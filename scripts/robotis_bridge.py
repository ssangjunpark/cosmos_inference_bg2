#!/usr/bin/env python3
"""Observe, preview, or bridge Cosmos predictions to existing ROBOTIS controllers."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import json
import threading
import time

from bg2_bridge.core import BridgeError, GROUPS, JointFeedback, Startup, all_names, hold_messages, load_home_yaml, require, trajectories, validate_profile
from bg2_bridge.observations import Observations
from bg2_bridge.ipc import predict, request

ROOT = Path(__file__).resolve().parents[1]


def replay(args, profile):
    """Offline event clock; no transport is imported or connected."""
    events = [json.loads(line) for line in args.events.read_text().splitlines() if line.strip()]
    require(events, "Replay is empty")
    startup = Startup(profile, load_home_yaml(args.initial_positions, profile), events[0]["now_ns"])
    output = []
    feedback = None
    for event in events:
        now = event["now_ns"]
        if event["kind"] == "homing_status":
            startup.homing_status(event["group"], event["status"], event["goal_started_ns"])
        elif event["kind"] == "joint_states":
            feedback = JointFeedback.from_arrays(event["names"], event["positions"], event["velocities"], event["source_ns"], now, profile)
            startup.update(feedback, now)
        elif event["kind"] == "prediction":
            require(feedback is not None, "Replay has no joint feedback")
            prediction = {**event["prediction"], "epoch": startup.epoch}
            output.append({"phase": startup.phase, "now_ns": now,
                           "commands": trajectories(prediction, feedback, startup, now, velocity_scale=args.velocity_scale)})
        else:
            raise BridgeError("Unknown replay event")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        json.dump({"mode": "offline_replay", "physical_publishing": False, "hold": startup.hold, "results": output}, handle, indent=2)
    print(json.dumps({"phase": startup.phase, "prediction_chunks": len(output), "output": str(args.output)}))


def live(args, profile):
    from bg2_bridge.transport import ZenohTransport
    execute = args.command == "execute"
    if args.command != "observe":
        validate_profile(profile, live=True, execute=execute)
        request(args.worker_socket, {"op": "health"}, timeout=5)
    observations = Observations(profile)
    home = load_home_yaml(args.initial_positions, profile) if args.command != "observe" else None
    startup = Startup(profile, home, time.time_ns()) if home is not None else None
    transport = ZenohTransport(profile, observations, startup)
    stopped = threading.Event()
    executor = ThreadPoolExecutor(max_workers=1)
    future, observation_id, active = None, 0, False
    end_ns, last_hold, last_request_ns, cycles = None, 0., 0, 0
    start, last_report = time.monotonic(), 0.
    guard = None
    args.output.mkdir(parents=True, exist_ok=False)
    journal = (args.output / "events.jsonl").open("x")

    def record(event):
        journal.write(json.dumps({"wall_time_ns": time.time_ns(), **event}, allow_nan=False) + "\n")
        journal.flush()

    def guard_loop():
        while not stopped.wait(.25):
            try:
                transport.check_ownership()
            except Exception as error:
                transport.guard_error = str(error)
                stopped.set()

    try:
        record({"event": "start", "mode": args.command, "physical_publishing": execute})
        while args.duration == 0 or time.monotonic() - start < args.duration:
            now = time.time_ns()
            require(not stopped.is_set(), transport.guard_error or "Bridge stopped")
            with observations.lock:
                require(observations.error is None, f"Observation callback failed: {observations.error}")
                feedback = observations.joints[-1] if observations.joints else None
                if startup is not None and feedback is not None:
                    startup.update(feedback, now)
                    require(startup.phase != "fault", "Homing failed")
            if args.command == "observe":
                if time.monotonic() - last_report >= 1:
                    with observations.lock:
                        record({"event": "observations", "joint_stamp_ns": feedback.source_ns if feedback else None,
                                "camera_stamp_ns": {key: items[-1][0] if items else None for key, items in observations.cameras.items()}})
                    last_report = time.monotonic()
                time.sleep(.02)
                continue
            if startup.phase in ("ready", "running"):
                feedback.fresh(now)
                if execute and not active:
                    transport.activate()
                    active = True
                    guard = threading.Thread(target=guard_loop, daemon=True)
                    guard.start()
                    transport.publish(hold_messages(feedback, startup, now, include_right=True))
                    record({"event": "hold_captured", "positions": startup.hold, "epoch": startup.epoch})
                if active and time.monotonic() - last_hold >= .1:
                    transport.publish(hold_messages(feedback, startup, now))
                    last_hold = time.monotonic()
                if end_ns is not None:
                    require(now < end_ns, "Last trajectory horizon ended before a new prediction arrived")
                if future is not None and future.done():
                    prediction = future.result()
                    feedback = observations.latest_feedback(time.time_ns())
                    commands = trajectories(prediction, feedback, startup, time.time_ns(), velocity_scale=args.velocity_scale)
                    if execute:
                        transport.publish(commands)
                        startup.phase = "running"
                    record({"event": "prediction", "prediction": prediction, "commands": commands, "physical_publishing": execute})
                    if execute:
                        end_ns = prediction["source_time_ns"] + 800_000_000
                    future = None
                    cycles += 1
                    if args.max_cycles and cycles >= args.max_cycles:
                        break
                if future is None:
                    try:
                        observation = observations.snapshot(args.instruction, time.time_ns())
                    except BridgeError as error:
                        # No motion is allowed without valid observations. During an active
                        # trajectory the watchdog still enforces its original forecast end.
                        if active:
                            raise
                        if time.monotonic() - last_report >= 1:
                            record({"event": "waiting", "reason": str(error), "phase": startup.phase})
                            last_report = time.monotonic()
                        time.sleep(.02)
                        continue
                    source_ns = int(observation["source_time_ns"])
                    if source_ns - last_request_ns >= 50_000_000:
                        last_request_ns = source_ns
                        future = executor.submit(predict, args.worker_socket, observation, startup.epoch, seed=args.seed)
            if time.monotonic() - last_report >= 1:
                record({"event": "state", "phase": startup.phase, "completed_homing": sorted(startup.completed)})
                last_report = time.monotonic()
            time.sleep(.02)
    except KeyboardInterrupt:
        record({"event": "stopped"})
    except Exception as error:
        record({"event": "fault", "reason": str(error)})
        raise
    finally:
        stopped.set()
        if guard is not None:
            guard.join(timeout=2)
        if active and feedback is not None:
            try:
                transport.publish(hold_messages(observations.latest_feedback(time.time_ns()), startup, time.time_ns(), include_right=True))
            except Exception as error:
                record({"event": "stop_hold_not_sent", "reason": str(error)})
        transport.close()
        if startup is not None:
            startup.stop()
        executor.shutdown(wait=False, cancel_futures=True)
        journal.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("replay", "observe", "preview", "execute"))
    parser.add_argument("--profile", type=Path, default=ROOT / "profiles/robotis_bg2_rev4.reference.json")
    parser.add_argument("--initial-positions", type=Path)
    parser.add_argument("--worker-socket", type=Path, default=ROOT / ".runtime/bridge/worker.sock")
    parser.add_argument("--instruction", default="")
    parser.add_argument("--events", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--velocity-scale", type=float, default=.25)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--duration", type=float, default=10., help="Seconds; zero runs until interrupted")
    parser.add_argument("--max-cycles", type=int, default=0)
    args = parser.parse_args()
    try:
        profile = json.loads(args.profile.read_text())
        validate_profile(profile)
        require(0 < args.velocity_scale <= 1 and args.duration >= 0 and args.max_cycles >= 0, "Invalid runtime settings")
        if args.command != "observe":
            require(args.initial_positions is not None, "Supply the exact initial-position YAML selected for robot bringup")
        if args.command == "replay":
            require(args.events is not None, "Replay needs an event journal")
            replay(args, profile)
        else:
            require(args.command == "observe" or args.instruction.strip(), "Supply an instruction")
            live(args, profile)
    except (ValueError, OSError, KeyError) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()
