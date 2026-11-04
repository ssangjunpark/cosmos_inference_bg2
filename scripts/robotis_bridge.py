#!/usr/bin/env python3
"""Legacy offline event replay only. Live observation/execution uses gateway.py and the FFW companion."""
import argparse
from pathlib import Path
import json
from bg2_bridge.core import BridgeError, JointFeedback, Startup, load_home_yaml, require, trajectories, validate_profile
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
                           "commands": trajectories(prediction, feedback, startup, now)})
        else:
            raise BridgeError("Unknown replay event")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        json.dump({"mode": "offline_replay", "physical_publishing": False, "hold": startup.hold, "results": output}, handle, indent=2)
    print(json.dumps({"phase": startup.phase, "prediction_chunks": len(output), "output": str(args.output)}))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('replay',))
    parser.add_argument('--profile',type=Path,default=ROOT/'profiles/robotis_bg2_rev4.reference.json')
    parser.add_argument('--initial-positions',type=Path,required=True)
    parser.add_argument('--events',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    try:
        profile=json.loads(args.profile.read_text());validate_profile(profile);replay(args,profile)
    except (ValueError,OSError,KeyError) as error:parser.exit(1,f'Error: {error}\n')


if __name__=='__main__':main()
