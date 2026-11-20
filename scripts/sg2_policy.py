#!/usr/bin/env python3
"""5090 SG2 WAM capture, offline prediction, and synchronized robot sessions."""
import argparse
import json
from pathlib import Path
import time

import numpy as np

from bg2_bridge.core import require
from bg2_bridge.gateway import Gateway, GatewayTransport, StatusCache
from bg2_bridge.observations import Observations, decode_view
from bg2_wam import save_rollout
from sg2_wam import (ARM_NAMES, DATA_FORMAT, FORMAT, ROOT, VARIANTS, SG2WAM, calibration, camera_specs, check_handoff,
                     controller_positions, inspect_weights, read, settings, site_profile, training_observation)


def save_observation(path, observation):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        np.savez_compressed(stream, **observation)


def prepare(args):
    state = read(args.state)
    specs = camera_specs(("", "", ""))
    observation = {key: decode_view(path.read_bytes(), key, specs[key]) for key, path in
                   (("head", args.head), ("wrist_left", args.left_wrist), ("wrist", args.right_wrist))}
    observation.update(format=np.asarray(DATA_FORMAT), source_time_ns=np.asarray(state["source_time_ns"]),
                       joint_names=np.asarray(state["joint_names"]), joint_position=np.asarray(state["positions"], dtype=float))
    training_observation(observation, settings(args.variant))
    save_observation(args.output, observation)


class Predictor:
    def __init__(self, args):
        self.spec = settings(args.variant)
        self.names = self.spec["joint_names"]
        self.conversion = calibration(args.calibration, self.names) if args.calibration else None
        self.sampling = dict(steps=args.steps, seed=args.seed, guidance=args.guidance, shift=args.shift,
                             guidance_interval=args.guidance_interval)
        self.model = SG2WAM(args.checkpoint, args.tokenizer, args.vae, args.variant)

    def infer(self, observation, *, decode_video=True):
        prepared = training_observation(observation, self.spec, self.conversion)
        return self.model.predict(prepared, **self.sampling, decode_video=decode_video)

    def predict(self, observation, epoch, request_id):
        result = self.infer(observation, decode_video=False)
        positions = controller_positions(result["positions"], observation, self.names, self.conversion)
        return dict(schema="cosmos_sg2_joint_chunk_v1", epoch=epoch, request_id=request_id, units=["rad"] * 16,
                    joint_names=ARM_NAMES, source_time_ns=int(observation["source_time_ns"]), positions=positions.tolist())


def offline(args):
    with np.load(args.observation, allow_pickle=False) as archive:
        observation = {key: archive[key] for key in archive.files}
    # Validate inputs before allocating the GPU or starting the sampler.
    spec = settings(args.variant)
    conversion = calibration(args.calibration, spec["joint_names"]) if args.calibration else None
    training_observation(observation, spec, conversion)
    args.output.mkdir(parents=True, exist_ok=False)
    predictor = Predictor(args)
    result = predictor.infer(observation)
    record = {**predictor.model.runtime_metadata, "prompt": result["prompt"], "sampling": predictor.sampling,
              "source_time_ns": int(observation["source_time_ns"]), "joint_names": predictor.names,
              "units": "dataset-native", "time_offsets_ns": list(range(50_000_000, 800_000_001, 50_000_000)),
              "positions": result["positions"].tolist()}
    (args.output / "joint_positions.json").write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    np.save(args.output / "normalized_actions.npy", result["normalized"], allow_pickle=False)
    save_rollout(args.output, result["imagined_video"])


def run_live(args):
    profile = site_profile(args)
    gate = None
    if getattr(args, 'jepa_config', None):
        from bg2_bridge.sg2_dual import from_config
        from bg2_bridge.sg2_history import HistoryObservations
        from bg2_bridge.sg2_monitor import SelectionMonitor
        config = read(args.jepa_config.expanduser())
        monitor = SelectionMonitor(args.selection_output)
        gate = from_config(config,event=monitor)
        monitor.gate = gate
        inputs = HistoryObservations(profile, joint_names=ARM_NAMES, observation_format=FORMAT,
            history_frames=config['history_frames'], frame_interval_ns=config['frame_interval_ns'],
            video_view=config['video_view'], video_size=config['video_size'],
            history_required=lambda: not gate.fallback_active)
    else:
        inputs = Observations(profile, joint_names=ARM_NAMES, observation_format=FORMAT)
    status = StatusCache(schema="cosmos_sg2_status_v3")
    capture = args.command == "capture"
    predictor = gate if gate is not None else (None if capture else Predictor(args))
    instruction = settings(args.variant)["instruction"]
    transport = GatewayTransport(profile, inputs, status, inspect=capture, messages_dir=ROOT / "deploy/gateway/messages")
    gateway = None
    try:
        if not capture:
            gateway = Gateway(inputs, status, predictor, transport.send_chunk, instruction, mode=args.command, selection_gate=gate)
        deadline = time.monotonic() + args.timeout
        reason = "Waiting for SG2 sensor messages"
        while time.monotonic() < deadline:
            now = time.monotonic_ns()
            if capture:
                try:
                    obs = inputs.snapshot(instruction, inputs.latest_source_ns(), receipt_now_ns=now)
                except (ValueError, OSError) as exc:
                    reason = str(exc)
                else:
                    save_observation(args.output, obs)
                    return
            else:
                cached = status.get(now)
                if cached:
                    state = cached[0]
                    require(state["mode"] == args.command, "Companion mode differs from the requested session")
                    require(state["phase"] not in ("fault", "stopped"), state.get("reason") or state["phase"])
                    if gateway.awaiting and state["epoch"] != gateway.epoch:
                        raise ValueError("SG2 session changed while awaiting chunk completion")
                    if args.chunks and gateway.forwarded == args.chunks and gateway.awaiting:
                        cycle, request_id = gateway.awaiting
                        if state["last_completed_request_id"] == request_id and state["cycle"] > cycle and state["needs_chunk"]:
                            if gate is not None:
                                gate.execution_event("completed")
                            print(json.dumps({"completed_chunks": gateway.forwarded, "last_completed_request_id": request_id}))
                            return
                if args.chunks == 0 or gateway.forwarded < args.chunks:
                    event = gateway.step(now)
                    if event and event["event"] == "discarded":
                        raise ValueError(event["reason"])
                    if event and event["event"] == "forwarded":
                        print(json.dumps(event), flush=True)
                        deadline = time.monotonic() + args.timeout
                    elif event:
                        reason = event.get("reason", reason)
                else:
                    reason = (cached[0].get("last_rejection") if cached else None) or "Waiting for SG2 chunk completion"
            time.sleep(.02)
        raise TimeoutError(transport.last_error or reason)
    finally:
        if gateway is not None:
            gateway.close()
        transport.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("check", "prepare", "capture", "predict", "preview", "execute", "jepa-preview", "jepa-execute"):
        cmd = commands.add_parser(name)
        cmd.add_argument("--variant", choices=VARIANTS, required=True)
        if name.startswith('jepa-'):
            cmd.add_argument('--jepa-config',type=Path,required=True)
            cmd.add_argument('--selection-output',type=Path,required=True)
        if name in ("check", "predict", "preview", "execute"):
            cmd.add_argument("--checkpoint", type=Path, required=name != "check")
        if name == "check":
            cmd.add_argument("--handoff", type=Path, help="Compare with the supplied training loader sample (CPU PyTorch)")
        if name == "prepare":
            for field in ("head", "left-wrist", "right-wrist", "state"):
                cmd.add_argument("--" + field, type=Path, required=True)
        if name in ("prepare", "capture", "predict"):
            cmd.add_argument("--output", type=Path, required=True)
        if name == "predict":
            cmd.add_argument("--observation", type=Path, required=True)
        if name in ("predict", "preview", "execute"):
            for field in ("tokenizer", "vae"):
                cmd.add_argument("--" + field, type=Path, required=True)
            cmd.add_argument("--calibration", type=Path, required=name != "predict")
            cmd.add_argument("--steps", type=int, required=True)
            cmd.add_argument("--guidance", type=float, required=True)
            cmd.add_argument("--shift", type=float, required=True)
            cmd.add_argument("--guidance-interval", nargs=2, type=float)
            cmd.add_argument("--seed", type=int, default=0)
        if name in ("preview", "execute", "capture", "jepa-preview", "jepa-execute"):
            for field in ("router", "head-topic", "left-topic", "right-topic"):
                cmd.add_argument("--" + field, required=True)
            cmd.add_argument("--port", type=int, default=7447)
            cmd.add_argument("--domain", type=int, required=True)
            cmd.add_argument("--robot-profile", type=Path, default=ROOT / "profiles/robotis_sg2_rev1.reference.json")
            cmd.add_argument("--timeout", type=float, default=30, help="Seconds to wait for sensor data or a robot acknowledgement")
        if name in ("preview", "execute", "jepa-preview", "jepa-execute"):
            cmd.add_argument("--chunks", type=int, default=1, help="Number of chunks; 0 continues until interrupted in JEPA mode")
    args = parser.parse_args()
    if args.command.startswith('jepa-'):
        args.command = args.command.removeprefix('jepa-')
        require(args.chunks >= 0 and np.isfinite(args.timeout) and args.timeout > 0, 'Specify a nonnegative chunk count and positive timeout')
        cfg = read(args.jepa_config.expanduser())
        require(args.variant == cfg['primary_variant'], 'Variant must match the CM1 configuration')
        run_live(args)
        return
    if args.command == "check":
        spec = settings(args.variant)
        receipt = inspect_weights(args.checkpoint, spec) if args.checkpoint else None
        audit = check_handoff(args.handoff, args.variant) if args.handoff else None
        print(json.dumps({"variant": args.variant, "joint_names": spec["joint_names"], "instruction": spec["instruction"],
                          "handoff_check": audit, "checkpoint_step": receipt["step"] if receipt else None, "gpu_started": False}))
    elif args.command == "prepare":
        prepare(args)
    elif args.command == "predict":
        offline(args)
    else:
        require(np.isfinite(args.timeout) and args.timeout > 0 and getattr(args, "chunks", 1) > 0,
                "Specify a positive timeout and chunk count")
        run_live(args)


if __name__ == "__main__":
    main()
