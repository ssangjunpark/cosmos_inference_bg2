#!/usr/bin/env python3
"""One synchronous, offline BG2 WAM prediction with decoded imagined-video outputs."""
import argparse
import json
from pathlib import Path
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["checkpoint", "observation", "tokenizer", "vae", "experiment-contract", "output"]:
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--adapter-dir", type=Path, default=Path("/adapter"))
    parser.add_argument("--step", type=int, default=20000)
    parser.add_argument("--instruction")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--guidance", type=float, default=3.)
    parser.add_argument("--shift", type=float, default=5.)
    parser.add_argument("--guidance-interval", type=float, nargs=2)
    args = parser.parse_args()

    import numpy as np
    import torch
    from PIL import Image
    from bg2_inference import load_observation, JOINT_NAMES
    from bg2_wam import save_rollout
    from wam_runtime import NativeWAM
    from wam_checkpoint import validate_checkpoint, validate_experiment, file_hash

    publication = validate_checkpoint(args.checkpoint, step=args.step)
    contract = publication["contract"]
    experiment = validate_experiment(args.experiment_contract, contract)
    observation = load_observation(args.observation)
    if args.instruction is not None:
        if not args.instruction.strip():
            raise ValueError("Instruction must not be empty")
        observation["instruction"] = args.instruction
    if args.steps < 1 or not all(np.isfinite(v) for v in [args.guidance, args.shift]) or args.shift <= 0:
        raise ValueError("Invalid sampling parameters")
    if args.guidance_interval is not None and (not all(np.isfinite(args.guidance_interval)) or args.guidance_interval[0] > args.guidance_interval[1]):
        raise ValueError("Invalid guidance interval")
    args.output.mkdir(parents=True, exist_ok=True)
    if any(p.name != "launch.json" for p in args.output.iterdir()):
        raise ValueError("WAM output directory must be empty")

    def write_json(name, value):
        with (args.output / name).open("x") as file:
            json.dump(value, file, indent=2, allow_nan=False)
            file.write("\n")

    def save_array(name, value):
        with (args.output / name).open("xb") as file:
            np.save(file, value, allow_pickle=False)

    # No gateway, ROS context, network client, execution queue or asynchronous worker.
    started = time.monotonic()
    policy = NativeWAM(args.checkpoint, args.tokenizer, args.vae, args.adapter_dir,
                       args.experiment_contract, step=args.step)
    prompt = policy.formatter.build_policy_prompt(observation["instruction"])
    runtime = {"mode": "wam", "execution": "offline_synchronous", "hardware_commands": False,
               "checkpoint": publication["ready"], "weight_manifest": publication["manifest"],
               "framework_commit": contract["framework_commit"], "training_experiment": experiment,
               "config_sha256": file_hash(args.checkpoint / "config.yaml"),
               "vision_contract_sha256": file_hash(args.checkpoint / "vision_contract.py"),
               "experiment_contract_sha256": file_hash(args.experiment_contract),
               "input_sha256": file_hash(args.observation), "input_contract": contract,
               "model_class_adapter": policy.provenance, "runtime_config_overrides": policy.config_overrides, "prompt": prompt,
               "instruction": observation["instruction"], "domain_id": experiment["action"]["domain_id"],
               "video_frames": 17, "future_actions": 16, "fps": 20,
               "condition_frame_indexes_vision": [0], "condition_frame_indexes_action": [0],
               "action_start_frame_offset": 0, "sampling": {"seed": args.seed, "steps": args.steps,
                   "guidance": args.guidance, "shift": args.shift, "guidance_interval": args.guidance_interval},
               "torch_version": torch.__version__, "gpu_name": torch.cuda.get_device_name(),
               "loading_seconds": policy.loading_seconds,
               "action_postprocessing": "Inverse training affine only; no clipping, inversion or hardware gripper floor."}
    write_json("request.json", runtime)
    with (args.output / "observation.npz").open("xb") as file:
        np.savez_compressed(file, **observation)
    composite = np.concatenate([observation["head"], np.concatenate([observation["wrist_left"], observation["wrist"]], axis=1)], axis=0)
    Image.fromarray(composite).save(args.output / "input_composite.png")
    print(json.dumps({"event": "generating", "mode": "wam", "video_frames": 17, "future_actions": 16}), flush=True)
    result = policy.predict(observation, seed=args.seed, steps=args.steps,
                            guidance=args.guidance, shift=args.shift, guidance_interval=args.guidance_interval)
    actions, latent, frames = result["positions"], result["vision"][0], result["imagined_video"]
    save_array("normalized_actions.npy", result["normalized"])
    save_array("joint_positions.npy", actions)
    save_array("vision_latent.npy", latent)
    write_json("joint_positions.json", {"mode": "wam", "target_type": "absolute measured-joint coordinates",
               "joint_names": JOINT_NAMES, "units": ["rad"] * 8,
               "source_time_ns": int(observation["source_time_ns"]),
               "forecast_time_offsets_ns": [i * 50_000_000 for i in range(1, 17)], "positions": actions.tolist()})
    print(json.dumps({"event": "saving_rollout", "timings": result["timings"]}), flush=True)
    rollout = save_rollout(args.output, frames)
    summary = {"mode": "wam", "hardware_commands": False, "generation_seconds": result["timings"]["inference_seconds"],
               "video_decoding_seconds": result["timings"]["video_decoding_seconds"], "total_seconds": time.monotonic() - started,
               "peak_gpu_allocated_bytes": result["timings"]["peak_gpu_allocated_bytes"],
               "peak_gpu_reserved_bytes": result["timings"]["peak_gpu_reserved_bytes"],
               "actions_shape": list(actions.shape), "vision_latent_shape": list(latent.shape),
               "decoded_video_shape": list(frames.shape), "rollout": rollout}
    write_json("outputs.json", summary)
    print(json.dumps({"event": "complete", **summary}), flush=True)


if __name__ == "__main__":
    main()
