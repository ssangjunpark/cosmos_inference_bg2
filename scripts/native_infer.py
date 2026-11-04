#!/usr/bin/env python3
"""Run one recorded observation and save normalized and decoded joint predictions."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import time



def relocate_model_config(config: dict, tokenizer: Path, vae: Path) -> dict:
    model = copy.deepcopy(config["model"])
    node = model["config"]["vlm_config"]["tokenizer"]
    if "tokenizer_type" in node:
        node["tokenizer_type"] = str(tokenizer)
    elif "pretrained_model_name" in node:
        node["pretrained_model_name"] = str(tokenizer)
    else:
        raise ValueError("Unknown tokenizer configuration; use the original adapter's inference path")
    model["config"]["tokenizer"]["vae_path"] = str(vae)
    model["config"]["ema"]["enabled"] = False
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--observation", type=Path, required=True)
    parser.add_argument("--adapter-dir", type=Path, default=Path("/adapter"))
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--vae", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--guidance", type=float, default=3.0)
    parser.add_argument("--shift", type=float, default=5.0)
    parser.add_argument("--guidance-interval", nargs=2, type=float)
    args = parser.parse_args()

    from cosmos_framework.inference.common.init import init_script
    init_script()

    import numpy as np
    import torch
    from cosmos_framework.configs.base.defaults.compile import CompileConfig
    from cosmos_framework.configs.base.defaults.parallelism import ParallelismConfig
    from cosmos_framework.inference.common.config import deserialize_config_dict
    from cosmos_framework.inference.model import Cosmos3OmniConfig, Cosmos3OmniModel
    from cosmos_framework.utils.lazy_config import instantiate
    from bg2_inference import JOINT_NAMES, build_batch, decode_actions, load_observation, verify_adapter
    from cosmos_framework.utils.lazy_config.registry import locate

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA must be available in the inference container")
    observation = load_observation(args.observation)
    contract = json.loads((args.checkpoint / "input_contract.json").read_text())
    provenance = verify_adapter(args.adapter_dir, contract)
    config = deserialize_config_dict(args.checkpoint / "config.yaml")
    model_dict = relocate_model_config(config, args.tokenizer, args.vae)
    target = model_dict["_target_"]
    if model_dict["config"].get("action_global_asinh") or model_dict["config"].get("action_metric_gripper"):
        raise ValueError("Checkpoint enables transforms outside the verified joint-position contract")
    if target != provenance["model_target"]:
        raise ValueError("Model target differs from verified training adapter")
    try:
        locate(target)
    except (ImportError, ModuleNotFoundError) as error:
        raise RuntimeError(
            f"The checkpoint uses {target}. Supply the original adapter source with --adapter-dir. "
            "The runner preserves the saved model class instead of substituting the public DROID model."
        ) from error

    args.output.mkdir(parents=True, exist_ok=True)
    if any(args.output.iterdir()):
        raise ValueError("Output directory must be empty to preserve earlier results")
    runtime = {
        "framework_commit": contract["framework_commit"],
        "model_target": target, "seed": args.seed, "steps": args.steps,
        "guidance": args.guidance, "guidance_interval": args.guidance_interval, "shift": args.shift,
        "checkpoint": str(args.checkpoint), "observation": str(args.observation),
        "observation_sha256": hashlib.sha256(args.observation.read_bytes()).hexdigest(),
        "config_sha256": hashlib.sha256((args.checkpoint / "config.yaml").read_bytes()).hexdigest(),
        "checkpoint_publication": json.loads((args.checkpoint / "READY.json").read_text()),
        "checkpoint_files": json.loads((args.checkpoint / "COPY_VERIFIED.json").read_text()),
        "adapter": provenance,
        "output_space": {"action_0.npy": "normalized joints, current row plus 16 predictions",
                         "joint_positions.npy": "absolute measured-joint coordinates, 16 predictions, radians"},
        "input_contract": contract,
    }
    prepare_start = time.monotonic()
    tokenizer = instantiate(model_dict["config"]["vlm_config"]["tokenizer"])
    batch, prompt = build_batch(observation, contract, tokenizer)
    del tokenizer
    runtime.update({"prompt": prompt, "instruction": observation["instruction"],
                    "source_time_ns": int(observation["source_time_ns"]),
                    "current_joint_positions": observation["joint_position"].tolist(),
                    "normalized_current_state": batch["action"][0][0][0, :8].tolist(),
                    "domain_id": 33, "condition_frame_indexes_vision": [0],
                    "condition_frame_indexes_action": [0], "action_start_frame_offset": 0})
    preprocessing_seconds = time.monotonic() - prepare_start
    (args.output / "request.json").write_text(json.dumps(runtime, indent=2) + "\n")
    parallelism = ParallelismConfig(
        data_parallel_shard_degree=1, data_parallel_replicate_degree=1,
        context_parallel_shard_degree=1, cfg_parallel_shard_degree=1,
        enable_inference_mode=True,
    )
    load_start = time.monotonic()
    wrapper = Cosmos3OmniModel.from_pretrained_dcp(
        args.checkpoint / "model", config=Cosmos3OmniConfig(model=model_dict),
        parallelism_config=parallelism,
        compile_config=CompileConfig(enabled=False, use_cuda_graphs=False),
    )
    model = wrapper.model.eval()
    torch.cuda.synchronize()
    loading_seconds = time.monotonic() - load_start
    torch.cuda.reset_peak_memory_stats()
    start = time.monotonic()
    with torch.inference_mode():
        outputs = model.generate_samples_from_batch(
            batch, seed=[args.seed], num_steps=args.steps, guidance=args.guidance,
            guidance_interval=args.guidance_interval, shift=args.shift,
        )
    torch.cuda.synchronize()
    summary = {"inference_seconds": time.monotonic() - start,
               "loading_seconds": loading_seconds, "preprocessing_seconds": preprocessing_seconds,
               "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated(),
               "peak_gpu_reserved_bytes": torch.cuda.max_memory_reserved(), "arrays": {}}
    for key in ("action", "vision"):
        values = outputs.get(key, [])
        for index, tensor in enumerate(values):
            value = tensor.detach().float().cpu().numpy()
            if not np.isfinite(value).all():
                raise RuntimeError(f"Nonfinite {key} output")
            filename = f"{key}_{index}.npy"
            with (args.output / filename).open("xb") as handle:
                np.save(handle, value, allow_pickle=False)
            summary["arrays"][filename] = {"shape": list(value.shape), "dtype": str(value.dtype)}
    if len(outputs.get("action", [])) != 1:
        raise RuntimeError("Expected exactly one predicted action sequence")
    decode_start = time.monotonic()
    normalized = outputs["action"][0].detach().float().cpu().numpy()
    positions = decode_actions(normalized, observation, contract)
    with (args.output / "joint_positions.npy").open("xb") as handle:
        np.save(handle, positions, allow_pickle=False)
    summary["arrays"]["joint_positions.npy"] = {"shape": list(positions.shape), "dtype": str(positions.dtype)}
    trajectory = {"joint_names": JOINT_NAMES, "units": ["rad"] * 8,
                  "target_type": "absolute joint position", "source_time_ns": int(observation["source_time_ns"]),
                  "forecast_time_offsets_ns": [50_000_000 * i for i in range(1, 17)],
                  "positions": positions.tolist()}
    (args.output / "joint_positions.json").write_text(json.dumps(trajectory, indent=2) + "\n")
    summary["decoding_seconds"] = time.monotonic() - decode_start
    (args.output / "outputs.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
