#!/usr/bin/env python3
"""Invoke the native framework API once and save its raw outputs, without a robot."""
from __future__ import annotations

import argparse
import copy
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


def load_sample(path: Path):
    import numpy as np
    import torch

    sample = {}
    with np.load(path, allow_pickle=False) as archive:
        for key in archive.files:
            value = archive[key]
            if value.dtype.kind in "US":
                if value.ndim != 0:
                    raise ValueError(f"{key}: string metadata must be a scalar")
                sample[key] = str(value.item())
            elif value.dtype.kind in "biuf":
                if value.dtype.kind == "f" and not np.isfinite(value).all():
                    raise ValueError(f"{key}: nonfinite input")
                sample[key] = torch.from_numpy(value.copy())
            else:
                raise ValueError(f"{key}: unsupported input dtype {value.dtype}")
    for required in ("ai_caption", "video", "action", "domain_id"):
        if required not in sample:
            raise ValueError(f"Native sample is missing {required!r}; export it using the original data adapter")
    if not isinstance(sample["ai_caption"], str):
        raise ValueError("ai_caption must contain the native formatted prompt as a scalar string")
    video = sample["video"]
    if video.ndim != 4 or video.shape[0] != 3 or video.shape[1] < 1 or tuple(video.shape[-2:]) != (640, 512) or video.dtype != torch.uint8:
        raise ValueError("RobotEscape sample must use the contract's RGB uint8 640x512 composite")
    if sample["action"].ndim != 2 or not sample["action"].is_floating_point():
        raise ValueError("action must be an unbatched floating-point [T,D] tensor")
    if sample["domain_id"].ndim != 0 or sample["domain_id"].dtype != torch.int64:
        raise ValueError("domain_id must be a native scalar int64")
    return sample


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--sample", type=Path, required=True)
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
    from cosmos_framework.scripts.action_policy_server_robolab import _build_data_batch_from_sample
    from cosmos_framework.utils.lazy_config.registry import locate

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA must be available in the inference container")
    sample = load_sample(args.sample)
    contract = json.loads((args.checkpoint / "input_contract.json").read_text())
    config = deserialize_config_dict(args.checkpoint / "config.yaml")
    model_dict = relocate_model_config(config, args.tokenizer, args.vae)
    target = model_dict["_target_"]
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
        "checkpoint": str(args.checkpoint), "sample": str(args.sample),
        "output_space": "raw framework output; no hardware conversion or action execution",
        "input_contract": contract,
    }
    (args.output / "request.json").write_text(json.dumps(runtime, indent=2) + "\n")
    parallelism = ParallelismConfig(
        data_parallel_shard_degree=1, data_parallel_replicate_degree=1,
        context_parallel_shard_degree=1, cfg_parallel_shard_degree=1,
        enable_inference_mode=True,
    )
    wrapper = Cosmos3OmniModel.from_pretrained_dcp(
        args.checkpoint / "model", config=Cosmos3OmniConfig(model=model_dict),
        parallelism_config=parallelism,
        compile_config=CompileConfig(enabled=False, use_cuda_graphs=False),
    )
    model = wrapper.model.eval()
    batch = _build_data_batch_from_sample(sample)
    start = time.monotonic()
    with torch.inference_mode():
        outputs = model.generate_samples_from_batch(
            batch, seed=[args.seed], num_steps=args.steps, guidance=args.guidance,
            guidance_interval=args.guidance_interval, shift=args.shift,
        )
    torch.cuda.synchronize()
    summary = {"inference_seconds": time.monotonic() - start, "arrays": {}}
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
    if not outputs.get("action"):
        raise RuntimeError("The native sample produced no predicted action output")
    (args.output / "outputs.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
