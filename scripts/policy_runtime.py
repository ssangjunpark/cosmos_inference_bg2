"""Reusable native Cosmos policy. Construction loads weights once; predict opens no robot connection."""
from pathlib import Path
import copy
import json
import time


def relocate_model_config(config, tokenizer, vae):
    model = copy.deepcopy(config["model"])
    node = model["config"]["vlm_config"]["tokenizer"]
    if "tokenizer_type" in node:
        node["tokenizer_type"] = str(tokenizer)
    elif "pretrained_model_name" in node:
        node["pretrained_model_name"] = str(tokenizer)
    else:
        raise ValueError("Unknown tokenizer configuration")
    model["config"]["tokenizer"]["vae_path"] = str(vae)
    model["config"]["ema"]["enabled"] = False
    return model


class NativePolicy:
    def __init__(self, checkpoint, tokenizer, vae, adapter_dir, *, model_config_transform=None):
        from cosmos_framework.inference.common.init import init_script
        init_script()
        import torch
        from cosmos_framework.configs.base.defaults.compile import CompileConfig
        from cosmos_framework.configs.base.defaults.parallelism import ParallelismConfig
        from cosmos_framework.inference.common.config import deserialize_config_dict
        from cosmos_framework.inference.model import Cosmos3OmniConfig, Cosmos3OmniModel
        from cosmos_framework.utils.lazy_config import instantiate
        from bg2_inference import verify_adapter

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA must be available in the inference container")
        start = time.monotonic()
        self.checkpoint = Path(checkpoint)
        self.contract = json.loads((self.checkpoint / "input_contract.json").read_text())
        self.provenance = verify_adapter(Path(adapter_dir), self.contract)
        model = relocate_model_config(deserialize_config_dict(self.checkpoint / "config.yaml"), tokenizer, vae)
        if model_config_transform is not None:
            model = model_config_transform(model)
        self.target = model["_target_"]
        if self.target != self.provenance["model_target"] or model["config"].get("action_global_asinh") or model["config"].get("action_metric_gripper"):
            raise ValueError("Checkpoint differs from the verified BG2 action contract")
        self.tokenizer = instantiate(model["config"]["vlm_config"]["tokenizer"])
        self.wrapper = Cosmos3OmniModel.from_pretrained_dcp(
            self.checkpoint / "model", config=Cosmos3OmniConfig(model=model),
            parallelism_config=ParallelismConfig(data_parallel_shard_degree=1, data_parallel_replicate_degree=1,
                                                context_parallel_shard_degree=1, cfg_parallel_shard_degree=1,
                                                enable_inference_mode=True),
            compile_config=CompileConfig(enabled=False, use_cuda_graphs=False))
        self.model = self.wrapper.model.eval()
        torch.cuda.synchronize()
        self.loading_seconds = time.monotonic() - start

    def predict(self, observation, *, seed=0, steps=4, guidance=3., shift=5., guidance_interval=None):
        import numpy as np
        import torch
        from bg2_inference import build_batch, decode_actions

        if steps < 1 or not all(np.isfinite(v) for v in (guidance, shift)) or shift <= 0:
            raise ValueError("Invalid sampling settings")
        start = time.monotonic()
        batch, prompt = build_batch(observation, self.contract, self.tokenizer)
        preprocessing_seconds = time.monotonic() - start
        torch.cuda.reset_peak_memory_stats()
        start = time.monotonic()
        with torch.inference_mode():
            outputs = self.model.generate_samples_from_batch(batch, seed=[seed], num_steps=steps,
                       guidance=guidance, shift=shift, guidance_interval=guidance_interval)
        torch.cuda.synchronize()
        generation_seconds = time.monotonic() - start
        if len(outputs.get("action", [])) != 1:
            raise RuntimeError("Expected one native action sequence")
        normalized = outputs["action"][0].detach().float().cpu().numpy()
        start = time.monotonic()
        positions = decode_actions(normalized, observation, self.contract)
        vision = [v.detach().float().cpu().numpy() for v in outputs.get("vision", [])]
        if any(not np.isfinite(v).all() for v in vision):
            raise RuntimeError("Nonfinite vision output")
        return {"positions": positions, "normalized": normalized, "vision": vision, "prompt": prompt,
                "normalized_current_state": batch["action"][0][0][0, :8].tolist(),
                "timings": {"loading_seconds": self.loading_seconds, "preprocessing_seconds": preprocessing_seconds,
                            "inference_seconds": generation_seconds, "decoding_seconds": time.monotonic() - start,
                            "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated(),
                            "peak_gpu_reserved_bytes": torch.cuda.max_memory_reserved()}}
