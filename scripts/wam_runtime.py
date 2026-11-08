"""Reusable blocking WAM generation for the one-shot runner and HTTP service."""
from pathlib import Path
import time

from policy_runtime import NativePolicy
from wam_checkpoint import validate_checkpoint, validate_experiment, file_hash
from bg2_wam import build_batch, decode_actions, decoded_rgb, exported_formatter, restore_wam_config


class NativeWAM(NativePolicy):
    mode = "wam"

    def __init__(self, checkpoint, tokenizer, vae, adapter_dir, experiment_contract, *, step=20000):
        self.publication = validate_checkpoint(checkpoint, step=step)
        self.experiment = validate_experiment(experiment_contract, self.publication["contract"])
        self.config_overrides = []
        super().__init__(checkpoint, tokenizer, vae, adapter_dir,
            model_config_transform=lambda model: restore_wam_config(model, self.config_overrides))
        import torch
        self.formatter = exported_formatter(checkpoint)
        self.runtime_metadata = {
            "mode": "wam", "checkpoint": self.publication["ready"],
            "torch_version": torch.__version__, "gpu_name": torch.cuda.get_device_name(),
            "framework_commit": self.contract["framework_commit"], "input_contract": self.contract,
            "training_experiment": self.experiment, "model_class_adapter": self.provenance,
            "runtime_config_overrides": self.config_overrides,
            "metadata_sha256": {name: file_hash(Path(checkpoint) / name) for name in
                ["config.yaml", "input_contract.json", "vision_contract.py", "VERIFIED.json"]},
            "experiment_contract_sha256": file_hash(experiment_contract),
            "video_frames": 17, "future_actions": 16, "fps": 20,
            "condition_frame_indexes_vision": [0], "condition_frame_indexes_action": [0],
            "action_start_frame_offset": 0,
            "action_postprocessing": "Inverse training affine only; raw predictions have no clipping or gripper floor."}

    def predict(self, observation, *, seed=0, steps=4, guidance=3., shift=5., guidance_interval=None):
        import numpy as np
        import torch
        if steps < 1 or not all(np.isfinite(v) for v in (guidance, shift)) or shift <= 0:
            raise ValueError("Invalid WAM sampling settings")
        if guidance_interval is not None and (len(guidance_interval) != 2 or
                not np.isfinite(guidance_interval).all() or guidance_interval[0] > guidance_interval[1]):
            raise ValueError("Invalid guidance interval")
        started = time.monotonic()
        batch, prompt = build_batch(observation, self.contract, self.experiment, self.formatter, self.tokenizer)
        preprocessing = time.monotonic() - started
        torch.cuda.reset_peak_memory_stats()
        tick = time.monotonic()
        with torch.inference_mode():
            generated = self.model.generate_samples_from_batch(batch, seed=[seed], num_steps=steps,
                guidance=guidance, guidance_interval=guidance_interval, shift=shift)
        torch.cuda.synchronize()
        generation = time.monotonic() - tick
        if len(generated.get("action", [])) != 1 or len(generated.get("vision", [])) != 1:
            raise RuntimeError("Expected one jointly generated WAM action/video sequence")
        normalized = generated["action"][0].detach().float().cpu().numpy()
        positions = decode_actions(normalized, observation, self.contract)
        latent = generated["vision"][0]
        if not torch.isfinite(latent).all():
            raise RuntimeError("Nonfinite imagined-video latent")
        tick = time.monotonic()
        with torch.inference_mode():
            frames = decoded_rgb(self.model.decode(latent))
        torch.cuda.synchronize()
        decoding = time.monotonic() - tick
        return {"mode": "wam", "positions": positions, "normalized": normalized,
                "vision": [latent.detach().float().cpu().numpy()], "imagined_video": frames,
                "prompt": prompt, "runtime_metadata": self.runtime_metadata,
                "timings": {"loading_seconds": self.loading_seconds, "preprocessing_seconds": preprocessing,
                    "inference_seconds": generation, "video_decoding_seconds": decoding,
                    "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated(),
                    "peak_gpu_reserved_bytes": torch.cuda.max_memory_reserved()}}
