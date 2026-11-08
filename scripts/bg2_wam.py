"""BG2 WAM inputs and raw action decoding. No ROS or hardware dependencies."""
import json
from pathlib import Path
from types import ModuleType

import numpy as np

from bg2_inference import JOINT_NAMES


def restore_wam_config(model, overrides):
    """Resolve the Nano export's abstract inherited classmethod, only in memory.

    Python serializes the inherited from_json_file method with _MoTConfigBase's
    qualified name. The saved causal-LM class and JSON identify its concrete
    family unambiguously; NVIDIA's Nano recipe uses Qwen3VLMoTConfig here.
    """
    import copy
    model = copy.deepcopy(model)
    instance = model["config"]["vlm_config"]["model_instance"]
    base = instance.get("config", {}).get("base_config", {})
    prefix = "cosmos_framework.model.generator.mot.unified_mot."
    abstract = prefix + "_MoTConfigBase.from_json_file"
    if base.get("_target_") != abstract:
        return model
    if (instance.get("_target_") != prefix + "Qwen3VLTextForCausalLM" or
            base.get("json_file") != "cosmos_framework/model/generator/reasoner/qwen3_vl/configs/Qwen3-VL-8B-Instruct.json"):
        raise ValueError("Cannot resolve abstract WAM config for an unknown saved architecture")
    concrete = prefix + "Qwen3VLMoTConfig.from_json_file"
    base["_target_"] = concrete
    overrides.append({"path": "model.config.vlm_config.model_instance.config.base_config._target_",
                      "saved": abstract, "runtime": concrete,
                      "reason": "Restore concrete Qwen3-VL config from saved model class/JSON, matching native Nano recipe."})
    return model


def exported_formatter(checkpoint):
    path = Path(checkpoint) / "vision_contract.py"
    # compile/exec deliberately avoids importlib's __pycache__ writes beside the source.
    module = ModuleType("bg2_checkpoint_wam_vision")
    module.__file__ = str(path)
    exec(compile(path.read_text(), str(path), "exec"), module.__dict__)
    return module


def prepare_tensors(observation, contract, experiment, formatter):
    import torch
    if experiment["training_mode"] != "wam" or experiment["action"]["names"] != JOINT_NAMES:
        raise ValueError("Expected the verified BG2 WAM experiment")
    stats = contract["normalization"]["joint_position"]
    lo, hi = np.asarray(stats["q01"]), np.asarray(stats["q99"])
    normalized = (observation["joint_position"].astype(np.float64) - (hi + lo) / 2) / np.maximum((hi - lo) / 2, 1e-6)
    if not np.isfinite(normalized).all():
        raise ValueError("Nonfinite WAM current state")
    views = []
    for key in ("head", "wrist_left", "wrist"):
        current = torch.from_numpy(observation[key].copy()).permute(2, 0, 1)
        video = torch.zeros((3, 17, *current.shape[1:]), dtype=torch.uint8)
        video[:, 0] = current
        views.append(video)
    action = torch.zeros((17, 64), dtype=torch.float32)
    action[0, :8] = torch.from_numpy(normalized).float()
    raw = torch.zeros((17, 8), dtype=torch.float32)
    raw[0] = torch.from_numpy(observation["joint_position"].copy())
    return {"video": formatter.compose_views(*views), "action": action, "action_raw": raw,
            "raw_action_dim": torch.tensor(8), "domain_id": torch.tensor(experiment["action"]["domain_id"])}


def build_batch(observation, contract, experiment, formatter, tokenizer):
    import torch
    from cosmos_framework.data.generator.action.utils.action_processing import ActionProcessingRecord
    from cosmos_framework.data.generator.joint_dataloader import custom_collate_fn
    from cosmos_framework.data.generator.sequence_packing import SequencePlan
    sample = prepare_tensors(observation, contract, experiment, formatter)
    prompt = formatter.build_policy_prompt(observation["instruction"])
    if json.loads(prompt).get("mode") == "policy":
        raise ValueError("Checkpoint formatter produced a policy-only prompt")
    tokens = tokenizer.tokenize_text(prompt, is_video=False, use_system_prompt=False)
    if not tokens:
        raise ValueError("Empty WAM instruction tokens")
    sample.update({"__key__": str(int(observation["source_time_ns"])), "mode": "wam", "ai_caption": prompt,
                   "text_token_ids": torch.tensor(tokens, dtype=torch.long), "conditioning_fps": torch.tensor(20),
                   "image_size": torch.tensor([640., 512., 640., 512.]),
                   "sequence_plan": SequencePlan(has_text=True, has_vision=True, has_action=True,
                       condition_frame_indexes_vision=[0], condition_frame_indexes_action=[0], action_start_frame_offset=0),
                   "action_processing_record": ActionProcessingRecord(raw_action_dim=8, action_normalizer=None)})
    batch = custom_collate_fn([sample])
    batch["dataset_name"] = ["bg2_room"]
    for key in ("text_token_ids", "video", "action", "action_raw"):
        batch[key] = [[value] for value in batch[key]]
    for key in ("image_size", "conditioning_fps"):
        batch[key] = [value.unsqueeze(0) for value in batch[key]]
    return batch, prompt


def decode_actions(normalized, observation, contract):
    """Invert training normalization; preserve every raw prediction, including negatives."""
    if normalized.shape != (17, 8) or not np.isfinite(normalized).all():
        raise ValueError("Expected finite WAM actions shaped (17, 8)")
    stats = contract["normalization"]["joint_position"]
    lo, hi = np.asarray(stats["q01"]), np.asarray(stats["q99"])
    physical = normalized.astype(np.float64) * np.maximum((hi - lo) / 2, 1e-6) + (hi + lo) / 2
    if not np.allclose(physical[0], observation["joint_position"], rtol=2e-5, atol=2e-5):
        raise ValueError("WAM changed the conditioned current-state row")
    targets = physical[1:].astype(np.float32)
    if not np.isfinite(targets).all():
        raise ValueError("Nonfinite decoded WAM actions")
    return targets


def decoded_rgb(decoded):
    """Native model.decode output [1,C,T,H,W] in [-1,1] to RGB uint8."""
    import torch
    if tuple(decoded.shape) != (1, 3, 17, 640, 512) or not torch.isfinite(decoded).all():
        raise ValueError(f"Expected finite decoded WAM video [1,3,17,640,512], got {tuple(decoded.shape)}")
    return ((decoded[0].clamp(-1, 1) + 1) * 127.5).to(torch.uint8).permute(1, 2, 3, 0).cpu().numpy()


def write_mp4(path, frames, fps=20):
    import os
    import subprocess
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    if frames.dtype != np.uint8 or frames.ndim != 4 or frames.shape[-1] != 3 or not len(frames):
        raise ValueError("Expected nonempty RGB uint8 video")
    # Keep FFmpeg outside the CUDA/PyTorch process: the image's in-process PyAV
    # round-trip stalled after encoding during GPU validation. No model work is
    # asynchronous; run() waits for each export and verification to finish.
    # Bound codec/filter and linked math-library pools in the export subprocesses.
    # This does not change the model process's CPU or CUDA configuration.
    media_env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    subprocess.run(["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-n",
        "-filter_threads", "1", "-filter_complex_threads", "1",
        "-f", "rawvideo", "-pixel_format", "rgb24", "-video_size", f"{frames.shape[2]}x{frames.shape[1]}",
        "-framerate", str(fps), "-i", "pipe:0", "-an", "-c:v", "libx264", "-threads", "2",
        "-crf", "18", "-preset", "fast", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path)],
        input=frames.tobytes(), check=True, env=media_env)
    result = subprocess.check_output(["ffprobe", "-v", "error", "-threads", "1", "-select_streams", "v:0", "-count_frames",
        "-show_entries", "stream=nb_read_frames,width,height,avg_frame_rate", "-of", "json", str(path)], text=True, env=media_env)
    stream = json.loads(result)["streams"][0]
    from fractions import Fraction
    if (int(stream["nb_read_frames"]) != len(frames) or int(stream["width"]) != frames.shape[2] or
            int(stream["height"]) != frames.shape[1] or Fraction(stream["avg_frame_rate"]) != fps):
        raise RuntimeError("Rendered WAM video frame count mismatch")


def save_rollout(directory, frames):
    directory = Path(directory)
    if frames.shape != (17, 640, 512, 3) or frames.dtype != np.uint8:
        raise ValueError("Expected 17 RGB WAM composite frames")
    # Preserve lossless decoded results even if a later MP4 export fails.
    with (directory / "decoded_rollout.npy").open("xb") as file:
        np.save(file, frames, allow_pickle=False)
    from PIL import Image
    Image.fromarray(frames[0]).save(directory / "context_reconstruction.png")
    Image.fromarray(frames[-1]).save(directory / "imagined_last_frame.png")
    metadata = {"fps": 20, "frame_count": 17, "frame_offsets_ns": [i * 50_000_000 for i in range(17)],
                "context_frame": 0, "future_frames": list(range(1, 17)),
                "full_clip_playback_seconds": 17 / 20, "future_only_seconds": 16 / 20,
                "note": "Frame 0 is VAE-reconstructed conditioned context. Frames 1..16 are generated future, not hardware observations."}
    with (directory / "rollout.json").open("x") as file:
        json.dump(metadata, file, indent=2)
    write_mp4(directory / "rollout_full.mp4", frames)
    future = frames[1:]
    write_mp4(directory / "imagined_future.mp4", future)
    write_mp4(directory / "imagined_head.mp4", future[:, :384])
    write_mp4(directory / "imagined_wrist_left.mp4", future[:, 384:, :256])
    write_mp4(directory / "imagined_wrist_right.mp4", future[:, 384:, 256:])
    return metadata
