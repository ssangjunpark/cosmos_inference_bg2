"""Recorded-observation adapter for the verified Cosmos future-joint training contract."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

JOINT_NAMES = [f"arm_r_joint{i}" for i in range(1, 8)] + ["gripper_r_joint1"]
FORMAT = "robotescape_recorded_observation_v1"


def canonical_hash(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def verify_adapter(directory: Path, contract: dict) -> dict:
    provenance = json.loads((directory / "SOURCE.json").read_text())
    for key in ("framework_commit", "data_manifest_sha256"):
        if provenance[key] != contract[key]:
            raise ValueError(f"Adapter/checkpoint {key} mismatch")
    if canonical_hash(contract["normalization"]) != provenance["normalization_sha256"]:
        raise ValueError("Adapter/checkpoint training normalization mismatch")
    for name, record in provenance["files"].items():
        path = (directory / name).resolve(strict=True)
        path.relative_to(directory.resolve())
        if hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]:
            raise ValueError(f"Adapter source changed: {name}")
    if contract["normalization"]["joint_names"] != JOINT_NAMES:
        raise ValueError("Checkpoint joint order does not match the verified adapter")
    if contract["vision"]["first_target_offset_frames"] != 1 or contract["vision"]["frame_interval_ns"] != 50_000_000:
        raise ValueError("Checkpoint forecast timing differs from the verified adapter")
    return provenance


def load_observation(path: Path) -> dict:
    expected = {"head", "wrist_left", "wrist", "joint_position", "source_time_ns", "instruction", "joint_names", "format"}
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != expected:
            raise ValueError("Observation must contain only the eight recorded current-observation fields")
        observation = {key: archive[key].copy() for key in archive.files}
    for key in ("format", "instruction"):
        if observation[key].ndim != 0 or observation[key].dtype.kind not in "US":
            raise ValueError(f"{key} must be a scalar string")
        observation[key] = str(observation[key].item())
    if observation["format"] != FORMAT or not observation["instruction"].strip():
        raise ValueError("Unknown observation format or empty instruction")
    names = observation["joint_names"]
    if names.dtype.kind not in "US" or names.tolist() != JOINT_NAMES:
        raise ValueError("Observation joint order differs from the checkpoint")
    for key, shape in (("head", (384, 512, 3)), ("wrist_left", (256, 256, 3)), ("wrist", (256, 256, 3))):
        if observation[key].shape != shape or observation[key].dtype != np.uint8:
            raise ValueError(f"{key} must be RGB uint8 with shape {shape}")
    state = observation["joint_position"]
    if state.shape != (8,) or state.dtype.kind != "f" or not np.isfinite(state).all():
        raise ValueError("Expected eight finite floating-point current joint positions")
    observation["joint_position"] = state.astype(np.float32)
    if not np.isfinite(observation["joint_position"]).all():
        raise ValueError("Current joint positions overflow float32")
    timestamp = observation["source_time_ns"]
    if timestamp.ndim != 0 or timestamp.dtype.kind not in "iu" or int(timestamp) < 0:
        raise ValueError("source_time_ns must be a nonnegative scalar integer")
    return observation


def prepare_tensors(observation: dict, contract: dict) -> dict:
    import torch
    from bg2_cosmos.data import quantile_affine
    from bg2_cosmos.vision_contract import compose_views

    views = [torch.from_numpy(observation[key].copy()).permute(2, 0, 1) for key in ("head", "wrist_left", "wrist")]
    normalized = quantile_affine(observation["joint_position"], contract["normalization"]["joint_position"])
    # Row zero is observed. Every future row starts as a placeholder, with no labels read.
    action = torch.zeros((17, 64), dtype=torch.float32)
    action[0, :8] = torch.from_numpy(normalized.copy()).float()
    raw = torch.zeros((17, 8), dtype=torch.float32)
    raw[0] = torch.from_numpy(observation["joint_position"].copy())
    if not torch.isfinite(action).all():
        raise ValueError("Nonfinite normalized observation")
    return {"video": compose_views(*views), "action": action, "action_raw": raw,
            "raw_action_dim": torch.tensor(8), "domain_id": torch.tensor(33)}


def build_batch(observation: dict, contract: dict, tokenizer):
    import torch
    from bg2_cosmos.vision_contract import build_policy_prompt
    from cosmos_framework.data.generator.action.utils.action_processing import ActionProcessingRecord
    from cosmos_framework.data.generator.joint_dataloader import custom_collate_fn
    from cosmos_framework.data.generator.sequence_packing import SequencePlan

    sample = prepare_tensors(observation, contract)
    prompt = build_policy_prompt(observation["instruction"])
    tokens = tokenizer.tokenize_text(prompt, is_video=False, use_system_prompt=False)
    if not tokens:
        raise ValueError("Tokenizer returned no instruction tokens")
    sample.update({
        "__key__": str(int(observation["source_time_ns"])), "mode": "policy", "ai_caption": prompt,
        "text_token_ids": torch.tensor(tokens, dtype=torch.long),
        "conditioning_fps": torch.tensor(20, dtype=torch.long),
        "image_size": torch.tensor([640., 512., 640., 512.]),
        "sequence_plan": SequencePlan(has_text=True, has_vision=True, has_action=True,
                                      condition_frame_indexes_vision=[0], condition_frame_indexes_action=[0],
                                      action_start_frame_offset=0),
        # Native generation requires this inverse record. Training already applied the
        # custom affine, so return eight normalized channels and invert explicitly once.
        "action_processing_record": ActionProcessingRecord(raw_action_dim=8, action_normalizer=None),
    })
    batch = custom_collate_fn([sample])
    batch["dataset_name"] = ["bg2_room"]
    for key in ("text_token_ids", "video", "action", "action_raw"):
        batch[key] = [[value] for value in batch[key]]
    for key in ("image_size", "conditioning_fps"):
        batch[key] = [value.unsqueeze(0) for value in batch[key]]
    return batch, prompt


def decode_actions(normalized: np.ndarray, observation: dict, contract: dict) -> np.ndarray:
    """Decode native output (including current row) into 16 absolute joint targets."""
    if normalized.shape != (17, 8) or not np.isfinite(normalized).all():
        raise ValueError("Expected finite native normalized actions with shape (17, 8)")
    stats = contract["normalization"]["joint_position"]
    lo, hi = np.asarray(stats["q01"], dtype=np.float64), np.asarray(stats["q99"], dtype=np.float64)
    scale = np.maximum((hi - lo) / 2, 1e-6)
    absolute = normalized.astype(np.float64) * scale + (hi + lo) / 2
    if not np.allclose(absolute[0], observation["joint_position"], rtol=2e-5, atol=2e-5):
        raise ValueError("Native generation changed the conditioned current-state row")
    targets = absolute[1:].astype(np.float32)
    if not np.isfinite(targets).all():
        raise ValueError("Nonfinite decoded joint positions")
    return targets
