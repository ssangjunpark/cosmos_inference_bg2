"""Read-only validation for the delivered BG2 WAM checkpoints."""
import hashlib
import json
from pathlib import Path

FRAMEWORK_COMMIT = "b98934eda43f4d31972538a527f8663f003c27dc"
DEFAULT_ROOT = Path("/mnt/sata1/RobotEscapeCkpt/cosmos3_wam_room20k_20261105")


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_experiment(path, contract):
    experiment = json.loads(Path(path).read_text())
    action = experiment["action"]
    clip = experiment["clip_audit"]
    if (experiment["training_mode"] != "wam" or
            experiment["framework_commit"] != FRAMEWORK_COMMIT or
            experiment["data"]["sha256"]["manifest"] != contract["data_manifest_sha256"]):
        raise ValueError("WAM experiment/checkpoint provenance mismatch")
    if (action["horizon"], action["fps"], action["dim"], action["domain_id"],
            action["num_embodiment_domains"], clip["frames"], clip["future_actions"]) != (16, 20, 8, 33, 34, 17, 16):
        raise ValueError("Unsupported WAM action/video contract")
    if clip.get("passed") is not True or clip.get("frame_interval_ns") != 50_000_000 or action.get("first_offset") != 1:
        raise ValueError("WAM clip audit or first-target timing mismatch")
    if action["names"] != contract["normalization"]["joint_names"]:
        raise ValueError("WAM experiment/checkpoint joint order mismatch")
    return experiment


def validate_checkpoint(checkpoint, variant=None, step=20000, hashes=False):
    """Never creates receipts or edits the supplied checkpoint directory."""
    checkpoint = Path(checkpoint).resolve(strict=True)
    ready = json.loads((checkpoint / "READY.json").read_text())
    transfer = json.loads((checkpoint / "TRANSFER_VERIFIED.json").read_text())
    manifest = json.loads((checkpoint / "VERIFIED.json").read_text())
    contract = json.loads((checkpoint / "input_contract.json").read_text())
    tier = variant or ready.get("model_tier")
    if (tier not in {"edge", "nano"} or ready.get("model_tier") != tier or
            ready.get("variant") != tier + "_wam" or ready.get("code_variant") != "cosmos_wam" or
            ready.get("step") != step or ready.get("format") != "pytorch_distributed_checkpoint"):
        raise ValueError("Checkpoint is not the requested BG2 WAM tier/step")
    if (transfer.get("state") != "complete" or transfer.get("verified") is not True or
            transfer.get("sha256_verified") is not True or transfer.get("tier") != tier or
            transfer.get("step") != step):
        raise ValueError("WAM transfer is not verified complete")
    if manifest.get("format") != "pytorch_distributed_checkpoint" or manifest.get("step") != step:
        raise ValueError("Invalid WAM weight manifest")
    if contract.get("framework_commit") != FRAMEWORK_COMMIT:
        raise ValueError("WAM framework revision mismatch")
    if contract["vision"]["frame_interval_ns"] != 50_000_000 or contract["vision"]["first_target_offset_frames"] != 1:
        raise ValueError("Unexpected WAM prediction timing")
    listed = set()
    for entry in manifest["files"]:
        relative = Path(entry["path"])
        if relative.is_absolute() or ".." in relative.parts or relative.parts[:1] != ("model",):
            raise ValueError("Unsafe WAM manifest path")
        name = relative.as_posix()
        if name in listed:
            raise ValueError("Duplicate WAM manifest entry")
        listed.add(name)
        source = (checkpoint / relative).resolve(strict=True)
        source.relative_to(checkpoint)
        if not source.is_file() or source.stat().st_size != entry["bytes"]:
            raise ValueError(f"Incomplete WAM checkpoint: {relative}")
        # Unused distributed ranks legitimately have zero-byte shards.
        if hashes and file_hash(source) != entry["sha256"]:
            raise ValueError(f"WAM checksum mismatch: {relative}")
    actual = {"model/.metadata"} | {p.relative_to(checkpoint).as_posix() for p in (checkpoint / "model").glob("*.distcp")}
    if listed != actual or not (checkpoint / "model/.metadata").is_file():
        raise ValueError("WAM shard set differs from VERIFIED.json")
    if not any(p.stat().st_size > 0 for p in (checkpoint / "model").glob("*.distcp")):
        raise ValueError("WAM checkpoint has no weight data")
    if not (checkpoint / "vision_contract.py").is_file():
        raise ValueError("Missing exported WAM vision/prompt formatter")
    import yaml
    model = yaml.safe_load((checkpoint / "config.yaml").read_text())["model"]
    cfg = model["config"]
    if (model["_target_"] != "bg2_cosmos.model.BG2PolicyModel" or
            cfg["tokenizer"]["encode_exact_durations"] != [17] or
            cfg["num_embodiment_domains"] != 34 or cfg.get("max_action_dim") != 64 or
            not cfg.get("vision_gen") or not cfg.get("action_gen") or
            cfg.get("action_global_asinh") or cfg.get("action_metric_gripper") or
            cfg["rectified_flow_training_config"]["image_loss_scale"] <= 0):
        raise ValueError("Saved model configuration differs from the BG2 WAM contract")
    return {"tier": tier, "ready": ready, "contract": contract, "manifest": manifest,
            "transfer": transfer, "full_checksums_checked": bool(hashes)}


def published_wam_checkpoint(root, variant, step=20000, hashes=False):
    root = Path(root).resolve(strict=True)
    checkpoint = (root / variant / f"{step:06d}").resolve(strict=True)
    checkpoint.relative_to(root)
    validate_checkpoint(checkpoint, variant, step, hashes)
    return checkpoint
