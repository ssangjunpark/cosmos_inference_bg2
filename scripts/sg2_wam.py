"""SG2 training coordinates, camera geometry, and saved WAM publications."""
import json
from pathlib import Path
import sys

import numpy as np

from bg2_bridge.core import SG2, require, validate_profile
from bg2_wam import decode_actions, exported_formatter, prepare_tensors, restore_wam_config
from wam_checkpoint import FRAMEWORK_COMMIT, file_hash
from wam_runtime import NativeWAM

ROOT = Path(__file__).resolve().parents[1]
VARIANTS = ("pnp1_normal", "pnp1_abnormal", "pnp2_normal", "pnp2_abnormal")
ARM_NAMES = [name for side in ("l", "r") for name in
             [*(f"arm_{side}_joint{i}" for i in range(1, 8)), f"gripper_{side}_joint1"]]
FORMAT = "cosmos_sg2_measured_radians_threeview_v1"
DATA_FORMAT = "cosmos_sg2_dataset_threeview_v1"


def read(path):
    return json.loads(Path(path).read_text())


def settings(variant):
    spec = read(ROOT / "profiles/sg2_wam.json")[variant]
    names = ARM_NAMES[8:] if variant.startswith("pnp1_") else ARM_NAMES
    require(spec["joint_names"] == names, "SG2 policy joint order mismatch")
    return spec


def site_profile(args):
    profile = read(args.robot_profile)
    require(profile["robot"] == SG2, "Expected an SG2 robot profile")
    validate_profile(profile, companion=True, execute=args.command == "execute")
    topics = [args.head_topic, args.left_topic, args.right_topic]
    require(len(set(topics)) == 3 and all(t.startswith("/") for t in topics), "Specify three distinct absolute camera topics")
    require(0 < args.port <= 65535 and 0 <= args.domain < 2**32, "Invalid Zenoh port or ROS domain")
    profile["middleware"] = dict(configured_router_host=args.router, configured_router_port=args.port, domain_id=args.domain)
    profile["cameras"] = camera_specs(topics)
    return profile


def camera_specs(topics):
    cameras = {}
    for key, topic in zip(("head", "wrist_left", "wrist"), topics):
        head = key == "head"
        # decode_view rotates before cropping. (0,184,240,424) in the rotated
        # wrist is the raw (0,0,240,240) square after a 90-degree CCW turn.
        cameras[key] = dict(topic=topic, msg_type="sensor_msgs/msg/CompressedImage",
            raw_wh=[672, 376] if head else [424, 240], raw_dimensions_verified=True,
            live_to_training_rotation_deg=0 if head else 90,
            crop_xyxy=[112, 88, 560, 376] if head else [0, 184, 240, 424],
            output_wh=[512, 384] if head else [256, 256])
    return cameras


def calibration(path, names):
    """Controller radians = dataset value * scale + offset, in selected order."""
    data = read(path)
    require(isinstance(data, dict) and set(names) <= set(data) <= set(ARM_NAMES),
            "Calibration must identify every predicted joint")
    for name in names:
        row = data[name]
        require(isinstance(row, dict) and set(row) == {"scale", "offset"} and
                all(type(v) in (int, float) and np.isfinite(v) for v in row.values()) and row["scale"] != 0,
                f"Invalid controller calibration for {name}")
    return tuple(np.array([data[n][k] for n in names], dtype=np.float64) for k in ("scale", "offset"))


def training_observation(measured, spec, conversion=None):
    names = spec["joint_names"]
    controller = str(measured["format"]) == FORMAT
    input_names = np.asarray(measured["joint_names"]).tolist()
    require(isinstance(input_names, list) and all(isinstance(n, str) for n in input_names) and
            len(set(input_names)) == len(input_names) and set(names) <= set(input_names), "Incomplete named SG2 state")
    require((controller and input_names == ARM_NAMES) or str(measured["format"]) == DATA_FORMAT,
            "Unknown SG2 observation format or controller order")
    q = np.asarray(measured["joint_position"])
    require(q.shape == (len(input_names),) and q.dtype.kind == "f" and np.isfinite(q).all(), "Invalid SG2 joint state")
    for key, shape in (("head", (384, 512, 3)), ("wrist_left", (256, 256, 3)), ("wrist", (256, 256, 3))):
        frame = measured[key]
        require(frame.shape == shape and frame.dtype == np.uint8, f"Invalid processed SG2 view: {key}")
    stamp = np.asarray(measured["source_time_ns"])
    require(stamp.shape == () and stamp.dtype.kind in "iu" and int(stamp) >= (1 if controller else 0), "Invalid source timestamp")
    selected = q[[input_names.index(n) for n in names]]
    if controller:
        require(conversion is not None, "Controller feedback requires a verified dataset-to-radian calibration")
        scale, offset = conversion
        selected = (selected - offset) / scale
    return {**measured, "joint_position": selected, "joint_names": names, "instruction": spec["instruction"]}


def controller_positions(predicted, measured, names, conversion):
    """Unselected PNP1 joints hold the robot's measured pose for this cycle."""
    require(predicted.shape == (16, len(names)) and np.isfinite(predicted).all(), "Invalid SG2 action horizon")
    result = np.tile(np.asarray(measured["joint_position"], dtype=np.float64), (16, 1))
    scale, offset = conversion
    result[:, [ARM_NAMES.index(n) for n in names]] = predicted * scale + offset
    require(result.shape == (16, 16) and np.isfinite(result).all(), "Invalid controller positions")
    return result


def inspect_weights(checkpoint, spec):
    """Validate the training export's sidecars, receipt, and DCP storage spans."""
    from torch.distributed.checkpoint import FileSystemReader

    for name, digest in spec["source_sha256"].items():
        require(file_hash(checkpoint / name) == digest, f"Checkpoint does not match the selected SG2 run: {name}")
    ready = read(checkpoint / "READY.json")
    require(ready.get("format") == "pytorch_distributed_checkpoint" and ready.get("checkpoint") == "model"
            and type(ready.get("step")) is int and ready["step"] > 0, "Expected an SG2 weights-only publication")
    require(Path(ready.get("run", "")).name == spec["run_name"], "Weight publication belongs to another SG2 run")
    model = (checkpoint / "model").resolve(strict=True)
    metadata = FileSystemReader(str(model)).read_metadata()
    require(bool(metadata.storage_data), "DCP metadata contains no storage entries")
    ends = {}
    for entry in metadata.storage_data.values():
        require(entry.offset >= 0 and entry.length > 0, "Invalid DCP storage range")
        ends[entry.relative_path] = max(ends.get(entry.relative_path, 0), entry.offset + entry.length)
    for name, end in ends.items():
        shard = (model / name).resolve(strict=True)
        shard.relative_to(model)
        require(shard.is_file() and shard.stat().st_size >= end, f"Incomplete SG2 shard: {shard.name}")
    manifest = checkpoint / "VERIFIED.json"
    if not manifest.is_file():
        require(ready.get("weights_only") is True and ready.get("native_save_complete") is True and
                ready.get("verified") is True, "Missing SG2 milestone weight receipt or verified last-snapshot receipt")
    if manifest.is_file():
        record = read(manifest)
        require(record["step"] == ready["step"], "Weight receipt step mismatch")
        listed = set()
        for entry in record["files"]:
            file = (checkpoint / entry["path"]).resolve(strict=True)
            file.relative_to(model)
            require(file not in listed and file.stat().st_size == entry["bytes"] and file_hash(file) == entry["sha256"],
                    f"SG2 weight receipt mismatch: {entry['path']}")
            listed.add(file)
        require(listed == {p.resolve() for p in model.rglob("*") if p.is_file()}, "Incomplete SG2 weight receipt")
    return ready


class SG2WAM(NativeWAM):
    dataset_name = "sg2"

    def __init__(self, checkpoint, tokenizer, vae, variant):
        from importlib import import_module

        spec = settings(variant)
        ready = inspect_weights(checkpoint, spec)
        self.joint_names = spec["joint_names"]
        self.contract = read(checkpoint / "input_contract.json")
        require(self.contract["framework_commit"] == FRAMEWORK_COMMIT, "Wrong Cosmos framework revision")
        self.experiment = {"training_mode": "wam", "action": {"names": self.joint_names, "domain_id": 33}}
        adapter = ROOT / "sg2_setup/adapter"
        sys.path.insert(0, str(adapter))
        source = Path(import_module("bg2_cosmos.model").__file__).resolve()
        require(source == (adapter / "bg2_cosmos/model.py").resolve(), "A different training model adapter is imported")
        self.config_overrides = []
        self.load_model(checkpoint / "config.yaml", checkpoint / "model", tokenizer, vae, "bg2_cosmos.model.BG2PolicyModel",
                        lambda model: restore_wam_config(model, self.config_overrides))
        self.formatter = exported_formatter(checkpoint)
        self.runtime_metadata = {"variant": variant, "step": ready["step"], "source_sha256": spec["source_sha256"]}

    def decode(self, normalized, observation):
        return decode_prediction(normalized, observation, self.contract, self.joint_names)


def decode_prediction(normalized, observation, contract, names):
    # Native inference includes the clean current row. Reject a changed layout.
    if normalized.shape == (17, 64):
        normalized = normalized[:, :len(names)]
    return decode_actions(normalized, observation, contract, joint_names=names)


def check_handoff(root, variant):
    """Exercise inference preprocessing on the supplied training-loader example."""
    source = root / "sources" / variant
    spec = settings(variant)
    for name, digest in spec["source_sha256"].items():
        require(file_hash(source / "run" / name) == digest, f"Training handoff mismatch: {name}")
    contract = read(source / "run/input_contract.json")
    profile = read(root / "inference_profiles" / (variant + ".json"))
    sample = read(source / "golden_sample.json")
    mapping = read(root / "ACTION_SPACE.json")["variants"][variant]
    names = spec["joint_names"]
    require(names == profile["joint_names"] == sample["joint_names"] and
            spec["instruction"] == profile["instruction"] and sample["prompt"] == profile["prompt_example"],
            "Training handoff names or prompt mismatch")
    stats = contract["normalization"]["joint_position"]
    require(all(stats[k] == spec[k] for k in ("q01", "q99")), "Training quantile mismatch")
    measured = dict(format=DATA_FORMAT, joint_names=mapping["raw_state_names"],
                    joint_position=np.array(sample["raw_full_state_first"]), source_time_ns=np.int64(sample["source_time_ns_first17"][0]),
                    head=np.zeros((384, 512, 3), np.uint8), wrist_left=np.zeros((256, 256, 3), np.uint8),
                    wrist=np.zeros((256, 256, 3), np.uint8))
    observation = training_observation(measured, spec)
    experiment = {"training_mode": "wam", "action": {"names": names, "domain_id": 33}}
    tensors = prepare_tensors(observation, contract, experiment, exported_formatter(source / "run"), joint_names=names)
    reference = np.asarray(sample["action_normalized_padded"])
    np.testing.assert_allclose(tensors["action"][0], reference[0], rtol=2e-6, atol=2e-6)
    require(not tensors["action"][1:].any() and not tensors["video"][:, 1:].any(), "Future inputs must be placeholders")
    positions = decode_prediction(reference, observation, contract, names)
    np.testing.assert_allclose(positions, sample["action_raw"][1:], rtol=2e-6, atol=2e-6)
    return {"reference": "training loader", "future_rows_compared": len(positions), "model_width": len(names)}
