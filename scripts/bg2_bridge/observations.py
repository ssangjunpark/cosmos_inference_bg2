"""Timestamp-aligned three-camera observations with the original PIL pixel transforms."""
from collections import deque
from io import BytesIO
from threading import RLock

import numpy as np
from PIL import Image, ImageOps

from bg2_inference import FORMAT, JOINT_NAMES
from .core import BridgeError, require, validate_profile

# Verified training/cosmos_futurejoint/tools/prepare_threeview.py contract.
CAMERAS = {"head": ((128, 0, 672, 376), (512, 384)),
           "wrist": ((0, 184, 240, 424), (256, 256)),
           "wrist_left": ((0, 184, 240, 424), (256, 256))}


def decode_view(payload, key, spec):
    require(0 < len(payload) <= 16 * 1024 * 1024, "Invalid compressed image size")
    crop, size = CAMERAS[key]
    require(tuple(spec["crop_xyxy"]) == crop and tuple(spec["output_wh"]) == size, "Camera crop/resize differs from training")
    rotation = spec.get("live_to_training_rotation_deg")
    require(rotation in (0, 90, 180, 270) and spec.get("raw_dimensions_verified") is True, f"{key}: camera transform is unverified")
    with Image.open(BytesIO(payload)) as source:
        require(list(source.size) == spec.get("raw_wh"), f"{key}: raw image dimensions changed")
        image = source.convert("RGB").rotate(rotation, expand=True)
        require(crop[2] <= image.width and crop[3] <= image.height, f"{key}: crop exceeds input image")
        image = image.crop(crop)
        image = (ImageOps.pad(image, size, method=Image.Resampling.BILINEAR, color=(0, 0, 0), centering=(.5, .5))
                 if key == "head" else image.resize(size, Image.Resampling.BILINEAR))
        return np.array(image, dtype=np.uint8, copy=True)


class Observations:
    def __init__(self, profile, max_samples=64):
        self.profile = profile
        self.lock = RLock()
        self.joints = deque(maxlen=max_samples)
        self.cameras = {key: deque(maxlen=max_samples) for key in CAMERAS}
        self.error = None

    def add_joints(self, frame):
        with self.lock:
            if self.joints:
                require(frame.source_ns > self.joints[-1].source_ns, "Joint timestamps must increase")
            self.joints.append(frame)

    def add_camera(self, key, source_ns, received_ns, payload):
        require(source_ns > 0 and 0 < len(payload) <= 16 * 1024 * 1024, "Invalid camera timestamp or payload")
        with self.lock:
            frames = self.cameras[key]
            require(not frames or source_ns > frames[-1][0], "Camera timestamps must increase")
            frames.append((int(source_ns), int(received_ns), bytes(payload)))

    def latest_source_ns(self):
        with self.lock:
            return self.joints[-1].source_ns if self.joints else 0

    def latest_feedback(self, now_ns):
        with self.lock:
            require(bool(self.joints), "Waiting for joint feedback")
            frame = self.joints[-1]
        frame.fresh(now_ns)
        return frame

    def snapshot(self, instruction, now_ns, *, receipt_now_ns=None):
        receipt_now_ns = now_ns if receipt_now_ns is None else receipt_now_ns
        validate_profile(self.profile, live=True)
        require(isinstance(instruction, str) and instruction.strip(), "Instruction is empty")
        with self.lock:
            require(self.error is None, f"Observation callback failed: {self.error}")
            require(self.joints and all(self.cameras.values()), "Waiting for all three cameras and joints")
            chosen = None
            for joint in reversed(self.joints):
                if now_ns - joint.source_ns > 100_000_000:
                    break
                frames = {key: min(items, key=lambda f: (abs(f[0] - joint.source_ns), f[0])) for key, items in self.cameras.items()}
                if all(abs(f[0] - joint.source_ns) <= 25_000_000 for f in frames.values()):
                    chosen = joint, frames
                    break
            require(chosen is not None, "Camera/joint synchronization exceeds 25 ms")
            joint, frames = chosen
        joint.fresh(now_ns, receipt_now_ns=receipt_now_ns)
        require(all(0 <= receipt_now_ns - f[1] <= 100_000_000 and -25_000_000 <= now_ns - f[0] <= 100_000_000
                    for f in frames.values()), "Stale camera or unsynchronized robot clock")
        result = {key: decode_view(frame[2], key, self.profile["cameras"][key]) for key, frame in frames.items()}
        result.update(joint_position=np.asarray([joint.positions[n] for n in JOINT_NAMES], dtype=np.float32),
                      source_time_ns=np.asarray(joint.source_ns, dtype=np.int64), instruction=np.asarray(instruction),
                      joint_names=np.asarray(JOINT_NAMES), format=np.asarray(FORMAT))
        return result
