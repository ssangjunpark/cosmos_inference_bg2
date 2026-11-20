"""Resident SG2 services, parallel CM1/CM2 selection, and a latched CM2 mode."""

import json
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from urllib.request import Request, urlopen

import numpy as np
from PIL import Image

from .jepa_gate import MAX_BODY, JepaClient, JepaGate, finite_positive, local_url


def video_view(frames, view="head", size=(256, 192)):
    frames = JepaGate.frames(frames)
    if frames.shape[1:3] != (640, 512):
        raise ValueError("Expected native 640x512 three-view WAM video")
    crops = {
        "head": (0, 0, 512, 384),
        "wrist_left": (0, 384, 256, 640),
        "wrist": (256, 384, 512, 640),
    }
    if view not in crops or len(size) != 2 or min(size) < 1 or max(size) > 512:
        raise ValueError("Invalid JEPA camera/view size")
    return np.stack(
        [
            np.asarray(
                Image.fromarray(frame)
                .crop(crops[view])
                .resize(tuple(size), Image.Resampling.BILINEAR)
            )
            for frame in frames
        ]
    )


def pack(arrays):
    stream = BytesIO()
    np.savez(stream, **arrays)
    payload = stream.getvalue()
    if len(payload) > MAX_BODY:
        raise ValueError("Oversized SG2 NPZ")
    return payload


def unpack(payload):
    if not 0 < len(payload) <= MAX_BODY:
        raise ValueError("Invalid SG2 NPZ size")
    with zipfile.ZipFile(BytesIO(payload)) as archive:
        if sum(item.file_size for item in archive.infolist()) > MAX_BODY:
            raise ValueError("Expanded SG2 NPZ exceeds limit")
    with np.load(BytesIO(payload), allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


class SG2PolicyClient:
    def __init__(self, url, timeout, model_id):
        self.url, self.timeout, self.model_id = (
            local_url(url),
            finite_positive(timeout),
            model_id,
        )

    def health(self):
        with urlopen(self.url + "/health", timeout=self.timeout) as response:
            return json.loads(response.read(128 * 1024))

    def predict(self, observation, epoch, request_id):
        data = {
            key: value
            for key, value in observation.items()
            if key
            in {
                "head",
                "wrist_left",
                "wrist",
                "joint_position",
                "joint_names",
                "source_time_ns",
                "format",
            }
        }
        data["metadata"] = np.asarray(
            json.dumps(
                {"epoch": epoch, "request_id": request_id, "model_id": self.model_id}
            )
        )
        request = Request(
            self.url + "/predict",
            data=pack(data),
            headers={"Content-Type": "application/x-npz"},
        )
        with urlopen(request, timeout=self.timeout) as response:
            arrays = unpack(response.read(MAX_BODY + 1))
        result = json.loads(str(arrays["metadata"]))
        if result.get("model_id") != self.model_id:
            raise ValueError("SG2 policy model mismatch")
        result["future"] = arrays["future"]
        return result

    def cancel(self):
        # Bounded socket timeouts; Gateway.closed prevents late forwarding.
        pass


class ParallelLatchedGate(JepaGate):
    """Both models infer before CM1 scoring; CM2 selection stays latched.

    Once latched only CM2 is requested, so CM1/JEPA outages cannot affect it.
    Recreate the object/process to reset. Epoch changes do not reset the mode.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fallback_active = False
        self.last_result = None
        self.last_observation = None
        self.switch_loss = None

    def validate(self, result, observation, epoch, request_id):
        actions = super().validate(result, observation, epoch, request_id)
        if (
            actions.shape != (16, 16)
            or result.get("schema") != "cosmos_sg2_joint_chunk_v1"
        ):
            raise ValueError(
                "SG2 requires sixteen targets for sixteen controller coordinates"
            )
        future = self.frames(result["future"])
        expected_size = tuple(reversed(self.config["video_size"]))
        if future.shape != (16, *expected_size, 3):
            raise ValueError(
                "Expected sixteen future frames in the configured JEPA view"
            )
        return actions

    def predict(self, observation, epoch, request_id):
        self.last_result = None
        self.last_observation = observation
        started = time.monotonic()
        base = {
            "observation_id": request_id,
            "epoch": epoch,
            "threshold": self.threshold,
            "primary_id": self.config["primary_id"],
            "selected_model": None,
            "loss": None,
            "fallback_active": self.fallback_active,
            "switch_loss": self.switch_loss,
        }
        self.emit(**base, state="inferring")
        try:
            if int(observation["frame_interval_ns"]) != self.interval:
                raise ValueError("Observation interval mismatch")
            joints = np.asarray(observation["joint_position"], dtype=float)
            if (
                joints.shape != (len(self.names),)
                or not np.isfinite(joints).all()
                or list(observation["joint_names"]) != self.names
            ):
                raise ValueError("Observed SG2 joint order/values mismatch")
            past = self.frames(observation["past"])
            if self.fallback_active:
                result = self.fallback.predict(observation, epoch, request_id)
                reason = "fallback_latched"
            else:
                with ThreadPoolExecutor(max_workers=2) as pool:
                    first = pool.submit(
                        self.primary.predict, observation, epoch, request_id
                    )
                    second = pool.submit(
                        self.fallback.predict, observation, epoch, request_id
                    )
                    primary, fallback = first.result(), second.result()
                for candidate in (primary, fallback):
                    self.validate(candidate, observation, epoch, request_id)
                    future = self.frames(candidate["future"])
                    if future.shape[1:] != past.shape[1:]:
                        raise ValueError("Observed/generated camera geometry mismatch")
                loss = float(
                    self.judge.score(
                        past,
                        self.frames(primary["future"]),
                        request_id,
                        self.config["primary_id"],
                    )
                )
                if not np.isfinite(loss) or loss < 0:
                    raise ValueError("Invalid JEPA raw loss")
                base["loss"] = loss
                if loss > self.threshold:
                    self.fallback_active, self.switch_loss = True, loss
                    result, reason = fallback, "threshold_exceeded"
                else:
                    result, reason = primary, "threshold_passed"
            self.validate(result, observation, epoch, request_id)
            future = self.frames(result["future"])
            if future.shape[1:] != past.shape[1:]:
                raise ValueError("Selected camera geometry mismatch")
            self.last_result = result
            base.update(
                selected_model=self.config["fallback_id"]
                if self.fallback_active
                else self.config["primary_id"],
                fallback_active=self.fallback_active,
                switch_loss=self.switch_loss,
            )
            self.emit(
                **base,
                state="selected",
                reason=reason,
                seconds=time.monotonic() - started,
            )
            return {**result, "selection": dict(self.last_event)}
        except Exception as error:
            self.emit(
                **base,
                state="discarded",
                reason=str(error),
                seconds=time.monotonic() - started,
            )
            raise

    def execution_event(self, state, reason=None):
        if self.last_event is not None:
            self.emit(
                **{
                    **self.last_event,
                    "state": state,
                    **({"reason": reason} if reason else {}),
                }
            )


def from_config(config, event=None, check_health=True):
    if config.get("selection_mode") != "parallel_latched":
        raise ValueError("SG2 dual mode requires selection_mode=parallel_latched")
    primary = SG2PolicyClient(
        config["primary_url"], config["timeout_seconds"], config["primary_id"]
    )
    fallback = SG2PolicyClient(
        config["fallback_url"], config["timeout_seconds"], config["fallback_id"]
    )
    if check_health:
        for client, variant in (
            (primary, config["primary_variant"]),
            (fallback, config["fallback_variant"]),
        ):
            health = client.health()
            for key, expected in {
                "model_id": client.model_id,
                "variant": variant,
                "joint_names": config["joint_names"],
                "units": config["units"],
                "frame_interval_ns": config["frame_interval_ns"],
                "video_view": config["video_view"],
                "video_size": config["video_size"],
            }.items():
                if health.get(key) != expected:
                    raise ValueError(
                        f"{client.model_id}: service contract mismatch for {key}"
                    )
            if health.get("ready") is not True:
                raise ValueError("SG2 model service not ready")
    return ParallelLatchedGate(
        primary,
        fallback,
        JepaClient(config["jepa_url"], config["timeout_seconds"]),
        config,
        event,
    )
