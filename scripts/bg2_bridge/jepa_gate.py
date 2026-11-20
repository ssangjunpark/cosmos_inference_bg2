"""Opt-in pre-execution selection, independent of robot joint count and torch."""

import json
import math
import time
from pathlib import Path
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

import numpy as np

MAX_BODY = 64 * 1024 * 1024


def finite_positive(value):
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError("Expected finite positive value")
    return value


def local_url(url):
    parsed = urlsplit(url)
    if (
        parsed.scheme != "http"
        or parsed.hostname not in ("localhost", "127.0.0.1", "::1")
        or parsed.username
        or parsed.password
    ):
        raise ValueError("Use localhost HTTP services, optionally through SSH tunnels")
    return url.rstrip("/")


def post(url, value, timeout):
    payload = json.dumps(value, allow_nan=False).encode()
    if len(payload) > MAX_BODY:
        raise ValueError("Request too large; resize replay clips before submission")
    with urlopen(
        Request(url, data=payload, headers={"Content-Type": "application/json"}),
        timeout=timeout,
    ) as response:
        body = response.read(MAX_BODY + 1)
    if len(body) > MAX_BODY:
        raise ValueError("Reply too large")
    return json.loads(body)


class JepaClient:
    def __init__(self, url, timeout):
        self.url, self.timeout = local_url(url), finite_positive(timeout)

    def score(self, past, future, observation_id, model_id):
        reply = post(
            self.url + "/score",
            {
                "past": past.tolist(),
                "future": future.tolist(),
                "observation_id": observation_id,
                "model_id": model_id,
            },
            self.timeout,
        )
        if (
            reply.get("observation_id") != observation_id
            or reply.get("model_id") != model_id
        ):
            raise ValueError("JEPA reply identity mismatch")
        loss = float(reply["loss"])
        if not math.isfinite(loss) or loss < 0:
            raise ValueError("JEPA raw loss must be finite and nonnegative")
        return loss


class JsonPolicy:
    """SG2 adapter contract: POST /predict -> positions, joint_names, units,
    frame_interval_ns, future (THWC RGB, context already excluded), and echoed IDs.
    Deliberately separate from the BG2-only NPZ service contract.
    """

    def __init__(self, url, timeout, model_id):
        self.url, self.timeout, self.model_id = (
            local_url(url),
            finite_positive(timeout),
            model_id,
        )

    def predict(self, observation, epoch, request_id):
        data = {
            key: value.tolist() if isinstance(value, np.ndarray) else value
            for key, value in observation.items()
        }
        result = post(
            self.url + "/predict",
            {
                "observation": data,
                "epoch": epoch,
                "request_id": request_id,
                "model_id": self.model_id,
            },
            self.timeout,
        )
        if result.get("model_id") != self.model_id:
            raise ValueError("Policy model identity mismatch")
        return result

    def cancel(self):
        pass


class JepaGate:
    def __init__(self, primary, fallback, judge, config, event=None):
        self.primary, self.fallback, self.judge = primary, fallback, judge
        self.config, self.event = dict(config), event or (lambda value: None)
        self.threshold = float(config["threshold"])
        if not math.isfinite(self.threshold) or self.threshold < 0:
            raise ValueError("Set an explicit finite nonnegative raw-loss threshold")
        self.interval = int(config["frame_interval_ns"])
        finite_positive(self.interval)
        self.names, self.units = list(config["joint_names"]), list(config["units"])
        if (
            not self.names
            or len(set(self.names)) != len(self.names)
            or len(self.units) != len(self.names)
        ):
            raise ValueError(
                "Explicit unique joint order and corresponding units required"
            )
        if config["primary_id"] == config["fallback_id"]:
            raise ValueError("Model IDs must differ")
        self.last_event = None

    def emit(self, **value):
        self.last_event = value
        self.event(value)

    def validate(self, result, observation, epoch, request_id):
        if (
            result.get("epoch") != epoch
            or result.get("request_id") != request_id
            or result.get("source_time_ns") != int(observation["source_time_ns"])
        ):
            raise ValueError("Policy reply observation/session mismatch")
        if (
            result.get("joint_names") != self.names
            or result.get("units") != self.units
            or result.get("frame_interval_ns") != self.interval
        ):
            raise ValueError("Policy action coordinate/time contract mismatch")
        actions = np.asarray(result["positions"], dtype=float)
        if (
            actions.ndim != 2
            or actions.shape[0] < 1
            or actions.shape[1] != len(self.names)
            or not np.isfinite(actions).all()
        ):
            raise ValueError("Invalid policy action shape/values")
        return actions

    @staticmethod
    def frames(value):
        x = np.asarray(value)
        if (
            x.ndim != 4
            or x.shape[-1] != 3
            or min(x.shape) < 1
            or not np.isfinite(x).all()
            or np.any(x < 0)
            or np.any(x > 255)
            or np.any(x != np.floor(x))
        ):
            raise ValueError("Expected nonempty uint8-compatible THWC RGB")
        return x.astype(np.uint8)

    def predict(self, observation, epoch, request_id):
        started = time.monotonic()
        base = {
            "observation_id": request_id,
            "epoch": epoch,
            "threshold": self.threshold,
            "primary_id": self.config["primary_id"],
            "loss": None,
            "selected_model": None,
        }
        self.emit(**base, state="inferring")
        try:
            if int(observation["frame_interval_ns"]) != self.interval:
                raise ValueError(
                    "Observed video sampling interval differs from generated future"
                )
            if "joint_position" in observation:
                joints = np.asarray(observation["joint_position"], dtype=float)
                if (
                    joints.shape != (len(self.names),)
                    or not np.isfinite(joints).all()
                    or list(observation.get("joint_names", [])) != self.names
                ):
                    raise ValueError(
                        "Observed joint coordinates differ from configured policy"
                    )
            past = self.frames(observation["past"])
            result = self.primary.predict(observation, epoch, request_id)
            self.validate(result, observation, epoch, request_id)
            future = self.frames(result["future"])
            if past.shape[1:] != future.shape[1:]:
                raise ValueError("Observed and generated camera geometry differs")
            loss = float(
                self.judge.score(past, future, request_id, self.config["primary_id"])
            )
            if not math.isfinite(loss) or loss < 0:
                raise ValueError("Invalid JEPA raw loss")
            base["loss"] = loss
            switched = loss > self.threshold
            if switched:
                self.emit(**base, state="switching")
                result = self.fallback.predict(observation, epoch, request_id)
                self.validate(result, observation, epoch, request_id)
                future = self.frames(result["future"])
                if future.shape[1:] != past.shape[1:]:
                    raise ValueError("Fallback camera geometry differs")
            selected = (
                self.config["fallback_id"] if switched else self.config["primary_id"]
            )
            base["selected_model"] = selected
            self.emit(
                **base,
                state="selected",
                reason="threshold_exceeded" if switched else "threshold_passed",
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

    def cancel(self):
        for client in (self.primary, self.fallback):
            cancel = getattr(client, "cancel", None)
            if cancel:
                cancel()


def from_config(config, event=None):
    if config.get('selection_mode') == 'parallel_latched':
        from .sg2_dual import from_config as dual_from_config
        return dual_from_config(config,event)
    return JepaGate(
        JsonPolicy(
            config["primary_url"], config["timeout_seconds"], config["primary_id"]
        ),
        JsonPolicy(
            config["fallback_url"], config["timeout_seconds"], config["fallback_id"]
        ),
        JepaClient(config["jepa_url"], config["timeout_seconds"]),
        config,
        event,
    )


def launch_models(config):
    """Explicit CLI opt-in only; checkpoint transfer is not checked at import/config load."""
    import subprocess

    processes = []
    try:
        for model in config["models"]:
            checkpoint = Path(model["checkpoint"]).expanduser().resolve()
            if not checkpoint.exists():
                raise FileNotFoundError(checkpoint)
            argv = [
                str(arg).replace("{checkpoint}", str(checkpoint))
                for arg in model["command"]
            ]
            if not argv:
                raise ValueError("Explicit model-service argv required")
            processes.append(subprocess.Popen(argv))
        return processes
    except Exception:
        for process in processes:
            process.terminate()
        raise
