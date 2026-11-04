"""Same-user Unix socket requests; observations use pickle-free NPZ files in shared storage."""
from pathlib import Path
import json
import os
import re
import socket
import uuid

import numpy as np

from .core import require

MAX_LINE = 128 * 1024


def read_json(connection):
    with connection.makefile("rb") as stream:
        data = stream.readline(MAX_LINE + 1)
    require(len(data) <= MAX_LINE and data.endswith(b"\n"), "Invalid/oversized worker response")
    return json.loads(data)


def write_json(connection, data):
    payload = json.dumps(data, allow_nan=False).encode() + b"\n"
    require(len(payload) <= MAX_LINE, "Worker message exceeds limit")
    connection.sendall(payload)


def request(socket_path, payload, timeout=60):
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
        connection.settimeout(timeout)
        connection.connect(str(socket_path))
        write_json(connection, payload)
        response = read_json(connection)
    require(response.get("ok") is True, response.get("error", "Worker failed"))
    return response


def predict(socket_path, observation, epoch, *, seed=0, timeout=60):
    socket_path = Path(socket_path)
    identifier = uuid.uuid4().hex
    sample = socket_path.parent / f"{identifier}.npz"
    descriptor = os.open(sample, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            np.savez(handle, **observation)
        response = request(socket_path, {"op": "predict", "request_id": identifier, "epoch": epoch, "seed": seed}, timeout)
        require(response.get("request_id") == identifier and response.get("epoch") == epoch, "Worker response identity mismatch")
        require(response.get("source_time_ns") == int(observation["source_time_ns"]), "Worker response timestamp mismatch")
        return response
    finally:
        sample.unlink(missing_ok=True)


def observation_path(directory, identifier):
    require(isinstance(identifier, str) and re.fullmatch(r"[0-9a-f]{32}", identifier), "Invalid worker request ID")
    directory = Path(directory).resolve()
    path = directory / f"{identifier}.npz"
    require(not path.is_symlink() and path.resolve().parent == directory, "Unsafe observation path")
    require(path.is_file() and path.stat().st_size <= 8 * 1024 * 1024, "Observation file missing or too large")
    return path
