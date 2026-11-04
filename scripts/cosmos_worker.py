#!/usr/bin/env python3
"""Persistent native Cosmos worker; requests arrive only through a same-user Unix socket."""
import argparse
from pathlib import Path
import json
import os
import signal
import socket
import stat
import struct

from bg2_inference import JOINT_NAMES, load_observation
from bg2_bridge.ipc import observation_path, read_json, write_json


def serve(policy, socket_path, sampling, warmup=None):
    socket_path = Path(socket_path)
    socket_path.parent.mkdir(parents=True, exist_ok=True)
    if socket_path.exists() or socket_path.is_symlink():
        raise ValueError("Worker socket path already exists; do not overwrite another worker")
    if warmup is not None:
        policy.predict(load_observation(warmup), **sampling)
    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        listener.bind(str(socket_path))
        os.chmod(socket_path, 0o600)
        listener.listen(2)
        print(json.dumps({"event": "ready", "socket": str(socket_path), "loading_seconds": policy.loading_seconds}), flush=True)
        while True:
            connection, _ = listener.accept()
            with connection:
                connection.settimeout(10)
                try:
                    _, uid, _ = struct.unpack("3i", connection.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, 12))
                    if uid != os.getuid():
                        raise ValueError("Worker only accepts the same UID")
                    data = read_json(connection)
                    if data.get("op") == "health":
                        result = {"ok": True, "loading_seconds": policy.loading_seconds, "joint_names": JOINT_NAMES}
                    elif data.get("op") == "predict":
                        path = observation_path(socket_path.parent, data.get("request_id"))
                        observation = load_observation(path)
                        epoch = data.get("epoch")
                        if not isinstance(epoch, str) or len(epoch) != 32:
                            raise ValueError("Invalid bridge session epoch")
                        result = policy.predict(observation, **{**sampling, "seed": int(data.get("seed", sampling["seed"]))})
                        result = {"ok": True, "request_id": data["request_id"], "epoch": epoch,
                                  "source_time_ns": int(observation["source_time_ns"]), "joint_names": JOINT_NAMES,
                                  "units": ["rad"] * 8, "positions": result["positions"].tolist(), "timings": result["timings"]}
                    else:
                        raise ValueError("Unknown worker operation")
                    write_json(connection, result)
                except Exception as error:
                    try:
                        write_json(connection, {"ok": False, "error": str(error)})
                    except (OSError, ValueError):
                        pass
    finally:
        listener.close()
        socket_path.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint", "tokenizer", "vae"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--adapter-dir", type=Path, default=Path("/adapter"))
    parser.add_argument("--socket", type=Path, default=Path("/bridge/worker.sock"))
    parser.add_argument("--warmup-observation", type=Path)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--guidance", type=float, default=3.)
    parser.add_argument("--shift", type=float, default=5.)
    parser.add_argument("--guidance-interval", nargs=2, type=float)
    args = parser.parse_args()
    from policy_runtime import NativePolicy
    policy = NativePolicy(args.checkpoint, args.tokenizer, args.vae, args.adapter_dir)
    sampling = {key: getattr(args, key) for key in ("seed", "steps", "guidance", "shift", "guidance_interval")}
    def shutdown(*_):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, shutdown)
    try:
        serve(policy, args.socket, sampling, args.warmup_observation)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
