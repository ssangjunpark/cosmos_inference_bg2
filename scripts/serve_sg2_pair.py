#!/usr/bin/env python3
"""Start two resident SG2 policy processes; never starts a robot gateway."""

import argparse
import json
import signal
import subprocess
import sys
import time
from pathlib import Path
from urllib.error import URLError

from bg2_bridge.sg2_dual import SG2PolicyClient

ROOT = Path(__file__).resolve().parents[1]


def model_commands(config):
    runtime = config["model_runtime"]
    sampling = runtime["sampling"]
    for field in ("steps", "guidance", "shift"):
        if sampling.get(field) is None:
            raise ValueError(
                f"Set model_runtime.sampling.{field} from the training evaluation settings"
            )
    commands = []
    for role, port in (("primary", 8090), ("fallback", 8091)):
        entry = next(
            item
            for item in config["models"]
            if item["model_id"] == config[role + "_id"]
        )
        command = [
            sys.executable,
            str(ROOT / "scripts/sg2_policy_service.py"),
            "--variant",
            config[role + "_variant"],
            "--model-id",
            entry["model_id"],
            "--listen-port",
            str(entry.get("port", port)),
            "--video-view",
            config["video_view"],
            "--video-size",
            *[str(v) for v in config["video_size"]],
        ]
        for key in ("checkpoint", "tokenizer", "vae", "calibration"):
            value = entry[key] if key == "checkpoint" else runtime[key]
            command += ["--" + key, str(Path(value).expanduser())]
        for key in ("steps", "guidance", "shift", "seed"):
            command += ["--" + key, str(sampling.get(key, 0))]
        if sampling.get("guidance_interval") is not None:
            command += [
                "--guidance-interval",
                *[str(v) for v in sampling["guidance_interval"]],
            ]
        commands.append(command)
    return commands


def run_pair(config, dry_run=False, startup_timeout=600):
    commands = model_commands(config)
    if dry_run:
        print(json.dumps(commands, indent=2))
        return
    processes = []
    stopped = False

    def stop(*unused):
        nonlocal stopped
        stopped = True

    previous = {
        sig: signal.signal(sig, stop) for sig in (signal.SIGINT, signal.SIGTERM)
    }
    try:
        # Load serially to avoid concurrent checkpoint-loader host-memory peaks.
        for role, command in zip(("primary", "fallback"), commands):
            process = subprocess.Popen(command)
            processes.append(process)
            client = SG2PolicyClient(config[role + "_url"], 2, config[role + "_id"])
            deadline = time.monotonic() + startup_timeout
            while not stopped:
                if process.poll() is not None:
                    raise RuntimeError(f"{role} service exited during startup")
                try:
                    health = client.health()
                    if health.get("model_id") != config[role + "_id"]:
                        raise ValueError("Port already belongs to another model")
                    if health.get("ready"):
                        break
                except (OSError, URLError):
                    pass
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"{role} service startup timed out")
                time.sleep(0.1)
            if stopped:
                return
        print(
            json.dumps(
                {
                    "event": "sg2_pair_ready",
                    "models": [config["primary_id"], config["fallback_id"]],
                }
            ),
            flush=True,
        )
        reported = set()
        while not stopped:
            alive = False
            for role, process in zip(("primary", "fallback"), processes):
                code = process.poll()
                if code is None:
                    alive = True
                elif role not in reported:
                    reported.add(role)
                    print(
                        json.dumps(
                            {
                                "event": "sg2_service_exited",
                                "role": role,
                                "code": code,
                                "other_service_preserved": True,
                            }
                        ),
                        flush=True,
                    )
            if not alive:
                raise RuntimeError("Both SG2 model services exited")
            time.sleep(0.2)
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
        for process in processes:
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=lambda p: Path(p).expanduser(), required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--startup-timeout", type=float, default=600)
    args = parser.parse_args()
    run_pair(json.loads(args.config.read_text()), args.dry_run, args.startup_timeout)


if __name__ == "__main__":
    main()
