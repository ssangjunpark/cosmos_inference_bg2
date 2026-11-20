#!/usr/bin/env python3
"""Launch SG2 Cosmos/JEPA services and gateway from one host-path JSON config."""

import argparse
import copy
import json
import math
import os
import signal
import socket
import subprocess
import time
import uuid
from pathlib import Path
from urllib.error import URLError
from urllib.parse import urlsplit
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]


def load_config(path):
    path = Path(path).expanduser().resolve()
    cfg = copy.deepcopy(json.loads(path.read_text()))

    def absolute(value):
        value = Path(os.path.expandvars(value)).expanduser()
        return (
            str((path.parent / value).resolve())
            if not value.is_absolute()
            else str(value.resolve())
        )

    for entry in cfg["models"]:
        entry["checkpoint"] = absolute(entry["checkpoint"])
    for key in ("tokenizer", "vae", "calibration"):
        cfg["model_runtime"][key] = absolute(cfg["model_runtime"][key])
    for key in ("checkpoint", "repo_root", "project_root"):
        cfg["jepa"][key] = absolute(cfg["jepa"][key])
    if cfg["jepa"].get("teacher_checkpoint"):
        cfg["jepa"]["teacher_checkpoint"] = absolute(cfg["jepa"]["teacher_checkpoint"])
    for section, key in (("jepa", "python"), ("launcher", "cosmos_python")):
        value = cfg[section].get(key, "python3")
        if "/" in value or value.startswith("~"):
            executable = Path(os.path.expandvars(value)).expanduser()
            if not executable.is_absolute():
                executable = path.parent / executable
            # Preserve venv/bin/python symlinks: resolving them selects system Python.
            cfg[section][key] = os.path.abspath(executable)
        else:
            cfg[section][key] = value
    for key in ("selection_output", "robot_profile"):
        cfg["gateway"][key] = absolute(cfg["gateway"][key])
    threshold = float(cfg["threshold"])
    if not math.isfinite(threshold) or threshold < 0:
        raise ValueError("threshold must be finite and nonnegative")
    if cfg["selection_mode"] != "parallel_latched":
        raise ValueError("This launcher requires parallel_latched mode")
    ports = []
    for role in ("primary", "fallback", "jepa"):
        url = urlsplit(cfg[role + "_url"])
        if (
            url.scheme != "http"
            or url.hostname not in ("localhost", "127.0.0.1", "::1")
            or url.username
            or url.password
            or url.path not in ("", "/")
            or url.query
            or url.fragment
        ):
            raise ValueError("Use localhost root HTTP endpoints")
        ports.append(url.port or 80)
    if len(set(ports)) != 3:
        raise ValueError("CM1, CM2 and JEPA must use distinct ports")
    if ports[2] != cfg["jepa"]["port"]:
        raise ValueError("jepa.port must match jepa_url")
    if cfg["primary_id"] == cfg["fallback_id"]:
        raise ValueError("CM1 and CM2 IDs must differ")
    for role, port in zip(("primary", "fallback"), ports):
        matches = [
            entry for entry in cfg["models"] if entry["model_id"] == cfg[role + "_id"]
        ]
        if len(matches) != 1 or matches[0]["port"] != port:
            raise ValueError(f"{role} model ID/port must match its URL")
    if cfg["launcher"]["cosmos_backend"] not in ("docker", "native"):
        raise ValueError("cosmos_backend must be docker or native")
    timeout = float(cfg["launcher"]["startup_timeout_seconds"])
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("Set a positive startup_timeout_seconds")
    for key in ("steps", "guidance", "shift"):
        if cfg["model_runtime"]["sampling"].get(key) is None:
            raise ValueError(f"Set model_runtime.sampling.{key} before starting")
    return cfg


def build_plan(cfg, action, run_dir):
    run_dir = Path(run_dir)
    gate_file, jepa_file = run_dir / "gate.json", run_dir / "jepa.json"
    project = Path(cfg["jepa"]["project_root"])
    jepa_env = dict(os.environ)
    jepa_env["PYTHONPATH"] = str(project / "src") + (
        os.pathsep + os.environ["PYTHONPATH"] if os.environ.get("PYTHONPATH") else ""
    )
    jepa = {
        "role": "jepa",
        "argv": [
            cfg["jepa"]["python"],
            "-m",
            "lgwm.judges.future_service",
            "--config",
            str(jepa_file),
        ],
        "cwd": str(project),
        "env": jepa_env,
    }

    def cosmos(role, script, argv):
        python = cfg["launcher"].get("cosmos_python", "python3")
        command = [python, str(ROOT / "scripts" / script), *argv]
        if cfg["launcher"]["cosmos_backend"] == "docker":
            # Host absolute paths are mounted at identical paths: no /data config rewrite.
            mounts = {ROOT: True, run_dir: True}
            for entry in cfg["models"]:
                mounts[Path(entry["checkpoint"]).parent] = True
            for key in ("tokenizer", "vae", "calibration"):
                mounts[Path(cfg["model_runtime"][key]).parent] = True
            mounts[Path(cfg["gateway"]["robot_profile"]).parent] = True
            command = [
                "docker",
                "run",
                "--rm",
                "--init",
                "--pull=never",
                "--network",
                "host",
                "--user",
                f"{os.getuid()}:{os.getgid()}",
                "-e",
                "HF_HUB_OFFLINE=1",
                "-e",
                "PYTHONDONTWRITEBYTECODE=1",
                "-e",
                f"PYTHONPATH=/workspace:{ROOT}/scripts",
                "--workdir",
                str(ROOT),
            ]
            if role == "cosmos":
                command += [
                    "--gpus",
                    f"device={cfg['launcher']['gpu']}",
                    "--shm-size=16g",
                ]
            if role == "gateway":
                mounts[Path(cfg["gateway"]["selection_output"])] = False
            for mount, readonly in mounts.items():
                command += ["-v", f"{mount}:{mount}" + (":ro" if readonly else "")]
            command += [
                "--entrypoint",
                "/workspace/.venv/bin/python",
                cfg["launcher"]["cosmos_image"],
                str(ROOT / "scripts" / script),
                *argv,
            ]
        return {
            "role": role,
            "argv": command,
            "cwd": str(ROOT),
            "env": dict(os.environ),
        }

    pair = cosmos(
        "cosmos",
        "serve_sg2_pair.py",
        [
            "--config",
            str(gate_file),
            "--startup-timeout",
            str(cfg["launcher"]["startup_timeout_seconds"]),
        ],
    )
    jobs = [jepa, pair] if action in ("services", "all") else []
    if action in ("gateway", "all"):
        gate = cfg["gateway"]
        if gate["mode"] not in ("preview", "execute"):
            raise ValueError("gateway.mode must be preview or execute")
        if (
            not gate.get("router")
            or gate.get("domain") is None
            or not all(
                gate.get(key) for key in ("head_topic", "left_topic", "right_topic")
            )
        ):
            raise ValueError("Fill gateway router, domain and all three camera topics")
        argv = [
            "jepa-" + gate["mode"],
            "--variant",
            cfg["primary_variant"],
            "--jepa-config",
            str(gate_file),
            "--selection-output",
            gate["selection_output"],
            "--router",
            gate["router"],
            "--domain",
            str(gate["domain"]),
            "--port",
            str(gate["port"]),
            "--robot-profile",
            gate["robot_profile"],
            "--chunks",
            str(gate["chunks"]),
            "--timeout",
            str(gate["timeout_seconds"]),
        ]
        for key in ("head_topic", "left_topic", "right_topic"):
            argv += ["--" + key.replace("_", "-"), gate[key]]
        jobs.append(cosmos("gateway", "sg2_policy.py", argv))
    return jobs


def ready(cfg):
    for role in ("primary", "fallback", "jepa"):
        try:
            with urlopen(
                cfg[role + "_url"].rstrip("/") + "/health", timeout=1
            ) as response:
                data = json.loads(response.read(128 * 1024))
            if data.get("ready") is not True:
                return False
            expected = (
                cfg["jepa"]["checkpoint"]
                if role == "jepa"
                else next(
                    entry["checkpoint"]
                    for entry in cfg["models"]
                    if entry["model_id"] == cfg[role + "_id"]
                )
            )
            if data.get("checkpoint") != expected:
                raise ValueError(
                    f"{role} service has a different checkpoint; restart services with this config"
                )
            if role != "jepa" and data.get("model_id") != cfg[role + "_id"]:
                raise ValueError(f"Port belongs to a different {role} model")
        except (OSError, URLError):
            return False
    return True


def ensure_free_ports(cfg):
    for role in ("primary", "fallback", "jepa"):
        port = urlsplit(cfg[role + "_url"]).port or 80
        with socket.socket() as probe:
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                probe.bind(("127.0.0.1", port))
            except OSError as error:
                raise RuntimeError(
                    f"{role} port {port} is occupied; stop existing services or use the gateway action"
                ) from error


def supervise(cfg, jobs, action):
    owned = []
    stopped = False

    def stop(*unused):
        nonlocal stopped
        stopped = True

    previous = {
        sig: signal.signal(sig, stop) for sig in (signal.SIGINT, signal.SIGTERM)
    }
    try:
        deadline = time.monotonic() + cfg["launcher"]["startup_timeout_seconds"]
        for job in jobs:
            if job["role"] == "gateway":
                while not stopped and not ready(cfg):
                    if time.monotonic() >= deadline:
                        raise TimeoutError(
                            "Services not ready; gateway was not started"
                        )
                    if any(process.poll() is not None for _, process in owned):
                        raise RuntimeError("A service exited before gateway startup")
                    time.sleep(0.2)
                if stopped:
                    return
            owned.append(
                (
                    job,
                    subprocess.Popen(
                        job["argv"],
                        cwd=job["cwd"],
                        env=job["env"],
                        start_new_session=True,
                    ),
                )
            )
        if action == "services":
            while not stopped and not ready(cfg):
                if time.monotonic() >= deadline:
                    raise TimeoutError("Service startup timed out")
                if any(process.poll() is not None for _, process in owned):
                    raise RuntimeError("Service exited during startup")
                time.sleep(0.2)
            if stopped:
                return
        print(
            json.dumps(
                {
                    "event": "sg2_jepa_started",
                    "action": action,
                    "threshold": cfg["threshold"],
                }
            ),
            flush=True,
        )
        reported = set()
        while not stopped:
            alive = False
            for job, process in owned:
                code = process.poll()
                if code is None:
                    alive = True
                elif job["role"] == "gateway":
                    if code:
                        raise RuntimeError(f"Gateway exited with code {code}")
                    return
                elif job["role"] not in reported:
                    reported.add(job["role"])
                    print(
                        json.dumps(
                            {
                                "event": "service_exited",
                                "role": job["role"],
                                "code": code,
                                "other_services_preserved": True,
                            }
                        ),
                        flush=True,
                    )
            if not alive:
                raise RuntimeError("All launched services exited")
            time.sleep(0.2)
    finally:
        for _, process in owned:
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
        for _, process in owned:
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait()
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "action", choices=("services", "gateway", "all"), default="services", nargs="?"
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    run_dir = ROOT / ".runtime/sg2-jepa" / uuid.uuid4().hex
    jobs = build_plan(cfg, args.action, run_dir)
    if args.dry_run:
        print(
            json.dumps(
                {
                    "threshold": cfg["threshold"],
                    "models": cfg["models"],
                    "jepa_checkpoint": cfg["jepa"]["checkpoint"],
                    "jobs": [
                        {key: value for key, value in job.items() if key != "env"}
                        for job in jobs
                    ],
                },
                indent=2,
            )
        )
        return
    if args.action in ("services", "all"):
        ensure_free_ports(cfg)
    run_dir.mkdir(parents=True)
    if args.action in ("gateway", "all"):
        Path(cfg["gateway"]["selection_output"]).mkdir(parents=True, exist_ok=True)
    (run_dir / "gate.json").write_text(json.dumps(cfg, indent=2))
    jepa_cfg = {
        key: value
        for key, value in cfg["jepa"].items()
        if key not in ("python", "project_root")
    }
    (run_dir / "jepa.json").write_text(json.dumps(jepa_cfg, indent=2))
    print(
        json.dumps({"event": "configuration_saved", "path": str(run_dir)}), flush=True
    )
    supervise(cfg, jobs, args.action)


if __name__ == "__main__":
    main()
