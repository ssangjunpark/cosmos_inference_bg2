#!/usr/bin/env python3
"""Build NVIDIA's environment and run BG2 policy or offline WAM inference."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
from datetime import datetime, timezone
import uuid

ROOT = Path(__file__).resolve().parents[1]
REVISION = "b98934eda43f4d31972538a527f8663f003c27dc"
UPSTREAM = ROOT / "third_party/cosmos-framework"
BASE_IMAGE = "cosmos-framework:robotescape-b98934e"
POLICY_IMAGE = "cosmos3-policy:robotescape-b98934e"
CHECKPOINT_ROOT = Path("/mnt/ssd2/RobotEscapeCkpt")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def published_checkpoint(root: Path, variant: str, step: int, hashes: bool = False) -> Path:
    root = root.resolve(strict=True)
    checkpoint = (root / f"cosmos3_{variant}" / f"{step:06d}").resolve(strict=True)
    checkpoint.relative_to(root)
    for name in ("READY.json", "COPY_VERIFIED.json", "ACCESS_VERIFIED.json", "input_contract.json", "config.yaml"):
        if not (checkpoint / name).is_file():
            raise ValueError(f"Checkpoint is not published: missing {checkpoint / name}")
    ready = json.loads((checkpoint / "READY.json").read_text())
    if ready.get("step") != step or ready.get("variant") != variant:
        raise ValueError("READY.json does not match the requested variant/step")
    contract = json.loads((checkpoint / "input_contract.json").read_text())
    if contract.get("framework_commit") != REVISION:
        raise ValueError(f"Checkpoint needs framework {contract.get('framework_commit')}, checkout is pinned to {REVISION}")
    manifest = json.loads((checkpoint / "COPY_VERIFIED.json").read_text())
    if manifest.get("format") != "pytorch_distributed_checkpoint":
        raise ValueError("This runner expects the published native DCP checkpoint")
    listed = set()
    for entry in manifest["files"]:
        relative = Path(entry["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Unsafe path in checkpoint manifest")
        path = (checkpoint / relative).resolve(strict=True)
        path.relative_to(checkpoint)
        listed.add(relative.as_posix())
        if path.stat().st_size != entry["bytes"]:
            raise ValueError(f"Checkpoint is incomplete or changed: {path}")
        if hashes and sha256(path) != entry["sha256"]:
            raise ValueError(f"SHA-256 mismatch: {path}")
    actual = {"model/.metadata"} | {
        p.relative_to(checkpoint).as_posix() for p in (checkpoint / "model").glob("*.distcp")
    }
    if not (checkpoint / "model/.metadata").is_file() or not (actual - {"model/.metadata"}):
        raise ValueError("Missing DCP metadata or shards")
    if actual != listed:
        raise ValueError("DCP shard set differs from COPY_VERIFIED.json")
    return checkpoint


def checked(command: list[str], dry_run: bool = False) -> None:
    print(shlex.join(command), flush=True)
    if not dry_run:
        subprocess.run(command, cwd=ROOT, check=True)


def build(dry_run: bool) -> None:
    if not (UPSTREAM / ".git").exists():
        raise ValueError("Clone NVIDIA/cosmos-framework into third_party/cosmos-framework first")
    revision = subprocess.check_output(["git", "-C", str(UPSTREAM), "rev-parse", "HEAD"], text=True).strip()
    if revision != REVISION:
        raise ValueError(f"Framework revision mismatch: {revision}")
    dirty = subprocess.check_output(
        ["git", "-C", str(UPSTREAM), "status", "--porcelain"], text=True,
    ).strip()
    if dirty:
        raise ValueError("Framework checkout must be clean before claiming verified source provenance")
    tree = subprocess.check_output(
        ["git", "-C", str(UPSTREAM), "rev-parse", "HEAD^{tree}"], text=True,
    ).strip()
    timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    checked(["docker", "build", "--build-arg", "INSTALL_APEX=0",
             "--build-arg", f"SOURCE_COMMIT={revision}",
             "--build-arg", f"SOURCE_TREE={tree}",
             "--build-arg", f"BUILD_TIMESTAMP={timestamp}",
             "--build-arg", "SOURCE_DIRTY=0",
             "--build-arg", "REQUIRE_SOURCE_PROVENANCE=1",
             "-t", BASE_IMAGE, str(UPSTREAM)], dry_run)
    checked(["docker", "build", "--build-arg", f"FRAMEWORK_IMAGE={BASE_IMAGE}",
             "-f", str(ROOT / "docker/Dockerfile.policy"), "-t", POLICY_IMAGE,
             str(ROOT / "docker")], dry_run)


def bind(source: Path, target: str, readonly: bool = True) -> list[str]:
    source = source.resolve()
    if "," in str(source):
        raise ValueError("Docker bind source paths containing commas are unsupported")
    option = f"type=bind,source={source},target={target}"
    if readonly:
        option += ",readonly"
    return ["--mount", option]


def container(gpu: str, output: Path, dry_run: bool) -> list[str]:
    if not gpu.isdecimal():
        raise ValueError("--gpu must select one physical GPU index")
    cache = ROOT / ".runtime/cache"
    if not dry_run:
        for path in (cache, cache / "home", output):
            path.mkdir(parents=True, exist_ok=True)
    return [
        "docker", "run", "--rm", "--init", "--runtime=nvidia", f"--gpus=device={gpu}",
        "--network=none", "--read-only", "--cap-drop=ALL",
        "--security-opt=no-new-privileges:true", "--shm-size=16g",
        "--tmpfs", "/tmp:rw,nosuid,nodev,size=8g",
        "--user", f"{os.getuid()}:{os.getgid()}", "--workdir", "/workspace",
        "--env", "USER=cosmos", "--env", "PYTHONDONTWRITEBYTECODE=1", "--env", "HF_HUB_OFFLINE=1",
        "--env", "TRANSFORMERS_OFFLINE=1", "--env", "HF_HOME=/cache/huggingface",
        "--env", "XDG_CACHE_HOME=/cache/xdg", "--env", "TRITON_CACHE_DIR=/cache/triton",
        "--env", "TORCHINDUCTOR_CACHE_DIR=/cache/torchinductor",
        "--env", "CUDA_CACHE_PATH=/cache/cuda", "--env", "MPLCONFIGDIR=/cache/matplotlib",
        "--env", "PYTHONPATH=/adapter:/app:/workspace",
        *bind(ROOT / "scripts", "/app"), *bind(cache, "/cache", False),
        *bind(cache / "home", "/root", False), *bind(output, "/results", False),
        "--entrypoint", "/workspace/.venv/bin/python",
    ]



def tokenizer_bind(tokenizer: Path) -> tuple[list[str], str]:
    # Hugging Face snapshot files often link to ../../blobs. Preserve those
    # read-only links by mounting just that model cache repository.
    source = tokenizer
    if tokenizer.parent.name == "snapshots" and (tokenizer.parent.parent / "blobs").is_dir():
        source = tokenizer.parent.parent
    for path in tokenizer.rglob("*"):
        if path.is_symlink():
            try:
                path.resolve(strict=True).relative_to(source)
            except (ValueError, OSError) as error:
                raise ValueError(f"Tokenizer symlink escapes its read-only mount: {path}") from error
    target = "/assets/tokenizer"
    relative = tokenizer.relative_to(source)
    inside = target if relative == Path(".") else f"{target}/{relative.as_posix()}"
    return bind(source, target), inside


def run(args: argparse.Namespace) -> None:
    checkpoint = published_checkpoint(args.checkpoint_root, args.variant, args.step)
    observation = args.observation.resolve(strict=True)
    tokenizer = args.tokenizer.resolve(strict=True)
    vae = args.vae.resolve(strict=True)
    if not observation.is_file() or observation.suffix != ".npz":
        raise ValueError("--observation must be a recorded observation .npz archive")
    if not tokenizer.is_dir() or not (tokenizer / "tokenizer_config.json").is_file() or not vae.is_file():
        raise ValueError("Provide a complete local tokenizer directory and the local Wan VAE file")
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    output = ROOT / "outputs" / f"{args.variant}-{args.step:06d}-{run_id}"
    command = container(args.gpu, output, args.dry_run)
    command += bind(checkpoint, "/checkpoint")
    command += bind(observation, "/input/observation.npz")
    tokenizer_mount, tokenizer_inside = tokenizer_bind(tokenizer)
    command += tokenizer_mount
    command += bind(vae, "/assets/Wan2.2_VAE.pth")
    if args.adapter_dir is not None:
        adapter = args.adapter_dir.resolve(strict=True)
        if not adapter.is_dir():
            raise ValueError("--adapter-dir must be the directory containing bg2_cosmos")
        command += bind(adapter, "/adapter")
    command += [
        BASE_IMAGE if args.image == "base" else POLICY_IMAGE, "/app/native_infer.py",
        "--checkpoint", "/checkpoint", "--observation", "/input/observation.npz",
        "--tokenizer", tokenizer_inside, "--vae", "/assets/Wan2.2_VAE.pth",
        "--output", "/results", "--seed", str(args.seed),
        "--steps", str(args.steps), "--guidance", str(args.guidance), "--shift", str(args.shift),
    ]
    if args.guidance_interval:
        command += ["--guidance-interval", *map(str, args.guidance_interval)]
    checked(command, args.dry_run)
    print(f"Results: {output}")


def serve(args: argparse.Namespace) -> None:
    checkpoint = published_checkpoint(args.checkpoint_root, args.variant, args.step)
    tokenizer, vae = args.tokenizer.resolve(strict=True), args.vae.resolve(strict=True)
    observation = args.warmup_observation.resolve(strict=True)
    if not tokenizer.is_dir() or not (tokenizer / "tokenizer_config.json").is_file() or not vae.is_file():
        raise ValueError("Provide the complete local tokenizer and Wan VAE")
    if not observation.is_file() or observation.suffix != ".npz" or not 1 <= args.port <= 65535 or args.warmup_runs < 1:
        raise ValueError("Invalid warm-up observation, port, or warm-up count")
    output = ROOT / "outputs" / ("server-" + args.variant)
    command = container(args.gpu, output, args.dry_run)
    # Serving is the one runtime that accepts network requests. The host port
    # remains loopback-only; an SSH tunnel is required from the 5090.
    command[command.index("--network=none")] = "--network=bridge"
    command += ["--name", "cosmos-bg2-" + args.variant + "-server", "-p", f"127.0.0.1:{args.port}:8008"]
    command += bind(checkpoint, "/checkpoint") + bind(observation, "/input/warmup.npz")
    mounts, inside = tokenizer_bind(tokenizer)
    command += mounts + bind(vae, "/assets/Wan2.2_VAE.pth") + bind(args.adapter_dir.resolve(strict=True), "/adapter")
    command += [BASE_IMAGE if args.image == "base" else POLICY_IMAGE, "/app/inference_server.py",
                "--bind", "0.0.0.0", "--port", "8008", "--checkpoint", "/checkpoint", "--tokenizer", inside,
                "--vae", "/assets/Wan2.2_VAE.pth", "--warmup-observation", "/input/warmup.npz",
                "--warmup-runs", str(args.warmup_runs),
                "--seed", str(args.seed), "--steps", str(args.steps), "--guidance", str(args.guidance), "--shift", str(args.shift)]
    if args.guidance_interval:
        command += ["--guidance-interval", *map(str, args.guidance_interval)]
    checked(command, args.dry_run)


def doctor(args: argparse.Namespace) -> None:
    command = container(args.gpu, ROOT / "outputs/doctor", args.dry_run)
    image = BASE_IMAGE if args.image == "base" else POLICY_IMAGE
    checked(command + [image, "/app/doctor_runtime.py", "--expected-commit", REVISION], args.dry_run)


def idle_gpu(gpu: str) -> dict:
    """Reject occupied GPUs; never stop another workload to make room."""
    if not gpu.isdecimal():
        raise ValueError("--gpu must select one physical GPU index")
    info = subprocess.check_output(["nvidia-smi", f"--id={gpu}",
        "--query-gpu=uuid,memory.used,utilization.gpu", "--format=csv,noheader,nounits"], text=True).strip().split(",")
    uuid_value, memory, utilization = [v.strip() for v in info]
    running = subprocess.check_output(["nvidia-smi", "--query-compute-apps=gpu_uuid", "--format=csv,noheader"], text=True).splitlines()
    if uuid_value in [v.strip() for v in running] or int(memory) > 512 or int(utilization) != 0:
        raise ValueError(f"GPU {gpu} is occupied; select an idle GPU. No existing process was stopped.")
    return {"physical_gpu": gpu, "uuid": uuid_value, "used_memory_mib": int(memory), "utilization_percent": int(utilization)}


def wam(args: argparse.Namespace) -> None:
    from wam_checkpoint import published_wam_checkpoint, validate_experiment, validate_checkpoint, file_hash
    checkpoint = published_wam_checkpoint(args.checkpoint_root, args.variant, args.step, args.sha256)
    experiment = (args.experiment_contract or args.checkpoint_root / "connection_test/experiment_contract.json").resolve(strict=True)
    publication = validate_checkpoint(checkpoint, args.variant, args.step)
    validate_experiment(experiment, publication["contract"])
    observation, tokenizer, vae = (p.resolve(strict=True) for p in (args.observation, args.tokenizer, args.vae))
    if not observation.is_file() or observation.suffix != ".npz":
        raise ValueError("--observation must be a recorded current-observation NPZ")
    if not (tokenizer / "tokenizer_config.json").is_file() or not vae.is_file():
        raise ValueError("Provide the matching local tokenizer and Wan VAE")
    if args.step < 1 or args.steps < 1:
        raise ValueError("--step and --steps must be positive")
    if args.check:
        print(json.dumps({"valid": True, "mode": "wam", "checkpoint": str(checkpoint),
                          "future_steps": 16, "video_frames": 17, "gpu_started": False,
                          "checkpoint_writes": False, "full_checksums_checked": args.sha256}))
        return
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    output = (args.output or ROOT / "outputs" / f"wam-{args.variant}-{args.step:06d}-{stamp}").resolve()
    output.relative_to((ROOT / "outputs").resolve())
    if output.exists():
        raise ValueError("Use a new WAM output directory; existing results are preserved")
    command = container(args.gpu, output, args.dry_run)
    command.insert(2, "--pull=never")
    command += bind(checkpoint, "/checkpoint") + bind(observation, "/input/observation.npz")
    command += bind(experiment, "/input/experiment_contract.json")
    mounts, inside = tokenizer_bind(tokenizer)
    command += mounts + bind(vae, "/assets/Wan2.2_VAE.pth") + bind(args.adapter_dir.resolve(strict=True), "/adapter")
    interval = args.guidance_interval
    if interval is None and args.variant == "edge" and not args.guidance_all_steps:
        interval = [960., 1001.]
    command += [BASE_IMAGE if args.image == "base" else POLICY_IMAGE, "/app/wam_infer.py",
                "--checkpoint", "/checkpoint", "--step", str(args.step),
                "--observation", "/input/observation.npz", "--experiment-contract", "/input/experiment_contract.json",
                "--tokenizer", inside, "--vae", "/assets/Wan2.2_VAE.pth", "--output", "/results",
                "--seed", str(args.seed), "--steps", str(args.steps), "--guidance", str(args.guidance), "--shift", str(args.shift)]
    if interval is not None:
        command += ["--guidance-interval", *map(str, interval)]
    if args.instruction is not None:
        command += ["--instruction", args.instruction]
    if not args.dry_run:
        gpu = idle_gpu(args.gpu)  # Check immediately before launch, not at plan time.
        receipt = {"mode": "wam", "execution": "offline_synchronous", "gpu_preflight": gpu,
                   "checkpoint": str(checkpoint), "checkpoint_writes": False, "command": command,
                   "metadata_sha256": {n: file_hash(checkpoint / n) for n in
                       ["READY.json", "VERIFIED.json", "TRANSFER_VERIFIED.json", "config.yaml", "input_contract.json", "vision_contract.py"]}}
        (output / "launch.json").write_text(json.dumps(receipt, indent=2) + "\n")
    checked(command, args.dry_run)
    print(f"WAM actions and imagined rollout: {output}")



def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build_parser = commands.add_parser("build", help="Run NVIDIA's Docker build and policy dependency installation")
    build_parser.add_argument("--dry-run", action="store_true")
    status = commands.add_parser("status", help="Inspect published checkpoints without touching in-progress downloads")
    status.add_argument("--checkpoint-root", type=Path, default=CHECKPOINT_ROOT)
    status.add_argument("--sha256", action="store_true", help="Rehash all published shards (reads many GB)")
    health = commands.add_parser("doctor", help="Verify container CUDA and media dependencies")
    health.add_argument("--gpu", required=True)
    health.add_argument("--image", choices=("base", "policy"), default="policy")
    health.add_argument("--dry-run", action="store_true")
    infer = commands.add_parser("infer", help="Save model outputs for one recorded observation")
    infer.add_argument("--checkpoint-root", type=Path, default=CHECKPOINT_ROOT)
    infer.add_argument("--variant", choices=("edge", "nano"), required=True)
    infer.add_argument("--step", type=int, default=20000)
    infer.add_argument("--gpu", required=True)
    infer.add_argument("--image", choices=("base", "policy"), default="policy")
    infer.add_argument("--observation", type=Path, required=True)
    infer.add_argument("--tokenizer", type=Path, required=True)
    infer.add_argument("--vae", type=Path, required=True)
    infer.add_argument("--adapter-dir", type=Path, default=ROOT / "adapters")
    infer.add_argument("--seed", type=int, default=0)
    infer.add_argument("--steps", type=int, default=4)
    infer.add_argument("--guidance", type=float, default=3.0)
    infer.add_argument("--shift", type=float, default=5.0)
    infer.add_argument("--guidance-interval", nargs=2, type=float)
    infer.add_argument("--dry-run", action="store_true")
    server = commands.add_parser("serve", help="Persistent localhost inference API; connect through SSH")
    server.add_argument("--checkpoint-root", type=Path, default=CHECKPOINT_ROOT)
    server.add_argument("--variant", choices=("edge", "nano"), required=True)
    server.add_argument("--step", type=int, default=20000)
    server.add_argument("--gpu", required=True)
    server.add_argument("--image", choices=("base", "policy"), default="base")
    server.add_argument("--tokenizer", type=Path, required=True)
    server.add_argument("--vae", type=Path, required=True)
    server.add_argument("--warmup-observation", type=Path, required=True)
    server.add_argument("--warmup-runs", type=int, default=2)
    server.add_argument("--adapter-dir", type=Path, default=ROOT / "adapters")
    server.add_argument("--port", type=int, default=8008)
    server.add_argument("--seed", type=int, default=0)
    server.add_argument("--steps", type=int, default=4)
    server.add_argument("--guidance", type=float, default=3.)
    server.add_argument("--shift", type=float, default=5.)
    server.add_argument("--guidance-interval", nargs=2, type=float)
    server.add_argument("--dry-run", action="store_true")
    dream = commands.add_parser("wam", help="Offline synchronous WAM actions + imagined MP4; no hardware/network")
    from wam_checkpoint import DEFAULT_ROOT
    dream.add_argument("--checkpoint-root", type=Path, default=DEFAULT_ROOT)
    dream.add_argument("--experiment-contract", type=Path)
    dream.add_argument("--variant", choices=("edge", "nano"), required=True)
    dream.add_argument("--step", type=int, default=20000)
    dream.add_argument("--gpu", required=True)
    dream.add_argument("--image", choices=("base", "policy"), default="base")
    dream.add_argument("--observation", type=Path, required=True)
    dream.add_argument("--tokenizer", type=Path, required=True)
    dream.add_argument("--vae", type=Path, required=True)
    dream.add_argument("--adapter-dir", type=Path, default=ROOT / "adapters")
    dream.add_argument("--instruction")
    dream.add_argument("--output", type=Path)
    dream.add_argument("--seed", type=int, default=0)
    dream.add_argument("--steps", type=int, default=4)
    dream.add_argument("--guidance", type=float, default=3.)
    dream.add_argument("--shift", type=float, default=5.)
    interval_options = dream.add_mutually_exclusive_group()
    interval_options.add_argument("--guidance-interval", type=float, nargs=2)
    interval_options.add_argument("--guidance-all-steps", action="store_true")
    dream.add_argument("--sha256", action="store_true", help="Rehash checkpoint shards read-only")
    dream.add_argument("--check", action="store_true", help="Validate paths/receipts only; no GPU/model launch")
    dream.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    try:
        if args.command == "build":
            build(args.dry_run)
        elif args.command == "doctor":
            doctor(args)
        elif args.command == "serve":
            if args.step < 1 or args.steps < 1:
                raise ValueError("--step and --steps must be positive")
            serve(args)
        elif args.command == "infer":
            if args.step < 1 or args.steps < 1:
                raise ValueError("--step and --steps must be positive")
            run(args)
        elif args.command == "wam":
            wam(args)
        else:
            results = []
            for variant in ("edge", "nano"):
                parent = args.checkpoint_root / f"cosmos3_{variant}"
                if parent.is_dir():
                    for path in sorted(parent.iterdir()):
                        if path.name.isdecimal() and path.is_dir():
                            try:
                                published_checkpoint(args.checkpoint_root, variant, int(path.name), args.sha256)
                                results.append({"variant": variant, "step": int(path.name), "status": "published"})
                            except (ValueError, OSError, KeyError) as error:
                                results.append({"variant": variant, "step": int(path.name), "status": str(error)})
            print(json.dumps(results, indent=2))
        return 0
    except (ValueError, OSError, KeyError, subprocess.CalledProcessError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
