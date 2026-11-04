#!/usr/bin/env python3
"""Build NVIDIA's environment and run one recorded native policy sample."""
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
        "--env", "PYTHONDONTWRITEBYTECODE=1", "--env", "HF_HUB_OFFLINE=1",
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
    sample = args.sample.resolve(strict=True)
    tokenizer = args.tokenizer.resolve(strict=True)
    vae = args.vae.resolve(strict=True)
    if not sample.is_file() or sample.suffix != ".npz":
        raise ValueError("--sample must be a native sample .npz archive")
    if not tokenizer.is_dir() or not (tokenizer / "tokenizer_config.json").is_file() or not vae.is_file():
        raise ValueError("Provide a complete local tokenizer directory and the local Wan VAE file")
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    output = ROOT / "outputs" / f"{args.variant}-{args.step:06d}-{run_id}"
    command = container(args.gpu, output, args.dry_run)
    command += bind(checkpoint, "/checkpoint")
    command += bind(sample, "/input/sample.npz")
    tokenizer_mount, tokenizer_inside = tokenizer_bind(tokenizer)
    command += tokenizer_mount
    command += bind(vae, "/assets/Wan2.2_VAE.pth")
    if args.adapter_dir is not None:
        adapter = args.adapter_dir.resolve(strict=True)
        if not adapter.is_dir():
            raise ValueError("--adapter-dir must be the directory containing bg2_cosmos")
        command += bind(adapter, "/adapter")
    command += [
        POLICY_IMAGE, "/app/native_infer.py",
        "--checkpoint", "/checkpoint", "--sample", "/input/sample.npz",
        "--tokenizer", tokenizer_inside, "--vae", "/assets/Wan2.2_VAE.pth",
        "--output", "/results", "--seed", str(args.seed),
        "--steps", str(args.steps), "--guidance", str(args.guidance), "--shift", str(args.shift),
    ]
    if args.guidance_interval:
        command += ["--guidance-interval", *map(str, args.guidance_interval)]
    checked(command, args.dry_run)
    print(f"Results: {output}")


def doctor(args: argparse.Namespace) -> None:
    command = container(args.gpu, ROOT / "outputs/doctor", args.dry_run)
    checked(command + [POLICY_IMAGE, "/app/doctor_runtime.py", "--expected-commit", REVISION], args.dry_run)



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
    health.add_argument("--dry-run", action="store_true")
    infer = commands.add_parser("infer", help="Save model outputs for one recorded native sample")
    infer.add_argument("--checkpoint-root", type=Path, default=CHECKPOINT_ROOT)
    infer.add_argument("--variant", choices=("edge", "nano"), required=True)
    infer.add_argument("--step", type=int, default=20000)
    infer.add_argument("--gpu", required=True)
    infer.add_argument("--sample", type=Path, required=True)
    infer.add_argument("--tokenizer", type=Path, required=True)
    infer.add_argument("--vae", type=Path, required=True)
    infer.add_argument("--adapter-dir", type=Path)
    infer.add_argument("--seed", type=int, default=0)
    infer.add_argument("--steps", type=int, default=4)
    infer.add_argument("--guidance", type=float, default=3.0)
    infer.add_argument("--shift", type=float, default=5.0)
    infer.add_argument("--guidance-interval", nargs=2, type=float)
    infer.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    try:
        if args.command == "build":
            build(args.dry_run)
        elif args.command == "doctor":
            doctor(args)
        elif args.command == "infer":
            if args.step < 1 or args.steps < 1:
                raise ValueError("--step and --steps must be positive")
            run(args)
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
