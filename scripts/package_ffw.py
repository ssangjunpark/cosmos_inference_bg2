#!/usr/bin/env python3
"""Build a small standalone folder for later SFTP; never opens SSH or robot connections."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def build(output):
    output = Path(output)
    if output.exists():
        raise ValueError("Use a new output directory; earlier bundles are preserved")
    output.mkdir(parents=True)
    template = ROOT / "deploy/ffw"
    names = ("compose.yaml", "run.sh", ".env.example", "README.md",
             "verify_bundle.py", "cosmos_ffw/__init__.py", "cosmos_ffw/node.py", "cosmos_ffw/ros_messages.py")
    sources = [template / name for name in names]
    for source in sources:
        target = output / source.relative_to(template)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    for name in ("core.py", "chunks.py"):
        shutil.copyfile(ROOT / "scripts/bg2_bridge" / name, output / "cosmos_ffw" / name)
    (output / "config").mkdir(exist_ok=True)
    shutil.copyfile(ROOT / "profiles/robotis_bg2_rev4.reference.json", output / "config/profile.reference.json")
    files = {}
    for path in sorted(output.rglob("*")):
        if path.is_file():
            files[path.relative_to(output).as_posix()] = {"bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip())
    manifest = {"schema": 1, "purpose": "FFW Cosmos companion; offline prepared, not live commissioned",
                "source_commit": commit, "source_dirty": dirty, "default_mode": "preview",
                "expected_robotis_image": "robotis/ai-worker:2.2.8",
                "runtime": "installed_robotis_image_with_read_only_code", "image_build_required": False, "files": files}
    (output / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/ffw_bundle")
    args = parser.parse_args()
    result = build(args.output)
    print(json.dumps({"folder": str(args.output), "files": len(result["files"]), "bytes": sum(f["bytes"] for f in result["files"].values()), "transferred": False}))
