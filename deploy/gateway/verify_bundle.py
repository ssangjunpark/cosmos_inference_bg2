#!/usr/bin/env python3
"""Verify this copied folder; does not import ROS or contact the robot."""
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parent
manifest = json.loads((root / "MANIFEST.json").read_text())
for name, expected in manifest["files"].items():
    path = (root / name).resolve(strict=True)
    path.relative_to(root)
    if path.stat().st_size != expected["bytes"] or hashlib.sha256(path.read_bytes()).hexdigest() != expected["sha256"]:
        raise SystemExit("Bundle file changed: " + name)
print(json.dumps({"verified_files": len(manifest["files"]), "source_commit": manifest["source_commit"], "source_dirty": manifest["source_dirty"], "robot_connection_opened": False}))
