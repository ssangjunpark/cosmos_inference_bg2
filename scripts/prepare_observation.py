#!/usr/bin/env python3
"""Extract only current recorded camera frames and joint state, without model execution."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np


def prepare(data_root: Path, episode_id: str, frame: int, output: Path) -> None:
    data_root = data_root.resolve(strict=True)
    manifest = json.loads((data_root / "manifest.json").read_text())
    episode = next((e for e in manifest["episodes"] if e["id"] == episode_id), None)
    if episode is None:
        raise ValueError(f"Unknown recorded episode: {episode_id}")
    arrays = {}
    for camera, expected in (("head", (384, 512, 3)), ("wrist_left", (256, 256, 3)), ("wrist", (256, 256, 3))):
        path = (data_root / episode["videos"][camera]).resolve(strict=True)
        path.relative_to(data_root)
        frames = np.load(path, mmap_mode="r", allow_pickle=False)
        if frame < 0 or frame >= len(frames) or frames.shape[1:] != expected or frames.dtype != np.uint8:
            raise ValueError(f"Invalid frame or camera format: {camera}")
        arrays[camera] = np.array(frames[frame], copy=True)
    joint_path = (data_root / episode["samples"]).resolve(strict=True)
    joint_path.relative_to(data_root)
    with np.load(joint_path, allow_pickle=False) as joints:
        q = joints["joint_positions"][frame].astype(np.float32)
        timestamp = joints["source_time_ns"][frame]
    if q.shape != (8,) or not np.isfinite(q).all():
        raise ValueError("Expected eight finite measured right-arm/gripper values")
    arrays.update(
        joint_position=q, source_time_ns=timestamp,
        instruction=np.asarray(episode["instruction_en"]),
        joint_names=np.asarray(manifest["joint_names"]),
        format=np.asarray("robotescape_recorded_observation_v1"),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as handle:
        np.savez_compressed(handle, **arrays)
    print(f"Saved current observation to {output}")
    print("This is a recorded observation, not a native inference batch or predicted output.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--episode", required=True)
    parser.add_argument("--frame", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("inputs/recorded_observation.npz"))
    args = parser.parse_args()
    try:
        prepare(args.data_root, args.episode, args.frame, args.output)
    except (ValueError, OSError, KeyError) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()
