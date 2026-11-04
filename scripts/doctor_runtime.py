#!/usr/bin/env python3
"""Check image identity, CUDA execution, and actual media decoding without model weights."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path
import subprocess
import tempfile


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-commit", required=True)
    args = parser.parse_args()

    import torch
    from cosmos_framework.utils.generator.torchcodec_dependencies import load_torchcodec

    provenance = json.loads(Path("/opt/cosmos/image-provenance.json").read_text())
    if not provenance.get("source_provenance_verified") or provenance.get("repository_commit") != args.expected_commit:
        raise RuntimeError("Image provenance does not match the expected verified framework source")
    torch_version = importlib.metadata.version("torch")
    if provenance.get("packages", {}).get("torch") != torch_version:
        raise RuntimeError("Image runtime provenance is stale after dependency installation")
    if torch.cuda.device_count() != 1:
        raise RuntimeError("The launcher must expose exactly the one selected GPU")
    torch.cuda.init()
    value = torch.ones(1, device="cuda").add_(1)
    torch.cuda.synchronize()
    if value.item() != 2:
        raise RuntimeError("CUDA arithmetic check failed")

    codec = load_torchcodec()
    with tempfile.TemporaryDirectory(prefix="cosmos-media-check-") as directory:
        clip = Path(directory) / "test.mp4"
        subprocess.run([
            "ffmpeg", "-hide_banner", "-loglevel", "error",
            "-f", "lavfi", "-i", "testsrc=size=32x32:rate=1",
            "-frames:v", "1", "-c:v", "mpeg4", str(clip),
        ], check=True)
        frame = codec.decoders.VideoDecoder(str(clip), device="cpu")[0]
        if tuple(frame.shape) != (3, 32, 32):
            raise RuntimeError(f"Unexpected decoded frame shape: {tuple(frame.shape)}")

    print(json.dumps({
        "framework_commit": provenance["repository_commit"],
        "source_provenance_verified": True,
        "torch": torch_version, "cuda_build": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(),
        "torchcodec": codec.__version__,
        "decoded_frame_shape": list(frame.shape),
        "cuda_and_media_checks": "passed",
    }, indent=2))


if __name__ == "__main__":
    main()
