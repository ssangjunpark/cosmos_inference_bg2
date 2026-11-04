#!/usr/bin/env python3
"""Fetch only the tokenizer and VAE revisions explicitly recorded in a saved config."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
PROCESSOR_FILES = [
    "config.json", "tokenizer*.json", "special_tokens_map.json",
    "vocab.json", "merges.txt", "preprocessor_config.json",
    "processor_config.json", "chat_template*", "video_preprocessor*",
]


def snapshot_reference(path: str) -> tuple[str, str, str]:
    match = re.search(r"/models--([^/]+)/snapshots/([0-9a-f]{40})(?:/(.*))?$", path)
    if not match or "--" not in match.group(1):
        raise ValueError(f"No exact Hugging Face repository/revision in saved path: {path}")
    owner, model = match.group(1).split("--", 1)
    return f"{owner}/{model}", match.group(2), match.group(3) or ""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--tokenizer-only", action="store_true")
    args = parser.parse_args()

    import yaml
    config = yaml.safe_load(args.config.read_text())["model"]["config"]
    node = config["vlm_config"]["tokenizer"]
    source = node.get("tokenizer_type") or node.get("pretrained_model_name")
    if not isinstance(source, str):
        raise ValueError("Saved config does not specify a supported tokenizer source")
    repo, revision, subpath = snapshot_reference(source)
    if subpath:
        raise ValueError("Tokenizer subdirectories require inspection of the original asset layout")
    vae_repo, vae_revision, vae_filename = snapshot_reference(config["tokenizer"]["vae_path"])
    if vae_filename != "Wan2.2_VAE.pth":
        raise ValueError("Unexpected VAE asset; inspect the original config before downloading")

    # Keep download/Xet caches in this workspace, without changing host HOME.
    cache = ROOT / ".runtime/cache"
    os.environ["HF_XET_CACHE"] = str(cache / "xet")
    from huggingface_hub import hf_hub_download, snapshot_download

    tokenizer = ROOT / "assets" / f"{repo.replace('/', '--')}-{revision}"
    snapshot_download(
        repo_id=repo, revision=revision, allow_patterns=PROCESSOR_FILES,
        local_dir=tokenizer, cache_dir=cache / "huggingface", max_workers=2,
    )
    print(f"Tokenizer: {tokenizer}", flush=True)
    result = {"config": str(args.config.resolve()), "tokenizer": str(tokenizer),
              "tokenizer_repo": repo, "tokenizer_revision": revision}
    if not args.tokenizer_only:
        vae_directory = ROOT / "assets" / f"{vae_repo.replace('/', '--')}-{vae_revision}"
        vae = hf_hub_download(
            repo_id=vae_repo, revision=vae_revision, filename=vae_filename,
            local_dir=vae_directory, cache_dir=cache / "huggingface",
        )
        result.update(vae=str(Path(vae).resolve()), vae_repo=vae_repo, vae_revision=vae_revision)
        print(f"VAE: {vae}", flush=True)
    filename = f"{args.config.parent.parent.name}-{args.config.parent.name}-assets.json"
    receipt = ROOT / "assets" / filename
    receipt.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Asset receipt: {receipt}", flush=True)


if __name__ == "__main__":
    main()
