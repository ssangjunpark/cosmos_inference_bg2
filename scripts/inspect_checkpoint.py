#!/usr/bin/env python3
"""Inspect the saved configuration, input contract, and DCP tensor metadata on CPU."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from cosmos3 import CHECKPOINT_ROOT, published_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-root", type=Path, default=CHECKPOINT_ROOT)
    parser.add_argument("--variant", choices=("edge", "nano"), default="edge")
    parser.add_argument("--step", type=int, default=20000)
    args = parser.parse_args()
    checkpoint = published_checkpoint(args.checkpoint_root, args.variant, args.step)
    import yaml
    from torch.distributed.checkpoint import FileSystemReader
    config = yaml.safe_load((checkpoint / "config.yaml").read_text())
    contract = json.loads((checkpoint / "input_contract.json").read_text())
    from bg2_inference import verify_adapter
    adapter = verify_adapter(Path(__file__).resolve().parents[1] / "adapters", contract)
    metadata = FileSystemReader(str(checkpoint / "model")).read_metadata()
    projections = {
        key: list(value.size) for key, value in metadata.state_dict_metadata.items()
        if any(part in key for part in ("action2llm", "llm2action", "action_modality_embed"))
    }
    result = {
        "checkpoint": str(checkpoint), "framework_commit": contract["framework_commit"],
        "saved_model_target": config["model"]["_target_"],
        "dcp_tensor_count": len(metadata.state_dict_metadata),
        "dcp_root_prefixes": sorted({key.split(".")[0] for key in metadata.state_dict_metadata}),
        "action_projection_shapes": projections,
        "joint_names": contract["normalization"]["joint_names"],
        "vision": contract["vision"], "prompt_format": contract["prompt_format"],
        "tokenizer_config": config["model"]["config"]["vlm_config"]["tokenizer"],
        "vae_path": config["model"]["config"]["tokenizer"]["vae_path"],
        "adapter": {"source_bundle": adapter["source_bundle"], "source_case": adapter["source_case"],
                    "domain_id": adapter["domain_id"], "action_dim": adapter["action_dim"],
                    "max_action_dim": adapter["max_action_dim"], "future_steps": adapter["future_steps"],
                    "condition_frame_indexes_vision": [0], "condition_frame_indexes_action": [0],
                    "action_start_frame_offset": 0, "source_hashes_verified": True},
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
