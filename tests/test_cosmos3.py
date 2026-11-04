"""CPU checks for checkpoint integrity, Docker mounts, and native sample validation."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
import cosmos3


class CheckpointTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.checkpoint = self.root / "cosmos3_edge/020000"
        (self.checkpoint / "model").mkdir(parents=True)
        (self.checkpoint / "model/.metadata").write_bytes(b"test metadata")
        (self.checkpoint / "model/__0_0.distcp").write_bytes(b"test shard")
        (self.checkpoint / "READY.json").write_text(json.dumps({"variant": "edge", "step": 20000}))
        (self.checkpoint / "ACCESS_VERIFIED.json").write_text("{}")
        (self.checkpoint / "input_contract.json").write_text(json.dumps({"framework_commit": cosmos3.REVISION}))
        (self.checkpoint / "config.yaml").write_text("model: {}")
        self.manifest = {
            "format": "pytorch_distributed_checkpoint",
            "files": [
                {"path": str(p.relative_to(self.checkpoint)), "bytes": p.stat().st_size, "sha256": cosmos3.sha256(p)}
                for p in sorted((self.checkpoint / "model").iterdir())
            ],
        }
        self.save_manifest()

    def save_manifest(self):
        (self.checkpoint / "COPY_VERIFIED.json").write_text(json.dumps(self.manifest))

    def tearDown(self):
        self.temporary.cleanup()

    def test_published_checkpoint_and_hashes(self):
        self.assertEqual(cosmos3.published_checkpoint(self.root, "edge", 20000, True), self.checkpoint)

    def test_incomplete_copy_rejected(self):
        (self.checkpoint / "ACCESS_VERIFIED.json").unlink()
        with self.assertRaisesRegex(ValueError, "not published"):
            cosmos3.published_checkpoint(self.root, "edge", 20000)

    def test_modified_shard_rejected(self):
        (self.checkpoint / "model/__0_0.distcp").write_bytes(b"short")
        with self.assertRaisesRegex(ValueError, "incomplete or changed"):
            cosmos3.published_checkpoint(self.root, "edge", 20000)

    def test_manifest_traversal_rejected(self):
        self.manifest["files"][0]["path"] = "../../escape"
        self.save_manifest()
        with self.assertRaisesRegex(ValueError, "Unsafe path"):
            cosmos3.published_checkpoint(self.root, "edge", 20000)

    def test_extra_replica_shard_rejected(self):
        (self.checkpoint / "model/__1_0.distcp").write_bytes(b"")
        with self.assertRaisesRegex(ValueError, "shard set"):
            cosmos3.published_checkpoint(self.root, "edge", 20000)

    def test_hash_detects_same_length_corruption(self):
        shard = self.checkpoint / "model/__0_0.distcp"
        shard.write_bytes(b"x" * shard.stat().st_size)
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            cosmos3.published_checkpoint(self.root, "edge", 20000, True)


class ContainerTests(unittest.TestCase):
    def test_output_only_container_is_isolated(self):
        command = cosmos3.container("5", cosmos3.ROOT / "outputs/test", True)
        self.assertIn("--network=none", command)
        self.assertIn("--read-only", command)
        self.assertIn("--user", command)
        self.assertNotIn("--privileged", command)
        self.assertNotIn("--net=host", command)
        self.assertNotIn("-p", command)
        self.assertNotIn("--ipc=host", command)
        for index, item in enumerate(command[:-1]):
            if item == "--mount" and not command[index + 1].endswith(",readonly"):
                source = command[index + 1].split("source=", 1)[1].split(",", 1)[0]
                Path(source).relative_to(cosmos3.ROOT)
        self.assertTrue(cosmos3.bind(Path("/mnt/checkpoint"), "/checkpoint")[1].endswith(",readonly"))

    def test_gpu_argument_cannot_inject_options(self):
        with self.assertRaisesRegex(ValueError, "physical GPU index"):
            cosmos3.container("all --privileged", cosmos3.ROOT / "outputs/test", True)


    def test_hf_snapshot_keeps_blob_symlinks_inside_mount(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            snapshot = root / "models--example--tokenizer/snapshots/revision"
            snapshot.mkdir(parents=True)
            blobs = snapshot.parent.parent / "blobs"
            blobs.mkdir()
            (blobs / "blob").write_text("tokenizer")
            (snapshot / "tokenizer_config.json").symlink_to("../../blobs/blob")
            mount, inside = cosmos3.tokenizer_bind(snapshot)
            self.assertIn(f"source={snapshot.parent.parent},", mount[1])
            self.assertTrue(mount[1].endswith(",readonly"))
            self.assertEqual(inside, "/assets/tokenizer/snapshots/revision")

    def test_external_tokenizer_symlink_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tokenizer = root / "tokenizer"
            tokenizer.mkdir()
            (root / "outside").write_text("private")
            (tokenizer / "tokenizer_config.json").symlink_to("../outside")
            with self.assertRaisesRegex(ValueError, "escapes its read-only mount"):
                cosmos3.tokenizer_bind(tokenizer)


class ConfigRelocationTests(unittest.TestCase):
    def test_edge_and_nano_keep_the_original_model_target(self):
        from native_infer import relocate_model_config
        for key in ("tokenizer_type", "pretrained_model_name"):
            with self.subTest(tokenizer_field=key):
                original = {"model": {"_target_": "bg2_cosmos.model.BG2PolicyModel", "config": {
                    "vlm_config": {"tokenizer": {key: "/scratch/tokenizer"}},
                    "tokenizer": {"vae_path": "/scratch/vae"}, "ema": {"enabled": True}}}}
                derived = relocate_model_config(original, Path("/assets/tokenizer"), Path("/assets/vae"))
                self.assertEqual(derived["_target_"], original["model"]["_target_"])
                self.assertEqual(derived["config"]["vlm_config"]["tokenizer"][key], "/assets/tokenizer")
                self.assertEqual(original["model"]["config"]["vlm_config"]["tokenizer"][key], "/scratch/tokenizer")
                self.assertFalse(derived["config"]["ema"]["enabled"])


@unittest.skipUnless(importlib.util.find_spec("torch") and importlib.util.find_spec("numpy"), "Optional sample dependencies unavailable")
class SampleTests(unittest.TestCase):
    def test_observation_cannot_be_mistaken_for_native_batch(self):
        import numpy as np
        from native_infer import load_sample
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "recorded.npz"
            np.savez(path, head=np.zeros((384, 512, 3), dtype=np.uint8))
            with self.assertRaisesRegex(ValueError, "missing 'ai_caption'"):
                load_sample(path)

    def test_pickle_input_is_rejected(self):
        import numpy as np
        from native_infer import load_sample
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "unsafe.npz"
            np.savez(path, value=np.array([{}], dtype=object))
            with self.assertRaises(ValueError):
                load_sample(path)


if __name__ == "__main__":
    unittest.main()
