"""CPU checks for checkpoint integrity, Docker mounts, and native sample validation."""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(SCRIPTS.parent / "adapters"))
import cosmos3


@unittest.skipUnless((cosmos3.UPSTREAM / "docker/write_image_provenance.py").is_file(), "Framework submodule unavailable")
class BuildProvenanceTests(unittest.TestCase):
    def test_wrapper_metadata_satisfies_the_upstream_build_validator(self):
        import os
        import runpy
        from datetime import datetime
        from unittest import mock

        validator = runpy.run_path(str(cosmos3.UPSTREAM / "docker/write_image_provenance.py"))["_source_provenance"]
        with mock.patch.object(cosmos3, "checked") as execute:
            cosmos3.build(dry_run=True)
        command = execute.call_args_list[0].args[0]
        build_args = dict(command[i + 1].split("=", 1) for i, item in enumerate(command[:-1]) if item == "--build-arg")
        with mock.patch.dict(os.environ, build_args):
            source = validator()
        self.assertTrue(source["verified"])
        self.assertFalse(source["dirty"])
        self.assertEqual(source["commit"], cosmos3.REVISION)
        self.assertEqual(len(source["tree"]), 40)
        self.assertIsNotNone(datetime.fromisoformat(source["build_timestamp"].replace("Z", "+00:00")).tzinfo)


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
        self.assertIn("USER=cosmos", command)
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


DEPENDENCIES = importlib.util.find_spec("torch") and importlib.util.find_spec("numpy")
BUNDLE = Path(os.environ.get("BG2_TRAINING_BUNDLE", "/home/sangjunpark/Documents/robotescape_training_example_20261104"))


@unittest.skipUnless(DEPENDENCIES, "Optional sample dependencies unavailable")
class ObservationTests(unittest.TestCase):
    def observation(self):
        import numpy as np
        from bg2_inference import FORMAT, JOINT_NAMES
        return dict(head=np.zeros((384, 512, 3), dtype=np.uint8),
                    wrist_left=np.zeros((256, 256, 3), dtype=np.uint8),
                    wrist=np.zeros((256, 256, 3), dtype=np.uint8),
                    joint_position=np.arange(8, dtype=np.float32), source_time_ns=np.asarray(123),
                    instruction=np.asarray("Pick up the car."), joint_names=np.asarray(JOINT_NAMES),
                    format=np.asarray(FORMAT))

    def load(self, observation):
        import numpy as np
        from bg2_inference import load_observation
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "observation.npz"
            np.savez(path, **observation)
            return load_observation(path)

    def test_current_observation_is_loaded_without_future_labels(self):
        observation = self.load(self.observation())
        self.assertEqual(observation["instruction"], "Pick up the car.")
        self.assertEqual(observation["joint_position"].shape, (8,))

    def test_future_label_field_rejected(self):
        import numpy as np
        observation = self.observation()
        observation["future_actions"] = np.zeros((16, 8))
        with self.assertRaisesRegex(ValueError, "only.*current-observation"):
            self.load(observation)

    def test_wrong_joint_order_rejected(self):
        observation = self.observation()
        observation["joint_names"] = observation["joint_names"][::-1]
        with self.assertRaisesRegex(ValueError, "joint order"):
            self.load(observation)

    def test_bad_pixels_or_nonfinite_state_rejected(self):
        import numpy as np
        observation = self.observation()
        observation["head"] = observation["head"].astype(np.float32)
        with self.assertRaisesRegex(ValueError, "RGB uint8"):
            self.load(observation)
        observation = self.observation()
        observation["joint_position"][0] = np.nan
        with self.assertRaisesRegex(ValueError, "finite"):
            self.load(observation)

    def test_pickle_input_rejected(self):
        import numpy as np
        observation = self.observation()
        observation["instruction"] = np.array({}, dtype=object)
        with self.assertRaises(ValueError):
            self.load(observation)

    def test_inverse_affine_selects_future_rows_without_clipping(self):
        import numpy as np
        from bg2_cosmos.data import quantile_affine
        from bg2_inference import decode_actions
        observation = self.load(self.observation())
        stats = {"q01": [-2.] * 8, "q99": [2.] * 8}
        physical = np.tile(np.arange(17, dtype=np.float32)[:, None], (1, 8))
        physical[0] = observation["joint_position"]
        normalized = quantile_affine(physical, stats).astype(np.float32)
        decoded = decode_actions(normalized, observation, {"normalization": {"joint_position": stats}})
        np.testing.assert_array_equal(decoded, physical[1:])
        self.assertGreater(float(decoded.max()), 2.)

    def test_decoder_floors_only_negative_future_right_gripper(self):
        import numpy as np
        from bg2_inference import decode_actions
        observation = self.load(self.observation())
        contract = {"normalization": {"joint_position": {"q01": [-1.] * 8, "q99": [1.] * 8}}}
        normalized = np.tile(observation["joint_position"], (17, 1))
        normalized[1:, 0] = -.3
        normalized[1:, 7] = np.asarray([-.00101, 0., .012, 1.2] + [.5] * 12)
        original = normalized.copy()
        decoded = decode_actions(normalized, observation, contract)
        np.testing.assert_array_equal(decoded[:, :7], original[1:, :7])
        np.testing.assert_array_equal(decoded[:, 7], [0., *original[2:, 7]])
        np.testing.assert_array_equal(normalized, original)

    def test_decoder_rejects_wrong_shape_and_changed_conditioning(self):
        import numpy as np
        from bg2_inference import decode_actions
        contract = {"normalization": {"joint_position": {"q01": [-1.] * 8, "q99": [1.] * 8}}}
        observation = self.load(self.observation())
        for bad in (np.zeros((16, 8)), np.full((17, 8), np.nan), np.zeros((17, 8))):
            with self.subTest(shape=bad.shape), self.assertRaises(ValueError):
                decode_actions(bad, observation, contract)


@unittest.skipUnless(DEPENDENCIES and (BUNDLE / "data/example/future_joint/manifest.json").is_file(),
                     "Matching training example bundle unavailable; set BG2_TRAINING_BUNDLE")
class TrainingReferenceTests(unittest.TestCase):
    def test_recorded_preparation_matches_actual_training_loader_at_first_middle_and_tail(self):
        import numpy as np
        import torch
        from bg2_cosmos.data import RoomSamples
        from bg2_inference import decode_actions, load_observation, prepare_tensors, verify_adapter
        from prepare_observation import prepare
        torch.set_num_threads(2)
        root = BUNDLE / "data/example/future_joint"
        instructions = BUNDLE / "training/cosmos_futurejoint/configs/instructions.json"
        original = RoomSamples(root, instructions, "train", "canonical")
        contract = {"normalization": json.loads((BUNDLE / "provenance/data/future_joint/joint_train_stats.json").read_text())}
        provenance = json.loads((SCRIPTS.parent / "adapters/SOURCE.json").read_text())
        contract.update(framework_commit=provenance["framework_commit"], data_manifest_sha256=provenance["data_manifest_sha256"],
                        vision={"first_target_offset_frames": 1, "frame_interval_ns": 50_000_000})
        verify_adapter(SCRIPTS.parent / "adapters", contract)
        with tempfile.TemporaryDirectory() as directory:
            for index in (0, len(original) // 2, len(original) - 1):
                with self.subTest(index=index):
                    path = Path(directory) / f"{index}.npz"
                    prepare(root, "3_0", index, path)
                    observation = load_observation(path)
                    prepared = prepare_tensors(observation, contract)
                    reference = original[index]
                    torch.testing.assert_close(prepared["video"], reference["video"], rtol=0, atol=0)
                    torch.testing.assert_close(prepared["action"][0, :8], reference["action"][0], rtol=0, atol=0)
                    self.assertEqual(int(prepared["domain_id"]), 33)
                    self.assertEqual(int(prepared["raw_action_dim"]), 8)
                    self.assertEqual(tuple(prepared["action"].shape), (17, 64))
                    self.assertEqual(int(torch.count_nonzero(prepared["action"][1:])), 0)
                    self.assertEqual(int(torch.count_nonzero(prepared["action"][:, 8:])), 0)
                    normalized = prepared["action"][:, :8].numpy().copy()
                    valid = reference["valid_future_steps"]
                    normalized[1:1 + valid] = reference["action"][1:].numpy()
                    decoded = decode_actions(normalized, observation, contract)
                    np.testing.assert_allclose(decoded[:valid], reference["absolute_targets"].numpy(), rtol=1e-5, atol=2e-7)


if __name__ == "__main__":
    unittest.main()
