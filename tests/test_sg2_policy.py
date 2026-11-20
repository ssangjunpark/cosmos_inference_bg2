import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from bg2_bridge.observations import decode_view
from sg2_policy import Predictor
from sg2_wam import (ARM_NAMES, DATA_FORMAT, FORMAT, VARIANTS, calibration, camera_specs,
                     decode_prediction, inspect_weights, settings, training_observation)


def observation():
    return dict(format=FORMAT, joint_names=ARM_NAMES, joint_position=np.linspace(-.3, .3, 16),
                source_time_ns=np.int64(37), head=np.zeros((384, 512, 3), np.uint8),
                wrist_left=np.zeros((256, 256, 3), np.uint8), wrist=np.zeros((256, 256, 3), np.uint8))


class SG2PolicyTests(unittest.TestCase):
    def test_four_distinct_normalizers_and_conditioning_rows(self):
        specs = [settings(v) for v in VARIANTS]
        self.assertEqual(len({tuple(s["q01"]) for s in specs}), 4)
        for spec in specs:
            names = spec["joint_names"]
            dim = len(names)
            measured = observation()
            measured.update(format=DATA_FORMAT, source_time_ns=np.int64(0), joint_names=ARM_NAMES[::-1], joint_position=measured["joint_position"][::-1])
            selected = training_observation(measured, spec)
            np.testing.assert_array_equal(selected["joint_position"], observation()["joint_position"][-dim:])
            contract = {"normalization": {"joint_position": spec}}
            center = (np.array(spec["q01"]) + spec["q99"]) / 2
            scale = np.maximum((np.array(spec["q99"]) - spec["q01"]) / 2, 1e-6)
            rows = np.zeros((17, 64))
            rows[0, :dim] = (selected["joint_position"] - center) / scale
            rows[1:, :dim] = 1.5
            future = decode_prediction(rows, selected, contract, names)
            self.assertEqual(future.shape, (16, dim))
            np.testing.assert_allclose(future[-1], center + 1.5 * scale, rtol=1e-6)
            with self.assertRaises(ValueError):
                decode_prediction(rows[1:], selected, contract, names)
            rows[0, 0] += 1.
            with self.assertRaises(ValueError):
                decode_prediction(rows, selected, contract, names)

    def test_pnp1_holds_left_and_maps_gripper_through_both_directions(self):
        obs = observation()
        for variant in ("pnp1_normal", "pnp2_normal"):
            spec = settings(variant)
            names = spec["joint_names"]
            count = len(names)
            scale = np.full(count, 2.)
            scale[-1] = -3.
            conversion = scale, np.full(count, .25)
            q = training_observation(obs, spec, conversion)["joint_position"]
            model_output = np.tile(q, (16, 1))
            model_output[:, -1] += .1
            prediction = Predictor.__new__(Predictor)
            prediction.names, prediction.conversion = names, conversion
            with patch.object(prediction, "infer", return_value={"positions": model_output}):
                chunk = prediction.predict(obs, "epoch", "request")
            expected = obs["joint_position"].copy()
            expected[-1] -= .3
            np.testing.assert_allclose(chunk["positions"], np.tile(expected, (16, 1)), atol=1e-10)
            self.assertEqual((chunk["joint_names"], chunk["source_time_ns"], chunk["schema"]),
                             (ARM_NAMES, 37, "cosmos_sg2_joint_chunk_v1"))

    def test_uncalibrated_controller_state_and_bad_calibration_rejected(self):
        spec = settings("pnp2_normal")
        with self.assertRaisesRegex(ValueError, "calibration"):
            training_observation(observation(), spec)
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "calibration.json"
            entries = {n: {"scale": 1, "offset": 0} for n in ARM_NAMES}
            entries[ARM_NAMES[-1]]["scale"] = 0
            path.write_text(json.dumps(entries))
            with self.assertRaises(ValueError):
                calibration(path, ARM_NAMES)
            del entries[ARM_NAMES[-1]]
            path.write_text(json.dumps(entries))
            with self.assertRaises(ValueError):
                calibration(path, ARM_NAMES)

    def test_camera_geometry_raw_square_orientation_and_head_letterbox(self):
        specs = camera_specs(("/head", "/left", "/right"))
        for key, spec in specs.items():
            w, h = spec["raw_wh"]
            rows, cols = np.indices((h, w))
            rgb = np.stack((rows, cols, (rows + cols) % 251), axis=-1).astype(np.uint8)
            raw = Image.fromarray(rgb)
            buf = io.BytesIO()
            raw.save(buf, format="PNG")
            if key == "head":
                expected = np.zeros((384, 512, 3), np.uint8)
                resized = raw.crop((112, 88, 560, 376)).resize((512, 329), Image.Resampling.BILINEAR)
                expected[28:357] = np.asarray(resized)
            else:
                expected = np.asarray(raw.crop((0, 0, 240, 240)).transpose(Image.Transpose.ROTATE_90)
                                      .resize((256, 256), Image.Resampling.BILINEAR))
            np.testing.assert_array_equal(decode_view(buf.getvalue(), key, spec), expected)

    @unittest.skipUnless(importlib.util.find_spec("torch"), "DCP requires PyTorch")
    def test_weight_publications_check_run_and_storage(self):
        import torch
        import torch.distributed.checkpoint as dcp
        with tempfile.TemporaryDirectory() as d:
            checkpoint = Path(d)
            model = checkpoint / "model"
            dcp.save({"weight": torch.arange(4)}, checkpoint_id=model)
            spec = {"run_name": "test_sg2_run", "source_sha256": {}}
            for name in ("config.yaml", "input_contract.json", "vision_contract.py"):
                path = checkpoint / name
                path.write_text("test")
                spec["source_sha256"][name] = hashlib.sha256(path.read_bytes()).hexdigest()
            ready_path = checkpoint / "READY.json"
            ready = dict(run="/training/test_sg2_run", step=10000, format="pytorch_distributed_checkpoint",
                         checkpoint="model", weights_only=True, native_save_complete=True, verified=True)
            ready_path.write_text(json.dumps(ready))
            self.assertEqual(inspect_weights(checkpoint, spec)["step"], 10000)
            with self.assertRaisesRegex(ValueError, "another SG2 run"):
                inspect_weights(checkpoint, {**spec, "run_name": "wrong_run"})
            # Milestone READY has a tier and a separate file manifest.
            ready_path.write_text(json.dumps({k: v for k, v in ready.items() if k != "weights_only"}))
            records = [dict(path="model/" + p.name, bytes=p.stat().st_size,
                            sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in model.iterdir()]
            (checkpoint / "VERIFIED.json").write_text(json.dumps(dict(step=10000, files=records)))
            self.assertEqual(inspect_weights(checkpoint, spec)["step"], 10000)
            next(model.glob("*.distcp")).write_bytes(b"x")
            with self.assertRaisesRegex(ValueError, "Incomplete SG2 shard"):
                inspect_weights(checkpoint, spec)
