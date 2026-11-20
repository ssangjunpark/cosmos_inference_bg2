"""CPU integration of resident SG2 RPC, parallel inference, history and latched mode."""

import json
import sys
import tempfile
import threading
import unittest
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "tests")]
from bg2_bridge.core import JointFeedback
from bg2_bridge.gateway import Gateway
from bg2_bridge.sg2_dual import (
    ParallelLatchedGate,
    SG2PolicyClient,
    from_config,
)
from bg2_bridge.sg2_history import HistoryObservations
from bg2_bridge.sg2_monitor import SelectionMonitor
from serve_sg2_pair import model_commands, run_pair
from sg2_policy_service import make_server
from sg2_wam import ARM_NAMES, FORMAT, camera_specs
from test_sg2_policy import observation


def config():
    cfg = json.loads((ROOT / "configs/sg2_jepa.example.json").read_text())
    return cfg


def obs():
    value = observation()
    value.update(
        past=np.zeros((2, 192, 256, 3), dtype=np.uint8), frame_interval_ns=50_000_000
    )
    return value


def policy(value=0, barrier=None):
    def predict(observation, epoch, request_id):
        if barrier:
            barrier.wait(timeout=2)
        return {
            "schema": "cosmos_sg2_joint_chunk_v1",
            "epoch": epoch,
            "request_id": request_id,
            "source_time_ns": int(observation["source_time_ns"]),
            "joint_names": ARM_NAMES,
            "units": ["rad"] * 16,
            "positions": np.full((16, 16), value).tolist(),
            "frame_interval_ns": 50_000_000,
            "future": np.full((16, 192, 256, 3), value, dtype=np.uint8),
        }

    return Mock(predict=Mock(side_effect=predict))


class DualTests(unittest.TestCase):
    def gate(self, loss=1, barrier=None):
        return ParallelLatchedGate(
            policy(0, barrier),
            policy(1, barrier),
            Mock(score=Mock(return_value=loss)),
            config(),
        )

    def test_parallel_same_input_and_latched_mode_survives_primary_jepa_outage(self):
        gate = self.gate(2.1, threading.Barrier(2))
        observed = obs()
        result = gate.predict(observed, "e", "r")
        self.assertEqual(result["selection"]["selected_model"], "CM2")
        self.assertIs(gate.primary.predict.call_args.args[0], observed)
        self.assertIs(gate.fallback.predict.call_args.args[0], observed)
        gate.primary.predict.side_effect = RuntimeError("CM1 down")
        gate.judge.score.side_effect = RuntimeError("JEPA down")
        gate.fallback.predict.side_effect = policy(1).predict.side_effect
        for i in range(3):
            self.assertEqual(
                gate.predict(observed, "new-epoch", str(i))["selection"]["reason"],
                "fallback_latched",
            )
        self.assertEqual(gate.primary.predict.call_count, 1)
        self.assertEqual(gate.judge.score.call_count, 1)
        self.assertEqual(gate.fallback.predict.call_count, 4)

    def test_threshold_equality_and_fallback_failures(self):
        gate = self.gate(2.0)
        self.assertEqual(
            gate.predict(obs(), "e", "r")["selection"]["selected_model"], "CM1"
        )
        self.assertFalse(gate.fallback_active)
        gate.judge.score.return_value = float("nan")
        with self.assertRaises(ValueError):
            gate.predict(obs(), "e", "r2")
        gate = self.gate(3)
        gate.fallback.predict.side_effect = TimeoutError("CM2 timeout")
        with self.assertRaises(TimeoutError):
            gate.predict(obs(), "e", "r")
        self.assertFalse(gate.fallback_active)
        gate = self.gate(3)
        gate.predict(obs(), "e", "r")
        gate.fallback.predict.side_effect = TimeoutError("CM2 unavailable")
        with self.assertRaises(TimeoutError):
            gate.predict(obs(), "e", "r2")
        self.assertTrue(gate.fallback_active)

    def test_gateway_one_session_switch_ack_and_no_video_on_robot_wire(self):
        gate = self.gate(3)
        current = {
            "epoch": "e",
            "cycle": 0,
            "needs_chunk": True,
            "phase": "holding",
            "mode": "preview",
            "robot_now_ns": 37,
            "ready_since_ns": 0,
        }
        status = Mock(get=Mock(side_effect=lambda now: (dict(current), 0)))
        observed = obs()
        inputs = Mock(
            latest_source_ns=Mock(side_effect=lambda: int(observed["source_time_ns"])),
            snapshot=Mock(side_effect=lambda *a, **k: dict(observed)),
        )
        send = Mock()
        gateway = Gateway(inputs, status, Mock(), send, "test", selection_gate=gate)
        self.assertEqual(gateway.step(now_ns=1)["event"], "forwarded")
        first = send.call_args.args[0]
        self.assertNotIn("future", first)
        self.assertEqual(np.asarray(first["positions"])[0, 0], 1)
        current.update(cycle=1, last_completed_request_id=first["request_id"])
        observed["source_time_ns"] = np.asarray(38)
        self.assertEqual(gateway.step(now_ns=2)["event"], "forwarded")
        self.assertEqual(send.call_count, 2)
        self.assertEqual(send.call_args.args[0]["epoch"], "e")
        self.assertEqual(gate.primary.predict.call_count, 1)
        # A changed session after prediction must discard even a latched CM2 chunk.
        current.update(
            cycle=2, last_completed_request_id=send.call_args.args[0]["request_id"]
        )
        observed["source_time_ns"] = np.asarray(39)
        original = gate.fallback.predict.side_effect

        def changed(*args):
            result = original(*args)
            current["epoch"] = "stale"
            return result

        gate.fallback.predict.side_effect = changed
        self.assertEqual(gateway.step(now_ns=3)["event"], "discarded")
        self.assertEqual(send.call_count, 2)

    def test_monitor_records_submitted_and_preserves_latch(self):
        with tempfile.TemporaryDirectory() as directory:
            monitor = SelectionMonitor(directory)
            gate = self.gate(3)
            gate.event = monitor
            monitor.gate = gate
            gate.predict(obs(), "e", "r")
            gate.execution_event("submitted")
            data = json.loads(Path(directory, "report.json").read_text())
            self.assertEqual(data["state"], "submitted")
            self.assertTrue(data["fallback_active"])
            self.assertIn(
                "data:image/gif;base64", Path(directory, "index.html").read_text()
            )

    def test_binary_native_service_decodes_future_and_maps_controller_units(self):
        observed = observation()
        native = np.zeros((17, 640, 512, 3), dtype=np.uint8)
        native[0] = 255  # Reconstructed current frame must never appear in future.
        native[1:, :, :, 1] = 50
        predictor = NS(
            names=ARM_NAMES,
            conversion=(np.ones(16), np.zeros(16)),
            model=NS(runtime_metadata={"variant": "pnp2_normal"}),
            infer=Mock(
                return_value={"positions": np.zeros((16, 16)), "imagined_video": native}
            ),
        )
        server = make_server(("127.0.0.1", 0), predictor, "CM1")
        thread = threading.Thread(target=server.serve_forever)
        thread.start()
        try:
            client = SG2PolicyClient(f"http://127.0.0.1:{server.server_port}", 2, "CM1")
            self.assertTrue(client.health()["ready"])
            result = client.predict(observed, "e", "r")
            self.assertEqual(result["future"].shape, (16, 192, 256, 3))
            self.assertEqual(result["future"][0, 0, 0].tolist(), [0, 50, 0])
            self.assertEqual(np.asarray(result["positions"]).shape, (16, 16))
            predictor.infer.assert_called_once()
            self.assertTrue(predictor.infer.call_args.kwargs["decode_video"])
        finally:
            server.shutdown()
            thread.join()
            server.server_close()

    def test_two_real_services_parallel_switch_and_continue_after_cm1_stops(self):
        servers, threads, predictors = [], [], []
        cfg = config()
        try:
            for role, value in (("primary", 0), ("fallback", 1)):
                predictor = NS(
                    names=ARM_NAMES,
                    conversion=(np.ones(16), np.zeros(16)),
                    model=NS(runtime_metadata={"variant": cfg[role + "_variant"]}),
                    infer=Mock(
                        return_value={
                            "positions": np.full((16, 16), value),
                            "imagined_video": np.full(
                                (17, 640, 512, 3), value, dtype=np.uint8
                            ),
                        }
                    ),
                )
                server = make_server(("127.0.0.1", 0), predictor, cfg[role + "_id"])
                thread = threading.Thread(target=server.serve_forever)
                servers.append(server)
                threads.append(thread)
                predictors.append(predictor)
                thread.start()
                cfg[role + "_url"] = f"http://127.0.0.1:{server.server_port}"
            gate = from_config(cfg)
            gate.judge = Mock(score=Mock(return_value=3))
            observed = obs()
            selected = gate.predict(observed, "e", "r1")
            self.assertEqual(selected["selection"]["selected_model"], "CM2")
            self.assertEqual(np.asarray(selected["positions"])[0, 0], 1)
            for predictor in predictors:
                self.assertEqual(predictor.infer.call_count, 1)
            servers[0].shutdown()
            threads[0].join()
            servers[0].server_close()
            gate.judge.score.side_effect = TimeoutError("JEPA stopped")
            selected = gate.predict(observed, "e", "r2")
            self.assertEqual(selected["selection"]["reason"], "fallback_latched")
            self.assertEqual(predictors[1].infer.call_count, 2)
            self.assertEqual(gate.judge.score.call_count, 1)
        finally:
            for server, thread in zip(servers, threads):
                if thread.is_alive():
                    server.shutdown()
                    thread.join()
                server.server_close()

    def test_service_health_rejects_mismatched_contract(self):
        with (
            patch.object(
                SG2PolicyClient,
                "health",
                return_value={"ready": True, "model_id": "other"},
            ),
            self.assertRaises(ValueError),
        ):
            from_config(config())

    def test_supervisor_preserves_cm2_when_cm1_exits(self):
        import signal

        handlers = {}

        def install(sig, handler):
            handlers[sig] = handler
            return signal.SIG_DFL

        first, second = Mock(), Mock()
        first.poll.side_effect = [None, 1, 1]
        second.poll.return_value = None

        def running(_seconds):
            self.assertFalse(second.terminate.called)
            handlers[signal.SIGINT]()

        cfg = config()
        with (
            patch("serve_sg2_pair.model_commands", return_value=[["cm1"], ["cm2"]]),
            patch("serve_sg2_pair.subprocess.Popen", side_effect=[first, second]),
            patch("serve_sg2_pair.signal.signal", side_effect=install),
            patch("serve_sg2_pair.time.sleep", side_effect=running),
            patch.object(
                SG2PolicyClient,
                "health",
                side_effect=[
                    {"model_id": "CM1", "ready": True},
                    {"model_id": "CM2", "ready": True},
                ],
            ),
        ):
            run_pair(cfg)
        second.terminate.assert_called_once()  # Only after the simulated operator interrupt.

    def test_pair_commands_expand_home_without_checking_unmoved_weights(self):
        cfg = config()
        cfg["model_runtime"]["sampling"].update(steps=4, guidance=3, shift=5)
        commands = model_commands(cfg)
        self.assertEqual(len(commands), 2)
        self.assertIn(str(Path("~/checkpoints/sg2-primary").expanduser()), commands[0])
        self.assertIn("CM2", commands[1])


class HistoryTests(unittest.TestCase):
    def setUp(self):
        self.profile = json.loads(
            (ROOT / "profiles/robotis_sg2_rev1.reference.json").read_text()
        )
        self.profile["cameras"] = camera_specs(("/h", "/l", "/r"))
        self.inputs = HistoryObservations(
            self.profile, joint_names=ARM_NAMES, observation_format=FORMAT
        )
        self.source = 1_800_000_000_000_000_000
        self.received = 10_000_000_000
        for offset in (-50_000_000, 0):
            source = self.source + offset
            for key, spec in self.profile["cameras"].items():
                image = Image.new(
                    "RGB", tuple(spec["raw_wh"]), (50 if offset else 100, 0, 0)
                )
                stream = BytesIO()
                image.save(stream, format="PNG")
                self.inputs.add_camera(
                    key, source, self.received + offset, stream.getvalue()
                )
        feedback = JointFeedback.from_arrays(
            ARM_NAMES, [0.0] * 16, [0.0] * 16, self.source, self.received, self.profile
        )
        self.inputs.add_joints(feedback)

    def test_history_order_geometry_and_hold_boundary(self):
        result = self.inputs.snapshot(
            "test",
            self.source,
            receipt_now_ns=self.received,
            not_before_ns=self.source - 50_000_000,
        )
        self.assertEqual(result["past"].shape, (2, 192, 256, 3))
        self.assertEqual(
            result["past_source_time_ns"].tolist(),
            [self.source - 50_000_000, self.source],
        )
        self.assertLess(result["past"][0].sum(), result["past"][1].sum())
        with self.assertRaises(ValueError):
            self.inputs.snapshot(
                "test",
                self.source,
                receipt_now_ns=self.received,
                not_before_ns=self.source,
            )

    def test_latched_mode_does_not_wait_for_jepa_history(self):
        self.inputs.history_required = lambda: False
        result = self.inputs.snapshot(
            "test", self.source, receipt_now_ns=self.received, not_before_ns=self.source
        )
        self.assertEqual(len(result["past"]), 1)
        self.assertEqual(result["past_source_time_ns"].tolist(), [self.source])

    def test_camera_gap_rejected(self):
        self.inputs.cameras["head"].popleft()
        with self.assertRaises(ValueError):
            self.inputs.snapshot("test", self.source, receipt_now_ns=self.received)


if __name__ == "__main__":
    unittest.main()
