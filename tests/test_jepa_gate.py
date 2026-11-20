"""CPU-only SG2 selection, real local HTTP, replay UI, and gateway guards."""

import json
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import Mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from bg2_bridge.gateway import Gateway
from bg2_bridge.jepa_gate import JepaClient, JepaGate, from_config
from bg2_bridge.sg2_replay import replay


def config():
    return {
        "threshold": 0.5,
        "frame_interval_ns": 50_000_000,
        "joint_names": ["sg2_joint"],
        "units": ["rad"],
        "primary_id": "primary",
        "fallback_id": "alternative",
        "timeout_seconds": 1,
    }


def observation():
    return {
        "past": np.zeros((2, 4, 4, 3), dtype=np.uint8),
        "source_time_ns": 12,
        "frame_interval_ns": 50_000_000,
        "joint_position": np.zeros(1),
        "joint_names": ["sg2_joint"],
    }


def policy(value=0):
    def predict(obs, epoch, request_id):
        return {
            "epoch": epoch,
            "request_id": request_id,
            "source_time_ns": int(obs["source_time_ns"]),
            "joint_names": ["sg2_joint"],
            "units": ["rad"],
            "positions": [[value]],
            "frame_interval_ns": 50_000_000,
            "future": np.full((2, 4, 4, 3), value, dtype=np.uint8).tolist(),
        }

    return Mock(predict=Mock(side_effect=predict))


class GateTests(unittest.TestCase):
    def gate(self, loss):
        return JepaGate(
            policy(), policy(1), Mock(score=Mock(return_value=loss)), config()
        )

    def test_threshold_and_reset(self):
        for loss, expected in (
            (0.49, "primary"),
            (0.5, "primary"),
            (0.51, "alternative"),
        ):
            gate = self.gate(loss)
            result = gate.predict(observation(), "e", "r")
            self.assertEqual(result["selection"]["selected_model"], expected)
            gate.judge.score.return_value = 0.1
            self.assertEqual(
                gate.predict(observation(), "e", "r2")["selection"]["selected_model"],
                "primary",
            )
            self.assertEqual(gate.primary.predict.call_count, 2)
            self.assertEqual(gate.fallback.predict.call_count, int(loss > 0.5))

    def test_bad_scores_and_service_errors_discard(self):
        for loss in (float("nan"), float("inf"), -1):
            gate = self.gate(loss)
            with self.assertRaises(ValueError):
                gate.predict(observation(), "e", "r")
            self.assertEqual(gate.last_event["state"], "discarded")
            gate.fallback.predict.assert_not_called()
        for error in (TimeoutError("timeout"), OSError("service unavailable")):
            gate = self.gate(0.1)
            gate.judge.score.side_effect = error
            with self.assertRaises(type(error)):
                gate.predict(observation(), "e", "r")

    def test_fallback_failure_and_contract_rejection(self):
        gate = self.gate(1)
        gate.fallback.predict.side_effect = RuntimeError("failed")
        with self.assertRaises(RuntimeError):
            gate.predict(observation(), "e", "r")
        for key, value in (
            ("joint_names", ["bg2"]),
            ("positions", [[float("nan")]]),
            ("epoch", "stale"),
            ("frame_interval_ns", 1),
            ("future", [[1]]),
        ):
            gate = self.gate(0.1)
            original = gate.primary.predict.side_effect
            gate.primary.predict.side_effect = (
                lambda *a, k=key, v=value, original=original: {
                    **original(*a),
                    k: v,
                }
            )
            with self.assertRaises(ValueError):
                gate.predict(observation(), "e", "r")

    def test_invalid_configuration_and_observation(self):
        for threshold in (None, float("nan"), -1):
            with self.assertRaises((ValueError, TypeError)):
                JepaGate(
                    policy(), policy(), Mock(), {**config(), "threshold": threshold}
                )
        gate = self.gate(0.1)
        with self.assertRaises(ValueError):
            gate.predict({**observation(), "frame_interval_ns": 1}, "e", "r")

    def test_ui_and_no_stale_actions(self):
        gate = self.gate(1)
        with tempfile.TemporaryDirectory() as directory:
            report = replay(gate, observation(), directory)
            self.assertEqual(report["execution_state"], "recorded_only")
            self.assertEqual(report["selected_model"], "alternative")
            self.assertIn(
                "data:image/gif;base64", Path(directory, "index.html").read_text()
            )
            self.assertTrue(Path(directory, "selection.json").exists())
            gate.judge.score.side_effect = TimeoutError("timeout")
            self.assertEqual(
                replay(gate, observation(), directory)["state"], "discarded"
            )
            self.assertFalse(Path(directory, "selection.json").exists())

    def test_gateway_failure_and_session_change_send_nothing(self):
        current = {
            "epoch": "e",
            "cycle": 0,
            "needs_chunk": True,
            "phase": "holding",
            "mode": "preview",
            "robot_now_ns": 12,
            "ready_since_ns": 0,
        }
        for changed, loss in ((False, float("nan")), (True, 0.1), (False, 0.1)):
            status = Mock(
                get=Mock(
                    side_effect=[
                        (current, 0),
                        ({**current, "epoch": "new"} if changed else current, 0),
                    ]
                )
            )
            obs = Mock(
                latest_source_ns=Mock(return_value=12),
                snapshot=Mock(return_value=observation()),
            )
            send = Mock()
            gate = self.gate(loss)
            gateway = Gateway(obs, status, Mock(), send, "test", selection_gate=gate)
            event = gateway.step(now_ns=1)
            self.assertEqual(
                event["event"],
                "forwarded" if not changed and loss == 0.1 else "discarded",
            )
            self.assertEqual(send.call_count, int(not changed and loss == 0.1))

    def test_real_http_timeout_and_wrong_identity(self):
        import time

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                self.rfile.read(int(self.headers["Content-Length"]))
                time.sleep(0.08)
                try:
                    self.send_response(200)
                    self.end_headers()
                    self.wfile.write(
                        b'{"observation_id":"wrong","model_id":"wrong","loss":0.1}'
                    )
                except (BrokenPipeError, ConnectionResetError):
                    pass

            def log_message(self, *args):
                pass

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever)
        thread.start()
        try:
            url = f"http://127.0.0.1:{server.server_port}"
            frames = observation()["past"]
            with self.assertRaises(TimeoutError):
                JepaClient(url, 0.01).score(frames, frames, "r", "m")
            with self.assertRaises(ValueError):
                JepaClient(url, 1).score(frames, frames, "r", "m")
        finally:
            server.shutdown()
            thread.join()
            server.server_close()

    def test_real_http_replay(self):
        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                data = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                if self.path == "/score":
                    reply = {
                        "observation_id": data["observation_id"],
                        "model_id": data["model_id"],
                        "loss": 0.7,
                    }
                else:
                    reply = policy(1).predict(
                        data["observation"], data["epoch"], data["request_id"]
                    )
                    reply["model_id"] = data["model_id"]
                payload = json.dumps(reply).encode()
                self.send_response(200)
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, *args):
                pass

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever)
        thread.start()
        try:
            url = f"http://127.0.0.1:{server.server_port}"
            gate = from_config(
                {**config(), "primary_url": url, "fallback_url": url, "jepa_url": url}
            )
            with tempfile.TemporaryDirectory() as directory:
                self.assertEqual(
                    replay(gate, observation(), directory)["selected_model"],
                    "alternative",
                )
        finally:
            server.shutdown()
            thread.join()
            server.server_close()


if __name__ == "__main__":
    unittest.main()
