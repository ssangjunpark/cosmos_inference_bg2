"""Unified-config launch plans: host paths, environments, Docker mounts and CLI."""

import copy
import json
import subprocess
import sys
import tempfile
import unittest
from io import BytesIO
from pathlib import Path
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from sg2_jepa_run import build_plan, load_config, ready, supervise


class LauncherTests(unittest.TestCase):
    def fixture(self, directory):
        cfg = json.loads((ROOT / "configs/sg2_jepa.example.json").read_text())
        cfg["model_runtime"]["sampling"].update(steps=4, guidance=3, shift=5)
        cfg["models"][0]["checkpoint"] = "weights/cm1"
        cfg["models"][1]["checkpoint"] = "weights/cm2"
        cfg["jepa"]["checkpoint"] = "weights/jepa.pt"
        cfg["gateway"].update(
            router="test-router",
            domain=0,
            head_topic="/head",
            left_topic="/left",
            right_topic="/right",
        )
        path = Path(directory) / "experiment with spaces.json"
        path.write_text(json.dumps(cfg))
        return path, cfg

    def test_all_three_checkpoint_paths_and_threshold_come_from_same_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path, cfg = self.fixture(directory)
            cfg["threshold"] = 1.25
            path.write_text(json.dumps(cfg))
            loaded = load_config(path)
            self.assertEqual(loaded["threshold"], 1.25)
            self.assertEqual(
                loaded["models"][0]["checkpoint"], str(Path(directory) / "weights/cm1")
            )
            self.assertEqual(
                loaded["models"][1]["checkpoint"], str(Path(directory) / "weights/cm2")
            )
            self.assertEqual(
                loaded["jepa"]["checkpoint"], str(Path(directory) / "weights/jepa.pt")
            )
            self.assertEqual(
                loaded["jepa"]["repo_root"],
                str(Path("~/projects/vjepa2").expanduser().resolve()),
            )

    def test_docker_plan_preserves_absolute_paths_and_does_not_give_gateway_gpu(self):
        with tempfile.TemporaryDirectory() as directory:
            path, _ = self.fixture(directory)
            cfg = load_config(path)
            jobs = build_plan(cfg, "all", Path(directory) / "runtime")
            self.assertEqual(
                [job["role"] for job in jobs], ["jepa", "cosmos", "gateway"]
            )
            jepa, pair, gateway = jobs
            self.assertIn("lgwm.judges.future_service", jepa["argv"])
            self.assertTrue(
                jepa["env"]["PYTHONPATH"].startswith(
                    str(Path(cfg["jepa"]["project_root"]) / "src")
                )
            )
            self.assertIn("--gpus", pair["argv"])
            self.assertNotIn("--gpus", gateway["argv"])
            self.assertIn("jepa-preview", gateway["argv"])
            self.assertIn(f"{directory}/weights:{directory}/weights:ro", pair["argv"])
            self.assertIn(str(Path(directory) / "runtime/gate.json"), pair["argv"])
            self.assertIn(str(Path(directory) / "runtime/gate.json"), gateway["argv"])

    def test_virtualenv_python_symlink_is_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            path, cfg = self.fixture(directory)
            executable = Path(directory) / "venv/bin/python"
            executable.parent.mkdir(parents=True)
            executable.symlink_to(sys.executable)
            cfg["jepa"]["python"] = "venv/bin/python"
            path.write_text(json.dumps(cfg))
            self.assertEqual(load_config(path)["jepa"]["python"], str(executable))

    def test_native_plan_and_actions(self):
        with tempfile.TemporaryDirectory() as directory:
            path, _ = self.fixture(directory)
            cfg = load_config(path)
            cfg["launcher"]["cosmos_backend"] = "native"
            services = build_plan(cfg, "services", Path(directory) / "runtime")
            gateway = build_plan(cfg, "gateway", Path(directory) / "runtime")
            self.assertEqual(len(services), 2)
            self.assertEqual(len(gateway), 1)
            self.assertNotIn("docker", services[1]["argv"])
            cfg["gateway"]["mode"] = "execute"
            self.assertIn(
                "jepa-execute",
                build_plan(cfg, "gateway", Path(directory) / "runtime")[0]["argv"],
            )

    def test_dry_run_script_from_other_directory_starts_nothing(self):
        with tempfile.TemporaryDirectory() as directory:
            path, _ = self.fixture(directory)
            result = subprocess.run(
                [
                    "bash",
                    str(ROOT / "sg2_setup/run_jepa.sh"),
                    str(path),
                    "all",
                    "--dry-run",
                ],
                cwd=directory,
                capture_output=True,
                text=True,
                check=True,
            )
            data = json.loads(result.stdout)
            self.assertEqual(data["threshold"], 2.0)
            self.assertEqual(len(data["jobs"]), 3)
            self.assertIn(
                str(Path(directory) / "weights/jepa.pt"), data["jepa_checkpoint"]
            )
            self.assertFalse(Path(directory, "weights").exists())

    def test_invalid_threshold_port_or_missing_sampling_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path, cfg = self.fixture(directory)
            for section, key, value in (
                (None, "threshold", float("nan")),
                ("jepa", "port", 8090),
                (
                    "model_runtime",
                    "sampling",
                    {"steps": None, "guidance": 3, "shift": 5},
                ),
            ):
                bad = copy.deepcopy(cfg)
                if section is None:
                    bad[key] = value
                else:
                    bad[section][key] = value
                path.write_text(json.dumps(bad))
                with self.assertRaises(ValueError):
                    load_config(path)

    def test_ready_rejects_an_old_checkpoint_on_the_same_model_port(self):
        with tempfile.TemporaryDirectory() as directory:
            path, _ = self.fixture(directory)
            cfg = load_config(path)
            reply = BytesIO(
                json.dumps(
                    {"ready": True, "model_id": "CM1", "checkpoint": "/old/checkpoint"}
                ).encode()
            )
            with (
                patch("sg2_jepa_run.urlopen", return_value=reply),
                self.assertRaisesRegex(ValueError, "different checkpoint"),
            ):
                ready(cfg)

    def test_ready_checks_all_three_checkpoint_identities(self):
        with tempfile.TemporaryDirectory() as directory:
            path, _ = self.fixture(directory)
            cfg = load_config(path)
            replies = [
                BytesIO(
                    json.dumps(
                        {
                            "ready": True,
                            "model_id": entry["model_id"],
                            "checkpoint": entry["checkpoint"],
                        }
                    ).encode()
                )
                for entry in cfg["models"]
            ]
            replies.append(
                BytesIO(
                    json.dumps(
                        {"ready": True, "checkpoint": cfg["jepa"]["checkpoint"]}
                    ).encode()
                )
            )
            with patch("sg2_jepa_run.urlopen", side_effect=replies):
                self.assertTrue(ready(cfg))

    def test_supervisor_keeps_cosmos_alive_when_jepa_exits_after_startup(self):
        import signal

        handlers = {}

        def install(sig, handler):
            handlers[sig] = handler
            return signal.SIG_DFL

        jepa, cosmos = Mock(pid=100), Mock(pid=200)
        jepa.poll.return_value = 1
        cosmos.poll.return_value = None

        def running(_seconds):
            killer.assert_not_called()
            handlers[signal.SIGINT]()

        jobs = [
            {"role": role, "argv": ["test"], "cwd": "/", "env": {}}
            for role in ("jepa", "cosmos")
        ]
        with (
            patch("sg2_jepa_run.signal.signal", side_effect=install),
            patch("sg2_jepa_run.subprocess.Popen", side_effect=[jepa, cosmos]),
            patch("sg2_jepa_run.ready", return_value=True),
            patch("sg2_jepa_run.time.sleep", side_effect=running),
            patch("sg2_jepa_run.os.killpg") as killer,
        ):
            supervise(
                {"launcher": {"startup_timeout_seconds": 1}, "threshold": 2},
                jobs,
                "services",
            )
        killer.assert_called_once_with(200, signal.SIGTERM)

    def test_unconfigured_camera_does_not_prevent_service_launch(self):
        with tempfile.TemporaryDirectory() as directory:
            path, cfg = self.fixture(directory)
            cfg["gateway"]["router"] = None
            path.write_text(json.dumps(cfg))
            loaded = load_config(path)
            self.assertEqual(
                len(build_plan(loaded, "services", Path(directory) / "runtime")), 2
            )
            with self.assertRaises(ValueError):
                build_plan(loaded, "gateway", Path(directory) / "runtime")


if __name__ == "__main__":
    unittest.main()
