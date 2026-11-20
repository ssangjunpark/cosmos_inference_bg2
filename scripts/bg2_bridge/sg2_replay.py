"""Local replay and self-contained dashboard. No ROS imports or robot transport."""

import argparse
import base64
import json
import uuid
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image

from .jepa_gate import JepaGate, from_config, launch_models


def animation(frames):
    frames = JepaGate.frames(frames)
    images = [Image.fromarray(frame).resize((256, 192)) for frame in frames]
    stream = BytesIO()
    images[0].save(
        stream,
        format="GIF",
        save_all=True,
        append_images=images[1:],
        duration=100,
        loop=0,
    )
    return "data:image/gif;base64," + base64.b64encode(stream.getvalue()).decode()


def replay(gate, observation, output):
    output = Path(output).expanduser()
    output.mkdir(parents=True, exist_ok=True)
    request_id, epoch = uuid.uuid4().hex, uuid.uuid4().hex
    result = None
    try:
        result = gate.predict(observation, epoch, request_id)
    except Exception:
        if gate.last_event is None:
            raise
    report = {
        **gate.last_event,
        "execution_state": "recorded_only",
        "observed_video": animation(observation["past"]),
    }
    if result is not None:
        report["predicted_video"] = animation(result["future"])
        saved = {
            key: value.tolist() if isinstance(value, np.ndarray) else value
            for key, value in result.items()
        }
        (output / "selection.json").write_text(
            json.dumps(saved, allow_nan=False, indent=2)
        )
    elif (output / "selection.json").exists():
        (
            output / "selection.json"
        ).unlink()  # Never leave a prior action as the current selection.
    (output / "report.json").write_text(json.dumps(report, allow_nan=False, indent=2))
    data = json.dumps(report, allow_nan=False).replace("<", "\\u003c")
    html = """<!doctype html><meta charset="utf-8"><title>SG2 Cosmos · JEPA replay</title>
<style>body{background:#111827;color:#eee;font:17px system-ui;max-width:1000px;margin:40px auto}section{display:flex;gap:24px}img{width:100%;image-rendering:auto}figure{flex:1;margin:0}pre{white-space:pre-wrap;background:#1f2937;padding:20px;border-radius:12px}</style>
<h1>SG2 Cosmos · JEPA</h1><p>녹화 입력 · 로봇 명령 전송 없음</p>
<section><figure><h2>실제 관측</h2><img id="past"></figure><figure><h2>선택 모델 예측</h2><img id="future"></figure></section><pre id="status"></pre>
<script>const report=REPORT;
document.getElementById('past').src=report.observed_video;
if(report.predicted_video)document.getElementById('future').src=report.predicted_video;
const {observed_video,predicted_video,...status}=report;
document.getElementById('status').textContent=JSON.stringify(status,null,2);</script>""".replace(
        "REPORT", data
    )
    (output / "index.html").write_text(html)
    return report


class FixturePolicy:
    def __init__(self, file, model_id):
        self.file, self.model_id = Path(file).expanduser(), model_id

    def predict(self, observation, epoch, request_id):
        result = json.loads(self.file.read_text())
        return {
            **result,
            "model_id": self.model_id,
            "epoch": epoch,
            "request_id": request_id,
            "source_time_ns": int(observation["source_time_ns"]),
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config")
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Synthetic offline fixture; no models or robot",
    )
    parser.add_argument(
        "--observation",
        help="NPZ: past THWC RGB, frame_interval_ns, source_time_ns, joint_position, joint_names",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--fixture-primary")
    parser.add_argument("--fixture-fallback")
    parser.add_argument("--fixture-loss", type=float)
    parser.add_argument("--launch-models", action="store_true")
    args = parser.parse_args()
    if args.demo:
        if args.launch_models:
            parser.error("--demo does not launch models")
        cfg = {
            "threshold": 0.5,
            "frame_interval_ns": 50_000_000,
            "joint_names": ["synthetic_joint"],
            "units": ["rad"],
            "primary_id": "fixture-primary",
            "fallback_id": "fixture-alternative",
        }
        frames = np.zeros((4, 48, 64, 3), dtype=np.uint8)
        for i in range(4):
            frames[i, 10:30, 5 + i * 8 : 20 + i * 8, 1] = 255

        class Policy:
            def predict(self, observation, epoch, request_id):
                return {
                    "epoch": epoch,
                    "request_id": request_id,
                    "source_time_ns": 1,
                    "joint_names": cfg["joint_names"],
                    "units": cfg["units"],
                    "positions": [[0.1]],
                    "frame_interval_ns": cfg["frame_interval_ns"],
                    "future": frames[::-1].tolist(),
                }

        class Judge:
            def score(self, *unused):
                return 0.7

        report = replay(
            JepaGate(Policy(), Policy(), Judge(), cfg),
            {
                "past": frames,
                "source_time_ns": 1,
                "frame_interval_ns": cfg["frame_interval_ns"],
            },
            args.output,
        )
        print(str(Path(args.output).expanduser() / "index.html"))
        return
    if not args.config or not args.observation:
        parser.error("--config and --observation required outside --demo")
    config = json.loads(Path(args.config).expanduser().read_text())
    if args.launch_models and args.fixture_primary:
        parser.error("Choose model launch or fixtures")
    processes = launch_models(config) if args.launch_models else []
    try:
        with np.load(
            Path(args.observation).expanduser(), allow_pickle=False
        ) as archive:
            observation = {key: archive[key] for key in archive.files}
        if args.fixture_primary:
            if args.fixture_loss is None or not args.fixture_fallback:
                parser.error("Fixture replay requires both policies and --fixture-loss")

            class Judge:
                def score(self, *unused):
                    return args.fixture_loss

            gate = JepaGate(
                FixturePolicy(args.fixture_primary, config["primary_id"]),
                FixturePolicy(args.fixture_fallback, config["fallback_id"]),
                Judge(),
                config,
            )
        else:
            gate = from_config(config)
        report = replay(gate, observation, args.output)
        print(
            json.dumps(
                {
                    key: value
                    for key, value in report.items()
                    if not key.endswith("_video")
                },
                ensure_ascii=False,
            )
        )
    finally:
        for process in processes:
            process.terminate()
        for process in processes:
            process.wait(timeout=10)


if __name__ == "__main__":
    main()
