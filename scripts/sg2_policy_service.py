#!/usr/bin/env python3
"""One resident Cosmos SG2 checkpoint, with binary action/video RPC on localhost."""

import argparse
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import numpy as np
from bg2_bridge.sg2_dual import MAX_BODY, pack, unpack, video_view
from sg2_policy import Predictor
from sg2_wam import ARM_NAMES, VARIANTS, controller_positions


def make_server(
    address, predictor, model_id, view="head", size=(256, 192), checkpoint=None
):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path != "/health":
                self.send_error(404)
                return
            payload = json.dumps(
                {
                    "ready": True,
                    "checkpoint": str(Path(checkpoint).expanduser().resolve())
                    if checkpoint
                    else None,
                    "model_id": model_id,
                    "variant": predictor.model.runtime_metadata["variant"],
                    "joint_names": ARM_NAMES,
                    "units": ["rad"] * 16,
                    "frame_interval_ns": 50_000_000,
                    "video_view": view,
                    "video_size": list(size),
                }
            ).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def do_POST(self):
            try:
                self.connection.settimeout(30)
                length = int(self.headers.get("Content-Length", "0"))
                if self.path != "/predict" or not 0 < length <= MAX_BODY:
                    raise ValueError("Invalid SG2 prediction request")
                observation = unpack(self.rfile.read(length))
                ids = json.loads(str(observation.pop("metadata")))
                if (
                    ids["model_id"] != model_id
                    or not ids["epoch"]
                    or not ids["request_id"]
                ):
                    raise ValueError("Invalid SG2 request identity")
                result = predictor.infer(observation, decode_video=True)
                positions = controller_positions(
                    result["positions"],
                    observation,
                    predictor.names,
                    predictor.conversion,
                )
                frames = np.asarray(result["imagined_video"])
                if frames.shape != (17, 640, 512, 3) or frames.dtype != np.uint8:
                    raise ValueError(
                        "Expected one context + sixteen generated RGB frames"
                    )
                metadata = dict(
                    schema="cosmos_sg2_joint_chunk_v1",
                    **ids,
                    source_time_ns=int(observation["source_time_ns"]),
                    joint_names=ARM_NAMES,
                    units=["rad"] * 16,
                    positions=positions.tolist(),
                    frame_interval_ns=50_000_000,
                    timings=result.get("timings", {}),
                )
                payload = pack(
                    {
                        "metadata": np.asarray(json.dumps(metadata, allow_nan=False)),
                        "future": video_view(frames[1:], view, size),
                    }
                )
                self.send_response(200)
                self.send_header("Content-Type", "application/x-npz")
            except Exception as error:  # noqa: BLE001 -- turn model/RPC errors into a failed request
                payload = json.dumps({"error": str(error)}).encode()
                self.send_response(400)
                self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            try:
                self.wfile.write(payload)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def log_message(self, *args):
            pass

    return HTTPServer(address, Handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    for name in ("checkpoint", "tokenizer", "vae", "calibration"):
        parser.add_argument(
            "--" + name, type=lambda p: Path(p).expanduser(), required=True
        )
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--listen-port", type=int, required=True)
    parser.add_argument(
        "--video-view", choices=("head", "wrist_left", "wrist"), default="head"
    )
    parser.add_argument(
        "--video-size",
        nargs=2,
        type=int,
        default=[256, 192],
        metavar=("WIDTH", "HEIGHT"),
    )
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--guidance", type=float, required=True)
    parser.add_argument("--shift", type=float, required=True)
    parser.add_argument("--guidance-interval", nargs=2, type=float)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    predictor = Predictor(args)
    server = make_server(
        ("127.0.0.1", args.listen_port),
        predictor,
        args.model_id,
        args.video_view,
        args.video_size,
        args.checkpoint,
    )
    print(
        json.dumps(
            {"ready": True, "model_id": args.model_id, "port": args.listen_port}
        ),
        flush=True,
    )
    try:
        server.serve_forever()
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
