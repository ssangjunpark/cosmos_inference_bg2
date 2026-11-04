# Policy debug without robot movement

The BG2 request uses `mode="policy"`, matching the supplied training loader and
both Edge/Nano checkpoints. It predicts 16 absolute right-arm/gripper targets at
50 ms intervals. No imagined future video or extra VAE decoding is requested.
The DROID WAM example is not used for BG2 inference.

The operator first brings up the robot, puts it in the intended initial pose, and
checks all three cameras and measured joints. Then start debug manually. Debug does
not initialize the robot, verify its pose against a home YAML, or require our FFW
companion. It subscribes only to live sensors, creates no chunk/control publishers,
and never sends hardware commands. It retains the normal freshness/alignment and
verified-camera-transform requirements. `execution_enabled` can remain false.

## PRO 6000

Start the model and tunnel using the commands in
[the three-machine guide](three_machine_pipeline.md#start-the-model-server-on-pro-6000).
The model service must be restarted after updating the Python code; verify:

```bash
curl --fail http://127.0.0.1:8008/health
```

The reply must include `"mode": "policy"` and `"records_outputs": true`.
The service is passive until a gateway request arrives.

## 5090

Use the newly packaged gateway image, after building it from the updated bundle.
The installed `config/profile.local.json` on 5090 was verified against live frames
and the supplied training sample on 2026-11-05 KST: head raw size 672×376, rotation
0°; both wrist raw sizes 424×240, PIL rotation 90° counter-clockwise before cropping.
Its debug startup check passes, while `execution_enabled` remains false. Unknown
camera settings still block startup in other profiles; the default/reference profile
remains unverified.

```bash
cd ~/cosmos_bg2_gateway
curl --fail http://127.0.0.1:18008/health
RUN_ID="debug-$(date -u +%Y%m%dT%H%M%SZ)"
docker compose run --rm --no-deps --name cosmos-bg2-debug gateway \
  execute --debug \
  --profile /config/profile.json \
  --messages-dir /opt/cosmos_gateway/messages \
  --server-url http://127.0.0.1:18008 \
  --instruction "Pick up the police car and place it on the top-right picture" \
  --output "/output/$RUN_ID"
```

The prompt is a CLI argument for each run. For a short test, append
`--max-predictions 1`. Ctrl+C ends debug and closes the camera files. There is no
automatic task-completion detector. Keep FFW untouched by our deployment for this
prediction-only pass. Do not start a second gateway concurrently.

Normal inference uses the same command without `--debug`, but requires the matching
FFW companion, a hardware-verified enabled profile, and fresh external homing for the
companion session. Normal execution follows the synchronous full-chunk contract;
debug does not simulate robot execution or claim to validate physical tracking.

## Saved files

On the 5090, `~/cosmos_bg2_gateway/logs/<run>/` contains:

- `cameras/{head,wrist_left,wrist}.mkv`: continuous original JPEG/PNG packets as received.
- Matching `.timestamps.jsonl`: exact source and receipt nanoseconds per received frame.
- `predictions/<request_id>/`: the exact raw selected images, processed RGB PNGs,
  combined policy input image, observation NPZ/JSON, and prediction JSON with all 16 targets.
- `events.jsonl`, `run.json`, `recording_summary.json`: run mode, progress and recording results.

The Matroska player time base may round timestamps; sidecars retain their exact values.
No fixed frame rate or missing frames are invented. Recording starts with the debug
process, not with model warmup. If a stream stops, new predictions wait for fresh
aligned data. A recording failure or inference connection error exits debug and closes
its files. Ctrl+C cannot cancel an already submitted GPU request, but no action is sent.

On PRO 6000, `outputs/server-edge/requests/<epoch>/<request_id>/` contains model inputs,
absolute/normalized actions, raw conditioned visual latents, sampling settings and prompt.
Each 5090 prediction links this directory in `artifacts`. Outputs and generated bundles
are ignored by Git. This policy run does not produce imagined future RGB videos.

## Verified camera orientation evidence

The original recorder on 5090 is in `~/dev/cyclo_intelligence`.
Its `shared/shared/robot_configs/ffw_bg2_rev4_config.yaml` specifies wrist rotation 270;
`cyclo_data/cyclo_data/recorder/transcoder.py::_transpose_filter` maps that value to
FFmpeg `transpose=2` (counter-clockwise). The rotation is baked into recorded MP4s.
Thus the corresponding PIL live preprocessing rotation is **90 degrees**, applied
before the training crop. Head rotation is 0.

A subscriber-only live capture confirmed the dimensions above and the processed
three-view layout against `data/example/future_joint/cache/3_0` in the supplied training
bundle. Capture evidence and exact inputs are on 5090 under
`logs/camera-check-20261104T151855Z/`, with a copy under the same folder in PRO 6000
`outputs/`. Selected camera/joint offsets were within 22.28 ms. No policy request or
command publisher was used. The old profile was saved as
`config/profile.local.before-camera-verification-20261105.json`.
The original recorder configuration SHA-256 remains
`ac30eb71a3910c9a8fe3d6cb1216e498141672038d70ae0b7665d4a4dbcbc74b`.
Hardware execution status and joint limits were left unchanged; this camera check
establishes no physical tracking or motor-stop validation.
