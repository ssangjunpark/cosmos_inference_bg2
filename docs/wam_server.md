# Persistent WAM server on PRO 6000

`serve --mode wam` loads the new WAM checkpoint once and handles the existing gateway's
blocking `/predict` requests. It jointly generates 16 future joint targets and 16 future
RGB frames, saves a raw snapshot, then returns the action chunk. A recording-only collector
on 5090 receives timestamped chunks and renders videos after the run stops. The
offline `wam` command and persistent server share `NativeWAM` in `scripts/wam_runtime.py`.

The gateway/FFW execution contract stays **execute all 16 points → hold the final command
→ capture a fresh observation → request the next chunk**. The model service does not
prefetch or initiate requests. Its single GPU worker keeps CUDA on one thread; the caller
waits for its result. A second concurrent request receives HTTP 503, with no request queue.

## Start on PRO 6000

Choose a currently free physical GPU with `nvidia-smi`; the examples use GPU 4. These commands
start the model service only. They neither connect to FFW nor start a gateway or SSH tunnel.
Checkpoint files remain mounted read-only, and the Nano config-class correction remains in
memory, as described in the [offline WAM guide](wam_inference.md).

**Edge:**

```bash
cd /home/sangjunpark/Documents/cosmos_inference_bg2
python3 scripts/cosmos3.py serve \
  --mode wam --variant edge --step 20000 --gpu 4 \
  --warmup-observation inputs/recorded_observation.npz \
  --tokenizer assets/nvidia--Cosmos3-Edge-344d602b128d1bbdacb43b08d0a3626f46343e29 \
  --vae assets/Wan-AI--Wan2.2-TI2V-5B-921dbaf3f1674a56f47e83fb80a34bac8a8f203e/Wan2.2_VAE.pth
```

**Nano, as an alternative:**

```bash
cd /home/sangjunpark/Documents/cosmos_inference_bg2
python3 scripts/cosmos3.py serve \
  --mode wam --variant nano --step 20000 --gpu 4 \
  --warmup-observation inputs/recorded_observation.npz \
  --tokenizer assets/nvidia--Cosmos3-Nano-e59a53c25979a090fa8706c9acc0c254a6e89b92 \
  --vae assets/Wan-AI--Wan2.2-TI2V-5B-921dbaf3f1674a56f47e83fb80a34bac8a8f203e/Wan2.2_VAE.pth
```

Wait for `"event": "ready"`. The two warmups use only the recorded observation and generate
no robot commands. Startup no longer encodes any MP4s. Edge defaults to guidance interval
[960, 1001], Nano to all steps. Sampling remains configurable with `--seed`, `--steps`,
`--guidance`, `--shift`, and `--guidance-interval` or `--guidance-all-steps`.
Add `--dry-run` to inspect the Docker command first.

The default root is `/mnt/sata1/RobotEscapeCkpt/cosmos3_wam_room20k_20261105` for `--mode wam`.
Omitting `--mode wam` retains the original policy mode and its checkpoint root.

In a second **PRO 6000** terminal:

```bash
curl --fail http://127.0.0.1:8008/health
```

Expect `"mode": "wam"`, `"warmed": true`, `"imagined_future_rollout": true`,
`"records_outputs": true`, `"future_steps": 16`, and `"video_frames": 17`.

## Gateway compatibility

The host endpoint remains `127.0.0.1:8008`; the existing SSH forwarding endpoint on 5090
remains `127.0.0.1:18008`. A local test can use `--port 19018` to avoid the deployment port.
The test must use a localhost client on PRO 6000, without starting the SSH tunnel or gateway.

The existing gateway still sends a current-observation NPZ with epoch/request identifiers.
It receives the same `cosmos_bg2_joint_chunk_v1` response: 16 × 8 absolute joint coordinates,
radians, original observation timestamp, epoch and request ID. Added `mode` and artifact
fields are metadata; the existing action chunk and companion stay compatible. The updated
5090 gateway downloads raw RGB chunks separately through the same SSH tunnel. Predictions
remain blocking and synchronous; recording transfer never requests predictions or robot commands.

Your earlier negative-right-gripper-to-zero rule remains at the service's command boundary.
All seven arm channels pass through unchanged. The WAM runtime and raw prediction files
retain every original value, including negative gripper predictions. No additional scaling,
arm clipping, timing adjustment or stop guard was added. Offline `wam` outputs stay raw.

The task instruction still comes from the gateway CLI's `--instruction` for each request.
Continuous real-camera recordings stay in the 5090 run folder. Imagined recordings now also
live in that folder. See [per-run recordings](run_recordings.md) for the layout, timestamp
meaning, interrupted-transfer recovery and collecting earlier runs.

Keep the existing `--timeout 120` option. MP4 encoding is no longer part of the action reply.
Ctrl+C interrupts an in-flight HTTP wait so the gateway can request the FFW pause promptly.
After `pause_confirmed`, it finishes recording collection and rendering. Ctrl+C again defers
that file-only work; the raw chunks remain recoverable.

## 5090: run normal inference

Run this on **5090**, after the WAM server and tunnel are healthy and the FFW follower
and execution companion are ready. Follow the [experiment runbook](experiment_runbook.md)
for FFW startup and SNUPI initialization. This command enables robot movement.

```bash
cd ~/cosmos_bg2_gateway

COSMOS_PROFILE=./config/profile.execute.json \
  docker compose run --rm --no-deps --name cosmos-bg2-execute gateway \
  execute \
  --profile /config/profile.json \
  --messages-dir /opt/cosmos_gateway/messages \
  --server-url http://127.0.0.1:18008 \
  --timeout 120 \
  --instruction "Pick up the police car and place it on the top-right picture" \
  --output "/output/execute-$(date -u +%Y%m%dT%H%M%SZ)"
```

This is the original execution command with `--timeout 120` added for WAM. Edit the
instruction directly; no `CAR_NAME`, `TASK_INSTRUCTION` or `RUN_ID` assignments are needed.
Each run creates a timestamped output directory. Ctrl+C requests a pause; wait for
`pause_confirmed` before assuming the robot paused.

## Saved files

**5090:** `~/cosmos_bg2_gateway/logs/execute-<timestamp>/imagined/` contains per-chunk raw
frames/timestamps, per-chunk MP4s, and the combined `rollout.mp4`, `head.mp4`,
`wrist_left.mp4` and `wrist_right.mp4`. Construction starts after control has ended.

**PRO 6000:** raw snapshots and model metadata are retained under
`outputs/server-wam-<tier>/runs/<5090-run-name>/requests/<epoch>/<request_id>/`.
They include `decoded_rollout.npy`, `rollout.json`, action arrays, inputs and provenance.
The persistent server never runs FFmpeg. Its recording worker only prepares timestamps
and checksums for transfer. The separate offline `wam` command still renders local MP4s.

`imagined/timeline.json` maps combined-video frame ranges to each original request and
observation timestamp. These are concatenated forecasts; inference/hold gaps are not
continuous physical time. Existing legacy PRO request folders remain available for collection.

## Stop on PRO 6000

Use Ctrl+C in the service terminal. From another PRO terminal, stop the matching container:

```bash
docker stop --timeout 20 cosmos-bg2-wam-edge-server
```

or `cosmos-bg2-wam-nano-server` for Nano. The container removes itself and releases its GPU.
For a future hardware run, pause the gateway first using the existing runbook; stopping a
model service does not itself control the robot.

## Local verification of the recording fix, 2026-11-08

All inference tests stayed on PRO 6000, with no SSH tunnel to the test server and no robot
publisher. An idle physical GPU 1 ran Edge at localhost port 19018. Two sequential recorded
requests through the gateway's replay mode returned in 1.29 s and 1.17 s. The replay collector
received both raw chunks, created their individual MP4s, and assembled four combined videos.
Each combined video was independently decoded: 32 frames, 20 fps, with expected dimensions.
The temporary test server was stopped afterward.

The full CPU suite passed 153 tests (one optional SDK test skipped). Coverage includes a
blocked recording worker with continued action delivery, immediate cancellation of a 120 s
HTTP wait, save failure isolation, checksum-verified downloads, request ordering, restartable
collection, legacy timed-out request recovery and the existing synchronous companion logic.
Physical WAM execution was not run by these tests. No checkpoint files or FFW code changed.
