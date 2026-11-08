# Persistent WAM server on PRO 6000

`serve --mode wam` loads the new WAM checkpoint once and handles the existing gateway's
blocking `/predict` requests. It jointly generates 16 future joint targets and 16 future
RGB frames, decodes and saves the imagined video, then returns the action chunk. The
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

Wait for `"event": "ready"`. The default two warmups use only the recorded observation;
they generate no robot commands. They exercise the complete video-saving path before the
service reports ready and save under `outputs/server-wam-<tier>/warmup/`, separately from
experiment requests. This pays the cold-start cost of video encoding during startup rather
than the first gateway request. Edge defaults
to guidance interval [960, 1001], Nano to all steps. Sampling remains configurable using
`--seed`, `--steps`, `--guidance`, `--shift`, and `--guidance-interval` or
`--guidance-all-steps`. Add `--dry-run` to inspect the Docker command first.

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
fields are metadata; the existing client and companion accept them. Only relative artifact
paths travel in JSON, never video frames or model latents. Files stay on PRO 6000.

Your earlier negative-right-gripper-to-zero rule remains at the service's command boundary.
All seven arm channels pass through unchanged. The WAM runtime and raw prediction files
retain every original value, including negative gripper predictions. No additional scaling,
arm clipping, timing adjustment or stop guard was added. Offline `wam` outputs stay raw.

The task instruction still comes from the gateway CLI's `--instruction` for each request.
Continuous recordings of the real cameras still belong to the existing 5090 recorder.
The authoritative model mode is `/health`, response `mode`, and PRO's `model_output.json`;
older 5090 `run.json` files have legacy hardcoded policy-mode labels.

WAM requests include video decoding and file export before replying. Use the gateway's
existing **`--timeout 120`** flag for WAM debug/execution/replay runs; the default 10 seconds
was exceeded during the local Nano test. This is a client command-line option, with no
gateway source change or FFW bundle update required. It changes how long the client waits,
not the 20 Hz execution timing. Encoding uses bounded CPU threads, but export latency can
still vary on a shared machine.

## Saved files

Each successful request writes:

```text
outputs/server-wam-edge/requests/<epoch>/<request_id>/
outputs/server-wam-nano/requests/<epoch>/<request_id>/
```

Each directory contains the exact inputs, `prediction.json`, `model_output.json`, normalized
actions, `raw_joint_positions.npy`, and `joint_positions.npy` containing the returned
commands. It also includes the vision latent, lossless decoded RGB array, reconstructed
context/final imagined PNGs, and the same five MP4s as the offline WAM runner:

- `imagined_future.mp4`: 16 future composite frames at 20 fps, 0.8 s.
- `rollout_full.mp4`: current reconstructed frame + 16 future frames, 0.85 s playback.
- `imagined_head.mp4`, `imagined_wrist_left.mp4`, `imagined_wrist_right.mp4`: future view crops.

`model_output.json` records the native prompt, sampling, checkpoint metadata hashes,
conditioning masks, training contract, adapter provenance and any Nano config-class override.
The response's `artifacts.directory` links 5090 prediction records to these PRO files.
A failed generation, decode or save returns an HTTP error rather than a successful chunk.

## Stop on PRO 6000

Use Ctrl+C in the service terminal. From another PRO terminal, stop the matching container:

```bash
docker stop --timeout 20 cosmos-bg2-wam-edge-server
```

or `cosmos-bg2-wam-nano-server` for Nano. The container removes itself and releases its GPU.
For a future hardware run, pause the gateway first using the existing runbook; stopping a
model service does not itself control the robot.

## Local verification, 2026-11-08

All tests stayed on PRO 6000. No SSH tunnel, 5090 process, FFW connection or robot publisher
was started. Idle physical GPU 4 ran the Edge and Nano services sequentially at localhost
port 19018. Two complete recorded-observation requests per model passed through the
existing, unchanged `InferenceClient`, including response identity/timestamp checks,
16 × 8 finite targets and all five MP4s present before each reply. Saved checkpoint hashes
and model-tier provenance were checked against the read-only exports afterward.

Edge round trips were 2.32 s and 2.57 s. Nano round trips were 28.01 s and 17.34 s before the
final export-thread tuning; its 10-second client timed out, while `--timeout 120` passed.
These are shared-machine observations, not isolated throughput benchmarks. Inference and
VAE decoding themselves took approximately 0.25 + 0.41 s for Edge and 0.84 + 0.41 s for Nano
after warmup. Video output is part of the blocking round trip.

The final writer bounds FFmpeg codec/filter threads and its linked math-library pools,
without changing model CPU/CUDA settings or pixels supplied to the encoder. A separate
CUDA-context check rendered a saved native Nano rollout through this final writer in
1.28 s, checking all five MP4s. Both full model RPC checks preceded this final child-process
environment change; the final writer was checked separately with CUDA active.

The final full CPU suite ran 146 tests: 145 passed, with one optional Zenoh SDK test skipped.
New coverage includes WAM health/response metadata, startup recording warmup, exact raw vs
returned gripper values, synchronous acceptance/final hold in the companion's pure executor,
video-save failure responses, the existing client, the offline runner's shared runtime and
the WAM server's read-only launch command. Existing busy-request, dropped-connection and
synchronous gateway tests also pass. Physical WAM execution has not been tested.

Only test containers were stopped afterward; GPU 4 returned to idle. No checkpoint or
third-party source was edited. No new Docker image or package installation was needed.
