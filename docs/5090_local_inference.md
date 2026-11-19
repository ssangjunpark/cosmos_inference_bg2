# Cosmos3 Edge: start, stop, and reset

For both v1/v2 models loaded at once and an independent fifth-terminal camera
recorder, use the [dual-model guide](5090_dual_yellowstick.md). This page describes
running one model server at a time.

Use four terminals on **5090**: model, follower, companion, and gateway.
From another computer, connect each terminal with `ssh 3sp5090`.
The commands below use the installed ROBOTIS environment on FFW, with its
`ai_worker` container and Zenoh router running.

Run commands on the host unless a step explicitly says **inside `ai_worker`**.
Conda `(base)` can remain active. The model launcher starts Docker automatically.

The Git checkout is `~/Documents/cosmos_inference_bg2` on 5090. Code, tokenizer/VAE
assets, recorded inputs, and outputs live there. The original room checkpoint is at
`/mnt/sata1/cosmos3/checkpoints/edge/020000/`. Yellowstick weights stay under
`/mnt/sata1/lg_wm/ckpt/cosmos3_yellowstick_20261114/`; their prepared metadata lives
in this checkout's `.runtime/5090/yellowstick/` directory.

For a fresh restart of an existing session, first follow **Stop and reset** below.

## One-time yellowstick preparation

Run preparation once per variant in a fresh checkout. Existing installations
can reuse their prepared files across restarts and policy switches.

1. Clone with submodules and build the model image as described in the
   [root README](../README.md#build-and-check). For the live gateway, build
   `cosmos-bg2-gateway:local` from the [packaged gateway](../deploy/gateway/README.md).
2. Obtain the Yellowstick checkpoint handoff, including its sidecars and transfer
   manifests. Set `source_root` in both `deploy/5090/yellowstick_v*.json` files
   if its location differs on your machine. Supply the [tokenizer and VAE](../README.md#supporting-assets)
   under `assets/` and a [recorded observation](../README.md#recorded-input-and-output)
   at `inputs/recorded_observation.npz` for model warmup.
3. The supplied profile records the camera calibration and execution settings for
   the documented 5090 / FFW-BG2 installation. For a fresh checkout on that setup,
   initialize the local copy with the commands below. On another installation,
   place its verified execution profile at the same path.

```bash
mkdir -p .runtime/5090
cp -n profiles/5090.execute.json .runtime/5090/profile.execute.json
```

```bash
cd ~/Documents/cosmos_inference_bg2
/usr/bin/python3 -B scripts/prepare_wam_export.py \
  --config deploy/5090/yellowstick_v1.json
/usr/bin/python3 -B scripts/prepare_wam_export.py \
  --config deploy/5090/yellowstick_v2.json
```

Preparation writes small metadata files, the supplied unchanged adapter source,
and a variant-specific gateway profile in this checkout. It checks sidecar SHA256
values and model shard sizes against the supplied transfer receipts. It does not
copy/rewrite weights or start Docker, a model, or a robot process. Repeating it
accepts identical generated files and rejects changes that would overwrite them.

## 1. Start the model — terminal 1, on 5090

Choose one model below. They share port `18008` and the container name
`cosmos-bg2-wam-edge-server`; run one at a time. Use the matching gateway command
in section 4. All commands here are for the operator to run manually.

### Original room/car model

```bash
cd ~/Documents/cosmos_inference_bg2
bash deploy/5090/run_edge.sh --allow-busy-gpu
```

Wait for `"event": "ready"`. Keep this terminal open.

### Yellowstick v1 — final step 18420

```bash
cd ~/Documents/cosmos_inference_bg2
bash deploy/5090/run_edge.sh \
  --checkpoint-root .runtime/5090/yellowstick/v1 \
  --model-dir /mnt/sata1/lg_wm/ckpt/cosmos3_yellowstick_20261114/v1/last_step_018420/model \
  --step 18420 \
  --experiment-contract .runtime/5090/yellowstick/v1/experiment_contract.json \
  --adapter-dir .runtime/5090/yellowstick/v1/adapter \
  --allow-busy-gpu
```

### Yellowstick v2 — final step 18340

```bash
cd ~/Documents/cosmos_inference_bg2
bash deploy/5090/run_edge.sh \
  --checkpoint-root .runtime/5090/yellowstick/v2 \
  --model-dir /mnt/sata1/lg_wm/ckpt/cosmos3_yellowstick_20261114/v2/last_step_018340/model \
  --step 18340 \
  --experiment-contract .runtime/5090/yellowstick/v2/experiment_contract.json \
  --adapter-dir .runtime/5090/yellowstick/v2/adapter \
  --allow-busy-gpu
```

The generated inference config contains only the saved model subtree, excluding
training warm-start, skipped-head and optimizer/trainer settings. The existing native
loader relocates assets and selects single-GPU inference placement.

Both yellowstick models use the **full head image**, letterboxed to 512×384; the
old room model crops off the left 128 pixels first. Wrist rotation/crops stay the
same. v1 and v2 each use their own normalization statistics and supplied WAM
formatter. Both use the instruction **"Move the yellow stick into the basket."**
The existing recorded observation is used for warmup only, not a yellowstick quality
evaluation. Wait for `"event": "ready"` before starting the matching gateway.

To inspect a launch without starting it, append `--dry-run` to the model command.
These commands passed metadata checks and Docker-command dry runs on 2026-11-15;
GPU loading, live inference and robot execution have not been tested for yellowstick.

## 2. Start the follower and cameras — terminal 2

From **5090**, connect to the **FFW host** and enter its existing container:

```bash
ssh ffw
docker exec -it ai_worker bash
```

Now **inside `ai_worker`**, start the follower:

```bash
ros2 launch ffw_bringup ffw_bg2_follower_ai.launch.py \
  launch_cameras:=true \
  init_position:=true \
  init_position_file:=bg2_vr_startup_snupi.yaml \
  use_head_eef_tracker:=false
```

This moves the robot to the SNUPI starting pose and starts the cameras and joint
feedback. Wait for homing to finish and the cameras to start before continuing.
Keep this terminal open. Run only one follower launch, and keep the Cosmos
companion stopped during homing.

## 3. Start the FFW companion — terminal 3

Open another terminal on **5090**:

```bash
ssh ffw
cd ~/cosmos_ffw_companion
FFW_PROFILE=./config/profile.execute.json \
COSMOS_MODE=execute COSMOS_INITIALIZATION=current \
  docker compose up --no-build --pull never
```

These commands run on the **FFW host**, outside `ai_worker`.
The companion captures and holds the current robot pose. Keep this terminal open.

## 4. Start inference and robot execution — terminal 4, on 5090

Start this after the model is ready and the follower and companion are running.

Choose the command that matches the model selected in section 1. The gateway
profile selects camera preprocessing; the running model server selects the weights.

### Original room/car model

```bash
cd ~/Documents/cosmos_inference_bg2
COSMOS_PROFILE=./.runtime/5090/profile.execute.json \
bash deploy/5090/run_gateway.sh \
  execute --profile /config/profile.json \
  --messages-dir /opt/cosmos_gateway/messages \
  --server-url http://127.0.0.1:18008 --timeout 120 \
  --instruction "Pick up the police car and place it on the top-right picture" \
  --output "/output/execute-$(date -u +%Y%m%dT%H%M%SZ)"
```

### Yellowstick v1

```bash
cd ~/Documents/cosmos_inference_bg2
COSMOS_PROFILE=./profiles/yellowstick_v1.execute.json \
bash deploy/5090/run_gateway.sh \
  execute --profile /config/profile.json \
  --messages-dir /opt/cosmos_gateway/messages \
  --server-url http://127.0.0.1:18008 --timeout 120 \
  --instruction "Move the yellow stick into the basket." \
  --output "/output/execute-yellowstick-v1-step18420-$(date -u +%Y%m%dT%H%M%SZ)"
```

### Yellowstick v2

```bash
cd ~/Documents/cosmos_inference_bg2
COSMOS_PROFILE=./profiles/yellowstick_v2.execute.json \
bash deploy/5090/run_gateway.sh \
  execute --profile /config/profile.json \
  --messages-dir /opt/cosmos_gateway/messages \
  --server-url http://127.0.0.1:18008 --timeout 120 \
  --instruction "Move the yellow stick into the basket." \
  --output "/output/execute-yellowstick-v2-step18340-$(date -u +%Y%m%dT%H%M%SZ)"
```

This starts robot execution. Change `--instruction` to your task.
Recordings are saved under `~/Documents/cosmos_inference_bg2/outputs/gateway/`
on 5090. Earlier runs from the SATA folder have been moved here too, including
their MP4 exports. Raw model outputs are in `outputs/server-wam-edge/`.

## Pause or change the task

1. Press **Ctrl+C** in the gateway terminal (terminal 4).
2. Wait for `pause_confirmed` and recording to finish.
3. Repeat step 4 with the new instruction.

Keep the model, follower, and companion running during an ordinary task change.

To **switch v1/v2 or return to the room model**, pause the gateway first and wait
for `pause_confirmed` and recording to finish. Ctrl+C the old model in terminal 1,
then start the desired model from section 1 and wait for `"event": "ready"`.
Run its matching section 4 command. A healthy follower and companion can keep
running; a model switch does not require homing again. Changing only the gateway
profile does not switch checkpoints.

## Stop and reset

1. **Gateway, terminal 4:** press **Ctrl+C**, wait for `pause_confirmed`, and let
   recording finish.
2. **Companion:** open another terminal on 5090 and run:

   ```bash
   ssh ffw
   cd ~/cosmos_ffw_companion
   docker compose down --timeout 3
   ```

3. **Follower, terminal 2:** press **Ctrl+C** in the original follower launch
   terminal inside `ai_worker`. Leave the `ai_worker` container running.
4. **Model, terminal 1:** press **Ctrl+C**. Alternatively, on 5090:

   ```bash
   docker stop --timeout 20 cosmos-bg2-wam-edge-server
   ```

To start again from the SNUPI starting pose, repeat steps **1 → 2 → 3 → 4**.
