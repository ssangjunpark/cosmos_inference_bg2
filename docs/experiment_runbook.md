# Experiment runbook: PRO 6000 → 5090 → FFW

Updated 2026-11-05 KST. Use this as the main operator guide for the installed setup.
All commands below are for you to run manually. Ordinary experiments need no SCP,
Docker build, image download, or package installation.

## Machines and terminals

| Machine | Where you are | Directory | What runs here |
| --- | --- | --- | --- |
| PRO 6000 | `sangjunpark@SNUPI-PRO-6000-Blackwell-01` | `~/Documents/cosmos_inference_bg2` | Edge/Nano model server and SSH tunnel |
| 5090 | `sangjunpark@snupi-5090-1` | `~/cosmos_bg2_gateway` | Gateway, task prompt, camera recording and predicted chunks |
| FFW host | `robotis@ffw-SNPR48A1011` | `~/cosmos_ffw_companion` | Our companion container |
| Inside FFW's existing `ai_worker` | Opened from the FFW host with `docker exec -it ai_worker bash` | Existing ROBOTIS environment | ROBOTIS follower, cameras and joint feedback |

To reach FFW from PRO 6000, open a terminal and run:

```bash
ssh 3sp5090
```

Then, **on 5090**:

```bash
ssh ffw
```

Open separate terminals for the PRO model, PRO tunnel, FFW follower, FFW companion,
and 5090 gateway. A foreground server/launch/Compose command occupies its terminal;
that terminal is expected to stay open. Open another SSH terminal for status or restart
commands. FFW Compose commands run on the **FFW host**, outside `ai_worker`, after
`cd ~/cosmos_ffw_companion`.

## 1. PRO 6000: start one model server

Use one of the two commands below. If the desired server is already warmed and running,
keep it. To change an existing server, follow [Switch models](#switch-models) first.

### Nano on physical GPU 2

**PRO 6000 — model terminal:**

```bash
cd ~/Documents/cosmos_inference_bg2
python3 scripts/cosmos3.py serve \
  --variant nano --step 20000 --gpu 2 --image base \
  --tokenizer assets/nvidia--Cosmos3-Nano-e59a53c25979a090fa8706c9acc0c254a6e89b92 \
  --vae assets/Wan-AI--Wan2.2-TI2V-5B-921dbaf3f1674a56f47e83fb80a34bac8a8f203e/Wan2.2_VAE.pth \
  --warmup-observation inputs/recorded_observation.npz \
  --warmup-runs 3 --steps 4 --port 8008
```

### Edge on physical GPU 5

**PRO 6000 — model terminal, as an alternative to Nano:**

```bash
cd ~/Documents/cosmos_inference_bg2
python3 scripts/cosmos3.py serve \
  --variant edge --step 20000 --gpu 5 --image base \
  --tokenizer assets/nvidia--Cosmos3-Edge-344d602b128d1bbdacb43b08d0a3626f46343e29 \
  --vae assets/Wan-AI--Wan2.2-TI2V-5B-921dbaf3f1674a56f47e83fb80a34bac8a8f203e/Wan2.2_VAE.pth \
  --warmup-observation inputs/recorded_observation.npz \
  --warmup-runs 3 --steps 4 --port 8008 \
  --guidance-interval 960 1001
```

Wait for `"event": "ready"` and keep the model terminal open. In another terminal
**on PRO 6000**, verify:

```bash
curl --fail http://127.0.0.1:8008/health
```

Expect `"ok": true`, `"warmed": true`, `"mode": "policy"`, `"records_outputs": true`,
and `"future_steps": 16`.

`--step 20000` selects the trained `020000` checkpoint. `--steps 4` selects four
denoising iterations. Both models predict **16 future actions**, matching BG2 fine-tuning.
Nano omits the Edge-specific guidance interval above. Changing models does not change
the gateway command, ports, camera transforms or FFW controller setup.

To check which checkpoint is actually mounted, **on PRO 6000**, use the command for
the server you started:

```bash
# Nano:
docker inspect cosmos-bg2-nano-server --format '{{range .Mounts}}{{if eq .Destination "/checkpoint"}}{{.Source}}{{end}}{{end}}'
```

```bash
# Edge:
docker inspect cosmos-bg2-edge-server --format '{{range .Mounts}}{{if eq .Destination "/checkpoint"}}{{.Source}}{{end}}{{end}}'
```

## 2. PRO 6000: keep one SSH tunnel open

First check **on 5090**:

```bash
curl --fail http://127.0.0.1:18008/health
```

If that succeeds through the intended model server, reuse the existing tunnel and skip
starting another. Otherwise, **on PRO 6000 — a separate tunnel terminal**, run:

```bash
ssh -N -T -o ExitOnForwardFailure=yes \
  -o ServerAliveInterval=15 -o ServerAliveCountMax=3 \
  -R 127.0.0.1:18008:127.0.0.1:8008 3sp5090
```

A healthy tunnel normally prints nothing and occupies this terminal. Keep it open.
Repeat the curl check **on 5090** before starting the gateway.

This tunnel must originate on PRO 6000, where the model listens. A tunnel from a laptop
would forward to that laptop's `127.0.0.1:8008` instead. If SSH reports
`remote port forwarding failed`, a listener already occupies 18008. Check the existing
tunnel; do not open duplicates. If it needs replacement, Ctrl+C its original PRO 6000
terminal, then run the tunnel command once.

Ports remain: PRO model `127.0.0.1:8008`, 5090 forwarded endpoint
`127.0.0.1:18008`, and the existing FFW Zenoh router `192.168.0.14:7447` (ROS domain 30).

## 3. FFW: start ROBOTIS and move to SNUPI, when wanted

If the follower, three cameras and joint feedback are already running and you want to
continue from the current pose, **skip this section**. Keep the follower running.

This launch **moves the robot to the SNUPI default pose**. For an intentional new home
move, first stop the 5090 gateway with Ctrl+C. If our companion is running, stop only
it **on the FFW host**:

```bash
cd ~/cosmos_ffw_companion
docker compose down --timeout 3
```

If an old follower launch is running, Ctrl+C that launch in its original FFW terminal
before launching another. Do not run two follower launches together.

**FFW host — follower terminal:**

```bash
docker exec -it ai_worker bash
```

Now **inside `ai_worker`**, run:

```bash
ros2 launch ffw_bringup ffw_bg2_follower_ai.launch.py \
  launch_cameras:=true \
  init_position:=true \
  init_position_file:=bg2_vr_startup_snupi.yaml \
  use_head_eef_tracker:=false
```

Wait for homing to finish and all three camera streams/joint feedback to be available.
Leave this follower terminal open. `init_position:=true` is what requests the home move;
`init_position:=false` does not move to the file's pose.

The companion is started **after** this move. Its current-pose mode then captures the
measured left arm/gripper, head and elevator positions to hold. The policy predicts only
the right arm and right gripper. Do not intentionally home while the companion is active.

## 4. FFW host: start our companion for robot execution

Open another terminal through **5090 → `ssh ffw`**. Stay on the **FFW host**, outside
`ai_worker`, and run:

```bash
cd ~/cosmos_ffw_companion
FFW_PROFILE=./config/profile.execute.json \
COSMOS_MODE=execute COSMOS_INITIALIZATION=current \
  docker compose up --no-build --pull never
```

Keep this terminal open. `Attaching to cosmos-ffw-companion` with no further log lines
can be normal; readiness is reported on the status topic below. `current` captures
fresh, settled measured feedback; it does not trigger or require another home move.
The companion may publish hold trajectories in this mode before the gateway starts.

For **prediction-only `--debug`**, this companion is not required. You can skip this
section; debug needs the follower's live sensors, not a new companion process.

### Check companion status in another FFW terminal

**FFW host:**

```bash
docker exec -it ai_worker bash
```

Then **inside that `ai_worker` shell**:

```bash
source /opt/ros/jazzy/setup.bash
export ROS_DOMAIN_ID=30
export RMW_IMPLEMENTATION=rmw_zenoh_cpp
export ZENOH_CONFIG_OVERRIDE='mode="client";connect/endpoints=["tcp/127.0.0.1:7447"];scouting/multicast/enabled=false;transport/shared_memory/enabled=false'
ros2 topic echo /cosmos/bg2/status --field data --once
```

Expect `"mode": "execute"`, `"initialization": "current"`,
`"controllers_ready": true`, and phase `ready`, `holding` or `paused`.
A fresh gateway resumes `paused`. If phase is `fault` or `stopped`, use
[Recover a companion fault](#recover-a-companion-fault) before starting execution.

`exit` leaves this extra status shell and returns to the FFW host. Leave the separate
follower shell running.

## 5. 5090: start the experiment last

Run only one gateway at a time. Choose **execution** or **debug** below. The instruction
is supplied here, on 5090, for every run. The server on PRO 6000 selects Edge versus Nano.

### Execute: the robot moves

**5090 — gateway terminal:**

```bash
cd ~/cosmos_bg2_gateway
curl --fail http://127.0.0.1:18008/health
CAR_NAME=police
TASK_INSTRUCTION="Pick up the police car and place it on the top-right picture"
RUN_ID="execute-$(date -u +%Y%m%dT%H%M%SZ)_${CAR_NAME}"
COSMOS_PROFILE=./config/profile.execute.json \
  docker compose run --rm --no-deps --name cosmos-bg2-execute gateway \
  execute \
  --profile /config/profile.json \
  --messages-dir /opt/cosmos_gateway/messages \
  --server-url http://127.0.0.1:18008 \
  --instruction "$TASK_INSTRUCTION" \
  --output "/output/$RUN_ID"
```

Change `CAR_NAME` for the recording folder suffix and `TASK_INSTRUCTION` for the actual
policy prompt. The folder name alone does not change the prompt. Generate a new `RUN_ID`
each time; an existing output directory is never overwritten. With this example,
files appear under `~/cosmos_bg2_gateway/logs/execute-..._police/` on 5090. Run-name
timestamps use UTC (`date -u`).

Normal execution is synchronous: **hold → fresh measured state and three camera inputs
→ prediction → all 16 targets over 0.8 seconds → hold the last command → next request**.
There is no automatic semantic task-completion detector; you stop the run yourself.

### Debug: predictions and recordings, no policy movement

**5090 — use this instead of the execution command:**

```bash
cd ~/cosmos_bg2_gateway
curl --fail http://127.0.0.1:18008/health
CAR_NAME=police
TASK_INSTRUCTION="Pick up the police car and place it on the top-right picture"
RUN_ID="debug-$(date -u +%Y%m%dT%H%M%SZ)_${CAR_NAME}"
COSMOS_PROFILE=./config/profile.local.json \
  docker compose run --rm --no-deps --name cosmos-bg2-debug gateway \
  execute --debug \
  --profile /config/profile.json \
  --messages-dir /opt/cosmos_gateway/messages \
  --server-url http://127.0.0.1:18008 \
  --instruction "$TASK_INSTRUCTION" \
  --output "/output/$RUN_ID"
```

Debug does not create chunk/control publishers or send policy commands to FFW. It also
does not home the robot. Any separately running ROBOTIS/companion process retains its
own behavior. To stop debug after one prediction, append `--max-predictions 1`.

## Pause, new prompt and resume

1. **5090:** Ctrl+C the gateway in its own terminal. For a normal execute run, wait for
   `pause_confirmed`. This closes recording files and requests a hold at measured positions.
2. Leave the PRO model, SSH tunnel, FFW follower and companion running.
3. Change the prompt/car label, generate a new run ID, then repeat the execute command
   in step 5 on **5090**. The fresh gateway resumes from measured state and new images.

An ordinary pause needs no companion restart, homing, transfer or Docker build.
`control_not_confirmed` means the pause was not acknowledged; inspect FFW status before
continuing. Debug Ctrl+C only closes that debug run and its recordings.

## Switch models

1. **5090:** Ctrl+C execution and wait for `pause_confirmed`.
2. **PRO 6000:** Ctrl+C the old model's original terminal. If that terminal is unavailable,
   use the matching explicit stop command below, only for the server currently running:

```bash
# PRO 6000, stop Nano if switching away from Nano:
docker stop --timeout 20 cosmos-bg2-nano-server
```

```bash
# PRO 6000, stop Edge if switching away from Edge:
docker stop --timeout 20 cosmos-bg2-edge-server
```

3. **PRO 6000:** run the desired model command from step 1 and wait for `"event": "ready"`.
4. **5090:** check `curl --fail http://127.0.0.1:18008/health`, then rerun step 5.

Both models use PRO port 8008. Keep the existing SSH tunnel, follower and healthy
companion running. No FFW restart or home move is needed just to change models.

## Recover a companion fault

Examples include `Stale feedback receipt`, tracking errors, and competing command
publishers. A fault is latched by our Python process even if the original condition
has gone away. The same applies to a terminal `stopped` session.

1. **5090:** Ctrl+C the gateway first. Inspect the reported reason; stop a competing
   control client if applicable, or restore feedback/controller health.
2. **FFW host — another SSH terminal, outside `ai_worker`:**

```bash
cd ~/cosmos_ffw_companion
docker compose restart companion
```

3. Check status using step 4. Wait for a ready/holding current-pose session.
4. **5090:** run a new gateway experiment with a new output directory.

Keep the ROBOTIS follower running. This restart clears the companion's process state,
reloads updated Python modules and captures the current measured pose. It does not
request homing. If the condition remains, it can fault again.

`docker compose up` creates/starts a container as needed; with an unchanged already-running
container it attaches to logs and keeps the existing Python state. `restart companion`
stops/starts the existing container. It preserves its configured execute/current mode;
to change those settings, use the full environment-prefixed `up` command in step 4.
If the container has been removed by `down`, use step 4 to create it again.

The installed bridge uses a **9999 rad/s measured angular-velocity threshold**, effectively
disabling that stop for practical motion. The **10-degree tracking stop** and **100 ms
joint-feedback freshness check** remain. Command-step speed rejection and bridge position
bounds are removed. Negative predicted right-gripper positions are floored to zero after
model decoding. Native ROBOTIS controller/driver configuration is unchanged. The full
remaining checks are in [the guard audit](runtime_guards_20261105.md).

## Finish the session: software cleanup, in this order

### 1. 5090: stop inference and finish recordings

Ctrl+C the gateway in its original terminal. For normal execution, wait for
`pause_confirmed`. Do this before stopping the model, tunnel or FFW companion.
Camera videos are finalized during gateway cleanup.

If the original gateway terminal is unavailable, **on 5090**, stop only the named
container for the mode that is actually running:

```bash
# Normal execution:
docker stop --timeout 15 cosmos-bg2-execute
```

```bash
# Debug, if that is the active run instead:
docker stop --timeout 15 cosmos-bg2-debug
```

Stopping by container name sends SIGTERM so the gateway can attempt the same cleanup;
check its last logs and FFW status if you could not see the acknowledgement.
A name-not-found message means that named container is already absent; inspect
`docker ps --format '{{.Names}}'` if it was started under a different name.
Use the explicit container name above for these one-off `docker compose run` gateways.

### 2. FFW host: stop only our companion

In a separate **FFW host** terminal, outside `ai_worker`:

```bash
cd ~/cosmos_ffw_companion
docker compose down --timeout 3
```

This removes `cosmos-ffw-companion`. It leaves the existing `ai_worker`, ROBOTIS follower,
image, code, configuration and saved recordings in place. Do not delete volumes/images
or run Docker prune as part of experiment cleanup.

### 3. PRO 6000: stop the model server

Ctrl+C the model server's original terminal. Alternatively use the matching explicit
Nano/Edge `docker stop --timeout 20 ...` command in [Switch models](#switch-models).
These model containers use `--rm` and disappear after stopping.

### 4. PRO 6000: close this experiment's SSH tunnel

Ctrl+C the original `ssh -N -T ... -R ...` terminal on PRO 6000. This closes its forwarded
5090 port 18008. Leave unrelated SSH sessions and services alone.

### 5. FFW: optionally stop the ROBOTIS follower

If the robot/sensors should remain available for other work, leave the follower running.
If you are finished with that bringup, Ctrl+C its original `ros2 launch` terminal
**inside FFW's `ai_worker`**, then type `exit` to leave that shell. Do not stop or remove
the entire existing `ai_worker` container just to stop Cosmos.

Software cleanup is not robot power-off or torque disable. Use the robot's normal
shutdown procedure when physical shutdown is wanted.

### Optional: hand control back to the prior SmolVLA client

Only after Cosmos execution and its companion are stopped, and only when you want the
previous SmolVLA robot-control client running again, **on 5090**:

```bash
docker exec lerobot_server /command/s6-svc -u /run/service/main-runtime
```

This gives the prior stack control again. That client was the only legacy service stopped
for Cosmos; its model engine was left running. Before using Cosmos again, stop that client
**on 5090** so it does not compete for the same controller topics:

```bash
docker exec lerobot_server /command/s6-svc -d /run/service/main-runtime
```

## Where recordings are saved

**5090:** `~/cosmos_bg2_gateway/logs/<RUN_ID>/`

- `cameras/head.mkv`, `wrist.mkv`, `wrist_left.mkv`: continuous camera videos.
- `cameras/*.timestamps.jsonl`: original source and receipt timestamps.
- `predictions/<request_id>/`: raw/input images, observation NPZ/JSON and predicted 16×8 targets.
- `events.jsonl`, `run.json`, `recording_summary.json`: run details and recording results.

MKV files are videos. MP4 copies and tiled comparison videos are separate exports;
they are not generated automatically by every gateway run. Keep original recordings.
Rename a run folder only after the gateway has stopped and finalized it.

**PRO 6000:** `~/Documents/cosmos_inference_bg2/outputs/server-nano/requests/` or
`outputs/server-edge/requests/`, organized by epoch and request ID. These contain model
inputs, predictions, normalized actions and conditioned visual latents. Policy mode does
not generate an imagined future RGB rollout. Generated outputs are ignored by Git.

## Quick end-of-session checks

**PRO 6000**, check model containers:

```bash
docker ps --format '{{.Names}}'
```

**5090**, check that your one-off gateway is gone:

```bash
docker ps --format '{{.Names}}'
```

**FFW host**, check our companion:

```bash
cd ~/cosmos_ffw_companion
docker compose ps
```

A full Cosmos cleanup leaves no running `cosmos-bg2-edge-server`,
`cosmos-bg2-nano-server`, `cosmos-bg2-execute`, `cosmos-bg2-debug`, or
`cosmos-ffw-companion`. Existing unrelated containers can remain running.
