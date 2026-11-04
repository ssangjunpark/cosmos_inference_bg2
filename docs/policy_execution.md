# Normal policy inference with robot movement

Prepared 2026-11-05 KST. The operator authorized normal execution and an upper-limit
override of head_joint1 from 0.6951 to 0.78 radians for the existing SNUPI home.
Only the separate `profile.execute.json` contains this override. The ROBOTIS model,
SNUPI YAML, reference profile and debug profile are unchanged. The execution profile
is marked `operator_enabled_for_execution`, not physically commissioned. The policy
predicts only right arm/gripper; the companion still publishes measured startup
holds for the left arm/gripper, head and elevator. Keep the independent robot stop
available for this first moving run; physical tracking and stopping remain untested.

Both 5090 and FFW are installed at source revision `940d5ee`, without starting the
gateway or companion. The one-time FFW copy was completed at the user's request on
2026-11-05 KST; see [the installation record](ffw_setup_20261105.md). Skip copying files
for ordinary inference runs. FFW reuses its installed `robotis/ai-worker:2.2.8` image;
no Docker build or download is needed.

## 1. Keep the robot stack running

An already running ROBOTIS follower and its cameras can stay up. Do not run a second
follower or a competing VR/teleoperation publisher. The companion defaults to
`COSMOS_INITIALIZATION=current`: it checks fresh, settled feedback for 500 ms and
captures the measured pose. It does not require another home move, including after
a companion restart. The existing YAML position/velocity tolerances are reused for
settling; current-position bounds, controller health and ownership checks remain.
If you want a specific initial pose, put the robot there using ROBOTIS first.
`COSMOS_INITIALIZATION=homing` restores the optional fresh-homing requirement.

## 2. PRO 6000: restart the model to load the gripper floor

```bash
cd /home/sangjunpark/Documents/cosmos_inference_bg2
docker stop --timeout 20 cosmos-bg2-edge-server
python3 scripts/cosmos3.py serve \
  --variant edge --step 20000 --gpu 5 --image base \
  --tokenizer assets/nvidia--Cosmos3-Edge-344d602b128d1bbdacb43b08d0a3626f46343e29 \
  --vae assets/Wan-AI--Wan2.2-TI2V-5B-921dbaf3f1674a56f47e83fb80a34bac8a8f203e/Wan2.2_VAE.pth \
  --warmup-observation inputs/recorded_observation.npz \
  --warmup-runs 3 --guidance-interval 960 1001
```

Wait for `event: ready`; keep this terminal open. The server uses policy mode,
no imagined future video, and floors only negative predicted right-gripper values to 0.

## 3. PRO 6000: open the tunnel in another terminal

The last check found the 5090 port closed. Open one tunnel, originating on PRO 6000:

```bash
ssh -N -T -o ExitOnForwardFailure=yes \
  -o ServerAliveInterval=15 -o ServerAliveCountMax=3 \
  -R 127.0.0.1:18008:127.0.0.1:8008 3sp5090
```

Keep it open. If a working tunnel already occupies 18008, reuse it instead of starting
another. Verify on 5090 with `curl --fail http://127.0.0.1:18008/health`.

## 4. FFW: installation complete; check and start

The copy is already complete. From a terminal on 5090:

```bash
ssh ffw
cd ~/cosmos_ffw_companion
python3 verify_bundle.py
```

The following commands are on FFW. Validate files without opening ROS or publishing:

```bash
FFW_PROFILE=./config/profile.execute.json COSMOS_MODE=execute \
  docker compose run --rm --no-deps companion \
  --profile /config/profile.json \
  --initial-positions /config/initial_positions.yaml --mode execute --initialization current --check
```

Proceed only when it reports `valid: true`. Start the companion from the current pose:

```bash
FFW_PROFILE=./config/profile.execute.json COSMOS_MODE=execute COSMOS_INITIALIZATION=current \
  docker compose up --no-build --pull never
```

Keep this terminal open between tasks. It waits for fresh settled feedback and healthy
controllers, then holds the measured pose. There is no copy/build/homing step between
prompts. This update uses status schema `cosmos_bg2_status_v3` and execution contract
`synchronous_full_chunk_v2`; update gateway and companion together once.

## 5. FFW: verify the robot and companion are ready

If the follower is already running, leave it running. Otherwise, enter the existing
container and start sensors/controllers without automatic homing:

```bash
docker exec -it ai_worker bash
ros2 launch ffw_bringup ffw_bg2_follower_ai.launch.py \
  launch_cameras:=true \
  init_position:=false \
  init_position_file:=bg2_vr_startup_snupi.yaml \
  use_head_eef_tracker:=false
```

Use `init_position:=true` only when you intentionally want ROBOTIS to move to SNUPI.
Finish any such homing before starting the companion in current-pose mode. The
companion does not trigger homing itself. In another ai_worker shell:

```bash
ros2 topic echo /cosmos/bg2/status --field data
```

Wait for `mode: execute`, `initialization: current`, and phase `ready`, `holding`,
or `paused`. Ready/holding has `needs_chunk: true`; paused intentionally has false.
A new gateway resumes a paused companion before capturing fresh inputs. Ctrl+C stops
only this status display. A fault or competing publisher must be diagnosed, not bypassed.

## 6. 5090: start normal inference last

```bash
cd ~/cosmos_bg2_gateway
curl --fail http://127.0.0.1:18008/health
COSMOS_PROFILE=./config/profile.execute.json \
  docker compose run --rm --no-deps --name cosmos-bg2-execute gateway \
  execute \
  --profile /config/profile.json \
  --messages-dir /opt/cosmos_gateway/messages \
  --server-url http://127.0.0.1:18008 \
  --instruction "Pick up the police car and place it on the top-right picture" \
  --output "/output/execute-$(date -u +%Y%m%dT%H%M%SZ)"
```

There is no `--debug`: this command enables policy-driven movement after FFW readiness.
Each cycle captures current measured state/images, infers while holding, executes all
16 targets over 0.8 seconds, then holds the final command before requesting again.
The loop does not detect semantic task completion; the operator stops it. Camera videos,
exact input frames and chunks are saved under the 5090 `logs/execute-*/` directory;
model artifacts stay on PRO 6000 under `outputs/server-edge/requests/`.

## 7. Pause and resume with a new prompt

Press Ctrl+C in the 5090 inference terminal. It closes recording files and requests a
measured-position pause. Check for `pause_confirmed`; the companion stays running and
publishes `phase: paused`. It invalidates the old prediction/session epoch so an in-flight
old result cannot execute after a pause. The acknowledgement wait is at most one second;
`control_not_confirmed` means the robot state must be checked, not assumed paused.

To continue, repeat step 6 with a new `--instruction` and a new output directory name.
The new gateway requests resume, waits for the FFW hold, then captures current measured
joints and new camera frames. No companion restart, SCP, or homing is needed. No old
chunk tail is replayed. Paused holds remain subject to tracking/velocity/feedback checks.

Faults are terminal and are never automatically resumed. After diagnosing one, restart
the companion in current-pose mode to capture fresh settled feedback; homing is not
mandatory. Unexpected gateway failures request a terminal stop rather than a resumable
pause. Debug still creates no command/control publishers.

To shut down the companion deliberately, on FFW:

```bash
cd ~/cosmos_ffw_companion
docker compose down --timeout 3
```

The separately managed ROBOTIS follower stays running. Software hold is not torque
disable or an independent emergency stop; use the robot's stop for urgent stopping.

The existing 2 rad/s, 10-degree tracking, position, freshness and controller/ownership
checks remain. The head override changes only the companion's accepted head range.

## Recreate the execution profile later

On PRO 6000, use a copy of the verified sensor profile as input:

```bash
python3 scripts/prepare_execution_profile.py \
  --source PATH_TO_VERIFIED_SENSOR_PROFILE.json \
  --output PATH_TO_NEW_EXECUTION_PROFILE.json \
  --snupi-head-override
```

The output is exclusive: existing profiles are not overwritten. This command opens
no robot connection. Disabling/removing this separate profile restores the default
execution-disabled reference/debug behavior; no ROBOTIS model edits need reverting.
