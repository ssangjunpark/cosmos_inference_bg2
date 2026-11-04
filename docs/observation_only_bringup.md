# Three-machine observation-only bringup

This pass starts the model service, SSH connection, ROBOTIS follower and sensor
inspection, with no Cosmos/teleoperation trajectory commands. FFW initialization is
disabled, and our FFW companion is not started. The standard follower still activates
its hardware/controllers; this is not a guarantee of zero motor writes, zero torque
or zero movement. An operator should be at the robot during hardware activation.

These are instructions for manual execution; they were not run during preparation.
Do not launch a second follower if one is already running.

## PRO 6000 — terminal 1

Check GPU availability with `nvidia-smi`; GPU 5 below is an example. From the repository:

```bash
cd /home/sangjunpark/Documents/cosmos_inference_bg2
python3 scripts/cosmos3.py serve \
  --variant edge --step 20000 --gpu 5 --image base \
  --tokenizer assets/nvidia--Cosmos3-Edge-344d602b128d1bbdacb43b08d0a3626f46343e29 \
  --vae assets/Wan-AI--Wan2.2-TI2V-5B-921dbaf3f1674a56f47e83fb80a34bac8a8f203e/Wan2.2_VAE.pth \
  --warmup-observation inputs/recorded_observation.npz \
  --warmup-runs 3 --guidance-interval 960 1001
```

Wait for the ready message. Warmup uses the local recorded observation only; it sends
no data or commands to the robot.

## PRO 6000 — terminal 2

```bash
ssh -N -T -o ExitOnForwardFailure=yes \
  -o ServerAliveInterval=15 -o ServerAliveCountMax=3 \
  -R 127.0.0.1:8008:127.0.0.1:8008 3sp5090
```

Keep it running. No output is expected on a successful tunnel connection.

## FFW — terminal 1

From the 5090, `ssh ffw` connects to the robot. On FFW, enter the existing container:

```bash
docker exec -it ai_worker bash
```

Inside it:

```bash
ros2 launch ffw_bringup ffw_bg2_follower_ai.launch.py \
  launch_cameras:=true init_position:=false \
  init_position_file:=bg2_vr_startup_snupi.yaml \
  use_head_eef_tracker:=false
```

`init_position:=false` skips the homing executors. The SNUPI file remains selected,
but its pose is not commanded in this pass. Do not start a leader, VR controller,
other policy runtime, or our companion. The existing Zenoh router should stay running.
The installed launch schedules head-camera startup after 10 seconds without homing,
and wrist-camera startup about another 10 seconds later; driver initialization may
add time. Wait for actual frames.

ROBOTIS documents the follower-only launch and initialization-disable option in its
[teleoperation guide](https://ai.robotis.com/ai_worker/operation_teleoperation_ai_worker).

## 5090 — inspect and check connectivity

From a terminal on the 5090:

```bash
cd /home/sangjunpark/cosmos_bg2_gateway
python3 -c 'import urllib.request; print(urllib.request.urlopen("http://127.0.0.1:8008/health",timeout=5).read().decode())'
mkdir -p logs
COSMOS_INSPECT_ID=$(TZ=Asia/Seoul date +%Y%m%d-%H%M%S)
docker compose run --rm --no-deps gateway inspect \
  --profile /config/profile.json \
  --messages-dir /opt/cosmos_gateway/messages \
  --output "/output/inspect-${COSMOS_INSPECT_ID}"
```

Health should report `ok: true` and `warmed: true`. Inspect reports joint timestamps
and each camera's raw dimensions and receipt age. It creates no chunk, stop, or
JointTrajectory publishers and makes no model prediction requests. The model server
is loaded and its connection is verified, but live camera inference is not running.
Unknown camera rotations do not block inspect; they must be verified before preview.

## FFW — terminal 2, optional joint-position display

In a second terminal on FFW:

```bash
docker exec -it ai_worker bash
```

Inside it:

```bash
ros2 topic echo /joint_states --once
```

This reads actual joint names, positions and velocities. Inspect's summary displays
timestamps rather than the complete position vector.

## Ending this pass

Use Ctrl+C in the inspect, tunnel and model-service terminals when finished. The
follower is a separate foreground process managed through the normal ROBOTIS
procedure; stopping our observation tools does not stop that robot stack.

Later policy execution is a different session: start the companion before fresh
homing, resolve/verify the SNUPI head-limit mismatch, verify camera transforms and
controller ownership, then test preview before enabling execution.
