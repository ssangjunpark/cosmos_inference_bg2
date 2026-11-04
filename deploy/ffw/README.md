# Cosmos FFW companion bundle

Prepare this folder on the PRO 6000 server; transfer it to FFW only later. It adds a
small companion alongside the existing ROBOTIS bringup. It contains no model weights,
IK implementation, motor driver, or replacement for the existing 100 Hz controllers.

## Prepare on PRO 6000

From the cosmos_inference_bg2 repository:

```bash
python3 scripts/package_ffw.py --output transfer/ffw
```

Send the generated `transfer/ffw/` folder when deployment is requested.
`deploy/ffw/` is the source template; the packager adds shared core modules, a
reference robot profile, and a SHA-256 manifest. Use a new output directory for later
versions. The packager never connects to either remote machine.

## What runs where

| Machine | Role |
| --- | --- |
| PRO 6000 | Persistent Cosmos model worker. Return eight absolute joint positions for 16 future steps. |
| 5090 | Align three camera views and measured right joints, issue asynchronous inference requests, forward returned chunks. |
| FFW | This companion validates chunks and submits full timed trajectories to the existing JointTrajectoryControllers. ROBOTIS handles interpolation and motor control at 100 Hz. |

The model's 20 Hz is waypoint spacing: 50 ms between predicted positions. It does
not require sending inference requests or network packets at 20 Hz. A whole chunk
is sent once. New chunks replace the unexecuted tail rather than joining a FIFO.
The companion's 20 Hz timer monitors feedback, timing, and expiry; it is not another
motor-control loop.

## Chunk and request lifecycle

1. Start the companion before the externally managed ROBOTIS initialization sequence.
   It does not start or restart the robot's bringup. It watches all four homing actions.
2. After successful homing, require fresh, complete positions and velocities at the
   selected YAML's final pose for a settling interval. Capture measured left-arm,
   left-gripper, head, and lift positions once for this session.
3. The companion publishes a session epoch and readiness on `/cosmos/bg2/status`.
   In execute mode it also verifies active position controllers, claimed interfaces,
   controller subscribers, and absence of other command publishers before sending holds.
4. The 5090 obtains fresh observations, sends them to PRO 6000, and forwards a decoded
   joint chunk to `/cosmos/bg2/chunk`. Only one request should be in flight; do not queue
   old camera observations behind a busy model.
5. FFW checks the session, request ID, original observation timestamp, joint order,
   shape, units, limits, velocities, fresh feedback, and held-joint drift. It discards
   elapsed rows. Fully expired chunks are rejected. It never restamps old predictions
   as fresh predictions.
6. FFW submits the remaining multi-point trajectory once. Controller interpolation
   executes the 20 Hz waypoints in the existing 100 Hz control loop. Left/head/lift
   targets remain the captured pose.
7. While that trajectory runs, the 5090 requests the next prediction from the newest
   observations, before the current forecast expires. `RequestPlanner` in `chunks.py`
   implements the timing decision from a measured latency budget plus margin.
8. If no fresh chunk arrives by the end, FFW sends a hold at the latest measured right
   position and keeps the captured passive-group holds. It reports `holding`/`needs_chunk`.
   A new valid chunk can resume a transiently starved session. It does not extrapolate.

A fresh chunk can replace an active chunk before its end. Frames and positions use
FFW ROS timestamps end to end. PRO 6000 must echo the original timestamp; its wall
clock must not manufacture new action timestamps. The gateway measures elapsed
latency with its local monotonic clock; FFW is the final timestamp authority.

Example with an already warmed 100 ms round trip: observation at t=0 predicts through
0.8 s. The first result arrives around 0.1 s. A new request at t=0.55 can arrive around
0.65, replacing the remaining tail before 0.8. The exact lead time comes from measured
latency, jitter, and command-delivery margin, not a fixed assumed model speed.

With observation request interval P and end-to-end latency L, continuous coverage
requires P+L plus margin to fit inside the 0.8 s horizon. A serial one-in-flight worker
also has P at least as large as its request latency. Cold recorded runs took about
2.99 s (Edge) and 1.57 s (Nano). The persistent service now warms before listening;
five local HTTP requests measured Edge 0.231 s median / 0.293 s max and Nano
0.401 s median / 0.440 s max. Real network/robot timing and sustained jitter still
need checking. See the [pipeline runbook](../../docs/three_machine_pipeline.md).

## Wire protocol

Input: `/cosmos/bg2/chunk`, `std_msgs/msg/String`, UTF-8 JSON no larger than 128 KiB:

- `schema`: `cosmos_bg2_joint_chunk_v1`.
- `epoch`: the current FFW session epoch from status.
- `request_id`: a unique 32-character lowercase hexadecimal ID.
- `source_time_ns`: the original observed joint timestamp in the FFW ROS clock.
- `joint_names`: arm_r_joint1 ... arm_r_joint7, gripper_r_joint1, in that order.
- `units`: eight `rad` entries.
- `positions`: 16 rows of eight absolute recorded joint positions. Row k is due at
  source_time_ns + (k+1)*50,000,000 ns. Do not include the conditioned current row.

Status: `/cosmos/bg2/status`, `std_msgs/msg/String`, schema `cosmos_bg2_status_v1`.
It contains epoch, phase, robot_now_ns, remaining_ns, valid_until_ns, last request ID,
mode, controller readiness, rejection reason, and count of physical command messages.
The gateway must treat stale status as unavailable. Robot state and camera messages
continue to come from the existing ROBOTIS nodes.

Stop: `/cosmos/bg2/control`, `std_msgs/msg/String`, JSON with schema
`cosmos_bg2_control_v1`, current epoch, and command `stop`. Stop holds the robot if
feedback and ownership permit, then disarms this companion instance. Start a new
companion session before a new homing sequence. No remote arm/resume command is provided.

## Offline inspection and later deployment

After an approved copy, these commands check files without opening a ROS connection:

```bash
python3 verify_bundle.py
python3 -B -m cosmos_ffw.node \
  --profile config/profile.reference.json \
  --initial-positions /path/to/the/actual/selected_initial_positions.yaml \
  --check
```

For later deployment, reuse the **already installed** `robotis/ai-worker:2.2.8`
image. There is no image build, package installation, or ROS/driver replacement.
The companion runs in a separate small container from that same image; the existing
`ai_worker` container and its bringup are not entered, edited, or restarted.

The image is the installed software environment. The companion container is just a
separate running process using that environment. Its entrypoint explicitly starts
our Python companion instead of the image's normal s6/robot-bringup entrypoint.
The inherited management-API healthcheck is disabled because that API is not started.
Image pulling is disabled; a missing local image fails instead of downloading another
version. A later deployment may set ROBOTIS_IMAGE to the verified installed image ID.

From the **generated bundle** on FFW, after filling `.env` from `.env.example` and
selecting the actual profile/home YAML, check the files without opening ROS:

```bash
docker compose run --rm companion \
  --profile /config/profile.json \
  --initial-positions /config/initial_positions.yaml \
  --check
```

Then start preview in the foreground:

```bash
docker compose up --no-build --pull never
```

No `docker compose build` or pip command is needed on FFW. All code, the launcher,
profile and chosen home YAML are mounted read-only. Missing source paths fail instead
of creating directories on the host. The root filesystem is read-only; temporary
logs/caches use `/tmp`. The companion uses the existing local Zenoh router and has no
serial devices, Docker socket, GPU, or privileged access. Default mode is preview.
Stopping this Compose service affects only the companion, leaving existing ROBOTIS
services running; physical stop limitations below still apply.

The local tests validate bundle portability, launch isolation and companion behavior.
The installed ARM64 image has not been run with this companion on FFW yet; live runtime
compatibility and controller/camera behavior remain commissioning work.

`.env.example` lists the later deployment settings. Set FFW_INITIAL_POSITIONS to the
same file selected by ROBOTIS init_position_file; changing that file requires no bridge
code changes. The user selected `bg2_vr_startup_snupi.yaml`, and the template now points
to it. Startup is pending review: its head_joint1 target 0.78 exceeds the actual installed
model upper limit 0.6951. The companion rejects that configuration. No target/limit has
been changed; resolve the mismatch before launching homing.

## Current validation boundary

The supplied reference profile is not live verified and explicitly disables execution.
Preview never creates hardware trajectory publishers. Execute additionally requires
an explicitly enabled verified profile and all runtime readiness/ownership conditions.
The 5090 network gateway and PRO 6000 SSH-tunneled HTTP RPC are implemented and tested
locally. The earlier Unix-socket bridge is a legacy co-located draft; use the three-machine
runbook. No transfer or live test is part of packaging this folder.

All homing statuses must correspond to goals started after this companion begins; an
old latched success cannot open readiness. External homing after capture invalidates
this inference session and clears its old holds so it cannot preempt reinitialization.

Feedback loss, clock jumps, controller health/ownership loss, or tracking divergence
fault the session. A stop command is sent only with usable feedback and ownership.
If those are unavailable, this code cannot guarantee an immediate physical stop; the
last submitted finite trajectory may finish and the existing controller retains its
last target. Keep the robot's independent stop/watchdog behavior available. This is
not a collision/path planner or a replacement for the original EE10 controller's IK/CBF.
Physical compatibility, acceleration behavior, live camera transforms, and sustained
latency still need commissioning before execute mode.
