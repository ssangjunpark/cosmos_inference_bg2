# Read-only FFW inspection — 2026-11-04

Inspection used PRO 6000 → SSH `3sp5090` → SSH `ffw`. Both remote machines remained
read-only: no edits, installations, new containers, restarts, ROS service calls,
homing, or robot commands. SSH used existing keys/known hosts, strict host-key
checking, and disabled host-key updates and agent/key additions.

Implementation remains paused. This report records inspection results and design implications.

## Confirmed topology

- PRO 6000 server: Cosmos inference and this development repository.
- 5090 (`3sp5090`): proposed observation/command gateway between robot LAN and GPU server.
- FFW (`ffw-SNPR48A1011`): robot-side control and sensors, ARM64 ROBOTIS Orin B/D,
  Ubuntu 22.04.5 host. ROS 2 Jazzy is provided inside the existing containers.
- FFW eth0 is `192.168.0.14`; the existing Zenoh router is listening on TCP 7447.

The earlier bridge draft assumes a local Unix socket between bridge and worker.
That IPC cannot span the PRO 6000 and 5090 hosts. A network transport and an explicit
clock/timestamp contract must be designed and tested before this topology can run.

## Installed services versus active robot functions

FFW has these running containers:

- `ai_worker`: `robotis/ai-worker:2.2.8`
- `zenoh_daemon`: `robotis/zenoh-daemon:latest`
- `cyclo_intelligence`: `robotis/cyclo-intelligence:1.4.0`
- `robotis-applications`: `robotis/robotis-applications:1.0.1`
- Cyclo manager/UI, physical AI server/manager, noVNC, and an Isaac GR00T container.

At inspection, ai_worker contained its s6 supervisors and management API, with no
robot bringup process. A host-wide process snapshot found no ros2_control_node,
robot_state_publisher, RealSense/ZED camera nodes, homing executor, or custom motion
controller. The application container contained an idle bash process. These are
point-in-time observations, not a controller-state guarantee for a later session.
No controllers or cameras were started to obtain live data.

## Actual source and saved startup selection

- `/home/robotis/ai_worker`: Jazzy, HEAD `c0282a1fe9865c46e1378146fc3aaab2491c6ebd`.
  This differs from our local 2.3.2 reference pin `7c6605c6f006e840566c8db4fecaddda47e9a786`.
- Cyclo checkout: `e648425e1f724ba61a8060369e200bd3e0f2a877`.
- Physical AI Tools checkout: Jazzy, `96f9b1d7ceb967594b8e7f09f020fb02bf1b2b80`.
- `/run/robot_type` in ai_worker contains `bg2`.
- Saved `/run/launch_args/ai_worker_bringup` selects real BG2 Rev4 hardware,
  `/dev/follower`, cameras enabled, initialization enabled, and
  `init_position_file:=ffw_bg2_follower_initial_positions.yaml`.
  These are saved launch arguments; the bringup was not running at inspection.
- The standard controller YAML configures position JointTrajectoryControllers for
  both arms, head, and lift, with a 100 Hz controller manager. Topic remappings match
  the existing right/left arm and head/lift reference topics.

A separate `bg2_vr_startup_snupi.yaml` exists but is not selected by the saved launch
arguments. It contains nonzero arm targets and lift -0.18. Its head_joint1 target
is 0.78, while the installed source URDF upper limit is 0.6951. That mismatch needs
resolution before choosing that alternate startup file. Neither file was changed.

## Existing motion package and reuse boundary

`/home/robotis/cyclo-bg2-motion` already contains an adapter, control loop, transport,
output guard, assets, a controller profile, and a launch file. Its sources reference
ROBOTIS cyclo_control commit `8731eda09ebf885319f47aa2f60537470922787d`.

- The launch starts Cyclo vr_controller_node/reference_checker_node at a configured
  100 Hz, on /smolvla topics, and publishes a controller profile.
- The policy adapter currently requires `bg2_absolute_ee10_v2`, action_dim 10,
  chunk_size 4, dataset_hz 20, and generates pose/gripper inputs.
- It includes session/deadline handling, passive-joint hold checks, and output guards.
- Its profile states controller outputs do not directly target hardware; the existing
  runtime transport separately chooses preview or the right-arm hardware trajectory topic.
- The package was not observed running, and no separate motion-controller container
  appeared in the running/stopped container list. Presence of code is not a completed deployment.

Cosmos supplies eight absolute joint positions for 16 future steps. Its output cannot
be fed into the EE10 adapter unchanged. Reuse the existing robot-side low-level
controllers; review where joint-target validation/chunk timing should live. Avoid
adding another IK implementation or another 100 Hz motor-control loop on the 5090.
This inspection does not establish which existing higher-level motion safeguards can
be reused unchanged for joint-space commands.

## Camera evidence and remaining unknowns

Installed launch source configures D405 color streams at 424×240×30 and the ZED Mini
configuration selects VGA. These are configuration defaults, not measured live frames.
Actual stream dimensions, timestamps, and the exact wrist rotation into the training
MP4 convention remain unverified because the camera nodes were not running.

Before implementation resumes, reconcile the three-machine transport, installed driver
revision, live camera transform, controller ownership, selected home YAML, and the
appropriate robot-side joint-target interface. Preserve the existing paused WIP commit
as untested; no deployment or physical execution is authorized by this inspection.
