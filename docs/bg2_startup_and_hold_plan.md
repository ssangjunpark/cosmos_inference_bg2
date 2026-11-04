# BG2 startup and inference hold plan

Agreed on 2026-11-04 for `cosmos_inference_bg2`, to be cloned onto the inference PC
that communicates with the physical ROBOTIS FFW-BG2.

Recorded-observation inference has passed for both Edge and Nano. Hardware integration
is now being prepared locally. The `3sp5090` inference PC is restricted to read-only
inspection: no file edits, installs, container restarts, homing, or controller changes there.
A local FFW companion now implements homing/hold coordination and chunk submission
for later deployment. It has been checked offline; live commissioning and the 5090 ↔ PRO
6000 network gateway remain pending. See [FFW bundle](../deploy/ffw/README.md).

## Agreed behavior

The PRO 6000 server runs Cosmos; the 5090 is the observation/inference gateway.
The robot's existing ROBOTIS controllers and DYNAMIXEL
hardware interface execute position targets and retain responsibility for low-level motor
control. LeRobot is not a required runtime dependency.

| Robot group | Behavior during inference |
| --- | --- |
| Right arm and right gripper | Follow verified Cosmos position predictions: `arm_r_joint1` through `arm_r_joint7`, then `gripper_r_joint1`. |
| Left arm and left gripper | Hold the measured positions captured after initialization: `arm_l_joint1` through `arm_l_joint7`, then `gripper_l_joint1`. |
| Elevator | Hold the captured position of `lift_joint`; the model does not predict it. |
| Head | Hold the captured `head_joint1` and `head_joint2` positions to keep the camera pose stable. |
| Mobile base | Remain stationary through the deployed base controller; the model supplies no base commands. |

“Hold” means active position control, not a mechanical lock. Hold targets stay constant for
the inference session. Continuing to replace them with the latest measured positions would
allow the held pose to drift.

## Startup sequence

1. Select the initial-position YAML through ROBOTIS's `init_position_file` launch argument,
   with initialization enabled. Use the existing ROBOTIS homing executor.
2. Initialize the robot to that YAML's pose. The right arm also initializes before it receives
   policy commands.
3. Wait for initialization to finish for all relevant groups. Confirm fresh, complete joint
   feedback and that positions and velocities satisfy the configured homing tolerances.
   Starting the launch process or sleeping for a fixed interval does not establish readiness.
4. Read `/joint_states` by joint name and capture the reached left-arm, left-gripper, lift,
   and head positions. Capture them **after homing and settling**, not when the script first starts.
5. Give those controllers the captured hold targets and establish a single command owner
   for each controlled group. Teleoperation publishers must not compete with inference.
6. Start right-arm policy execution only after the hold state, observations, model, and first
   decoded prediction are ready. Until then, retain the right arm's initialized target.

The model can load while the robot initializes, but loading it must not enable motion.
A failed initialization or missing/stale feedback must keep the policy execution gate closed.

Changing the initial-position YAML should require no bridge code changes. On the next
initialization, the newly reached measured pose becomes the session's hold pose. The bridge
must not hardcode the left arm or elevator to zero.

## Default initial pose

The official FFW-BG2 revision 2, 3, and 4 initial-position files specify this left-arm home:

```text
joint order: arm_l_joint1 ... arm_l_joint7, gripper_l_joint1
home:        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
duration:    5.0 seconds
```

The default lift home is `0.0`, with a configured duration of 10 seconds. These are
documented defaults, not values to force on every inference session. An installation may
select a different YAML.

The homing executor uses measured start positions, generates a smooth trajectory, and checks
position and velocity tolerances. The startup supervisor must wait for actual completion.

## Runtime behavior

Keep a persistent GPU inference worker and a separate ROS 2 communication bridge. The 5090 gateway
collects feedback and cameras using the installed-stack-matched Zenoh SDK, sends prepared
observations over an SSH-tunneled HTTP connection to PRO 6000, and forwards
whole timestamped chunks. The FFW companion validates and sends right-arm trajectories
to existing controllers. The [implemented three-machine runbook](three_machine_pipeline.md)
replaces the earlier co-located local-IPC proposal. All code remains local until deployment.

The current checkpoint contract describes:

- Eight right-arm/gripper channels in the joint order listed above.
- Sixteen future measured joint targets on a 20 Hz grid: a nominal 0.8-second horizon.
- One current measured joint state and three camera views.
- Training-specific joint normalization, camera preprocessing, and native prompt formatting.

The matching `cosmos_futurejoint` source is integrated. Recorded Edge/Nano runs return
16-by-8 absolute right-arm/gripper positions in recorded radians after one inverse affine.
Domain 33, padding, current-state conditioning, and row selection are verified. Live camera
orientation, controller configuration, hardware limits, observation age, and execution timing
still need validation before publishing commands. Preserve the recorded joint coordinates
and gripper values.

Warm Edge/Nano HTTP latency is now measured on PRO 6000 in the runbook. Verify real
network and robot timing during commissioning before selecting the execution prefix,
trajectory timestamps, buffering, and refresh cadence. Preserve observation timestamps;
a prediction based on stale images cannot be treated as a fresh 0.8-second forecast.
The robot continues moving during inference, unlike a simulation that pauses its clock.

Hold targets remain fixed even if the right arm replans. The controller may maintain its last
target or require refreshes depending on its deployed configuration; a refresh must reuse the
captured target rather than recapture the measured pose.

For an explicit reinitialization, pause policy updates, finish the chosen initialization
sequence, discard old observations and predictions, recapture hold targets, and reopen the
execution gate after readiness checks. Editing a YAML file during inference must not
automatically move the robot.

## Controller interfaces

The reviewed standard FFW-BG2 AI bringup uses position-command
`JointTrajectoryController` instances. Its controller-manager configuration specifies 100 Hz.
The hardware interface sends DYNAMIXEL Goal Position commands; Cosmos does not compute or
send joint torques.

| Purpose | Official reference interface |
| --- | --- |
| Measured joint feedback | `/joint_states`, `sensor_msgs/msg/JointState` |
| Right-arm position trajectories | `/leader/joint_trajectory_command_broadcaster_right/joint_trajectory`, `trajectory_msgs/msg/JointTrajectory` |
| Left-arm homing action | `/arm_l_controller/follow_joint_trajectory` |
| Lift homing action | `/lift_controller/follow_joint_trajectory` |
| Head homing action | `/head_controller/follow_joint_trajectory` |

Confirm namespaces, controller activation, interfaces, and command ownership against the
installed robot stack before using these reference names.

The official LeRobot BG2 preset is only a source reference, not our proposed implementation.
It lists both arms, and its sender fills omitted joint commands with zero. It also sends a
single immediate trajectory point. Our bridge must explicitly separate right-arm policy
commands from left/lift/head hold commands and provide deliberate trajectory timing.

## Read-only inference-PC inspection (2026-11-04)

`3sp5090` is an x86_64 Ubuntu 24.04.5 inference PC. Its existing containers use ROS 2
Jazzy, `rmw_zenoh_cpp`, and ROS domain 30. Existing configuration names the robot router
at `192.168.0.14:7447`; reachability was not tested during this inspection.

The existing deployments include a pi0.5 EE10 runtime and a separate motion-preview setup.
Their existence does not establish controller ownership or authorize their reuse for Cosmos
joint-position commands. No management scripts, ROS services, or robot command topics were invoked.

Local pinned source dependencies:

- `third_party/ai_worker`: official Jazzy branch, commit `7c6605c6f006e840566c8db4fecaddda47e9a786` (2.3.2).
  This is a reference checkout; the installed onboard driver revision is unverified.
- `third_party/zenoh_ros2_sdk`: commit `be2c4d4595305a9c282fca09820a8b3bfb8076a3` (0.1.8), matching
  the Git revision read from both inspected SDK checkouts on the inference PC.

Both are proper Git submodules with local lock files. Neither repository has been modified
or installed on `3sp5090`. The standalone Zenoh SDK can supply communication independently
of the existing LeRobot model runtime. Keep the network-facing communication bridge separate
from the offline GPU worker and reuse the original ROBOTIS controllers on the robot.

The local [reference profile](../profiles/robotis_bg2_rev4.reference.json) records camera topics,
command topics, 19 named joint limits from a reference URDF, source hashes, and middleware
settings. It explicitly disables execution and marks live compatibility as unverified.
It is reference data, not a running controller bridge. The source config maps the right head,
left wrist, and right wrist to the required training camera labels.

## Information still needed

- Actual robot-side bringup revision, selected initial-position YAML, active controllers,
  namespaces, limits, and command ownership. The reference URDF does not establish installed limits.
- Fresh live camera resolutions, frame timestamps, and rotation conventions. The reference
  config specifies 270-degree wrist rotation, but the exact transformation from live images
  into the MP4 frames used by training remains unverified. Do not apply that rotation blindly.
- Confirm all three streams: `cam_right_head`, `cam_right_wrist`, and `cam_left_wrist`.
  One existing policy deployment config includes only right head and right wrist. Cosmos needs both wrists.
- Validate the chosen lift/head pose against the training camera pose and benchmark the persistent
  worker on the inference PC. The current first-call generation timings exceed the 0.8-second horizon.

The training head crop extends to pixel x=672. An assumed 640-pixel source width would be
incompatible. The live observation adapter must reproduce the original PIL bilinear crop/resize
and head letterboxing after establishing the source orientation.

## Next implementation work

Use the verified output decoder from the [inference pipeline implementation plan](inference_pipeline_plan.md).
Continue local implementation while keeping `3sp5090` read-only.

1. Local FFW startup/hold coordination and offline/mock tests are implemented; commission
   them against the installed controllers only after deployment is authorized.
2. Add the live observation adapter after verifying camera orientation, and add a persistent
   inference interface using the already integrated training adapter.
3. The tested chunk builder validates names, shape, units, timestamps, finite positions,
   reference limits, tracking, and velocity. Confirm real hardware limits/profile before use.
4. Benchmark warmed repeated inference and choose timestamp handling and execution cadence.
5. Verify live feedback and controller ownership before exercising the complete command path.

## Source references

ROBOTIS source was reviewed at commit
`b1874d523cadca40776d87a373392c0ada3d7da1`. The physical installation may use another revision.

- [BG2 AI bringup and initial-position file selection](https://github.com/ROBOTIS-GIT/ai_worker/blob/b1874d523cadca40776d87a373392c0ada3d7da1/ffw_bringup/launch/ffw_bg2_follower_ai.launch.py)
- [BG2 revision 4 default initial positions](https://github.com/ROBOTIS-GIT/ai_worker/blob/b1874d523cadca40776d87a373392c0ada3d7da1/ffw_bringup/config/ffw_bg2_rev4_follower/ffw_bg2_follower_initial_positions.yaml)
- [ROBOTIS homing executor](https://github.com/ROBOTIS-GIT/ai_worker/blob/b1874d523cadca40776d87a373392c0ada3d7da1/ffw_bringup/ffw_bringup/joint_trajectory_executor.py)
- [BG2 AI controller configuration](https://github.com/ROBOTIS-GIT/ai_worker/blob/b1874d523cadca40776d87a373392c0ada3d7da1/ffw_bringup/config/ffw_bg2_rev4_follower/ffw_bg2_follower_ai_hardware_controller.yaml)
- [BG2 hardware command interfaces](https://github.com/ROBOTIS-GIT/ai_worker/blob/b1874d523cadca40776d87a373392c0ada3d7da1/ffw_description/ros2_control/ffw_bg2_rev4_follower/ffw_bg2_follower.ros2_control.xacro)
- [ROBOTIS ROS and Zenoh sender used as a reference](https://github.com/ROBOTIS-GIT/lerobot_robot_ros2_zenoh/blob/48d1fa5be557d352ee2e0e2a69c6527e77303d1d/lerobot_robot_ros2_zenoh/ros2_zenoh.py)
- Checkpoint input contract on the current checkpoint server:
  `/mnt/ssd2/RobotEscapeCkpt/cosmos3_edge/020000/input_contract.json`.
  Framework commit: `b98934eda43f4d31972538a527f8663f003c27dc`.
  Keep the contract with the deployed checkpoint; checkpoints and generated artifacts stay outside Git.
