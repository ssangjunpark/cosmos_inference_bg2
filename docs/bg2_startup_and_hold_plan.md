# BG2 startup and inference hold plan

Agreed on 2026-11-04 for `cosmos_inference_bg2`, to be cloned onto the inference PC
that communicates with the physical ROBOTIS FFW-BG2.

This is the design for the next integration stage. The current runner saves outputs from
recorded native samples; robot communication, homing coordination, and hold behavior are
not implemented yet.

## Agreed behavior

The inference PC runs Cosmos. The robot's existing ROBOTIS controllers and DYNAMIXEL
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

Keep a persistent GPU inference worker and a separate ROS 2 communication bridge. The bridge
collects robot feedback and camera observations, passes model inputs to the worker, validates
the returned predictions, and sends right-arm trajectories to the existing controller.
The bridge can use the deployed ROS middleware directly; choose any Zenoh transport only
after confirming the robot's actual setup. Model-to-bridge communication can use local IPC.

The current checkpoint contract describes:

- Eight right-arm/gripper channels in the joint order listed above.
- Sixteen future measured joint targets on a 20 Hz grid: a nominal 0.8-second horizon.
- One current measured joint state and three camera views.
- Training-specific joint normalization, camera preprocessing, and native prompt formatting.

Raw saved framework outputs are not yet validated controller commands. The matching model
and data adapter must establish output row indexing, padding, normalization, units, signs,
and gripper semantics before any values are published. Do not apply the public DROID
gripper conversion to this custom policy without evidence.

Measure inference latency on the inference PC before selecting the execution prefix,
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

## Information still needed

- The matching `bg2_cosmos/model.py` and `bg2_cosmos/data.py`, or a self-contained inference
  adapter. The full optimizer and training loop are not required.
- The custom embodiment ID and exact native inference batch/mask schema. The DCP metadata
  has 34 embodiment slots but does not identify which slot RobotEscape used.
- A verified conversion from model output to the eight physical joint targets, with
  normalization applied exactly as training requires.
- The deployed robot's controller configuration, middleware/domain/router settings, and
  camera topics, resolutions, frame selection, and timestamps.
- Validation that the chosen lift/head pose and camera preprocessing remain compatible with
  training. Changing those poses changes the images the policy receives.

The training camera labels are `cam_right_head`, `cam_right_wrist`, and `cam_left_wrist`.
Do not assume the standard SDK's camera presets select those same streams. The head crop in
the contract extends to pixel x=672, so an assumed 640-pixel input width would be incompatible.

## Next implementation work

First finish the [inference pipeline implementation plan](inference_pipeline_plan.md), so
the command path consumes verified model outputs.

1. Add configurable startup/hold coordination with states for initialization, settling,
   capture, ready, running, and failure. Test it against mocked feedback/controllers.
2. Add the observation adapter and persistent inference interface after reviewing the
   matching training adapter.
3. Add an offline trajectory builder and inspector that validate joint names, shapes, units,
   timestamps, finite values, limits, and target changes before constructing commands.
4. Verify recorded-sample inference and benchmark latency on the inference PC.
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
