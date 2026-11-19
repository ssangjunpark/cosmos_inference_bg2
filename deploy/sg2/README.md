# FFW-SG2 fixed-base arm companion

The companion runs in the SG2 follower's ROS 2 Jazzy environment. It reads
`/joint_states` and exchanges arm chunks, control messages, and status with
the 5090 on `/cosmos/sg2/{chunk,control,status}`. Both grippers are part of the
arm trajectories. The only actuator publishers are the two arm topics below.
The operator's bringup owns the head, lift and base.

| Controller | Default JointTrajectory topic | Message joint order |
| --- | --- | --- |
| `arm_l_controller` | `/leader/joint_trajectory_command_broadcaster_left/joint_trajectory` | `arm_l_joint1` through `arm_l_joint7`, then `gripper_l_joint1` |
| `arm_r_controller` | `/leader/joint_trajectory_command_broadcaster_right/joint_trajectory` | `arm_r_joint1` through `arm_r_joint7`, then `gripper_r_joint1` |

The July 9 Jazzy source remaps each arm controller's input to its leader topic.
It also has separate swerve initialization and drive spawners. The companion
does not launch the follower or immobilize the chassis. Use the robot's
fixed-base setup independently. [Follower launch](https://github.com/ROBOTIS-GIT/ai_worker/blob/6961f7e7368329d57ebf2adf0fdba08680a43949/ffw_bringup/launch/ffw_sg2_follower_ai.launch.py)
and [controller configuration](https://github.com/ROBOTIS-GIT/ai_worker/blob/6961f7e7368329d57ebf2adf0fdba08680a43949/ffw_bringup/config/ffw_sg2_rev1_follower/ffw_sg2_follower_ai_hardware_controller.yaml).

## Build the bundle and check it offline

From the repository root on the 5090, with the ai_worker submodule available:

```bash
python3 scripts/package_ffw.py --robot sg2
cd transfer/sg2
python3 verify_bundle.py
python3 -m cosmos_ffw.node --profile config/profile.reference.json \
  --initial-positions config/initial_positions.yaml --check
```

The offline check uses NumPy and PyYAML. The included initial-position YAML
provides the two arm groups and settling tolerances. The companion never sends
the file's startup poses. Current-pose initialization is the default.

## Verify the installed interface

In the SG2 follower's sourced ROS shell, record the installed revisions, domain,
RMW implementation, and active endpoints. Use the source paths for that installation.

```bash
for repo in ai_worker robotis_applications physical_ai_tools; do
  git -C "$ROBOTIS_SRC/$repo" rev-parse HEAD
done
ros2 doctor --report
ros2 control list_controllers --verbose
ros2 topic info /joint_states --verbose
for side in left right; do
  ros2 topic info "/leader/joint_trajectory_command_broadcaster_${side}/joint_trajectory" --verbose
done
```

Set `ROBOTIS_SRC` to the workspace's source directory before the loop. Check
that feedback includes positions and velocities for all 16 arm coordinates.
If the installed launch uses the native JTC inputs, the reference profile also
accepts `/arm_l_controller/joint_trajectory` and `/arm_r_controller/joint_trajectory`
for their corresponding arms. In execute mode the companion checks that the
named controller subscribes, claims the expected position interfaces, and has
no competing command publisher on that topic.

To preview in the SG2 ROS environment, run the same command from the transferred
bundle without `--check`. Preview subscribes to both bridge inputs and publishes
status. It creates no actuator publishers. Check from the 5090 with
`ros2 topic echo /cosmos/sg2/status --once` using compatible ROS discovery and QoS.
This bundle inherits the running robot's middleware setup. The BG2 deployment's
Zenoh address and container image are specific to that installation.

Physical execution requires an SG2 deployment profile with verified joint limits
and topic mappings. Copy the reference profile to a new file, set
`execution_enabled` to `true` and `status` to `verified_on_robot` after checking
the installation, and select it with `--profile` and `--mode execute`.

## 5090 wire interface

All three bridge topics carry JSON in `std_msgs/msg/String`. These are robot
command coordinates in radians. A future 5090 adapter can convert the trained
model's outputs into this format.

- Status uses `cosmos_sg2_status_v3`. Wait for `needs_chunk` and use the reported
  `epoch`, `cycle`, and `ready_since_ns`. Check `mode` to distinguish preview
  from execution.
- A chunk uses `schema: cosmos_sg2_joint_chunk_v1` and
  `execution_contract: synchronous_full_chunk_v2`. Include the current epoch
  and integer cycle, a unique 32-character lowercase hexadecimal `request_id`,
  and the fresh observation's `source_time_ns` in the robot clock domain.
  Supply all 16 unique `joint_names` in the order of the columns, 16 `"rad"`
  entries in `units`, and 16 rows of 16 absolute `positions`. The companion
  maps columns by name to each controller. It checks the entire chunk against
  the SG2 profile's joint ranges and velocity limits before publishing either arm.
- Both trajectories use the same header timestamp and the same 50–800 ms
  point offsets. ROS publishes two separate messages. After the scheduled
  interval the companion holds each arm's final target. Readiness acknowledges
  publication of those holds, not measured arrival or atomic delivery.
- Control uses `schema: cosmos_sg2_control_v1`, the current `epoch`, and
  `command` set to `pause`, `resume`, or `stop`. Pause and resume hold fresh
  measured arm positions and rotate the epoch. Old chunks cannot resume motion.

The dated Jazzy revisions used as interface references are
[ai_worker](https://github.com/ROBOTIS-GIT/ai_worker/commit/6961f7e7368329d57ebf2adf0fdba08680a43949),
[robotis_applications](https://github.com/ROBOTIS-GIT/robotis_applications/commit/ed07c1a996fa8392ec0defe2e51423f301ada2e9),
and [physical_ai_tools](https://github.com/ROBOTIS-GIT/physical_ai_tools/blob/3d83836193e545fc51fb7b9713e83fcf88b8606a/physical_ai_server/config/ffw_sg2_rev1_config.yaml).
