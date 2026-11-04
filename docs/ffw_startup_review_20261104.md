# FFW follower startup for the Cosmos pipeline

Reviewed 2026-11-04 against ROBOTIS documentation and the installed FFW stack.
This review was read-only: no file changes, installations, service starts/stops,
ROS initialization, follower launch, homing or motion commands on FFW.

## Current observed state

- FFW is reachable through PRO 6000 → `3sp5090` → `ffw`.
- The existing `ai_worker` container uses `robotis/ai-worker:2.2.8`; the management
  and Zenoh containers are running. No follower controller, state publisher or
  camera processes were found during the process check.
- `/dev/follower` exists. This does not establish controller readiness or physical safety.
- The manager UI listens on port 3000, its API on 8081, and Zenoh on TCP 7447.
  From the 5090 robot LAN, the UI address is `http://192.168.0.14:3000`.
- The existing container's shell setup selects ROS domain 30 and `rmw_zenoh_cpp`.
- Read-only imports of NumPy, PyYAML, rclpy, controller-manager messages, action
  status and trajectory messages passed in its Python 3.12.3 environment. No ROS
  context was initialized, and bytecode writes were disabled.

The installed follower launch and home YAML match the inspected host source byte
for byte. Source revision: `c0282a1fe9865c46e1378146fc3aaab2491c6ebd`.

| Installed file | SHA-256 |
| --- | --- |
| `ffw_bg2_follower_ai.launch.py` | `ad9f326701261066519a66dd4b24b88c70a7d6d90ccd67fb7f65f0742f7701da` |
| `ffw_bg2_follower_initial_positions.yaml` | `88b914128a4d41d1e7b5b9ce6866619d8ce275deead56b7e9d37396ba54b0a38` |

## ROBOTIS startup and our adaptation

ROBOTIS starts Zenoh before the robot stack, then brings up the BG2 follower.
Cyclo Manager's System page provides model selection, launch parameters and a
bringup button. See the [official bringup guide](https://docs.robotis.com/docs/systems/aiworker/quick_start_guide/operation_guide/cyclo_manager/bringup/).

Their [inference guide](https://ai.robotis.com/ai_worker/model_inference_ai_worker.html)
also starts the follower before policy inference and disables teleoperation first.
We reuse that robot bringup. Our model execution stays on PRO 6000; our 5090 gateway
and FFW companion supply the Cosmos request/trajectory path instead of their LeRobot
model-loading/UI inference steps. No Cosmos checkpoint needs to be copied to FFW.

## Planned order

1. Prepare/start our separate FFW companion in **preview**, using the installed
   ROBOTIS image and the same home YAML the follower will use. This is still pending.
2. Confirm the existing Zenoh router is available and middleware/domain match.
3. Start **only the BG2 follower** through Cyclo Manager → System → BG2 Robot Bringup.
   Use the gear/settings control to check the saved parameters below. Alternatively,
   enter the existing AI Worker container and use the documented command shown below.
   Use one bringup method; do not launch another copy through the other method.
4. Wait for successful initialization of all four groups and settled feedback. Our
   companion captures measured left-arm/gripper, head and elevator holds once.
5. Check `/joint_states`, controller health and the head/right-wrist/left-wrist
   image streams. Verify actual raw dimensions and orientation against training.
6. Start/warm the PRO 6000 model service and open the existing reverse SSH tunnel.
   Configure the instruction and verified camera profile on the 5090.
7. Start gateway **preview** and verify fresh chunks, acceptance and timing. Preview
   creates no policy trajectory publishers on FFW; normal follower homing still moves.
8. Commission execute only after profile/limits/ownership are verified. Teleoperation
   leader, VR and old motion adapters must not compete for command topics. The existing
   management containers can remain running.

Our companion requires fresh homing goals started after its own startup. Launch it
before follower initialization. Changing companion mode/restarting it creates a new
session and requires fresh homing again; a previous session's success is insufficient.

## Saved follower parameters to retain

| Parameter | Observed selection |
| --- | --- |
| model | `ffw_bg2_rev4_follower` |
| port_name | `/dev/follower` |
| use_sim / use_mock_hardware | `false` / `false` |
| launch_cameras | `true` |
| head_camera_type | `zed` (installed default) |
| init_position | `true` |
| init_position_file | `ffw_bg2_follower_initial_positions.yaml` |
| use_head_eef_tracker | `false` |
| ros2_control_type | `ffw_bg2_follower` |

The selected YAML initializes both arms/grippers and head to zero over configured
five-second trajectories, and the lift to zero over ten seconds. These are targets,
not a guarantee of completion by that time. The installed launch schedules the head
camera launch at 20 seconds when initialization is enabled; its nested RealSense
launch adds another ten-second delay for the wrists. Wait for actual fresh messages.
Do not substitute the alternate YAML with the previously identified 0.78-radian head
target until its limit mismatch is resolved.

## CLI alternative — not executed

The [official teleoperation guide](https://ai.robotis.com/ai_worker/operation_teleoperation_ai_worker)
provides the follower-only launch. Enter the already-running AI Worker container:

```bash
cd ~/ai_worker/docker
./container.sh enter
```

Inside that container, after our companion is running:

```bash
ros2 launch ffw_bringup ffw_bg2_follower_ai.launch.py \
  launch_cameras:=true init_position:=true \
  init_position_file:=ffw_bg2_follower_initial_positions.yaml
```

This starts real hardware and initialization can move both arms, head and lift.
A person at the robot must confirm the area is clear and the independent stop is
available before this step. We have not started it during the review. The existing
container is already running, so there is no need to invoke its start/update helper,
which can pull images. No software update is proposed.
