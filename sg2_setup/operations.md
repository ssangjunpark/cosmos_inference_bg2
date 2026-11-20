# SG2 commissioning and operation

Start with the [fresh setup guide](README.md). The steps below assume that a single companion is running on the SG2, the 5090 is on the same ROS graph, and the terminal labels refer to those two machines.

## Prove both directions in preview

**5090 ROS shell.** Run this with a freshly started, idle preview. A status echo proves SG2 → 5090. An acknowledged pause followed by an acknowledged resume proves 5090 → SG2 as well. The test uses only the bridge control and status topics.

These Bash functions read the full JSON status inside a ROS String, verify preview mode and the expected phase, and send one control message with the current epoch. They use the 5090's system Python and PyYAML from the setup guide. Keep the functions and the commands that use them in the same terminal.

```bash
sg2_status() {
  timeout 12s ros2 topic echo /cosmos/sg2/status std_msgs/msg/String --full-length --once
}

sg2_preview_control() {
  local request
  case "${1:-}" in
    pause|resume) ;;
    *) echo "Usage: sg2_preview_control pause|resume" >&2; return 1 ;;
  esac
  request=$(sg2_status | /usr/bin/python3 -c '
import json, sys, yaml
command = sys.argv[1]
sample = next((doc for doc in yaml.safe_load_all(sys.stdin) if doc), None)
if sample is None:
    sys.exit("No SG2 status received. Check the companion, ROS domain, middleware and discovery settings.")
s = json.loads(sample["data"])
phases = {"pause": ("ready", "holding"), "resume": ("paused",)}
if (s.get("schema") != "cosmos_sg2_status_v3" or s.get("mode") != "preview"
        or s.get("physical_publishing") is not False or s.get("command_messages") != 0
        or s.get("phase") not in phases[command]):
    sys.exit("Expected an idle SG2 preview in the appropriate phase. Read its status first.")
control = {"schema": "cosmos_sg2_control_v1", "epoch": s["epoch"], "command": command}
print(json.dumps({"data": json.dumps(control)}))
' "$1") || return 1
  timeout 12s ros2 topic pub --once /cosmos/sg2/control std_msgs/msg/String "$request"
}
```

```bash
sg2_preview_control pause
sg2_status
```

Expect `phase="paused"`, `needs_chunk=false`, and a new epoch. Then run:

```bash
sg2_preview_control resume
sg2_status
```

Expect `phase="holding"`, `needs_chunk=true`, another new epoch, and `command_messages=0` throughout. The changing epoch proves the companion processed the command. A CLI message saying it published once only establishes that the publisher sent it. Press Ctrl+C in the **SG2 companion terminal** when finished.

## Match the profile to the installed robot

**SG2 ROS shell.** These were the dated Jazzy references used for the branch. A pull date alone does not identify a local checkout that has edits or tracks a different branch.

| Repository | Supplied pull date | Jazzy reference | Relevance |
| --- | --- | --- | --- |
| `ai_worker` | July 9, 2026 | [`6961f7e`](https://github.com/ROBOTIS-GIT/ai_worker/commit/6961f7e7368329d57ebf2adf0fdba08680a43949) | Arm controller configuration, launch remaps, SG2 URDF limits, and initial-position YAML |
| `robotis_applications` | May 29, 2026 | [`ed07c1a`](https://github.com/ROBOTIS-GIT/robotis_applications/commit/ed07c1a996fa8392ec0defe2e51423f301ada2e9) | Existing SG2 teleoperation publishers that may share arm command topics |
| `physical_ai_tools` | June 26, 2025 | [`3d83836`](https://github.com/ROBOTIS-GIT/physical_ai_tools/commit/3d83836193e545fc51fb7b9713e83fcf88b8606a) | Recorder field names and the distinction between arm, head, lift and mobile data |

The parent repo pins a November `ai_worker` submodule (`7c6605c`). The SG2 packager copies only its SG2 initial-position YAML from that submodule. Its bytes match the July reference, SHA-256 `a81ba37a5891d12ee47f8f9383e711146d4ed582e5bf1bd0ac43136c4d52e4a2`. The 16 reference limits and the URDF source hash in `config/profile.reference.json` come from July. The bundle adds a Python companion beside ROBOTIS and leaves the installed drivers in place.

To identify the actual installation, run these in the follower's ROS shell. Set the source root to the real checkout location. Repositories may live on different machines or be absent from a binary image. Run the Git checks wherever each checkout lives.

```bash
ROBOTIS_SRC=/root/ros2_ws/src
for repo in ai_worker robotis_applications physical_ai_tools; do
  if test -e "$ROBOTIS_SRC/$repo/.git"; then
    git -C "$ROBOTIS_SRC/$repo" rev-parse --show-toplevel HEAD
    git -C "$ROBOTIS_SRC/$repo" status --short
  else
    echo "No Git checkout at $ROBOTIS_SRC/$repo"
  fi
done
ros2 pkg prefix --share ffw_bringup
ros2 pkg prefix --share ffw_description
```

From the **release folder**, compare the installed files with the references. Local modifications and robot-specific launch arguments can change the graph even when the Git revisions agree.

```bash
SG2_BRINGUP=$(ros2 pkg prefix --share ffw_bringup)
SG2_DESCRIPTION=$(ros2 pkg prefix --share ffw_description)
sha256sum config/initial_positions.yaml \
  "$SG2_BRINGUP/config/ffw_sg2_rev1_follower/ffw_sg2_follower_initial_positions.yaml"
sha256sum "$SG2_DESCRIPTION/urdf/ffw_sg2_rev1_follower/ffw_sg2_follower.urdf"
ros2 control list_controllers --verbose
ros2 topic info /leader/joint_trajectory_command_broadcaster_left/joint_trajectory --verbose
ros2 topic info /leader/joint_trajectory_command_broadcaster_right/joint_trajectory --verbose
```

Look for `arm_l_controller` and `arm_r_controller` in namespace `/` as subscribers of their respective topics. Each controller must claim its seven arm joint position interfaces plus its gripper position interface. An extra subscriber can be a recorder. A second **publisher** on an arm command topic can be a teleoperation process. Check the publisher's node name before handing command ownership to the companion. Stop a competing command source using its own normal session controls while keeping the follower and controller manager running.

If the installed launch uses native controller inputs, the supported alternatives are `/arm_l_controller/joint_trajectory` and `/arm_r_controller/joint_trajectory`. Set the respective `commands.arm_left.topic` and `commands.arm_right.topic` in a site-specific copy of the profile. If the joint set, limits, namespace, or remaps differ in another way, keep the observed values and revise the integration against that installation before executing.

## Later physical execution

The initial installation is complete at the preview round trip. The following is a **separate operating mode**. In execute mode, the companion publishes physical holds as soon as the arms settle, the controllers pass their health checks, and the command topics have exclusive ownership. This happens even with no model connected.

1. Stop preview with Ctrl+C. Confirm a fixed chassis, the installation's normal means of stopping motion, the actual arm interfaces and limits, and who owns the two command topics. Keep the reference profile intact for future verification.
2. In the **SG2 ROS shell**, copy the profile once, review it, and set `status` to `verified_on_robot` and `execution_enabled` to `true` **in the deployment copy** after verification. The example below puts the copy in the release directory. For a container, retain an additional copy on the robot host if it should survive recreation.
3. Validate the selected files offline before running the foreground process:

```bash
cp -n config/profile.reference.json config/profile.site.json
# Edit config/profile.site.json with the verified values described above.
/usr/bin/python3 -B -m cosmos_ffw.node \
  --profile config/profile.site.json \
  --initial-positions config/initial_positions.yaml \
  --mode execute --initialization current --check
```

For an intentional hardware session, run the validated invocation **without `--check`**. From a second terminal verify `mode="execute"`, `controllers_ready=true`, a current epoch, and a ready or holding phase. Current-pose startup captures measured positions. The supplied initial-position YAML provides the arm groups and settling tolerances.

**5090 model sender.** Use the [SG2 policy guide](policy.md) to select a PNP1 or PNP2 checkpoint, confirm its dataset-to-controller calibration, and run a synchronized preview. The sender fills a 16-joint chunk at 50 ms intervals using the current session and hold timestamp, then waits for the matching completion acknowledgement. PNP1 holds the measured left arm. PNP2 predicts both arms. [SG2 protocol](../deploy/sg2/README.md#5090-wire-interface)

## Stop, restart, update, or roll back

- **Stop the companion** with Ctrl+C in its foreground SG2 ROS shell. In execute mode it attempts a measured-position hold on each arm using fresh feedback and valid ownership. Keep the robot's own stop procedure available for a hardware fault or lost feedback.
- **Restart** from the same release and ROS environment, beginning in preview. Each new session gets a new epoch. A terminal fault requires diagnosing the reason and starting a new session.
- **Update** by building the desired clean `sg2` commit on the 5090, copying it to its own release directory, verifying its manifest, and repeating preview. An existing `transfer/sg2` can be preserved by building with `--output /path/to/new/directory`. Update `SG2_BUNDLE` accordingly in the transfer commands.
- **Roll back** by stopping the current companion, selecting the previous verified release on the robot host, copying that release back into the ROS environment if needed, and starting it in preview. Site-specific profiles should be reviewed against the release that loads them. Restarting the companion does not require replacing the ROBOTIS workspace.

## Troubleshooting by symptom

| Symptom | What to check |
| --- | --- |
| GitHub cannot find `sg2` | Check the exact remote and `git ls-remote --heads` output. Use the local-branch Git bundle route in the [setup guide](README.md#1-get-the-sg2-branch-onto-the-5090). |
| Packaging cannot find `ffw_sg2_follower_initial_positions.yaml` | Initialize `third_party/ai_worker` at the parent gitlink in the 5090 checkout. A parent Git bundle contains the gitlink, while the submodule content comes from its own repository. |
| `Use a new output directory` or `Bundle file changed` | Build to a fresh path. Compare the verified source commit at both ends and copy the complete directory again if it differs. Keep deployment edits in a separate profile. |
| SSH transfer succeeded, but the container cannot find `cosmos_ffw` | Check `pwd` and `ls MANIFEST.json cosmos_ffw` inside the ROS shell. Repeat the host-to-container copy if the container was recreated. Run the module from the directory containing `cosmos_ffw/`. |
| `ModuleNotFoundError` for `rclpy` or `controller_manager_msgs` | Use the follower's sourced Jazzy environment and `/usr/bin/python3`. Check whether the ROS interface package is installed **in that environment**. NumPy and PyYAML are required by the offline check too. |
| Status works on SG2 but times out on the 5090 | Compare the actual domain and RMW. For DDS inspect `ROS_AUTOMATIC_DISCOVERY_RANGE` and `ROS_STATIC_PEERS` on both endpoints. For Zenoh check the router address. A container-local loopback address points to a different machine from the 5090. Recheck the 5090 CLI daemon after changing middleware. |
| `settling` persists, or status says stale feedback | Inspect current `/joint_states` in the follower's ROS shell. Both arms need fresh, aligned position and velocity samples. Current-pose startup needs a stable 0.5 s interval. The companion rejects feedback older than 100 ms or more than 25 ms in the future relative to the robot clock. |
| `ready` in preview but `controllers_ready=false` | Confirm `/controller_manager/list_controllers` is visible and the two arm controllers are active with the required interfaces. Preview readiness alone does not establish controller ownership. |
| `Competing command publisher` or `No controller subscriber` | Inspect **both** configured topics with `ros2 topic info ... --verbose`. Check for teleop publishers, alternate remaps, namespaced controllers, and duplicate companions. |
| `Control message session mismatch` | Pause, resume, and restart rotate epochs. Read a fresh status sample. Inspect `last_rejection` for a rejected request and `reason` for a fault. |
| `needs_chunk=true` with no arm motion | This is the expected preview result. In execute mode it means the companion is holding and waiting for a valid 5090 chunk. Installing the bundle does not start inference. |
