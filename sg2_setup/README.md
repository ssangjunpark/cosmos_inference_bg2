# Set up the 5090 and an FFW-SG2 AI Worker from a fresh checkout

**First-session goal:** from a terminal on the 5090, see a ready SG2 companion, pause it in preview, and see it resume. This exercises communication in both directions without commanding an actuator or loading a model. The trained SG2 policy and its 5090 adapter are a later step.

| Where you are | What belongs there |
| --- | --- |
| **5090 host** | The `sg2` Git branch, one ROBOTIS submodule for packaging, the generated transfer folder, and a Jazzy ROS shell for testing communication. |
| **SG2 host** | A persistent copy of the small companion bundle under the robot login's home directory. SSH and rsync terminate here. |
| **SG2 ROS shell** | The environment that can already see the follower's ROS graph. This is often a shell **inside** the running `ai_worker` container. The companion is a separate Python process in this environment. |

This guide assumes the SG2 already has its ROBOTIS driver, follower configuration and usual bringup. Start with the robot at its intended fixed location. The companion only creates actuator publishers for the two arms and grippers. Immobilizing the chassis belongs to the robot installation. The stock July follower launch also starts steering or swerve controllers, including with `init_position:=false`. Use the installation's fixed-base bringup if you need to start the hardware. [July SG2 follower launch](https://github.com/ROBOTIS-GIT/ai_worker/blob/6961f7e7368329d57ebf2adf0fdba08680a43949/ffw_bringup/launch/ffw_sg2_follower_ai.launch.py)

## 1. Get the SG2 branch onto the 5090

**5090 host, Bash.** Pick an unused checkout directory. On Ubuntu 24.04 the following supplies the packaging tools and the two Python dependencies for offline validation. Keep using `/usr/bin/python3` in these commands so the distro packages are visible.

```bash
sudo apt update
sudo apt install git openssh-client rsync python3-numpy python3-yaml
mkdir -p ~/Documents
cd ~/Documents
git ls-remote --heads https://github.com/ssangjunpark/cosmos_inference_bg2.git sg2
```

**If the last command prints a hash and `refs/heads/sg2`**, clone that branch on the 5090:

```bash
git clone --branch sg2 --single-branch \
  https://github.com/ssangjunpark/cosmos_inference_bg2.git cosmos_inference_sg2
```

**If it succeeds with empty output**, the branch is still local to the machine where it was developed. Use this route for a fresh install before publication. A GitHub authentication or network error is a separate issue.

1. On the **machine holding the local `sg2` branch**, run the following in the existing checkout. Replace the example SSH destination with your 5090 login. The bundle contains the committed parent repository history.

   ```bash
   git bundle create ~/cosmos-sg2.bundle sg2
   git bundle list-heads ~/cosmos-sg2.bundle
   scp ~/cosmos-sg2.bundle YOUR_5090_LOGIN@YOUR_5090_HOST:cosmos-sg2.bundle
   ```

2. On the **5090 host**, clone from that file. If the original checkout is already on the 5090, skip `scp` and use its local bundle.

   ```bash
   cd ~/Documents
   git clone --branch sg2 ~/cosmos-sg2.bundle cosmos_inference_sg2
   ```

**Continue here for either route.** Only the `ai_worker` submodule is needed to build this bundle. Its source is fetched from GitHub even when the parent checkout came from a local Git bundle.

```bash
cd ~/Documents/cosmos_inference_sg2
git branch --show-current
git submodule update --init --depth 1 -- third_party/ai_worker
git submodule status -- third_party/ai_worker
git rev-parse HEAD
```

Expect `sg2` as the branch. The root commit names the submodule revision. Leave it at that revision for reproducible packaging. No checkpoint, GPU runtime, recording, or Cosmos Framework build is required for this procedure.

## 2. Build and verify the transfer folder

**5090 host, from the checkout.**

```bash
/usr/bin/python3 -B scripts/package_ffw.py --robot sg2
(
  cd transfer/sg2 &&
  /usr/bin/python3 -B verify_bundle.py &&
  /usr/bin/python3 -B -m cosmos_ffw.node \
    --profile config/profile.reference.json \
    --initial-positions config/initial_positions.yaml \
    --mode preview --initialization current --check
)
```

The packager reports `transferred: false`. The verifier prints the source commit and number of files checked. For a clean clone, `source_dirty` is `false`. The final check should report `valid: true`, `homing_groups: ["arm_left", "arm_right"]`, and `robot_connection_opened: false`.

```text
transfer/sg2/
  MANIFEST.json                   commit and file digests
  verify_bundle.py                verifies the copy at its current location
  config/profile.reference.json   arm topics, names, limits, execution disabled
  config/initial_positions.yaml   SG2 startup reference and settling tolerances
  cosmos_ffw/                     runnable Python companion
  README.md                       robot interface and wire protocol
```

The packager expects a new output directory. On a repeat installation you can reuse a verified bundle from the intended commit, or build a fresh one with `--output /path/to/new/directory` and set `SG2_BUNDLE` below to that directory.

## 3. Copy exactly one release to the robot host

**5090 host, same terminal.** Replace the SSH login with the actual SG2 host. Use a new release path for a new build. Both hosts need `rsync`. If the robot host lacks it, the `scp` alternative below uses the same directory layout.

```bash
cd ~/Documents/cosmos_inference_sg2
SG2_HOST='YOUR_ROBOT_LOGIN@YOUR_SG2_HOST'
SG2_BUNDLE="$PWD/transfer/sg2"
SG2_RELEASE=$(/usr/bin/python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["source_commit"][:12])' "$SG2_BUNDLE/MANIFEST.json")
SG2_DEST="cosmos-sg2/releases/$SG2_RELEASE"
ssh "$SG2_HOST" "mkdir -p '$SG2_DEST'"
rsync -av "$SG2_BUNDLE/" "$SG2_HOST:$SG2_DEST/"
ssh "$SG2_HOST" "cd '$SG2_DEST' && python3 -B verify_bundle.py && pwd"
```

The trailing slash copies the folder's **contents**. The last command prints a release directory with `MANIFEST.json` at its root. Compare the source commit with step 2. If using `scp`, substitute `scp -r "$SG2_BUNDLE/." "$SG2_HOST:$SG2_DEST/"` for the rsync command, then run the same remote verification.

## 4. Open the release in the SG2 ROS environment

**5090 host, same terminal.** Choose the path that matches how the existing follower is running.

**Containerized follower.** Find the actual container name. If it is `ai_worker`, the following commands copy the release into the running container and open a shell in it. No image build or container restart is involved.

```bash
ssh "$SG2_HOST" 'docker ps --format "{{.Names}}  {{.Image}}"'
SG2_CONTAINER=ai_worker
SG2_INSIDE="/tmp/cosmos-sg2/$SG2_RELEASE"
ssh "$SG2_HOST" "docker exec '$SG2_CONTAINER' mkdir -p '$SG2_INSIDE' && tar -C '$SG2_DEST' -cf - . | docker exec -i '$SG2_CONTAINER' tar -C '$SG2_INSIDE' -xf -"
ssh -t "$SG2_HOST" "docker exec -it -w '$SG2_INSIDE' '$SG2_CONTAINER' bash"
```

The host copy under `~/cosmos-sg2/releases/` persists if the container is recreated. Repeat the host-to-container transfer after recreation. This command uses `tar` on the host and in the container, including when the container mounts `/tmp` as a temporary filesystem. If your installation exposes the host release through a bind mount, you can instead open a shell in its mounted path.

**Native follower on the SG2 host.** Open a host shell in the release from step 3:

```bash
ssh -t "$SG2_HOST" "cd '$SG2_DEST' && exec bash -i"
```

**Inside the chosen SG2 ROS shell.** Source the same Jazzy environment and workspace overlay that the follower uses. Here `/root/ros2_ws` is a common container path. Substitute the installed overlay if different. Check `pwd` before launching the Python module.

```bash
source /opt/ros/jazzy/setup.bash
if test -f /root/ros2_ws/install/setup.bash; then source /root/ros2_ws/install/setup.bash; fi
pwd
ls MANIFEST.json config/profile.reference.json
/usr/bin/python3 -B verify_bundle.py
/usr/bin/python3 -c 'import rclpy, numpy, yaml; from controller_manager_msgs.srv import ListControllers; from trajectory_msgs.msg import JointTrajectory; from sensor_msgs.msg import JointState; from action_msgs.msg import GoalStatusArray; from std_msgs.msg import String'
```

A successful import prints nothing. If an import fails, check that this is the follower's ROS shell and interpreter. The ROS package for `ListControllers` is `ros-jazzy-controller-manager-msgs`. Install missing dependencies in the chosen ROS environment using that installation's normal package procedure.

## 5. Identify the live robot interface before starting the companion

**SG2 ROS shell.** The robot's usual follower bringup must be running to produce feedback. If you are identifying the installed launch, `ros2 launch ffw_bringup ffw_sg2_follower_ai.launch.py --show-args` displays its arguments without starting the robot. The following commands inspect the running follower.

```bash
/usr/bin/python3 -c 'from rclpy.utilities import get_rmw_implementation_identifier as rmw; print(rmw())'
echo "ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-0}"
ros2 control list_controllers --verbose
ros2 topic info /joint_states --verbose
ros2 topic echo /joint_states sensor_msgs/msg/JointState --qos-reliability best_effort --full-length --once
for side in left right; do
  ros2 topic info "/leader/joint_trajectory_command_broadcaster_${side}/joint_trajectory" --verbose
done
```

Record the effective middleware and domain for step 7. The expected controllers are `arm_l_controller` and `arm_r_controller`, both active. Each claims eight `/position` interfaces. Feedback must have a current stamp, unique names, and aligned name, position, and velocity arrays. Check for seven arm joints and `gripper_l_joint1` or `gripper_r_joint1` on each side. Wheel, lift and head feedback can coexist in the message.

The July launch remaps each controller to its `/leader/...` topic. In the verbose topic output, look for the corresponding controller **subscriber** in namespace `/`. If your installation uses the native JointTrajectoryController (JTC) input, check `/arm_l_controller/joint_trajectory` and `/arm_r_controller/joint_trajectory`. Record the actual topic for each arm. See [profile selection and software revisions](operations.md#match-the-profile-to-the-installed-robot) if they differ.

## 6. Start the companion in preview

**SG2 ROS shell, from the release directory.** Keep this terminal open. Current-pose initialization waits for settled arm feedback for 0.5 s. The YAML supplies joint names and settling tolerances. This launch does not issue a homing action.

```bash
/usr/bin/python3 -B -m cosmos_ffw.node \
  --profile config/profile.reference.json \
  --initial-positions config/initial_positions.yaml \
  --mode preview --initialization current
```

In a **second SG2 ROS shell** with the same environment, inspect the node and read a full status sample:

```bash
ros2 node info /cosmos_ffw_companion
ros2 topic echo /cosmos/sg2/status std_msgs/msg/String --full-length --once
```

Expect JSON in the message's `data` field. For a settled, newly started preview with no commands, look for `schema="cosmos_sg2_status_v3"`, `mode="preview"`, `phase="ready"`, `needs_chunk=true`, `physical_publishing=false`, and `command_messages=0`. A brief `settling` phase is normal. `controllers_ready` is a separate observation of the controller-manager service. Preview can report `ready` while that field is false.

## 7. Check the actual 5090 ↔ SG2 link

**New terminal on the 5090 host**, outside SSH. This test needs a Jazzy ROS CLI with the middleware used by SG2. On a fresh Ubuntu 24.04 installation, configure the official ROS apt repository and install `ros-jazzy-ros-base`. Install the matching RMW package if needed. [Jazzy installation](https://docs.ros.org/en/jazzy/Installation/Ubuntu-Install-Debs.html)

```bash
source /opt/ros/jazzy/setup.bash
read -rp 'ROS domain ID observed on SG2: ' ROS_DOMAIN_ID
read -rp 'RMW implementation observed on SG2: ' RMW_IMPLEMENTATION
export ROS_DOMAIN_ID RMW_IMPLEMENTATION
```

- **DDS** (`rmw_fastrtps_cpp` or `rmw_cyclonedds_cpp`). Use the same discovery configuration as the SG2 network. With normal subnet discovery, both hosts need a route on the robot LAN and working multicast. If the installation uses static peers, a discovery server, or a DDS profile, apply its corresponding **5090-side** settings. [Jazzy discovery settings](https://docs.ros.org/en/jazzy/Tutorials/Advanced/Improved-Dynamic-Discovery.html)
- **Zenoh** (`rmw_zenoh_cpp`). Use the existing router endpoint reachable from the 5090. If the site uses a custom session config, use its 5090 client config. For a plain TCP router, the following sets its endpoint in this shell. Use the real router address and port. [Jazzy Zenoh configuration](https://github.com/ros2/rmw_zenoh/blob/jazzy/README.md#configuration)

```bash
# Only for a plain TCP Zenoh deployment.
read -rp 'Reachable Zenoh endpoint, e.g. tcp/ROBOT_LAN_IP:7447: ' SG2_ROUTER
export ZENOH_CONFIG_OVERRIDE="connect/endpoints=[\"$SG2_ROUTER\"]"
```

After changing the 5090's CLI middleware, refresh its local CLI daemon. Then check the actual ROS traffic:

```bash
ros2 daemon stop
ros2 topic info /cosmos/sg2/status --verbose
ros2 topic echo /cosmos/sg2/status std_msgs/msg/String --full-length --once
```

Confirm the **single SG2 companion** is the status publisher and that its `epoch` matches the robot-side preview. SSH success alone only proves the file-transfer path. Finish with the [preview pause/resume round trip](operations.md#prove-both-directions-in-preview). A successful round trip leaves the companion in `holding` with `needs_chunk=true` and zero actuator messages.

This procedure was rehearsed from a fresh Git-bundle clone with the public pinned submodule, a staged rsync copy, and a network-isolated Jazzy container with simulated arm controllers. The preview test reached `ready → paused → holding` with zero actuator messages. The SSH login, live controller graph, physical fixed-base setup, and inter-machine discovery must be established on the actual installation.

Stop preview with **Ctrl+C in the companion terminal**. The [operations guide](operations.md) also covers restarting, troubleshooting, rollback, and the later handoff to a trained 5090 policy.
