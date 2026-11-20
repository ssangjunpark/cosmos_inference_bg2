# Run the SG2 WAM policy on the 5090

Use the SG2 companion from the [robot setup guide](README.md). The policy, camera preprocessing, and Zenoh subscriptions run on the 5090. The AI Worker needs only its existing ROS publishers and the SG2 companion. Capture can run as soon as joint feedback and the three cameras are available.

| Variant | Predicted coordinates | Fixed instruction |
| --- | --- | --- |
| `pnp1_normal` and `pnp1_abnormal` | Right arm and gripper, 8 | Pick up the red spool and place it into the empty metal basket. |
| `pnp2_normal` | Both arms and grippers, 16 | Pick up the red spool and place it into the wooden tray on the right. |
| `pnp2_abnormal` | Both arms and grippers, 16 | Move the wooden tray to the right, then place the red spool into it. |

Each variant selects its own q01/q99 statistics. The model conditions on the current measured positions and three RGB views. It predicts absolute **dataset-native measured positions** at 50 ms intervals through 800 ms. Row zero is the conditioned observation. The native batch has 17 rows, 64 padded columns and domain ID 33. For a PNP1 robot chunk, the 5090 repeats all eight measured left-side coordinates from that observation and fills the right side from the prediction. Head, lift and mobile coordinates have no output channels. [Variant settings](../profiles/sg2_wam.json)

## Build and check on a fresh 5090

From the `sg2` checkout, use Docker with NVIDIA Container Toolkit. The following builds the Cosmos runtime at the pinned framework revision and adds the pinned ROBOTIS SDK. A workstation that already has `cosmos-framework:robotescape-b98934e` can skip the Cosmos build.

```bash
git submodule update --init --depth 1 -- third_party/cosmos-framework third_party/zenoh_ros2_sdk
python3 scripts/cosmos3.py build
docker build -f sg2_setup/Dockerfile -t cosmos-sg2-wam:local third_party/zenoh_ros2_sdk
bash sg2_setup/run_policy.sh check --variant pnp2_normal
```

`SG2_DATA` is the input directory on the 5090, mounted read-only at `/data`. `SG2_OUTPUT` defaults to `outputs/sg2` and appears at `/output`. Choose a new output file or directory per run. The optional training handoff at `handoff/sg2_handoff_20261119` appears at `/handoff`. Use `SG2_HANDOFF` to point at a different local copy. Weights, captures, and the original handoff stay outside the Git commit.

```bash
export SG2_DATA=/path/to/local/sg2-inputs
export SG2_GPU=0
for variant in pnp1_normal pnp1_abnormal pnp2_normal pnp2_abnormal; do
  bash sg2_setup/run_policy.sh check --variant "$variant" --handoff /handoff || break
done
```

This CPU check exercises the inference tensor preparation and inverse transform with the training loader's saved 22-coordinate state and 16 future targets. To check the entire supplied package, run `python3 -B handoff/sg2_handoff_20261119/verify_handoff.py`.

## Capture a live observation over Zenoh

On the **SG2**, in the ROS environment where the camera drivers are running, use `ros2 topic list -t` and `ros2 topic info TOPIC --verbose` to identify the three `sensor_msgs/msg/CompressedImage` topics and their physical views. Record the router address reachable from the 5090 and the SG2's ROS domain. These settings come from the installation. A successful DDS-only test of the robot companion does not establish a Zenoh camera route.

On the **5090**, replace the example values below with the observed values. The Bash array keeps left and right assignments the same across capture and preview.

```bash
link=(--router SG2_ROUTER_HOST --port 7447 --domain YOUR_DOMAIN_ID
      --head-topic /YOUR_HEAD_COMPRESSED --left-topic /YOUR_LEFT_WRIST_COMPRESSED
      --right-topic /YOUR_RIGHT_WRIST_COMPRESSED)
bash sg2_setup/run_policy.sh capture --variant pnp2_normal "${link[@]}" --output /output/first.npz
```

Capture subscribes to those cameras and `/joint_states`. It writes one timestamp-aligned observation and exits. Joints are selected by name from the feedback message. Each camera must be within 25 ms of that joint sample. The head must be 672×376 and each wrist 424×240. Inspect the saved views against the installation and training orientation. The head crop is `(112,88,560,376)`, letterboxed to 512×384. Each wrist uses the raw leftmost 240×240 square, rotated 90° counterclockwise and resized to 256×256. The composite places the left wrist below the head on the left.

To use an existing synchronized **dataset** observation directly, put the raw images and a JSON state in `SG2_DATA`. Give the state fields `joint_names`, `positions` (dataset units), and `source_time_ns` (integer nanoseconds, including zero at episode start). All 22 named recorder coordinates are accepted. An observation containing only the selected joints also works.

```bash
bash sg2_setup/run_policy.sh prepare --variant pnp2_normal \
  --head /data/head.png --left-wrist /data/left.png --right-wrist /data/right.png \
  --state /data/state.json --output /output/dataset.npz
```

## Load the selected checkpoint and predict offline

Copy the training exporter’s **weights publication directory** for the chosen variant under `SG2_DATA`. It contains `READY.json`, `config.yaml`, `input_contract.json`, `vision_contract.py`, and `model/` with `.metadata` and all DCP shards. Preserve `VERIFIED.json` for milestone publications. The exporter's `weights/last_NNNNNN` also works with its verified final-snapshot receipt. Keep the model and sidecars from the same publication. The loader checks the source run, sidecar hashes, DCP byte ranges, and the weight checksums when supplied. It loads the inference model state directly, including the trained action heads. [Model loader](../scripts/sg2_wam.py)

Put the Edge tokenizer snapshot (revision `344d602b128d1bbdacb43b08d0a3626f46343e29`) and Wan VAE (`921dbaf3f1674a56f47e83fb80a34bac8a8f203e`) under `SG2_DATA` as well. Any tokenizer symlinks must resolve inside that mount. Set `STEPS`, `CFG` and `SHIFT` to the evaluation settings you choose with the training team. Supply `--guidance-interval LOW HIGH` if their evaluation uses one.

```bash
variant=pnp2_normal
weights=/data/pnp2_normal/weights/010000  # substitute the delivered publication
bash sg2_setup/run_policy.sh check --variant "$variant" --checkpoint "$weights"
model=(--variant "$variant" --checkpoint "$weights" --tokenizer /data/tokenizer
       --vae /data/Wan2.2_VAE.pth --steps "${STEPS:?set STEPS}" --guidance "${CFG:?set CFG}"
       --shift "${SHIFT:?set SHIFT}" --seed 0)
bash sg2_setup/run_policy.sh predict "${model[@]}" --observation /output/dataset.npz --output /output/prediction-001
```

The prediction directory contains named dataset-native actions and timestamps in `joint_positions.json`, native normalized actions in `normalized_actions.npy`, and the reconstructed context and imagined future videos. Compare one prediction against the same observation, checkpoint and sampling settings on the training system.

## Map controller units and run one preview chunk

For controller observations, supply a site calibration JSON with one entry per **selected joint**, each containing numeric `scale` and `offset`. The relation is **controller radians = scale × dataset position + offset**. Provide a nonzero scale and verify both directions for each gripper. An identity mapping also needs confirmation against the recording/export path. A nonlinear gripper mapping requires a corresponding conversion in the adapter before using this interface. To replay `/output/first.npz` offline, add `--calibration /data/calibration.json` to `predict`.

Start the companion in preview as described in the setup guide. From the 5090 run the following with the same `link` and `model` arrays. It waits for a ready session, captures feedback and frames after the start of the hold, predicts and sends one chunk, then exits after the matching completion acknowledgement. `--chunks N` repeats this cycle N times.

```bash
bash sg2_setup/run_policy.sh preview "${model[@]}" "${link[@]}" --calibration /data/calibration.json
```

For an intentional physical session, follow the [execute-mode commissioning procedure](operations.md#later-physical-execution). Select `execute` on both ends and pass `--robot-profile /data/profile.site.json` on the 5090 with the reviewed site profile. The reference profile supports preview. The November 19 handoff includes loader examples, camera preprocessing and model settings. It leaves checkpoint selection, sampling parameters, physical calibration and hardware rollout verification to commissioning.

In the 5090 runtime, all four native batches matched the handoff's current-state tensor, prompt and token count. An isolated Zenoh test sent synthetic camera frames and joint feedback through the SDK, then ran two acknowledged chunks each for PNP1 and PNP2 with a stand-in model. The supplied golden examples contain measured targets. A trained SG2 prediction and a physical rollout still require a selected checkpoint and the actual installation.
