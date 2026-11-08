# cosmos_inference_bg2

RobotEscape Cosmos 3 Edge/Nano inference, with a persistent PRO 6000 model service, a thin
5090 observation gateway, and a companion for the existing FFW-BG2 controllers. All code is
prepared here first. The gateway is installed on 3sp5090 and the companion is installed on
FFW; the existing ROBOTIS stack remains separately managed.

For the new WAM checkpoints, use the [offline WAM guide](docs/wam_inference.md).
It generates joint predictions and imagined videos on PRO 6000 from a recorded observation,
with networking disabled and checkpoints mounted read-only.
The [persistent WAM server guide](docs/wam_server.md) connects the same WAM runtime to the
existing gateway API and records timestamped imagined chunks without delaying action replies.
The updated [per-run recording guide](docs/run_recordings.md) describes timestamped chunk
downloads to 5090 and per-chunk/combined MP4 construction after the run stops.

The current control contract is **hold → fresh observation → infer → execute all 16 points
→ hold the last command → repeat**. See the [synchronous execution agreement](docs/synchronous_execution_20261104.md).

Start with the [experiment runbook](docs/experiment_runbook.md): exact commands by machine
for Nano/Edge startup, SNUPI initialization, debug/execution, pause/resume, model switching
and cleanup. The [pipeline reference](docs/three_machine_pipeline.md) provides architecture
and earlier integration details.

NVIDIA's [policy-server guide](https://github.com/NVIDIA/cosmos-framework/blob/b98934eda43f4d31972538a527f8663f003c27dc/docs/action_policy_droid_server.md)
provides the Dockerfile, `INSTALL_APEX=0` build flag, and exact policy dependency installation.
The checkout in `third_party/cosmos-framework` is pinned to the framework commit in the
checkpoint's `input_contract.json`: `b98934eda43f4d31972538a527f8663f003c27dc`.
Upstream source is unchanged; the root scripts contain the local integration.

Inference implementation and recorded verification are tracked in the [inference pipeline plan](docs/inference_pipeline_plan.md).

The planned physical-robot startup, configurable homing, and left-arm/lift hold behavior are
recorded in the [BG2 startup and inference hold plan](docs/bg2_startup_and_hold_plan.md).

ROBOTIS integration source is pinned in `third_party/ai_worker` and
`third_party/zenoh_ros2_sdk`. The SDK revision matches the inspected inference-PC checkout.
The [BG2 reference profile](profiles/robotis_bg2_rev4.reference.json) records discovered
interfaces with execution disabled. Installed execution uses a separate operator-enabled
profile, as described in the experiment runbook. See the historical
[5090 setup record](docs/5090_setup_20261104.md) and [FFW installation record](docs/ffw_setup_20261105.md).


## Clone

Clone this repository with its pinned Cosmos Framework submodule:

```bash
git clone --recurse-submodules <repository-url> cosmos_inference_bg2
cd cosmos_inference_bg2
```

For an existing checkout, run `git submodule update --init --recursive`.
Model weights, downloaded assets, caches, recorded inputs, and generated outputs are local-only.

## Build and check

```bash
python3 scripts/cosmos3.py status
python3 scripts/inspect_checkpoint.py --variant edge --step 20000
python3 scripts/cosmos3.py build
python3 scripts/cosmos3.py doctor --gpu 5 --image base
```

Choose an available physical GPU explicitly; the scripts do not claim all GPUs.
The first image uses the upstream Dockerfile. `docker/Dockerfile.policy` runs the guide's
`uv sync --all-extras --group=cu130-torch213-train --group=policy-server` command during the
image build so runtime containers can run offline as your UID. No host Python packages are installed.
`--image base` selects the unchanged upstream image, verified for this native inference API.
`--image policy` (the default) selects the additional policy dependency image.

## RobotEscape checkpoint compatibility

The downloaded checkpoints are custom native PyTorch Distributed Checkpoints, not the public
DROID safetensors policies used by the guide's server example. Keep `model/.metadata`
and every `model/*.distcp` together, including empty replica shards.

The saved configuration names `bg2_cosmos.model.BG2PolicyModel` and `bg2_cosmos.data.RoomLoader`.
The matching source was supplied in `robotescape_training_example_20261104`, under
`training/cosmos_futurejoint`. Six small, unchanged adapter files are included in
[`adapters/`](adapters/README.md), with source hashes. The full training pipeline is unnecessary.
The saved model class inherits native inference without an inference override.

Source verification establishes domain **33**, eight right-arm/gripper channels padded to
64, one observed composite image, current-state action row zero, and 16 future action rows
at 20 Hz. The adapter reuses the original compositor, prompt formatter, and quantile affine.
The original training manifest and normalization statistics match the checkpoints.

Also provide the tokenizer/processor snapshot and Wan VAE file: the training config references
paths on another machine under `/scratch/t558a02/`. Supply local paths below. Matching snapshot
revisions can be found in the saved configuration. Checkpoint exports are unnecessary for this
runner: it uses NVIDIA's supported `Cosmos3OmniModel.from_pretrained_dcp` API directly, as in
the upstream `examples/inference.py`.

## Supporting assets

The DCP metadata contains only network tensors; tokenizer files and VAE weights are separate.
This helper reads the exact Hugging Face repositories and snapshot revisions from a saved
configuration and fetches only processor/tokenizer files and the named Wan VAE (about 2.8 GB):

```bash
python3 scripts/fetch_assets.py \
  --config /mnt/ssd2/RobotEscapeCkpt/cosmos3_edge/020000/config.yaml
```

It uses `huggingface_hub` and `PyYAML` from the existing Python environment, or can be run in
the framework container. It writes under this workspace's `assets/` and `.runtime/cache/`.
The resulting asset receipt records exact paths for `infer --tokenizer ... --vae ...`.
No base Cosmos network weights are downloaded. Use the corresponding saved Nano configuration
for Nano, and `--tokenizer-only` to reuse the same VAE. Hub access/authentication errors need to
be resolved before running offline; credentials are not written to the scripts or receipts.

## Recorded input and output

A recorded observation is available in the existing validation data. Extract its current frame:

```bash
python3 scripts/prepare_observation.py \
  --data-root /mnt/sata1/RobotEscapeCkpt/external_validation_20261103/data/futurejoint \
  --episode 4_0 --frame 0 \
  --output inputs/recorded_observation.npz
```

This saves only current images, current measured joints, a timestamp, and the instruction.
It is an observation archive, not a native inference batch or a model prediction.

Run inference directly from this recorded observation:

```bash
python3 scripts/cosmos3.py infer \
  --variant edge --step 20000 --gpu 5 --image base \
  --observation inputs/recorded_observation.npz \
  --tokenizer /path/to/matching/tokenizer-snapshot \
  --vae /path/to/Wan2.2_VAE.pth \
  --guidance-interval 960 1001
```

The bundled adapter is mounted automatically. Current measured joints are normalized with
the checkpoint's training-only statistics; all future action rows are zero placeholders.
The native sequence plan conditions only the observed image and action row zero.
Input archives reject extra fields, including future labels, and must use the saved joint order.

For Nano, select `--variant nano` and its matching tokenizer; omit the Edge guidance interval
unless the original evaluation configuration specifies it. `--seed`, `--steps`, `--guidance`,
and `--shift` are configurable; defaults are the upstream server's 0, 4, 3.0, and 5.0.
Add `--dry-run` to review the Docker command before launching it.

Successful runs write a fresh `outputs/<variant>-<step>-<timestamp>-<id>/` directory containing:

- `request.json`: checkpoint file identity, observation hash, adapter hashes, prompt, conditioning, and sampling settings.
- `action_0.npy`: the native normalized 17-by-8 action output, including the conditioned current row.
- `joint_positions.npy` and `joint_positions.json`: 16-by-8 absolute predictions in recorded radians, with joint names and forecast offsets of 0.05 through 0.8 seconds.
- `outputs.json`: array shapes, loading/preprocessing/generation/decoding timings, and peak GPU memory.
- `vision_0.npy` when the framework returns vision latents.

The decoder checks that generation preserved current-state conditioning, drops row zero, and
inverts the training affine once. It applies no clipping or gripper remapping. Hardware
integration remains a separate stage. Both 20k checkpoints passed recorded-input GPU inference
in the official base image. Measured first-generation times and artifact paths are recorded
in the [implementation plan](docs/inference_pipeline_plan.md).

## Mounts

Model runtime containers have a read-only filesystem and run as your UID/GID. One-shot inference
and doctor disable networking; the persistent server publishes only a localhost HTTP port
for an SSH-tunneled connection from the 5090; no API password file is required.
Only the selected checkpoint, tokenizer, VAE, input archive, adapter directory, and local scripts
are mounted read-only. For a Hugging Face tokenizer snapshot, its model cache repository is
mounted read-only to preserve links into the adjacent blobs directory. Writable bind mounts are restricted to this workspace's `.runtime/cache/`
and the fresh output directory. The container does not mount the Docker socket, host home,
serial devices, or robot services. Shared memory uses `--shm-size=16g`, not host IPC.
Runtime launches the image's Python directly; the environment is installed during the image build.
The build wrapper supplies the complete source provenance required by the pinned Dockerfile.
The runtime check validates image identity, CUDA arithmetic, and actual video decoding.

The launcher requires publication markers and checks shard presence and sizes. To recompute
the original SHA-256 digests as well, run `python3 scripts/cosmos3.py status --sha256`.
It does not access the transfer service's `.copy_in_progress` directory.

## Local verification

```bash
python3 -m unittest discover -s tests -v
```

These CPU checks cover incomplete checkpoints, corrupt shards, manifest traversal, Docker
mount/network settings, rejection of pickle inputs and future labels, and decoding without clipping.
When the supplied training bundle is available, they also compare exact pixels and normalized states at the
first, middle, and tail samples against its actual loader. Set `BG2_TRAINING_BUNDLE` if that
bundle is elsewhere; this reference check is skipped when the bundle is absent. GPU inference
additionally needs the completed Docker image and matching assets.

## Source and transfer folders

| Folder | Purpose | Git |
| --- | --- | --- |
| `scripts/` | Shared implementation, inference service and packaging tools. Edit here. | Tracked |
| `deploy/ffw/` and `deploy/gateway/` | Machine-specific source code, launch settings and deployment instructions. Edit here. | Tracked |
| `transfer/ffw/` and `transfer/5090/` | Generated standalone copies to send to the named machine. | Ignored; regenerate from source |
| `outputs/` | Local inference results and measurements. | Ignored |

See [transfer instructions](transfer/README.md). The generated folders include required
shared modules so you can copy one complete folder without the rest of the repository.

## Portable FFW companion

Prepare a small, self-contained folder on the PRO 6000 server for later SFTP:

```bash
python3 scripts/package_ffw.py --output transfer/ffw
```

This command never transfers files or contacts either remote machine. The bundle contains
a native ROS 2 companion, a Compose launcher, a reference profile, and file checksums.
It defaults to preview and reuses the already-installed ROBOTIS image with the companion
code mounted read-only. No image build or package installation is needed on FFW, and
existing ROBOTIS bringup/driver containers stay unchanged.

Read [FFW bundle documentation](deploy/ffw/README.md) for the chunk wire protocol and
request/execution lifecycle. The 5090 forwards complete timestamped chunks; FFW validates
and submits all 16 points to its existing trajectory controllers. A new request starts
only after full execution and the final-command hold. The synchronous gateway and
companion are implemented and tested locally.
The matching gateway folder is prepared with:

```bash
python3 scripts/package_gateway.py --output transfer/5090
```

Warm local HTTP requests measured Edge 0.231 s median / 0.293 s max and Nano
0.401 s median / 0.440 s max over five requests each. Edge recorded output also passed
the earlier async trajectory gate; that is not validation of this revised executor. Nano produced a small negative gripper
target and was correctly rejected against reference limits; outputs are never silently
clipped. Live camera transforms, installed hardware compatibility, and sustained network/robot
behavior remain unverified. The installed 5090 copy predates this synchronous revision until updated. FFW deployment remains pending.
