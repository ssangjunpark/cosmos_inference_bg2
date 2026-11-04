# cosmos_inference_bg2

Local setup for RobotEscape's Cosmos 3 Edge/Nano checkpoints. The endpoint is saved predicted
outputs from a recorded observation. Hardware integration is deferred.

NVIDIA's [policy-server guide](https://github.com/NVIDIA/cosmos-framework/blob/b98934eda43f4d31972538a527f8663f003c27dc/docs/action_policy_droid_server.md)
provides the Dockerfile, `INSTALL_APEX=0` build flag, and exact policy dependency installation.
The checkout in `third_party/cosmos-framework` is pinned to the framework commit in the
checkpoint's `input_contract.json`: `b98934eda43f4d31972538a527f8663f003c27dc`.
Upstream source is unchanged; the root scripts contain the local integration.

The planned physical-robot startup, configurable homing, and left-arm/lift hold behavior are
recorded in the [BG2 startup and inference hold plan](docs/bg2_startup_and_hold_plan.md).

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
python3 scripts/cosmos3.py doctor --gpu 5
```

Choose an available physical GPU explicitly; the scripts do not claim all GPUs.
The first image uses the upstream Dockerfile. `docker/Dockerfile.policy` runs the guide's
`uv sync --all-extras --group=cu130-torch213-train --group=policy-server` command during the
image build so runtime containers can run offline as your UID. No host Python packages are installed.

## RobotEscape checkpoint compatibility

The downloaded checkpoints are custom native PyTorch Distributed Checkpoints, not the public
DROID safetensors policies used by the guide's server example. Keep `model/.metadata`
and every `model/*.distcp` together, including empty replica shards.

The saved configuration names `bg2_cosmos.model.BG2PolicyModel` and `bg2_cosmos.data.RoomLoader`.
The full training pipeline is not an inference requirement. The matching custom model adapter
and exact inference sample preparation are required until their compatibility with the native
base class has been checked. This runner preserves the saved model target.

The recorded contract supplies the camera composition, eight-channel joint order, training
normalization statistics, and 16-step future target horizon. It does not record the complete
native sample/mask schema or custom embodiment ID. Do not substitute the public DROID defaults.

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

For inference, the matching adapter must prepare a **single unbatched native inference sample**
in a pickle-free NPZ archive. Use arrays for native tensor fields and scalar Unicode arrays for
string fields, including the already formatted `ai_caption`. Preserve native dtypes, masks,
embodiment ID, and current-state conditioning. Future target values must be inference placeholders.
The runner follows the upstream policy service's native sample-to-batch conversion.

Once that sample and the adapter/assets are available:

```bash
python3 scripts/cosmos3.py infer \
  --variant edge --step 20000 --gpu 5 \
  --sample inputs/native_sample.npz \
  --adapter-dir /path/to/source-containing-bg2_cosmos \
  --tokenizer /path/to/matching/tokenizer-snapshot \
  --vae /path/to/Wan2.2_VAE.pth \
  --guidance-interval 960 1001
```

For Nano, select `--variant nano` and its matching tokenizer; omit the Edge guidance interval
unless the original evaluation configuration specifies it. `--seed`, `--steps`, `--guidance`,
and `--shift` are configurable; defaults are the upstream server's 0, 4, 3.0, and 5.0.
Add `--dry-run` to review the Docker command before launching it.

Successful runs write a fresh `outputs/<variant>-<step>-<timestamp>-<id>/` directory containing
`request.json`, `outputs.json`, `action_0.npy`, and vision latents when returned by the model.
Actions are saved exactly as returned by the framework; no DROID gripper conversion, controller
mapping, or action execution is applied. These are raw outputs, not validated hardware commands.

## Mounts

Runtime containers have a read-only filesystem, run as your UID/GID, and have networking disabled.
Only the selected checkpoint, tokenizer, VAE, input archive, adapter directory, and local scripts
are mounted read-only. For a Hugging Face tokenizer snapshot, its model cache repository is
mounted read-only to preserve links into the adjacent blobs directory. Writable bind mounts are restricted to this workspace's `.runtime/cache/`
and the fresh output directory. The container does not mount the Docker socket, host home,
serial devices, or robot services. Shared memory uses `--shm-size=16g`, not host IPC.
The upstream entrypoint is bypassed at inference because it attempts an editable install;
the environment was already installed during the image build.

The launcher requires publication markers and checks shard presence and sizes. To recompute
the original SHA-256 digests as well, run `python3 scripts/cosmos3.py status --sha256`.
It does not access the transfer service's `.copy_in_progress` directory.

## Local verification

```bash
python3 -m unittest discover -s tests -v
```

These CPU checks cover incomplete checkpoints, corrupt shards, manifest traversal, Docker
mount/network settings, and rejection of pickle inputs. GPU inference additionally needs the
completed Docker image, matching adapter/assets, and a valid native inference sample.
