# Inference pipeline implementation plan

Updated 2026-11-04. Finish recorded-observation inference and verify the decoded outputs
before connecting the [hardware startup and hold workflow](bg2_startup_and_hold_plan.md).

The first milestone is one real Edge inference result from a recorded observation, saved
with enough metadata to establish exactly which checkpoint, inputs, transforms, and sampling
settings produced it. Repeat that validation for Nano. Physical robot commands are a later stage.

## Available now

- Cosmos Framework is a clean Git submodule pinned to the checkpoint's recorded commit.
- Edge step 20000 and Nano step 20000 have completed publication checks. Edge step 15000
  is also available; checkpoint availability is reported by `scripts/cosmos3.py status`.
- The exact tokenizer snapshots and shared Wan VAE required by both 20k configs are downloaded.
- A current recorded observation is saved locally; the recorded camera/joint dataset is available.
- The launcher mounts checkpoints and inputs read-only, runs as the user, disables networking,
  and writes results to a fresh directory.
- Matching training source is available at `/home/sangjunpark/Documents/robotescape_training_example_20261104`,
  specifically `training/cosmos_futurejoint`. The required six adapter files are vendored with verified hashes.
- The runner accepts current recorded observations and builds the source-matched native batch.
  Both 20k checkpoints have produced finite 16-by-8 absolute joint predictions on the same recorded observation.

## Implementation order

| Step | Remaining work | Completion evidence |
| --- | --- | --- |
| 1 | **Official base image built and checked.** Policy dependency overlay built; native inference uses the verified base image. | Base and policy images build; verified image provenance, CUDA arithmetic, and real video decoding pass inside the container. |
| 2 | **Complete.** Source verified and native class loaded in Docker. | The saved `bg2_cosmos.model.BG2PolicyModel` imports; its inference behavior, custom embodiment ID, and data schema are established from source. |
| 3 | **Complete.** CPU reference and native Edge/Nano tokenization/batch checks passed. | Camera preprocessing, prompt formatting, normalization, state conditioning, padding, and masks match training; future ground truth is not supplied as conditioning. |
| 4 | **Complete.** CPU inverse checks and both generated-output decodes passed. | Correctly select the predicted rows and eight joint channels; apply the verified inverse transform exactly once; save joint names, units, and forecast times with a finite 16-by-8 result. |
| 5 | **Complete in the official base image.** Both 20k recorded-input runs passed. | Both 20k checkpoints produce real saved predictions for recorded inputs; input/checkpoint identity and sampling parameters are recorded. |
| 6 | **Complete.** Persistent SSH-tunneled HTTP service, one warmed model and GPU thread. | Both real Edge/Nano services passed five local HTTP requests; warm latency is recorded in the three-machine runbook. |
| 7 | **Software complete locally; physical commissioning remains.** 5090 gateway and portable FFW companion. | Local raw-observation → HTTP → gateway → FFW executor tests pass. Actual recorded Edge reply passes FFW validation; no remote deployment or actuation. |

The matching source has been found and integrated. Steps 2–5 passed using NVIDIA's official
base image. The guide's additional policy dependency image has built; native inference and
serving measurements below use the verified official base image.

## Immediate environment fix

The first Docker attempt failed at its last step because the local wrapper supplied
`SOURCE_COMMIT` but omitted `SOURCE_TREE` and `BUILD_TIMESTAMP`. NVIDIA's provenance
validator requires the complete set when any source metadata is supplied.

The wrapper now derives the source tree from Git, checks that the upstream checkout is clean,
and passes the complete metadata with strict provenance enabled. A regression test sends the
wrapper's arguments through NVIDIA's actual validator. The policy image also regenerates
runtime provenance after installing its documented dependency group.

The corrected base build passed verified provenance, CUDA arithmetic, and actual TorchCodec
video decoding on the selected GPU. Both native inference runs also passed in that image.
The additional policy dependency overlay has built. Native model inference and serving are
verified in the official base image; the overlay is not required for this custom DCP API.

## Matching training source and completed integration

Use `/home/sangjunpark/Documents/robotescape_training_example_20261104/training/cosmos_futurejoint`.
The separate `cosmos_futureee` case and older simulation experiments have different action contracts.

The original training manifest hash and complete normalization statistics match both 20k
checkpoints. The supplied framework's model, sequence plan, collation, prompt formatter,
and inference loader files match our pinned upstream checkout byte for byte.

The source confirms:

- `BG2PolicyModel` changes training/validation checks only and inherits native inference.
- Domain 33, eight measured right-arm/gripper channels, padded to 64.
- Current image and current joint row zero are conditioned; action temporal offset is zero.
- Sixteen future joint rows at 20 Hz use zero placeholders during inference.
- Exact original image composition, prompt formatter, and train-only quantile affine, without clipping.

The native generator also requires `ActionProcessingRecord`. The inference adapter supplies
an identity inverse record for eight channels, preserving normalized native output. The
local decoder removes the conditioned row and applies the training inverse affine once.

Twenty CPU tests pass, including exact pixel/current-state comparisons against the actual
training loader at the first, middle, and tail example frames, decoder recovery of recorded
targets, rejection of future labels, and input/joint-order validation. These checks establish
input/decoder agreement. Native tokenization, model loading, generation, and output decoding
have also passed for both variants in the official base image.

## Recorded GPU results (2026-11-04)

NVIDIA's unchanged base image, PyTorch 2.10.0+cu130, RTX PRO 6000 Blackwell GPU 5.
Input: `inputs/recorded_observation.npz`, episode `4_0`, frame zero. Seed 0,
4 sampling steps, guidance 3.0, shift 5.0; Edge guidance interval [960,1001], Nano unrestricted.
These settings follow the upstream service defaults; they are not a tuned BG2 quality benchmark.

| Variant / step | Load (s) | First generation (s) | Peak allocated GPU (GiB) | Saved physical predictions |
| --- | ---: | ---: | ---: | --- |
| Edge / 20000 | 15.44 | 2.99 | 8.02 | `outputs/edge-020000-20261104T072321Z-3d5f49a3/joint_positions.npy` |
| Nano / 20000 | 57.95 | 1.57 | 30.02 | `outputs/nano-020000-20261104T072501Z-2e9bec53/joint_positions.npy` |

Both results are finite float32 arrays of shape [16,8], with preserved current-state conditioning
and JSON joint names, recorded radian units, and forecast offsets 50–800 ms. Request files
record observation/checkpoint identity and complete sampling settings. Logs are under
`.runtime/logs/`. Outputs and logs remain local and ignored by Git.

Generation timings above are first requests and include native vision preparation. The persistent
HTTP service is now implemented and warms before accepting requests. Warm five-request local
HTTP results were Edge 0.231 s median / 0.293 s max and Nano 0.401 s median / 0.440 s max.
The [three-machine runbook](three_machine_pipeline.md) records full serving commands, chunk
timing, deployment boundaries, and remaining physical checks. Actual recorded Edge output
passed the FFW bounds/velocity gate; Nano was rejected for a negative gripper target.
Both model outputs remain unchanged and no robot commands have been published.

## Validation boundaries

Use the recorded-observation path first. Keep model weights, downloaded assets, recordings,
generated predictions, and build logs outside Git. Preserve upstream source and use local
adapters. Keep commits local unless publishing is explicitly requested.

Choose the physical GPU explicitly for runtime checks. Select execution cadence and
prediction prefixes only after measuring latency against the checkpoint's 0.8-second horizon.

## Entry points

- Build images: `python3 scripts/cosmos3.py build`
- Check the environment: `python3 scripts/cosmos3.py doctor --gpu <index>`
- Inspect available checkpoints: `python3 scripts/cosmos3.py status`
- Inspect checkpoint metadata: `python3 scripts/inspect_checkpoint.py --variant edge --step 20000`
- Review current build logs under `.runtime/logs/`.
- Resume physical integration from [BG2 startup and inference hold plan](bg2_startup_and_hold_plan.md).
