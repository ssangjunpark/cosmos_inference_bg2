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
  A decoder saves 16-by-8 absolute joint predictions; a real GPU model run remains pending.

## Implementation order

| Step | Remaining work | Completion evidence |
| --- | --- | --- |
| 1 | Finish the Docker environment and run the runtime check. | Base and policy images build; verified image provenance, CUDA arithmetic, and real video decoding pass inside the container. |
| 2 | **Source review and integration complete.** Validate its native runtime in Docker. | The saved `bg2_cosmos.model.BG2PolicyModel` imports; its inference behavior, custom embodiment ID, and data schema are established from source. |
| 3 | **Implemented; CPU reference checks passed.** Verify native tokenization and packing in Docker. | Camera preprocessing, prompt formatting, normalization, state conditioning, padding, and masks match training; future ground truth is not supplied as conditioning. |
| 4 | **Implemented; CPU decoding checks passed.** Verify on generated outputs. | Correctly select the predicted rows and eight joint channels; apply the verified inverse transform exactly once; save joint names, units, and forecast times with a finite 16-by-8 result. |
| 5 | Run Edge and Nano end to end. | Both 20k checkpoints produce real saved predictions for recorded inputs; input/checkpoint identity and sampling parameters are recorded. |
| 6 | Measure runtime and add a persistent inference worker. | Separate loading, preprocessing, generation, and decoding timings; record GPU memory; reuse the loaded model between requests. |
| 7 | Connect the later hardware bridge. | Use verified predictions with the homing/hold coordinator, fresh robot observations, explicit command ownership, and tested trajectory timing. |

The matching source has been found and integrated. Steps 2–4 no longer depend on receiving
training code; the remaining checks require the completed Docker environment and real model outputs.

## Immediate environment fix

The first Docker attempt failed at its last step because the local wrapper supplied
`SOURCE_COMMIT` but omitted `SOURCE_TREE` and `BUILD_TIMESTAMP`. NVIDIA's provenance
validator requires the complete set when any source metadata is supplied.

The wrapper now derives the source tree from Git, checks that the upstream checkout is clean,
and passes the complete metadata with strict provenance enabled. A regression test sends the
wrapper's arguments through NVIDIA's actual validator. The policy image also regenerates
runtime provenance after installing its documented dependency group.

A corrected build is being retried. Successful builds and GPU runtime verification are
still required; passing the CPU regression check alone does not establish a working image.

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
input/decoder agreement; GPU tokenization, model loading, and generation still need validation.

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
