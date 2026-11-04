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
- The native runner can load a supplied model class and a prepared native sample. It is still
  an integration scaffold; raw output tensors are not yet verified physical joint targets.

## Implementation order

| Step | Remaining work | Completion evidence |
| --- | --- | --- |
| 1 | Finish the Docker environment and run the runtime check. | Base and policy images build; verified image provenance, CUDA arithmetic, and real video decoding pass inside the container. |
| 2 | Review and integrate the matching training adapter. | The saved `bg2_cosmos.model.BG2PolicyModel` imports; its inference behavior, custom embodiment ID, and data schema are established from source. |
| 3 | Construct a native inference sample from recorded observations. | Camera preprocessing, prompt formatting, normalization, state conditioning, padding, and masks match training; future ground truth is not supplied as conditioning. |
| 4 | Implement and verify the output decoder. | Correctly select the predicted rows and eight joint channels; apply the verified inverse transform exactly once; save joint names, units, and forecast times with a finite 16-by-8 result. |
| 5 | Run Edge and Nano end to end. | Both 20k checkpoints produce real saved predictions for recorded inputs; input/checkpoint identity and sampling parameters are recorded. |
| 6 | Measure runtime and add a persistent inference worker. | Separate loading, preprocessing, generation, and decoding timings; record GPU memory; reuse the loaded model between requests. |
| 7 | Connect the later hardware bridge. | Use verified predictions with the homing/hold coordinator, fresh robot observations, explicit command ownership, and tested trajectory timing. |

Steps 2–4 depend on the matching training adapter. Environment verification and independent
diagnostics can proceed while that source is being retrieved.

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

## Information needed from the trainer

Provide the original `bg2_cosmos/model.py` and `bg2_cosmos/data.py` with any local imports
they require, or a self-contained inference adapter. The full optimizer and training loop
are unnecessary.

The checkpoint already supplies the framework commit, model architecture settings,
normalization statistics, camera contract, joint names, and 16-step target horizon.
It does not identify the active embodiment slot or reproduce the full input/mask schema.
Do not infer those from the number of action-projector slots or substitute DROID defaults.

The existing older `/mnt/ssd2/sangjunpark/bg2_cosmos` simulation experiment uses different
training code and contracts. It is not established as the adapter for these checkpoints.

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
