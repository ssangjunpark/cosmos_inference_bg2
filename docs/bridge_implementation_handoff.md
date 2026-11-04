# Bridge implementation handoff

Updated 2026-11-04. The user requires synchronous full-chunk execution. The previous
asynchronous scheduling design is superseded by [the agreed execution contract](synchronous_execution_20261104.md).
Use the [three-machine runbook](three_machine_pipeline.md) for topology and startup.

## Current implementation

- PRO 6000: persistent output-only native Cosmos model, existing checkpoint/adapter and
  normalization unchanged. API and SSH endpoints remain localhost-only with no API password.
- 5090: blocking request while FFW holds; all inputs must be captured after the prior hold.
  Wait for completed cycle/request ID before requesting again. No prefetch/overlap remains.
- FFW companion: observe fresh external homing, 500 ms settling, capture passive holds;
  execute every one of the 16 targets at 50 ms spacing; hold the final command afterward.
  Angular planned/measured velocity >2 rad/s or tracking error >10 degrees latches a fault
  and attempts a measured-position stop hold. Do not convert degrees/rad/s into lift limits.
- Old local-IPC live bridge execution was removed. `robotis_bridge.py` is offline replay only.
- Source modules, both transfer bundles and diagrams must describe this same contract.

## Validation and deployment

90 CPU tests: 89 pass and the optional SDK check skips in the default host environment.
The tests exercise real local HTTP plus synthetic robot inputs and mocked ROS boundaries.
They establish software behavior, not physical tracking or stopping. Historical Edge/Nano
recorded-input timings and the earlier async validator reports are not revised-executor
acceptance evidence.

The revised source/bundles are local until separately transferred. The existing installed
5090 image predates this change; FFW has not received our companion. Update both deployment
copies together because the FFW status/handshake changed. The model RPC needs no changes.
Do not push, install/run on FFW, or issue homing/motion without authorization. Commit major
local milestones. Keep generated bundles, weights, observations and outputs ignored.

## Remaining inputs

Camera orientation/FOV, actual instruction, enabled verified hardware profile and physical
tracking/stopping remain unverified. SNUPI remains selected unchanged, but head target 0.78
exceeds installed upper limit 0.6951 and the loader rejects it. Do not silently change the
pose or limits. FFW remains read-only. See the startup review for the previously verified
image, controller configuration, paths and source hashes.
