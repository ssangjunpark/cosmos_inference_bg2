# Bridge implementation handoff

Paused at the user's explicit request. Resume only when the user asks.

## Constraints

- All implementation belongs in this local cosmos_inference_bg2 repository.
- 3sp5090 is read-only. No remote edits, installs, container operations that change state,
  homing, or motion commands. This implementation turn made no remote connections.
- Commit major milestones locally; do not push. Keep weights, recordings, and generated
  outputs outside Git. Preserve pinned upstream submodules.
- Use matching cosmos_futurejoint training source and verified output decoding. Ask for
  missing hardware information instead of assuming camera rotations or controller details.

## Completed before this work

- Recorded-observation Edge/Nano step-20000 inference passed in the official NVIDIA base image.
- Twenty CPU tests passed before bridge implementation.
- ROBOTIS dependencies and the read-only inspection profile were committed as 46ee6eb.
- The reference profile deliberately has execution disabled and unknown live camera transforms.

## Draft code saved at pause (not tested)

- scripts/bg2_bridge/core.py: profile/feedback validation, duplicate-root ROBOTIS homing YAML
  parsing, startup/settling/hold capture, timestamp-preserving trajectories and hold messages.
- scripts/bg2_bridge/observations.py: three-camera buffering, 25 ms alignment, explicit camera
  orientation/dimension checks, original PIL crop/resize and head letterboxing.
- scripts/bg2_bridge/ipc.py: same-user Unix socket JSON requests and pickle-free NPZ handoff.
- scripts/bg2_bridge/transport.py: pinned Zenoh SDK subscriptions, explicit publisher activation,
  controller-manager and competing-publisher checks, ROS trajectory conversion.
- scripts/policy_runtime.py: reusable native policy object that loads the model once.
- scripts/cosmos_worker.py: persistent Unix socket model worker.
- scripts/robotis_bridge.py: draft observe/preview/execute/offline-replay CLI and watchdog loop.

This is a work-in-progress checkpoint, not a verified or deployment-ready bridge.
No bridge process or hardware publisher was run. No bridge tests were run.

## Resume next

1. Review the draft and fix known issues before running it:
   - transport.check_ownership currently invalidates its previous good lease while a new
     check is in progress; retain the valid lease until the check completes or fails.
   - Invalidate/fault an active session if new homing starts after capture; current
     Startup.homing_status ignores statuses once ready/running.
   - Review resource cleanup on startup failures, ownership loss, shutdown, stale feedback,
     and partially sent bundles; document controller-side stop limitations honestly.
   - Check native message types/serialization against the pinned SDK using local mocks
     and real CDR serialization, without a robot connection.
2. Wire a cosmos3.py worker command with the existing container isolation and a narrowly
   scoped writable IPC directory. Do not expose robot networking/devices to the GPU worker.
3. Decide whether to share NativePolicy with the existing one-shot runner to avoid duplicated
   loading/generation code. Preserve CLI/config relocation compatibility and saved outputs.
4. Add meaningful tests for camera pixels/timestamps, all homing groups and custom YAML,
   immutable session holds, invalid/expired trajectories, velocity/position limits,
   default preview creating no publishers, ownership failures, worker protocol, and cleanup.
5. Run CPU tests and an entirely offline replay. Then validate the persistent worker locally
   with a free explicitly selected GPU and recorded observations; measure warmed latency.
6. Update README and hardware plan to distinguish implemented, tested, and live-unverified
   behavior. Commit a tested milestone locally. Never deploy or actuate on 3sp5090.

## Live prerequisites still unknown

Raw image dimensions and orientation matching the training MP4 pipeline; installed onboard
bringup/limits, actual selected homing YAML, active controllers and command ownership;
router/live observation availability; warmed runtime compatible with the 0.8 s forecast.
The reference profile must remain non-executable until real compatibility is established.
