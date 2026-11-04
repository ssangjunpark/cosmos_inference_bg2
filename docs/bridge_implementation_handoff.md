# Bridge implementation handoff

Updated 2026-11-04 after the user resumed full local implementation. The three-role
software is implemented; physical deployment and commissioning have not been performed.
Use the [three-machine runbook](three_machine_pipeline.md) for commands and topology.

## Constraints

- All code is in the local cosmos_inference_bg2 repository on PRO 6000.
- 3sp5090 and FFW remain strictly read-only. This implementation made no remote connections,
  transfers, installs, container changes, homing calls, or motion commands.
- Commit major milestones locally; never push without a new explicit instruction.
- Keep weights, assets, observations, generated bundles, logs, caches, and token files ignored.
  Upstream Framework, ai_worker, and Zenoh SDK remain pinned and unchanged.
- Do not guess camera transforms or enabled hardware configuration.

## Implemented entry points

- `scripts/cosmos3.py serve`: protected Docker launch, localhost HTTP port, private token,
  verified DCP/assets, recorded-input warmup. `scripts/inference_server.py` uses one persistent
  model on one GPU thread with authenticated bounded requests and no request queue.
- `scripts/gateway.py`: inspect, replay, preview, execute. Align three views and measured
  joints, use original FFW timestamps and local monotonic elapsed times, one async request,
  discard stale/old-epoch replies, prefetch during execution, forward chunks only.
- `deploy/ffw/cosmos_ffw/node.py`: native ROS companion. Fresh homing success for all groups,
  settling, one post-homing hold snapshot, controller/position-interface/ownership gates,
  validated timed chunks to existing controllers, tracking/expiry/stop behavior.
- `scripts/package_gateway.py` and `scripts/package_ffw.py`: small standalone whitelist
  bundles with checksums/source commit. No SSH or transfer; refuse existing output paths.
- `deploy/gateway` and `deploy/ffw`: Compose/Docker templates, preview defaults, read-only
  roots/config/secrets, no GPU/serial/socket access. Existing ROBOTIS bringup stays separate.

The older `robotis_bridge.py`, `cosmos_worker.py`, Unix IPC and direct 5090 trajectory
transport are legacy co-located drafts. They are not the deployment path for this topology.

## Validation evidence

- 87 CPU tests: 86 pass and the optional SDK test skips in the default host environment.
  The SDK test passes separately in the isolated gateway environment.
- Real SDK pub/sub passed on loopback with bundled schemas and runtime downloads forbidden,
  also inside the built read-only, network-disabled Python 3.12 gateway container.
- Existing local native Jazzy CDR trajectory serialization checks passed.
- Real step-20000 Edge and Nano warmed services each processed five authenticated HTTP
  recorded-input requests, returning finite 16×8 absolute radian positions. Original
  observation timestamps are preserved. Median/max: Edge 0.231/0.293 s, Nano 0.401/0.440 s.
- Actual Edge reply passes the FFW trajectory validator at worst observed local latency
  with nine future points. Nano includes a -0.0081 rad gripper value and is correctly
  rejected against reference limits. No model-output clipping or silent coordinate change.
- Added actual loopback TCP refusal/drop and HTTP timeout tests, fresh-observation recovery,
  and mocked ROS-node feedback/controller-health loss checks. Physical SSH/LAN disconnects
  and actuator stopping remain unverified; see the runbook failure-case matrix.
- Synthetic three-camera/joint observations traverse the actual HTTP service, asynchronous
  gateway, and FFW chunk executor. Negative cases cover clocks, expiry, old epochs,
  homing, ownership, stale feedback/status, model concurrency, and authentication.
- Local test model containers were stopped after measurements; no model server is left
  running as an unintended deployment.

## Remaining physical work

1. Verify raw camera dimensions, orientation and field of view against training preparation.
2. Confirm actual home YAML, installed 2.2.8 controller interfaces/limits and exclusive
   ownership. The earlier inspection found no active robot services. The reference profile
   stays disabled; alternate head target 0.78 violates the reference 0.6951 limit.
3. Supply the PRO 6000 SSH destination from 5090, transfer the reviewed small bundles/token
   only when authorized, and measure real network timing and sustained jitter.
4. Start preview companion before fresh ROBOTIS homing, exercise end-to-end chunk acceptance,
   then commission supervised execute mode, tracking, shutdown and starvation behavior.

The user has not authorized transfer or modifying either remote machine. Do not interpret
local software completion as live hardware verification. Edge is the initial deployment
example; Nano timing and the observed gripper-limit rejection need evaluation before motion.
