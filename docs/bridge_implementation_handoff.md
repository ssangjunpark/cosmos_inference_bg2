# Bridge implementation handoff

Updated 2026-11-04 after the user resumed full local implementation. The three-role
software is implemented; physical deployment and commissioning have not been performed.
Use the [three-machine runbook](three_machine_pipeline.md) for commands and topology.

## Constraints

- All code is in the local cosmos_inference_bg2 repository on PRO 6000.
- The user authorized installing the gateway on 3sp5090. FFW remains strictly read-only:
  no file changes, container changes, homing calls, or motion commands there.
- The user explicitly requested removing the extra API password. RPC now relies on
  localhost endpoints through SSH; do not reintroduce a token file or password check.
- Commit major milestones locally; never push without a new explicit instruction.
- Keep weights, assets, observations, generated bundles, logs and caches ignored.
  Upstream Framework, ai_worker, and Zenoh SDK remain pinned and unchanged.
- Do not guess camera transforms or enabled hardware configuration.

## Implemented entry points

- `scripts/cosmos3.py serve`: protected Docker launch, localhost HTTP port,
  verified DCP/assets, recorded-input warmup. `scripts/inference_server.py` uses one persistent
  model on one GPU thread with bounded requests and no request queue.
- `scripts/gateway.py`: inspect, replay, preview, execute. Align three views and measured
  joints, use original FFW timestamps and local monotonic elapsed times, one async request,
  discard stale/old-epoch replies, prefetch during execution, forward chunks only.
- `deploy/ffw/cosmos_ffw/node.py`: native ROS companion. Fresh homing success for all groups,
  settling, one post-homing hold snapshot, controller/position-interface/ownership gates,
  validated timed chunks to existing controllers, tracking/expiry/stop behavior.
- `scripts/package_gateway.py` and `scripts/package_ffw.py`: small standalone whitelist
  bundles with checksums/source commit. No SSH or transfer; refuse existing output paths.
- `deploy/gateway`: gateway Docker/Compose templates with pinned dependencies.
- `deploy/ffw`: reuse the installed `robotis/ai-worker:2.2.8` image in a separate
  companion container, mounting code/launcher/profile/home YAML read-only. No FFW
  Dockerfile, image build, package install, or automatic pull. Override the inherited
  s6 entrypoint and API healthcheck so only the companion starts. Preview remains default;
  no GPU/serial/socket access. Existing ROBOTIS bringup stays separate.

The older `robotis_bridge.py`, `cosmos_worker.py`, Unix IPC and direct 5090 trajectory
transport are legacy co-located drafts. They are not the deployment path for this topology.

## Validation evidence

- 87 CPU tests: 86 pass and the optional SDK test skips in the default host environment.
  The SDK test passes separately in the isolated gateway environment.
- Real SDK pub/sub passed on loopback with bundled schemas and runtime downloads forbidden,
  also inside the built read-only, network-disabled Python 3.12 gateway container.
- Existing local native Jazzy CDR trajectory serialization checks passed.
- Real step-20000 Edge and Nano warmed services each processed five HTTP
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
  homing, ownership, stale feedback/status, model concurrency, and password-free localhost RPC.
- Local test model containers were stopped after measurements; no model server is left
  running as an unintended deployment.

## Minimal FFW runtime revision

The user requested minimal changes and reuse of existing software. The FFW launcher
now uses the installed image directly instead of building a derived image. The 39
affected FFW tests pass and Docker Compose configuration validation passes locally.
The refreshed generated `transfer/ffw/` folder contains the code and Compose file for later copy;
that FFW-only preparation made no remote changes. The 5090 gateway is now installed;
see [5090 setup record](5090_setup_20261104.md) for verified checks and pending configuration.

## Transfer-folder layout

Ready-to-copy folders now live in `transfer/ffw/` and `transfer/5090/`, not `outputs/`.
They are generated and ignored by Git; tracked source remains under `deploy/`, `scripts/`
and the pinned SDK submodule. `transfer/README.md` explains edit versus copy locations.
Both package scripts default to the new paths. Previous generated bundles are preserved
under `.runtime/bundle_archive/`; inference outputs remain under `outputs/`.

## FFW startup review

The user requested documentation review before FFW setup. A fresh read-only check
found the existing containers up but no follower controller/camera processes. The
installed image supplies companion dependencies; domain 30/rmw_zenoh_cpp and launch/home
file hashes were checked. See [FFW startup review](ffw_startup_review_20261104.md).
Start the preview companion before follower initialization. Follower bringup performs
real homing and has not been started; FFW remains unchanged.

## Selected SNUPI home configuration

The user selected `bg2_vr_startup_snupi.yaml`; the local FFW template and generated
folder now point to that existing FFW file. Its installed/source copies match. The
installed head_joint1 upper limit is 0.6951, while this YAML targets 0.78. Actual
offline home validation rejects it. A user clarification is pending; do not change
the target/limits by assumption or start homing. The robot YAML and launch settings
remain untouched. See the startup review for paths and checksum.

## Remaining physical work

1. Verify raw camera dimensions, orientation and field of view against training preparation.
2. Confirm actual home YAML, installed 2.2.8 controller interfaces/limits and exclusive
   ownership. The earlier inspection found no active robot services. The reference profile
   stays disabled; alternate head target 0.78 violates the reference 0.6951 limit.
3. Use the existing PRO 6000-to-5090 SSH alias for a reverse localhost forward; verify
   sustained timing/jitter. Only transfer the FFW folder when authorized.
4. Start preview companion before fresh ROBOTIS homing, exercise end-to-end chunk acceptance,
   then commission supervised execute mode, tracking, shutdown and starvation behavior.

The user authorized 5090 setup, not modifying FFW or running robot execution. Do not interpret
local software completion as live hardware verification. Edge is the initial deployment
example; Nano timing and the observed gripper-limit rejection need evaluation before motion.
