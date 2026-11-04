# Cosmos 5090 gateway bundle

This is the CPU-only communication process for the 5090. It receives measured joints,
three compressed cameras, and companion status through ROBOTIS's pinned Zenoh SDK 0.1.8.
It sends prepared observations to the PRO 6000 service through a localhost SSH tunnel,
then forwards complete timestamped JSON chunks to the FFW companion. It creates no
JointTrajectory publishers, controller clients, IK, or low-level control loop.

Prepare the standalone folder on PRO 6000:

```bash
python3 scripts/package_gateway.py --output outputs/gateway_bundle
```

The builder copies only the required local scripts, pinned SDK package and license,
five Apache-licensed ROS message definitions, reference profile, and deployment files.
Its manifest records source commit and SHA-256 checksums. It contains no secrets,
weights, framework checkout, ROS installation, or recorded images. It never connects
to either remote machine. Do not build Docker directly from this template directory;
build from the generated folder, which includes the shared code and SDK.

For later deployment, use the [three-machine runbook](../../docs/three_machine_pipeline.md)
in the repository. In a copied folder, verify it with `python3 verify_bundle.py`.
Create `logs/`, copy `.env.example` to `.env`, set your numeric UID/GID and instruction,
and supply the matching PRO 6000 `inference.token` with mode 600. The token is a separate
file mount; never put its contents in Compose or Git. Supply a camera-verified profile
for preview. Choose a new COSMOS_RUN_ID for each run; existing outputs are preserved.

The Docker image uses Python 3.12, exact dependency versions from requirements.lock,
and unchanged SDK source. Local ROS message definitions prevent SDK runtime downloads
and Git configuration changes. Host networking is needed for the FFW router and the
SSH tunnel on the 5090; no inbound gateway port is opened. The root filesystem and
profile/token mounts are read-only. Only `logs/` and temporary files are writable.
There are no GPU, serial-device, privileged, or Docker-socket mounts.

Modes:

- `inspect`: subscriptions only; no inference or chunk/control publishers. Reports raw
  image dimensions and timestamps. It does not establish correct image orientation.
- `preview`: real inference and chunk forwarding to a preview-mode FFW companion, which
  creates no hardware trajectory publishers. Requires verified camera transforms.
- `execute`: requires an explicitly enabled hardware-verified profile and a matching
  execute-mode FFW session. The FFW still applies all local runtime gates.
- `replay`: recorded observation to the HTTP service only; never opens a Zenoh session.

Gateway/companion modes must match. A stale status, new homing epoch, late result, or
missing observation blocks forwarding. One request runs at a time, using the newest
aligned observations. The FFW keeps original action deadlines, replaces remaining
trajectory points, and holds when a chunk ends. The 5090's clock measures elapsed time;
its wall clock and the PRO 6000 wall clock never replace robot observation timestamps.
