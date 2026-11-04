# Cosmos 5090 gateway bundle

This is the CPU-only communication process for the 5090. It receives measured joints,
three compressed cameras, and companion status through ROBOTIS's pinned Zenoh SDK 0.1.8.
It sends prepared observations to the PRO 6000 service through a localhost SSH tunnel,
then forwards complete timestamped JSON chunks to the FFW companion. It creates no
JointTrajectory publishers, controller clients, IK, or low-level control loop.

Prepare the standalone folder on PRO 6000:

```bash
python3 scripts/package_gateway.py --output transfer/5090
```

The builder copies only the required local scripts, pinned SDK package and license,
five Apache-licensed ROS message definitions, reference profile, and deployment files.
Its manifest records source commit and SHA-256 checksums. It contains no secrets,
weights, framework checkout, ROS installation, or recorded images. It never connects
to either remote machine. Do not build Docker directly from this template directory;
build from the generated folder, which includes the shared code and SDK.

For later deployment, use the [three-machine runbook](../../docs/three_machine_pipeline.md)
in the repository. In a copied folder, verify it with `python3 verify_bundle.py`.
Create `logs/`, copy `.env.example` to `.env`, and set your numeric UID/GID,
instruction and camera-verified profile. No API password or credential file is required;
the model endpoint is reached through the existing SSH connection. Use a reverse
forward from PRO 6000 to 5090 as shown in the runbook. The 5090 endpoint is
`http://127.0.0.1:18008`, configured by `COSMOS_SERVER_URL` in `.env`; the PRO
6000 model keeps port `8008`. Choose a new COSMOS_RUN_ID for each run; existing outputs are preserved.

The Docker image uses Python 3.12, exact dependency versions from requirements.lock,
and unchanged SDK source. Local ROS message definitions prevent SDK runtime downloads
and Git configuration changes. Host networking is needed for the FFW router and the
SSH tunnel on the 5090; no inbound gateway port is opened. The root filesystem and
profile mount are read-only. Only `logs/` and temporary files are writable.
There are no GPU, serial-device, privileged, or Docker-socket mounts.

Modes:

- `inspect`: subscriptions only; no inference or chunk/control publishers. Reports raw
  image dimensions and timestamps. It does not establish correct image orientation.
- `preview`: real inference and chunk forwarding to a preview-mode FFW companion, which
  creates no hardware trajectory publishers. Requires verified camera transforms.
- `execute`: requires an explicitly enabled hardware-verified profile and a matching
  execute-mode FFW session. The FFW still applies all local runtime gates.
- `replay`: recorded observation to the HTTP service only; never opens a Zenoh session.

Gateway/companion modes must match. The gateway makes a blocking inference request only
after FFW has completed its preceding trajectory and established the final-command hold.
It captures measured joints and three camera frames at or after `ready_since_ns`; it never
uses a previous prediction as the current state. It forwards all 16 unchanged targets.

No prefetch or request during motion remains. After forwarding, even an old ready status
cannot trigger another request: FFW must acknowledge the matching completed request/cycle.
The updated companion uses status schema `cosmos_bg2_status_v2` and execution contract
`synchronous_full_chunk_v1`. Update both copies together; old async copies are incompatible.
The output-only model RPC remains unchanged. See the synchronous execution contract in
the main repository for stop limits and completion semantics.
