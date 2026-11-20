# SG2: resident CM1/CM2, JEPA gate and persistent CM2 execution

For a single config containing all three checkpoint paths and threshold, use the
[unified config and shell launcher](jepa_config.md).

This is the live dual-model path. It replaces the per-chunk primary retry policy
for configurations using `selection_mode: parallel_latched`. The older
[local replay guide](../docs/sg2_jepa_local.md) describes the legacy JSON adapter.

## Runtime behavior

The supervisor loads CM1 and CM2 serially, then keeps both in separate processes.
Before switching, every chunk submits the **same current three-view observation
and joint state** to both services concurrently. Both generate actions and decode
video. JEPA receives actual camera history and CM1's generated future. Raw loss
`<= 2.0` selects CM1; loss `> 2.0` selects the already-generated CM2 chunk.

CM2 mode stays active until the gateway process is restarted. After switching,
only CM2 is requested; CM1 and JEPA are no longer on its execution path.
The gateway also stops waiting for JEPA history; the current camera frame is
sufficient for monitoring once latched. CM2 is
not gated a second time. A CM2 error still discards the chunk and ends the live
run. Session/cycle changes, invalid coordinates and stale inputs still prevent
forwarding. A mode switch never creates a new gateway or robot session.

The default JEPA view is the processed **head camera**. The gateway samples two
real frames 50 ms apart, ending at the current observation, all captured after
the current hold began. Camera gaps cause waiting, not repeated or fabricated
frames. Generated frame zero is reconstructed context and is excluded; the next
16 frames are the target. Both clips use the same configured view and 256x192
size before JEPA's own 384x384 preprocessing. Cosmos still conditions on all three
full processed camera views. Wrist-only judging can be selected explicitly via
`video_view: wrist_left` or `wrist`. This judges one view per run.

Inference and JEPA scoring take place while the companion holds. Chunk execution
and matching completion acknowledgements remain synchronous; this adds no
prediction/execution overlap or guarantee of continuous physical movement.

## 1. Configure paths, checkpoints and sampling

Copy `configs/sg2_jepa.example.json` to your data directory as `sg2-jepa.json`.
The template threshold is **2.0**, user-selected and uncalibrated. Its example
pair is `pnp2_normal` / `pnp2_abnormal`; choose your actual `primary_variant` and
`fallback_variant` before running. PNP2 abnormal has a different trained task
instruction from PNP2 normal. No equivalence of their task goals is assumed.
For PNP1, both variants must be the selected PNP1 pair; the service returns all
16 controller coordinates, holding the measured left arm as usual.

Fill these fields using your actual publication locations and evaluation setup:

- `models[].checkpoint`: the two complete SG2 publication directories.
- `model_runtime.tokenizer`, `vae`, `calibration`: the runtime assets and verified
  controller/dataset conversion. Both services use the same physical calibration.
- `model_runtime.sampling.steps`, `guidance`, `shift`: required; nulls reject launch.
- IDs, URLs and ports must agree (defaults CM1:8090, CM2:8091, JEPA:8092).

For direct execution in an already configured Cosmos environment, paths may be
`~/checkpoints/...`. The supervisor expands them when constructing service argv.
For Docker, set `SG2_DATA` to the host directory containing the publications and
assets, and use **container `/data/...` paths** in this config. A host file at
`$SG2_DATA/sg2-jepa.json` appears as `/data/sg2-jepa.json`. The container cannot
resolve host home paths; its home is `/tmp`. Use `/output` for gateway reports.
The existing [policy setup](policy.md) describes runtime image construction,
checkpoint sidecars, calibration and camera transforms.

## 2. Start the two Cosmos services (terminal A)

```sh
export SG2_DATA="$HOME/checkpoints"  # use the actual data root
export SG2_GPU=0
# Check argv only; does not load weights or open robot transport:
bash sg2_setup/run_policy.sh serve-pair --config /data/sg2-jepa.json --dry-run
# Start both models and keep them resident:
bash sg2_setup/run_policy.sh serve-pair --config /data/sg2-jepa.json
```

Wait for `sg2_pair_ready`. The pair supervisor never starts the gateway and
terminates its own model processes on Ctrl+C. If one service exits, it reports
the failure while preserving the other service; CM1 failure therefore does not
stop a gateway already latched to CM2. Direct host equivalent:
`python3 scripts/serve_sg2_pair.py --config ~/checkpoints/sg2-jepa.json`.
Models are loaded one at a time; inference requests run concurrently in their
separate processes on the configured GPU. Dual residency and concurrent GPU
memory/latency still need measurement with the actual weights.

## 3. Start JEPA (terminal B)

In sibling `project-lg-wm`'s configured environment, fill the JEPA checkpoint and
upstream checkout paths in `configs/jepa_future_service.example.json`, then run:

```sh
set -a; source .env; set +a
PYTHONPATH=src uv run python -m lgwm.judges.future_service \
  --config configs/jepa_future_service.example.json
```

The existing service stays separate from Cosmos dependencies. Use the same
host network (or appropriate localhost SSH tunnels). Its default preprocessing
has temporal stride 1 and source FPS 20, matching SG2's 50 ms forecast grid.

## 4. Start companion, then the gateway (terminal C)

Start the robot companion in **preview** following [operations](operations.md).
Use the installation's router, ROS domain and three camera topics; the following
values must be replaced with the observed installation values:

```sh
export SG2_DATA="$HOME/checkpoints"
link=(--router YOUR_ROUTER --port 7447 --domain YOUR_DOMAIN
      --head-topic /YOUR_HEAD_COMPRESSED --left-topic /YOUR_LEFT_COMPRESSED
      --right-topic /YOUR_RIGHT_COMPRESSED)
bash sg2_setup/run_policy.sh jepa-preview --variant pnp2_normal \
  --jepa-config /data/sg2-jepa.json --selection-output /output/sg2-jepa \
  "${link[@]}" --chunks 0
```

The variant must match `primary_variant`. `--chunks 0` continues until interrupted;
use `--chunks 1` for one acknowledged chunk. Preview submits chunks to the preview
companion without physical motion. Physical operation uses `jepa-execute` with
an execute-mode companion and the reviewed `--robot-profile /data/profile.site.json`,
following the existing commissioning procedure. No robot execution is performed
by the model or JEPA startup commands.

The live dashboard is `$SG2_OUTPUT/sg2-jepa/index.html` (default host path
`outputs/sg2/sg2-jepa/index.html`). It refreshes every two seconds and displays
observed/selected videos, loss, threshold, selected model, the latched mode and
`selected` / `submitted` / `completed` / `discarded` states. `events.jsonl` retains
decision history and `report.json` the latest report. After switching, `loss` is
null for unscored chunks; `switch_loss` preserves the triggering score. Camera
arrays stay on the gateway/service side and are never published in robot chunks.

## Verification

```sh
python3 -m unittest discover -s tests -p test_sg2_dual.py -q
python3 -m unittest discover -s tests -p test_sg2_policy.py -q
python3 -m unittest discover -s tests -p test_gateway.py -q
python3 -m unittest discover -s tests -p test_jepa_gate.py -q
```

CPU tests exercise concurrent calls, latching across later requests/epochs,
CM1/JEPA outages after switching, invalid loss/CM2 failure, camera history gaps
and hold boundaries, native binary video/action RPC, model health contracts and
same-gateway acknowledged chunk execution. They do not establish model quality,
GPU capacity, latency, physical tracking or a calibrated threshold.

Local verification (2026-11-20): 12 new dual-mode tests passed, plus 33 existing
SG2/gateway/legacy-gate tests; 2 optional dependency tests skipped. A separate CPU
smoke connected two native policy HTTP services to the actual JEPA scoring HTTP
service using tiny substitute towers, observed raw loss 3.0 trigger CM2, and
verified the following request remained on CM2. Model-service argv dry-run, CLI
help, shell syntax and lint checks passed. Broader regression still encounters
the pre-existing missing `ffmpeg` and local WAM asset errors (PyAV supplied from
a temporary dependency directory). No actual checkpoints, GPU inference or robot
commands were used for this verification.
