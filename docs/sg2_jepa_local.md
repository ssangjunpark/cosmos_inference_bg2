# SG2 Cosmos / JEPA local integration

This local integration implements pre-execution model selection and replay. It
has not commissioned SG2 hardware or measured model quality, GPU memory or latency.

## Offline demonstration

From this repository, using Python with numpy and Pillow installed:

```sh
PYTHONPATH=scripts python3 -m bg2_bridge.sg2_replay --demo --output outputs/sg2-jepa-demo
```

Open `outputs/sg2-jepa-demo/index.html`. It embeds animated observations and selected
predictions, raw loss, threshold, selected model, reason and processing time. The
synthetic demo uses loss 0.7 and threshold 0.5 to demonstrate fallback. It starts
no services and sends no robot commands. `report.json` records the decision;
`selection.json` records the chosen actions. A failed replay removes an earlier
selection file. Use separate output directories to preserve previous results.

## Real recorded observations

Copy `configs/sg2_jepa.example.json` to a local configuration. The configured
raw-loss **threshold is 2.0**, as selected by the user; it has not been calibrated.
Fill **joint_names** and **units** explicitly; the example does not invent SG2
coordinates. Set the observation and prediction
frame interval in nanoseconds. Paths such as `~/checkpoints/sg2-primary` are
expanded only when launching models; partial checkpoint transfers do not prevent
fixture development. `models[].command` is an argv array, with `{checkpoint}`
substituted as one argument, without a shell. Model services can instead be started
independently. `--launch-models` starts explicit commands and stops them on exit;
services must be ready before replay (startup errors discard the request).

```sh
PYTHONPATH=scripts python3 -m bg2_bridge.sg2_replay \
  --config /path/to/local-config.json --observation ~/recordings/observation.npz \
  --output outputs/sg2-recorded
```

NPZ fields: `past` is nonempty uint8 THWC RGB from one selected camera;
`joint_position` is the current vector in configured order; `joint_names` is that
order; `source_time_ns` and `frame_interval_ns` are scalars. Use camera-consistent
crop/orientation for both observed and predicted videos. All frames must share
one interval. Future clips exclude the generated reconstruction of the current
frame. Align and resample upstream; this integration rejects mismatched intervals.
Both sampled past and future lengths must align to the JEPA tubelet size; their
combined length must fit the configured model capacity. No silent trimming occurs.

For fixture policies, add `--fixture-primary primary.json --fixture-fallback
fallback.json --fixture-loss 0.7`. Fixture JSON uses the policy response fields
below; replay supplies epoch, request ID, model ID and source timestamp. This
mode tests software contracts only.

## Service interfaces

All HTTP addresses must be localhost (SSH tunnels are suitable for the 5090).
The gate timeout is explicit and applies to each HTTP call. There is no background
fallback after timeout. Errors discard the current chunk.

The SG2 policy adapter posts JSON to `/predict`:

```json
{"observation": {"past": [], "joint_position": [], "joint_names": [], "source_time_ns": 123, "frame_interval_ns": 50000000}, "epoch": "session", "request_id": "request", "model_id": "sg2-primary"}
```

The model service returns matching `epoch`, `request_id`, `model_id`,
`source_time_ns`, plus `joint_names`, `units`, `frame_interval_ns`, finite
`positions` shaped `[steps,joints]`, and uint8-compatible `future` shaped
`[frames,height,width,3]`. Its checkpoint loader must implement the new SG2
training contract. **The existing BG2 `/predict` NPZ service is not this interface**;
it fixes eight BG2 coordinates. Do not point the SG2 JSON adapter at it. No SG2
checkpoint architecture or control mapping is inferred from the BG2 service.

JEPA `/score` accepts `past`, `future`, `observation_id` (request ID), `model_id`;
it returns matching IDs, `loss` (nonnegative raw L1, lower is better), `seconds`
and scorer `identity`. The implementation lives in sibling `project-lg-wm`:

```sh
# In project-lg-wm's configured environment, following its .env/uv setup:
PYTHONPATH=src uv run python -m lgwm.judges.future_service \
  --config configs/jepa_future_service.example.json
```

Edit its example with the actual local JEPA checkpoint, external upstream vjepa2
checkout, model size, device and preprocessing. Teacher-space B/L checkpoints
also require `teacher_checkpoint`; self-space vitg in the example does not.
Checkpoint presence and weight compatibility are checked at service startup.
The scorer uses one explicit observed/generated boundary, not the old sliding
window minimum. Encoder/predictor weights stay fp32; CUDA forwards use autocast.

## Gateway integration

`Gateway(..., selection_gate=gate)` enables selection immediately before the
existing readiness/session/cycle recheck and `send_chunk`. Call `from_config`
to build the gate. The observation provider must additionally supply the aligned
`past` and `frame_interval_ns` fields above. The existing BG2 observation collector
does not collect this history and remains unchanged. SG2 ROS collection and motion
transport remain follow-up work; local replay does not import either.

Every request evaluates the primary model. `loss <= threshold` selects it;
`loss > threshold` selects fallback on the same observation and IDs without a
second JEPA gate. The next request starts from primary again. Nonfinite loss,
JEPA timeout/error, invalid action/video contracts and fallback failure discard
the chunk. Gateway readiness is rechecked after the entire selection operation,
so a session change during inference/scoring sends nothing and the robot remains
in its existing hold. Existing behavior is unchanged when no gate is supplied.

## Checks

```sh
python3 -m unittest discover -s tests -p test_jepa_gate.py -q
python3 -m unittest discover -s tests -p test_gateway.py -q
```

Sibling `project-lg-wm` includes `tests/test_future_gate.py` and the existing
`tests/test_judges_surprise.py` CPU regression tests. These checks establish
software behavior using synthetic towers and services, not physical safety or
success prediction.

Local verification on 2026-11-20: SG2 gate/replay tests 8 passed; existing gateway
suite 21 passed, 1 optional SDK import skipped; sibling JEPA suites 55 passed,
1 real-upstream-checkout test skipped. JEPA tests used an isolated temporary
CPU torch 2.5.1 environment rather than the production locked environment.
The broader inference suite (226 tests) had 2 environment errors: unavailable
`ffmpeg` and missing local assets for the existing WAM dry-run. PyAV was supplied
from an existing temporary dependency directory for that broad check. No real
Cosmos/JEPA checkpoints, GPU runs or robot commands were used in this validation.
