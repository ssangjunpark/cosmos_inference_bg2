# Optional post-pause diagnostic for the dual yellowstick setup

Use this on top of the [5090 dual-model guide](5090_dual_yellowstick.md).
**The option is off by default.** Enable it for a run by adding
`--pause-diagnostic` to the terminal 4 gateway command.

For a script that starts continuous recording, runs v2, switches to v1 after
Enter and complete cleanup, then saves everything after a second Enter, use the
[v2 → v1 sequence guide](5090_v2_then_v1.md). Model servers and robot bringup
remain separately managed.

Terminals 1, 2, 3 and 5 keep exactly the same commands. Both models stay loaded,
the FFW follower/companion stay unchanged, and the continuous camera recorder
keeps running. No FFW deployment or model-server restart is needed for this option.

## What happens on Ctrl+C in terminal 4

1. Cancel the normal gateway's inference wait and request the existing measured-position pause.
2. Wait for **`pause_confirmed`**. If the pause is not confirmed, skip the diagnostic.
3. While the robot remains paused, capture fresh measured right-arm/gripper state
   and all three camera images after that acknowledgement. Use the usual camera/joint
   alignment checks and wait for 0.2 seconds of stable right-arm/gripper feedback
   (maximum change 0.01 in the recorded joint units, with a 5-second capture timeout).
4. Request **one new prediction from the selected policy**, using that stopped-state
   observation and the same instruction. If an interrupted earlier request is still
   computing, wait for it to finish before the new request is accepted.
5. Save the diagnostic action chunk and full imagined rollout. **Never send that
   chunk to FFW, execute its actions, or resume the robot.**
6. Finish the normal imagined videos, then write separate **original + diagnostic**
   versions named `*_extra.mp4` and exit the gateway. The models, companion, and
   terminal 5 recorder stay running.

This extra prediction is a forecast from the new measured state after pausing;
it does not reuse the last moving-state observation or just rename its video.

## Terminal 4: enable the option for v1

```bash
cd ~/Documents/cosmos_inference_bg2
COSMOS_PROFILE=./profiles/yellowstick_v1.execute.json \
bash deploy/5090/run_gateway.sh \
  execute --profile /config/profile.json \
  --messages-dir /opt/cosmos_gateway/messages \
  --server-url http://127.0.0.1:18008 --expected-model v1 --timeout 120 \
  --pause-diagnostic \
  --instruction "Move the yellow stick into the basket." \
  --output "/output/execute-yellowstick-v1-step18420-$(date -u +%Y%m%dT%H%M%SZ)"
```

## Terminal 4: enable the option for v2

```bash
cd ~/Documents/cosmos_inference_bg2
COSMOS_PROFILE=./profiles/yellowstick_v2.execute.json \
bash deploy/5090/run_gateway.sh \
  execute --profile /config/profile.json \
  --messages-dir /opt/cosmos_gateway/messages \
  --server-url http://127.0.0.1:18009 --expected-model v2 --timeout 120 \
  --pause-diagnostic \
  --instruction "Move the yellow stick into the basket." \
  --output "/output/execute-yellowstick-v2-step18340-$(date -u +%Y%m%dT%H%M%SZ)"
```

Remove `--pause-diagnostic` to get the existing behavior with no extra inference.
The flag is only supported in normal `execute` mode, not `--debug`, preview,
replay or continuous camera recording.

## Where the extra results are saved

The usual run folder is:

```text
~/Documents/cosmos_inference_bg2/outputs/gateway/<RUN_ID>/
```

Extra previews are saved beside the normal imagined videos:

| Extra video | Contents |
| --- | --- |
| `imagined/head_extra.mp4` | Original `head.mp4` followed by the extra 16-frame head forecast |
| `imagined/wrist_left_extra.mp4` | Original left-wrist video followed by its extra forecast |
| `imagined/wrist_right_extra.mp4` | Original right-wrist video followed by its extra forecast |
| `imagined/rollout_extra.mp4` | Original `rollout.mp4` followed by the extra 16 future frames |
| `imagined/rollout_full_extra.mp4` | Original composite rollout, then diagnostic context + 16 future frames |

The original `imagined/head.mp4`, `rollout.mp4`, other normal previews, chunk
manifest and normal timeline keep their original contents. The **separate
`*_extra.mp4` versions contain original + extra**. The diagnostic-only clips
remain available under `diagnostics/pause/artifacts/`.

Diagnostic evidence is separate from the normal `predictions/` directory:

```text
diagnostics/pause/
  diagnostic.json          status, model, pause epoch, timestamps; executed=false
  inputs/
    observation.npz        stopped measured state and processed camera inputs
    observation.json       joint values, joint order and original image timestamps
    input_composite.png    exact composite supplied to the model
    head.png, wrist_left.png, wrist.png
    raw_*                  selected original compressed camera frames
  prediction.json          returned 16×8 actions; submitted_to_ffw=false
  joint_positions.npy      those same actions as a numeric array
  artifacts/               collected forecast data, metadata and rendered clips
```

The model service also retains its raw prediction under the selected v1/v2
server's existing run/request folders, linked by the diagnostic's epoch and
request ID. Normal per-run recordings and terminal 5's continuous recording
retain their existing locations.

## Completion and cancellation

Press **Ctrl+C once** and let `pause_diagnostic_saved` and the normal
`recordings_saved` cleanup finish before starting another experiment.
Starting a fresh terminal 4 command then resumes normally from measured state.

The diagnostic uses `--timeout` for its model wait and `--recording-wait` for
artifact readiness. Explicit "model busy" refusals can be retried; an uncertain
network/inference failure is not automatically retried as a new prediction.

A second Ctrl+C cancels the extra prediction wait or export, and also skips or
defers the usual recording cleanup. Already saved inputs/actions/raw artifacts
remain available. A failed or cancelled diagnostic is recorded in
`diagnostics/pause/diagnostic.json`; it never changes the robot's paused state.

SIGTERM, an ordinary duration limit, and unexpected gateway failures keep the
existing cleanup behavior and do not trigger this optional Ctrl+C diagnostic.

This mode is verified with fake inference/control endpoints and CPU media checks.
Live post-pause inference and robot execution are not started by implementation
or documentation checks.
