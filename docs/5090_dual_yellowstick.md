# 5090: keep yellowstick v1 and v2 ready, with continuous camera recording

This mode loads both Edge WAM policies into the 5090's VRAM once. Terminal 4
selects the policy for each run; changing policy does not restart either model.
To save one additional, never-executed forecast after Ctrl+C, see the optional
[post-pause diagnostic guide](5090_pause_diagnostic.md). It is off by default
and only changes the terminal 4 command.
For an Enter-controlled v2 → v1 sequence that also starts and finalizes the
continuous recorder, use the [sequence guide](5090_v2_then_v1.md). The models
and robot processes are still started separately.
The FFW commands in terminals 2 and 3 are exactly the same as in the
[single-model guide](5090_local_inference.md).

Use five terminals on **5090** (`ssh 3sp5090` from another computer). Start the
models in terminal 1 and bring up the robot in terminals 2–3. Start terminal 5
before the first experiment to capture the whole session, then use terminal 4.

For a new checkout or missing generated profiles and metadata, use the
[one-time preparation commands](5090_local_inference.md#one-time-yellowstick-preparation).

If switching from a single-model session, Ctrl+C its gateway first, wait for
`pause_confirmed` and recording cleanup, then Ctrl+C the old model server.
The paired launcher requires ports 18008 and 18009 to be free; it does not stop
existing services. A healthy follower and companion can remain running.

## 1. Load both models — terminal 1, on 5090

```bash
cd ~/Documents/cosmos_inference_bg2
bash deploy/5090/run_yellowstick_pair.sh
```

The launcher checks for at least 26 GiB free on GPU 0, starts v1, waits for its
warmup, then starts and warms v2. Both remain resident. Wait for
**`"event": "dual_ready"`**, which means both model endpoints are ready; the
first individual `ready` message is only one model.

| Model | Step | Local endpoint | Container |
| --- | --- | --- | --- |
| v1 | 18420 | `http://127.0.0.1:18008` | `cosmos-bg2-wam-edge-v1-server` |
| v2 | 18340 | `http://127.0.0.1:18009` | `cosmos-bg2-wam-edge-v2-server` |

Keep this terminal open. Ctrl+C stops the two containers created by this launcher.
If either model fails during startup or exits later, the launcher cleans up its
other model too. The separate recorder and FFW processes are not stopped by it.

To inspect the two Docker commands without loading anything:

```bash
cd ~/Documents/cosmos_inference_bg2
bash deploy/5090/run_yellowstick_pair.sh --dry-run
```

The deployment settings are in `deploy/5090/yellowstick_pair.json`. CPU tests and
command dry runs validate the setup. Simultaneous GPU loading and live dual-model
operation have not been tested by this implementation session. The 26 GiB check
is a startup estimate based on prior single-model measurements, not a guarantee
against running out of memory.

## 2. Start the follower and cameras — terminal 2

From **5090**, connect to the **FFW host** and enter its existing container:

```bash
ssh ffw
docker exec -it ai_worker bash
```

Now **inside `ai_worker`**:

```bash
ros2 launch ffw_bringup ffw_bg2_follower_ai.launch.py \
  launch_cameras:=true \
  init_position:=true \
  init_position_file:=bg2_vr_startup_snupi.yaml \
  use_head_eef_tracker:=false
```

This moves the robot to the SNUPI starting pose and starts cameras and feedback.
Wait for homing and camera startup to finish. Keep this terminal open. Keep the
Cosmos companion stopped during homing, and run only one follower launch.
If the follower and cameras are already running and you want the current pose,
leave them running and skip this launch.

## 3. Start the companion — terminal 3

```bash
ssh ffw
cd ~/cosmos_ffw_companion
FFW_PROFILE=./config/profile.execute.json \
COSMOS_MODE=execute COSMOS_INITIALIZATION=current \
  docker compose up --no-build --pull never
```

These commands run on the **FFW host**, outside `ai_worker`. Keep the companion
running through policy changes; it captures and holds the current pose.

## 4. Select a policy and run the experiment — terminal 4, on 5090

Wait for `dual_ready` and for the follower/companion to be ready. Start the
independent recorder in terminal 5 before this step if you want session-wide
camera coverage. Run one gateway at a time.

### Use v1

```bash
cd ~/Documents/cosmos_inference_bg2
COSMOS_PROFILE=./profiles/yellowstick_v1.execute.json \
bash deploy/5090/run_gateway.sh \
  execute --profile /config/profile.json \
  --messages-dir /opt/cosmos_gateway/messages \
  --server-url http://127.0.0.1:18008 --expected-model v1 --timeout 120 \
  --instruction "Move the yellow stick into the basket." \
  --output "/output/execute-yellowstick-v1-step18420-$(date -u +%Y%m%dT%H%M%SZ)"
```

### Use v2

```bash
cd ~/Documents/cosmos_inference_bg2
COSMOS_PROFILE=./profiles/yellowstick_v2.execute.json \
bash deploy/5090/run_gateway.sh \
  execute --profile /config/profile.json \
  --messages-dir /opt/cosmos_gateway/messages \
  --server-url http://127.0.0.1:18009 --expected-model v2 --timeout 120 \
  --instruction "Move the yellow stick into the basket." \
  --output "/output/execute-yellowstick-v2-step18340-$(date -u +%Y%m%dT%H%M%SZ)"
```

These commands start robot execution. `--expected-model` checks the endpoint's
checkpoint identity before opening robot transport. Each gateway keeps its usual
per-run camera recording, predictions and imagined-video collection.

To switch policies, **Ctrl+C only terminal 4**, wait for `pause_confirmed` and
recording cleanup, then run the other command above. Leave terminals 1, 2, 3,
and 5 running. Neither reloading models nor homing is required for a normal switch.

## 5. Continuous head and wrist recording — terminal 5, on 5090

Start this once the follower's cameras are publishing:

```bash
cd ~/Documents/cosmos_inference_bg2
bash deploy/5090/record_cameras.sh
```

It records the **head, left wrist and right wrist** continuously until Ctrl+C,
including pauses and policy changes in terminal 4. It subscribes only to those
camera topics; it does not require a model server, joint feedback or a companion.
It sends no robot commands. The original JPEG packets are remuxed into MKV files
without GPU encoding, cropping or frame resampling.

Ctrl+C finalizes these videos and leaves the model servers, gateway, and robot
processes running. You can also set a recording duration, for example:

```bash
cd ~/Documents/cosmos_inference_bg2
bash deploy/5090/record_cameras.sh --duration 600
```

## Saved files

All paths below are relative to `~/Documents/cosmos_inference_bg2` on 5090:

| Recording | Location |
| --- | --- |
| Continuous terminal 5 session | `outputs/gateway/continuous-cameras-<UTC timestamp>/` |
| Terminal 4 v1 experiment | `outputs/gateway/execute-yellowstick-v1-step18420-<UTC timestamp>/` |
| Terminal 4 v2 experiment | `outputs/gateway/execute-yellowstick-v2-step18340-<UTC timestamp>/` |
| Raw v1 model outputs | `outputs/server-wam-edge-v1/runs/` |
| Raw v2 model outputs | `outputs/server-wam-edge-v2/runs/` |

The continuous session contains `cameras/head.mkv`, `cameras/wrist_left.mkv`,
`cameras/wrist.mkv`, their `*.timestamps.jsonl` files, `run.json`, `events.jsonl`,
and `recording_summary.json`. Terminal 4 run folders retain the usual
`predictions/` and `imagined/` directories in addition to their own camera videos.

## Stop the session

1. **Terminal 4:** Ctrl+C; wait for `pause_confirmed` and recording cleanup.
2. **Terminal 5:** Ctrl+C; let the continuous videos finalize.
3. **Terminal 1:** Ctrl+C; the launcher stops its two model containers.
4. If shutting down the robot session too, use the unchanged companion/follower
   steps in [Stop and reset](5090_local_inference.md#stop-and-reset).

For an ordinary v1/v2 switch, only perform the terminal 4 pause and restart.
