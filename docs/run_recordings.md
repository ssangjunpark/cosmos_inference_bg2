# Imagined recordings organized by run

Keep using the same 5090 execution command in the [WAM guide](wam_server.md#5090-run-normal-inference).
The name from `--output` becomes the run ID automatically. No extra car-name variables,
recording command, model request or FFW update is needed for an ordinary run.

## Episode boundaries and the normal operator workflow

Each launch of the 5090 gateway with a fresh `--output` is a **new episode**, even if its
prompt is identical to the previous run. The output folder name is its episode ID, and
`run.json` records that ID, start/end times, prompt and lifecycle state. The PRO recorder,
chunk manifest and combined-video timeline use the same ID.

Ordinarily, toggle only the gateway command. Leave the PRO model service, SSH tunnel,
ROBOTIS follower/cameras and FFW companion running between episodes. Ctrl+C ends the
current episode and requests the FFW pause; the next gateway launch resumes the companion,
waits for fresh measured state/images, and uses its new CLI prompt. Chunk counters, inputs
and videos start fresh in the new episode folder. No chunk tail or video is carried over.

Occasional intentional SNUPI rehoming is separate from episode changes. For that case,
pause the gateway and stop the companion before homing with ROBOTIS; afterward start
the companion in current-pose mode to capture the new hold. This avoids competing holds
during homing. Ordinary episode changes do not require these steps.

During inference, PRO 6000 saves the generated RGB frames and returns the actions without
waiting for MP4s. A recording-only worker downloads each chunk into the matching 5090 run.
The inference sequence stays **execute all 16 → hold → fresh observations → next request**.
No model request is prefetched, and no RTC is introduced.

## On 5090

```text
~/cosmos_bg2_gateway/logs/execute-<timestamp>/
  cameras/                         actual camera streams and timestamp sidecars
  predictions/<request_id>/        exact input images/state and received action prediction
  events.jsonl                     gateway control events
  run.json                         actual model mode, including WAM
  imagined/
    manifest.json                  request order, download/render status and errors
    timeline.json                  combined-frame ranges → original requests/timestamps
    rollout.mp4                    combined imagined future: all three views
    head.mp4                       combined imagined head camera
    wrist_left.mp4                 combined imagined left wrist
    wrist_right.mp4                combined imagined right wrist
    chunks/
      000001_<request_id>/
        decoded_rollout.npy        exact 17 RGB frames, without video compression
        rollout.json               observation timestamp and per-frame forecast times
        model_output.json          native prompt, checkpoint/sampling provenance
        imagined_future.mp4        future 16-frame composite
        imagined_head.mp4
        imagined_wrist_left.mp4
        imagined_wrist_right.mp4
        rollout_full.mp4           reconstructed context + 16 future frames
      000002_<request_id>/
        ...
```

Chunk directories are created as raw downloads complete. **Per-chunk and combined MP4s
are built after the control loop stops**, using the gateway image's existing PyAV package.
There is no Docker rebuild or new package dependency. Existing older clips are remuxed
without re-encoding when collected.

## Ctrl+C and recording completion

1. Press Ctrl+C in the 5090 gateway terminal. It cancels an in-flight HTTP wait, discards any
   late action reply and requests the normal measured-position pause.
2. Wait for `pause_confirmed`. FFW has acknowledged the pause; the control loop has ended.
3. The gateway prints `collecting_recordings` if files are still arriving (up to 30 seconds
   by default), followed by `rendering_recordings` and `recordings_saved`.
4. Open `imagined/rollout.mp4` for the run, or any individual chunk's MP4.

Press Ctrl+C again to skip the recording wait or defer rendering. Already saved raw chunks
are kept. `--recording-wait 0` skips the final download wait, and a larger value waits longer;
this option affects file collection after control ends, not robot timing.

## Resume collection or collect an earlier WAM run

Run on **5090**, after the original gateway command has exited. Keep the PRO server and
SSH tunnel running. Substitute the actual run folder below:

```bash
cd ~/cosmos_bg2_gateway
docker compose run --rm --no-deps gateway \
  collect-recordings \
  --server-url http://127.0.0.1:18008 \
  --output /output/execute-20261108T060901Z \
  --recording-wait 300
```

This command **only downloads and renders saved files**. It creates no ROS/Zenoh node,
sends no robot commands, and requests no predictions. It can reconstruct a manifest from
older `predictions/` folders. Timed-out requests are matched to their saved PRO output by
request ID and exact observation timestamp, even when their action reply was never received.
Only one collector can own a run at a time; interrupted transfers resume from verified files.

## Timestamp meaning

Each chunk contains 17 frames: reconstructed current context at offset 0, then 16 imagined
future frames at +50 through +800 ms. `rollout.json` retains the source observation timestamp,
relative frame offsets and absolute forecast times. These are **forecasts relative to the
observation**, not actual controller execution timestamps.

The combined video appends the 16 future frames of each completed chunk in request order.
At 20 fps its duration is `0.8 × completed chunks` seconds. It does not insert inference,
pause or hold gaps. `timeline.json` marks missing chunks and partial previews explicitly.
`prediction_received` distinguishes a received action response from a request interrupted
before delivery; actual forwarding/completion should be checked against `events.jsonl`.

## PRO storage and recovery

PRO retains raw snapshots under
`outputs/server-wam-<tier>/runs/<run-name>/requests/<epoch>/<request_id>/`. Slow transfers,
recording errors and rendering never hold up action delivery. Pending manifest work survives
a server restart. Old `outputs/server-wam-<tier>/requests/<epoch>/<request_id>/` recordings
are still available through the collector; existing recordings are preserved.

The gateway source is mounted read-only from `~/cosmos_bg2_gateway/scripts`, so the recording
update uses the existing image. Its code update includes a backup and file-hash receipt.
FFW follower/controller/companion code is unchanged.
