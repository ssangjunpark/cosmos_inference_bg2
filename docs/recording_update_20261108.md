# Run-recording update installed 2026-11-08

Source commit: `9e27e1d`.

The 5090 gateway code and Compose read-only script mount were updated at
`/home/sangjunpark/cosmos_bg2_gateway`. All installed bundle file hashes matched the new
manifest. The existing image was reused; no package installation or image build was needed.
Core, chunk-executor and observation code matched their prior installed hashes. Configured
profiles, camera recordings and FFW software were preserved.

Backup:

```text
/home/sangjunpark/cosmos_bg2_gateway.before-recordings-20261108T064621Z
```

`~/cosmos_bg2_gateway/RECORDING_UPDATE.json` records the installed revision, file scope and
backup path. The gateway was stopped during the update and was not started for robot control.

Only our PRO 6000 Nano service was restarted. GPU 4 and localhost port 8008 were verified free
after stopping the old instance. The new service is healthy in WAM mode and advertises
`cosmos_run_recordings_v1`. Its two recorded-input warmups completed without MP4 encoding.
No operator instruction or robot observation was submitted for prediction during installation.

Three existing WAM runs were collected through recording-only HTTP GETs and reconstructed
on 5090. Each has per-request clips and combined `imagined/rollout.mp4`, `head.mp4`,
`wrist_left.mp4` and `wrist_right.mp4`:

| 5090 run folder under `~/cosmos_bg2_gateway/logs/` | Chunks | Combined future frames | Duration |
|---|---:|---:|---:|
| `execute-20261108T060556Z` | 21 | 336 | 16.8 s |
| `execute-20261108T060901Z` | 1 | 16 | 0.8 s |
| `execute-20261108T061801Z` | 23 | 368 | 18.4 s |

All 45 chunks completed with zero pending/failed collections. The combined composites were
copied temporarily to PRO 6000 and decoded there to independently verify the exact frame
counts and 20 fps rate. Collection containers exited. The updated model's prediction counter
remained zero throughout collection; no ROS/Zenoh node or robot publisher was started.

The full local CPU suite passed 153 checks, with one optional SDK test skipped. A separate
real-model replay on free GPU 1 at localhost port 19018 returned two action responses in
1.29 s and 1.17 s and produced correct per-chunk and 32-frame combined videos. That temporary
test service was stopped. Tests used recorded inputs on PRO 6000, without hardware execution.

Continue with the same execution command. Ctrl+C now interrupts the HTTP wait and requests
the normal FFW pause before recording collection/rendering. The full workflow and file layout
are in [run recordings](run_recordings.md).
