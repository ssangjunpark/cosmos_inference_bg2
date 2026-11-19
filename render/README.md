# Render saved gateway videos

Run these scripts from the repository root with Python 3.11 or later, PyAV,
NumPy, Pillow, and a TrueType font. The Yellowstick renderer defaults to
`/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf`. Use `--font` to select another.

## Yellowstick comparisons

Use the original run or sequence directory under `outputs/gateway/`. It supplies
the camera recordings, timestamps, and imagined videos. `--server-outputs` points
to the parent of the server run directories, usually `outputs/`.

To recreate the white, original-forecasts-only comparison from the November 15
sequence, reuse the forecast timing saved in the white export. Choose a fresh
output directory for each render.

```bash
python3 render/render_yellowstick_comparison.py \
  --sequence outputs/gateway/sequence-yellowstick-v2-v1-20261115T062317Z-2db950ec \
  --server-outputs outputs \
  --theme white --no-pause-extra \
  --timing-reference outputs/gateway/exports/yellowstick-comparison-20261115T062317Z-white/comparison.json \
  --output outputs/gateway/exports/yellowstick-white-rerender
```

For a new sequence, omit `--timing-reference` and render on the recording host
during the same boot, with prediction file modification times preserved. The
renderer maps those times to the camera clock to estimate when replies arrived.
For later renders or another machine, bring the matching verified
`comparison.json` with the source videos and pass it as the timing reference.

For a single policy, use `--run` in place of `--sequence`. Single runs use only
the forecasts saved during the run. For example, on the recording host:

```bash
python3 render/render_yellowstick_comparison.py \
  --run outputs/gateway/execute-yellowstick-v2-step18340-20261115T051551Z-test \
  --server-outputs outputs --theme white \
  --timing-reference outputs/gateway/exports/yellowstick-single-policy-comparisons-20261115/v2-test/comparison.json \
  --output outputs/gateway/exports/v2-test-rerender
```

| Export style | Options | Canvas |
| --- | --- | --- |
| Dark v2 + v1, including pause forecasts | `--sequence` | 1280 × 1600 |
| White v2 + v1, including pause forecasts | `--sequence --theme white` | 1280 × 1600 |
| White v2 + v1, original forecasts only | `--sequence --theme white --no-pause-extra` | 1280 × 1600 |
| White single policy | `--run --theme white` | 1280 × 1176 |

The [Yellowstick renderer](render_yellowstick_comparison.py) writes
`comparison.mp4`, JPG previews, and a `comparison.json` with the source runs,
forecast timing, and frame counts. The physical head camera fills the top row.
Each policy row shows imagined head, left wrist, and right wrist views. In the
script, edit `THEMES` for colors, `PolicyStrip` for labels and panel sizes, and
`render()` for the title, row positions, and encoder settings. The MP4 uses
H.264, 30 fps, yuv420p, CRF 18, and the veryfast preset. Forecasts play at 20 fps.

## Side-by-side trial comparisons

[The trial renderer](render_trial_comparisons.py) places actual cameras beside
the imagined rollout at 1072 × 784 and 20 fps. Its `--jobs` JSON is an array with
one entry per output movie, for example
`[{"run":"my-run","instruction":"Instruction saved in run.json","file":"my-run.mp4"}]`.
Pass the parent recording directory and the camera profile used for those runs.

```bash
python3 render/render_trial_comparisons.py --jobs jobs.json \
  --runs-dir outputs/gateway --output outputs/trial-comparisons \
  --profile profiles/5090.execute.json --font /usr/share/fonts/truetype/dejavu/DejaVuSans.ttf
```
