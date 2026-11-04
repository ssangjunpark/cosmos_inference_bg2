# BG2 checkpoint adapter

`bg2_cosmos/` contains six unchanged source files from
`robotescape_training_example_20261104/training/cosmos_futurejoint/bg2_cosmos`.
`SOURCE.json` records their original paths, sizes, and SHA-256 hashes, verified against
the supplied bundle's `PACKAGE_MANIFEST.json`. The checkpoint's original training
manifest and complete normalization statistics match that bundle's provenance.

The saved class changes only training/validation checks and inherits native inference.
The matching loader establishes domain 33, eight right-arm/gripper channels padded to
64, one observed image and current action row, and an action offset of zero. The shared
compositor and quantile affine are reused directly by `scripts/bg2_inference.py`.
Its dataset helpers are included to preserve imports and allow CPU comparisons against
the actual training loader. Inference does not construct a training loader.

Native generation additionally requires `ActionProcessingRecord`. Our inference batch
adds an identity inverse record with eight channels; native generation then returns
normalized actions. The local decoder removes current row zero and applies the training
quantile affine inverse once. No clipping or gripper remapping is applied.

The supplied `tools/prepare_joint_data.py` specifies recorded radians for arm and gripper
joints. Camera arrays are already cropped/resized by that export pipeline; the recorded
observation adapter composes those exact RGB pixels without further resizing.

No weights, recordings, optimizer states, embedded framework copy, or training environment
are vendored here. The independent NVIDIA framework remains a pinned Git submodule.
