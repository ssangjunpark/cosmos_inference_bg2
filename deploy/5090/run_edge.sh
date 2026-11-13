#!/usr/bin/env bash
# Run the Edge WAM service alongside the existing 5090 gateway.
set -euo pipefail

COSMOS_ROOT="${COSMOS_ROOT:-/mnt/sata1/cosmos3}"
cd "$COSMOS_ROOT/runtime"
# Host preflight uses the system Python/PyYAML, even with Conda activated.
# Model inference uses /workspace/.venv/bin/python inside the Docker image.
exec "${COSMOS_LAUNCHER_PYTHON:-/usr/bin/python3}" scripts/cosmos3.py serve \
  --mode wam --variant edge --step 20000 --gpu 0 --image base \
  --checkpoint-root "$COSMOS_ROOT/checkpoints" \
  --tokenizer "$COSMOS_ROOT/assets/nvidia--Cosmos3-Edge-344d602b128d1bbdacb43b08d0a3626f46343e29" \
  --vae "$COSMOS_ROOT/assets/Wan-AI--Wan2.2-TI2V-5B-921dbaf3f1674a56f47e83fb80a34bac8a8f203e/Wan2.2_VAE.pth" \
  --warmup-observation inputs/recorded_observation.npz \
  --warmup-runs 2 --steps 4 --seed 0 --port 18008 "$@"
