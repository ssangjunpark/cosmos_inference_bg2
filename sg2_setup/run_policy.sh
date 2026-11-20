#!/usr/bin/env bash
# Paths passed to the CLI refer to /data, /output or /handoff inside the runtime.
set -euo pipefail
repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
data=$(realpath -e -- "${SG2_DATA:-$repo/inputs}")
output=${SG2_OUTPUT:-$repo/outputs/sg2}
mkdir -p -- "$output"
output=$(realpath -e -- "$output")
command=${1:-}
script=/repo/scripts/sg2_policy.py
case "$command" in
  serve-policy) script=/repo/scripts/sg2_policy_service.py; shift ;;
  serve-pair) script=/repo/scripts/serve_sg2_pair.py; shift ;;
esac
network=none
devices=()
case "$command" in
  predict|preview|execute|serve-policy|serve-pair) devices+=(--gpus "device=${SG2_GPU:-0}" --shm-size=16g) ;;
esac
case "$command" in
  capture|preview|execute|serve-policy|serve-pair|jepa-preview|jepa-execute) network=host ;;
esac
handoff=${SG2_HANDOFF:-$repo/handoff/sg2_handoff_20261119}
mounts=()
if [[ -d "$handoff" ]]; then mounts+=(-v "$(realpath -e -- "$handoff"):/handoff:ro"); fi
exec docker run --rm --init --pull=never --network "$network" "${devices[@]}" \
  --user "$(id -u):$(id -g)" --workdir /workspace \
  -e HOME=/tmp -e HF_HUB_OFFLINE=1 -e PYTHONDONTWRITEBYTECODE=1 -e GIT_PYTHON_REFRESH=quiet \
  -e PYTHONPATH=/workspace:/repo/scripts \
  -v "$repo/scripts:/repo/scripts:ro" -v "$repo/profiles:/repo/profiles:ro" \
  -v "$repo/configs:/repo/configs:ro" -v "$repo/sg2_setup:/repo/sg2_setup:ro" -v "$repo/deploy/gateway/messages:/repo/deploy/gateway/messages:ro" \
  -v "$data:/data:ro" -v "$output:/output" "${mounts[@]}" \
  --entrypoint /workspace/.venv/bin/python cosmos-sg2-wam:local "$script" "$@"
