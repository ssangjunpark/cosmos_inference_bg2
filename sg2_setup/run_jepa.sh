#!/usr/bin/env bash
# Usage: run_jepa.sh CONFIG.json [services|gateway|all] [--dry-run]
set -euo pipefail
repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
if [[ $# -eq 0 ]]; then
  echo "Usage: $0 CONFIG.json [services|gateway|all] [--dry-run]" >&2
  exit 2
fi
config=$1
shift
exec python3 "$repo/scripts/sg2_jepa_run.py" --config "$config" "$@"
