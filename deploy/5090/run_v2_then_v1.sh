#!/usr/bin/env bash
# Own only the camera recorder and sequential gateways; models/FFW run separately.
set -euo pipefail
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"
exec "${COSMOS_LAUNCHER_PYTHON:-/usr/bin/python3}" -B scripts/run_yellowstick_sequence.py "$@"
