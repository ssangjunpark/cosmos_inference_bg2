#!/usr/bin/env bash
# Load and warm both yellowstick policies, then supervise their resident servers.
set -euo pipefail
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"
exec "${COSMOS_LAUNCHER_PYTHON:-/usr/bin/python3}" -B scripts/serve_yellowstick_pair.py "$@"
