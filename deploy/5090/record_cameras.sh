#!/usr/bin/env bash
# Independent camera subscriber; does not connect to a model or publish commands.
set -euo pipefail
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"
export COSMOS_PROFILE="${COSMOS_PROFILE:-./profiles/yellowstick_v1.execute.json}"
export COSMOS_GATEWAY_CONTAINER="${COSMOS_GATEWAY_CONTAINER:-cosmos-bg2-cameras}"
exec bash deploy/5090/run_gateway.sh record-cameras \
  --profile /config/profile.json \
  --messages-dir /opt/cosmos_gateway/messages \
  --output "/output/continuous-cameras-$(date -u +%Y%m%dT%H%M%S.%NZ)" "$@"
