#!/usr/bin/env bash
# Run the existing gateway image with source and outputs from this Git checkout.
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"
export COSMOS_UID="$(id -u)"
export COSMOS_GID="$(id -g)"
mkdir -p "$REPO_ROOT/outputs/gateway"
run_options=()
if [[ "${COSMOS_GATEWAY_NO_TTY:-0}" == "1" ]]; then
  run_options+=(-T)
fi
if [[ -n "${COSMOS_GATEWAY_SESSION_ID:-}" ]]; then
  run_options+=(--label "cosmos.sequence=${COSMOS_GATEWAY_SESSION_ID}")
fi
exec docker compose --project-directory "$REPO_ROOT" \
  -f "$REPO_ROOT/deploy/5090/gateway.local.yaml" \
  run --rm --no-deps --name "${COSMOS_GATEWAY_CONTAINER:-cosmos-bg2-execute}" \
  "${run_options[@]}" gateway "$@"
