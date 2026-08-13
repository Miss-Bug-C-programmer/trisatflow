#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${SATEDGESIM_ROOT:-}" ]]; then
  echo "SATEDGESIM_ROOT must point to the SatEdgeSim checkout/root" >&2
  exit 2
fi

PORT="${1:-${SATEDGE_PORT:-8088}}"
export SATEDGE_BASE_URL="${SATEDGE_BASE_URL:-http://127.0.0.1:${PORT}}"
export SATEDGESIM_SETTINGS_ROOT="${SATEDGESIM_SETTINGS_ROOT:-SatEdgeSim/settings/paper_v3_actual}"

cd "$SATEDGESIM_ROOT"
if [[ -z "${SATEDGESIM_GIT_COMMIT:-}" ]]; then
  if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    export SATEDGESIM_GIT_COMMIT="$(git rev-parse --short=12 HEAD)"
  else
    export SATEDGESIM_GIT_COMMIT="external-tree-no-git"
  fi
fi

bash scripts/run_rl_server.sh "$PORT"
