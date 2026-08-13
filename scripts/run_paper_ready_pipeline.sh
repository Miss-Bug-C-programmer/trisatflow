#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

cat >&2 <<'EOF'
[deprecated] scripts/run_paper_ready_pipeline.sh is superseded by
scripts/run_paper_ready_pipeline_v3.sh.
EOF

if [[ $# -eq 0 ]]; then
  cat >&2 <<'EOF'
Use one of:
  bash scripts/run_paper_ready_pipeline_v3.sh preflight-offline
  bash scripts/run_paper_ready_pipeline_v3.sh dry-run --device cpu
  bash scripts/run_paper_ready_pipeline_v3.sh preflight-satedgesim --base-url "$SATEDGE_BASE_URL"
EOF
  exit 2
fi

case "${1:-}" in
  preflight-offline|preflight-satedgesim|build-traces|dry-run|formal-main|formal-rules|formal-ablation|formal-learning|formal-replay|formal-report)
    exec bash scripts/run_paper_ready_pipeline_v3.sh "$@"
    ;;
  --stage)
    stage="${2:-}"
    dry_run=0
    shift 2
    while [[ $# -gt 0 ]]; do
      case "$1" in
        --dry-run)
          dry_run=1; shift ;;
        *)
          echo "Legacy argument is no longer supported: $1" >&2
          exit 2 ;;
      esac
    done
    if [[ "$dry_run" == "1" && "$stage" =~ ^(all|offline)$ ]]; then
      exec bash scripts/run_paper_ready_pipeline_v3.sh dry-run --device "${DEVICE:-cpu}"
    fi
    echo "Legacy --stage $stage is deprecated. Choose an explicit v3 mode." >&2
    exit 2
    ;;
  --dry-run)
    exec bash scripts/run_paper_ready_pipeline_v3.sh dry-run --device "${DEVICE:-cpu}"
    ;;
  -h|--help)
    exec bash scripts/run_paper_ready_pipeline_v3.sh --help
    ;;
  *)
    echo "Unknown legacy entrypoint argument: $1" >&2
    exit 2
    ;;
esac
