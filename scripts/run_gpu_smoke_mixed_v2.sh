#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
DEVICE="${DEVICE:-cuda}"
SEEDS="${SEEDS:-13}"
EPISODES="${EPISODES:-10}"
STEPS="${STEPS:-64}"
RUN_ROOT="${RUN_ROOT:-outputs/gpu_smoke_mixed_v2_peragent_joint_logq_best_mappo_maddpg}"
CONFIG="${CONFIG:-trisatflow/configs/satedgesim_trace_mixed_v2_peragent_joint_logq_best.yaml}"
N_LEO="${N_LEO:-16}"
UPPER="${UPPER:-mappo}"
LOWER="${LOWER:-maddpg}"
MAX_SEC_PER_EP="${MAX_SEC_PER_EP:-120}"

LOG_PATH="$RUN_ROOT/smoke_train.log"
mkdir -p "$RUN_ROOT"

echo "[smoke] training start: run_root=$RUN_ROOT seeds=$SEEDS episodes=$EPISODES steps=$STEPS"
CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" python scripts/sweep_algorithm_combinations.py \
  --config "$CONFIG" \
  --upper "$UPPER" \
  --lower "$LOWER" \
  --episodes "$EPISODES" \
  --steps "$STEPS" \
  --n-leo "$N_LEO" \
  --seeds "$SEEDS" \
  --device "$DEVICE" \
  --output-root "$RUN_ROOT" | tee "$LOG_PATH"

seed_first="${SEEDS%%,*}"
RUN_DIR="$RUN_ROOT/seed_${seed_first}/upper_${UPPER}__lower_${LOWER}"
METRICS="$RUN_DIR/metrics.csv"
CHECKPOINT="$RUN_DIR/checkpoint.pt"

test -f "$METRICS"
test -f "$CHECKPOINT"

echo "[smoke] file checks passed"

python - "$METRICS" <<'PY'
import csv
import math
import sys

metrics = sys.argv[1]
with open(metrics, "r", encoding="utf-8", newline="") as f:
    rows = list(csv.DictReader(f))
if not rows:
    raise SystemExit("metrics.csv is empty")
last = rows[-1]

def f(key, default=0.0):
    try:
        v = float(last.get(key, default))
    except Exception:
        return default
    if math.isnan(v) or math.isinf(v):
        return default
    return v

trace_hit = f("trace_hit_ratio")
trace_fb = f("trace_fallback_count")
feas = f("mean_feasibility")
if trace_hit < 1.0 - 1e-9:
    raise SystemExit(f"trace_hit_ratio check failed: {trace_hit}")
if trace_fb > 0.0:
    raise SystemExit(f"trace_fallback_count check failed: {trace_fb}")
if feas < 0.95:
    raise SystemExit(f"mean_feasibility check failed: {feas}")

for key in ("mean_reward_local_selected", "upper_loss", "upper_actor_loss", "upper_value_loss", "lower_actor_loss", "lower_critic_loss"):
    raw = last.get(key, "")
    if raw not in ("", None):
        try:
            v = float(raw)
        except Exception:
            continue
        if math.isnan(v) or math.isinf(v):
            raise SystemExit(f"NaN/Inf detected in {key}: {raw}")

print("SMOKE_METRICS_OK", {"trace_hit_ratio": trace_hit, "trace_fallback_count": trace_fb, "mean_feasibility": feas})
PY

if grep -Eqi "CUDA out of memory|out of memory|device mismatch|Expected all tensors to be on the same device" "$LOG_PATH"; then
  echo "[smoke] found OOM/device mismatch in log: $LOG_PATH"
  exit 1
fi

if grep -Eqi "\bnan\b" "$LOG_PATH"; then
  echo "[smoke] found NaN token in log: $LOG_PATH"
  exit 1
fi

python - "$RUN_ROOT/sweep_summary.csv" "$EPISODES" "$MAX_SEC_PER_EP" <<'PY'
import csv
import math
import sys

summary_csv = sys.argv[1]
episodes = max(1, int(sys.argv[2]))
max_sec = float(sys.argv[3])
with open(summary_csv, "r", encoding="utf-8", newline="") as f:
    rows = list(csv.DictReader(f))
if not rows:
    raise SystemExit("sweep_summary.csv empty")
row = rows[-1]
elapsed = float(row.get("elapsed_sec", 0.0) or 0.0)
sec_per_ep = elapsed / episodes
print("SMOKE_TIME", {"elapsed_sec": elapsed, "episodes": episodes, "sec_per_episode": sec_per_ep})
if not math.isfinite(sec_per_ep):
    raise SystemExit("sec_per_episode is not finite")
if sec_per_ep > max_sec:
    raise SystemExit(f"sec_per_episode too high: {sec_per_ep:.3f} > {max_sec:.3f}")
PY

echo "[smoke] all checks passed"
