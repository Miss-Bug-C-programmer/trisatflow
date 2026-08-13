from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

import torch


ACTION_NAMES = ["local", "neighbor", "geo", "ground"]


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        if value in (None, ""):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _mean(xs: List[float]) -> float:
    if not xs:
        return 0.0
    return float(sum(xs) / len(xs))


def _std(xs: List[float]) -> float:
    if len(xs) < 2:
        return 0.0
    t = torch.tensor(xs, dtype=torch.float32)
    return float(t.std(unbiased=False))


def _corr(xs: List[float], ys: List[float]) -> float:
    if len(xs) != len(ys) or len(xs) < 2:
        return 0.0
    x = torch.tensor(xs, dtype=torch.float32)
    y = torch.tensor(ys, dtype=torch.float32)
    x = x - x.mean()
    y = y - y.mean()
    denom = torch.sqrt((x.pow(2).sum() * y.pow(2).sum()).clamp_min(1.0e-12))
    return float((x * y).sum() / denom)


def _load_rollout_debug(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _load_metrics(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose advantage state-signal quality from rollout_debug or metrics.")
    parser.add_argument("--run-dir", type=str, required=True)
    parser.add_argument("--trace", type=str, default="")
    parser.add_argument("--n-leo", type=int, default=16)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    rollout_path = run_dir / "rollout_debug.csv"
    metrics_path = run_dir / "metrics.csv"
    rollout_rows = _load_rollout_debug(rollout_path)
    metrics_rows = _load_metrics(metrics_path)

    mean_adv_by_selected = {name: 0.0 for name in ACTION_NAMES}
    mean_adv_by_oracle = {name: 0.0 for name in ACTION_NAMES}
    mean_reward_by_selected = {name: 0.0 for name in ACTION_NAMES}
    mean_return_by_selected = {name: 0.0 for name in ACTION_NAMES}
    mean_value_by_selected = {name: 0.0 for name in ACTION_NAMES}
    diag_flags: List[str] = []

    advantage_values: List[float] = []
    oracle_prob_values: List[float] = []
    phase_adv: Dict[str, List[float]] = defaultdict(list)
    selected_adv: Dict[int, List[float]] = defaultdict(list)
    oracle_adv: Dict[int, List[float]] = defaultdict(list)
    reward_sel: Dict[int, List[float]] = defaultdict(list)
    ret_sel: Dict[int, List[float]] = defaultdict(list)
    val_sel: Dict[int, List[float]] = defaultdict(list)
    selected_is_oracle: List[float] = []
    adv_align_signal: List[float] = []

    if rollout_rows:
        for row in rollout_rows:
            sel = int(_to_float(row.get("selected_action"), 0.0))
            oracle = int(_to_float(row.get("oracle_action"), 0.0))
            adv = _to_float(row.get("advantage"), 0.0)
            rew = _to_float(row.get("reward"), 0.0)
            ret = _to_float(row.get("return"), 0.0)
            val = _to_float(row.get("value"), 0.0)
            p_oracle = _to_float(row.get("policy_oracle_prob"), 0.0)
            phase = str(row.get("scenario_phase", "unknown"))

            selected_adv[sel].append(adv)
            oracle_adv[oracle].append(adv)
            reward_sel[sel].append(rew)
            ret_sel[sel].append(ret)
            val_sel[sel].append(val)
            phase_adv[phase].append(adv)
            advantage_values.append(adv)
            oracle_prob_values.append(p_oracle)
            selected_is_oracle.append(1.0 if sel == oracle else 0.0)
            adv_align_signal.append(adv)

        for i, name in enumerate(ACTION_NAMES):
            mean_adv_by_selected[name] = _mean(selected_adv[i])
            mean_adv_by_oracle[name] = _mean(oracle_adv[i])
            mean_reward_by_selected[name] = _mean(reward_sel[i])
            mean_return_by_selected[name] = _mean(ret_sel[i])
            mean_value_by_selected[name] = _mean(val_sel[i])
    else:
        tail = metrics_rows[-10:] if len(metrics_rows) > 10 else metrics_rows
        for name in ACTION_NAMES:
            mean_adv_by_selected[name] = _mean([_to_float(row.get(f"mean_advantage_{name}_selected"), 0.0) for row in tail])
            mean_reward_by_selected[name] = _mean([_to_float(row.get(f"mean_reward_{name}_selected"), 0.0) for row in tail])
            mean_return_by_selected[name] = _mean([_to_float(row.get(f"mean_return_{name}_selected"), 0.0) for row in tail])
            mean_value_by_selected[name] = _mean([_to_float(row.get(f"mean_value_{name}_selected"), 0.0) for row in tail])
            mean_adv_by_oracle[name] = mean_adv_by_selected[name]
        advantage_values = [v for v in mean_adv_by_selected.values()]
        oracle_prob_values = [_to_float(row.get("prob_oracle_action_mean"), 0.0) for row in tail]
        selected_is_oracle = [0.0]
        adv_align_signal = [0.0]

    advantage_std = _std(advantage_values)
    advantage_mean_abs = _mean([abs(v) for v in advantage_values])
    snr = float(advantage_mean_abs / max(1.0e-6, advantage_std))
    oracle_alignment = _corr(selected_is_oracle, adv_align_signal)
    value_std = _std([v for values in val_sel.values() for v in values]) if rollout_rows else _std(list(mean_value_by_selected.values()))

    geo_adv = mean_adv_by_selected["geo"]
    others_adv = [mean_adv_by_selected[name] for name in ACTION_NAMES if name != "geo"]
    if snr < 0.25:
        diag_flags.append("advantage_noisy_or_weak")
    if oracle_alignment < 0.05:
        diag_flags.append("advantage_not_aligned_with_oracle")
    if value_std < 1.0e-4:
        diag_flags.append("value_baseline_washes_out_state_signal")
    if others_adv and geo_adv > max(others_adv) + 0.01:
        diag_flags.append("geo_selected_advantage_bias")
    if rollout_rows and _mean(oracle_prob_values) < 0.35 and snr < 0.4:
        diag_flags.append("lower_policy_induced_advantage_bias")
    if not diag_flags:
        diag_flags.append("advantage_signal_ok")

    payload = {
        "run_dir": str(run_dir),
        "trace": args.trace,
        "n_leo": int(args.n_leo),
        "data_source": "rollout_debug" if rollout_rows else "metrics_fallback",
        "rollout_debug_path": str(rollout_path),
        "metrics_path": str(metrics_path),
        "num_rollout_rows": len(rollout_rows),
        "mean_advantage_by_selected_action": mean_adv_by_selected,
        "mean_advantage_by_oracle_action": mean_adv_by_oracle,
        "mean_reward_by_selected_action": mean_reward_by_selected,
        "mean_return_by_selected_action": mean_return_by_selected,
        "mean_value_by_selected_action": mean_value_by_selected,
        "advantage_std": advantage_std,
        "advantage_signal_to_noise_ratio": snr,
        "advantage_oracle_alignment": oracle_alignment,
        "phase_advantage_mean": {k: _mean(v) for k, v in sorted(phase_adv.items())},
        "diagnosis": sorted(set(diag_flags)),
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
