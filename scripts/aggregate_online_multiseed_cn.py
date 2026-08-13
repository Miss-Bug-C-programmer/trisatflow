"""Aggregate SatEdgeSim online multiseed replay runs for Computer Networks figures.

Run:
python scripts/aggregate_online_multiseed_cn.py \
  --rl-dir outputs/paper_ready_v3/satedgesim_replay_multiseed \
  --baseline-dir outputs/paper_ready_v3/satedgesim_replay_baselines_multiseed \
  --expected-online-seeds 202,303,404,505,606,707,808,909,1001,1103 \
  --output-dir outputs/paper_ready_v3/figures_v4_cn/figure_data
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path
import re

import numpy as np
import pandas as pd


METHOD_LABELS = {
    "ippo_maddpg": "IPPO+MADDPG",
    "mappo_maddpg": "MAPPO+MADDPG",
    "ippo_masac": "IPPO+MASAC",
    "mappo_masac": "MAPPO+MASAC",
    "geo_only": "GEO only",
    "ground_only": "Ground only",
    "neighbor_only": "Neighbor only",
    "local_only": "Local only",
    "random_visible": "Random-visible",
    "min_delay_greedy": "Min-delay greedy",
    "min_energy_greedy": "Min-energy greedy",
    "queue_aware_greedy": "Queue-aware greedy",
    "mobility_risk_greedy": "Mobility-risk greedy",
    "lyapunov_dpp_greedy": "Lyapunov-DPP greedy",
}

METHOD_ORDER = [
    "MAPPO+MADDPG",
    "IPPO+MADDPG",
    "IPPO+MASAC",
    "MAPPO+MASAC",
    "Min-energy greedy",
    "Min-delay greedy",
    "GEO only",
    "Lyapunov-DPP greedy",
    "Queue-aware greedy",
    "Mobility-risk greedy",
    "Ground only",
    "Neighbor only",
    "Random-visible",
    "Local only",
]

ACTION_ORDER = ["Local", "Neighbor", "GEO", "Ground"]
ACTION_MAP = {0: "Local", 1: "Neighbor", 2: "GEO", 3: "Ground"}
ACTION_COLUMNS = [
    "executed_abstract_action_name",
    "policyUpperActionName",
    "final_policy_action_name",
    "upper_action_name",
    "action_name",
]
METRICS = [
    "successRate",
    "averageEteDelay",
    "energyConsumption",
    "energy_norm",
    "delayFailureRate",
    "mobilityFailureRate",
    "resourcesFailureRate",
    "receipt_accept_ratio",
    "intent_execution_match_ratio",
    "fallback_none_ratio",
    "local_ratio",
    "neighbor_ratio",
    "geo_ratio",
    "ground_ratio",
]
PROPORTION_METRICS = {
    "successRate",
    "delayFailureRate",
    "mobilityFailureRate",
    "resourcesFailureRate",
    "receipt_accept_ratio",
    "intent_execution_match_ratio",
    "fallback_none_ratio",
    "local_ratio",
    "neighbor_ratio",
    "geo_ratio",
    "ground_ratio",
}
TEST_METRICS = [
    "successRate",
    "averageEteDelay",
    "energy_norm",
    "delayFailureRate",
    "mobilityFailureRate",
]
HIGHER_BETTER = {"successRate"}
T_CRIT_95 = {
    1: 12.706,
    2: 4.303,
    3: 3.182,
    4: 2.776,
    5: 2.571,
    6: 2.447,
    7: 2.365,
    8: 2.306,
    9: 2.262,
    10: 2.228,
    11: 2.201,
    12: 2.179,
    13: 2.160,
    14: 2.145,
    15: 2.131,
    16: 2.120,
    17: 2.110,
    18: 2.101,
    19: 2.093,
    20: 2.086,
    21: 2.080,
    22: 2.074,
    23: 2.069,
    24: 2.064,
    25: 2.060,
    26: 2.056,
    27: 2.052,
    28: 2.048,
    29: 2.045,
    30: 2.042,
}


def tcrit(n: int) -> float:
    df = max(int(n) - 1, 1)
    return T_CRIT_95.get(df, 1.96)


def mean_ci95_t(values, clip_01: bool = False):
    arr = pd.to_numeric(pd.Series(values), errors="coerce").dropna().to_numpy(dtype=float)
    if len(arr) == 0:
        return np.nan, np.nan, np.nan, 0.0, 0
    mean = float(np.mean(arr))
    if len(arr) < 2:
        low = high = np.nan
        std = np.nan
    else:
        std = float(np.std(arr, ddof=1))
        half = tcrit(len(arr)) * std / math.sqrt(len(arr))
        low = mean - half
        high = mean + half
        if clip_01:
            low = max(0.0, low)
            high = min(1.0, high)
    return mean, low, high, std, len(arr)


def read_json_safe(path: Path, audit: dict) -> dict:
    if not path.exists() or path.stat().st_size == 0:
        audit["missing_files"].append(str(path))
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        audit["missing_files"].append(f"{path} ({exc})")
        return {}


def read_csv_safe(path: Path, audit: dict) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        audit["missing_files"].append(str(path))
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception as exc:
        audit["missing_files"].append(f"{path} ({exc})")
        return pd.DataFrame()


def normalize_action(value):
    if pd.isna(value):
        return None
    text = str(value).strip()
    if text == "":
        return None
    try:
        numeric = int(float(text))
        if numeric in ACTION_MAP:
            return ACTION_MAP[numeric]
    except ValueError:
        pass
    upper = text.upper()
    if upper in {"LOCAL", "LOC"}:
        return "Local"
    if upper in {"NEIGHBOR", "NEIGHBOUR"}:
        return "Neighbor"
    if upper == "GEO":
        return "GEO"
    if upper in {"GROUND", "CLOUD"}:
        return "Ground"
    return None


def action_distribution(decision_df: pd.DataFrame, audit: dict) -> dict:
    if decision_df.empty:
        return {f"{a.lower()}_ratio": np.nan for a in ACTION_ORDER}
    col = next((c for c in ACTION_COLUMNS if c in decision_df.columns), None)
    if col is None:
        audit["missing_columns"].append("decision_log action field")
        return {f"{a.lower()}_ratio": np.nan for a in ACTION_ORDER}
    actions = decision_df[col].map(normalize_action).dropna()
    total = len(actions)
    if total == 0:
        return {f"{a.lower()}_ratio": 0.0 for a in ACTION_ORDER}
    counts = actions.value_counts()
    return {f"{a.lower()}_ratio": float(counts.get(a, 0) / total) for a in ACTION_ORDER}


def receipt_integrity(summary: dict, decision_df: pd.DataFrame) -> dict:
    out = {}
    for key in ["receipt_accept_ratio", "intent_execution_match_ratio", "fallback_none_ratio"]:
        if key in summary:
            out[key] = summary.get(key)
    if decision_df.empty:
        return {k: out.get(k, np.nan) for k in ["receipt_accept_ratio", "intent_execution_match_ratio", "fallback_none_ratio"]}
    if "receipt_accept_ratio" not in out:
        if "receipt_accepted" in decision_df.columns:
            out["receipt_accept_ratio"] = pd.to_numeric(decision_df["receipt_accepted"], errors="coerce").mean()
        elif "actionAccepted" in decision_df.columns:
            out["receipt_accept_ratio"] = pd.to_numeric(decision_df["actionAccepted"], errors="coerce").mean()
    if "intent_execution_match_ratio" not in out and "intent_execution_match" in decision_df.columns:
        out["intent_execution_match_ratio"] = pd.to_numeric(decision_df["intent_execution_match"], errors="coerce").mean()
    if "fallback_none_ratio" not in out and "fallback_reason" in decision_df.columns:
        vals = decision_df["fallback_reason"].astype(str).str.strip().str.lower()
        out["fallback_none_ratio"] = vals.isin(["", "none", "nan", "null"]).mean()
    return {k: float(out.get(k, np.nan)) for k in ["receipt_accept_ratio", "intent_execution_match_ratio", "fallback_none_ratio"]}


def metric_value(summary: dict, final_metrics: dict, key: str):
    if key in final_metrics:
        return final_metrics.get(key)
    if key in summary:
        return summary.get(key)
    nested = summary.get("final_metrics")
    if isinstance(nested, dict) and key in nested:
        return nested.get(key)
    return np.nan


def parse_rl_dir_name(name: str):
    match = re.match(r"(.+)_seed(\d+)$", name)
    if not match:
        return name, np.nan
    return match.group(1), int(match.group(2))


def collect_run(run_dir: Path, typ: str, method_raw: str, train_seed, online_seed: int, audit: dict) -> dict:
    summary = read_json_safe(run_dir / "summary.json", audit)
    final_metrics = read_json_safe(run_dir / "final_metrics.json", audit)
    decision_df = read_csv_safe(run_dir / "decision_log.csv", audit)
    row = {
        "type": typ,
        "method": METHOD_LABELS.get(method_raw, method_raw),
        "method_raw": method_raw,
        "online_seed": int(online_seed),
        "train_seed": train_seed,
        "run_dir": str(run_dir),
        "source_is_test_seed_dir": run_dir.name.startswith("test_seed_"),
        "actual_decisions": summary.get("actual_decisions", summary.get("num_decisions", len(decision_df) if not decision_df.empty else np.nan)),
    }
    for metric in [
        "successRate",
        "averageEteDelay",
        "energyConsumption",
        "delayFailureRate",
        "mobilityFailureRate",
        "resourcesFailureRate",
    ]:
        row[metric] = metric_value(summary, final_metrics, metric)
    row.update(receipt_integrity(summary, decision_df))
    row.update(action_distribution(decision_df, audit))
    return row


def collect_rl_runs(rl_dir: Path, expected_seeds: list[int], audit: dict) -> list[dict]:
    rows = []
    stale = []
    for method_dir in sorted([p for p in rl_dir.iterdir() if p.is_dir()]):
        method_raw, train_seed = parse_rl_dir_name(method_dir.name)
        top_files = [method_dir / n for n in ["summary.json", "final_metrics.json", "decision_log.csv"] if (method_dir / n).exists()]
        stale.extend(str(p) for p in top_files)
        for seed in expected_seeds:
            run_dir = method_dir / f"test_seed_{seed}"
            if run_dir.exists():
                rows.append(collect_run(run_dir, "RL", method_raw, train_seed, seed, audit))
            else:
                audit["missing_files"].append(str(run_dir))
    audit["ignored_stale_rl_top_level_files"] = bool(stale)
    audit["ignored_stale_rl_top_level_file_paths"] = stale
    return rows


def collect_baseline_runs(baseline_dir: Path, expected_seeds: list[int], audit: dict) -> list[dict]:
    rows = []
    for run_dir in sorted([p for p in baseline_dir.iterdir() if p.is_dir()]):
        match = re.match(r"(.+)_seed(\d+)$", run_dir.name)
        if not match:
            continue
        method_raw, seed = match.group(1), int(match.group(2))
        if seed not in expected_seeds:
            continue
        rows.append(collect_run(run_dir, "Rule", method_raw, np.nan, seed, audit))
    return rows


def method_sort_key(method: str) -> int:
    return METHOD_ORDER.index(method) if method in METHOD_ORDER else 999


def write_summary(df: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    rows = []
    for (typ, method, method_raw), group in df.groupby(["type", "method", "method_raw"], dropna=False):
        row = {"type": typ, "method": method, "method_raw": method_raw, "n": int(len(group))}
        for metric in METRICS:
            mean, low, high, std, n = mean_ci95_t(group[metric], clip_01=metric in PROPORTION_METRICS)
            row[f"{metric}_mean"] = mean
            row[f"{metric}_std"] = std
            row[f"{metric}_ci95_low"] = low
            row[f"{metric}_ci95_high"] = high
            row[f"{metric}_n"] = n
        rows.append(row)
    out = pd.DataFrame(rows)
    out["method_order"] = out["method"].map(method_sort_key)
    out = out.sort_values(["method_order", "method"]).drop(columns=["method_order"])
    out.to_csv(output_dir / "online_summary_by_method.csv", index=False)
    return out


def write_matrices(df: pd.DataFrame, expected_seeds: list[int], output_dir: Path):
    ordered_methods = [m for m in METHOD_ORDER if m in set(df["method"])]
    for metric, filename in [
        ("successRate", "online_seed_matrix_success.csv"),
        ("averageEteDelay", "online_seed_matrix_delay.csv"),
        ("energy_norm", "online_seed_matrix_energy_norm.csv"),
        ("delayFailureRate", "online_seed_matrix_delay_failure.csv"),
        ("mobilityFailureRate", "online_seed_matrix_mobility_failure.csv"),
    ]:
        matrix = df.pivot_table(index="method", columns="online_seed", values=metric, aggfunc="mean").reindex(index=ordered_methods, columns=expected_seeds)
        matrix.to_csv(output_dir / filename)
    rank = df.pivot_table(index="method", columns="online_seed", values="successRate", aggfunc="mean").reindex(index=ordered_methods, columns=expected_seeds)
    rank_matrix = rank.rank(axis=0, ascending=False, method="min")
    rank_matrix.to_csv(output_dir / "online_seed_matrix_rank_success.csv")


def write_action_and_receipt(df: pd.DataFrame, output_dir: Path):
    action_cols = ["local_ratio", "neighbor_ratio", "geo_ratio", "ground_ratio"]
    receipt_cols = ["receipt_accept_ratio", "intent_execution_match_ratio", "fallback_none_ratio"]
    action = df.groupby(["type", "method", "method_raw"], dropna=False)[action_cols].mean().reset_index()
    receipt = df.groupby(["type", "method", "method_raw"], dropna=False)[receipt_cols].mean().reset_index()
    action["method_order"] = action["method"].map(method_sort_key)
    receipt["method_order"] = receipt["method"].map(method_sort_key)
    action.sort_values(["method_order", "method"]).drop(columns=["method_order"]).to_csv(output_dir / "online_action_distribution.csv", index=False)
    receipt.sort_values(["method_order", "method"]).drop(columns=["method_order"]).to_csv(output_dir / "online_receipt_integrity.csv", index=False)


def paired_p_values(diff: np.ndarray) -> dict:
    diff = np.asarray(diff, dtype=float)
    diff = diff[np.isfinite(diff)]
    out = {"p_value": np.nan, "p_value_ttest": np.nan, "p_value_wilcoxon": np.nan, "test": "sign_flip"}
    if len(diff) < 2:
        return out
    try:
        from scipy import stats  # type: ignore

        t_res = stats.ttest_1samp(diff, 0.0, nan_policy="omit")
        w_res = stats.wilcoxon(diff, zero_method="wilcox", alternative="two-sided")
        out.update({"p_value": float(t_res.pvalue), "p_value_ttest": float(t_res.pvalue), "p_value_wilcoxon": float(w_res.pvalue), "test": "paired_t_and_wilcoxon"})
        return out
    except Exception:
        pass
    observed = abs(float(np.mean(diff)))
    centered = diff.copy()
    n = len(centered)
    if n <= 15:
        signs = np.array(list(itertools.product([-1.0, 1.0], repeat=n)))
        means = np.abs((signs * centered).mean(axis=1))
        p = float((np.sum(means >= observed - 1e-12)) / len(means))
    else:
        rng = np.random.default_rng(0)
        signs = rng.choice([-1.0, 1.0], size=(10000, n))
        means = np.abs((signs * centered).mean(axis=1))
        p = float((np.sum(means >= observed - 1e-12) + 1) / 10001)
    out["p_value"] = p
    return out


def holm_correct(tests: pd.DataFrame) -> pd.DataFrame:
    tests = tests.copy()
    tests["p_holm"] = np.nan
    tests["significant_holm_0.05"] = False
    for metric, group in tests.groupby("metric"):
        valid = group.dropna(subset=["p_value"]).sort_values("p_value")
        m = len(valid)
        prev = 0.0
        for rank, idx in enumerate(valid.index):
            adj = min(1.0, max(prev, tests.loc[idx, "p_value"] * (m - rank)))
            tests.loc[idx, "p_holm"] = adj
            tests.loc[idx, "significant_holm_0.05"] = bool(adj < 0.05)
            prev = adj
    return tests


def write_pairwise_tests(df: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    methods = [m for m in METHOD_ORDER if m in set(df["method"])]
    rows = []
    for metric in TEST_METRICS:
        pivot = df.pivot_table(index="online_seed", columns="method", values=metric, aggfunc="mean")
        for method_a, method_b in itertools.combinations(methods, 2):
            pair = pivot[[method_a, method_b]].dropna() if method_a in pivot and method_b in pivot else pd.DataFrame()
            a = pair[method_a].to_numpy(dtype=float) if not pair.empty else np.array([])
            b = pair[method_b].to_numpy(dtype=float) if not pair.empty else np.array([])
            diff = a - b
            mean_diff, low, high, _, n = mean_ci95_t(diff)
            pvals = paired_p_values(diff)
            mean_a = float(np.mean(a)) if n else np.nan
            mean_b = float(np.mean(b)) if n else np.nan
            if metric in HIGHER_BETTER:
                direction = "method_a_higher" if mean_diff > 0 else "method_a_lower" if mean_diff < 0 else "tie"
            else:
                direction = "method_a_lower" if mean_diff < 0 else "method_a_higher" if mean_diff > 0 else "tie"
            rows.append(
                {
                    "metric": metric,
                    "method_a": method_a,
                    "method_b": method_b,
                    "n_paired_seeds": n,
                    "mean_a": mean_a,
                    "mean_b": mean_b,
                    "mean_diff": mean_diff,
                    "ci95_diff_low": low,
                    "ci95_diff_high": high,
                    "p_value": pvals["p_value"],
                    "p_value_ttest": pvals["p_value_ttest"],
                    "p_value_wilcoxon": pvals["p_value_wilcoxon"],
                    "test": pvals["test"],
                    "direction": direction,
                }
            )
    out = holm_correct(pd.DataFrame(rows))
    out.to_csv(output_dir / "online_pairwise_tests.csv", index=False)
    return out


def build_audit(df: pd.DataFrame, expected_seeds: list[int], audit: dict) -> dict:
    actual_rl = int((df["type"] == "RL").sum())
    actual_rule = int((df["type"] == "Rule").sum())
    expected_rl = 4 * len(expected_seeds)
    expected_rule = 10 * len(expected_seeds)
    receipt_cols = ["receipt_accept_ratio", "intent_execution_match_ratio", "fallback_none_ratio"]
    receipt_all_consistent = bool((df[receipt_cols].fillna(0).round(12) == 1.0).all().all())
    return {
        "expected_online_seeds": expected_seeds,
        "expected_rl_runs": expected_rl,
        "actual_rl_runs": actual_rl,
        "expected_rule_runs": expected_rule,
        "actual_rule_runs": actual_rule,
        "ignored_stale_rl_top_level_files": bool(audit.get("ignored_stale_rl_top_level_files", False)),
        "ignored_stale_rl_top_level_file_paths": audit.get("ignored_stale_rl_top_level_file_paths", []),
        "complete_for_main_claims": bool(actual_rl == expected_rl and actual_rule == expected_rule),
        "ci_method": "Student-t over online seed-level runs",
        "statistical_test_unit": "one SatEdgeSim replay run per online seed",
        "decision_level_samples_used_for_tests": False,
        "energy_normalization": "energyConsumption divided by min positive energyConsumption across all runs",
        "receipt_all_consistent": receipt_all_consistent,
        "success_rate_significance_warning": "Do not claim online success superiority unless Holm-corrected paired tests support it.",
        "generated_main_figures": [],
        "generated_appendix_figures": [],
        "generated_tables": [],
        "missing_files": sorted(set(audit.get("missing_files", []))),
        "missing_columns": sorted(set(audit.get("missing_columns", []))),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rl-dir", required=True, type=Path)
    parser.add_argument("--baseline-dir", required=True, type=Path)
    parser.add_argument("--expected-online-seeds", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    expected_seeds = [int(x) for x in args.expected_online_seeds.split(",") if x.strip()]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    audit = {"missing_files": [], "missing_columns": []}
    rows = collect_rl_runs(args.rl_dir, expected_seeds, audit) + collect_baseline_runs(args.baseline_dir, expected_seeds, audit)
    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit("No online multiseed runs found.")
    for metric in ["successRate", "averageEteDelay", "energyConsumption", "delayFailureRate", "mobilityFailureRate", "resourcesFailureRate"]:
        df[metric] = pd.to_numeric(df[metric], errors="coerce")
    min_energy = df.loc[pd.to_numeric(df["energyConsumption"], errors="coerce") > 0, "energyConsumption"].min()
    df["energy_norm"] = df["energyConsumption"] / min_energy if pd.notna(min_energy) and min_energy > 0 else np.nan
    df["method_order"] = df["method"].map(method_sort_key)
    df = df.sort_values(["type", "method_order", "online_seed", "method"]).drop(columns=["method_order"])
    df.to_csv(args.output_dir / "online_runs_long.csv", index=False)
    write_summary(df, args.output_dir)
    write_matrices(df, expected_seeds, args.output_dir)
    write_action_and_receipt(df, args.output_dir)
    write_pairwise_tests(df, args.output_dir)
    final_audit = build_audit(df, expected_seeds, audit)
    (args.output_dir / "online_multiseed_cn_v4_audit_data.json").write_text(json.dumps(final_audit, indent=2, ensure_ascii=False), encoding="utf-8")
    print("Online multiseed aggregation complete")
    print(f"  RL runs: {final_audit['actual_rl_runs']} / {final_audit['expected_rl_runs']}")
    print(f"  Rule runs: {final_audit['actual_rule_runs']} / {final_audit['expected_rule_runs']}")
    print(f"  Ignored stale RL top-level files: {final_audit['ignored_stale_rl_top_level_files']}")
    print(f"  Complete for main claims: {final_audit['complete_for_main_claims']}")


if __name__ == "__main__":
    main()
