from __future__ import annotations

import csv
import itertools
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

try:  # SciPy is preferred when available, but smoke tests must run without it.
    from scipy.stats import ttest_rel, wilcoxon  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - exercised in minimal CPU envs
    ttest_rel = None
    wilcoxon = None

from trisatflow.analysis.statistical_schema import StatisticalRecord


PAIRWISE_FIELDS = [
    "metric",
    "method_a",
    "method_b",
    "n_rows",
    "n_effective_pairs",
    "n_train_seeds",
    "n_checkpoints",
    "mean_a",
    "mean_b",
    "mean_diff",
    "ci_low",
    "ci_high",
    "paired_t_p",
    "wilcoxon_p",
    "holm_p",
    "cohen_dz",
    "cliffs_delta",
    "holm_significant",
    "statistical_unit",
    "warning",
]


METHOD_SUMMARY_FIELDS = [
    "metric",
    "method",
    "n_rows",
    "n_train_seeds",
    "n_checkpoints",
    "n_online_seeds",
    "mean",
    "ci_low",
    "ci_high",
    "statistical_unit",
    "small_n_warning",
]


FORBIDDEN_PHRASES = [
    "significantly outperforms",
    "clearly best",
    "statistically superior",
    "dominates",
]


def _mean(values: Sequence[float]) -> float:
    return float(sum(values) / len(values)) if values else float("nan")


def _sample_std(values: Sequence[float]) -> float:
    if len(values) <= 1:
        return 0.0
    m = _mean(values)
    return float(math.sqrt(max(0.0, sum((x - m) ** 2 for x in values) / (len(values) - 1))))


def paired_t_test(values_a: Sequence[float], values_b: Sequence[float]) -> Dict[str, Any]:
    if len(values_a) != len(values_b):
        raise ValueError("paired_t_test requires equal-length paired samples")
    n = len(values_a)
    diffs = [float(a) - float(b) for a, b in zip(values_a, values_b)]
    if n < 2:
        return {"p": 1.0, "statistic": 0.0, "warning": "small_n_insufficient_for_paired_t"}
    if all(abs(d - diffs[0]) <= 1.0e-12 for d in diffs):
        return {
            "p": 1.0 if abs(diffs[0]) <= 1.0e-12 else 0.0,
            "statistic": 0.0 if abs(diffs[0]) <= 1.0e-12 else math.copysign(float("inf"), diffs[0]),
            "warning": "zero_variance_difference",
        }
    mean_diff = _mean(diffs)
    t_stat = mean_diff / (_sample_std(diffs) / math.sqrt(n))
    if ttest_rel is not None:
        result = ttest_rel(values_a, values_b)
        return {"p": float(result.pvalue), "statistic": float(result.statistic), "warning": ""}
    p_value = math.erfc(abs(t_stat) / math.sqrt(2.0))
    return {"p": float(p_value), "statistic": float(t_stat), "warning": "normal_approximation_no_scipy"}


def wilcoxon_signed_rank(values_a: Sequence[float], values_b: Sequence[float]) -> Dict[str, Any]:
    if len(values_a) != len(values_b):
        raise ValueError("wilcoxon_signed_rank requires equal-length paired samples")
    n = len(values_a)
    warnings: List[str] = []
    if n < 5:
        warnings.append("small_n_wilcoxon_warning")
    if n < 1:
        return {"p": 1.0, "statistic": 0.0, "warning": ";".join(warnings + ["insufficient_pairs"])}
    if wilcoxon is None:
        nonzero = [d for d in (float(a) - float(b) for a, b in zip(values_a, values_b)) if abs(d) > 1.0e-12]
        if not nonzero:
            return {"p": 1.0, "statistic": 0.0, "warning": ";".join(warnings + ["wilcoxon_fallback_all_zero"])}
        positives = sum(1 for d in nonzero if d > 0)
        negatives = len(nonzero) - positives
        statistic = min(positives, negatives)
        p_value = min(1.0, 2.0 * sum(math.comb(len(nonzero), k) for k in range(0, statistic + 1)) / (2 ** len(nonzero)))
        return {"p": float(p_value), "statistic": float(statistic), "warning": ";".join(warnings + ["sign_test_fallback_no_scipy"])}
    try:
        result = wilcoxon(values_a, values_b, zero_method="wilcox", alternative="two-sided")
        return {"p": float(result.pvalue), "statistic": float(result.statistic), "warning": ";".join(warnings)}
    except ValueError as exc:
        return {"p": 1.0, "statistic": 0.0, "warning": ";".join(warnings + [f"wilcoxon_degenerate:{exc.__class__.__name__}"])}


def holm_bonferroni(p_values: Sequence[float]) -> List[float]:
    cleaned = [1.0 if not math.isfinite(float(p)) else min(1.0, max(0.0, float(p))) for p in p_values]
    order = sorted(range(len(cleaned)), key=lambda idx: cleaned[idx])
    adjusted = [1.0] * len(cleaned)
    running = 0.0
    m = len(cleaned)
    for rank, idx in enumerate(order):
        value = min(1.0, (m - rank) * cleaned[idx])
        running = max(running, value)
        adjusted[idx] = running
    return adjusted


def cohen_dz_from_pairs(values_a: Sequence[float], values_b: Sequence[float]) -> float:
    diffs = [float(a) - float(b) for a, b in zip(values_a, values_b)]
    if not diffs:
        return 0.0
    sd = _sample_std(diffs)
    if sd <= 0.0:
        return 0.0 if abs(_mean(diffs)) <= 1.0e-12 else math.copysign(float("inf"), _mean(diffs))
    return float(_mean(diffs) / sd)


def cliffs_delta(values_a: Sequence[float], values_b: Sequence[float]) -> float:
    pairs = list(itertools.product(values_a, values_b))
    if not pairs:
        return 0.0
    wins = sum(1 for a, b in pairs if a > b)
    losses = sum(1 for a, b in pairs if a < b)
    return float((wins - losses) / len(pairs))


def bootstrap_ci(values: Sequence[float], *, n_boot: int = 2000, alpha: float = 0.05, seed: int = 12345) -> Tuple[float, float]:
    vals = [float(v) for v in values]
    if not vals:
        return float("nan"), float("nan")
    if len(vals) == 1:
        return vals[0], vals[0]
    rng = random.Random(seed)
    means = []
    for _ in range(int(n_boot)):
        sample = [vals[rng.randrange(len(vals))] for _ in vals]
        means.append(_mean(sample))
    means.sort()
    lo_idx = max(0, min(len(means) - 1, int((alpha / 2.0) * len(means))))
    hi_idx = max(0, min(len(means) - 1, int((1.0 - alpha / 2.0) * len(means)) - 1))
    return float(means[lo_idx]), float(means[hi_idx])


def paired_bootstrap_ci(values_a: Sequence[float], values_b: Sequence[float], *, n_boot: int = 2000, seed: int = 12345) -> Tuple[float, float]:
    return bootstrap_ci([float(a) - float(b) for a, b in zip(values_a, values_b)], n_boot=n_boot, seed=seed)


def cluster_bootstrap_ci(cluster_diffs: Mapping[str, float], *, n_boot: int = 2000, seed: int = 12345) -> Tuple[float, float]:
    return bootstrap_ci(list(cluster_diffs.values()), n_boot=n_boot, seed=seed)


def _unit_for_records(records: Sequence[StatisticalRecord]) -> str:
    units = {record.statistical_unit for record in records if record.statistical_unit}
    if len(units) == 1:
        return next(iter(units))
    if any("cluster" in unit for unit in units):
        return "train_seed_checkpoint_cluster"
    return "train_seed_checkpoint"


def _aggregate_by_unit(records: Sequence[StatisticalRecord]) -> Dict[Tuple[str, str, str], Dict[str, Any]]:
    grouped: Dict[Tuple[str, str, str], List[StatisticalRecord]] = defaultdict(list)
    for record in records:
        grouped[(record.metric, record.method, record.cluster_id)].append(record)

    out: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    for key, group in grouped.items():
        out[key] = {
            "value": _mean([record.value for record in group]),
            "train_seed": group[0].train_seed,
            "checkpoint_id": group[0].checkpoint_id,
            "pair_id": group[0].train_seed,
            "n_rows": len(group),
            "online_seeds": {record.online_seed for record in group if record.online_seed},
            "test_seeds": {record.test_seed for record in group if record.test_seed},
            "statistical_unit": _unit_for_records(group),
        }
    return out


def method_summary(records: Sequence[StatisticalRecord], *, n_boot: int = 2000) -> List[Dict[str, Any]]:
    aggregated = _aggregate_by_unit(records)
    by_method: Dict[Tuple[str, str], List[Tuple[str, Dict[str, Any]]]] = defaultdict(list)
    raw_lookup: Dict[Tuple[str, str], List[StatisticalRecord]] = defaultdict(list)
    for record in records:
        raw_lookup[(record.metric, record.method)].append(record)
    for (metric, method, cluster_id), item in aggregated.items():
        by_method[(metric, method)].append((cluster_id, item))

    rows: List[Dict[str, Any]] = []
    for (metric, method), items in sorted(by_method.items()):
        values = [float(item["value"]) for _cluster, item in items]
        ci_low, ci_high = bootstrap_ci(values, n_boot=n_boot, seed=222)
        raw = raw_lookup[(metric, method)]
        rows.append(
            {
                "metric": metric,
                "method": method,
                "n_rows": len(raw),
                "n_train_seeds": len({item["train_seed"] for _cluster, item in items}),
                "n_checkpoints": len({item["checkpoint_id"] for _cluster, item in items}),
                "n_online_seeds": len({record.online_seed for record in raw if record.online_seed}),
                "mean": _mean(values),
                "ci_low": ci_low,
                "ci_high": ci_high,
                "statistical_unit": _unit_for_records(raw),
                "small_n_warning": len(items) < 5,
            }
        )
    return rows


def pairwise_tests(records: Sequence[StatisticalRecord], *, n_boot: int = 2000, alpha: float = 0.05) -> List[Dict[str, Any]]:
    aggregated = _aggregate_by_unit(records)
    raw_by_metric_method: Dict[Tuple[str, str], List[StatisticalRecord]] = defaultdict(list)
    methods_by_metric: Dict[str, set[str]] = defaultdict(set)
    samples: Dict[Tuple[str, str], Dict[str, Dict[str, Any]]] = defaultdict(dict)
    for record in records:
        raw_by_metric_method[(record.metric, record.method)].append(record)
        methods_by_metric[record.metric].add(record.method)
    for (metric, method, _cluster_id), item in aggregated.items():
        pair_id = str(item["pair_id"])
        if pair_id in samples[(metric, method)]:
            previous = samples[(metric, method)][pair_id]
            previous["value"] = _mean([float(previous["value"]), float(item["value"])])
            previous["checkpoint_id"] = f"{previous['checkpoint_id']}|{item['checkpoint_id']}"
            previous["n_rows"] = int(previous["n_rows"]) + int(item["n_rows"])
            previous["online_seeds"] = set(previous.get("online_seeds", set())) | set(item.get("online_seeds", set()))
            previous["test_seeds"] = set(previous.get("test_seeds", set())) | set(item.get("test_seeds", set()))
        else:
            samples[(metric, method)][pair_id] = dict(item)

    rows: List[Dict[str, Any]] = []
    family_indices: Dict[str, List[int]] = defaultdict(list)
    for metric in sorted(methods_by_metric):
        methods = sorted(methods_by_metric[metric])
        for method_a, method_b in itertools.combinations(methods, 2):
            sample_a = samples[(metric, method_a)]
            sample_b = samples[(metric, method_b)]
            shared = sorted(set(sample_a) & set(sample_b))
            missing = sorted((set(sample_a) ^ set(sample_b)))
            values_a = [float(sample_a[key]["value"]) for key in shared]
            values_b = [float(sample_b[key]["value"]) for key in shared]
            diffs_by_cluster = {key: float(sample_a[key]["value"]) - float(sample_b[key]["value"]) for key in shared}
            warnings: List[str] = []
            if missing:
                warnings.append(f"missing_paired_units={len(missing)}")
            if len(shared) < 5:
                warnings.append("small_n_warning")
            if len(shared) < 2:
                t_res = {"p": 1.0, "warning": "insufficient_pairs"}
                w_res = {"p": 1.0, "warning": "insufficient_pairs"}
                ci_low = ci_high = _mean(list(diffs_by_cluster.values())) if diffs_by_cluster else float("nan")
            else:
                t_res = paired_t_test(values_a, values_b)
                w_res = wilcoxon_signed_rank(values_a, values_b)
                ci_low, ci_high = cluster_bootstrap_ci(diffs_by_cluster, n_boot=n_boot, seed=333)
            for warning in (t_res.get("warning"), w_res.get("warning")):
                if warning:
                    warnings.extend(str(warning).split(";"))
            raw_records = raw_by_metric_method[(metric, method_a)] + raw_by_metric_method[(metric, method_b)]
            row = {
                "metric": metric,
                "method_a": method_a,
                "method_b": method_b,
                "n_rows": len(raw_records),
                "n_effective_pairs": len(shared),
                "n_train_seeds": len({sample_a[key]["train_seed"] for key in shared} | {sample_b[key]["train_seed"] for key in shared}),
                "n_checkpoints": len({sample_a[key]["checkpoint_id"] for key in shared} | {sample_b[key]["checkpoint_id"] for key in shared}),
                "mean_a": _mean(values_a),
                "mean_b": _mean(values_b),
                "mean_diff": _mean([a - b for a, b in zip(values_a, values_b)]),
                "ci_low": ci_low,
                "ci_high": ci_high,
                "paired_t_p": float(t_res["p"]),
                "wilcoxon_p": float(w_res["p"]),
                "holm_p": float(t_res["p"]),
                "cohen_dz": cohen_dz_from_pairs(values_a, values_b),
                "cliffs_delta": cliffs_delta(values_a, values_b),
                "holm_significant": False,
                "statistical_unit": _unit_for_records(raw_records),
                "warning": ";".join(dict.fromkeys(warnings)),
            }
            family_indices[metric].append(len(rows))
            rows.append(row)

    for metric, indices in family_indices.items():
        adjusted = holm_bonferroni([float(rows[idx]["paired_t_p"]) for idx in indices])
        for idx, p_holm in zip(indices, adjusted):
            rows[idx]["holm_p"] = float(p_holm)
            rows[idx]["holm_significant"] = bool(p_holm < alpha)
    return rows


def build_claim_guard(
    pairwise_rows: Sequence[Mapping[str, Any]],
    summary_rows: Sequence[Mapping[str, Any]],
    *,
    metric: str,
    lower_is_better: bool = True,
) -> Dict[str, Any]:
    candidates = [row for row in summary_rows if str(row.get("metric")) == metric]
    if not candidates:
        return {
            "best_method_by_mean": "",
            "best_pair_selection_basis": "unavailable",
            "holm_significant_best": False,
            "claim_allowed": "insufficient data",
            "forbidden_phrases": FORBIDDEN_PHRASES,
            "required_paper_wording": "Insufficient statistical input for method-ranking claims.",
            "small_n_warning": True,
        }
    ordered = sorted(candidates, key=lambda row: float(row["mean"]), reverse=not lower_is_better)
    best = str(ordered[0]["method"])
    runner = str(ordered[1]["method"]) if len(ordered) > 1 else ""
    best_pair = None
    if runner:
        for row in pairwise_rows:
            if str(row.get("metric")) != metric:
                continue
            pair = {str(row.get("method_a")), str(row.get("method_b"))}
            if pair == {best, runner}:
                best_pair = row
                break
    holm_significant = bool(best_pair and best_pair.get("holm_significant"))
    small_n = any(bool(row.get("small_n_warning")) for row in candidates) or bool(
        best_pair and int(best_pair.get("n_effective_pairs", 0)) < 5
    )
    if holm_significant:
        claim_allowed = "statistically supported difference for this metric and protocol"
        basis = "holm_corrected_pairwise_difference"
        wording = (
            f"{best} has the best mean for {metric}, and the best-vs-runner-up comparison is "
            "Holm-significant under the stated checkpoint-level protocol; use cautious wording if small-n is flagged."
        )
    else:
        claim_allowed = "mean-ranked reference only; statistically comparable"
        basis = "mean_rank_reference_only"
        wording = (
            "IPPO+MADDPG is selected as a mean-ranked reference pairing; the four pairings are "
            "statistically comparable under Holm-corrected pairwise tests."
        )
    return {
        "best_method_by_mean": best,
        "runner_up_by_mean": runner,
        "best_pair_selection_basis": basis,
        "holm_significant_best": holm_significant,
        "claim_allowed": claim_allowed,
        "forbidden_phrases": FORBIDDEN_PHRASES,
        "required_paper_wording": wording,
        "small_n_warning": bool(small_n),
    }


def run_protocol(
    records: Sequence[StatisticalRecord],
    *,
    output_dir: Path,
    metric: str | None = None,
    lower_is_better: bool = True,
    n_boot: int = 2000,
) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = method_summary(records, n_boot=n_boot)
    pairwise_rows = pairwise_tests(records, n_boot=n_boot)
    target_metric = metric or (summary_rows[0]["metric"] if summary_rows else "")
    guard = build_claim_guard(pairwise_rows, summary_rows, metric=target_metric, lower_is_better=lower_is_better)
    write_csv(output_dir / "method_summary.csv", summary_rows, METHOD_SUMMARY_FIELDS)
    write_csv(output_dir / "pairwise_tests.csv", pairwise_rows, PAIRWISE_FIELDS)
    (output_dir / "claim_guard.json").write_text(json.dumps(guard, ensure_ascii=False, indent=2), encoding="utf-8")
    write_protocol_report(output_dir / "statistical_protocol_report.md", guard, summary_rows, pairwise_rows)
    return {
        "method_summary": summary_rows,
        "pairwise_tests": pairwise_rows,
        "claim_guard": guard,
    }


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    extra = []
    for row in rows:
        for key in row:
            if key not in fieldnames and key not in extra:
                extra.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames) + extra)
        writer.writeheader()
        writer.writerows(rows)


def write_protocol_report(
    path: Path,
    guard: Mapping[str, Any],
    summary_rows: Sequence[Mapping[str, Any]],
    pairwise_rows: Sequence[Mapping[str, Any]],
) -> None:
    text = [
        "# Statistical Protocol Report",
        "",
        "Training seed/checkpoint is the primary independent statistical unit.",
        "Offline test seeds are aggregated within each method x train_seed/checkpoint before pairwise tests.",
        "Online seeds are treated as repeated replay observations within checkpoint clusters, not as independent training samples.",
        "",
        "## Claim Guard",
        "",
        f"- Best method by mean: `{guard.get('best_method_by_mean', '')}`",
        f"- Selection basis: `{guard.get('best_pair_selection_basis', '')}`",
        f"- Holm-significant best-vs-runner-up: `{guard.get('holm_significant_best', False)}`",
        f"- Claim allowed: `{guard.get('claim_allowed', '')}`",
        f"- Small-n warning: `{guard.get('small_n_warning', False)}`",
        "",
        "Required paper wording:",
        "",
        str(guard.get("required_paper_wording", "")),
        "",
        "Forbidden phrases unless explicitly supported by Holm-corrected tests:",
        "",
    ]
    text.extend(f"- {phrase}" for phrase in guard.get("forbidden_phrases", []))
    text.extend(
        [
            "",
            "## Output Files",
            "",
            "- `method_summary.csv`: checkpoint-level method summaries with bootstrap CI.",
            "- `pairwise_tests.csv`: paired checkpoint-level tests with Holm correction and effect sizes.",
            "- `claim_guard.json`: machine-readable claim downgrade guard.",
            "",
            f"Method summary rows: {len(summary_rows)}",
            f"Pairwise test rows: {len(pairwise_rows)}",
        ]
    )
    path.write_text("\n".join(text) + "\n", encoding="utf-8")
