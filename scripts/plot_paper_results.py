from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("MPLCONFIGDIR", "/tmp/trisatflow_matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from trisatflow.reporting.input_validation import ReportingInputError, load_reporting_input


FIGURE_NAMES = [
    "fig_rl_component_selection",
    "fig_rl_delay_energy_tradeoff",
    "fig_rule_baseline_comparison",
    "fig_policy_adaptivity_by_topology_phase",
    "fig_pairwise_cost_difference_forest",
]


def _method_groups(rows: Iterable[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    out: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        out[str(row.get("method", "unknown"))].append(row)
    return out


def _method_means(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for method, group in sorted(_method_groups(rows).items()):
        out.append(
            {
                "method": method,
                "cost": _mean([_to_float(row.get("metric_value")) for row in group]),
                "delay": _mean([_to_float(row.get("mean_delay_s")) for row in group]),
                "energy": _mean([_to_float(row.get("mean_energy_j")) for row in group]),
                "remote": _mean([_to_float(row.get("upper_remote_ratio")) for row in group]),
                "geo": _mean([_to_float(row.get("upper_geo_ratio")) for row in group]),
                "ground": _mean([_to_float(row.get("upper_ground_ratio")) for row in group]),
            }
        )
    return out


def _save(fig: plt.Figure, output_dir: Path, name: str) -> Dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf = output_dir / f"{name}.pdf"
    png = output_dir / f"{name}.png"
    fig.tight_layout()
    fig.savefig(pdf, format="pdf")
    fig.savefig(png, format="png", dpi=180)
    plt.close(fig)
    return {"pdf": str(pdf), "png": str(png)}


def _bar_figure(items: List[Dict[str, Any]], *, value_key: str, title: str, ylabel: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    labels = [str(item["method"]) for item in items]
    values = [float(item.get(value_key, 0.0) or 0.0) for item in items]
    ax.bar(range(len(labels)), values, color="#4c78a8")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    return fig


def _scatter_tradeoff(items: List[Dict[str, Any]]) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    for item in items:
        x = float(item.get("delay", 0.0) or 0.0)
        y = float(item.get("energy", 0.0) or 0.0)
        ax.scatter([x], [y], s=60)
        ax.annotate(str(item["method"]), (x, y), xytext=(4, 4), textcoords="offset points", fontsize=8)
    ax.set_xlabel("Mean delay (s)")
    ax.set_ylabel("Mean energy (J)")
    ax.set_title("RL delay-energy tradeoff")
    ax.grid(alpha=0.25)
    return fig


def _adaptivity_figure(rows: List[Dict[str, Any]]) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    items = _method_means(rows)
    labels = [str(item["method"]) for item in items]
    geo = [float(item.get("geo", 0.0) or 0.0) for item in items]
    ground = [float(item.get("ground", 0.0) or 0.0) for item in items]
    remote = [float(item.get("remote", 0.0) or 0.0) for item in items]
    x = list(range(len(labels)))
    ax.bar(x, geo, label="geo")
    ax.bar(x, ground, bottom=geo, label="ground")
    residual = [max(0.0, remote[i] - geo[i] - ground[i]) for i in range(len(labels))]
    ax.bar(x, residual, bottom=[geo[i] + ground[i] for i in range(len(labels))], label="other remote")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylim(0.0, max(1.0, max(remote or [1.0])))
    ax.set_ylabel("Selected action ratio")
    ax.set_title("Policy adaptivity by topology phase proxy")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    return fig


def _forest_figure(significance_rows: List[Dict[str, Any]]) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    rows = [row for row in significance_rows if str(row.get("mean_difference", "")).strip()]
    if not rows:
        ax.axvline(0.0, color="black", linewidth=1)
        ax.text(0.5, 0.5, "No pairwise tests available", transform=ax.transAxes, ha="center", va="center")
        ax.set_yticks([])
        ax.set_xlabel("Mean cost difference")
        ax.set_title("Pairwise cost difference forest")
        return fig
    labels = [f"{row.get('method_a', '')} vs {row.get('method_b', '')}" for row in rows]
    diffs = [float(row.get("mean_difference", 0.0) or 0.0) for row in rows]
    lows = [_to_float(row.get("ci95_low")) for row in rows]
    highs = [_to_float(row.get("ci95_high")) for row in rows]
    y = list(range(len(rows)))
    xerr_low = [max(0.0, diffs[i] - (lows[i] if lows[i] is not None else diffs[i])) for i in range(len(rows))]
    xerr_high = [max(0.0, (highs[i] if highs[i] is not None else diffs[i]) - diffs[i]) for i in range(len(rows))]
    ax.errorbar(diffs, y, xerr=[xerr_low, xerr_high], fmt="o", color="#4c78a8", ecolor="#777777", capsize=3)
    ax.axvline(0.0, color="black", linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Mean normalized cost difference")
    ax.set_title("Pairwise cost difference forest")
    ax.grid(axis="x", alpha=0.25)
    return fig


def _rule_items(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = [item for item in items if item["method"] in {"local_only", "neighbor_only", "geo_only", "ground_only", "random_visible", "min_delay_greedy", "min_energy_greedy", "queue_aware_greedy", "mobility_risk_greedy", "lyapunov_dpp_greedy"}]
    return out or items


def _to_float(value: Any) -> float | None:
    try:
        if value in (None, "", "NA"):
            return None
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _mean(values: List[float | None]) -> float:
    clean = [float(v) for v in values if v is not None]
    return mean(clean) if clean else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot audited paper-ready TriSatFlow results.")
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--allow-smoke-small-n", action="store_true")
    parser.add_argument("--formal", action="store_true")
    parser.add_argument(
        "--primary-semantic-class",
        default="",
        help="Require every reporting row to come from this trace_semantic_class.",
    )
    args = parser.parse_args()

    try:
        report = load_reporting_input(
            args.input_root,
            allow_smoke_small_n=bool(args.allow_smoke_small_n),
            formal=bool(args.formal),
            primary_semantic_class=str(args.primary_semantic_class),
        )
    except ReportingInputError as exc:
        raise SystemExit(f"plot_paper_results input validation failed: {exc}") from exc

    out = Path(args.output_dir)
    items = _method_means(report.rows)
    artifacts = {
        "fig_rl_component_selection": _save(
            _bar_figure(items, value_key="cost", title="RL component selection", ylabel="Normalized system cost"),
            out,
            "fig_rl_component_selection",
        ),
        "fig_rl_delay_energy_tradeoff": _save(_scatter_tradeoff(items), out, "fig_rl_delay_energy_tradeoff"),
        "fig_rule_baseline_comparison": _save(
            _bar_figure(_rule_items(items), value_key="cost", title="Rule baseline comparison", ylabel="Normalized system cost"),
            out,
            "fig_rule_baseline_comparison",
        ),
        "fig_policy_adaptivity_by_topology_phase": _save(
            _adaptivity_figure(report.rows),
            out,
            "fig_policy_adaptivity_by_topology_phase",
        ),
        "fig_pairwise_cost_difference_forest": _save(
            _forest_figure(report.significance_rows),
            out,
            "fig_pairwise_cost_difference_forest",
        ),
    }
    manifest = {
        "status": "ok",
        "input_root": str(report.input_root),
        "contract_sha256": report.contract_sha256,
        "metric_schema_version": report.metric_schema_version,
        "primary_semantic_class": str(args.primary_semantic_class),
        "smoke_mode": bool(report.smoke_mode),
        "figures": artifacts,
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    print(f"PAPER_FIGURES_OK output_dir={out} figures={len(artifacts)}")


if __name__ == "__main__":
    main()
