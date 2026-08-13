from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


THRESHOLDS = (0.01, 0.05, 0.10)


def _load_rows(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("rows") if isinstance(payload, dict) else payload
    if not isinstance(rows, list):
        raise ValueError(f"summary input must contain a rows list: {path}")
    return [dict(row) for row in rows]


def _best_at_threshold(rows: list[dict[str, Any]], threshold: float) -> dict[str, Any] | None:
    feasible = [
        row
        for row in rows
        if bool(row.get("main_ablation_deployable", True))
        and not bool(row.get("uses_cost_prior", False))
        and not bool(row.get("uses_oracle_cost", False))
        and float(row.get("deadline_violation_ratio", 0.0)) <= threshold
    ]
    if not feasible:
        return None
    return min(feasible, key=lambda row: (float(row.get("normalized_cost", row.get("cost", 0.0))), str(row.get("variant", ""))))


def build_pareto_payload(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ranking: dict[str, Any] = {}
    for threshold in THRESHOLDS:
        best = _best_at_threshold(rows, threshold)
        ranking[f"cost_at_violation_le_{threshold:.2f}"] = None if best is None else {
            "variant": best.get("variant"),
            "cost": float(best.get("normalized_cost", best.get("cost", 0.0))),
            "deadline_violation_ratio": float(best.get("deadline_violation_ratio", 0.0)),
        }

    deployable_safe = [
        row
        for row in rows
        if bool(row.get("main_ablation_deployable", True))
        and not bool(row.get("uses_cost_prior", False))
        and not bool(row.get("uses_oracle_cost", False))
    ]
    safe_ref = [row for row in deployable_safe if float(row.get("deadline_violation_ratio", 0.0)) <= 0.05]
    ref_cost = min((float(row.get("normalized_cost", row.get("cost", 0.0))) for row in safe_ref), default=None)
    annotated = []
    for row in rows:
        item = dict(row)
        cost = float(item.get("normalized_cost", item.get("cost", 0.0)))
        violation = float(item.get("deadline_violation_ratio", 0.0))
        item["unsafe_lower_cost"] = bool(ref_cost is not None and cost < ref_cost and violation > 0.05)
        annotated.append(item)

    return {
        "rows": annotated,
        "constrained_ranking": ranking,
        "unsafe_lower_cost_variants": [str(row.get("variant")) for row in annotated if bool(row.get("unsafe_lower_cost"))],
        "interpretation": "Lower cost with higher deadline violation is a cost-safety operating point, not a clean ablation win.",
    }


def _write_plot(rows: list[dict[str, Any]], output_dir: Path) -> str:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - optional plotting dependency
        return f"plot_skipped:{type(exc).__name__}"

    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    for row in rows:
        x = float(row.get("normalized_cost", row.get("cost", 0.0)))
        y = float(row.get("deadline_violation_ratio", 0.0))
        marker = "x" if bool(row.get("unsafe_lower_cost", False)) else "o"
        ax.scatter([x], [y], marker=marker)
        ax.annotate(str(row.get("variant", "")), (x, y), textcoords="offset points", xytext=(4, 4), fontsize=8)
    ax.set_xlabel("normalized cost")
    ax.set_ylabel("deadline violation ratio")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    path = output_dir / "safe_ablation_cost_safety_scatter.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return str(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot safe ablation cost-safety Pareto scatter.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = build_pareto_payload(_load_rows(Path(args.input)))
    plot_status = _write_plot(payload["rows"], output_dir)
    payload["plot_status"] = plot_status
    out_path = output_dir / "pareto_summary.json"
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
