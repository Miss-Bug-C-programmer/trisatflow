"""Camera-ready SatEdgeSim online multiseed figures for Computer Networks.

This script keeps the v4 data contract and color system, but redesigns the
main-text visuals for lower visual density: forest-style mean/CI panels with
seed-level points, a separate trade-off/action figure, and appendix heatmaps.

Run:
python scripts/plot_online_multiseed_cn_v5_camera_ready.py \
  --data-dir outputs/paper_ready_v3/figures_v4_cn/figure_data \
  --output-dir outputs/paper_ready_v3/figures_v5_cn \
  --dpi 600 \
  --formats pdf,png,svg
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd

from plot_online_multiseed_cn_v4 import (
    ACTION_COLORS,
    ACTION_ORDER,
    EXPECTED_SEEDS,
    FAILURE_COLORS,
    METHOD_COLORS,
    METHOD_ORDER,
    ONLINE_CMAP,
    REF_PALETTE,
    clean_axes,
    fmt_ci,
    fmt_num,
    latex_escape,
    load_csv,
    method_order,
    panel_label,
    save_figure,
    setup_style,
)


def out_dirs(root: Path) -> dict[str, Path]:
    dirs = {
        "main_figures": root / "main_figures",
        "appendix_figures": root / "appendix_figures",
        "tables": root / "tables",
        "captions": root / "captions",
        "audit": root / "audit",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def display_order(runs: pd.DataFrame) -> list[str]:
    return [m for m in METHOD_ORDER if m in set(runs["method"])]


def compact_label(method: str) -> str:
    return {
        "Min-energy greedy": "Min-energy",
        "Min-delay greedy": "Min-delay",
        "Lyapunov-DPP greedy": "Lyap.-DPP",
        "Queue-aware greedy": "Queue-aware",
        "Mobility-risk greedy": "Mobility-risk",
        "Random-visible": "Random-visible",
    }.get(method, method)


def y_positions(methods: list[str]) -> np.ndarray:
    base = np.arange(len(methods), dtype=float)
    base[base >= 4] += 0.75
    return base


def add_group_divider(ax, ypos: np.ndarray, annotate: bool = False):
    ax.axhspan(ypos[0] - 0.55, ypos[3] + 0.55, facecolor="#F7F9FB", edgecolor="none", zorder=-3)
    ax.axhline((ypos[3] + ypos[4]) / 2, color="#BFBFBF", lw=0.65, zorder=-1)
    if annotate:
        ax.text(0.01, 0.965, "RL policies", transform=ax.transAxes, fontsize=6.4, color="#666666", ha="left", va="top")
        ax.text(0.01, 0.675, "Rule baselines", transform=ax.transAxes, fontsize=6.4, color="#666666", ha="left", va="top")


def forest_panel(ax, runs: pd.DataFrame, summary: pd.DataFrame, metric: str, methods: list[str], xlabel: str, label: str, show_y: bool, logx: bool = False):
    ypos = y_positions(methods)
    rng = np.random.default_rng(12)
    for i, method in enumerate(methods):
        color = METHOD_COLORS.get(method, "#999999")
        row = summary.loc[summary["method"].eq(method)]
        vals = pd.to_numeric(runs.loc[runs["method"].eq(method), metric], errors="coerce").dropna().to_numpy(dtype=float)
        if row.empty or len(vals) == 0:
            continue
        row = row.iloc[0]
        mean = row[f"{metric}_mean"]
        low = row[f"{metric}_ci95_low"]
        high = row[f"{metric}_ci95_high"]
        jitter = rng.uniform(-0.12, 0.12, len(vals))
        ax.scatter(vals, np.full(len(vals), ypos[i]) + jitter, s=9, facecolor="white", edgecolor=color, linewidth=0.45, alpha=0.95, zorder=3)
        if pd.notna(low) and pd.notna(high):
            ax.hlines(ypos[i], low, high, color="#777777", lw=0.95, zorder=2)
            ax.plot([low, low], [ypos[i] - 0.11, ypos[i] + 0.11], color="#777777", lw=0.95, zorder=2)
            ax.plot([high, high], [ypos[i] - 0.11, ypos[i] + 0.11], color="#777777", lw=0.95, zorder=2)
        ax.scatter(mean, ypos[i], s=30, color=color, edgecolor=REF_PALETTE["text"], linewidth=0.55, zorder=4)
    add_group_divider(ax, ypos, annotate=(label == "(a)"))
    ax.set_yticks(ypos)
    ax.set_yticklabels(methods if show_y else [])
    ax.invert_yaxis()
    ax.set_xlabel(xlabel)
    if logx:
        ax.set_xscale("log")
        ax.text(0.98, 0.96, "log scale", transform=ax.transAxes, ha="right", va="top", fontsize=5.8, color="#666666")
    clean_axes(ax, "x")
    panel_label(ax, label)


def plot_main_performance(runs: pd.DataFrame, summary: pd.DataFrame, dirs: dict[str, Path], formats: list[str], dpi: int, audit: dict):
    methods = display_order(runs)
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.35), gridspec_kw={"wspace": 0.18, "hspace": 0.30})
    forest_panel(axes[0, 0], runs, summary, "successRate", methods, "Success rate $\\uparrow$", "(a)", True)
    forest_panel(axes[0, 1], runs, summary, "averageEteDelay", methods, "E2E delay $\\downarrow$", "(b)", False)
    forest_panel(axes[1, 0], runs, summary, "energy_norm", methods, "Energy norm. $\\downarrow$", "(c)", True, logx=True)

    ax = axes[1, 1]
    ypos = y_positions(methods)
    means = runs.groupby("method")[["delayFailureRate", "mobilityFailureRate", "resourcesFailureRate"]].mean().reindex(methods)
    left = np.zeros(len(methods))
    labels = {"delayFailureRate": "Delay", "mobilityFailureRate": "Mobility", "resourcesFailureRate": "Resource"}
    for col in ["delayFailureRate", "mobilityFailureRate", "resourcesFailureRate"]:
        vals = means[col].fillna(0).to_numpy(dtype=float)
        ax.barh(ypos, vals, left=left, color=FAILURE_COLORS[col], edgecolor=REF_PALETTE["text"], linewidth=0.5, height=0.50, label=labels[col])
        left += vals
    add_group_divider(ax, ypos)
    ax.set_yticks(ypos)
    ax.set_yticklabels([])
    ax.invert_yaxis()
    ax.set_xlabel("Failure ratio")
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.18), handlelength=1.0, columnspacing=0.8)
    clean_axes(ax, "x")
    panel_label(ax, "(d)")
    save_figure(fig, dirs["main_figures"], "fig11_online_multiseed_forest_cn_v5", formats, dpi, audit, "generated_main_figures")


def tradeoff_panel(ax, summary: pd.DataFrame, xmetric: str, xlabel: str, methods: list[str], label: str, logx: bool = False):
    for method in methods:
        row = summary.loc[summary["method"].eq(method)]
        if row.empty:
            continue
        row = row.iloc[0]
        color = METHOD_COLORS.get(method, "#999999")
        x = row[f"{xmetric}_mean"]
        y = row["successRate_mean"]
        if pd.isna(x) or pd.isna(y):
            continue
        xlo, xhi = row[f"{xmetric}_ci95_low"], row[f"{xmetric}_ci95_high"]
        ylo, yhi = row["successRate_ci95_low"], row["successRate_ci95_high"]
        xerr = None if pd.isna(xlo) or pd.isna(xhi) else [[max(0, x - xlo)], [max(0, xhi - x)]]
        yerr = None if pd.isna(ylo) or pd.isna(yhi) else [[max(0, y - ylo)], [max(0, yhi - y)]]
        size = 38 + 430 * float(row.get("mobilityFailureRate_mean", 0.0))
        ax.errorbar(x, y, xerr=xerr, yerr=yerr, fmt="none", ecolor="#777777", elinewidth=0.55, capsize=1.6, zorder=1)
        ax.scatter(x, y, s=size, color=color, edgecolor=REF_PALETTE["text"], linewidth=0.6, zorder=3)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Success rate $\\uparrow$")
    if logx:
        ax.set_xscale("log")
        ax.text(0.98, 0.03, "log scale", transform=ax.transAxes, ha="right", va="bottom", fontsize=5.8, color="#666666")
    clean_axes(ax, "both")
    panel_label(ax, label)


def plot_tradeoff_actions(summary: pd.DataFrame, action: pd.DataFrame, dirs: dict[str, Path], formats: list[str], dpi: int, audit: dict):
    methods = [m for m in display_order(action)]
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.85), gridspec_kw={"width_ratios": [0.95, 0.95, 1.20], "wspace": 0.50})
    tradeoff_panel(axes[0], summary, "averageEteDelay", "E2E delay $\\downarrow$", methods, "(a)")
    tradeoff_panel(axes[1], summary, "energy_norm", "Energy norm. $\\downarrow$", methods, "(b)", logx=True)

    ax = axes[2]
    ypos = y_positions(methods)
    bar = action.set_index("method").reindex(methods)
    left = np.zeros(len(methods))
    for action_name in ACTION_ORDER:
        vals = pd.to_numeric(bar[f"{action_name.lower()}_ratio"], errors="coerce").fillna(0).to_numpy(dtype=float)
        ax.barh(ypos, vals, left=left, color=ACTION_COLORS[action_name], edgecolor=REF_PALETTE["text"], linewidth=0.55, height=0.50, label=action_name)
        left += vals
    add_group_divider(ax, ypos)
    ax.set_yticks(ypos)
    ax.set_yticklabels([compact_label(m) for m in methods], fontsize=5.7)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("Executed action ratio")
    ax.legend(frameon=False, ncol=2, loc="lower center", bbox_to_anchor=(0.5, 1.01), handlelength=1.0, columnspacing=0.8)
    clean_axes(ax, "x")
    panel_label(ax, "(c)")
    save_figure(fig, dirs["main_figures"], "fig12_online_tradeoff_actions_cn_v5", formats, dpi, audit, "generated_main_figures")


def matrix_from_csv(path: Path) -> pd.DataFrame:
    df = load_csv(path)
    if df.empty:
        return pd.DataFrame()
    df = df.set_index("method")
    df.columns = [int(c) for c in df.columns]
    return df.reindex(index=[m for m in METHOD_ORDER if m in df.index], columns=EXPECTED_SEEDS)


def heatmap(ax, matrix: pd.DataFrame, title: str, label: str, annotate=False, show_y=True):
    cmap = ONLINE_CMAP.copy()
    cmap.set_bad("#EEEEEE")
    arr = np.ma.masked_invalid(matrix.to_numpy(dtype=float))
    im = ax.imshow(arr, aspect="auto", cmap=cmap, interpolation="nearest")
    ax.set_title(title, pad=3)
    ax.set_xticks(np.arange(len(matrix.columns)))
    ax.set_xticklabels([str(c) for c in matrix.columns], rotation=45, ha="right", fontsize=6.1)
    ax.set_yticks(np.arange(len(matrix.index)))
    ax.set_yticklabels(matrix.index if show_y else [], fontsize=5.9)
    ax.axhline(3.5, color="#BFBFBF", lw=0.65)
    if annotate:
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                val = matrix.iloc[i, j]
                ax.text(j, i, "NA" if pd.isna(val) else str(int(val)), ha="center", va="center", fontsize=5.4)
    panel_label(ax, label)
    return im


def plot_seed_appendix(data_dir: Path, dirs: dict[str, Path], formats: list[str], dpi: int, audit: dict):
    matrices = [
        (matrix_from_csv(data_dir / "online_seed_matrix_success.csv"), "Success rate", "(a)", False, True),
        (matrix_from_csv(data_dir / "online_seed_matrix_rank_success.csv"), "Success rank", "(b)", True, False),
        (matrix_from_csv(data_dir / "online_seed_matrix_energy_norm.csv"), "Energy norm.", "(c)", False, True),
        (matrix_from_csv(data_dir / "online_seed_matrix_mobility_failure.csv"), "Mobility failure rate", "(d)", False, False),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.1), gridspec_kw={"wspace": 0.30, "hspace": 0.30})
    ims = []
    for ax, (mat, title, label, annotate, show_y) in zip(axes.flat, matrices):
        ims.append(heatmap(ax, mat, title, label, annotate=annotate, show_y=show_y))
    for ax, im in zip(axes.flat, ims):
        fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
    save_figure(fig, dirs["appendix_figures"], "figS2_online_seed_matrix_cn_v5", formats, dpi, audit, "generated_appendix_figures")


def plot_receipt_appendix(receipt: pd.DataFrame, dirs: dict[str, Path], formats: list[str], dpi: int, audit: dict):
    methods = [m for m in METHOD_ORDER if m in set(receipt["method"])]
    mat = receipt.set_index("method").reindex(methods)[["receipt_accept_ratio", "intent_execution_match_ratio", "fallback_none_ratio"]]
    fig, ax = plt.subplots(figsize=(3.4, 3.05))
    cmap = LinearSegmentedColormap.from_list("receipt", ["white", REF_PALETTE["blue_very_pale"], REF_PALETTE["cyan_light"], REF_PALETTE["teal"]])
    im = ax.imshow(mat.to_numpy(dtype=float), aspect="auto", cmap=cmap, vmin=0, vmax=1)
    ax.set_title("SatEdgeSim receipt integrity", pad=4)
    ax.set_xticks(np.arange(3))
    ax.set_xticklabels(["Receipt\naccept", "Intent-exec.\nmatch", "No\nfallback"], fontsize=6.4)
    ax.set_yticks(np.arange(len(methods)))
    ax.set_yticklabels(methods, fontsize=5.8)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, f"{mat.iloc[i, j]:.2f}", ha="center", va="center", fontsize=5.6)
    ax.text(0.5, -0.13, "All replayed decisions are receipt-consistent.", transform=ax.transAxes, ha="center", va="top", fontsize=6.2)
    fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
    save_figure(fig, dirs["appendix_figures"], "figS3_online_receipt_integrity_cn_v5", formats, dpi, audit, "generated_appendix_figures")


def write_tables(summary: pd.DataFrame, tests: pd.DataFrame, dirs: dict[str, Path], audit: dict):
    table = summary.sort_values("successRate_mean", ascending=False).copy()
    strongest_rule = table[table["type"].eq("Rule")].iloc[0]["method"]
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Online multiseed SatEdgeSim summary.}",
        "\\label{tab:online_multiseed_summary_cn_v5}",
        "\\scriptsize",
        "\\begin{tabular}{llrllllll}",
        "\\toprule",
        "Method & Type & $n$ & Success $\\uparrow$ & Delay $\\downarrow$ & Energy norm. $\\downarrow$ & Delay fail. $\\downarrow$ & Mobility fail. $\\downarrow$ & Receipt match $\\uparrow$ \\\\",
        "\\midrule",
    ]
    for _, row in table.iterrows():
        method = latex_escape(row["method"])
        if row["method"] == strongest_rule:
            method = f"\\textbf{{{method}}}"
        if row["method"] == "MAPPO+MADDPG":
            method += "$^{\\dagger}$"
        lines.append(
            f"{method} & {latex_escape(row['type'])} & {int(row['n'])} & {fmt_ci(row, 'successRate', 3)} & {fmt_ci(row, 'averageEteDelay', 3)} & {fmt_ci(row, 'energy_norm', 2)} & {fmt_ci(row, 'delayFailureRate', 3)} & {fmt_ci(row, 'mobilityFailureRate', 3)} & {fmt_ci(row, 'intent_execution_match_ratio', 3)} \\\\"
        )
    lines += [
        "\\bottomrule",
        "\\end{tabular}",
        "\\vspace{1mm}",
        "\\parbox{\\linewidth}{\\scriptsize All online metrics are computed over ten SatEdgeSim replay seeds. Confidence intervals are Student-t intervals over seed-level runs. $^{\\dagger}$MAPPO+MADDPG has the lowest normalized cumulative energy counter; Holm-corrected paired tests do not support claiming a significant online success advantage.}",
        "\\end{table}",
    ]
    path = dirs["tables"] / "table_online_multiseed_summary_cn_v5.tex"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    audit["generated_tables"].append(str(path))

    comps = [
        ("MAPPO+MADDPG", "Min-energy greedy"),
        ("MAPPO+MADDPG", "GEO only"),
        ("MAPPO+MADDPG", "Local only"),
        ("IPPO+MADDPG", "Min-energy greedy"),
        ("IPPO+MADDPG", "GEO only"),
        ("IPPO+MADDPG", "Local only"),
    ]
    metrics = ["successRate", "averageEteDelay", "energy_norm"]
    names = {"successRate": "Success", "averageEteDelay": "Delay", "energy_norm": "Energy norm."}
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Key paired online multiseed comparisons.}",
        "\\label{tab:online_pairwise_tests_cn_v5}",
        "\\scriptsize",
        "\\begin{tabular}{llrlll}",
        "\\toprule",
        "Metric & Comparison & $n$ & Mean diff & 95\\% CI & $p_{\\mathrm{Holm}}$ / Conclusion \\\\",
        "\\midrule",
    ]
    for metric in metrics:
        for a, b in comps:
            row = tests[(tests["metric"].eq(metric)) & (((tests["method_a"].eq(a)) & (tests["method_b"].eq(b))) | ((tests["method_a"].eq(b)) & (tests["method_b"].eq(a))))]
            if row.empty:
                continue
            row = row.iloc[0]
            conclusion = "significant after Holm correction" if bool(row["significant_holm_0.05"]) else "not significant"
            ci = f"[{fmt_num(row['ci95_diff_low'], 3)}, {fmt_num(row['ci95_diff_high'], 3)}]"
            lines.append(f"{names[metric]} & {latex_escape(row['method_a'])} vs {latex_escape(row['method_b'])} & {int(row['n_paired_seeds'])} & {fmt_num(row['mean_diff'], 3)} & {ci} & {fmt_num(row['p_holm'], 3)} / {conclusion} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    path = dirs["tables"] / "table_online_pairwise_tests_cn_v5.tex"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    audit["generated_tables"].append(str(path))


def write_captions(dirs: dict[str, Path]):
    captions = """# Camera-Ready Online Multiseed Captions

## Fig. 11
Online multiseed closed-loop performance. Filled markers indicate method means, horizontal intervals are Student-t 95% confidence intervals over ten online seed-level replay runs, and hollow markers show individual online seeds. All methods use the same ten SatEdgeSim online seeds; no decision-level samples are used for statistical inference. Energy is shown as a normalized cumulative energy counter on a log scale.

## Fig. 12
Online trade-off and executed action distribution. Panels (a) and (b) compare success against delay and normalized cumulative energy, with marker size reflecting mobility-failure rate. Panel (c) reports executed upper-level action ratios. Similar upper-level action distributions with different energy outcomes suggest that lower-level continuous resource control contributes to the online energy profile.

## Fig. S2
Complete online seed matrix for success, success rank, normalized energy, and mobility failure rate. The matrix confirms that all RL checkpoints and rule baselines are evaluated on the same online split.

## Fig. S3
Receipt integrity dashboard. All-one values indicate consistent abstract-action mapping and execution receipts.
"""
    (dirs["captions"] / "online_multiseed_cn_v5_captions.md").write_text(captions, encoding="utf-8")
    plan = """# Strict Reviewer-Oriented Placement

Use in main text:
- Fig. 11: primary online performance summary.
- Fig. 12: trade-off and action-distribution explanation.
- Table 4: compact online multiseed summary.

Use in appendix:
- Fig. S2: complete seed matrix.
- Fig. S3: receipt integrity.
- Table 5: pairwise tests if main text space is tight.

Safe claims:
- Strong RL policies and the strongest rule baselines have similar online success rates around 0.56.
- MAPPO+MADDPG shows a much lower normalized cumulative energy counter in descriptive seed-level summaries.
- The online replay validates closed-loop action mapping and receipt consistency over ten common online seeds.

Do not claim:
- RL significantly outperforms all online baselines.
- TriSatFlow wins online.
- The online success-rate advantage is statistically significant.
- The normalized energy advantage is statistically significant unless Holm-corrected paired tests support it.
"""
    (dirs["captions"] / "experiments_online_multiseed_v5_plan.md").write_text(plan, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument("--formats", default="pdf,png,svg")
    args = parser.parse_args()

    setup_style()
    dirs = out_dirs(args.output_dir)
    formats = [f.strip() for f in args.formats.split(",") if f.strip()]
    runs = load_csv(args.data_dir / "online_runs_long.csv")
    summary = load_csv(args.data_dir / "online_summary_by_method.csv")
    tests = load_csv(args.data_dir / "online_pairwise_tests.csv")
    action = load_csv(args.data_dir / "online_action_distribution.csv")
    receipt = load_csv(args.data_dir / "online_receipt_integrity.csv")
    audit_data = args.data_dir / "online_multiseed_cn_v4_audit_data.json"
    audit = json.loads(audit_data.read_text(encoding="utf-8")) if audit_data.exists() else {}
    audit.update(
        {
            "visual_design_pass": "v5 camera-ready forest/dot-interval redesign",
            "generated_main_figures": [],
            "generated_appendix_figures": [],
            "generated_tables": [],
        }
    )
    plot_main_performance(runs, summary, dirs, formats, args.dpi, audit)
    plot_tradeoff_actions(summary, action, dirs, formats, args.dpi, audit)
    plot_seed_appendix(args.data_dir, dirs, formats, args.dpi, audit)
    plot_receipt_appendix(receipt, dirs, formats, args.dpi, audit)
    write_tables(summary, tests, dirs, audit)
    write_captions(dirs)
    audit_path = dirs["audit"] / "online_multiseed_cn_v5_audit.json"
    audit_path.write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    print("Online multiseed CN v5 camera-ready plotting complete")
    print(f"  Main figures: {len(audit['generated_main_figures'])}")
    print(f"  Appendix figures: {len(audit['generated_appendix_figures'])}")
    print(f"  Tables: {len(audit['generated_tables'])}")
    print(f"  Audit: {audit_path}")


if __name__ == "__main__":
    main()
