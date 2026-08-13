"""Strict camera-ready online multiseed figures for Computer Networks.

This script reuses the v4 aggregate data, keeps the same pastel palette, and
redesigns the online-validation figures around reviewer-readable questions:
performance/energy in the main result, failure/action diagnostics in the
mechanism figure, and full seed matrices in the appendix.

Run:
python scripts/plot_online_multiseed_cn_v6_camera_ready.py \
  --data-dir outputs/paper_ready_v3/figures_v4_cn/figure_data \
  --output-dir outputs/paper_ready_v3/figures_v6_cn \
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
    panel_label,
    save_figure,
    setup_style,
)


def make_dirs(root: Path) -> dict[str, Path]:
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


def ordered_methods(df: pd.DataFrame) -> list[str]:
    present = set(df["method"])
    return [m for m in METHOD_ORDER if m in present]


def compact_label(method: str) -> str:
    return {
        "MAPPO+MADDPG": "MAPPO+\nMADDPG",
        "IPPO+MADDPG": "IPPO+\nMADDPG",
        "IPPO+MASAC": "IPPO+\nMASAC",
        "MAPPO+MASAC": "MAPPO+\nMASAC",
        "Min-energy greedy": "Min-energy",
        "Min-delay greedy": "Min-delay",
        "Lyapunov-DPP greedy": "Lyap.-DPP",
        "Queue-aware greedy": "Queue-aware",
        "Mobility-risk greedy": "Mobility-risk",
        "Random-visible": "Random-visible",
    }.get(method, method)


def y_positions(methods: list[str]) -> np.ndarray:
    ypos = np.arange(len(methods), dtype=float)
    ypos[ypos >= 4] += 0.72
    return ypos


def group_guides(ax, ypos: np.ndarray, annotate: bool = False):
    ax.axhspan(ypos[0] - 0.52, ypos[3] + 0.52, facecolor="#F7F9FB", edgecolor="none", zorder=-4)
    ax.axhline((ypos[3] + ypos[4]) / 2, color="#BFBFBF", lw=0.65, zorder=-1)
    if annotate:
        ax.text(0.01, 0.97, "RL policies", transform=ax.transAxes, ha="left", va="top", fontsize=6.3, color="#666666")
        ax.text(0.01, 0.64, "Rule baselines", transform=ax.transAxes, ha="left", va="top", fontsize=6.3, color="#666666")


def forest_axis(ax, runs: pd.DataFrame, summary: pd.DataFrame, metric: str, methods: list[str], xlabel: str, label: str, show_y: bool, logx: bool = False):
    ypos = y_positions(methods)
    rng = np.random.default_rng(17)
    for i, method in enumerate(methods):
        vals = pd.to_numeric(runs.loc[runs["method"].eq(method), metric], errors="coerce").dropna().to_numpy(float)
        row = summary.loc[summary["method"].eq(method)]
        if row.empty or len(vals) == 0:
            continue
        row = row.iloc[0]
        color = METHOD_COLORS.get(method, "#999999")
        jitter = rng.uniform(-0.105, 0.105, len(vals))
        ax.scatter(vals, np.full(len(vals), ypos[i]) + jitter, s=10, facecolor="white", edgecolor=color, linewidth=0.5, zorder=3)
        mean = row[f"{metric}_mean"]
        low = row[f"{metric}_ci95_low"]
        high = row[f"{metric}_ci95_high"]
        if pd.notna(low) and pd.notna(high):
            ax.hlines(ypos[i], low, high, color="#707070", lw=1.0, zorder=2)
            ax.plot([low, low], [ypos[i] - 0.10, ypos[i] + 0.10], color="#707070", lw=1.0, zorder=2)
            ax.plot([high, high], [ypos[i] - 0.10, ypos[i] + 0.10], color="#707070", lw=1.0, zorder=2)
        ax.scatter(mean, ypos[i], s=34, facecolor=color, edgecolor=REF_PALETTE["text"], linewidth=0.6, zorder=4)
    group_guides(ax, ypos, annotate=show_y)
    ax.set_yticks(ypos)
    ax.set_yticklabels([compact_label(m) for m in methods] if show_y else [], fontsize=6.2)
    ax.invert_yaxis()
    ax.set_xlabel(xlabel)
    if logx:
        ax.set_xscale("log")
        ax.text(0.98, 0.96, "log scale", transform=ax.transAxes, ha="right", va="top", fontsize=5.9, color="#666666")
    clean_axes(ax, "x")
    panel_label(ax, label)


def plot_core_performance(runs: pd.DataFrame, summary: pd.DataFrame, dirs: dict[str, Path], formats: list[str], dpi: int, audit: dict):
    methods = ordered_methods(summary)
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 4.25), gridspec_kw={"wspace": 0.17})
    forest_axis(axes[0], runs, summary, "successRate", methods, "Success rate $\\uparrow$", "(a)", True)
    forest_axis(axes[1], runs, summary, "energy_norm", methods, "Energy norm. $\\downarrow$", "(b)", False, logx=True)
    save_figure(fig, dirs["main_figures"], "fig11_online_core_performance_cn_v6", formats, dpi, audit, "generated_main_figures")


def plot_diagnostics(runs: pd.DataFrame, action: pd.DataFrame, dirs: dict[str, Path], formats: list[str], dpi: int, audit: dict):
    methods = ordered_methods(action)
    ypos = y_positions(methods)
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 4.25), gridspec_kw={"wspace": 0.18})

    fail = runs.groupby("method")[["delayFailureRate", "mobilityFailureRate", "resourcesFailureRate"]].mean().reindex(methods)
    left = np.zeros(len(methods))
    fail_labels = {"delayFailureRate": "Delay", "mobilityFailureRate": "Mobility", "resourcesFailureRate": "Resource"}
    for col in ["delayFailureRate", "mobilityFailureRate", "resourcesFailureRate"]:
        vals = fail[col].fillna(0).to_numpy(float)
        axes[0].barh(ypos, vals, left=left, height=0.52, color=FAILURE_COLORS[col], edgecolor=REF_PALETTE["text"], linewidth=0.5, label=fail_labels[col])
        left += vals
    group_guides(axes[0], ypos, annotate=False)
    axes[0].set_yticks(ypos)
    axes[0].set_yticklabels([compact_label(m) for m in methods], fontsize=6.2)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Failure ratio")
    axes[0].legend(frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.01), handlelength=1.0, columnspacing=0.8)
    clean_axes(axes[0], "x")
    panel_label(axes[0], "(a)")

    action_by_method = action.set_index("method").reindex(methods)
    left = np.zeros(len(methods))
    for action_name in ACTION_ORDER:
        vals = pd.to_numeric(action_by_method[f"{action_name.lower()}_ratio"], errors="coerce").fillna(0).to_numpy(float)
        axes[1].barh(ypos, vals, left=left, height=0.52, color=ACTION_COLORS[action_name], edgecolor=REF_PALETTE["text"], linewidth=0.5, label=action_name)
        left += vals
    group_guides(axes[1], ypos)
    axes[1].set_yticks(ypos)
    axes[1].set_yticklabels([])
    axes[1].invert_yaxis()
    axes[1].set_xlim(0, 1)
    axes[1].set_xlabel("Executed action ratio")
    axes[1].legend(frameon=False, ncol=2, loc="lower center", bbox_to_anchor=(0.5, 1.01), handlelength=1.0, columnspacing=0.8)
    clean_axes(axes[1], "x")
    panel_label(axes[1], "(b)")
    save_figure(fig, dirs["main_figures"], "fig12_online_failure_action_cn_v6", formats, dpi, audit, "generated_main_figures")


def representative_tradeoff_methods(summary: pd.DataFrame) -> list[str]:
    methods = ["MAPPO+MADDPG", "IPPO+MADDPG", "IPPO+MASAC", "MAPPO+MASAC"]
    rules = summary[summary["type"].eq("Rule")].sort_values("successRate_mean", ascending=False)["method"].tolist()
    for method in rules:
        if method not in methods:
            methods.append(method)
        if len(methods) >= 8:
            break
    if "Min-energy greedy" not in methods and "Min-energy greedy" in set(summary["method"]):
        methods[-1] = "Min-energy greedy"
    return [m for m in methods if m in set(summary["method"])]


def plot_tradeoff(summary: pd.DataFrame, dirs: dict[str, Path], formats: list[str], dpi: int, audit: dict):
    methods = representative_tradeoff_methods(summary)
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.85), gridspec_kw={"wspace": 0.25})
    specs = [
        ("averageEteDelay", "E2E delay $\\downarrow$", "(a)", False),
        ("energy_norm", "Energy norm. $\\downarrow$", "(b)", True),
    ]
    for ax, (xmetric, xlabel, label, logx) in zip(axes, specs):
        handles = []
        labels = []
        for method in methods:
            row = summary.loc[summary["method"].eq(method)]
            if row.empty:
                continue
            row = row.iloc[0]
            color = METHOD_COLORS.get(method, "#999999")
            x = row[f"{xmetric}_mean"]
            y = row["successRate_mean"]
            size = 36 + 260 * float(row["mobilityFailureRate_mean"])
            handle = ax.scatter(x, y, s=size, facecolor=color, edgecolor=REF_PALETTE["text"], linewidth=0.55, zorder=3)
            if ax is axes[1]:
                handles.append(handle)
                labels.append(compact_label(method).replace("\n", ""))
            xlo, xhi = row[f"{xmetric}_ci95_low"], row[f"{xmetric}_ci95_high"]
            ylo, yhi = row["successRate_ci95_low"], row["successRate_ci95_high"]
            if pd.notna(xlo) and pd.notna(xhi):
                ax.hlines(y, xlo, xhi, color="#777777", lw=0.65, zorder=1)
            if pd.notna(ylo) and pd.notna(yhi):
                ax.vlines(x, ylo, yhi, color="#777777", lw=0.65, zorder=1)
        if logx:
            ax.set_xscale("log")
            ax.text(0.98, 0.03, "log scale", transform=ax.transAxes, ha="right", va="bottom", fontsize=5.9, color="#666666")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Success rate $\\uparrow$")
        clean_axes(ax, "both")
        panel_label(ax, label)
    axes[1].legend(handles, labels, frameon=False, ncol=2, loc="lower center", bbox_to_anchor=(0.5, 1.03), fontsize=5.9, handletextpad=0.4, columnspacing=0.7)
    save_figure(fig, dirs["main_figures"], "fig13_online_tradeoff_cn_v6", formats, dpi, audit, "generated_main_figures")


def matrix(path: Path) -> pd.DataFrame:
    df = load_csv(path).set_index("method")
    df.columns = [int(c) for c in df.columns]
    return df.reindex(index=[m for m in METHOD_ORDER if m in df.index], columns=EXPECTED_SEEDS)


def draw_heat(ax, mat: pd.DataFrame, title: str, label: str, annotate=False, show_y=True):
    cmap = ONLINE_CMAP.copy()
    cmap.set_bad("#EEEEEE")
    im = ax.imshow(np.ma.masked_invalid(mat.to_numpy(float)), aspect="auto", cmap=cmap, interpolation="nearest")
    ax.set_title(title, pad=3)
    ax.set_xticks(np.arange(len(mat.columns)))
    ax.set_xticklabels([str(c) for c in mat.columns], rotation=45, ha="right", fontsize=6.0)
    ax.set_yticks(np.arange(len(mat.index)))
    ax.set_yticklabels([compact_label(m).replace("\n", " ") for m in mat.index] if show_y else [], fontsize=5.8)
    ax.axhline(3.5, color="#BFBFBF", lw=0.65)
    if annotate:
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                val = mat.iloc[i, j]
                ax.text(j, i, "NA" if pd.isna(val) else str(int(val)), ha="center", va="center", fontsize=5.2)
    panel_label(ax, label)
    return im


def plot_seed_matrix(data_dir: Path, dirs: dict[str, Path], formats: list[str], dpi: int, audit: dict):
    specs = [
        ("online_seed_matrix_success.csv", "Success rate", "(a)", False, True),
        ("online_seed_matrix_rank_success.csv", "Success rank", "(b)", True, False),
        ("online_seed_matrix_energy_norm.csv", "Energy norm.", "(c)", False, True),
        ("online_seed_matrix_mobility_failure.csv", "Mobility failure rate", "(d)", False, False),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.05), gridspec_kw={"wspace": 0.30, "hspace": 0.30})
    ims = []
    for ax, (fname, title, label, annotate, show_y) in zip(axes.flat, specs):
        ims.append(draw_heat(ax, matrix(data_dir / fname), title, label, annotate, show_y))
    for ax, im in zip(axes.flat, ims):
        fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
    save_figure(fig, dirs["appendix_figures"], "figS2_online_seed_matrix_cn_v6", formats, dpi, audit, "generated_appendix_figures")


def plot_receipt(receipt: pd.DataFrame, dirs: dict[str, Path], formats: list[str], dpi: int, audit: dict):
    methods = ordered_methods(receipt)
    mat = receipt.set_index("method").reindex(methods)[["receipt_accept_ratio", "intent_execution_match_ratio", "fallback_none_ratio"]]
    fig, ax = plt.subplots(figsize=(3.4, 3.05))
    cmap = LinearSegmentedColormap.from_list("receipt", ["white", REF_PALETTE["blue_very_pale"], REF_PALETTE["cyan_light"], REF_PALETTE["teal"]])
    im = ax.imshow(mat.to_numpy(float), aspect="auto", cmap=cmap, vmin=0, vmax=1)
    ax.set_title("SatEdgeSim receipt integrity", pad=4)
    ax.set_xticks(np.arange(3))
    ax.set_xticklabels(["Receipt\naccept", "Intent-exec.\nmatch", "No\nfallback"], fontsize=6.2)
    ax.set_yticks(np.arange(len(methods)))
    ax.set_yticklabels([compact_label(m).replace("\n", " ") for m in methods], fontsize=5.6)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, f"{mat.iloc[i, j]:.2f}", ha="center", va="center", fontsize=5.6)
    if bool((mat.round(12) == 1.0).all().all()):
        ax.text(0.5, -0.13, "All replayed decisions are receipt-consistent.", transform=ax.transAxes, ha="center", va="top", fontsize=6.2)
    fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
    save_figure(fig, dirs["appendix_figures"], "figS3_online_receipt_integrity_cn_v6", formats, dpi, audit, "generated_appendix_figures")


def write_tables(summary: pd.DataFrame, tests: pd.DataFrame, dirs: dict[str, Path], audit: dict):
    table = summary.sort_values("successRate_mean", ascending=False)
    strongest_rule = table[table["type"].eq("Rule")].iloc[0]["method"]
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Online multiseed SatEdgeSim summary.}",
        "\\label{tab:online_multiseed_summary_cn_v6}",
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
        "\\parbox{\\linewidth}{\\scriptsize All online metrics are computed over ten SatEdgeSim replay seeds. Confidence intervals are Student-t intervals over seed-level runs. $^{\\dagger}$MAPPO+MADDPG has the lowest normalized cumulative energy counter in the descriptive summary; statistical superiority claims should follow Holm-corrected paired tests.}",
        "\\end{table}",
    ]
    path = dirs["tables"] / "table_online_multiseed_summary_cn_v6.tex"
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
    metric_names = {"successRate": "Success", "averageEteDelay": "Delay", "energy_norm": "Energy norm."}
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Key paired online multiseed comparisons.}",
        "\\label{tab:online_pairwise_tests_cn_v6}",
        "\\scriptsize",
        "\\begin{tabular}{llrlll}",
        "\\toprule",
        "Metric & Comparison & $n$ & Mean diff & 95\\% CI & $p_{\\mathrm{Holm}}$ / Conclusion \\\\",
        "\\midrule",
    ]
    for metric in ["successRate", "averageEteDelay", "energy_norm"]:
        for a, b in comps:
            row = tests[(tests["metric"].eq(metric)) & (((tests["method_a"].eq(a)) & (tests["method_b"].eq(b))) | ((tests["method_a"].eq(b)) & (tests["method_b"].eq(a))))]
            if row.empty:
                continue
            row = row.iloc[0]
            conclusion = "significant after Holm correction" if bool(row["significant_holm_0.05"]) else "not significant"
            ci = f"[{fmt_num(row['ci95_diff_low'], 3)}, {fmt_num(row['ci95_diff_high'], 3)}]"
            lines.append(f"{metric_names[metric]} & {latex_escape(row['method_a'])} vs {latex_escape(row['method_b'])} & {int(row['n_paired_seeds'])} & {fmt_num(row['mean_diff'], 3)} & {ci} & {fmt_num(row['p_holm'], 3)} / {conclusion} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    path = dirs["tables"] / "table_online_pairwise_tests_cn_v6.tex"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    audit["generated_tables"].append(str(path))


def write_captions(summary: pd.DataFrame, dirs: dict[str, Path]):
    best_rl = summary.loc[summary["type"].eq("RL"), "successRate_mean"].max()
    best_rule = summary.loc[summary["type"].eq("Rule"), "successRate_mean"].max()
    mappo_energy = summary.loc[summary["method"].eq("MAPPO+MADDPG"), "energy_norm_mean"].iloc[0]
    captions = f"""# Camera-Ready Online Multiseed Captions

## Fig. 11
Online closed-loop performance over ten common SatEdgeSim replay seeds. Hollow points are seed-level runs, filled points are method means, and horizontal bars are Student-t 95% confidence intervals. Panel (b) uses normalized cumulative energy on a log scale. Each point represents one replay run, not a task-level decision.

## Fig. 12
Failure and action diagnostics. Panel (a) decomposes delay, mobility, and resource failure ratios. Panel (b) reports executed upper-level action ratios. These diagnostics are descriptive and should be interpreted together with the paired seed-level tests.

## Fig. 13
Seed-level mean trade-off view. Marker size reflects mobility-failure ratio. The figure summarizes success-delay and success-energy profiles without claiming statistical superiority unless Holm-corrected paired tests support it.

## Fig. S2
Complete online seed matrix. The matrix confirms that all RL checkpoints and rule baselines are evaluated on the same online split.

## Fig. S3
Receipt integrity dashboard. All-one values indicate consistent abstract-action mapping and execution receipts.
"""
    (dirs["captions"] / "online_multiseed_cn_v6_captions.md").write_text(captions, encoding="utf-8")
    plan = f"""# Strict Reviewer-Oriented Online Figure Plan

Reviewer critique addressed:
- The main figures avoid over-dense violin panels and separate performance, diagnostics, and seed completeness.
- Statistical unit is explicitly seed-level replay run, not decision-level sample.
- No significance stars are drawn unless Holm-corrected paired tests support them.
- Full seed matrices are kept as appendix evidence rather than crowding the main result.

Recommended wording:
Across ten common SatEdgeSim online replay seeds, the best RL mean success rate is {best_rl:.3f}, while the best rule-baseline mean success rate is {best_rule:.3f}. MAPPO+MADDPG has a mean normalized cumulative energy counter of {mappo_energy:.2f}. These are descriptive seed-level summaries; statistical superiority claims should be limited to Holm-corrected paired tests.

Avoid:
- RL significantly outperforms all online baselines.
- TriSatFlow wins the SatEdgeSim online experiment.
- The online success-rate or energy advantage is statistically significant, unless the paired Holm-corrected tests support it.
"""
    (dirs["captions"] / "experiments_online_multiseed_v6_plan.md").write_text(plan, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument("--formats", default="pdf,png,svg")
    args = parser.parse_args()

    setup_style()
    dirs = make_dirs(args.output_dir)
    formats = [f.strip() for f in args.formats.split(",") if f.strip()]
    runs = load_csv(args.data_dir / "online_runs_long.csv")
    summary = load_csv(args.data_dir / "online_summary_by_method.csv")
    tests = load_csv(args.data_dir / "online_pairwise_tests.csv")
    action = load_csv(args.data_dir / "online_action_distribution.csv")
    receipt = load_csv(args.data_dir / "online_receipt_integrity.csv")
    audit_path = args.data_dir / "online_multiseed_cn_v4_audit_data.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8")) if audit_path.exists() else {}
    audit.update(
        {
            "visual_design_pass": "v6 strict reviewer camera-ready split figures",
            "generated_main_figures": [],
            "generated_appendix_figures": [],
            "generated_tables": [],
        }
    )
    plot_core_performance(runs, summary, dirs, formats, args.dpi, audit)
    plot_diagnostics(runs, action, dirs, formats, args.dpi, audit)
    plot_tradeoff(summary, dirs, formats, args.dpi, audit)
    plot_seed_matrix(args.data_dir, dirs, formats, args.dpi, audit)
    plot_receipt(receipt, dirs, formats, args.dpi, audit)
    write_tables(summary, tests, dirs, audit)
    write_captions(summary, dirs)
    out = dirs["audit"] / "online_multiseed_cn_v6_audit.json"
    out.write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    print("Online multiseed CN v6 camera-ready plotting complete")
    print(f"  Main figures: {len(audit['generated_main_figures'])}")
    print(f"  Appendix figures: {len(audit['generated_appendix_figures'])}")
    print(f"  Tables: {len(audit['generated_tables'])}")
    print(f"  Audit: {out}")


if __name__ == "__main__":
    main()
