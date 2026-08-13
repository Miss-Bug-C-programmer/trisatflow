"""Plot Computer Networks style SatEdgeSim online multiseed figures.

Run:
python scripts/plot_online_multiseed_cn_v4.py \
  --data-dir outputs/paper_ready_v3/figures_v4_cn/figure_data \
  --output-dir outputs/paper_ready_v3/figures_v4_cn \
  --dpi 600 \
  --formats pdf,png,svg
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd


REF_PALETTE = {
    "salmon": "#EDB1AD",
    "green": "#A8D0A2",
    "lavender": "#C3BFDE",
    "cyan_light": "#91D8DB",
    "purple": "#857BB8",
    "pink": "#E96A97",
    "rose": "#E9A1C3",
    "pink_pale": "#F8D8E3",
    "pink_very_pale": "#FBE5EC",
    "blue_very_pale": "#E2EEF6",
    "blue_light": "#92D4EB",
    "blue_mid": "#85B4E0",
    "teal": "#32B9CD",
    "blue": "#3E92CE",
    "text": "#222222",
    "grid": "#D9D9D9",
}

METHOD_COLORS = {
    "IPPO+MADDPG": "#857BB8",
    "MAPPO+MADDPG": "#3E92CE",
    "IPPO+MASAC": "#E96A97",
    "MAPPO+MASAC": "#32B9CD",
    "GEO only": "#A8D0A2",
    "Ground only": "#91D8DB",
    "Neighbor only": "#C3BFDE",
    "Local only": "#EDB1AD",
    "Random-visible": "#F8D8E3",
    "Min-delay greedy": "#92D4EB",
    "Min-energy greedy": "#E9A1C3",
    "Queue-aware greedy": "#85B4E0",
    "Mobility-risk greedy": "#FBE5EC",
    "Lyapunov-DPP greedy": "#32B9CD",
}

ACTION_COLORS = {
    "Local": "#EDB1AD",
    "Neighbor": "#C3BFDE",
    "GEO": "#A8D0A2",
    "Ground": "#91D8DB",
}

FAILURE_COLORS = {
    "delayFailureRate": "#E9A1C3",
    "mobilityFailureRate": "#857BB8",
    "resourcesFailureRate": "#92D4EB",
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

EXPECTED_SEEDS = [202, 303, 404, 505, 606, 707, 808, 909, 1001, 1103]
ACTION_ORDER = ["Local", "Neighbor", "GEO", "Ground"]
ONLINE_CMAP = LinearSegmentedColormap.from_list(
    "pink_blue_teal",
    ["#FBE5EC", "#E9A1C3", "#E2EEF6", "#92D4EB", "#32B9CD"],
)
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


def setup_style():
    plt.rcParams.update(
        {
            "mathtext.fontset": "stix",
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "STIXGeneral", "DejaVu Serif"],
            "axes.unicode_minus": False,
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 8.5,
            "xtick.labelsize": 7.2,
            "ytick.labelsize": 7.2,
            "legend.fontsize": 7,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "axes.edgecolor": REF_PALETTE["text"],
            "text.color": REF_PALETTE["text"],
        }
    )


def clean_axes(ax, grid_axis="x"):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)
    if grid_axis:
        ax.grid(True, axis=grid_axis, color=REF_PALETTE["grid"], lw=0.45, alpha=0.85)
        ax.set_axisbelow(True)


def panel_label(ax, label):
    ax.text(-0.08, 1.03, label, transform=ax.transAxes, ha="right", va="bottom", fontsize=9, fontweight="bold")


def tcrit(n: int) -> float:
    return T_CRIT_95.get(max(int(n) - 1, 1), 1.96)


def mean_ci(values, clip_01=False):
    arr = pd.to_numeric(pd.Series(values), errors="coerce").dropna().to_numpy(dtype=float)
    if len(arr) == 0:
        return np.nan, np.nan, np.nan
    mean = float(arr.mean())
    if len(arr) < 2:
        return mean, np.nan, np.nan
    half = tcrit(len(arr)) * float(arr.std(ddof=1)) / math.sqrt(len(arr))
    low, high = mean - half, mean + half
    if clip_01:
        low, high = max(0.0, low), min(1.0, high)
    return mean, low, high


def method_order(methods):
    present = list(methods)
    ordered = [m for m in METHOD_ORDER if m in present]
    ordered.extend(sorted([m for m in present if m not in ordered]))
    return ordered


def save_figure(fig, out_dir: Path, name: str, formats: list[str], dpi: int, audit: dict, key: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for fmt in formats:
        path = out_dir / f"{name}.{fmt}"
        fig.savefig(path, dpi=dpi if fmt.lower() == "png" else None, bbox_inches="tight")
        paths.append(str(path))
    plt.close(fig)
    audit[key].extend(paths)


def load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def violin_box_jitter(ax, df: pd.DataFrame, metric: str, methods: list[str], xlabel: str):
    positions = np.arange(len(methods))
    groups = [pd.to_numeric(df.loc[df["method"].eq(m), metric], errors="coerce").dropna().to_numpy(dtype=float) for m in methods]
    parts = ax.violinplot(groups, positions=positions, vert=False, widths=0.72, showmeans=False, showmedians=False, showextrema=False)
    for body, method in zip(parts["bodies"], methods):
        body.set_facecolor(METHOD_COLORS.get(method, "#CCCCCC"))
        body.set_edgecolor(REF_PALETTE["text"])
        body.set_linewidth(0.55)
        body.set_alpha(0.48)
    ax.boxplot(
        groups,
        positions=positions,
        vert=False,
        widths=0.25,
        patch_artist=True,
        boxprops={"facecolor": "white", "edgecolor": REF_PALETTE["text"], "linewidth": 0.55, "alpha": 0.75},
        medianprops={"color": REF_PALETTE["text"], "linewidth": 0.75},
        whiskerprops={"color": REF_PALETTE["text"], "linewidth": 0.55},
        capprops={"color": REF_PALETTE["text"], "linewidth": 0.55},
        flierprops={"marker": "", "markersize": 0},
    )
    rng = np.random.default_rng(4)
    for pos, method, vals in zip(positions, methods, groups):
        jitter = rng.uniform(-0.11, 0.11, len(vals))
        ax.scatter(vals, pos + jitter, s=10, facecolor="white", edgecolor=METHOD_COLORS.get(method, "#666666"), linewidth=0.55, zorder=4)
    shade_groups(ax, len(methods))
    ax.set_xlabel(xlabel)
    ax.set_yticks(positions)
    ax.set_yticklabels(methods)
    ax.invert_yaxis()
    clean_axes(ax, "x")


def shade_groups(ax, n_methods: int):
    rl_count = min(4, n_methods)
    ax.axhspan(-0.5, rl_count - 0.5, facecolor="#F7F9FB", edgecolor="none", zorder=-2)
    if n_methods > rl_count:
        ax.axhline(rl_count - 0.5, color="#BFBFBF", lw=0.65, zorder=1)


def add_success_significance(ax, tests: pd.DataFrame, methods: list[str]):
    sig = tests[(tests["metric"].eq("successRate")) & (tests["significant_holm_0.05"].astype(bool))]
    sig = sig[sig["method_a"].eq("MAPPO+MADDPG") | sig["method_b"].eq("MAPPO+MADDPG")].head(1)
    if sig.empty:
        return
    row = sig.iloc[0]
    m1, m2 = row["method_a"], row["method_b"]
    if m1 not in methods or m2 not in methods:
        return
    y1, y2 = methods.index(m1), methods.index(m2)
    x = max(ax.get_xlim()) * 0.98
    ax.plot([x, x], [y1, y2], color=REF_PALETTE["text"], lw=0.65, clip_on=False)
    ax.text(x, (y1 + y2) / 2, "*", ha="left", va="center", fontsize=9, clip_on=False)


def plot_fig11(runs: pd.DataFrame, tests: pd.DataFrame, out_dirs: dict, formats: list[str], dpi: int, audit: dict):
    methods = method_order(runs["method"].unique())
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.7), gridspec_kw={"wspace": 0.22, "hspace": 0.26})
    specs = [
        ("successRate", "Success rate $\\uparrow$", "(a)"),
        ("averageEteDelay", "E2E delay $\\downarrow$", "(b)"),
        ("energy_norm", "Energy norm. $\\downarrow$", "(c)"),
    ]
    for ax, (metric, xlabel, label) in zip([axes[0, 0], axes[0, 1], axes[1, 0]], specs):
        violin_box_jitter(ax, runs, metric, methods, xlabel)
        panel_label(ax, label)
        if label in {"(b)"}:
            ax.set_yticklabels([])
        if metric == "successRate":
            add_success_significance(ax, tests, methods)

    ax = axes[1, 1]
    means = runs.groupby("method")[["delayFailureRate", "mobilityFailureRate", "resourcesFailureRate"]].mean().reindex(methods)
    y = np.arange(len(methods))
    left = np.zeros(len(methods))
    labels = {
        "delayFailureRate": "Delay",
        "mobilityFailureRate": "Mobility",
        "resourcesFailureRate": "Resource",
    }
    for col in ["delayFailureRate", "mobilityFailureRate", "resourcesFailureRate"]:
        vals = means[col].fillna(0).to_numpy(dtype=float)
        ax.barh(y, vals, left=left, color=FAILURE_COLORS[col], edgecolor=REF_PALETTE["text"], linewidth=0.5, label=labels[col], height=0.68)
        left += vals
    shade_groups(ax, len(methods))
    ax.set_yticks(y)
    ax.set_yticklabels([])
    ax.invert_yaxis()
    ax.set_xlabel("Failure ratio")
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.20), handlelength=1.0, columnspacing=0.8)
    clean_axes(ax, "x")
    panel_label(ax, "(d)")
    save_figure(fig, out_dirs["main_figures"], "fig11_online_multiseed_performance_cn_v4", formats, dpi, audit, "generated_main_figures")


def heatmap_panel(ax, matrix: pd.DataFrame, title: str, label: str, annotate_rank=False, show_y=True):
    matrix = matrix.reindex(index=method_order(matrix.index), columns=EXPECTED_SEEDS)
    arr = matrix.to_numpy(dtype=float)
    cmap = ONLINE_CMAP.copy()
    cmap.set_bad("#EEEEEE")
    im = ax.imshow(np.ma.masked_invalid(arr), aspect="auto", cmap=cmap, interpolation="nearest")
    ax.set_title(title, pad=3)
    ax.set_xticks(np.arange(len(matrix.columns)))
    ax.set_xticklabels([str(c) for c in matrix.columns], rotation=45, ha="right")
    ax.set_yticks(np.arange(len(matrix.index)))
    ax.set_yticklabels(matrix.index if show_y else [])
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = matrix.iloc[i, j]
            if pd.isna(val):
                ax.text(j, i, "NA", ha="center", va="center", fontsize=5.5, color="#777777")
            elif annotate_rank:
                ax.text(j, i, str(int(val)), ha="center", va="center", fontsize=5.6, color=REF_PALETTE["text"])
    ax.axhline(3.5, color="#BFBFBF", lw=0.65)
    panel_label(ax, label)
    return im


def plot_fig12(data_dir: Path, out_dirs: dict, formats: list[str], dpi: int, audit: dict):
    matrices = {
        "success": load_csv(data_dir / "online_seed_matrix_success.csv").set_index("method"),
        "rank": load_csv(data_dir / "online_seed_matrix_rank_success.csv").set_index("method"),
        "delay_failure": load_csv(data_dir / "online_seed_matrix_delay_failure.csv").set_index("method"),
        "mobility_failure": load_csv(data_dir / "online_seed_matrix_mobility_failure.csv").set_index("method"),
    }
    for key in matrices:
        matrices[key].columns = [int(c) for c in matrices[key].columns]
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.0), gridspec_kw={"wspace": 0.20, "hspace": 0.30})
    ims = []
    ims.append(heatmap_panel(axes[0, 0], matrices["success"], "Success rate", "(a)", show_y=True))
    ims.append(heatmap_panel(axes[0, 1], matrices["rank"], "Success rank", "(b)", annotate_rank=True, show_y=False))
    ims.append(heatmap_panel(axes[1, 0], matrices["delay_failure"], "Delay failure rate", "(c)", show_y=True))
    ims.append(heatmap_panel(axes[1, 1], matrices["mobility_failure"], "Mobility failure rate", "(d)", show_y=False))
    for ax, im in zip(axes.flat, ims):
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    save_figure(fig, out_dirs["main_figures"], "fig12_online_seed_consistency_heatmap_cn_v4", formats, dpi, audit, "generated_main_figures")


def summary_value(summary: pd.DataFrame, method: str, metric: str, suffix: str):
    row = summary.loc[summary["method"].eq(method)]
    if row.empty:
        return np.nan
    return float(row.iloc[0].get(f"{metric}_{suffix}", np.nan))


def plot_tradeoff_panel(ax, summary: pd.DataFrame, xmetric: str, xlabel: str, annotations: dict, label: str):
    methods = method_order(summary["method"].unique())
    for method in methods:
        row = summary.loc[summary["method"].eq(method)]
        if row.empty:
            continue
        row = row.iloc[0]
        x = row[f"{xmetric}_mean"]
        y = row["successRate_mean"]
        xerr = [[max(0, x - row[f"{xmetric}_ci95_low"])], [max(0, row[f"{xmetric}_ci95_high"] - x)]] if pd.notna(row[f"{xmetric}_ci95_low"]) else None
        yerr = [[max(0, y - row["successRate_ci95_low"])], [max(0, row["successRate_ci95_high"] - y)]] if pd.notna(row["successRate_ci95_low"]) else None
        size = 42 + 420 * max(0.0, float(row.get("mobilityFailureRate_mean", 0.0)))
        ax.errorbar(x, y, xerr=xerr, yerr=yerr, fmt="none", ecolor="#777777", elinewidth=0.55, capsize=1.8, zorder=1)
        ax.scatter(x, y, s=size, color=METHOD_COLORS.get(method, "#CCCCCC"), edgecolor=REF_PALETTE["text"], linewidth=0.55, zorder=3)
        if method in annotations:
            dx, dy, text = annotations[method]
            ax.annotate(text, (x, y), xytext=(dx, dy), textcoords="offset points", fontsize=6.2, arrowprops={"arrowstyle": "-", "lw": 0.4, "color": "#777777"})
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Success rate $\\uparrow$")
    clean_axes(ax, "both")
    panel_label(ax, label)


def plot_fig13(summary: pd.DataFrame, action: pd.DataFrame, out_dirs: dict, formats: list[str], dpi: int, audit: dict):
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.75), gridspec_kw={"width_ratios": [0.95, 0.95, 1.18], "wspace": 0.52})
    annotations_a = {
        "IPPO+MADDPG": (-78, -22, "offline-selected RL"),
    }
    annotations_b = {
        "MAPPO+MADDPG": (12, 10, "low-energy RL"),
        "Min-energy greedy": (-10, 25, "strongest rule"),
    }
    plot_tradeoff_panel(axes[0], summary, "averageEteDelay", "E2E delay $\\downarrow$", annotations_a, "(a)")
    plot_tradeoff_panel(axes[1], summary, "energy_norm", "Energy norm. $\\downarrow$", annotations_b, "(b)")
    methods = method_order(action["method"].unique())
    bar = action.set_index("method").reindex(methods)
    y = np.arange(len(methods))
    left = np.zeros(len(methods))
    for action_name in ACTION_ORDER:
        vals = pd.to_numeric(bar[f"{action_name.lower()}_ratio"], errors="coerce").fillna(0).to_numpy(dtype=float)
        axes[2].barh(y, vals, left=left, color=ACTION_COLORS[action_name], edgecolor=REF_PALETTE["text"], linewidth=0.55, height=0.68, label=action_name)
        left += vals
    shade_groups(axes[2], len(methods))
    axes[2].set_yticks(y)
    axes[2].set_yticklabels(methods, fontsize=5.7)
    axes[2].invert_yaxis()
    axes[2].set_xlim(0, 1)
    axes[2].set_xlabel("Executed action ratio")
    axes[2].legend(frameon=False, ncol=2, loc="lower center", bbox_to_anchor=(0.5, 1.02), handlelength=1.0, columnspacing=0.8)
    clean_axes(axes[2], "x")
    panel_label(axes[2], "(c)")
    save_figure(fig, out_dirs["main_figures"], "fig13_online_tradeoff_and_actions_cn_v4", formats, dpi, audit, "generated_main_figures")


def plot_figS2(receipt: pd.DataFrame, out_dirs: dict, formats: list[str], dpi: int, audit: dict):
    methods = method_order(receipt["method"].unique())
    mat = receipt.set_index("method").reindex(methods)[["receipt_accept_ratio", "intent_execution_match_ratio", "fallback_none_ratio"]]
    fig, ax = plt.subplots(1, 1, figsize=(3.4, 3.1))
    cmap = LinearSegmentedColormap.from_list("receipt_integrity", ["white", REF_PALETTE["blue_very_pale"], REF_PALETTE["cyan_light"], REF_PALETTE["teal"]])
    im = ax.imshow(mat.to_numpy(dtype=float), aspect="auto", cmap=cmap, vmin=0, vmax=1)
    ax.set_title("SatEdgeSim receipt integrity", pad=4)
    ax.set_xticks(np.arange(3))
    ax.set_xticklabels(["Receipt\naccept", "Intent-exec.\nmatch", "No\nfallback"])
    ax.set_yticks(np.arange(len(methods)))
    ax.set_yticklabels(methods, fontsize=6.2)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, f"{mat.iloc[i, j]:.2f}", ha="center", va="center", fontsize=5.8)
    if bool((mat.round(12) == 1.0).all().all()):
        ax.text(0.5, -0.15, "All replayed decisions are receipt-consistent.", transform=ax.transAxes, ha="center", va="top", fontsize=6.4, color=REF_PALETTE["text"])
    fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
    save_figure(fig, out_dirs["appendix_figures"], "figS2_online_receipt_integrity_cn_v4", formats, dpi, audit, "generated_appendix_figures")


def fmt_num(x, digits=3):
    if pd.isna(x):
        return "--"
    return f"{float(x):.{digits}f}"


def fmt_ci(row, metric, digits=3):
    mean = row.get(f"{metric}_mean", np.nan)
    low = row.get(f"{metric}_ci95_low", np.nan)
    high = row.get(f"{metric}_ci95_high", np.nan)
    if pd.isna(mean):
        return "--"
    if pd.isna(low) or pd.isna(high):
        return fmt_num(mean, digits)
    return f"{fmt_num(mean, digits)} [{fmt_num(low, digits)}, {fmt_num(high, digits)}]"


def latex_escape(text: str) -> str:
    return str(text).replace("&", "\\&").replace("_", "\\_")


def write_table_summary(summary: pd.DataFrame, out_dir: Path, audit: dict):
    table = summary.sort_values("successRate_mean", ascending=False).copy()
    rule = table[table["type"].eq("Rule")]
    strongest_rule = rule.iloc[0]["method"] if not rule.empty else None
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Online multiseed SatEdgeSim summary.}",
        "\\label{tab:online_multiseed_summary_cn_v4}",
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
            method = f"{method}$^{{\\dagger}}$"
        lines.append(
            " & ".join(
                [
                    method,
                    latex_escape(row["type"]),
                    str(int(row["n"])),
                    fmt_ci(row, "successRate", 3),
                    fmt_ci(row, "averageEteDelay", 3),
                    fmt_ci(row, "energy_norm", 2),
                    fmt_ci(row, "delayFailureRate", 3),
                    fmt_ci(row, "mobilityFailureRate", 3),
                    fmt_ci(row, "intent_execution_match_ratio", 3),
                ]
            )
            + " \\\\"
        )
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "\\vspace{1mm}",
            "\\parbox{\\linewidth}{\\scriptsize All online metrics are computed over ten SatEdgeSim replay seeds. Confidence intervals are Student-t intervals over seed-level runs. $^{\\dagger}$MAPPO+MADDPG has the lowest normalized cumulative energy counter in the descriptive summary; statistical superiority claims should follow the Holm-corrected paired tests.}",
            "\\end{table}",
        ]
    )
    path = out_dir / "table_online_multiseed_summary_cn_v4.tex"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    audit["generated_tables"].append(str(path))


def write_table_tests(tests: pd.DataFrame, out_dir: Path, audit: dict):
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
        "\\caption{Paired online multiseed comparisons.}",
        "\\label{tab:online_pairwise_tests_cn_v4}",
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
            comp = f"{latex_escape(row['method_a'])} vs {latex_escape(row['method_b'])}"
            if bool(row["significant_holm_0.05"]):
                if metric == "energy_norm":
                    conclusion = "lower energy counter" if row["direction"] == "method_a_lower" else "higher energy counter"
                elif metric == "successRate":
                    conclusion = "higher success" if row["direction"] == "method_a_higher" else "lower success"
                else:
                    conclusion = "lower delay" if row["direction"] == "method_a_lower" else "higher delay"
                conclusion = f"significant after Holm correction; {conclusion}"
            else:
                conclusion = "not significant"
            ci = f"[{fmt_num(row['ci95_diff_low'], 3)}, {fmt_num(row['ci95_diff_high'], 3)}]"
            lines.append(
                f"{names[metric]} & {comp} & {int(row['n_paired_seeds'])} & {fmt_num(row['mean_diff'], 3)} & {ci} & {fmt_num(row['p_holm'], 3)} / {conclusion} \\\\"
            )
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table}",
        ]
    )
    path = out_dir / "table_online_pairwise_tests_cn_v4.tex"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    audit["generated_tables"].append(str(path))


def write_captions(out_dirs: dict, complete: bool, summary: pd.DataFrame):
    incomplete = "" if complete else " The current run matrix is incomplete; the figure should be interpreted diagnostically."
    best_rl_success = summary.loc[summary["type"].eq("RL"), "successRate_mean"].max()
    best_rule_success = summary.loc[summary["type"].eq("Rule"), "successRate_mean"].max()
    mappo_energy = summary.loc[summary["method"].eq("MAPPO+MADDPG"), "energy_norm_mean"]
    mappo_energy_text = "MAPPO+MADDPG shows a low normalized cumulative energy counter"
    if not mappo_energy.empty and pd.notna(mappo_energy.iloc[0]):
        mappo_energy_text = f"MAPPO+MADDPG has a mean normalized cumulative energy counter of {mappo_energy.iloc[0]:.2f}"
    text = f"""# Online Multiseed Captions

## Fig. 11
Online SatEdgeSim multiseed closed-loop performance. Panels (a)-(c) show seed-level distributions using violin, embedded box, and jittered points; Panel (d) decomposes failure ratios. All methods are evaluated over the same ten online seeds. Each point represents one SatEdgeSim replay run, not one task-level decision.{incomplete} Energy is the normalized cumulative energy counter.

## Fig. 12
Online seed consistency heatmaps. The complete seed matrix confirms that all RL checkpoints and rule baselines are evaluated on the same online split.{incomplete} Rank is computed within each online seed using success rate, where rank 1 is best.

## Fig. 13
Online success-delay-energy trade-off and executed action distributions. Similar upper-level action distributions with different energy outcomes suggest that the lower-level continuous resource control contributes to the online energy difference.

## Fig. S2
SatEdgeSim receipt integrity for online multiseed replay. All-one values indicate consistent abstract-action mapping and execution receipts.
"""
    (out_dirs["captions"] / "online_multiseed_cn_captions.md").write_text(text, encoding="utf-8")
    plan = f"""# Online Multiseed Experiment Placement

Recommended main-text items:
- Fig. 11: online multiseed performance distribution and failure decomposition.
- Fig. 12: seed consistency heatmap proving the common online split.
- Fig. 13: success-delay-energy trade-off and online action mix.
- Table 4: compact multiseed summary.
- Table 5: key paired comparisons.

Recommended appendix item:
- Fig. S2: receipt integrity dashboard.

Recommended wording:
Across ten SatEdgeSim online replay seeds, the best RL mean success rate is {best_rl_success:.3f}, while the best rule-baseline mean success rate is {best_rule_success:.3f}. {mappo_energy_text}, indicating a different online energy profile. Claims of statistical superiority should be limited to comparisons supported by Holm-corrected paired tests.

Avoid:
- RL significantly outperforms all online baselines.
- TriSatFlow wins the SatEdgeSim online experiment.
- The success-rate improvement is statistically significant.
"""
    (out_dirs["captions"] / "experiments_online_multiseed_plan.md").write_text(plan, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument("--formats", default="pdf,png,svg")
    args = parser.parse_args()

    setup_style()
    out_dirs = {
        "main_figures": args.output_dir / "main_figures",
        "appendix_figures": args.output_dir / "appendix_figures",
        "tables": args.output_dir / "tables",
        "captions": args.output_dir / "captions",
        "audit": args.output_dir / "audit",
    }
    for path in out_dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    formats = [f.strip() for f in args.formats.split(",") if f.strip()]
    runs = load_csv(args.data_dir / "online_runs_long.csv")
    summary = load_csv(args.data_dir / "online_summary_by_method.csv")
    tests = load_csv(args.data_dir / "online_pairwise_tests.csv")
    action = load_csv(args.data_dir / "online_action_distribution.csv")
    receipt = load_csv(args.data_dir / "online_receipt_integrity.csv")
    audit_path = args.data_dir / "online_multiseed_cn_v4_audit_data.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8")) if audit_path.exists() else {}
    audit.setdefault("generated_main_figures", [])
    audit.setdefault("generated_appendix_figures", [])
    audit.setdefault("generated_tables", [])

    plot_fig11(runs, tests, out_dirs, formats, args.dpi, audit)
    plot_fig12(args.data_dir, out_dirs, formats, args.dpi, audit)
    plot_fig13(summary, action, out_dirs, formats, args.dpi, audit)
    plot_figS2(receipt, out_dirs, formats, args.dpi, audit)
    write_table_summary(summary, out_dirs["tables"], audit)
    write_table_tests(tests, out_dirs["tables"], audit)
    write_captions(out_dirs, bool(audit.get("complete_for_main_claims", False)), summary)
    final_audit = out_dirs["audit"] / "online_multiseed_cn_v4_audit.json"
    final_audit.write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    print("Online multiseed CN v4 plotting complete")
    print(f"  Main figures: {len(audit['generated_main_figures'])}")
    print(f"  Appendix figures: {len(audit['generated_appendix_figures'])}")
    print(f"  Tables: {len(audit['generated_tables'])}")
    print(f"  Audit: {final_audit}")


if __name__ == "__main__":
    main()
