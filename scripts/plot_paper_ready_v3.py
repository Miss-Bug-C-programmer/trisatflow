"""Paper-ready visualizations for outputs/paper_ready_v3.

Smoke test command when pytest is unavailable:

python scripts/plot_paper_ready_v3.py \
  --input-root outputs/paper_ready_v3 \
  --output-dir outputs/paper_ready_v3/figures \
  --dpi 600 \
  --formats pdf,png,svg
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd


PALETTE = {
    "blue": "#4080C0",
    "blue_dark": "#205090",
    "blue_light": "#D0E0F0",
    "cyan": "#10A090",
    "cyan_light": "#C0F0F0",
    "pink": "#F05A9D",
    "pink_light": "#F0B0D0",
    "purple": "#9B59C4",
    "purple_light": "#E0D0F0",
    "gray": "#708090",
    "gray_light": "#A0B0C0",
    "text": "#222222",
    "grid": "#D9D9D9",
}

CAMERA_COLORS = {
    "rose": "#EDB1AD",
    "green": "#A8D0A2",
    "lavender": "#C3BFDE",
    "aqua": "#91D8DB",
    "violet": "#857BB8",
    "pink": "#E96A97",
    "pink_mid": "#E9A1C3",
    "pink_light": "#F8D8E3",
    "pink_lightest": "#FBE5EC",
    "blue_lightest": "#E2EEF6",
    "blue_light": "#85B4E0",
    "blue_mid": "#92D4EB",
    "cyan": "#32B9CD",
    "blue": "#3E92CE",
    "ink": "#2B2B2B",
    "soft_grid": "#E5E5E5",
}

ALGO_COLORS = {
    "IPPO+MADDPG": "#3E92CE",
    "MAPPO+MADDPG": "#32B9CD",
    "IPPO+MASAC": "#E96A97",
    "MAPPO+MASAC": "#857BB8",
}

ALGO_FILLS = {
    "IPPO+MADDPG": "#E2EEF6",
    "MAPPO+MADDPG": "#D7F1F2",
    "IPPO+MASAC": "#F8D8E3",
    "MAPPO+MASAC": "#C3BFDE",
}

ACTION_COLORS = {
    "LOCAL": "#85B4E0",
    "NEIGHBOR": "#3E92CE",
    "GEO": "#32B9CD",
    "GROUND": "#E9A1C3",
}

BASELINE_COLORS = {
    "local_only": "#E2EEF6",
    "neighbor_only": "#3E92CE",
    "geo_only": "#32B9CD",
    "ground_only": "#F8D8E3",
    "random_visible": "#C3BFDE",
    "min_delay_greedy": "#A8D0A2",
    "min_energy_greedy": "#EDB1AD",
    "queue_aware_greedy": "#91D8DB",
    "mobility_risk_greedy": "#E9A1C3",
    "lyapunov_dpp_greedy": "#857BB8",
}

ALGO_ORDER = ["IPPO+MADDPG", "MAPPO+MADDPG", "MAPPO+MASAC", "IPPO+MASAC"]
RULE_ORDER = [
    "local_only",
    "neighbor_only",
    "geo_only",
    "ground_only",
    "random_visible",
    "min_delay_greedy",
    "min_energy_greedy",
    "queue_aware_greedy",
    "mobility_risk_greedy",
    "lyapunov_dpp_greedy",
]
ABLATION_ORDER = [
    "no_mask",
    "visibility_only",
    "completion_safe",
    "full_mask",
    "no_gnn",
    "static_gnn",
    "temporal_gnn",
    "no_cost_prior",
]
ACTION_ORDER = ["LOCAL", "NEIGHBOR", "GEO", "GROUND"]
ACTION_LABELS = {"LOCAL": "Local", "NEIGHBOR": "Neighbor", "GEO": "GEO", "GROUND": "Ground"}

OUTPUT_DIR = None
DPI = 600
AUDIT = {
    "warnings": [],
    "missing files": [],
    "missing columns": [],
    "generated figures": [],
    "generated tables": [],
}


def configure_style():
    plt.rcParams["mathtext.fontset"] = "stix"
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman", "Times", "STIXGeneral", "DejaVu Serif"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42
    plt.rcParams["svg.fonttype"] = "none"
    plt.rcParams["figure.facecolor"] = "white"
    plt.rcParams["axes.facecolor"] = "white"
    plt.rcParams["axes.edgecolor"] = CAMERA_COLORS["ink"]
    plt.rcParams["axes.labelcolor"] = CAMERA_COLORS["ink"]
    plt.rcParams["xtick.color"] = CAMERA_COLORS["ink"]
    plt.rcParams["ytick.color"] = CAMERA_COLORS["ink"]
    plt.rcParams["text.color"] = CAMERA_COLORS["ink"]
    plt.rcParams["axes.linewidth"] = 0.85
    plt.rcParams["lines.linewidth"] = 1.0
    plt.rcParams["patch.linewidth"] = 0.7
    plt.rcParams["font.size"] = 8
    plt.rcParams["axes.labelsize"] = 8
    plt.rcParams["axes.titlesize"] = 9
    plt.rcParams["xtick.labelsize"] = 7
    plt.rcParams["ytick.labelsize"] = 7
    plt.rcParams["legend.fontsize"] = 7
    plt.rcParams["savefig.bbox"] = "tight"


def warn(message):
    if message not in AUDIT["warnings"]:
        AUDIT["warnings"].append(message)
    print("WARNING:", message)


def note_missing_file(path):
    text = str(path)
    if text not in AUDIT["missing files"]:
        AUDIT["missing files"].append(text)


def note_missing_column(column):
    if column not in AUDIT["missing columns"]:
        AUDIT["missing columns"].append(column)


def mean_ci95(values):
    arr = pd.to_numeric(pd.Series(values), errors="coerce").dropna().to_numpy(dtype=float)
    if len(arr) == 0:
        return np.nan, np.nan, np.nan, np.nan
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
    ci = float(1.96 * std / np.sqrt(len(arr))) if len(arr) > 1 else 0.0
    return mean, std, mean - ci, mean + ci


def load_csv_safe(path):
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        note_missing_file(path)
        warn(f"Missing or empty CSV: {path}")
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception as exc:
        warn(f"Could not read CSV {path}: {exc}")
        return pd.DataFrame()


def load_json_safe(path):
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        note_missing_file(path)
        warn(f"Missing or empty JSON: {path}")
        return {}
    try:
        with path.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception as exc:
        warn(f"Could not read JSON {path}: {exc}")
        return {}


def save_figure(fig, name, formats):
    global OUTPUT_DIR, DPI
    paths = []
    for fmt in formats:
        fmt = fmt.strip().lower()
        if not fmt:
            continue
        path = OUTPUT_DIR / f"{name}.{fmt}"
        kwargs = {"dpi": DPI} if fmt == "png" else {}
        kwargs.update({"facecolor": "white", "edgecolor": "none", "bbox_inches": "tight", "pad_inches": 0.025})
        fig.savefig(path, format=fmt, **kwargs)
        paths.append(str(path))
    plt.close(fig)
    AUDIT["generated figures"].extend(paths)
    return paths


def clean_axes(ax, grid_axis="y"):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.85)
    ax.spines["bottom"].set_linewidth(0.85)
    if grid_axis:
        ax.grid(True, axis=grid_axis, color=CAMERA_COLORS["soft_grid"], linewidth=0.45, alpha=0.9)
        ax.set_axisbelow(True)


def panel_label(ax, label):
    ax.text(-0.08, 1.035, label, transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom")


def method_tick_label(method):
    return str(method).replace("+", "+\n")


def compact_label(name):
    mapping = {
        "IPPO+MADDPG": "IPPO+\nMADDPG",
        "MAPPO+MADDPG": "MAPPO+\nMADDPG",
        "IPPO+MASAC": "IPPO+\nMASAC",
        "MAPPO+MASAC": "MAPPO+\nMASAC",
        "local_only": "local",
        "neighbor_only": "neighbor",
        "geo_only": "geo",
        "ground_only": "ground",
        "random_visible": "random",
        "min_delay_greedy": "min delay",
        "min_energy_greedy": "min energy",
        "queue_aware_greedy": "queue",
        "mobility_risk_greedy": "mobility",
        "lyapunov_dpp_greedy": "Lyapunov",
    }
    return mapping.get(str(name), str(name).replace("_", " "))


def method_name(upper, lower):
    if pd.isna(upper) or pd.isna(lower):
        return ""
    return f"{str(upper).upper()}+{str(lower).upper()}"


def parse_int_token(text, prefix):
    text = str(text)
    if prefix not in text:
        return np.nan
    tail = text.split(prefix, 1)[1]
    digits = ""
    for ch in tail:
        if ch.isdigit():
            digits += ch
        else:
            break
    return int(digits) if digits else np.nan


def first_existing_column(df, columns, required_name=None):
    for col in columns:
        if col in df.columns:
            return col
    if required_name:
        note_missing_column(required_name)
    return None


def select_main_dir(input_root):
    legacy = input_root / "main_actua"
    corrected = input_root / "main_actual"
    if legacy.exists():
        return legacy
    if corrected.exists():
        return corrected
    note_missing_file(legacy)
    note_missing_file(corrected)
    warn("Neither main_actua nor main_actual directory exists.")
    return legacy


def aggregate_main_by_train_seed(sweep_summary):
    if sweep_summary.empty:
        return pd.DataFrame(), pd.DataFrame()
    df = sweep_summary.copy()
    if "status" in df.columns:
        df = df[df["status"].fillna("ok").astype(str).str.lower().eq("ok")]
    cost_col = first_existing_column(
        df,
        ["final_normalized_system_cost", "normalized_system_cost", "final_mean_system_cost", "mean_system_cost", "system_cost"],
        "main_cost",
    )
    if cost_col is None:
        return pd.DataFrame(), pd.DataFrame()
    if "phase" in df.columns and df["phase"].astype(str).str.lower().eq("test").any():
        df = df[df["phase"].astype(str).str.lower().eq("test")]
    elif "phase" in df.columns and df["phase"].astype(str).str.lower().eq("val").any():
        df = df[df["phase"].astype(str).str.lower().eq("val")]
    for col in ["upper_algo", "lower_algo", "train_seed"]:
        if col not in df.columns:
            note_missing_column(col)
            return pd.DataFrame(), pd.DataFrame()
    df["method"] = [method_name(u, l) for u, l in zip(df["upper_algo"], df["lower_algo"])]
    df["train_seed"] = pd.to_numeric(df["train_seed"], errors="coerce")
    df["eval_seed"] = pd.to_numeric(df["eval_seed"], errors="coerce") if "eval_seed" in df.columns else np.nan
    seed_rows = (
        df.groupby(["method", "upper_algo", "lower_algo", "train_seed"], dropna=False)
        .agg(
            mean_test_cost=(cost_col, "mean"),
            std_over_eval_seed=(cost_col, "std"),
            n_eval_or_test_seeds=(cost_col, "count"),
            eval_test_seeds=("eval_seed", lambda s: ",".join(str(int(x)) for x in sorted(pd.to_numeric(s, errors="coerce").dropna().unique()))),
        )
        .reset_index()
    )
    rows = []
    for method, group in seed_rows.groupby("method"):
        mean, std, low, high = mean_ci95(group["mean_test_cost"])
        rows.append(
            {
                "method": method,
                "mean_cost": mean,
                "std": std,
                "ci95_low": low,
                "ci95_high": high,
                "n_train_seeds": int(group["train_seed"].nunique()),
                "train_seeds": ",".join(str(int(x)) for x in sorted(group["train_seed"].dropna().unique())),
            }
        )
    summary = pd.DataFrame(rows)
    return seed_rows, summary


def aggregate_ablation_by_train_seed(ablation_sweep_summary):
    if ablation_sweep_summary.empty:
        return pd.DataFrame(), pd.DataFrame()
    df = ablation_sweep_summary.copy()
    if "status" in df.columns:
        df = df[df["status"].fillna("ok").astype(str).str.lower().eq("ok")]
    if "phase" in df.columns and df["phase"].astype(str).str.lower().eq("test").any():
        df = df[df["phase"].astype(str).str.lower().eq("test")]
    elif "phase" in df.columns and df["phase"].astype(str).str.lower().eq("val").any():
        df = df[df["phase"].astype(str).str.lower().eq("val")]
    cost_col = first_existing_column(
        df,
        ["final_normalized_system_cost", "normalized_system_cost", "final_mean_system_cost", "mean_system_cost", "system_cost"],
        "ablation_cost",
    )
    deadline_col = first_existing_column(
        df,
        ["final_mean_deadline_violation_ratio", "deadline_violation_ratio", "mean_deadline_violation_ratio", "final_mean_deadline_violation"],
        "ablation_deadline_violation",
    )
    if cost_col is None or deadline_col is None or "train_seed" not in df.columns:
        if "train_seed" not in df.columns:
            note_missing_column("train_seed")
        return pd.DataFrame(), pd.DataFrame()
    df["train_seed"] = pd.to_numeric(df["train_seed"], errors="coerce")
    seed_rows = (
        df.groupby(["ablation", "train_seed"], dropna=False)
        .agg(
            mean_test_cost=(cost_col, "mean"),
            mean_deadline_violation=(deadline_col, "mean"),
            n_eval_or_test_seeds=(cost_col, "count"),
        )
        .reset_index()
    )
    rows = []
    for ablation, group in seed_rows.groupby("ablation"):
        cost_mean, cost_std, cost_low, cost_high = mean_ci95(group["mean_test_cost"])
        dl_mean, dl_std, dl_low, dl_high = mean_ci95(group["mean_deadline_violation"])
        rows.append(
            {
                "ablation": ablation,
                "cost_mean": cost_mean,
                "cost_std": cost_std,
                "cost_ci95_low": cost_low,
                "cost_ci95_high": cost_high,
                "deadline_violation_mean": dl_mean,
                "deadline_violation_std": dl_std,
                "deadline_violation_ci95_low": dl_low,
                "deadline_violation_ci95_high": dl_high,
                "n_train_seeds": int(group["train_seed"].nunique()),
                "train_seeds": ",".join(str(int(x)) for x in sorted(group["train_seed"].dropna().unique())),
            }
        )
    return seed_rows, pd.DataFrame(rows)


def normalize_action(value):
    if pd.isna(value):
        return None
    if isinstance(value, (int, np.integer)):
        return {0: "LOCAL", 1: "NEIGHBOR", 2: "GEO", 3: "GROUND"}.get(int(value))
    text = str(value).strip().upper()
    if text in ["0", "0.0"]:
        return "LOCAL"
    if text in ["1", "1.0"]:
        return "NEIGHBOR"
    if text in ["2", "2.0"]:
        return "GEO"
    if text in ["3", "3.0"]:
        return "GROUND"
    for action in ACTION_ORDER:
        if action in text:
            return action
    return None


def bool_series(series):
    return series.astype(str).str.strip().str.lower().isin(["1", "1.0", "true", "yes", "accepted"])


def action_distribution_from_decision_log(path):
    df = load_csv_safe(path)
    result = {"num_decisions": 0}
    for action in ACTION_ORDER:
        result[f"action_{action.lower()}_ratio"] = np.nan
    result["receipt_accept_ratio"] = np.nan
    result["intent_execution_match_ratio"] = np.nan
    result["fallback_none_ratio"] = np.nan
    if df.empty:
        return result
    result["num_decisions"] = int(len(df))
    action_col = first_existing_column(
        df,
        [
            "executed_abstract_action_name",
            "policyUpperActionName",
            "final_policy_action_name",
            "upper_action_name",
            "action_name",
            "executed_abstract_action",
            "policy_upper_action",
        ],
        "online_action_name",
    )
    if action_col:
        actions = df[action_col].map(normalize_action)
        counts = actions.value_counts(dropna=True)
        denom = max(float(actions.notna().sum()), 1.0)
        for action in ACTION_ORDER:
            result[f"action_{action.lower()}_ratio"] = float(counts.get(action, 0) / denom)
    receipt_col = first_existing_column(df, ["receipt_accepted", "actionAccepted", "executionScheduled"], None)
    if receipt_col:
        result["receipt_accept_ratio"] = float(bool_series(df[receipt_col]).mean())
    intent_col = first_existing_column(df, ["intent_execution_match"], None)
    if intent_col:
        result["intent_execution_match_ratio"] = float(pd.to_numeric(df[intent_col], errors="coerce").fillna(0).clip(0, 1).mean())
    elif {"final_policy_action_name", "executed_abstract_action_name"}.issubset(df.columns):
        result["intent_execution_match_ratio"] = float(
            (df["final_policy_action_name"].map(normalize_action) == df["executed_abstract_action_name"].map(normalize_action)).mean()
        )
    fallback_col = first_existing_column(df, ["fallback_reason"], None)
    if fallback_col:
        fallback = df[fallback_col].astype(str).str.strip().str.lower()
        result["fallback_none_ratio"] = float(fallback.isin(["", "none", "nan", "null"]).mean())
    return result


def get_nested_metric(summary, final_metrics, key):
    if key in final_metrics:
        return final_metrics.get(key)
    if key in summary:
        return summary.get(key)
    nested = summary.get("final_metrics", {})
    if isinstance(nested, dict):
        return nested.get(key)
    return None


def method_from_online_name(name, run_type):
    if run_type == "RL":
        bits = name.split("_seed")[0].split("_")
        if len(bits) >= 2:
            return f"{bits[0].upper()}+{bits[1].upper()}"
        return name.upper()
    if "_seed" in name:
        return name.split("_seed")[0]
    return name


def collect_online_runs(input_root):
    rows = []
    replay_parent = input_root / "satedgesim_replay"
    baseline_parent = input_root / "satedgesim_replay_baselines"
    for parent_summary in [replay_parent / "summary.csv", baseline_parent / "summary.csv"]:
        if parent_summary.exists():
            warn("Parent SatEdgeSim summary.csv ignored; recursively using child run JSON/CSV files instead.")
    for summary_path in replay_parent.glob("*/*/summary.json"):
        run_dir = summary_path.parent
        checkpoint_dir = run_dir.parent.name
        summary = load_json_safe(summary_path)
        final_metrics = load_json_safe(run_dir / "final_metrics.json")
        decision_log = run_dir / "decision_log.csv"
        action_stats = action_distribution_from_decision_log(decision_log)
        train_seed = parse_int_token(checkpoint_dir, "_seed")
        test_seed = parse_int_token(run_dir.name, "test_seed_")
        row = {
            "method": method_from_online_name(checkpoint_dir, "RL"),
            "type": "RL",
            "run_name": f"{checkpoint_dir}/{run_dir.name}",
            "train_seed": train_seed,
            "test_seed": test_seed if not pd.isna(test_seed) else summary.get("seed"),
            "summary_path": str(summary_path),
            "final_metrics_path": str(run_dir / "final_metrics.json"),
            "decision_log_path": str(decision_log),
        }
        for key in ["successRate", "averageEteDelay", "energyConsumption", "delayFailureRate", "mobilityFailureRate"]:
            row[key] = get_nested_metric(summary, final_metrics, key)
        for key, value in action_stats.items():
            row[key] = value
        for key in ["receipt_accept_ratio", "intent_execution_match_ratio", "fallback_none_ratio"]:
            if key in summary:
                row[key] = summary.get(key)
        rows.append(row)
    for summary_path in baseline_parent.glob("*/summary.json"):
        run_dir = summary_path.parent
        summary = load_json_safe(summary_path)
        final_metrics = load_json_safe(run_dir / "final_metrics.json")
        decision_log = run_dir / "decision_log.csv"
        action_stats = action_distribution_from_decision_log(decision_log)
        test_seed = parse_int_token(run_dir.name, "_seed")
        row = {
            "method": method_from_online_name(run_dir.name, "Rule"),
            "type": "Rule",
            "run_name": run_dir.name,
            "train_seed": np.nan,
            "test_seed": test_seed if not pd.isna(test_seed) else summary.get("seed"),
            "summary_path": str(summary_path),
            "final_metrics_path": str(run_dir / "final_metrics.json"),
            "decision_log_path": str(decision_log),
        }
        for key in ["successRate", "averageEteDelay", "energyConsumption", "delayFailureRate", "mobilityFailureRate"]:
            row[key] = get_nested_metric(summary, final_metrics, key)
        for key, value in action_stats.items():
            row[key] = value
        for key in ["receipt_accept_ratio", "intent_execution_match_ratio", "fallback_none_ratio"]:
            if key in summary:
                row[key] = summary.get(key)
        rows.append(row)
    return pd.DataFrame(rows)


def write_csv(df, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def fmt_num(value, digits=3):
    if pd.isna(value):
        return "--"
    value = float(value)
    if value != 0 and (abs(value) >= 10000 or abs(value) < 0.001):
        return f"{value:.2e}"
    return f"{value:.{digits}f}"


def latex_escape(value):
    text = str(value)
    for old, new in [("\\", "\\textbackslash{}"), ("_", "\\_"), ("%", "\\%"), ("&", "\\&")]:
        text = text.replace(old, new)
    return text


def write_latex_table(df, path, columns, headers):
    path.parent.mkdir(parents=True, exist_ok=True)
    aligns = "l" + "c" * (len(columns) - 1)
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        f"\\begin{{tabular}}{{{aligns}}}",
        "\\hline",
        " & ".join(headers) + " \\\\",
        "\\hline",
    ]
    for _, row in df.iterrows():
        vals = [latex_escape(row.get(col, "--")) for col in columns]
        lines.append(" & ".join(vals) + " \\\\")
    lines.extend(["\\hline", "\\end{tabular}", "\\end{table}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")
    AUDIT["generated tables"].append(str(path))


def significance_note(significance_df):
    if significance_df.empty:
        return ""
    col = first_existing_column(significance_df, ["p_value_holm_float", "p_value_holm", "p_holm"], "p_holm")
    if col is None:
        return ""
    vals = pd.to_numeric(significance_df[col], errors="coerce").dropna()
    if len(vals) and (vals < 0.05).any():
        return ""
    return "Pairwise differences are not significant after Holm correction."


def plot_fig7(seed_rows, summary, significance_df, formats, figure_data_dir):
    data = seed_rows.merge(summary, on="method", how="left", suffixes=("", "_summary"))
    write_csv(data, figure_data_dir / "fig7_offline_main_comparison.csv")
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.75), gridspec_kw={"width_ratios": [1.0, 1.15]})
    ax = axes[0]
    ordered = summary.set_index("method").reindex(ALGO_ORDER).dropna(how="all").reset_index()
    x = np.arange(len(ordered))
    colors = [ALGO_COLORS.get(m, PALETTE["gray"]) for m in ordered["method"]]
    fills = [ALGO_FILLS.get(m, "#EEEEEE") for m in ordered["method"]]
    ax.bar(x, ordered["mean_cost"], width=0.58, color=fills, edgecolor=colors, linewidth=0.9)
    yerr = np.vstack([ordered["mean_cost"] - ordered["ci95_low"], ordered["ci95_high"] - ordered["mean_cost"]])
    ax.errorbar(x, ordered["mean_cost"], yerr=yerr, fmt="o", color=CAMERA_COLORS["ink"], ms=3, capsize=2.2, lw=0.8, zorder=3)
    for idx, method in enumerate(ordered["method"]):
        vals = seed_rows.loc[seed_rows["method"].eq(method), "mean_test_cost"].dropna().to_numpy(dtype=float)
        if len(vals):
            jitter = np.linspace(-0.12, 0.12, len(vals)) if len(vals) > 1 else np.array([0.0])
            ax.scatter(np.full(len(vals), idx) + jitter, vals, s=15, facecolor="white", edgecolor=colors[idx], linewidth=0.9, zorder=4)
    ax.set_xticks(x)
    ax.set_xticklabels([method_tick_label(m) for m in ordered["method"]], rotation=0, ha="center")
    ax.set_ylabel("Final normalized system cost\n(lower is better)")
    clean_axes(ax, "y")
    panel_label(ax, "(a)")

    ax = axes[1]
    heat = seed_rows.pivot_table(index="method", columns="train_seed", values="mean_test_cost", aggfunc="mean")
    heat = heat.reindex(ALGO_ORDER)
    cmap = LinearSegmentedColormap.from_list("paper_heat", [CAMERA_COLORS["blue_light"], CAMERA_COLORS["blue_lightest"], "white", CAMERA_COLORS["pink_light"], CAMERA_COLORS["pink"]])
    im = ax.imshow(heat.to_numpy(dtype=float), aspect="auto", cmap=cmap, interpolation="nearest")
    ax.set_xticks(np.arange(len(heat.columns)))
    ax.set_xticklabels([str(int(c)) for c in heat.columns])
    ax.set_yticks(np.arange(len(heat.index)))
    ax.set_yticklabels(heat.index)
    ax.set_xlabel("Independent train seed")
    for i in range(heat.shape[0]):
        for j in range(heat.shape[1]):
            val = heat.iloc[i, j]
            if pd.notna(val):
                ax.text(j, i, fmt_num(val, 1), ha="center", va="center", fontsize=6.5)
    note = significance_note(significance_df)
    if note:
        ax.text(0.02, -0.24, note, transform=ax.transAxes, fontsize=7, color=PALETTE["gray"], va="top")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03, label="Mean test cost")
    panel_label(ax, "(b)")
    fig.tight_layout()
    return save_figure(fig, "fig7_offline_main_comparison", formats)


def plot_fig8(rule_df, main_summary, formats, figure_data_dir):
    rule_rows = []
    if not rule_df.empty:
        for _, row in rule_df.iterrows():
            baseline = row.get("baseline", "")
            mean = row.get("normalized_system_cost_mean", row.get("final_normalized_system_cost_mean", np.nan))
            ci = row.get("normalized_system_cost_ci95", row.get("final_normalized_system_cost_ci95", np.nan))
            rule_rows.append(
                {
                    "method": baseline,
                    "type": "Rule",
                    "cost_mean": pd.to_numeric(pd.Series([mean]), errors="coerce").iloc[0],
                    "ci95_low": pd.to_numeric(pd.Series([mean]), errors="coerce").iloc[0] - pd.to_numeric(pd.Series([ci]), errors="coerce").fillna(0).iloc[0],
                    "ci95_high": pd.to_numeric(pd.Series([mean]), errors="coerce").iloc[0] + pd.to_numeric(pd.Series([ci]), errors="coerce").fillna(0).iloc[0],
                }
            )
    rl_rows = [
        {"method": r["method"], "type": "RL", "cost_mean": r["mean_cost"], "ci95_low": r["ci95_low"], "ci95_high": r["ci95_high"]}
        for _, r in main_summary.iterrows()
    ]
    data = pd.DataFrame(rule_rows + rl_rows)
    order = RULE_ORDER + ALGO_ORDER
    data["order"] = data["method"].map({m: i for i, m in enumerate(order)}).fillna(999)
    data = data.sort_values(["order", "cost_mean"]).drop(columns=["order"])
    write_csv(data, figure_data_dir / "fig8_offline_rule_baselines.csv")
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.6), gridspec_kw={"width_ratios": [1.0, 1.15]})
    for ax, zoom in zip(axes, [False, True]):
        shown = data.copy()
        if zoom:
            shown = shown[(shown["type"].eq("RL")) | (shown["method"].isin(["geo_only", "ground_only", "random_visible"]))]
            title = "Competitive range"
        else:
            title = "Full range"
        y = np.arange(len(shown))
        colors = []
        for _, row in shown.iterrows():
            if row["type"] == "RL":
                colors.append(ALGO_COLORS.get(row["method"], PALETTE["blue"]))
            else:
                colors.append(BASELINE_COLORS.get(row["method"], "#AEBBC6"))
        ax.hlines(y, 0, shown["cost_mean"], color=colors, lw=1.0, alpha=0.82)
        ax.scatter(shown["cost_mean"], y, s=24, color=colors, edgecolor=CAMERA_COLORS["ink"], linewidth=0.35, zorder=3)
        low = shown["cost_mean"] - shown["ci95_low"]
        high = shown["ci95_high"] - shown["cost_mean"]
        ax.errorbar(shown["cost_mean"], y, xerr=np.vstack([low, high]), fmt="none", ecolor=PALETTE["gray"], capsize=2, lw=0.65, zorder=2)
        ax.axvline(0, color=CAMERA_COLORS["soft_grid"], lw=0.75)
        ax.set_yticks(y)
        ax.set_yticklabels([compact_label(m) for m in shown["method"]])
        ax.invert_yaxis()
        ax.set_xlabel("Normalized system cost (lower is better)")
        ax.set_title(title)
        clean_axes(ax, "x")
        if zoom and not shown["cost_mean"].dropna().empty:
            lo = shown["ci95_low"].min()
            hi = shown["ci95_high"].max()
            pad = max((hi - lo) * 0.08, 1.0)
            ax.set_xlim(lo - pad, hi + pad)
    panel_label(axes[0], "(a)")
    panel_label(axes[1], "(b)")
    fig.tight_layout()
    return save_figure(fig, "fig8_offline_rule_baselines", formats)


def collect_training_curves(main_dir):
    rows = []
    for metrics_path in main_dir.glob("train/seed_*/upper_*__lower_*/metrics.csv"):
        df = load_csv_safe(metrics_path)
        if df.empty:
            continue
        cost_col = first_existing_column(df, ["normalized_system_cost", "system_cost", "mean_system_cost"], "training_cost")
        if cost_col is None or "episode" not in df.columns:
            continue
        parent = metrics_path.parent.name
        upper = parent.split("__")[0].replace("upper_", "")
        lower = parent.split("__")[1].replace("lower_", "") if "__" in parent else ""
        seed = parse_int_token(metrics_path.parent.parent.name, "seed_")
        method = method_name(upper, lower)
        curve = df[["episode", cost_col]].copy()
        curve["episode"] = pd.to_numeric(curve["episode"], errors="coerce")
        curve[cost_col] = pd.to_numeric(curve[cost_col], errors="coerce")
        curve = curve.dropna().sort_values("episode")
        curve["smooth_cost"] = curve[cost_col].rolling(window=800, min_periods=1, center=False).mean()
        curve["method"] = method
        curve["train_seed"] = seed
        rows.append(curve.rename(columns={cost_col: "raw_cost"}))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def action_summary_from_main(sweep_df):
    if sweep_df.empty:
        return pd.DataFrame()
    df = sweep_df.copy()
    pairs = {
        "LOCAL": ["upper_local_ratio", "local_ratio"],
        "NEIGHBOR": ["upper_neighbor_ratio", "neighbor_ratio"],
        "GEO": ["upper_geo_ratio", "geo_ratio"],
        "GROUND": ["upper_ground_ratio", "ground_ratio"],
    }

    def has_action_data(frame):
        for candidates in pairs.values():
            for col in candidates:
                if col in frame.columns and pd.to_numeric(frame[col], errors="coerce").notna().any():
                    return True
        return False

    if "phase" in df.columns:
        test_df = df[df["phase"].astype(str).str.lower().eq("test")]
        train_df = df[df["phase"].astype(str).str.lower().eq("train")]
        if not test_df.empty and has_action_data(test_df):
            df = test_df
        elif not train_df.empty and has_action_data(train_df):
            df = train_df
    if not {"upper_algo", "lower_algo"}.issubset(df.columns):
        return pd.DataFrame()
    df["method"] = [method_name(u, l) for u, l in zip(df["upper_algo"], df["lower_algo"])]
    rows = []
    for method, group in df.groupby("method"):
        row = {"method": method}
        for action, candidates in pairs.items():
            col = first_existing_column(group, candidates, None)
            if col:
                row[action] = pd.to_numeric(group[col], errors="coerce").mean()
            else:
                row[action] = np.nan
        rows.append(row)
    result = pd.DataFrame(rows)
    if not result.empty:
        vals = result[ACTION_ORDER].sum(axis=1)
        for action in ACTION_ORDER:
            result[action] = np.where(vals > 0, result[action] / vals, result[action])
    return result


def action_summary_from_metrics(main_dir):
    rows = []
    pairs = {
        "LOCAL": ["upper_local_ratio", "local_ratio"],
        "NEIGHBOR": ["upper_neighbor_ratio", "neighbor_ratio"],
        "GEO": ["upper_geo_ratio", "geo_ratio"],
        "GROUND": ["upper_ground_ratio", "ground_ratio"],
    }
    for metrics_path in main_dir.glob("train/seed_*/upper_*__lower_*/metrics.csv"):
        df = load_csv_safe(metrics_path)
        if df.empty:
            continue
        parent = metrics_path.parent.name
        upper = parent.split("__")[0].replace("upper_", "")
        lower = parent.split("__")[1].replace("lower_", "") if "__" in parent else ""
        row = {"method": method_name(upper, lower), "train_seed": parse_int_token(metrics_path.parent.parent.name, "seed_")}
        last = df.tail(1)
        for action, candidates in pairs.items():
            col = first_existing_column(last, candidates, None)
            row[action] = pd.to_numeric(last[col], errors="coerce").iloc[0] if col else np.nan
        rows.append(row)
    if not rows:
        return pd.DataFrame()
    seed_df = pd.DataFrame(rows)
    summary = seed_df.groupby("method", as_index=False)[ACTION_ORDER].mean()
    vals = summary[ACTION_ORDER].sum(axis=1)
    for action in ACTION_ORDER:
        summary[action] = np.where(vals > 0, summary[action] / vals, summary[action])
    return summary


def plot_fig9(main_dir, sweep_df, formats, figure_data_dir):
    curves = collect_training_curves(main_dir)
    action_summary = action_summary_from_main(sweep_df)
    if action_summary.empty or action_summary[ACTION_ORDER].isna().all().all():
        action_summary = action_summary_from_metrics(main_dir)
    write_csv(action_summary, figure_data_dir / "fig9_training_policy_summary.csv")
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.75), gridspec_kw={"width_ratios": [1.35, 1.0]})
    ax = axes[0]
    if not curves.empty:
        for method in ALGO_ORDER:
            group = curves[curves["method"].eq(method)]
            if group.empty:
                continue
            pivot = group.pivot_table(index="episode", columns="train_seed", values="smooth_cost", aggfunc="mean").sort_index()
            mean = pivot.mean(axis=1)
            std = pivot.std(axis=1).fillna(0)
            ci = 1.96 * std / np.sqrt(max(pivot.shape[1], 1))
            color = ALGO_COLORS.get(method, PALETTE["gray"])
            ax.plot(mean.index, mean.values, label=method, color=color, lw=1.2)
            ax.fill_between(mean.index.to_numpy(dtype=float), (mean - ci).to_numpy(dtype=float), (mean + ci).to_numpy(dtype=float), color=color, alpha=0.16, linewidth=0)
    ax.set_xlabel("Episode")
    ax.set_ylabel("800-episode moving average cost")
    ax.legend(frameon=False, ncol=2, loc="best", handlelength=1.6, columnspacing=0.8)
    clean_axes(ax, "y")
    panel_label(ax, "(a)")

    ax = axes[1]
    action_summary = action_summary.set_index("method").reindex(ALGO_ORDER).dropna(how="all")
    x = np.arange(len(action_summary))
    bottom = np.zeros(len(action_summary))
    for action in ACTION_ORDER:
        vals = pd.to_numeric(action_summary[action], errors="coerce").fillna(0).to_numpy(dtype=float)
        ax.bar(x, vals, bottom=bottom, color=ACTION_COLORS[action], edgecolor="white", linewidth=0.55, label=ACTION_LABELS[action])
        bottom += vals
    ax.set_xticks(x)
    ax.set_xticklabels([method_tick_label(m) for m in action_summary.index], rotation=0, ha="center")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Final action ratio")
    ax.legend(frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.18), handlelength=1.1, columnspacing=0.8)
    clean_axes(ax, "y")
    panel_label(ax, "(b)")
    fig.tight_layout()
    return save_figure(fig, "fig9_training_convergence_policy_mix", formats)


def plot_fig10(ablation_seed_rows, ablation_summary, formats, figure_data_dir):
    data = ablation_seed_rows.merge(ablation_summary, on="ablation", how="left", suffixes=("", "_summary"))
    write_csv(data, figure_data_dir / "fig10_ablation_summary.csv")
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.25), gridspec_kw={"width_ratios": [1.02, 1.08]})
    ax = axes[0]
    summary = ablation_summary.set_index("ablation").reindex(ABLATION_ORDER).dropna(how="all").reset_index()
    label_map = {
        "no_mask": "no mask",
        "visibility_only": "visibility",
        "completion_safe": "completion",
        "full_mask": "full",
        "no_gnn": "no GNN",
        "static_gnn": "static GNN",
        "temporal_gnn": "temporal GNN",
        "no_cost_prior": "no prior",
    }
    label_pos = {
        "no_mask": (-75.0, 0.385),
        "visibility_only": (-22.0, 0.064),
        "completion_safe": (-56.5, 0.128),
        "full_mask": (-56.5, 0.105),
        "no_gnn": (-38.0, 0.045),
        "static_gnn": (-38.0, 0.028),
        "temporal_gnn": (-74.0, 0.350),
        "no_cost_prior": (-18.5, 0.043),
    }
    for _, row in summary.iterrows():
        ab = row["ablation"]
        if ab in ["no_mask", "visibility_only", "completion_safe", "full_mask"]:
            marker, color, fill = "o", CAMERA_COLORS["blue"], "#CFE3F4"
        elif ab in ["no_gnn", "static_gnn", "temporal_gnn"]:
            marker, color, fill = "s", CAMERA_COLORS["violet"], "#DCD8EE"
        else:
            marker, color, fill = "D", CAMERA_COLORS["pink"], "#F8D8E3"
        size = 35 + min(abs(float(row.get("cost_std", 0) or 0)) * 1.2, 90)
        ax.scatter(row["cost_mean"], row["deadline_violation_mean"], s=size, marker=marker, color=fill, alpha=0.98, edgecolor=color, linewidth=1.0, zorder=3)
        ax.errorbar(row["cost_mean"], row["deadline_violation_mean"], xerr=[[row["cost_mean"] - row["cost_ci95_low"]], [row["cost_ci95_high"] - row["cost_mean"]]], fmt="none", ecolor=color, lw=0.7, capsize=2)
        ax.annotate(
            label_map.get(ab, ab),
            xy=(row["cost_mean"], row["deadline_violation_mean"]),
            xytext=label_pos.get(ab, (row["cost_mean"], row["deadline_violation_mean"])),
            textcoords="data",
            fontsize=6.6,
            va="center",
            arrowprops={"arrowstyle": "-", "color": "#A8A8A8", "lw": 0.45, "shrinkA": 1.5, "shrinkB": 2.0},
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.72, "pad": 0.3},
        )
    for ref in ["full_mask", "completion_safe"]:
        ref_row = summary[summary["ablation"].eq(ref)]
        if not ref_row.empty:
            ax.axvline(ref_row["cost_mean"].iloc[0], color="#CFCFCF", lw=0.75, ls="--", zorder=1)
            ax.axhline(ref_row["deadline_violation_mean"].iloc[0], color="#CFCFCF", lw=0.75, ls="--", zorder=1)
    ax.set_xlabel("Final normalized system cost\n(lower is better)")
    ax.set_ylabel("Deadline violation ratio\n(lower is better)")
    ax.text(0.98, 0.96, "Lower-left is better", transform=ax.transAxes, fontsize=7, color=PALETTE["gray"], ha="right", va="top")
    clean_axes(ax, "both")
    panel_label(ax, "(a)")

    ax = axes[1]
    heat = ablation_seed_rows.pivot_table(index="ablation", columns="train_seed", values="mean_test_cost", aggfunc="mean").reindex(ABLATION_ORDER)
    cmap = LinearSegmentedColormap.from_list("ablation_heat", [CAMERA_COLORS["aqua"], CAMERA_COLORS["blue_lightest"], "white", CAMERA_COLORS["pink_light"], CAMERA_COLORS["pink"]])
    cmap.set_bad(color="#F5F5F5")
    arr = heat.to_numpy(dtype=float)
    im = ax.imshow(np.ma.masked_invalid(arr), aspect="auto", cmap=cmap, interpolation="nearest")
    ax.set_xticks(np.arange(len(heat.columns)))
    ax.set_xticklabels([str(int(c)) for c in heat.columns])
    ax.set_yticks(np.arange(len(heat.index)))
    ax.set_yticklabels(heat.index)
    ax.set_xlabel("Independent train seed")
    for i in range(heat.shape[0]):
        for j in range(heat.shape[1]):
            val = heat.iloc[i, j]
            label = "--" if pd.isna(val) else fmt_num(val, 1)
            ax.text(j, i, label, ha="center", va="center", fontsize=6.4, color=PALETTE["text"])
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03, label="Mean test cost")
    panel_label(ax, "(b)")
    fig.tight_layout()
    return save_figure(fig, "fig10_ablation_multiobjective", formats)


def aggregate_online_for_plot(online_df):
    rows = []
    if online_df.empty:
        return pd.DataFrame()
    for (method, typ), group in online_df.groupby(["method", "type"]):
        row = {"method": method, "type": typ, "n_runs": len(group)}
        for metric in ["successRate", "averageEteDelay", "energyConsumption"]:
            mean, std, low, high = mean_ci95(pd.to_numeric(group[metric], errors="coerce"))
            row[f"{metric}_mean"] = mean
            row[f"{metric}_std"] = std
            row[f"{metric}_ci95_low"] = low
            row[f"{metric}_ci95_high"] = high
        rows.append(row)
    result = pd.DataFrame(rows)
    order = {m: i for i, m in enumerate(ALGO_ORDER + RULE_ORDER)}
    result["order"] = result["method"].map(order).fillna(999)
    return result.sort_values(["type", "order", "method"]).drop(columns=["order"])


def plot_fig11(online_df, formats, figure_data_dir):
    data = online_df.copy()
    min_energy = pd.to_numeric(data.get("energyConsumption", pd.Series(dtype=float)), errors="coerce").replace(0, np.nan).min()
    data["energyConsumption_norm_min"] = pd.to_numeric(data.get("energyConsumption", np.nan), errors="coerce") / min_energy if pd.notna(min_energy) else np.nan
    write_csv(data, figure_data_dir / "fig11_satedgesim_online_summary.csv")
    plot_df = aggregate_online_for_plot(data)
    if pd.notna(min_energy):
        plot_df["energyConsumption_plot_mean"] = plot_df["energyConsumption_mean"] / min_energy
        plot_df["energyConsumption_plot_ci95_low"] = plot_df["energyConsumption_ci95_low"] / min_energy
        plot_df["energyConsumption_plot_ci95_high"] = plot_df["energyConsumption_ci95_high"] / min_energy
    order = {m: i for i, m in enumerate(ALGO_ORDER + RULE_ORDER)}
    plot_df["order"] = plot_df["method"].map(order).fillna(999)
    plot_df = plot_df.sort_values(["type", "order", "method"]).drop(columns=["order"])
    y = np.arange(len(plot_df))
    labels = [compact_label(m) for m in plot_df["method"]]
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 3.75), sharey=True)
    specs = [
        ("successRate", "Success rate\n(higher is better)", "(a)", False),
        ("averageEteDelay", "Average E2E delay\n(lower is better)", "(b)", False),
        ("energyConsumption_plot", "Energy / min energy\n(lower is better)", "(c)", True),
    ]
    for ax, (metric, ylabel, label, energy_norm) in zip(axes, specs):
        values = pd.to_numeric(plot_df[f"{metric}_mean"], errors="coerce")
        colors = [
            ALGO_COLORS.get(m, "#AEBBC6") if t == "RL" else BASELINE_COLORS.get(m, "#AEBBC6")
            for m, t in zip(plot_df["method"], plot_df["type"])
        ]
        for yi, val, color in zip(y, values, colors):
            if pd.notna(val):
                ax.hlines(yi, 0, val, color=color, lw=1.0, alpha=0.75)
                ax.scatter(val, yi, s=24, facecolor=color, edgecolor=CAMERA_COLORS["ink"], linewidth=0.35, zorder=3)
        low = values - pd.to_numeric(plot_df[f"{metric}_ci95_low"], errors="coerce")
        high = pd.to_numeric(plot_df[f"{metric}_ci95_high"], errors="coerce") - values
        ax.errorbar(values, y, xerr=np.vstack([low.fillna(0), high.fillna(0)]), fmt="none", ecolor=PALETTE["gray"], lw=0.55, capsize=1.6, zorder=2)
        ax.set_xlabel(ylabel)
        clean_axes(ax, "x")
        panel_label(ax, label)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(labels)
    axes[0].invert_yaxis()
    if data["test_seed"].nunique(dropna=True) <= 1:
        axes[2].text(0.98, 0.02, "single test seed;\nvalidation only", transform=axes[2].transAxes, ha="right", va="bottom", fontsize=6.5, color=PALETTE["gray"])
    fig.tight_layout()
    return save_figure(fig, "fig11_satedgesim_online_validation", formats)


def plot_fig12(online_df, formats, figure_data_dir):
    data = online_df.copy()
    write_csv(data, figure_data_dir / "fig12_action_receipt_summary.csv")
    order = {m: i for i, m in enumerate(ALGO_ORDER + RULE_ORDER)}
    data["order"] = data["method"].map(order).fillna(999)
    data = data.sort_values(["type", "order", "run_name"]).drop(columns=["order"])
    labels = [compact_label(m) for m in data["method"]]
    y = np.arange(len(data))
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.85), gridspec_kw={"width_ratios": [1.15, 1.0]}, sharey=True)
    ax = axes[0]
    bottom = np.zeros(len(data))
    for action in ACTION_ORDER:
        vals = pd.to_numeric(data[f"action_{action.lower()}_ratio"], errors="coerce").fillna(0).to_numpy(dtype=float)
        ax.barh(y, vals, left=bottom, color=ACTION_COLORS[action], edgecolor="white", linewidth=0.55, label=ACTION_LABELS[action], height=0.72)
        bottom += vals
    ax.set_xlim(0, 1)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("Executed action ratio")
    ax.legend(frameon=False, ncol=2, loc="lower center", bbox_to_anchor=(0.5, 1.01), handlelength=1.1, columnspacing=0.8)
    clean_axes(ax, "x")
    panel_label(ax, "(a)")

    ax = axes[1]
    metrics = ["receipt_accept_ratio", "intent_execution_match_ratio", "fallback_none_ratio"]
    mat = data[metrics].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    cmap = LinearSegmentedColormap.from_list("receipt", ["#FFFFFF", CAMERA_COLORS["blue_lightest"], "#CFEFF1", CAMERA_COLORS["aqua"]])
    im = ax.imshow(np.ma.masked_invalid(mat), aspect="auto", vmin=0, vmax=1, cmap=cmap)
    ax.set_xticks(np.arange(len(metrics)))
    ax.set_xticklabels(["Receipt\naccept", "Intent =\nexecuted", "No\nfallback"])
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            val = mat[i, j]
            ax.text(j, i, "--" if pd.isna(val) else fmt_num(val, 2), ha="center", va="center", fontsize=6.2)
    if np.isfinite(mat).all() and np.allclose(mat, 1.0):
        ax.text(0.5, 1.03, "interface receipt consistent", transform=ax.transAxes, ha="center", va="bottom", fontsize=7, color=CAMERA_COLORS["cyan"])
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    panel_label(ax, "(b)")
    fig.tight_layout()
    return save_figure(fig, "fig12_online_action_and_receipt", formats)


T_CRITICAL_975 = {
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

METHOD_DISPLAY = {
    "local_only": "Local only",
    "neighbor_only": "Neighbor only",
    "geo_only": "GEO only",
    "ground_only": "Ground only",
    "random_visible": "Random-visible",
    "min_delay_greedy": "Min-delay greedy",
    "min_energy_greedy": "Min-energy greedy",
    "queue_aware_greedy": "Queue-aware greedy",
    "mobility_risk_greedy": "Mobility-risk greedy",
    "lyapunov_dpp_greedy": "Lyapunov-DPP greedy",
}


def reset_audit():
    AUDIT["warnings"] = []
    AUDIT["missing files"] = []
    AUDIT["missing columns"] = []
    AUDIT["generated figures"] = []
    AUDIT["generated tables"] = []


def display_method(method):
    return METHOD_DISPLAY.get(str(method), str(method))


def display_method_with_type(method, typ):
    return display_method(method)


def t_critical_975(df):
    if df <= 0:
        return np.nan
    if df <= 30:
        return T_CRITICAL_975.get(int(df), 1.96)
    return 1.96


def mean_ci95_t(values):
    arr = pd.to_numeric(pd.Series(values), errors="coerce").dropna().to_numpy(dtype=float)
    if len(arr) == 0:
        return np.nan, np.nan, np.nan
    mean = float(np.mean(arr))
    if len(arr) < 2:
        return mean, np.nan, np.nan
    std = float(np.std(arr, ddof=1))
    half = float(t_critical_975(len(arr) - 1) * std / np.sqrt(len(arr)))
    return mean, mean - half, mean + half


def ci95_t_from_stats(mean, std, n):
    mean = pd.to_numeric(pd.Series([mean]), errors="coerce").iloc[0]
    std = pd.to_numeric(pd.Series([std]), errors="coerce").iloc[0]
    n = pd.to_numeric(pd.Series([n]), errors="coerce").iloc[0]
    if pd.isna(mean) or pd.isna(std) or pd.isna(n) or n < 2:
        return np.nan, np.nan
    half = float(t_critical_975(int(n) - 1) * float(std) / np.sqrt(float(n)))
    return float(mean) - half, float(mean) + half


def mean_std_t_summary(values, clip_proportion=False):
    vals = pd.to_numeric(pd.Series(values), errors="coerce").dropna()
    mean, low, high = mean_ci95_t(vals)
    std = float(vals.std(ddof=1)) if len(vals) > 1 else np.nan
    clipped = False
    if clip_proportion and pd.notna(low) and pd.notna(high):
        old_low, old_high = low, high
        low = max(0.0, low)
        high = min(1.0, high)
        clipped = (low != old_low) or (high != old_high)
    return mean, std, low, high, clipped


def summarize_main_t(main_seed_rows):
    rows = []
    for method, group in main_seed_rows.groupby("method"):
        mean, std, low, high, _ = mean_std_t_summary(group["mean_test_cost"])
        rows.append(
            {
                "method": method,
                "mean_cost": mean,
                "std": std,
                "ci95_low": low,
                "ci95_high": high,
                "n_train_seeds": int(group["train_seed"].nunique()),
                "train_seeds": ",".join(str(int(x)) for x in sorted(group["train_seed"].dropna().unique())),
            }
        )
    return pd.DataFrame(rows)


def summarize_ablation_t(ablation_seed_rows):
    rows = []
    proportion_clipped = False
    for ablation, group in ablation_seed_rows.groupby("ablation"):
        cost_mean, cost_std, cost_low, cost_high, _ = mean_std_t_summary(group["mean_test_cost"])
        dl_mean, dl_std, dl_low, dl_high, clipped = mean_std_t_summary(group["mean_deadline_violation"], clip_proportion=True)
        proportion_clipped = proportion_clipped or clipped
        rows.append(
            {
                "ablation": ablation,
                "cost_mean": cost_mean,
                "cost_std": cost_std,
                "cost_ci95_low": cost_low,
                "cost_ci95_high": cost_high,
                "deadline_violation_mean": dl_mean,
                "deadline_violation_std": dl_std,
                "deadline_violation_ci95_low": dl_low,
                "deadline_violation_ci95_high": dl_high,
                "n_train_seeds": int(group["train_seed"].nunique()),
                "train_seeds": ",".join(str(int(x)) for x in sorted(group["train_seed"].dropna().unique())),
            }
        )
    return pd.DataFrame(rows), proportion_clipped


def save_figure_v2(fig, directory, name, formats, audit_v2, audit_key):
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    for fmt in formats:
        path = directory / f"{name}.{fmt}"
        kwargs = {"dpi": DPI} if fmt == "png" else {}
        kwargs.update({"facecolor": "white", "edgecolor": "none", "bbox_inches": "tight", "pad_inches": 0.025})
        fig.savefig(path, format=fmt, **kwargs)
        paths.append(str(path))
    plt.close(fig)
    audit_v2[audit_key].extend(paths)
    return paths


def write_caption(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.strip() + "\n", encoding="utf-8")


def add_no_holm_note(ax, significance_df):
    if significance_note(significance_df):
        ax.text(
            0.98,
            -0.30,
            "No Holm-significant pairwise difference",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=5.8,
            color=PALETTE["gray"],
            clip_on=False,
        )


def plot_fig7_v2(main_seed_rows, main_summary, significance_df, formats, dirs, audit_v2):
    data = main_seed_rows.merge(main_summary, on="method", how="left", suffixes=("", "_summary"))
    write_csv(data, dirs["figure_data"] / "fig7_offline_main_comparison_v2.csv")
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.35), gridspec_kw={"width_ratios": [1.0, 1.08]})

    ax = axes[0]
    ax.set_title("Offline RL combination comparison", fontsize=8, pad=4)
    ordered = main_summary.set_index("method").reindex(ALGO_ORDER).dropna(how="all").reset_index()
    x = np.arange(len(ordered))
    colors = [ALGO_COLORS.get(m, PALETTE["gray"]) for m in ordered["method"]]
    fills = [ALGO_FILLS.get(m, "#EEEEEE") for m in ordered["method"]]
    ax.bar(x, ordered["mean_cost"], width=0.58, color=fills, edgecolor=colors, linewidth=0.9)
    yerr = np.vstack([ordered["mean_cost"] - ordered["ci95_low"], ordered["ci95_high"] - ordered["mean_cost"]])
    ax.errorbar(x, ordered["mean_cost"], yerr=yerr, fmt="o", color=PALETTE["text"], ms=3, capsize=2.2, lw=0.8, zorder=3)
    for idx, method in enumerate(ordered["method"]):
        vals = main_seed_rows.loc[main_seed_rows["method"].eq(method), "mean_test_cost"].dropna().to_numpy(dtype=float)
        jitter = np.linspace(-0.12, 0.12, len(vals)) if len(vals) > 1 else np.array([0.0])
        ax.scatter(np.full(len(vals), idx) + jitter, vals, s=15, facecolor="white", edgecolor=colors[idx], linewidth=0.9, zorder=4)
    ax.set_xticks(x)
    ax.set_xticklabels([method_tick_label(m) for m in ordered["method"]])
    ax.set_ylabel("Normalized system cost ↓")
    clean_axes(ax, "y")
    panel_label(ax, "(a)")

    ax = axes[1]
    heat = main_seed_rows.pivot_table(index="method", columns="train_seed", values="mean_test_cost", aggfunc="mean").reindex(ALGO_ORDER)
    cmap = LinearSegmentedColormap.from_list("fig7_v2_heat", [CAMERA_COLORS["blue_light"], CAMERA_COLORS["blue_lightest"], "white", CAMERA_COLORS["pink_light"], CAMERA_COLORS["pink"]])
    im = ax.imshow(heat.to_numpy(dtype=float), aspect="auto", cmap=cmap, interpolation="nearest")
    ax.set_title("Training-seed sensitivity", fontsize=8, pad=4)
    ax.set_xticks(np.arange(len(heat.columns)))
    ax.set_xticklabels([str(int(c)) for c in heat.columns])
    ax.set_yticks(np.arange(len(heat.index)))
    ax.set_yticklabels(heat.index)
    ax.set_xlabel("Independent train seed")
    for i in range(heat.shape[0]):
        for j in range(heat.shape[1]):
            val = heat.iloc[i, j]
            if pd.notna(val):
                ax.text(j, i, fmt_num(val, 1), ha="center", va="center", fontsize=6.3)
    add_no_holm_note(ax, significance_df)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03, label="Mean test cost")
    panel_label(ax, "(b)")
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.22)
    save_figure_v2(fig, dirs["main_figures"], "fig7_offline_main_comparison_v2", formats, audit_v2, "generated_main_figures")
    write_caption(
        dirs["captions"] / "fig7_caption.md",
        "The bars report mean and Student-t 95% confidence intervals over independent training seeds. Lower normalized system cost is better. Pairwise differences are not claimed as statistically significant unless supported by Holm-corrected tests.",
    )


def rule_rows_with_t_ci(rule_df):
    rows = []
    for _, row in rule_df.iterrows():
        mean = row.get("normalized_system_cost_mean", row.get("final_normalized_system_cost_mean", np.nan))
        std = row.get("normalized_system_cost_std", row.get("final_normalized_system_cost_std", np.nan))
        n = row.get("n_seeds", row.get("n_train_seeds", np.nan))
        low, high = ci95_t_from_stats(mean, std, n)
        rows.append(
            {
                "method": row.get("baseline", ""),
                "type": "Rule",
                "cost_mean": pd.to_numeric(pd.Series([mean]), errors="coerce").iloc[0],
                "ci95_low": low,
                "ci95_high": high,
            }
        )
    return pd.DataFrame(rows)


def plot_fig8_v2(rule_df, main_summary, formats, dirs, audit_v2):
    rule_data = rule_rows_with_t_ci(rule_df)
    rl_data = main_summary.rename(columns={"mean_cost": "cost_mean"})[["method", "cost_mean", "ci95_low", "ci95_high"]].copy()
    rl_data["type"] = "RL"
    data = pd.concat([rule_data, rl_data], ignore_index=True)
    data["display_method"] = data["method"].map(display_method)
    rule_rank = rule_data.sort_values("cost_mean")["method"].head(6).tolist()
    rule_cost_subset = rule_data.loc[pd.to_numeric(rule_data["cost_mean"], errors="coerce") <= 10, "method"].tolist()
    competitive_rules = sorted(set(rule_rank + rule_cost_subset), key=lambda m: rule_data.set_index("method").loc[m, "cost_mean"])
    data["competitive_subset_rule"] = "all RL + rule cost top 6 or normalized_system_cost <= 10"
    data["in_competitive_subset"] = data["type"].eq("RL") | data["method"].isin(competitive_rules)
    write_csv(data, dirs["figure_data"] / "fig8_offline_rule_baselines_v2.csv")

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.15), gridspec_kw={"width_ratios": [1.0, 1.08]})
    for ax, shown, title in [
        (axes[0], data.sort_values("cost_mean"), "All methods"),
        (axes[1], data[data["in_competitive_subset"]].sort_values("cost_mean"), "Competitive subset"),
    ]:
        y = np.arange(len(shown))
        colors = [ALGO_COLORS.get(m, BASELINE_COLORS.get(m, PALETTE["gray_light"])) for m in shown["method"]]
        ax.hlines(y, 0, shown["cost_mean"], color=colors, lw=1.0, alpha=0.78)
        ax.scatter(shown["cost_mean"], y, s=24, color=colors, edgecolor=PALETTE["text"], linewidth=0.35, zorder=3)
        low = shown["cost_mean"] - shown["ci95_low"]
        high = shown["ci95_high"] - shown["cost_mean"]
        ax.errorbar(shown["cost_mean"], y, xerr=np.vstack([low.fillna(0), high.fillna(0)]), fmt="none", ecolor=PALETTE["gray"], capsize=2, lw=0.6)
        ax.axvline(0, color=PALETTE["grid"], lw=0.8)
        ax.set_yticks(y)
        ax.set_yticklabels(shown["display_method"], fontsize=6.5 if title == "All methods" else 7)
        ax.invert_yaxis()
        ax.set_title(title, fontsize=8)
        ax.set_xlabel("Normalized system cost ↓")
        clean_axes(ax, "x")
    panel_label(axes[0], "(a)")
    panel_label(axes[1], "(b)")
    fig.tight_layout()
    save_figure_v2(fig, dirs["main_figures"], "fig8_offline_rule_baselines_v2", formats, audit_v2, "generated_main_figures")
    write_caption(
        dirs["captions"] / "fig8_caption.md",
        "Panel (b) zooms into the competitive subset; Panel (a) preserves the full baseline range. Lower normalized cost is better.",
    )


def collect_training_curves_v2(main_dir, window=50):
    rows = []
    for metrics_path in main_dir.glob("train/seed_*/upper_*__lower_*/metrics.csv"):
        df = load_csv_safe(metrics_path)
        if df.empty:
            continue
        cost_col = first_existing_column(df, ["normalized_system_cost", "system_cost", "mean_system_cost"], "training_cost")
        if cost_col is None or "episode" not in df.columns:
            continue
        parent = metrics_path.parent.name
        upper = parent.split("__")[0].replace("upper_", "")
        lower = parent.split("__")[1].replace("lower_", "") if "__" in parent else ""
        seed = parse_int_token(metrics_path.parent.parent.name, "seed_")
        method = method_name(upper, lower)
        curve = df[["episode", cost_col]].copy()
        curve["episode"] = pd.to_numeric(curve["episode"], errors="coerce")
        curve[cost_col] = pd.to_numeric(curve[cost_col], errors="coerce")
        curve = curve.dropna().sort_values("episode")
        curve["smooth_cost"] = curve[cost_col].rolling(window=window, min_periods=1, center=False).mean()
        curve["method"] = method
        curve["train_seed"] = seed
        rows.append(curve.rename(columns={cost_col: "raw_cost"}))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def find_action_feasibility(main_dir):
    candidates = [
        "local_feasibility_ratio",
        "neighbor_feasibility_ratio",
        "geo_feasibility_ratio",
        "ground_feasibility_ratio",
        "upper_local_feasibility_ratio",
        "upper_neighbor_feasibility_ratio",
        "upper_geo_feasibility_ratio",
        "upper_ground_feasibility_ratio",
    ]
    rows = []
    for metrics_path in main_dir.glob("train/seed_*/upper_*__lower_*/metrics.csv"):
        df = load_csv_safe(metrics_path)
        if df.empty or not any(c in df.columns for c in candidates):
            continue
        parent = metrics_path.parent.name
        upper = parent.split("__")[0].replace("upper_", "")
        lower = parent.split("__")[1].replace("lower_", "") if "__" in parent else ""
        row = {"method": method_name(upper, lower)}
        last = df.tail(1)
        for action in ACTION_ORDER:
            cols = [c for c in candidates if c.lower().startswith(action.lower()) or c.lower().startswith(f"upper_{action.lower()}")]
            col = first_existing_column(last, cols, None)
            row[action] = pd.to_numeric(last[col], errors="coerce").iloc[0] if col else np.nan
        rows.append(row)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).groupby("method", as_index=False)[ACTION_ORDER].mean()


def plot_fig9_v2(main_dir, sweep_df, formats, dirs, audit_v2):
    curves = collect_training_curves_v2(main_dir, window=50)
    action_summary = action_summary_from_main(sweep_df)
    if action_summary.empty or action_summary[ACTION_ORDER].isna().all().all():
        action_summary = action_summary_from_metrics(main_dir)
    feasibility = find_action_feasibility(main_dir)
    audit_v2["action_feasibility_ratio_missing"] = feasibility.empty
    write_csv(action_summary, dirs["figure_data"] / "fig9_training_policy_summary_v2.csv")
    ncols = 3 if not feasibility.empty else 2
    fig, axes = plt.subplots(1, ncols, figsize=(7.0, 2.5), gridspec_kw={"width_ratios": [1.55, 1.0] + ([0.75] if ncols == 3 else [])})
    ax = axes[0]
    for method in ALGO_ORDER:
        group = curves[curves["method"].eq(method)]
        if group.empty:
            continue
        pivot = group.pivot_table(index="episode", columns="train_seed", values="smooth_cost", aggfunc="mean").sort_index()
        mean = pivot.mean(axis=1)
        std = pivot.std(axis=1).fillna(0)
        ci = 1.96 * std / np.sqrt(max(pivot.shape[1], 1))
        color = ALGO_COLORS.get(method, PALETTE["gray"])
        ax.plot(mean.index, mean.values, label=method, color=color, lw=1.05)
        ax.fill_between(mean.index.to_numpy(dtype=float), (mean - ci).to_numpy(dtype=float), (mean + ci).to_numpy(dtype=float), color=color, alpha=0.075, linewidth=0)
    ax.text(0.01, 0.04, "validation-based checkpoint selection", transform=ax.transAxes, fontsize=6.5, color=PALETTE["gray"])
    ax.set_xlabel("Episode")
    ax.set_ylabel("50-episode moving average cost")
    ax.legend(frameon=False, ncol=4, loc="lower center", bbox_to_anchor=(0.5, 1.01), handlelength=1.2, columnspacing=0.7, fontsize=6.4)
    clean_axes(ax, "y")
    panel_label(ax, "(a)")

    ax = axes[1]
    stacked = action_summary.set_index("method").reindex(ALGO_ORDER).dropna(how="all")
    x = np.arange(len(stacked))
    bottom = np.zeros(len(stacked))
    for action in ACTION_ORDER:
        vals = pd.to_numeric(stacked[action], errors="coerce").fillna(0).to_numpy(dtype=float)
        ax.bar(x, vals, bottom=bottom, color=ACTION_COLORS[action], edgecolor=PALETTE["text"], linewidth=0.55, label=ACTION_LABELS[action])
        bottom += vals
    ax.set_xticks(x)
    ax.set_xticklabels([method_tick_label(m) for m in stacked.index])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Final action ratio")
    ax.legend(frameon=False, ncol=2, loc="lower center", bbox_to_anchor=(0.5, 1.01), handlelength=1.0, columnspacing=0.7, fontsize=6.4)
    clean_axes(ax, "y")
    panel_label(ax, "(b)")

    if not feasibility.empty:
        ax = axes[2]
        feas = feasibility.set_index("method").reindex(ALGO_ORDER).dropna(how="all")
        mat = feas[ACTION_ORDER].to_numpy(dtype=float)
        cmap = LinearSegmentedColormap.from_list("feas_v2", ["white", PALETTE["cyan_light"], PALETTE["cyan"]])
        im = ax.imshow(np.ma.masked_invalid(mat), aspect="auto", vmin=0, vmax=1, cmap=cmap)
        ax.set_xticks(np.arange(len(ACTION_ORDER)))
        ax.set_xticklabels([ACTION_LABELS[a] for a in ACTION_ORDER], rotation=45, ha="right", fontsize=6)
        ax.set_yticks(np.arange(len(feas.index)))
        ax.set_yticklabels([method_tick_label(m) for m in feas.index], fontsize=6)
        ax.set_title("Feasibility", fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.05, pad=0.03)
        panel_label(ax, "(c)")
    fig.tight_layout()
    save_figure_v2(fig, dirs["main_figures"], "fig9_training_convergence_policy_mix_v2", formats, audit_v2, "generated_main_figures")
    write_caption(
        dirs["captions"] / "fig9_caption.md",
        "The curves are smoothed for readability and checkpoint selection is validation-based. The action distribution reports the learned upper-level offloading directions. A low Neighbor ratio should be interpreted together with action feasibility and cost-profile statistics.",
    )


def plot_fig10_v2(ablation_seed_rows, ablation_summary, formats, dirs, audit_v2, tradeoff_ablations):
    data = ablation_seed_rows.merge(ablation_summary, on="ablation", how="left", suffixes=("", "_summary"))
    write_csv(data, dirs["figure_data"] / "fig10_ablation_summary_v2.csv")
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.05), gridspec_kw={"width_ratios": [1.02, 1.08]})
    ax = axes[0]
    summary = ablation_summary.set_index("ablation").reindex(ABLATION_ORDER).dropna(how="all").reset_index()
    label_pos = {
        "no_mask": (-75.0, 0.385),
        "temporal_gnn": (-75.0, 0.345),
        "completion_safe": (-55.0, 0.125),
        "full_mask": (-56.5, 0.105),
        "visibility_only": (-19.0, 0.061),
        "no_cost_prior": (-9.0, 0.038),
        "no_gnn": (-37.0, 0.052),
        "static_gnn": (-37.0, 0.026),
    }
    for _, row in summary.iterrows():
        ab = row["ablation"]
        if ab in ["no_mask", "visibility_only", "completion_safe", "full_mask"]:
            marker, color, fill = "o", PALETTE["blue"], PALETTE["blue_light"]
        elif ab in ["no_gnn", "static_gnn", "temporal_gnn"]:
            marker, color, fill = "s", PALETTE["purple"], PALETTE["purple_light"]
        else:
            marker, color, fill = "D", PALETTE["pink"], PALETTE["pink_light"]
        face = "white" if ab in tradeoff_ablations or ab in ["no_mask", "temporal_gnn"] else fill
        lw = 1.2 if ab in tradeoff_ablations or ab in ["no_mask", "temporal_gnn"] else 0.9
        ax.scatter(row["cost_mean"], row["deadline_violation_mean"], s=58, marker=marker, color=face, edgecolor=color, linewidth=lw, zorder=3)
        xerr = [[row["cost_mean"] - row["cost_ci95_low"]], [row["cost_ci95_high"] - row["cost_mean"]]]
        yerr = [[row["deadline_violation_mean"] - row["deadline_violation_ci95_low"]], [row["deadline_violation_ci95_high"] - row["deadline_violation_mean"]]]
        ax.errorbar(row["cost_mean"], row["deadline_violation_mean"], xerr=xerr, yerr=yerr, fmt="none", ecolor=color, lw=0.6, capsize=1.8, zorder=2)
        ax.annotate(
            display_method(ab).replace("_", " "),
            xy=(row["cost_mean"], row["deadline_violation_mean"]),
            xytext=label_pos.get(ab, (row["cost_mean"], row["deadline_violation_mean"])),
            textcoords="data",
            fontsize=6.2,
            va="center",
            arrowprops={"arrowstyle": "-", "color": "#A8A8A8", "lw": 0.45, "shrinkA": 1.5, "shrinkB": 2.0},
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.72, "pad": 0.3},
        )
    full = summary[summary["ablation"].eq("full_mask")]
    if not full.empty:
        ax.axvline(full["cost_mean"].iloc[0], color=PALETTE["grid"], lw=0.75, ls="--")
        ax.axhline(full["deadline_violation_mean"].iloc[0], color=PALETTE["grid"], lw=0.75, ls="--")
    comp = summary[summary["ablation"].eq("completion_safe")]
    if not comp.empty:
        ax.axhline(comp["deadline_violation_mean"].iloc[0], color=PALETTE["gray_light"], lw=0.7, ls=":")
    ax.text(0.98, 0.96, "Lower-left is preferable", transform=ax.transAxes, fontsize=7, color=PALETTE["gray"], ha="right", va="top")
    ax.set_xlabel("Final normalized system cost ↓")
    ax.set_ylabel("Deadline violation ratio ↓")
    clean_axes(ax, "both")
    panel_label(ax, "(a)")

    ax = axes[1]
    heat = ablation_seed_rows.pivot_table(index="ablation", columns="train_seed", values="mean_test_cost", aggfunc="mean").reindex(ABLATION_ORDER)
    cmap = LinearSegmentedColormap.from_list("ablation_v2", [CAMERA_COLORS["aqua"], CAMERA_COLORS["blue_lightest"], "white", CAMERA_COLORS["pink_light"], CAMERA_COLORS["pink"]])
    cmap.set_bad(color="#F5F5F5")
    im = ax.imshow(np.ma.masked_invalid(heat.to_numpy(dtype=float)), aspect="auto", cmap=cmap, interpolation="nearest")
    ax.set_xticks(np.arange(len(heat.columns)))
    ax.set_xticklabels([str(int(c)) for c in heat.columns])
    ax.set_yticks(np.arange(len(heat.index)))
    ax.set_yticklabels(heat.index)
    ax.set_xlabel("Independent train seed")
    for i in range(heat.shape[0]):
        for j in range(heat.shape[1]):
            val = heat.iloc[i, j]
            ax.text(j, i, "--" if pd.isna(val) else fmt_num(val, 1), ha="center", va="center", fontsize=6.2)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03, label="Mean test cost")
    panel_label(ax, "(b)")
    fig.tight_layout()
    save_figure_v2(fig, dirs["main_figures"], "fig10_ablation_multiobjective_v2", formats, audit_v2, "generated_main_figures")
    write_caption(
        dirs["captions"] / "fig10_caption.md",
        "Lower-left is preferable. A lower cost with a higher deadline-violation ratio indicates a cost-safety trade-off rather than an unconditional improvement.",
    )


def plot_fig11_v2(online_df, formats, dirs, audit_v2):
    data = online_df.copy()
    min_energy = pd.to_numeric(data.get("energyConsumption", pd.Series(dtype=float)), errors="coerce").replace(0, np.nan).min()
    data["energy_norm"] = pd.to_numeric(data.get("energyConsumption", np.nan), errors="coerce") / min_energy if pd.notna(min_energy) else np.nan
    data["display_method"] = [display_method_with_type(m, t) for m, t in zip(data["method"], data["type"])]
    order = {m: i for i, m in enumerate(ALGO_ORDER + RULE_ORDER)}
    data["order"] = data["method"].map(order).fillna(999)
    data = data.sort_values(["type", "order", "method"]).drop(columns=["order"])
    write_csv(data, dirs["figure_data"] / "fig11_satedgesim_closed_loop_v2.csv")
    y = np.arange(len(data))
    fig, axes = plt.subplots(1, 4, figsize=(7.0, 3.8), sharey=True, gridspec_kw={"width_ratios": [0.95, 0.95, 0.95, 1.2]})
    rl_count = int((data["type"] == "RL").sum())
    for ax in axes:
        if rl_count > 0:
            ax.axhspan(-0.5, rl_count - 0.5, facecolor="#F7F9FB", edgecolor="none", zorder=0)
            ax.axhline(rl_count - 0.5, color=PALETTE["grid"], lw=0.7, zorder=1)
    specs = [
        ("successRate", "Success rate ↑", "(a)"),
        ("averageEteDelay", "Average E2E delay ↓", "(b)"),
        ("energy_norm", "Energy normalized\nby best run ↓", "(c)"),
    ]
    colors = [ALGO_COLORS.get(m, BASELINE_COLORS.get(m, PALETTE["gray_light"])) for m in data["method"]]
    for ax, (metric, xlabel, label) in zip(axes[:3], specs):
        vals = pd.to_numeric(data[metric], errors="coerce")
        for yi, val, color, typ in zip(y, vals, colors, data["type"]):
            if pd.notna(val):
                ax.hlines(yi, 0, val, color=color, lw=1.0, alpha=0.75)
                ax.scatter(val, yi, s=24, facecolor=color, edgecolor=PALETTE["text"] if typ == "Rule" else "white", linewidth=0.55, zorder=3)
        ax.set_xlabel(xlabel)
        clean_axes(ax, "x")
        panel_label(ax, label)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(data["display_method"], fontsize=6.4)
    axes[0].invert_yaxis()
    online_single_seed = data["test_seed"].nunique(dropna=True) <= 1

    ax = axes[3]
    left = np.zeros(len(data))
    for action in ACTION_ORDER:
        vals = pd.to_numeric(data[f"action_{action.lower()}_ratio"], errors="coerce").fillna(0).to_numpy(dtype=float)
        ax.barh(y, vals, left=left, height=0.72, color=ACTION_COLORS[action], edgecolor=PALETTE["text"], linewidth=0.55, label=ACTION_LABELS[action])
        left += vals
    ax.set_xlim(0, 1)
    ax.set_xlabel("Executed action ratio")
    ax.legend(frameon=False, ncol=2, loc="lower center", bbox_to_anchor=(0.5, 1.01), fontsize=6.2, handlelength=1.0)
    clean_axes(ax, "x")
    panel_label(ax, "(d)")
    fig.tight_layout()
    if online_single_seed:
        fig.subplots_adjust(bottom=0.13)
        fig.text(0.985, 0.018, "single online seed; validation only", ha="right", va="bottom", fontsize=6.0, color=PALETTE["gray"])
    save_figure_v2(fig, dirs["main_figures"], "fig11_satedgesim_closed_loop_v2", formats, audit_v2, "generated_main_figures")
    write_caption(
        dirs["captions"] / "fig11_caption.md",
        "The SatEdgeSim replay evaluates closed-loop action mapping and candidate-level execution. If only one online test seed is available, the results should be interpreted as validation evidence rather than statistical superiority evidence.",
    )


def plot_figS1_v2(online_df, formats, dirs, audit_v2):
    data = online_df.copy()
    data["display_method"] = [display_method_with_type(m, t) for m, t in zip(data["method"], data["type"])]
    write_csv(data, dirs["figure_data"] / "figS1_online_receipt_integrity.csv")
    metrics = ["receipt_accept_ratio", "intent_execution_match_ratio", "fallback_none_ratio"]
    mat = data[metrics].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(5.0, 3.6))
    cmap = LinearSegmentedColormap.from_list("receipt_s1", ["white", CAMERA_COLORS["blue_lightest"], CAMERA_COLORS["aqua"], CAMERA_COLORS["cyan"]])
    im = ax.imshow(np.ma.masked_invalid(mat), aspect="auto", vmin=0, vmax=1, cmap=cmap)
    ax.set_title("SatEdgeSim receipt integrity", fontsize=9)
    ax.set_xticks(np.arange(len(metrics)))
    ax.set_xticklabels(["Receipt\naccept", "Intent =\nexecuted", "No\nfallback"])
    ax.set_yticks(np.arange(len(data)))
    ax.set_yticklabels(data["display_method"], fontsize=6.5)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            val = mat[i, j]
            ax.text(j, i, "--" if pd.isna(val) else fmt_num(val, 2), ha="center", va="center", fontsize=6.2)
    ax.text(
        0.5,
        -0.12,
        "All-one values indicate consistent abstract-action mapping and execution receipts.",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=6.7,
        color=PALETTE["gray"],
    )
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    fig.tight_layout()
    save_figure_v2(fig, dirs["appendix_figures"], "figS1_online_receipt_integrity", formats, audit_v2, "generated_appendix_figures")


def latex_table_v2(path, caption, columns, rows, note=None, resize=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    body = []
    body.append("\\begin{table}[t]")
    body.append("\\centering")
    body.append(f"\\caption{{{caption}}}")
    if resize:
        body.append("\\resizebox{\\linewidth}{!}{%")
    body.append("\\begin{tabular}{" + "l" + "c" * (len(columns) - 1) + "}")
    body.append("\\toprule")
    body.append(" & ".join(columns) + " \\\\")
    body.append("\\midrule")
    body.extend(rows)
    body.append("\\bottomrule")
    body.append("\\end{tabular}")
    if resize:
        body.append("}%")
    if note:
        body.append(f"\\vspace{{1mm}}\\footnotesize{{{note}}}")
    body.append("\\end{table}")
    path.write_text("\n".join(body) + "\n", encoding="utf-8")


def parse_best_seed(path_text):
    val = parse_int_token(path_text, "seed_")
    return "--" if pd.isna(val) else str(int(val))


def rl_metrics_from_sweep(sweep_df):
    if sweep_df.empty:
        return pd.DataFrame()
    df = sweep_df.copy()
    if "phase" in df.columns and df["phase"].astype(str).str.lower().eq("test").any():
        df = df[df["phase"].astype(str).str.lower().eq("test")]
    if not {"upper_algo", "lower_algo"}.issubset(df.columns):
        return pd.DataFrame()
    df["method"] = [method_name(u, l) for u, l in zip(df["upper_algo"], df["lower_algo"])]
    metric_map = {
        "delay": ["final_mean_delay_s", "final_mean_delay", "mean_delay_s", "mean_delay"],
        "energy": ["final_mean_energy_j", "final_mean_energy", "mean_energy_j", "mean_energy"],
        "deadline": ["final_mean_deadline_violation_ratio", "final_mean_deadline_violation", "mean_deadline_violation_ratio"],
    }
    rows = []
    for method, group in df.groupby("method"):
        row = {"method": method}
        for out, cols in metric_map.items():
            col = first_existing_column(group, cols, None)
            row[out] = pd.to_numeric(group[col], errors="coerce").mean() if col else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def make_tables_v2(main_summary, summary_by_algorithm, significance_df, rule_df, sweep_df, online_df, dirs, audit_v2):
    main = main_summary.copy()
    if not summary_by_algorithm.empty and "best_checkpoint" in summary_by_algorithm.columns:
        s = summary_by_algorithm.copy()
        s["method"] = [method_name(u, l) for u, l in zip(s["upper_algo"], s["lower_algo"])]
        s["Best seed"] = s["best_checkpoint"].map(parse_best_seed)
        main = main.merge(s[["method", "Best seed"]], on="method", how="left")
    main = main.sort_values("mean_cost").reset_index(drop=True)
    main["Rank"] = np.arange(1, len(main) + 1)
    rows = []
    for _, row in main.iterrows():
        ci = "--" if pd.isna(row["ci95_low"]) else f"[{row['ci95_low']:.2f}, {row['ci95_high']:.2f}]"
        rows.append(
            f"{latex_escape(row['method'])} & {row['mean_cost']:.2f} & {row['std']:.2f} & {ci} & {row.get('Best seed', '--')} & {int(row['Rank'])} \\\\"
        )
    latex_table_v2(
        dirs["tables"] / "table_offline_main_v2.tex",
        "Offline RL algorithm-combination comparison.",
        ["Method", "Cost $\\downarrow$", "Std.", "95\\% CI", "Best seed", "Rank"],
        rows,
        note="Pairwise differences are not claimed as significant unless supported by Holm-corrected tests.",
    )
    audit_v2["generated_tables"].append(str(dirs["tables"] / "table_offline_main_v2.tex"))

    rl_metrics = rl_metrics_from_sweep(sweep_df)
    rl_rule = main_summary.rename(columns={"mean_cost": "cost"})[["method", "cost"]].merge(rl_metrics, on="method", how="left")
    rl_rule["type"] = "RL"
    rule_rows = []
    for _, row in rule_df.iterrows():
        rule_rows.append(
            {
                "method": row.get("baseline", ""),
                "type": "Rule",
                "cost": row.get("normalized_system_cost_mean", np.nan),
                "delay": row.get("mean_delay_s_mean", row.get("mean_delay_mean", np.nan)),
                "energy": row.get("mean_energy_j_mean", row.get("mean_energy_mean", np.nan)),
                "deadline": row.get("mean_deadline_violation_ratio_mean", row.get("mean_deadline_violation_mean", np.nan)),
            }
        )
    combined = pd.concat([rl_rule, pd.DataFrame(rule_rows)], ignore_index=True).sort_values("cost")
    best_rule = combined[combined["type"].eq("Rule")].sort_values("cost")["method"].iloc[0] if (combined["type"].eq("Rule")).any() else None
    rows = []
    for _, row in combined.iterrows():
        name = display_method(row["method"])
        if row["method"] == best_rule:
            name = f"\\textbf{{{latex_escape(name)}}}"
        else:
            name = latex_escape(name)
        rows.append(
            f"{name} & {row['type']} & {fmt_num(row['cost'], 2)} & {fmt_num(row['delay'], 2)} & {fmt_num(row['energy'], 3)} & {fmt_num(row['deadline'], 3)} \\\\"
        )
    latex_table_v2(
        dirs["tables"] / "table_rule_baselines_v2.tex",
        "Offline RL and rule-baseline comparison.",
        ["Method", "Type", "Cost $\\downarrow$", "Delay $\\downarrow$", "Energy $\\downarrow$", "Deadline viol. $\\downarrow$"],
        rows,
        note="The table reports offline normalized metrics; lower is better for all columns.",
        resize=True,
    )
    audit_v2["generated_tables"].append(str(dirs["tables"] / "table_rule_baselines_v2.tex"))

    online = online_df.copy()
    min_energy = pd.to_numeric(online.get("energyConsumption", pd.Series(dtype=float)), errors="coerce").replace(0, np.nan).min()
    online["energy_norm"] = pd.to_numeric(online.get("energyConsumption", np.nan), errors="coerce") / min_energy if pd.notna(min_energy) else np.nan
    rows = []
    for _, row in online.iterrows():
        rows.append(
            f"{latex_escape(display_method(row['method']))} & {row['type']} & {fmt_num(row.get('successRate'), 3)} & {fmt_num(row.get('averageEteDelay'), 2)} & {fmt_num(row.get('energy_norm'), 2)} & {fmt_num(row.get('intent_execution_match_ratio'), 2)} \\\\"
        )
    note = "Single online test seed; descriptive closed-loop validation only." if online["test_seed"].nunique(dropna=True) <= 1 else None
    latex_table_v2(
        dirs["tables"] / "table_satedgesim_online_v2.tex",
        "SatEdgeSim online closed-loop replay summary.",
        ["Method", "Type", "Success $\\uparrow$", "Delay $\\downarrow$", "Energy norm. $\\downarrow$", "Receipt match $\\uparrow$"],
        rows,
        note=note,
        resize=True,
    )
    audit_v2["generated_tables"].append(str(dirs["tables"] / "table_satedgesim_online_v2.tex"))


def write_experiments_plan(dirs):
    text = """
# Experiments Figure and Table Plan

## Recommended main-text items

- Fig. 7 -> Offline main results.
- Fig. 8 -> Rule baseline comparison.
- Fig. 9 -> Training behavior and learned offloading pattern.
- Fig. 10 -> Ablation study.
- Fig. 11 -> SatEdgeSim closed-loop validation.
- Table 1 -> Offline RL algorithm-combination comparison.
- Table 2 -> Offline rule baseline comparison.
- Table 3 -> Optional in main text; move to appendix if space is tight.

## Recommended appendix items

- Fig. S1 receipt integrity.
- Full checkpoint path table, if extra reproducibility detail is needed.

## Do not claim

- `RL significantly outperforms all methods`
- `RL wins SatEdgeSim online validation`
- `no_mask is better than full_mask`

These claims require supporting statistical tests and multi-seed online replay.

## Recommended phrasing

- `The method achieves the lowest mean cost, while Holm-corrected tests do not support a statistically significant pairwise difference.`
- `SatEdgeSim replay validates closed-loop action mapping and receipt consistency.`
- `Lower cost under no_mask or temporal_gnn comes with higher deadline violations, indicating a cost-safety trade-off.`
"""
    write_caption(dirs["captions"] / "experiments_figure_table_plan.md", text)


def run_paper_polish_v2(args):
    global OUTPUT_DIR, DPI
    reset_audit()
    input_root = Path(args.input_root)
    OUTPUT_DIR = Path(args.output_dir)
    DPI = args.dpi
    formats = [f.strip() for f in args.formats.split(",") if f.strip()]
    dirs = {
        "root": OUTPUT_DIR,
        "main_figures": OUTPUT_DIR / "main_figures",
        "appendix_figures": OUTPUT_DIR / "appendix_figures",
        "figure_data": OUTPUT_DIR / "figure_data",
        "tables": OUTPUT_DIR / "tables",
        "captions": OUTPUT_DIR / "captions",
    }
    for directory in dirs.values():
        directory.mkdir(parents=True, exist_ok=True)
    configure_style()

    audit_v2 = {
        "warnings": [],
        "ci_method": "Student-t over independent training seeds",
        "proportion_ci_clipped": False,
        "num_train_seeds": 0,
        "num_eval_seeds": 0,
        "num_online_rl_runs": 0,
        "num_online_rule_runs": 0,
        "online_single_seed": False,
        "holm_significant_pairs": 0,
        "no_holm_significant_difference_warning": False,
        "cost_safety_tradeoff_ablations": [],
        "fig12_moved_to_appendix": True,
        "table_checkpoint_paths_removed": True,
        "latex_pm_fixed": True,
        "generated_main_figures": [],
        "generated_appendix_figures": [],
        "generated_tables": [],
        "missing_files": [],
        "missing_columns": [],
    }

    main_dir = select_main_dir(input_root)
    summary_by_algorithm = load_csv_safe(input_root / "main_actual_summary" / "summary_by_algorithm.csv")
    significance_df = load_csv_safe(input_root / "main_actual_summary" / "significance_tests.csv")
    sweep_df = load_csv_safe(main_dir / "sweep_summary.csv")
    main_seed_rows, _ = aggregate_main_by_train_seed(sweep_df)
    main_summary = summarize_main_t(main_seed_rows)
    rule_df = load_csv_safe(input_root / "rules_actual" / "baseline_summary.csv")

    ablation_seed_frames = []
    for sweep_path in sorted((input_root / "ablations").glob("*/sweep_summary.csv")):
        df = load_csv_safe(sweep_path)
        if df.empty:
            continue
        df["ablation"] = sweep_path.parent.name
        seed_rows, _ = aggregate_ablation_by_train_seed(df)
        ablation_seed_frames.append(seed_rows)
    ablation_seed_rows = pd.concat(ablation_seed_frames, ignore_index=True) if ablation_seed_frames else pd.DataFrame()
    ablation_summary, clipped = summarize_ablation_t(ablation_seed_rows)
    audit_v2["proportion_ci_clipped"] = bool(clipped)
    online_df = collect_online_runs(input_root)

    sig_col = first_existing_column(significance_df, ["p_value_holm_float", "p_value_holm", "p_holm"], None) if not significance_df.empty else None
    holm_significant = int((pd.to_numeric(significance_df[sig_col], errors="coerce") < 0.05).sum()) if sig_col else 0
    audit_v2["holm_significant_pairs"] = holm_significant
    audit_v2["no_holm_significant_difference_warning"] = holm_significant == 0
    if holm_significant == 0:
        warn("No Holm-corrected significant pairwise difference detected; do not claim statistical superiority.")

    online_single = int(online_df["test_seed"].nunique(dropna=True)) <= 1 if not online_df.empty and "test_seed" in online_df.columns else True
    audit_v2["online_single_seed"] = bool(online_single)
    if online_single:
        warn("Online SatEdgeSim replay has only one test seed; use as closed-loop validation, not statistical superiority evidence.")

    tradeoff = []
    if not ablation_summary.empty and "full_mask" in set(ablation_summary["ablation"]):
        full = ablation_summary[ablation_summary["ablation"].eq("full_mask")].iloc[0]
        trade = ablation_summary[
            (ablation_summary["cost_mean"] < full["cost_mean"])
            & (ablation_summary["deadline_violation_mean"] > full["deadline_violation_mean"])
        ]
        tradeoff = trade["ablation"].tolist()
        if tradeoff:
            warn("Some ablations reduce cost but increase deadline violation; interpret as multi-objective trade-off.")
    audit_v2["cost_safety_tradeoff_ablations"] = tradeoff

    plot_fig7_v2(main_seed_rows, main_summary, significance_df, formats, dirs, audit_v2)
    plot_fig8_v2(rule_df, main_summary, formats, dirs, audit_v2)
    plot_fig9_v2(main_dir, sweep_df, formats, dirs, audit_v2)
    plot_fig10_v2(ablation_seed_rows, ablation_summary, formats, dirs, audit_v2, tradeoff)
    plot_fig11_v2(online_df, formats, dirs, audit_v2)
    plot_figS1_v2(online_df, formats, dirs, audit_v2)
    make_tables_v2(main_summary, summary_by_algorithm, significance_df, rule_df, sweep_df, online_df, dirs, audit_v2)
    write_experiments_plan(dirs)

    audit_v2["num_train_seeds"] = int(main_seed_rows["train_seed"].nunique(dropna=True)) if not main_seed_rows.empty else 0
    audit_v2["num_eval_seeds"] = int(sweep_df["eval_seed"].nunique(dropna=True)) if "eval_seed" in sweep_df.columns else 0
    audit_v2["num_online_rl_runs"] = int((online_df["type"] == "RL").sum()) if not online_df.empty else 0
    audit_v2["num_online_rule_runs"] = int((online_df["type"] == "Rule").sum()) if not online_df.empty else 0
    audit_v2["warnings"] = list(AUDIT["warnings"])
    audit_v2["missing_files"] = list(AUDIT["missing files"])
    audit_v2["missing_columns"] = list(AUDIT["missing columns"])
    audit_path = OUTPUT_DIR / "visualization_audit_v2.json"
    audit_path.write_text(json.dumps(audit_v2, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\nVisualization audit v2 summary")
    print(f"  CI method: {audit_v2['ci_method']}")
    print(f"  Main figures: {len(audit_v2['generated_main_figures'])}")
    print(f"  Appendix figures: {len(audit_v2['generated_appendix_figures'])}")
    print(f"  Tables: {len(audit_v2['generated_tables'])}")
    print(f"  Warnings: {len(audit_v2['warnings'])}")
    for message in audit_v2["warnings"]:
        print(f"   - {message}")
    print(f"  Audit JSON: {audit_path}")


def make_tables(main_summary_df, summary_by_algorithm, significance_df, rule_df, online_df, tables_dir):
    note = significance_note(significance_df)
    main = main_summary_df.copy()
    if not summary_by_algorithm.empty and "best_checkpoint" in summary_by_algorithm.columns:
        summary_by_algorithm = summary_by_algorithm.copy()
        summary_by_algorithm["method"] = [method_name(u, l) for u, l in zip(summary_by_algorithm["upper_algo"], summary_by_algorithm["lower_algo"])]
        main = main.merge(summary_by_algorithm[["method", "best_checkpoint"]], on="method", how="left")
    main["mean cost"] = main["mean_cost"].map(fmt_num)
    main["95% CI"] = ["[" + fmt_num(l) + ", " + fmt_num(h) + "]" for l, h in zip(main["ci95_low"], main["ci95_high"])]
    main["std_fmt"] = main["std"].map(fmt_num)
    main["significance note"] = note if note else "--"
    main = main.set_index("method").reindex(ALGO_ORDER).dropna(how="all").reset_index()
    write_latex_table(
        main,
        tables_dir / "table_offline_main.tex",
        ["method", "mean cost", "std_fmt", "95% CI", "best_checkpoint", "significance note"],
        ["method", "mean cost", "std", "95\\% CI", "best checkpoint", "significance note"],
    )

    rule_rows = []
    for _, row in rule_df.iterrows():
        baseline = row.get("baseline", "--")
        mean = row.get("normalized_system_cost_mean", np.nan)
        ci = row.get("normalized_system_cost_ci95", np.nan)
        rule_rows.append(
            {
                "baseline": baseline,
                "normalized cost mean ± 95% CI": f"{fmt_num(mean)} $\\pm$ {fmt_num(ci)}",
                "delay": fmt_num(row.get("mean_delay_s_mean", row.get("mean_delay_mean", np.nan))),
                "energy": fmt_num(row.get("mean_energy_j_mean", row.get("mean_energy_mean", np.nan))),
                "queue": fmt_num(row.get("mean_queue_length_tasks_mean", row.get("mean_queue_mean", np.nan))),
                "deadline violation ratio": fmt_num(row.get("mean_deadline_violation_ratio_mean", row.get("mean_deadline_violation_mean", np.nan))),
                "feasibility": fmt_num(row.get("mean_feasibility_mean", np.nan)),
            }
        )
    rule_table = pd.DataFrame(rule_rows)
    if not rule_table.empty:
        rule_table["order"] = rule_table["baseline"].map({m: i for i, m in enumerate(RULE_ORDER)}).fillna(999)
        rule_table = rule_table.sort_values(["order", "baseline"]).drop(columns=["order"])
    write_latex_table(
        rule_table,
        tables_dir / "table_rule_baselines.tex",
        ["baseline", "normalized cost mean ± 95% CI", "delay", "energy", "queue", "deadline violation ratio", "feasibility"],
        ["baseline", "normalized cost mean $\\pm$ 95\\% CI", "delay", "energy", "queue", "deadline violation ratio", "feasibility"],
    )

    online_rows = []
    for _, row in online_df.iterrows():
        online_rows.append(
            {
                "method": row.get("method", "--"),
                "type": row.get("type", "--"),
                "successRate": fmt_num(row.get("successRate", np.nan)),
                "averageEteDelay": fmt_num(row.get("averageEteDelay", np.nan)),
                "energyConsumption": fmt_num(row.get("energyConsumption", np.nan)),
                "delayFailureRate": fmt_num(row.get("delayFailureRate", np.nan)),
                "mobilityFailureRate": fmt_num(row.get("mobilityFailureRate", np.nan)),
                "receipt_accept_ratio": fmt_num(row.get("receipt_accept_ratio", np.nan)),
                "intent_execution_match_ratio": fmt_num(row.get("intent_execution_match_ratio", np.nan)),
            }
        )
    online_table = pd.DataFrame(online_rows)
    write_latex_table(
        online_table,
        tables_dir / "table_satedgesim_online.tex",
        [
            "method",
            "type",
            "successRate",
            "averageEteDelay",
            "energyConsumption",
            "delayFailureRate",
            "mobilityFailureRate",
            "receipt_accept_ratio",
            "intent_execution_match_ratio",
        ],
        [
            "method",
            "type",
            "successRate",
            "averageEteDelay",
            "energyConsumption",
            "delayFailureRate",
            "mobilityFailureRate",
            "receipt accept",
            "intent match",
        ],
    )


def main():
    global OUTPUT_DIR, DPI
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", default="outputs/paper_ready_v3")
    parser.add_argument("--output-dir", default="outputs/paper_ready_v3/figures")
    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument("--formats", default="pdf,png,svg")
    parser.add_argument("--paper-polish-v2", action="store_true")
    args = parser.parse_args()
    if args.paper_polish_v2:
        run_paper_polish_v2(args)
        return
    input_root = Path(args.input_root)
    OUTPUT_DIR = Path(args.output_dir)
    DPI = args.dpi
    formats = [f.strip() for f in args.formats.split(",") if f.strip()]
    figure_data_dir = OUTPUT_DIR / "figure_data"
    tables_dir = OUTPUT_DIR / "tables"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    figure_data_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    configure_style()

    main_dir = select_main_dir(input_root)
    summary_by_algorithm = load_csv_safe(input_root / "main_actual_summary" / "summary_by_algorithm.csv")
    significance_df = load_csv_safe(input_root / "main_actual_summary" / "significance_tests.csv")
    sweep_df = load_csv_safe(main_dir / "sweep_summary.csv")
    main_seed_rows, main_summary = aggregate_main_by_train_seed(sweep_df)

    rule_df = load_csv_safe(input_root / "rules_actual" / "baseline_summary.csv")
    baseline_episode_metrics = load_csv_safe(input_root / "rules_actual" / "baseline_episode_metrics.csv")
    if baseline_episode_metrics.empty:
        warn("baseline_episode_metrics.csv unavailable; using baseline_summary.csv only.")

    ablation_seed_frames = []
    ablation_summary_frames = []
    for sweep_path in sorted((input_root / "ablations").glob("*/sweep_summary.csv")):
        ablation = sweep_path.parent.name
        df = load_csv_safe(sweep_path)
        if df.empty:
            continue
        df["ablation"] = ablation
        seed_rows, summary = aggregate_ablation_by_train_seed(df)
        ablation_seed_frames.append(seed_rows)
        ablation_summary_frames.append(summary)
    ablation_seed_rows = pd.concat(ablation_seed_frames, ignore_index=True) if ablation_seed_frames else pd.DataFrame()
    ablation_summary = pd.concat(ablation_summary_frames, ignore_index=True) if ablation_summary_frames else pd.DataFrame()

    online_df = collect_online_runs(input_root)

    note = significance_note(significance_df)
    sig_col = first_existing_column(significance_df, ["p_value_holm_float", "p_value_holm", "p_holm"], None) if not significance_df.empty else None
    any_holm = bool((pd.to_numeric(significance_df[sig_col], errors="coerce") < 0.05).any()) if sig_col else False
    AUDIT["whether significance tests contain any p_holm < 0.05"] = any_holm
    if note:
        warn("No Holm-corrected significant pairwise difference detected; do not claim statistical superiority.")

    online_test_seed_count = int(online_df["test_seed"].nunique(dropna=True)) if not online_df.empty and "test_seed" in online_df.columns else 0
    AUDIT["whether online replay has only one test seed"] = online_test_seed_count <= 1
    if online_test_seed_count <= 1:
        warn("Online SatEdgeSim replay has only one test seed; use as closed-loop validation, not statistical superiority evidence.")

    if not ablation_summary.empty and "full_mask" in set(ablation_summary["ablation"]):
        full = ablation_summary[ablation_summary["ablation"].eq("full_mask")].iloc[0]
        trade = ablation_summary[
            (ablation_summary["cost_mean"] < full["cost_mean"])
            & (ablation_summary["deadline_violation_mean"] > full["deadline_violation_mean"])
        ]
        AUDIT["whether any ablation has lower cost but higher deadline violation than full_mask"] = bool(len(trade))
        if len(trade):
            warn("Some ablations reduce cost but increase deadline violation; interpret as multi-objective trade-off.")
    else:
        AUDIT["whether any ablation has lower cost but higher deadline violation than full_mask"] = False

    plot_fig7(main_seed_rows, main_summary, significance_df, formats, figure_data_dir)
    plot_fig8(rule_df, main_summary, formats, figure_data_dir)
    plot_fig9(main_dir, sweep_df, formats, figure_data_dir)
    plot_fig10(ablation_seed_rows, ablation_summary, formats, figure_data_dir)
    plot_fig11(online_df, formats, figure_data_dir)
    plot_fig12(online_df, formats, figure_data_dir)
    make_tables(main_summary, summary_by_algorithm, significance_df, rule_df, online_df, tables_dir)

    AUDIT["number of RL train seeds"] = int(main_seed_rows["train_seed"].nunique(dropna=True)) if not main_seed_rows.empty else 0
    AUDIT["number of RL eval/test seeds"] = int(sweep_df["eval_seed"].nunique(dropna=True)) if "eval_seed" in sweep_df.columns else 0
    AUDIT["number of online RL runs"] = int((online_df["type"] == "RL").sum()) if not online_df.empty else 0
    AUDIT["number of online baseline runs"] = int((online_df["type"] == "Rule").sum()) if not online_df.empty else 0
    AUDIT["whether parent satedgesim summary.csv was ignored"] = bool(
        (input_root / "satedgesim_replay" / "summary.csv").exists()
        or (input_root / "satedgesim_replay_baselines" / "summary.csv").exists()
    )
    AUDIT["input root"] = str(input_root)
    AUDIT["output dir"] = str(OUTPUT_DIR)
    audit_path = OUTPUT_DIR / "visualization_audit.json"
    audit_path.write_text(json.dumps(AUDIT, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\nVisualization audit summary")
    print(f"  RL train seeds: {AUDIT['number of RL train seeds']}")
    print(f"  RL eval/test seeds: {AUDIT['number of RL eval/test seeds']}")
    print(f"  Online RL runs: {AUDIT['number of online RL runs']}")
    print(f"  Online baseline runs: {AUDIT['number of online baseline runs']}")
    print(f"  Generated figures: {len(AUDIT['generated figures'])}")
    print(f"  Generated tables: {len(AUDIT['generated tables'])}")
    print(f"  Warnings: {len(AUDIT['warnings'])}")
    for message in AUDIT["warnings"]:
        print(f"   - {message}")
    print(f"  Audit JSON: {audit_path}")


if __name__ == "__main__":
    main()
