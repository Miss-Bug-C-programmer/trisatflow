import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "paper_ready_v3" / "figures_v4_cn"
SEEDS = "202,303,404,505,606,707,808,909,1001,1103"


def run_cmd(args):
    subprocess.run(args, cwd=ROOT, check=True)


def test_online_multiseed_cn_v4_outputs():
    run_cmd(
        [
            sys.executable,
            "scripts/aggregate_online_multiseed_cn.py",
            "--rl-dir",
            "outputs/paper_ready_v3/satedgesim_replay_multiseed",
            "--baseline-dir",
            "outputs/paper_ready_v3/satedgesim_replay_baselines_multiseed",
            "--expected-online-seeds",
            SEEDS,
            "--output-dir",
            "outputs/paper_ready_v3/figures_v4_cn/figure_data",
        ]
    )
    run_cmd(
        [
            sys.executable,
            "scripts/plot_online_multiseed_cn_v4.py",
            "--data-dir",
            "outputs/paper_ready_v3/figures_v4_cn/figure_data",
            "--output-dir",
            "outputs/paper_ready_v3/figures_v4_cn",
            "--dpi",
            "600",
            "--formats",
            "pdf,png,svg",
        ]
    )

    runs = pd.read_csv(OUT / "figure_data" / "online_runs_long.csv")
    assert int((runs["type"] == "RL").sum()) == 40
    assert int((runs["type"] == "Rule").sum()) == 100
    assert runs.loc[runs["type"].eq("RL"), "source_is_test_seed_dir"].astype(bool).all()
    assert not runs.loc[runs["type"].eq("RL"), "run_dir"].str.endswith("summary.json").any()

    expected = [
        OUT / "main_figures" / "fig11_online_multiseed_performance_cn_v4.pdf",
        OUT / "main_figures" / "fig12_online_seed_consistency_heatmap_cn_v4.pdf",
        OUT / "main_figures" / "fig13_online_tradeoff_and_actions_cn_v4.pdf",
        OUT / "appendix_figures" / "figS2_online_receipt_integrity_cn_v4.pdf",
        OUT / "tables" / "table_online_multiseed_summary_cn_v4.tex",
        OUT / "tables" / "table_online_pairwise_tests_cn_v4.tex",
        OUT / "audit" / "online_multiseed_cn_v4_audit.json",
    ]
    for path in expected:
        assert path.exists(), path

    audit = json.loads((OUT / "audit" / "online_multiseed_cn_v4_audit.json").read_text(encoding="utf-8"))
    assert audit["complete_for_main_claims"] is True
    assert audit["decision_level_samples_used_for_tests"] is False
    assert audit["ignored_stale_rl_top_level_files"] is True
