import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "paper_ready_v3" / "figures_v6_cn"


def test_online_multiseed_cn_v6_outputs():
    subprocess.run(
        [
            sys.executable,
            "scripts/plot_online_multiseed_cn_v6_camera_ready.py",
            "--data-dir",
            "outputs/paper_ready_v3/figures_v4_cn/figure_data",
            "--output-dir",
            "outputs/paper_ready_v3/figures_v6_cn",
            "--dpi",
            "600",
            "--formats",
            "pdf,png,svg",
        ],
        cwd=ROOT,
        check=True,
    )
    expected = [
        OUT / "main_figures" / "fig11_online_core_performance_cn_v6.pdf",
        OUT / "main_figures" / "fig12_online_failure_action_cn_v6.pdf",
        OUT / "main_figures" / "fig13_online_tradeoff_cn_v6.pdf",
        OUT / "appendix_figures" / "figS2_online_seed_matrix_cn_v6.pdf",
        OUT / "appendix_figures" / "figS3_online_receipt_integrity_cn_v6.pdf",
        OUT / "tables" / "table_online_multiseed_summary_cn_v6.tex",
        OUT / "tables" / "table_online_pairwise_tests_cn_v6.tex",
        OUT / "audit" / "online_multiseed_cn_v6_audit.json",
    ]
    for path in expected:
        assert path.exists(), path
    audit = json.loads((OUT / "audit" / "online_multiseed_cn_v6_audit.json").read_text(encoding="utf-8"))
    assert audit["complete_for_main_claims"] is True
    assert audit["decision_level_samples_used_for_tests"] is False
    assert audit["ignored_stale_rl_top_level_files"] is True
