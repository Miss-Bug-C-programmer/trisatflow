import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs" / "paper_ready_v3" / "figures"


def test_plot_paper_ready_v3_smoke():
    script = ROOT / "scripts" / "plot_paper_ready_v3.py"
    subprocess.run(
        [
            sys.executable,
            str(script),
            "--input-root",
            str(ROOT / "outputs" / "paper_ready_v3"),
            "--output-dir",
            str(OUTPUT_DIR),
            "--dpi",
            "600",
            "--formats",
            "pdf,png,svg",
        ],
        check=True,
        cwd=str(ROOT),
    )

    figure_names = [
        "fig7_offline_main_comparison",
        "fig8_offline_rule_baselines",
        "fig9_training_convergence_policy_mix",
        "fig10_ablation_multiobjective",
        "fig11_satedgesim_online_validation",
        "fig12_online_action_and_receipt",
    ]
    for name in figure_names:
        for ext in ["pdf", "png", "svg"]:
            assert (OUTPUT_DIR / f"{name}.{ext}").exists()

    for name in [
        "fig7_offline_main_comparison.csv",
        "fig8_offline_rule_baselines.csv",
        "fig9_training_policy_summary.csv",
        "fig10_ablation_summary.csv",
        "fig11_satedgesim_online_summary.csv",
        "fig12_action_receipt_summary.csv",
    ]:
        assert (OUTPUT_DIR / "figure_data" / name).exists()

    for name in ["table_offline_main.tex", "table_rule_baselines.tex", "table_satedgesim_online.tex"]:
        assert (OUTPUT_DIR / "tables" / name).exists()

    audit_path = OUTPUT_DIR / "visualization_audit.json"
    assert audit_path.exists()
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    assert audit["generated figures"]

    online_summary = (OUTPUT_DIR / "figure_data" / "fig11_satedgesim_online_summary.csv").read_text(encoding="utf-8")
    assert "RL" in online_summary or "Rule" in online_summary


def test_optional_missing_decision_log_does_not_crash():
    import importlib.util

    script = ROOT / "scripts" / "plot_paper_ready_v3.py"
    spec = importlib.util.spec_from_file_location("plot_paper_ready_v3", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    result = module.action_distribution_from_decision_log(ROOT / "outputs" / "paper_ready_v3" / "__missing_decision_log.csv")
    assert result["num_decisions"] == 0
    assert "receipt_accept_ratio" in result
