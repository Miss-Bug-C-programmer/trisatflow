import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs" / "paper_ready_v3" / "figures_v2"


def test_plot_paper_ready_v3_v2_outputs():
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
            "--paper-polish-v2",
        ],
        cwd=str(ROOT),
        check=True,
    )

    expected = [
        OUTPUT_DIR / "main_figures" / "fig7_offline_main_comparison_v2.pdf",
        OUTPUT_DIR / "main_figures" / "fig8_offline_rule_baselines_v2.pdf",
        OUTPUT_DIR / "main_figures" / "fig9_training_convergence_policy_mix_v2.pdf",
        OUTPUT_DIR / "main_figures" / "fig10_ablation_multiobjective_v2.pdf",
        OUTPUT_DIR / "main_figures" / "fig11_satedgesim_closed_loop_v2.pdf",
        OUTPUT_DIR / "appendix_figures" / "figS1_online_receipt_integrity.pdf",
        OUTPUT_DIR / "tables" / "table_offline_main_v2.tex",
        OUTPUT_DIR / "tables" / "table_rule_baselines_v2.tex",
        OUTPUT_DIR / "tables" / "table_satedgesim_online_v2.tex",
        OUTPUT_DIR / "captions" / "experiments_figure_table_plan.md",
        OUTPUT_DIR / "visualization_audit_v2.json",
    ]
    for path in expected:
        assert path.exists(), path

    for tex_path in (OUTPUT_DIR / "tables").glob("*.tex"):
        text = tex_path.read_text(encoding="utf-8")
        assert "textbackslash{}pm" not in text
        assert "checkpoint.pt" not in text

    audit = json.loads((OUTPUT_DIR / "visualization_audit_v2.json").read_text(encoding="utf-8"))
    assert audit["ci_method"] == "Student-t over independent training seeds"
    assert audit["fig12_moved_to_appendix"] is True
