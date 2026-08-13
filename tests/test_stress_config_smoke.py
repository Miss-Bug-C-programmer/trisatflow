from __future__ import annotations

from pathlib import Path

from trisatflow.config import load_config
from scripts.run_stress_suite import run_suite


STRESS_DIR = Path("trisatflow/configs/stress")


def test_stress_configs_load() -> None:
    names = [
        "scale_16.yaml",
        "scale_32.yaml",
        "scale_64.yaml",
        "isl_sparse.yaml",
        "isl_dense.yaml",
        "gateway_low_visibility.yaml",
        "gateway_high_visibility.yaml",
        "burst_low.yaml",
        "burst_high.yaml",
        "deadline_tight.yaml",
        "deadline_loose.yaml",
        "mask_noise_mild.yaml",
        "mask_noise_severe.yaml",
        "domain_shift_satedgesim.yaml",
    ]
    for name in names:
        cfg = load_config(STRESS_DIR / name)
        assert cfg.scenario.episode_len <= 8
        assert cfg.device == "cpu"


def test_scale_16_tiny_step_runs(tmp_path) -> None:
    summary = run_suite(
        [STRESS_DIR / "scale_16.yaml"],
        policy="random_visible",
        checkpoint=None,
        episodes=1,
        steps=2,
        device="cpu",
        output_dir=tmp_path / "scale16",
    )
    assert summary["reset_ok_count"] == 1
    assert summary["step_ok_count"] == 1
    assert (tmp_path / "scale16" / "stress_results.csv").exists()


def test_scale_32_reset_shape_check_reports_checkpoint_blocker(tmp_path) -> None:
    summary = run_suite(
        [STRESS_DIR / "scale_32.yaml"],
        policy="checkpoint",
        checkpoint=None,
        episodes=1,
        steps=1,
        device="cpu",
        output_dir=tmp_path / "scale32",
    )
    assert summary["reset_ok_count"] == 1
    assert summary["blocked_count"] == 1
    text = (tmp_path / "scale32" / "stress_results.csv").read_text(encoding="utf-8")
    assert "checkpoint transfer not shape-verified" in text


def test_transfer_blocked_is_not_marked_success(tmp_path) -> None:
    summary = run_suite(
        [STRESS_DIR / "scale_32.yaml"],
        policy="checkpoint",
        checkpoint=None,
        episodes=1,
        steps=1,
        device="cpu",
        output_dir=tmp_path / "blocked",
    )
    assert summary["transfer_claim_supported"] is False
    text = (tmp_path / "blocked" / "stress_results.csv").read_text(encoding="utf-8")
    assert "transfer_blocked" in text
    assert "success" not in text.lower()

