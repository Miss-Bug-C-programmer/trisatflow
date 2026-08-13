from __future__ import annotations

import warnings

import pytest

from trisatflow.config import load_config


def test_legacy_configs_wrapper_emits_deprecation_warning() -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        cfg = load_config("configs/satedgesim_trace_mixed_v2.yaml")
    assert cfg.scenario.n_leo > 0
    assert any("deprecated" in str(item.message).lower() for item in caught)


def test_invalid_action_mask_mode_fails_validation(tmp_path) -> None:
    cfg_path = tmp_path / "bad_mask.yaml"
    cfg_path.write_text(
        "\n".join(
            [
                "scenario:",
                "  action_mask_mode: invalid_mask",
            ]
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="action_mask_mode"):
        load_config(cfg_path)


def test_invalid_algorithm_name_fails_validation(tmp_path) -> None:
    cfg_path = tmp_path / "bad_algo.yaml"
    cfg_path.write_text(
        "\n".join(
            [
                "algo:",
                "  upper_algo: unknown_algo",
            ]
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="upper_algo"):
        load_config(cfg_path)


def test_invalid_physical_units_fails_validation(tmp_path) -> None:
    cfg_path = tmp_path / "bad_units.yaml"
    cfg_path.write_text(
        "\n".join(
            [
                "scenario:",
                "  delay_s_per_unit: 0",
            ]
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="delay_s_per_unit"):
        load_config(cfg_path)


def test_seed_split_overlap_rejected_without_debug_flag(tmp_path) -> None:
    cfg_path = tmp_path / "bad_seed_split.yaml"
    cfg_path.write_text(
        "\n".join(
            [
                "experiment:",
                "  split:",
                "    train_seeds: [13, 21]",
                "    val_seeds: [21]",
                "    test_seeds: [202]",
                "    allow_debug_seed_overlap: false",
            ]
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="overlap"):
        load_config(cfg_path)
