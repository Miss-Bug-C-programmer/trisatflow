from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from trisatflow.config import canonical_train_config_dict, load_config, save_config


def _write_yaml(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_scenario_physical_is_canonical_source(tmp_path: Path) -> None:
    cfg_path = tmp_path / "scenario_physical.yaml"
    _write_yaml(
        cfg_path,
        [
            "output_dir: outputs/scenario_physical",
            "scenario:",
            "  physical:",
            "    enabled: true",
            "reward:",
            "  mode: physical_weighted",
        ],
    )

    cfg = load_config(cfg_path)

    assert cfg.scenario.physical.enabled is True
    assert cfg.physical is cfg.scenario.physical


def test_top_level_physical_is_migrated_with_deprecation_warning(tmp_path: Path) -> None:
    cfg_path = tmp_path / "legacy_top_level_physical.yaml"
    _write_yaml(
        cfg_path,
        [
            "output_dir: outputs/legacy_top_level_physical",
            "physical:",
            "  enabled: true",
            "reward:",
            "  mode: physical_weighted",
        ],
    )

    with pytest.warns(UserWarning, match="top-level physical has been migrated to scenario.physical"):
        cfg = load_config(cfg_path)

    assert cfg.scenario.physical.enabled is True
    assert cfg.physical is cfg.scenario.physical


def test_conflicting_top_level_and_scenario_physical_fails_fast(tmp_path: Path) -> None:
    cfg_path = tmp_path / "conflicting_physical.yaml"
    _write_yaml(
        cfg_path,
        [
            "output_dir: outputs/conflicting_physical",
            "physical:",
            "  enabled: true",
            "scenario:",
            "  physical:",
            "    enabled: false",
            "reward:",
            "  mode: legacy_remote_biased",
        ],
    )

    with pytest.raises(ValueError, match="Conflicting physical config sources"):
        load_config(cfg_path)


def test_saved_canonical_config_keeps_physical_under_scenario_only(tmp_path: Path) -> None:
    cfg_path = tmp_path / "legacy_top_level_physical.yaml"
    out_path = tmp_path / "resolved_config.yaml"
    _write_yaml(
        cfg_path,
        [
            "output_dir: outputs/legacy_top_level_physical",
            "physical:",
            "  enabled: true",
            "reward:",
            "  mode: physical_weighted",
        ],
    )

    with pytest.warns(UserWarning, match="top-level physical has been migrated to scenario.physical"):
        cfg = load_config(cfg_path)
    save_config(cfg, out_path)

    metadata_payload = canonical_train_config_dict(cfg)
    assert "physical" not in metadata_payload
    assert metadata_payload["scenario"]["physical"]["enabled"] is True

    payload = yaml.safe_load(out_path.read_text(encoding="utf-8"))
    assert "physical" not in payload
    assert payload["scenario"]["physical"]["enabled"] is True
