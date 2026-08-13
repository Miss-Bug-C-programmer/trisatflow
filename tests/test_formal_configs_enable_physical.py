from __future__ import annotations

from pathlib import Path

import yaml

from trisatflow.config import load_config
from trisatflow.config_validation import is_formal_or_paper_config
from trisatflow.experiment_contracts import assert_paper_safe


REPO_ROOT = Path(__file__).resolve().parents[1]
PAPER_DIR = REPO_ROOT / "trisatflow" / "configs" / "paper"
BASE_DIR = REPO_ROOT / "trisatflow" / "configs" / "base"
STRESS_DIR = REPO_ROOT / "trisatflow" / "configs" / "stress"


def _payload(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def test_paper_configs_explicitly_enable_scenario_physical_model() -> None:
    paper_configs = sorted(PAPER_DIR.glob("*.yaml"))
    assert paper_configs

    for path in paper_configs:
        payload = _payload(path)
        assert payload.get("scenario", {}).get("physical", {}).get("enabled") is True
        cfg = load_config(path)
        assert cfg.scenario.physical.enabled is True
        assert cfg.physical.enabled is True
        assert is_formal_or_paper_config(cfg, path) is True
        assert_paper_safe(cfg)


def test_paper_ready_base_configs_enable_scenario_physical_model() -> None:
    base_configs = sorted(BASE_DIR.glob("*.yaml"))
    assert base_configs

    for path in base_configs:
        cfg = load_config(path)
        if not is_formal_or_paper_config(cfg, path):
            continue
        payload = _payload(path)
        assert payload.get("scenario", {}).get("physical", {}).get("enabled") is True
        assert cfg.scenario.physical.enabled is True
        assert cfg.experiment.paper_ready is True


def test_stress_configs_are_not_marked_formal_or_paper_ready() -> None:
    stress_configs = sorted(STRESS_DIR.glob("*.yaml"))
    assert stress_configs

    for path in stress_configs:
        cfg = load_config(path)
        assert cfg.experiment.paper_ready is False
        assert is_formal_or_paper_config(cfg, path) is False
