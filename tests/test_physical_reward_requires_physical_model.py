from __future__ import annotations

from pathlib import Path

import torch
import pytest

from trisatflow.config import load_config
from trisatflow.config_validation import is_formal_or_paper_config
from trisatflow.envs import GeoLeoGroundEnv


def test_physical_weighted_reward_requires_scenario_physical_enabled(tmp_path: Path) -> None:
    cfg_path = tmp_path / "bad_physical_reward.yaml"
    cfg_path.write_text(
        "\n".join(
            [
                "output_dir: outputs/debug_bad_physical_reward",
                "scenario:",
                "  physical:",
                "    enabled: false",
                "reward:",
                "  mode: physical_weighted",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="physical_weighted reward requires scenario.physical.enabled=true"):
        load_config(cfg_path)


def test_paper_ready_flag_requires_scenario_physical_enabled(tmp_path: Path) -> None:
    cfg_path = tmp_path / "bad_paper_ready.yaml"
    cfg_path.write_text(
        "\n".join(
            [
                "output_dir: outputs/paper_ready_bad",
                "reward:",
                "  mode: legacy_remote_biased",
                "experiment:",
                "  paper_ready: true",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="paper-ready/formal config requires scenario.physical.enabled=true"):
        load_config(cfg_path)


def test_legacy_smoke_config_remains_non_formal_and_runs_one_step(tmp_path: Path) -> None:
    cfg_path = tmp_path / "legacy_smoke.yaml"
    cfg_path.write_text(
        "\n".join(
            [
                "total_episodes: 1",
                "output_dir: outputs/smoke_legacy",
                "scenario:",
                "  n_leo: 2",
                "  episode_len: 1",
                "  seed: 3",
                "reward:",
                "  mode: legacy_remote_biased",
                "experiment:",
                "  paper_ready: false",
            ]
        ),
        encoding="utf-8",
    )

    cfg = load_config(cfg_path)

    assert cfg.reward.mode == "legacy_remote_biased"
    assert cfg.scenario.physical.enabled is False
    assert cfg.experiment.paper_ready is False
    assert is_formal_or_paper_config(cfg, cfg_path) is False

    env = GeoLeoGroundEnv(cfg.scenario, cfg.reward, "cpu")
    env.reset()
    step = env.step(
        torch.zeros(cfg.scenario.n_leo, dtype=torch.long),
        torch.ones((cfg.scenario.n_leo, env.LOWER_ACTION_DIM), dtype=torch.float32),
    )
    assert step.done is True

