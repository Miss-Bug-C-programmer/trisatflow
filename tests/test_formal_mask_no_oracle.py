from __future__ import annotations

import pytest
import torch

from trisatflow.config import load_config
from trisatflow.envs import GeoLeoGroundEnv


def _write_config(path, *, diagnostic_oracle_allowed: bool) -> None:
    path.write_text(
        "\n".join(
            [
                "total_episodes: 1",
                "output_dir: outputs/paper/oracle_mask_test",
                "experiment:",
                "  paper_ready: true",
                f"  diagnostic_oracle_allowed: {str(diagnostic_oracle_allowed).lower()}",
                "scenario:",
                "  n_leo: 2",
                "  episode_len: 1",
                "  mask_source: oracle_trace",
                "  action_mask_layer_mode: full",
                "  physical:",
                "    enabled: true",
                "reward:",
                "  mode: physical_weighted",
            ]
        ),
        encoding="utf-8",
    )


def test_formal_config_rejects_oracle_trace_mask_by_default(tmp_path) -> None:
    cfg_path = tmp_path / "paper_oracle.yaml"
    _write_config(cfg_path, diagnostic_oracle_allowed=False)

    with pytest.raises(ValueError, match="formal/paper-ready config cannot use scenario.mask_source='oracle_trace'"):
        load_config(cfg_path)


def test_diagnostic_oracle_mode_runs_but_metadata_is_non_formal(tmp_path) -> None:
    cfg_path = tmp_path / "paper_oracle_diagnostic.yaml"
    _write_config(cfg_path, diagnostic_oracle_allowed=True)
    cfg = load_config(cfg_path)
    env = GeoLeoGroundEnv(cfg.scenario, cfg.reward)
    env.reset()

    upper = torch.zeros(cfg.scenario.n_leo, dtype=torch.long)
    lower = torch.ones((cfg.scenario.n_leo, env.LOWER_ACTION_DIM), dtype=torch.float32)
    step = env.step(upper, lower)

    assert step.info["mask_source"] == "oracle_trace"
    assert step.info["formal_claim_allowed"] is False
    assert step.info["diagnostic_oracle_allowed"] is True
    assert float(step.info["uses_oracle_trace_mask"].max().item()) == 1.0
