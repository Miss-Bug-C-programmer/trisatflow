from __future__ import annotations

import pytest
import torch

from trisatflow.config import RewardWeights, ScenarioConfig
from trisatflow.envs import GeoLeoGroundEnv


def _step_env(scenario: ScenarioConfig):
    env = GeoLeoGroundEnv(scenario, RewardWeights(mode="legacy_remote_biased"), "cpu")
    env.reset()
    upper = torch.zeros(scenario.n_leo, dtype=torch.long)
    lower = torch.ones((scenario.n_leo, env.LOWER_ACTION_DIM), dtype=torch.float32)
    return env.step(upper, lower)


def test_formal_legacy_simulator_path_fails_fast() -> None:
    scenario = ScenarioConfig(n_leo=2, episode_len=1, seed=3, formal_claim_required=True)
    scenario.physical.enabled = False

    with pytest.raises(RuntimeError, match="formal/paper-ready experiments require scenario.physical.enabled=true"):
        _step_env(scenario)


def test_debug_legacy_simulator_runs_with_non_formal_metadata() -> None:
    scenario = ScenarioConfig(n_leo=2, episode_len=1, seed=3, formal_claim_required=False)
    scenario.physical.enabled = False

    step = _step_env(scenario)

    assert step.done is True
    assert step.info["simulator_semantics"] == "legacy_normalized_debug"
    assert step.info["formal_claim_allowed"] is False
