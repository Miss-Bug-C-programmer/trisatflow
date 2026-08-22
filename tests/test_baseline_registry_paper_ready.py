from __future__ import annotations

import importlib.util
from pathlib import Path
import random

from trisatflow.baselines.registry import (
    baseline_metadata_json,
    baseline_metadata_registry,
    build_baseline_policy,
    paper_ready_baseline_names,
)
from trisatflow.baselines.hmadrl_baseline import HMADRLMaddqnDdpgBaseline


def _load_run_experiment_matrix_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "run_experiment_matrix.py"
    spec = importlib.util.spec_from_file_location("run_experiment_matrix", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _dummy_candidate_info():
    return {
        0: {"estimated_delay": 0.12, "estimated_queue": 2.0, "estimated_cost": 1.1, "estimated_energy_j": 0.6, "mobility_risk": 0.2},
        1: {"estimated_delay": 0.08, "estimated_queue": 3.0, "estimated_cost": 1.0, "estimated_energy_j": 0.8, "mobility_risk": 0.4},
        2: {"estimated_delay": 0.20, "estimated_queue": 1.0, "estimated_cost": 1.3, "estimated_energy_j": 0.5, "mobility_risk": 0.1},
        3: {"estimated_delay": 0.15, "estimated_queue": 1.5, "estimated_cost": 1.2, "estimated_energy_j": 0.4, "mobility_risk": 0.3},
    }


def test_baseline_metadata_schema_complete():
    required_keys = {
        "baseline_name",
        "name",
        "type",
        "implemented",
        "uses_oracle",
        "uses_privileged_info",
        "trainable",
        "requires_checkpoint",
        "checkpoint_loaded",
        "paper_ready",
        "is_placeholder",
        "allows_formal_eval",
        "fallback_policy",
        "update_implemented",
    }
    for name, meta in baseline_metadata_json().items():
        assert required_keys <= set(meta.keys())
        assert meta["name"] == name


def test_placeholder_not_paper_ready_and_filtered_by_default_matrix():
    metadata = baseline_metadata_registry()
    assert metadata["hmadrl_maddqn_ddpg"].type == "placeholder"
    assert metadata["hmadrl_maddqn_ddpg"].paper_ready is False
    assert HMADRLMaddqnDdpgBaseline.paper_ready is False
    assert HMADRLMaddqnDdpgBaseline.placeholder is True

    mod = _load_run_experiment_matrix_module()
    resolved = mod._resolve_matrix_baselines(
        ["local_only", "hmadrl_maddqn_ddpg", "weight_greedy"],
        allow_placeholder_baselines=False,
        allow_non_paper_ready_baselines=False,
    )
    assert resolved["selected"] == ["local_only"]
    assert resolved["blocked_placeholders"] == ["hmadrl_maddqn_ddpg"]
    assert resolved["skipped_non_paper_ready"] == ["weight_greedy"]


def test_paper_ready_set_contains_required_formal_baselines():
    names = set(paper_ready_baseline_names())
    expected = {
        "local_only",
        "neighbor_only",
        "geo_only",
        "ground_only",
        "random_visible",
        "min_delay_greedy",
        "min_energy_greedy",
        "queue_aware_greedy",
        "mobility_risk_greedy",
        "lyapunov_dpp_greedy",
    }
    assert expected.issubset(names)


def test_new_greedy_baselines_return_decision_info():
    baseline_names = [
        "local_only",
        "neighbor_only",
        "geo_only",
        "ground_only",
        "random_visible",
        "min_delay_greedy",
        "min_energy_greedy",
        "queue_aware_greedy",
        "mobility_risk_greedy",
        "lyapunov_dpp_greedy",
    ]
    mask = [1, 1, 1, 1]
    state = {}
    candidate_info = _dummy_candidate_info()
    rng = random.Random(13)

    for name in baseline_names:
        policy = build_baseline_policy(name)
        out = policy.select_action(obs=None, state=state, mask=mask, candidate_info=candidate_info, rng=rng)
        assert "decision_info" in out
        decision_info = out["decision_info"]
        assert decision_info["baseline_name"] in {name, "cost_greedy", "weight_greedy"}
        assert isinstance(out["upper_action"], int)
        assert 0 <= out["upper_action"] <= 3
