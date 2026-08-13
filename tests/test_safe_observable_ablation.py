from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.plot_safe_ablation_pareto import build_pareto_payload
from scripts.run_safe_ablation_suite import _write_outputs, run_variant
from trisatflow.config import load_config
from trisatflow.envs.obs_builder import build_shared_observation
from trisatflow.envs.obs_schema import IDX_LOCAL_NORMALIZED_COST


REPO_ROOT = Path(__file__).resolve().parents[1]
ABLATION_ROOT = REPO_ROOT / "trisatflow" / "configs" / "ablations"


def test_safe_observable_configs_do_not_expose_cost_prior() -> None:
    for path in sorted(ABLATION_ROOT.glob("safe_*.yaml")):
        cfg = load_config(path)
        assert cfg.observation.mode == "safe_observable"
        assert cfg.observation.include_cost_prior_features is False
        assert cfg.observation.include_oracle_cost is False
        assert cfg.policy_regularization.enabled is False
        assert cfg.reward.use_oracle_cost_components is False


def test_safe_observable_builder_warns_and_drops_cost_prior() -> None:
    row = {
        "local_visible": 1.0,
        "neighbor_visible": 1.0,
        "geo_visible": 1.0,
        "ground_visible": 1.0,
        "local_delay": 0.1,
        "neighbor_delay": 0.2,
        "geo_delay": 0.3,
        "ground_delay": 0.4,
    }
    with pytest.warns(UserWarning, match="safe_observable disallows"):
        batch = build_shared_observation(
            [row],
            node_feature_dim=20,
            access_mode="safe_observable",
            include_cost_prior_features=True,
            include_oracle_cost=True,
        )
    assert float(batch.obs[0, IDX_LOCAL_NORMALIZED_COST].item()) == 0.0


def test_diagnostic_cost_prior_only_is_not_deployable() -> None:
    cfg = load_config(ABLATION_ROOT / "diagnostic_cost_prior_only.yaml")
    payload = json.loads(json.dumps(__import__("yaml").safe_load((ABLATION_ROOT / "diagnostic_cost_prior_only.yaml").read_text(encoding="utf-8"))))

    assert cfg.observation.mode == "cost_prior_ablation"
    assert cfg.observation.include_cost_prior_features is True
    assert cfg.observation.include_oracle_cost is False
    assert payload["ablation"]["main_ablation_deployable"] is False


def test_no_mask_variant_runs_and_summary_has_observation_fields() -> None:
    output_dir = REPO_ROOT / "outputs" / "reviewer_repair" / "safe_ablation" / "test_no_mask"
    row = run_variant("safe_no_mask", episodes=1, steps=2, device="cpu", output_dir=output_dir)
    _write_outputs([row], output_dir)
    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))

    assert row["variant"] == "safe_no_mask"
    assert row["observation_policy"] == "safe_observable"
    assert row["uses_cost_prior"] is False
    assert row["uses_oracle_cost"] is False
    assert row["main_ablation_deployable"] is True
    assert summary["rows"][0]["observation_policy"] == "safe_observable"
    assert summary["rows"][0]["uses_cost_prior"] is False


def test_constrained_ranking_marks_lower_cost_unsafe_variant() -> None:
    payload = build_pareto_payload(
        [
            {
                "variant": "safe_observable_full",
                "normalized_cost": 2.0,
                "deadline_violation_ratio": 0.01,
                "main_ablation_deployable": True,
                "uses_cost_prior": False,
                "uses_oracle_cost": False,
            },
            {
                "variant": "safe_no_mask",
                "normalized_cost": 1.0,
                "deadline_violation_ratio": 0.20,
                "main_ablation_deployable": True,
                "uses_cost_prior": False,
                "uses_oracle_cost": False,
            },
        ]
    )

    assert payload["constrained_ranking"]["cost_at_violation_le_0.01"]["variant"] == "safe_observable_full"
    unsafe = {row["variant"]: row["unsafe_lower_cost"] for row in payload["rows"]}
    assert unsafe["safe_no_mask"] is True
