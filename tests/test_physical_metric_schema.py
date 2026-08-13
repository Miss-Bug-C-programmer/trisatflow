from __future__ import annotations

import torch

from trisatflow.config import PhysicalModelConfig, RewardWeights, ScenarioConfig
from trisatflow.envs.geo_leo_ground_env import GeoLeoGroundEnv
from trisatflow.envs.physical_metrics import build_metric_records, metric_schema_manifest


def _physical_scenario() -> ScenarioConfig:
    return ScenarioConfig(
        n_leo=4,
        episode_len=2,
        seed=5,
        arrival_rate=1.0,
        max_queue=10.0,
        physical=PhysicalModelConfig(
            enabled=True,
            slot_duration_s=0.1,
            task_size_bits_mean=1000.0,
            cycles_per_bit_mean=100.0,
            leo_cpu_hz=1.0e6,
            geo_cpu_hz=2.0e6,
            ground_cpu_hz=3.0e6,
            local_rate_bps=1.0e6,
            isl_base_rate_bps=2.0e5,
            geo_base_rate_bps=1.0e5,
            ground_base_rate_bps=1.5e5,
            max_tx_power_w=2.0,
            kappa=1.0e-27,
            queue_cap_cycles=1.0e7,
            metric_mode="dual",
        ),
    )


def test_physical_env_exports_seconds_joules_and_cycle_queue() -> None:
    env = GeoLeoGroundEnv(_physical_scenario(), RewardWeights(cost_normalization_enabled=True), device="cpu")
    obs, _, _ = env.reset()
    upper = torch.zeros(obs.shape[0], dtype=torch.long)
    lower = torch.ones((obs.shape[0], 3), dtype=torch.float32)
    step = env.step(upper, lower)

    assert "physical_delay_s" in step.info
    assert "physical_energy_j" in step.info
    assert "physical_queue_cycles" in step.info
    assert "compute_energy_j" in step.info
    assert float(step.info["physical_delay_s"].min()) >= 0.0
    assert float(step.info["physical_energy_j"].min()) >= 0.0


def test_summary_metric_records_have_unit_source_and_comparable_scope() -> None:
    values = {
        "mean_delay_s": 0.5,
        "mean_energy_j": 1.25,
        "mean_queue_cycles": 1000.0,
        "normalized_system_cost": 0.7,
    }
    records = build_metric_records(values, normalized_source="affine_normalized_offline_objective")

    assert len(records) == len(values)
    for record in records:
        assert set(["metric", "value", "unit", "source", "normalizer", "comparable_scope"]).issubset(record)
        assert record["unit"]
        assert record["source"]
        assert record["comparable_scope"]

    normalized = next(item for item in records if item["metric"] == "normalized_system_cost")
    assert normalized["unit"] == "dimensionless"
    assert normalized["source"] == "affine_normalized_offline_objective"
    assert normalized["comparable_scope"] == "same_scenario_profile_only"


def test_metric_schema_manifest_declares_physical_and_normalized_semantics() -> None:
    scenario = _physical_scenario()
    manifest = metric_schema_manifest(scenario)

    assert manifest["physical_model_enabled"] is True
    normalized = manifest["metric_descriptors"]["normalized_system_cost"]
    assert normalized["unit"] == "dimensionless"
    assert normalized["comparable_scope"] == "same_scenario_profile_only"
    assert manifest["queue_unit"] == "cycles"
