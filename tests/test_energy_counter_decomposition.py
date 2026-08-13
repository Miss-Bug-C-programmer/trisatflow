from __future__ import annotations

import torch

from trisatflow.config import RewardWeights, ScenarioConfig
from trisatflow.envs import GeoLeoGroundEnv


def _physical_scenario() -> ScenarioConfig:
    scenario = ScenarioConfig(
        n_leo=2,
        episode_len=1,
        seed=17,
        arrival_rate=0.0,
        burst_prob=0.0,
        max_queue=1.0e12,
        queue_cap_mode="unbounded_eval",
        action_mask_layer_mode="visibility",
        mask_source="measured",
        mask_min_rate=0.0,
        geo_coverage_width_rad=10.0,
        ground_coverage_width_rad=10.0,
    )
    scenario.physical.enabled = True
    scenario.physical.slot_duration_s = 1.0
    scenario.physical.task_size_bits_mean = 2.0e6
    scenario.physical.task_size_bits_std = 0.0
    scenario.physical.cycles_per_bit_mean = 1000.0
    scenario.physical.cycles_per_bit_std = 0.0
    scenario.physical.leo_cpu_hz = 1.0e8
    scenario.physical.geo_cpu_hz = 1.0e8
    scenario.physical.ground_cpu_hz = 1.0e8
    scenario.physical.geo_base_rate_bps = 1.0e9
    scenario.physical.ground_base_rate_bps = 1.0e9
    scenario.physical.queue_cap_cycles = 1.0e12
    return scenario


def _prepared_env(action: int) -> tuple[GeoLeoGroundEnv, torch.Tensor, torch.Tensor]:
    scenario = _physical_scenario()
    env = GeoLeoGroundEnv(scenario, RewardWeights(mode="physical_weighted"), "cpu")
    env.reset()
    task_cycles = torch.full((scenario.n_leo,), 2.0e9)
    env.queue = task_cycles.clone()
    env.leo_queue = env.queue
    env.geo_queue.zero_()
    env.ground_queue.zero_()
    env.ground_station_queue.zero_()
    env.last_task_bits = torch.full((scenario.n_leo,), 2.0e6)
    env.last_cycles_per_bit = torch.full((scenario.n_leo,), 1000.0)
    upper = torch.full((scenario.n_leo,), action, dtype=torch.long)
    lower = torch.full((scenario.n_leo, env.LOWER_ACTION_DIM), 0.05, dtype=torch.float32)
    return env, upper, lower


def test_remote_compute_energy_is_not_deducted_from_source_leo_battery() -> None:
    env, upper, lower = _prepared_env(GeoLeoGroundEnv.ACTION_GEO)
    before_battery = env.energy.clone()

    step = env.step(upper, lower)

    source_energy = step.info["leo_tx_energy_j"] + step.info["leo_local_compute_energy_j"]
    assert torch.all(step.info["geo_compute_energy_j"] > 0.0)
    assert torch.all(step.info["leo_local_compute_energy_j"] == 0.0)
    assert torch.allclose(env.energy, before_battery - source_energy)


def test_energy_counter_decomposition_sums_to_total_system_energy() -> None:
    env, upper, lower = _prepared_env(GeoLeoGroundEnv.ACTION_GEO)

    step = env.step(upper, lower)

    split_total = (
        step.info["leo_tx_energy_j"]
        + step.info["leo_local_compute_energy_j"]
        + step.info["leo_remote_compute_energy_j"]
        + step.info["geo_compute_energy_j"]
        + step.info["ground_compute_energy_j"]
        + step.info["network_energy_j"]
    )
    assert torch.allclose(step.info["total_system_energy_j"], split_total)
    assert torch.allclose(step.info["cumulative_total_system_energy_j"], split_total)


def test_local_compute_energy_stays_in_source_leo_counter() -> None:
    env, upper, lower = _prepared_env(GeoLeoGroundEnv.ACTION_LOCAL)
    before_battery = env.energy.clone()

    step = env.step(upper, lower)

    assert torch.all(step.info["leo_tx_energy_j"] == 0.0)
    assert torch.all(step.info["leo_local_compute_energy_j"] > 0.0)
    assert torch.all(step.info["geo_compute_energy_j"] == 0.0)
    assert torch.all(step.info["ground_compute_energy_j"] == 0.0)
    assert torch.allclose(env.energy, before_battery - step.info["leo_local_compute_energy_j"])
