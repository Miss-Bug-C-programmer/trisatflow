from __future__ import annotations

import torch

from trisatflow.config import ScenarioConfig
from trisatflow.envs import GeoLeoGroundEnv
from trisatflow.metrics.lyapunov_diagnostics import compute_lyapunov_diagnostics


def _tiny_queue_env(queue_cap_mode: str) -> GeoLeoGroundEnv:
    return GeoLeoGroundEnv(
        ScenarioConfig(
            n_leo=4,
            episode_len=2,
            seed=13,
            arrival_rate=8.0,
            burst_prob=0.0,
            max_queue=1.0,
            leo_cpu_capacity=0.01,
            queue_cap_mode=queue_cap_mode,
        )
    )


def test_finite_buffer_queue_does_not_exceed_cap() -> None:
    env = _tiny_queue_env("finite_buffer")
    env.reset()
    upper = torch.zeros(env.n_agents, dtype=torch.long)
    lower = torch.zeros(env.n_agents, env.LOWER_ACTION_DIM)
    step = env.step(upper, lower, minimal_info=True)

    assert float(step.info["queue"].max().item()) <= env.cfg.max_queue
    assert step.info["queue_cap_mode_is_unbounded_eval"].max().item() == 0.0


def test_unbounded_eval_queue_can_exceed_cap() -> None:
    env = _tiny_queue_env("unbounded_eval")
    env.reset()
    env.queue = torch.full((env.n_agents,), 2.5)
    upper = torch.zeros(env.n_agents, dtype=torch.long)
    lower = torch.zeros(env.n_agents, env.LOWER_ACTION_DIM)
    step = env.step(upper, lower, minimal_info=True)

    assert float(step.info["queue"].max().item()) > env.cfg.max_queue
    assert step.info["queue_cap_mode_is_unbounded_eval"].min().item() == 1.0


def test_positive_drift_ratio_and_overflow_diagnostics() -> None:
    diagnostics = compute_lyapunov_diagnostics(
        [
            {
                "queue": [0.5, 2.5, 4.0],
                "virtual_delay_queue": [0.0, 1.0],
                "lyapunov_drift": [1.0, -2.0, 3.0],
                "system_cost": [0.2, 0.4, 0.6],
            }
        ],
        queue_cap_mode="finite_buffer",
        queue_cap=2.0,
    )

    assert diagnostics["positive_drift_ratio"] == 2.0 / 3.0
    assert diagnostics["finite_buffer_overflow_count"] == 2
    assert diagnostics["queue_stability_claim_allowed"] is False
    assert diagnostics["lyapunov_semantics"] == "reward_shaping_no_stability_theorem"
