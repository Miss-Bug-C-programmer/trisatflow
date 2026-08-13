from __future__ import annotations

from typing import Any, Dict


def cadence_report(
    *,
    upper_update_count: int,
    lower_update_count: int,
    env_steps_since_upper_update: int,
    env_steps_since_lower_update: int,
    replay_buffer_size: int,
    rollout_buffer_size: int,
    upper_update_every: int,
    lower_update_every: int,
    lower_updates_per_upper_update: int,
) -> Dict[str, Any]:
    lag = max(0, int(env_steps_since_lower_update) - int(env_steps_since_upper_update))
    non_stationarity = bool(
        int(lower_updates_per_upper_update) > 1
        or int(upper_update_every) != int(lower_update_every)
        or lag > max(1, int(lower_update_every))
    )
    return {
        "upper_update_count": int(upper_update_count),
        "lower_update_count": int(lower_update_count),
        "env_steps_since_upper_update": int(env_steps_since_upper_update),
        "env_steps_since_lower_update": int(env_steps_since_lower_update),
        "replay_buffer_size": int(replay_buffer_size),
        "rollout_buffer_size": int(rollout_buffer_size),
        "upper_update_every": int(upper_update_every),
        "lower_update_every": int(lower_update_every),
        "lower_updates_per_upper_update": int(lower_updates_per_upper_update),
        "off_policy_lag_estimate": int(lag),
        "non_stationarity_warning": non_stationarity,
    }
