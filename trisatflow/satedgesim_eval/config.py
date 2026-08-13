from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict

import yaml


@dataclass
class ScenarioConfig:
    """Physical and traffic parameters for the GEO-LEO-Ground toy simulator."""

    n_leo: int = 6
    n_geo: int = 1
    n_ground: int = 1
    episode_len: int = 32
    seed: int = 7

    # Queue and workload parameters. Units are normalized "work units".
    arrival_rate: float = 2.0
    burst_prob: float = 0.08
    burst_multiplier: float = 3.5
    max_queue: float = 80.0
    deadline_threshold: float = 8.0

    # Normalized resource ranges.
    leo_cpu_capacity: float = 5.0
    geo_cpu_capacity: float = 16.0
    ground_cpu_capacity: float = 24.0
    leo_energy_init: float = 100.0
    tx_power_max: float = 1.0
    bandwidth_max: float = 10.0

    # Link model. Larger propagation delays intentionally make GEO useful for
    # stable coordination / tolerant loads rather than low-latency execution.
    isl_base_rate: float = 8.0
    geo_base_rate: float = 12.0
    ground_base_rate: float = 10.0
    local_prop_delay: float = 0.05
    isl_prop_delay: float = 0.6
    geo_prop_delay: float = 4.0
    ground_prop_delay: float = 1.2

    # Dynamic topology knobs.
    orbit_speed: float = 0.31
    visibility_threshold: float = 0.15
    edge_feature_dim: int = 4
    node_feature_dim: int = 12

    # Reviewer-facing ablation switches. They are deliberately part of the
    # scenario rather than hidden script flags so every CSV can be traced back
    # to the exact experimental condition through resolved_config.yaml.
    enable_geo: bool = True
    enable_ground: bool = True
    enable_isl: bool = True
    enable_dynamic_skip_isl: bool = True
    enable_gnn: bool = True
    enable_lyapunov_reward: bool = True
    enable_cross_layer_feedback: bool = True


@dataclass
class RewardWeights:
    delay: float = 1.0
    energy: float = 0.08
    queue: float = 0.04
    violation: float = 1.5
    infeasible: float = 2.0
    load_balance: float = 0.05
    lyapunov_v: float = 0.4


@dataclass
class AlgoConfig:
    """Algorithm configuration for the lightweight algorithm-combination trainer.

    Names follow BenchMARL algorithm families where possible:
    upper discrete: mappo, ippo, iql, vdn, qmix
    lower continuous: maddpg, iddpg, masac, isac
    """

    upper_algo: str = "mappo"
    lower_algo: str = "maddpg"
    encoder_mode: str = "shared_frozen"  # shared_frozen | shared_joint | separate
    encoder_lr: float = 3.0e-4
    joint_encoder_loss_weight: float = 0.5
    gnn_hidden_dim: int = 64
    policy_hidden_dim: int = 128
    ppo_update_mode: str = "standard_ppo"  # standard_ppo | legacy_compact
    ppo_epochs: int = 4
    minibatch_size: int = 256
    gamma: float = 0.96
    gae_lambda: float = 0.90
    clip_param: float = 0.20
    value_clip_param: float = 0.20
    value_loss_coef: float = 0.5
    value_loss_rescale_mode: str = "batch_std"  # none | batch_std | batch_rms
    value_loss_rescale_eps: float = 1.0e-6
    ppo_clip: float = 0.20
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 5.0
    target_kl: float = 0.02
    advantage_normalization: bool = True
    upper_lr: float = 3.0e-4
    lower_lr: float = 3.0e-4
    critic_lr: float = 7.0e-4
    tau: float = 0.02
    exploration_noise: float = 0.12
    lower_batch_size: int = 64
    lower_warmup: int = 64

    # Off-policy discrete-control knobs for IQL/VDN/QMIX-style upper sweeps.
    upper_batch_size: int = 64
    upper_warmup: int = 64
    epsilon_start: float = 0.35
    epsilon_end: float = 0.05
    epsilon_decay_episodes: int = 80

    # SAC-style lower-layer knobs for MASAC/ISAC.
    sac_alpha: float = 0.15
    sac_target_entropy_scale: float = 1.0


@dataclass
class TrainConfig:
    total_episodes: int = 10
    log_interval: int = 1
    device: str = "cpu"
    output_dir: str = "outputs"
    scenario: ScenarioConfig = field(default_factory=ScenarioConfig)
    reward: RewardWeights = field(default_factory=RewardWeights)
    algo: AlgoConfig = field(default_factory=AlgoConfig)


def load_config(path: str | Path | None = None) -> TrainConfig:
    if path is None:
        return TrainConfig()
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return _from_dict(TrainConfig, data)


def save_config(config: TrainConfig, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(asdict(config), f, sort_keys=False, allow_unicode=True)


def _from_dict(cls, data: Dict[str, Any]):
    fields = getattr(cls, "__dataclass_fields__", {})
    kwargs = {}
    for name, field_def in fields.items():
        if name not in data:
            continue
        value = data[name]
        field_type = field_def.type
        # Dataclass forward annotations are strings under from __future__.
        if name == "scenario" and isinstance(value, dict):
            value = _from_dict(ScenarioConfig, value)
        elif name == "reward" and isinstance(value, dict):
            value = _from_dict(RewardWeights, value)
        elif name == "algo" and isinstance(value, dict):
            value = _from_dict(AlgoConfig, value)
        kwargs[name] = value
    return cls(**kwargs)
