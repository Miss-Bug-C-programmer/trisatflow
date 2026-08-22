from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict

import yaml
from trisatflow.encoder_modes import canonicalize_encoder_mode
import warnings

from trisatflow.config_validation import (
    canonicalize_train_config_path,
    validate_train_config,
    validate_wrapper_payload,
)


@dataclass
class PhysicalModelConfig:
    enabled: bool = False
    slot_duration_s: float = 1.0
    task_size_bits_mean: float = 1.0e6
    task_size_bits_std: float = 0.0
    cycles_per_bit_mean: float = 1000.0
    cycles_per_bit_std: float = 0.0
    leo_cpu_hz: float = 5.0e9
    geo_cpu_hz: float = 16.0e9
    ground_cpu_hz: float = 24.0e9
    local_rate_bps: float = 1.0e9
    isl_base_rate_bps: float = 8.0e6
    geo_base_rate_bps: float = 12.0e6
    ground_base_rate_bps: float = 10.0e6
    max_tx_power_w: float = 1.0
    kappa: float = 1.0e-28
    compute_energy_model: str = "kappa_cycles_f2"  # kappa_cycles_f2 | kappa_f3_time
    queue_unit: str = "cycles"
    queue_cap_cycles: float = 8.0e10
    unbounded_queue_eval: bool = False
    metric_mode: str = "dual"  # normalized | physical | dual


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
    queue_mode: str = "single_queue"  # single_queue | multi_queue
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
    observation_mode: str = "shared_tier_summary"
    include_cost_features_in_obs: bool = False
    observation_access_mode: str = "safe_observable"
    observation_include_oracle_cost: bool = False
    observation_include_cost_prior_features: bool = False
    obs_normalization_mode: str = "legacy"  # legacy | trace_p95 | trace_log_quantile
    obs_normalization_path: str = ""
    action_mask_mode: str = "visible_only"  # legacy alias: visible_only | mobility_safe | completion_safe
    action_mask_layer_mode: str = "legacy"  # legacy | none | visibility | completion_safe | mobility_risk | full
    enable_visibility_mask: bool = True
    enable_completion_safe_mask: bool = True
    enable_mobility_risk_mask: bool = True
    success_profile: str = "default"  # default | paper_strict | preflight_lenient
    profile_name: str = ""
    action_space_architecture: str = "full"  # only_leo | leo_geo | leo_ground | full
    min_link_survival_margin_sec: float = 0.0

    # Topology realism knobs. The default analytic model is still lightweight,
    # but it now mimics SatEdgeSim-style dynamic candidate availability instead
    # of treating GEO/ground as static remote buckets. A JSONL trace exported
    # from SatEdgeSim can override the analytic visibility/rate process.
    topology_mode: str = "analytic"  # analytic | satedgesim_trace
    topology_trace_path: str = ""
    topology_trace_repeat: bool = True
    topology_trace_strict: bool = False
    action_mask_enabled: bool = True
    mask_source: str = "predicted"  # measured | predicted | oracle_trace
    mask_prediction_horizon_s: float = 8.0
    link_lifetime_noise_std_s: float = 0.0
    completion_time_noise_std_s: float = 0.0
    mask_false_positive_rate: float = 0.0
    mask_false_negative_rate: float = 0.0
    mask_staleness_slots: int = 0
    mask_min_rate: float = 1.0e-4
    geo_coverage_width_rad: float = 2.45
    ground_coverage_width_rad: float = 1.75
    gateway_drift_rate: float = 0.17
    ground_backhaul_congestion: float = 0.20
    geo_backhaul_congestion: float = 0.10

    # Reviewer-facing ablation switches. They are deliberately part of the
    # scenario rather than hidden script flags so every CSV can be traced back
    # to the exact experimental condition through resolved_config.yaml.
    enable_geo: bool = True
    enable_ground: bool = True
    enable_isl: bool = True
    enable_dynamic_skip_isl: bool = True
    enable_gnn: bool = True
    enable_lyapunov_reward: bool = True
    lyapunov_claim_mode: str = "inspired_reward"  # inspired_reward | theoretical_dpp
    queue_cap_mode: str = "finite_buffer"  # finite_buffer | unbounded_eval
    enable_cross_layer_feedback: bool = True

    # Metric/units export controls. Defaults preserve prior numerical behavior.
    export_physical_metrics: bool = True
    delay_s_per_unit: float = 1.0
    energy_j_per_unit: float = 1.0
    queue_cycles_per_unit: float = 1.0
    cpu_ghz_per_unit: float = 1.0
    rate_mbps_per_unit: float = 1.0
    bandwidth_mbps_per_unit: float = 1.0
    power_w_per_unit: float = 1.0
    task_size_bits_per_unit: float = 1.0
    workload_cycles_per_unit: float = 1.0
    trace_delay_anomaly_threshold_s: float = 1.0e3
    trace_treat_large_delay_as_legacy_score: bool = True
    physical: PhysicalModelConfig = field(default_factory=PhysicalModelConfig)
    paper_ready: bool = False
    diagnostic_oracle_allowed: bool = False
    # Explicit execution-semantic guard used by formal callers.  Legacy
    # normalized simulation remains available for debug/backward-compatible
    # runs, but cannot be presented as a formal physical result.
    formal_claim_required: bool = False


@dataclass
class RewardWeights:
    mode: str = "physical_weighted"  # physical_weighted | legacy_remote_biased | oracle_aligned_cost
    delay: float = 1.0
    energy: float = 0.08
    queue: float = 0.04
    violation: float = 1.5
    infeasible: float = 2.0
    load_balance: float = 0.05
    lyapunov_v: float = 0.4

    # Optional anti-local-collapse shaping used by the offload_pressure profile.
    # These terms do not force remote execution. They only change the reward when
    # remote execution is feasible and reduces estimated delay, or when local
    # execution is selected under severe local queue pressure. Defaults are zero
    # to keep existing experiment configs backward compatible.
    offload_gain: float = 0.0
    local_queue_pressure: float = 0.0
    remote_feasible_bonus: float = 0.0
    action_balance_bonus: float = 0.0
    selected_when_visible_bonus: float = 0.0
    cost_normalization_enabled: bool = False
    per_tier_cost_normalization: bool = False
    ground_congestion_penalty: float = 0.0
    geo_delay_penalty: float = 0.0
    local_queue_penalty: float = 0.0
    neighbor_link_penalty: float = 0.0
    remote_bonus: float = 0.0
    local_penalty: float = 0.0
    neighbor_penalty: float = 0.0
    geo_penalty: float = 0.0
    ground_penalty: float = 0.0

    # Oracle-aligned reward knobs (kept separate from legacy weights so older
    # experiments remain reproducible as-is).
    use_oracle_cost_components: bool = False
    use_lower_effect_in_upper_reward: bool = True
    include_energy: bool = False
    delay_weight: float = 1.0
    queue_weight: float = 0.5
    transmission_weight: float = 0.2
    compute_weight: float = 0.2
    feasibility_weight: float = 10.0
    include_failure_risk: bool = False
    failure_penalty_weight: float = 0.0


@dataclass
class PolicyRegularizationConfig:
    enabled: bool = False
    mode: str = "none"  # none | cost_rank_kl | cost_prior_ce
    weight: float = 0.0
    temperature: float = 0.5


@dataclass
class ObservationConfig:
    mode: str = "safe_observable"  # safe_observable | cost_prior_ablation | oracle_debug
    include_oracle_cost: bool = False
    include_cost_prior_features: bool = False
    # Set by config alias normalization when loading old configs.
    legacy_auto_enabled: bool = False


@dataclass
class AlgoConfig:
    """Algorithm configuration for the lightweight algorithm-combination trainer.

    Names follow BenchMARL algorithm families where possible:
    upper discrete: mappo, ippo, iql, vdn, qmix
    lower continuous: maddpg, iddpg, masac, isac
    """

    upper_algo: str = "mappo"
    lower_algo: str = "maddpg"
    # Lower-layer encoder coupling:
    # - shared_frozen: lower reads shared upper encoder features without updating encoder.
    # - shared_joint: lower and upper jointly update shared encoder.
    # - separate: lower uses an independent encoder (+ target encoder for off-policy targets).
    #   Recommended for paper-grade experiments to reduce gradient interference.
    encoder_mode: str = "shared_upper_detached_lower"  # shared_upper_detached_lower | shared_joint | separate_lower_encoder | shared_frozen
    lower_observation_mode: str = "shared_embedding"  # shared_embedding | raw_obs_reencode
    stop_gradient_to_encoder_from_lower: bool = True
    detach_embedding_during_action_collection: bool = True
    upper_update_every: int = 1
    lower_update_every: int = 1
    lower_updates_per_upper_update: int = 1
    freeze_upper_during_lower_update: bool = True
    log_gradient_diagnostics: bool = False
    gradient_diagnostics_interval: int = 1
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
    entropy_coef_schedule: str = ""
    eval_interval: int = 1
    credit_assignment: str = "global_team"  # global_team | per_agent
    policy_head: str = "gnn_only"  # gnn_only | hybrid_gnn_cost
    logit_centering: bool = False
    action_bias_regularization: float = 0.0


@dataclass
class EvaluationConfig:
    deterministic_eval_mode: str = "raw_argmax"
    tie_break_eps: float = 0.05
    stochastic_seed: int = 13
    report_raw_argmax_always: bool = True


@dataclass
class TemporalModelConfig:
    enabled: bool = False
    type: str = "gnn_gru"  # gnn_gru
    history_len: int = 4
    hidden_dim: int = 128


@dataclass
class ModelConfig:
    topology_encoder: str = "static_gnn"  # no_gnn | static_gnn | temporal_gnn
    temporal: TemporalModelConfig = field(default_factory=TemporalModelConfig)


@dataclass
class ExperimentSplitConfig:
    train_seeds: list[int] = field(default_factory=list)
    val_seeds: list[int] = field(default_factory=list)
    test_seeds: list[int] = field(default_factory=list)
    allow_debug_seed_overlap: bool = False


@dataclass
class ExperimentConfig:
    paper_ready: bool = False
    diagnostic_oracle_allowed: bool = False
    split: ExperimentSplitConfig = field(default_factory=ExperimentSplitConfig)


@dataclass
class TrainConfig:
    config_source_chain: list[str] = field(default_factory=list)
    total_episodes: int = 10
    steps_per_episode: int | None = None
    upper_pretrain_episodes: int = 0
    joint_train_episodes: int = 0
    lower_training_enabled: bool = True
    lower_action_mode: str = "learned"  # learned | neutral_allocator
    log_interval: int = 1
    device: str = "cpu"
    requested_device: str = "cpu"
    actual_device: str = "cpu"
    device_fallback_reason: str = ""
    output_dir: str = "outputs"
    observation: ObservationConfig = field(default_factory=ObservationConfig)
    scenario: ScenarioConfig = field(default_factory=ScenarioConfig)
    physical: PhysicalModelConfig = field(default_factory=PhysicalModelConfig)
    reward: RewardWeights = field(default_factory=RewardWeights)
    policy_regularization: PolicyRegularizationConfig = field(default_factory=PolicyRegularizationConfig)
    algo: AlgoConfig = field(default_factory=AlgoConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)


def _deep_merge_dict(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge a YAML override into a base mapping.

    Nested mappings are merged rather than replaced so compact ablation files can
    override a single controlled variable without silently discarding the canonical
    SatEdgeSim scenario.
    """

    merged: Dict[str, Any] = dict(base)
    for key, value in override.items():
        if key == "extends":
            continue
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge_dict(merged[key], value)
        else:
            merged[key] = value
    return merged


def _load_yaml_with_extends(path: str | Path, *, stack: tuple[Path, ...] = ()) -> tuple[Dict[str, Any], list[str]]:
    """Load a YAML config and resolve an optional relative ``extends`` chain."""

    resolved = Path(path).expanduser().resolve()
    if resolved in stack:
        chain = " -> ".join(p.as_posix() for p in (*stack, resolved))
        raise ValueError(f"Cyclic config extends chain detected: {chain}")
    with open(resolved, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config YAML must contain a mapping: {resolved.as_posix()}")
    parent = data.get("extends")
    if not parent:
        return dict(data), [resolved.as_posix()]
    parent_path = Path(str(parent))
    if not parent_path.is_absolute():
        parent_path = resolved.parent / parent_path
    base, chain = _load_yaml_with_extends(parent_path, stack=(*stack, resolved))
    return _deep_merge_dict(base, data), [*chain, resolved.as_posix()]


def load_config(path: str | Path | None = None) -> TrainConfig:
    if path is None:
        cfg = TrainConfig()
        cfg.config_source_chain = ["<default>"]
        cfg.physical = cfg.scenario.physical
        cfg.scenario.paper_ready = bool(getattr(cfg.experiment, "paper_ready", False))
        cfg.scenario.diagnostic_oracle_allowed = bool(getattr(cfg.experiment, "diagnostic_oracle_allowed", False))
        validate_train_config(cfg, source="<default>")
        return cfg
    resolved_path, deprecation_msg = canonicalize_train_config_path(path)
    if deprecation_msg:
        warnings.warn(f"[DEPRECATED] {deprecation_msg}", UserWarning, stacklevel=2)
    data, source_chain = _load_yaml_with_extends(resolved_path)
    wrapped, canonical = validate_wrapper_payload(data)
    if wrapped:
        canonical_path = Path(canonical)
        warnings.warn(
            f"[DEPRECATED] deprecated config wrapper '{Path(path)}' -> '{canonical_path.as_posix()}'",
            UserWarning,
            stacklevel=2,
        )
        return load_config(canonical_path)
    data = _normalize_config_aliases(data)
    reward_mode_explicit = isinstance(data.get("reward"), dict) and "mode" in data.get("reward", {})
    data = _canonicalize_physical_config_sources(data)
    physical_config_explicit = isinstance(data.get("scenario"), dict) and "physical" in data.get("scenario", {})
    data["config_source_chain"] = source_chain
    cfg = _from_dict(TrainConfig, data)
    setattr(cfg, "_reward_mode_explicit", bool(reward_mode_explicit))
    setattr(cfg, "_physical_config_explicit", bool(physical_config_explicit))
    cfg.physical = cfg.scenario.physical
    cfg.scenario.paper_ready = bool(getattr(cfg.experiment, "paper_ready", False))
    cfg.scenario.diagnostic_oracle_allowed = bool(getattr(cfg.experiment, "diagnostic_oracle_allowed", False))
    validate_train_config(cfg, source=str(resolved_path))
    return cfg


def save_config(config: TrainConfig, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(canonical_train_config_dict(config), f, sort_keys=False, allow_unicode=True)


def canonical_train_config_dict(config: TrainConfig) -> Dict[str, Any]:
    """Return a canonical serializable config payload.

    The only valid physical-model source in persisted metadata is
    scenario.physical.  The top-level TrainConfig.physical attribute is kept as
    a runtime compatibility alias and is intentionally omitted here.
    """

    payload = asdict(config)
    payload.pop("physical", None)
    scenario = payload.setdefault("scenario", {})
    scenario.setdefault("physical", asdict(config.scenario.physical))
    return payload


def _canonicalize_physical_config_sources(data: Dict[str, Any]) -> Dict[str, Any]:
    normalized = dict(data)
    top_physical = normalized.pop("physical", None)
    scenario = dict(normalized.get("scenario") or {})
    scenario_physical = scenario.get("physical")
    if top_physical is None:
        normalized["scenario"] = scenario
        return normalized
    if not isinstance(top_physical, dict):
        raise ValueError("top-level physical must be a mapping when provided")
    if scenario_physical is None:
        scenario["physical"] = dict(top_physical)
        normalized["scenario"] = scenario
        warnings.warn(
            "top-level physical has been migrated to scenario.physical; please update the YAML",
            UserWarning,
            stacklevel=3,
        )
        return normalized
    if not isinstance(scenario_physical, dict):
        raise ValueError("scenario.physical must be a mapping when provided")
    top_clean = dict(top_physical)
    scenario_clean = dict(scenario_physical)
    if top_clean != scenario_clean:
        raise ValueError("Conflicting physical config sources: use scenario.physical as the canonical source")
    scenario["physical"] = scenario_clean
    normalized["scenario"] = scenario
    warnings.warn(
        "top-level physical duplicates scenario.physical and has been removed from canonical config",
        UserWarning,
        stacklevel=3,
    )
    return normalized


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
        elif name == "physical" and isinstance(value, dict):
            value = _from_dict(PhysicalModelConfig, value)
        elif name == "observation" and isinstance(value, dict):
            value = _from_dict(ObservationConfig, value)
        elif name == "reward" and isinstance(value, dict):
            value = _from_dict(RewardWeights, value)
        elif name == "policy_regularization" and isinstance(value, dict):
            value = _from_dict(PolicyRegularizationConfig, value)
        elif name == "algo" and isinstance(value, dict):
            value = _from_dict(AlgoConfig, value)
        elif name == "model" and isinstance(value, dict):
            value = _from_dict(ModelConfig, value)
        elif name == "temporal" and isinstance(value, dict):
            value = _from_dict(TemporalModelConfig, value)
        elif name == "evaluation" and isinstance(value, dict):
            value = _from_dict(EvaluationConfig, value)
        elif name == "experiment" and isinstance(value, dict):
            value = _from_dict(ExperimentConfig, value)
        elif name == "split" and isinstance(value, dict):
            value = _from_dict(ExperimentSplitConfig, value)
        kwargs[name] = value
    return cls(**kwargs)


def _normalize_config_aliases(data: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(data, dict):
        return {}
    normalized = dict(data)

    training = normalized.pop("training", None)
    if isinstance(training, dict):
        if "episodes" in training and "total_episodes" not in normalized:
            normalized["total_episodes"] = training.get("episodes")
        if "steps_per_episode" in training and "steps_per_episode" not in normalized:
            normalized["steps_per_episode"] = training.get("steps_per_episode")
        if "steps" in training and "steps_per_episode" not in normalized:
            normalized["steps_per_episode"] = training.get("steps")
        for key in ("upper_pretrain_episodes", "joint_train_episodes", "lower_training_enabled", "lower_action_mode"):
            if key in training and key not in normalized:
                normalized[key] = training.get(key)

    algorithm = normalized.pop("algorithm", None)
    if isinstance(algorithm, dict):
        algo_cfg = dict(normalized.get("algo") or {})
        upper = algorithm.get("upper")
        if isinstance(upper, dict):
            for key, value in upper.items():
                algo_cfg[key] = value
        lower = algorithm.get("lower")
        if isinstance(lower, dict):
            for key, value in lower.items():
                algo_cfg[key] = value
        for key, value in algorithm.items():
            if key not in {"upper", "lower"}:
                algo_cfg[key] = value
        normalized["algo"] = algo_cfg
    # Allow nested algo.policy_regularization alias.
    algo_cfg_alias = normalized.get("algo")
    if isinstance(algo_cfg_alias, dict):
        nested_reg = algo_cfg_alias.pop("policy_regularization", None)
        if isinstance(nested_reg, dict) and "policy_regularization" not in normalized:
            normalized["policy_regularization"] = nested_reg
        if "encoder_mode" in algo_cfg_alias:
            raw_mode = str(algo_cfg_alias.get("encoder_mode") or "").strip().lower()
            canonical_mode = canonicalize_encoder_mode(raw_mode, warn=raw_mode not in {"shared_upper_detached_lower", "shared_joint", "separate_lower_encoder", "shared_frozen"})
            algo_cfg_alias["encoder_mode"] = canonical_mode
            if canonical_mode == "shared_upper_detached_lower" and "stop_gradient_to_encoder_from_lower" not in algo_cfg_alias:
                algo_cfg_alias["stop_gradient_to_encoder_from_lower"] = True
            elif canonical_mode in {"shared_joint", "separate_lower_encoder"} and "stop_gradient_to_encoder_from_lower" not in algo_cfg_alias:
                algo_cfg_alias["stop_gradient_to_encoder_from_lower"] = False
            elif canonical_mode == "shared_frozen" and "stop_gradient_to_encoder_from_lower" not in algo_cfg_alias:
                algo_cfg_alias["stop_gradient_to_encoder_from_lower"] = True

    observation = normalized.get("observation")
    if isinstance(observation, dict):
        observation = dict(observation)
    else:
        observation = {}
    if "observation_policy" in normalized and "mode" not in observation:
        observation["mode"] = normalized.pop("observation_policy")
    if "include_cost_prior" in normalized and "include_cost_prior_features" not in observation:
        observation["include_cost_prior_features"] = bool(normalized.pop("include_cost_prior"))
    if "include_diagnostic_features" in normalized and "include_cost_prior_features" not in observation:
        observation["include_cost_prior_features"] = bool(normalized.pop("include_diagnostic_features"))

    scenario = normalized.get("scenario")
    if isinstance(scenario, dict):
        scenario = dict(scenario)
        if "steps_per_episode" in normalized and "episode_len" not in scenario:
            scenario["episode_len"] = normalized["steps_per_episode"]
        if bool(scenario.get("include_cost_features_in_obs", False)) and "node_feature_dim" not in scenario:
            scenario["node_feature_dim"] = 20
        if "mode" not in observation and scenario.get("observation_access_mode"):
            observation["mode"] = scenario.get("observation_access_mode")
        if "include_cost_prior_features" not in observation and "include_cost_features_in_obs" in scenario:
            observation["include_cost_prior_features"] = bool(scenario.get("include_cost_features_in_obs", False))
        if "include_oracle_cost" not in observation and "observation_include_oracle_cost" in scenario:
            observation["include_oracle_cost"] = bool(scenario.get("observation_include_oracle_cost", False))
        normalized["scenario"] = scenario

    environment = normalized.get("environment")
    if isinstance(environment, dict):
        environment = dict(environment)
        action_mask_cfg = environment.get("action_mask")
        sc = dict(normalized.get("scenario") or {})
        if isinstance(action_mask_cfg, dict):
            mode = action_mask_cfg.get("mode")
            if mode is not None:
                sc["action_mask_layer_mode"] = mode
            if "enable_visibility_mask" in action_mask_cfg:
                sc["enable_visibility_mask"] = bool(action_mask_cfg.get("enable_visibility_mask"))
            if "enable_completion_safe_mask" in action_mask_cfg:
                sc["enable_completion_safe_mask"] = bool(action_mask_cfg.get("enable_completion_safe_mask"))
            if "enable_mobility_risk_mask" in action_mask_cfg:
                sc["enable_mobility_risk_mask"] = bool(action_mask_cfg.get("enable_mobility_risk_mask"))
            if "enabled" in action_mask_cfg:
                sc["action_mask_enabled"] = bool(action_mask_cfg.get("enabled"))
        if sc:
            normalized["scenario"] = sc

    reward = normalized.get("reward")
    if isinstance(reward, dict):
        reward = dict(reward)
        if "remote_bonus" not in reward and "remote_feasible_bonus" in reward:
            reward["remote_bonus"] = reward["remote_feasible_bonus"]
        if "delay_weight" not in reward and "delay" in reward:
            reward["delay_weight"] = reward["delay"]
        if "queue_weight" not in reward and "queue" in reward:
            reward["queue_weight"] = reward["queue"]
        if "feasibility_weight" not in reward and "infeasible" in reward:
            reward["feasibility_weight"] = reward["infeasible"]
        normalized["reward"] = reward

    policy_reg = normalized.get("policy_regularization")
    if isinstance(policy_reg, dict):
        policy_reg = dict(policy_reg)
    else:
        policy_reg = {}
        normalized["policy_regularization"] = policy_reg

    algo_cfg = normalized.get("algo")
    if isinstance(algo_cfg, dict):
        algo_cfg = dict(algo_cfg)
        if "clip_param" not in algo_cfg and "ppo_clip" in algo_cfg:
            algo_cfg["clip_param"] = algo_cfg["ppo_clip"]
        if "ppo_clip" not in algo_cfg and "clip_param" in algo_cfg:
            algo_cfg["ppo_clip"] = algo_cfg["clip_param"]
        if "value_loss_coef" not in algo_cfg and "value_coef" in algo_cfg:
            algo_cfg["value_loss_coef"] = algo_cfg["value_coef"]
        if "value_coef" not in algo_cfg and "value_loss_coef" in algo_cfg:
            algo_cfg["value_coef"] = algo_cfg["value_loss_coef"]
        normalized["algo"] = algo_cfg
    else:
        algo_cfg = {}

    # Legacy auto-mode for old configs: do not silently leak privileged info.
    explicit_obs = bool(observation)
    if "mode" not in observation:
        observation["mode"] = "safe_observable"
    if "include_oracle_cost" not in observation:
        observation["include_oracle_cost"] = False
    if "include_cost_prior_features" not in observation:
        observation["include_cost_prior_features"] = False
    observation["legacy_auto_enabled"] = False

    legacy_oracle = False
    if isinstance(reward, dict):
        legacy_oracle = bool(reward.get("use_oracle_cost_components", False)) or str(reward.get("mode", "")).strip().lower() == "oracle_aligned_cost"
    legacy_cost_prior = False
    if isinstance(scenario, dict):
        legacy_cost_prior = legacy_cost_prior or bool(scenario.get("include_cost_features_in_obs", False))
    if isinstance(algo_cfg, dict):
        legacy_cost_prior = legacy_cost_prior or str(algo_cfg.get("policy_head", "")).strip().lower() == "hybrid_gnn_cost"
    if isinstance(policy_reg, dict):
        legacy_cost_prior = legacy_cost_prior or bool(policy_reg.get("enabled", False))

    if not explicit_obs and (legacy_oracle or legacy_cost_prior):
        observation["legacy_auto_enabled"] = True
        if legacy_oracle:
            observation["mode"] = "oracle_debug"
            observation["include_oracle_cost"] = True
            observation["include_cost_prior_features"] = True
        else:
            observation["mode"] = "cost_prior_ablation"
            observation["include_cost_prior_features"] = True

    normalized["observation"] = observation

    if isinstance(normalized.get("scenario"), dict):
        sc = dict(normalized["scenario"])
        sc["observation_access_mode"] = str(observation.get("mode", "safe_observable"))
        sc["observation_include_oracle_cost"] = bool(observation.get("include_oracle_cost", False))
        sc["observation_include_cost_prior_features"] = bool(observation.get("include_cost_prior_features", False))
        sc["include_cost_features_in_obs"] = bool(observation.get("include_cost_prior_features", False))
        if sc["include_cost_features_in_obs"] and int(sc.get("node_feature_dim", 12)) < 20:
            sc["node_feature_dim"] = 20
        normalized["scenario"] = sc

    evaluation = dict(normalized.get("evaluation") or {})
    legacy_eval_keys = {
        "deterministic_eval_mode": "deterministic_eval_mode",
        "tie_break_eps": "tie_break_eps",
        "stochastic_seed": "stochastic_seed",
        "report_raw_argmax_always": "report_raw_argmax_always",
    }
    for legacy_key, eval_key in legacy_eval_keys.items():
        if legacy_key in normalized and eval_key not in evaluation:
            evaluation[eval_key] = normalized.pop(legacy_key)
    if evaluation:
        normalized["evaluation"] = evaluation

    model_cfg = dict(normalized.get("model") or {})
    temporal_cfg = dict(model_cfg.get("temporal") or {})
    scenario_cfg = normalized.get("scenario")
    scenario_enable_gnn = True
    if isinstance(scenario_cfg, dict):
        scenario_enable_gnn = bool(scenario_cfg.get("enable_gnn", True))
    topo = str(model_cfg.get("topology_encoder", "") or "").strip().lower()
    if not topo:
        topo = "static_gnn" if scenario_enable_gnn else "no_gnn"
    if topo not in {"no_gnn", "static_gnn", "temporal_gnn"}:
        topo = "static_gnn" if scenario_enable_gnn else "no_gnn"
    temporal_enabled = bool(temporal_cfg.get("enabled", False))
    if topo == "temporal_gnn":
        temporal_enabled = True
    if temporal_enabled and topo == "static_gnn":
        topo = "temporal_gnn"
    temporal_cfg["enabled"] = temporal_enabled
    model_cfg["topology_encoder"] = topo
    model_cfg["temporal"] = temporal_cfg
    normalized["model"] = model_cfg

    experiment = dict(normalized.get("experiment") or {})
    split = dict(experiment.get("split") or {})
    for key in ("train_seeds", "val_seeds", "test_seeds", "allow_debug_seed_overlap"):
        if key in normalized and key not in split:
            split[key] = normalized.pop(key)
    if split:
        experiment["split"] = split
    if experiment:
        normalized["experiment"] = experiment

    return normalized
