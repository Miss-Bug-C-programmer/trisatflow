from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping


CANONICAL_CONFIG_ROOT = Path("trisatflow/configs")
LEGACY_CONFIG_ROOT = Path("configs")

_OBSERVATION_MODES = {"safe_observable", "cost_prior_ablation", "oracle_debug"}
_REWARD_MODES = {"physical_weighted", "legacy_remote_biased", "oracle_aligned_cost"}
_OBS_NORMALIZATION_MODES = {"legacy", "trace_p95", "trace_log_quantile"}
_ACTION_MASK_LEGACY_MODES = {"visible_only", "mobility_safe", "completion_safe"}
_ACTION_MASK_LAYER_MODES = {"legacy", "none", "visibility", "completion_safe", "mobility_risk", "full"}
_QUEUE_MODES = {"single_queue", "multi_queue"}
_UPPER_ALGOS = {"mappo", "ippo", "iql", "vdn", "qmix"}
_LOWER_ALGOS = {"maddpg", "iddpg", "masac", "isac"}
_ENCODER_MODES = {"shared_upper_detached_lower", "shared_joint", "separate_lower_encoder", "shared_frozen"}
_PPO_UPDATE_MODES = {"standard_ppo", "legacy_compact"}
_PHYSICAL_METRIC_MODES = {"normalized", "physical", "dual"}
_COMPUTE_ENERGY_MODELS = {"kappa_cycles_f2", "kappa_f3_time"}
_LYAPUNOV_CLAIM_MODES = {"inspired_reward", "theoretical_dpp"}
_QUEUE_CAP_MODES = {"finite_buffer", "unbounded_eval"}
_MASK_SOURCES = {"measured", "predicted", "oracle_trace"}


def validate_train_config(cfg: Any, *, source: str = "") -> None:
    errors: list[str] = []
    where = f" ({source})" if source else ""

    source_chain = getattr(cfg, "config_source_chain", [])
    if source_chain and (
        not isinstance(source_chain, list) or not all(isinstance(item, str) and item for item in source_chain)
    ):
        errors.append(f"config_source_chain{where} must be a list of non-empty strings")

    obs_mode = str(getattr(getattr(cfg, "observation", None), "mode", "")).strip().lower()
    if obs_mode not in _OBSERVATION_MODES:
        errors.append(f"observation.mode{where} must be one of {sorted(_OBSERVATION_MODES)}, got {obs_mode!r}")
    observation = getattr(cfg, "observation", None)
    include_oracle_cost = bool(getattr(observation, "include_oracle_cost", False))
    include_cost_prior = bool(getattr(observation, "include_cost_prior_features", False))

    reward_mode = str(getattr(getattr(cfg, "reward", None), "mode", "")).strip().lower()
    if reward_mode not in _REWARD_MODES:
        errors.append(f"reward.mode{where} must be one of {sorted(_REWARD_MODES)}, got {reward_mode!r}")
    reward = getattr(cfg, "reward", None)
    use_oracle_components = bool(getattr(reward, "use_oracle_cost_components", False))

    if obs_mode == "safe_observable":
        if include_oracle_cost:
            errors.append(f"observation.include_oracle_cost{where} must be false when observation.mode='safe_observable'")
        if include_cost_prior:
            errors.append(f"observation.include_cost_prior_features{where} must be false when observation.mode='safe_observable'")
        if use_oracle_components:
            errors.append(f"reward.use_oracle_cost_components{where} must be false when observation.mode='safe_observable'")

    scenario = getattr(cfg, "scenario", None)
    queue_mode = str(getattr(scenario, "queue_mode", "single_queue")).strip().lower()
    if queue_mode not in _QUEUE_MODES:
        errors.append(f"scenario.queue_mode{where} must be one of {sorted(_QUEUE_MODES)}, got {queue_mode!r}")
    lyapunov_claim_mode = str(getattr(scenario, "lyapunov_claim_mode", "inspired_reward")).strip().lower()
    if lyapunov_claim_mode not in _LYAPUNOV_CLAIM_MODES:
        errors.append(
            f"scenario.lyapunov_claim_mode{where} must be one of {sorted(_LYAPUNOV_CLAIM_MODES)}, got {lyapunov_claim_mode!r}"
        )
    queue_cap_mode = str(getattr(scenario, "queue_cap_mode", "finite_buffer")).strip().lower()
    if queue_cap_mode not in _QUEUE_CAP_MODES:
        errors.append(f"scenario.queue_cap_mode{where} must be one of {sorted(_QUEUE_CAP_MODES)}, got {queue_cap_mode!r}")

    obs_norm_mode = str(getattr(scenario, "obs_normalization_mode", "legacy")).strip().lower()
    if obs_norm_mode not in _OBS_NORMALIZATION_MODES:
        errors.append(
            f"scenario.obs_normalization_mode{where} must be one of {sorted(_OBS_NORMALIZATION_MODES)}, got {obs_norm_mode!r}"
        )

    mask_mode = str(getattr(scenario, "action_mask_mode", "visible_only")).strip().lower()
    if mask_mode not in _ACTION_MASK_LEGACY_MODES:
        errors.append(f"scenario.action_mask_mode{where} must be one of {sorted(_ACTION_MASK_LEGACY_MODES)}, got {mask_mode!r}")
    layer_mode = str(getattr(scenario, "action_mask_layer_mode", "legacy")).strip().lower()
    if layer_mode not in _ACTION_MASK_LAYER_MODES:
        errors.append(f"scenario.action_mask_layer_mode{where} must be one of {sorted(_ACTION_MASK_LAYER_MODES)}, got {layer_mode!r}")
    mask_source = str(getattr(scenario, "mask_source", "predicted")).strip().lower()
    if mask_source not in _MASK_SOURCES:
        errors.append(f"scenario.mask_source{where} must be one of {sorted(_MASK_SOURCES)}, got {mask_source!r}")
    _validate_nonnegative(getattr(scenario, "mask_prediction_horizon_s", 0.0), "scenario.mask_prediction_horizon_s", errors, where)
    _validate_nonnegative(getattr(scenario, "link_lifetime_noise_std_s", 0.0), "scenario.link_lifetime_noise_std_s", errors, where)
    _validate_nonnegative(getattr(scenario, "completion_time_noise_std_s", 0.0), "scenario.completion_time_noise_std_s", errors, where)
    _validate_probability(getattr(scenario, "mask_false_positive_rate", 0.0), "scenario.mask_false_positive_rate", errors, where)
    _validate_probability(getattr(scenario, "mask_false_negative_rate", 0.0), "scenario.mask_false_negative_rate", errors, where)
    _validate_nonnegative(getattr(scenario, "mask_staleness_slots", 0), "scenario.mask_staleness_slots", errors, where)

    algo = getattr(cfg, "algo", None)
    upper_algo = str(getattr(algo, "upper_algo", "")).strip().lower()
    lower_algo = str(getattr(algo, "lower_algo", "")).strip().lower()
    if upper_algo not in _UPPER_ALGOS:
        errors.append(f"algo.upper_algo{where} must be one of {sorted(_UPPER_ALGOS)}, got {upper_algo!r}")
    if lower_algo not in _LOWER_ALGOS:
        errors.append(f"algo.lower_algo{where} must be one of {sorted(_LOWER_ALGOS)}, got {lower_algo!r}")
    encoder_mode = str(getattr(algo, "encoder_mode", "shared_frozen")).strip().lower()
    if encoder_mode not in _ENCODER_MODES:
        errors.append(f"algo.encoder_mode{where} must be one of {sorted(_ENCODER_MODES)}, got {encoder_mode!r}")
    ppo_mode = str(getattr(algo, "ppo_update_mode", "standard_ppo")).strip().lower()
    if ppo_mode not in _PPO_UPDATE_MODES:
        errors.append(f"algo.ppo_update_mode{where} must be one of {sorted(_PPO_UPDATE_MODES)}, got {ppo_mode!r}")

    _validate_seed_value(getattr(scenario, "seed", 0), "scenario.seed", errors, where)
    experiment = getattr(cfg, "experiment", None)
    split = getattr(experiment, "split", None)
    train_seeds = _as_seed_list(getattr(split, "train_seeds", []), "experiment.split.train_seeds", errors, where)
    val_seeds = _as_seed_list(getattr(split, "val_seeds", []), "experiment.split.val_seeds", errors, where)
    test_seeds = _as_seed_list(getattr(split, "test_seeds", []), "experiment.split.test_seeds", errors, where)
    if not bool(getattr(split, "allow_debug_seed_overlap", False)):
        overlap = (set(train_seeds) & set(val_seeds)) | (set(train_seeds) & set(test_seeds)) | (set(val_seeds) & set(test_seeds))
        if overlap:
            errors.append(
                f"experiment split seeds overlap{where} with allow_debug_seed_overlap=false: {sorted(overlap)}"
            )

    _validate_positive(getattr(scenario, "delay_s_per_unit", 1.0), "scenario.delay_s_per_unit", errors, where)
    _validate_positive(getattr(scenario, "energy_j_per_unit", 1.0), "scenario.energy_j_per_unit", errors, where)
    _validate_positive(getattr(scenario, "queue_cycles_per_unit", 1.0), "scenario.queue_cycles_per_unit", errors, where)
    _validate_positive(getattr(scenario, "cpu_ghz_per_unit", 1.0), "scenario.cpu_ghz_per_unit", errors, where)
    _validate_positive(getattr(scenario, "rate_mbps_per_unit", 1.0), "scenario.rate_mbps_per_unit", errors, where)
    _validate_positive(getattr(scenario, "bandwidth_mbps_per_unit", 1.0), "scenario.bandwidth_mbps_per_unit", errors, where)
    _validate_positive(getattr(scenario, "power_w_per_unit", 1.0), "scenario.power_w_per_unit", errors, where)
    _validate_positive(getattr(scenario, "task_size_bits_per_unit", 1.0), "scenario.task_size_bits_per_unit", errors, where)
    _validate_positive(getattr(scenario, "workload_cycles_per_unit", 1.0), "scenario.workload_cycles_per_unit", errors, where)

    scenario_physical = getattr(scenario, "physical", None)
    physical = scenario_physical
    if physical is not None:
        metric_mode = str(getattr(physical, "metric_mode", "dual")).strip().lower()
        if metric_mode not in _PHYSICAL_METRIC_MODES:
            errors.append(f"physical.metric_mode{where} must be one of {sorted(_PHYSICAL_METRIC_MODES)}, got {metric_mode!r}")
        compute_model = str(getattr(physical, "compute_energy_model", "kappa_cycles_f2")).strip().lower()
        if compute_model not in _COMPUTE_ENERGY_MODELS:
            errors.append(
                f"physical.compute_energy_model{where} must be one of {sorted(_COMPUTE_ENERGY_MODELS)}, got {compute_model!r}"
            )
        for field in (
            "slot_duration_s",
            "task_size_bits_mean",
            "cycles_per_bit_mean",
            "leo_cpu_hz",
            "geo_cpu_hz",
            "ground_cpu_hz",
            "local_rate_bps",
            "isl_base_rate_bps",
            "geo_base_rate_bps",
            "ground_base_rate_bps",
            "max_tx_power_w",
            "queue_cap_cycles",
        ):
            _validate_positive(getattr(physical, field, 1.0), f"physical.{field}", errors, where)
        for field in ("task_size_bits_std", "cycles_per_bit_std", "kappa"):
            _validate_nonnegative(getattr(physical, field, 0.0), f"physical.{field}", errors, where)
        queue_unit = str(getattr(physical, "queue_unit", "cycles")).strip().lower()
        if queue_unit != "cycles":
            errors.append(f"physical.queue_unit{where} must be 'cycles' for dimensioned mode, got {queue_unit!r}")

    experiment = getattr(cfg, "experiment", None)
    paper_ready = bool(getattr(experiment, "paper_ready", False))
    diagnostic_oracle_allowed = bool(getattr(experiment, "diagnostic_oracle_allowed", False))
    physical_enabled = bool(getattr(scenario_physical, "enabled", False))
    src_text = str(source or "").replace("\\", "/").lower()
    nonformal_diagnostic_config = any(
        part in src_text for part in ("/stress/", "/debug/", "configs/stress/", "configs/debug/", "configs/small.yaml")
    )
    reward_mode_explicit = bool(getattr(cfg, "_reward_mode_explicit", False))
    formal_or_paper_path = bool(is_formal_or_paper_config(cfg, source)) if "is_formal_or_paper_config" in globals() else False
    if (reward_mode_explicit or formal_or_paper_path) and source and source != "<default>" and reward_mode == "physical_weighted" and not physical_enabled and not nonformal_diagnostic_config:
        errors.append("physical_weighted reward requires scenario.physical.enabled=true")
    if paper_ready and not physical_enabled:
        errors.append("paper-ready/formal config requires scenario.physical.enabled=true")
    if paper_ready and mask_source == "oracle_trace" and not diagnostic_oracle_allowed:
        errors.append("formal/paper-ready config cannot use scenario.mask_source='oracle_trace'")

    output_dir = str(getattr(cfg, "output_dir", "") or "").strip()
    if not output_dir:
        errors.append(f"output_dir{where} must not be empty")
    elif Path(output_dir).as_posix() in {"/", "."}:
        errors.append(f"output_dir{where} must not be '/' or '.'")

    if errors:
        raise ValueError("Invalid TrainConfig:\n- " + "\n- ".join(errors))


def validate_experiment_matrix_config(payload: Mapping[str, Any], *, source: str = "") -> None:
    errors: list[str] = []
    where = f" ({source})" if source else ""
    if not isinstance(payload, Mapping):
        raise ValueError(f"experiment matrix config{where} must be a mapping")
    for key in ("profiles", "architectures", "baselines", "seeds"):
        values = payload.get(key)
        if not isinstance(values, list) or not values:
            errors.append(f"{key}{where} must be a non-empty list")
    seeds = payload.get("seeds", [])
    if isinstance(seeds, list):
        for idx, seed in enumerate(seeds):
            _validate_seed_value(seed, f"seeds[{idx}]", errors, where)
    if errors:
        raise ValueError("Invalid experiment matrix config:\n- " + "\n- ".join(errors))


def canonicalize_train_config_path(path: str | Path) -> tuple[Path, str | None]:
    input_path = Path(path)
    normalized = Path(input_path.as_posix())
    legacy_prefix = LEGACY_CONFIG_ROOT.as_posix() + "/"
    if normalized.as_posix().startswith(legacy_prefix):
        candidate = CANONICAL_CONFIG_ROOT / normalized.as_posix()[len(legacy_prefix) :]
        if candidate.exists():
            msg = (
                f"config path '{input_path}' is deprecated; canonical path is '{candidate.as_posix()}'. "
                "Please migrate to canonical config root."
            )
            return candidate, msg
    return input_path, None


def validate_wrapper_payload(payload: Mapping[str, Any]) -> tuple[bool, str]:
    if not isinstance(payload, Mapping):
        return False, ""
    if not bool(payload.get("_deprecated_wrapper", False)):
        return False, ""
    canonical = str(payload.get("canonical_config", "")).strip()
    if not canonical:
        raise ValueError("deprecated config wrapper missing 'canonical_config'")
    return True, canonical


def _validate_positive(value: Any, field: str, errors: list[str], where: str) -> None:
    try:
        val = float(value)
    except (TypeError, ValueError):
        errors.append(f"{field}{where} must be numeric, got {value!r}")
        return
    if not (val > 0.0):
        errors.append(f"{field}{where} must be > 0, got {val}")


def _validate_nonnegative(value: Any, field: str, errors: list[str], where: str) -> None:
    try:
        val = float(value)
    except (TypeError, ValueError):
        errors.append(f"{field}{where} must be numeric, got {value!r}")
        return
    if val < 0.0:
        errors.append(f"{field}{where} must be >= 0, got {val}")


def _validate_probability(value: Any, field: str, errors: list[str], where: str) -> None:
    try:
        val = float(value)
    except (TypeError, ValueError):
        errors.append(f"{field}{where} must be numeric, got {value!r}")
        return
    if val < 0.0 or val > 1.0:
        errors.append(f"{field}{where} must be in [0, 1], got {val}")


def _validate_seed_value(value: Any, field: str, errors: list[str], where: str) -> None:
    try:
        seed = int(value)
    except (TypeError, ValueError):
        errors.append(f"{field}{where} must be integer >= 0, got {value!r}")
        return
    if seed < 0:
        errors.append(f"{field}{where} must be >= 0, got {seed}")


def _as_seed_list(value: Any, field: str, errors: list[str], where: str) -> list[int]:
    if value in (None, ""):
        return []
    if not isinstance(value, Iterable) or isinstance(value, (str, bytes)):
        errors.append(f"{field}{where} must be a list of integers")
        return []
    out: list[int] = []
    for idx, item in enumerate(value):
        try:
            seed = int(item)
        except (TypeError, ValueError):
            errors.append(f"{field}[{idx}]{where} must be integer >= 0, got {item!r}")
            continue
        if seed < 0:
            errors.append(f"{field}[{idx}]{where} must be >= 0, got {seed}")
            continue
        out.append(seed)
    return out


def is_formal_or_paper_config(cfg: Any, source: str | Path | None = None) -> bool:
    """Return whether a config should be treated as formal/paper-ready.

    This is intentionally conservative: explicit experiment.paper_ready wins,
    otherwise only canonical paper/base paths or paper_ready output directories
    identify paper-style configs. Stress/debug configs remain non-formal unless
    explicitly marked.
    """

    experiment = getattr(cfg, "experiment", None)
    if bool(getattr(experiment, "paper_ready", False)):
        return True
    src = Path(source).as_posix().lower() if source is not None else ""
    if "/debug/" in src or "/stress/" in src or "debug" in src or "stress" in src:
        return False
    if "/trisatflow/configs/paper/" in src or "trisatflow/configs/paper/" in src:
        return True
    if "/trisatflow/configs/base/" in src or "trisatflow/configs/base/" in src:
        return "paper_ready" in str(getattr(cfg, "output_dir", "")).lower()
    return False
