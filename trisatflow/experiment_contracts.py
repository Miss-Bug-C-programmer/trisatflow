from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Iterable

from trisatflow.envs.physical_metrics import METRIC_SCHEMA_VERSION

SOFTWARE_SCHEMA_VERSION = "paper_ready_v3.stage1"


def file_sha256(path: str | Path) -> str:
    trace_path = Path(path)
    if not str(path).strip():
        return ""
    h = hashlib.sha256()
    with trace_path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def trace_sha256_for_config(cfg: Any, *, base_dir: str | Path | None = None) -> str:
    scenario = getattr(cfg, "scenario", None)
    trace_path = str(getattr(scenario, "topology_trace_path", "") or "").strip()
    if not trace_path:
        return ""
    path = Path(trace_path)
    if not path.is_absolute():
        root = Path(base_dir) if base_dir is not None else Path.cwd()
        path = root / path
    return file_sha256(path)


def write_contract_artifacts(output_dir: str | Path, cfg: Any, *, base_dir: str | Path | None = None) -> tuple[dict[str, Any], str]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    contract = resolve_contract(cfg, trace_sha256_for_config(cfg, base_dir=base_dir))
    digest = contract_sha256(contract)
    (out / "experiment_contract.json").write_text(
        json.dumps(contract, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (out / "experiment_contract_sha256.txt").write_text(f"{digest}\n", encoding="utf-8")
    return contract, digest


def resolve_contract(cfg: Any, trace_sha256: str) -> dict[str, Any]:
    scenario = getattr(cfg, "scenario", None)
    observation = getattr(cfg, "observation", None)
    reward = getattr(cfg, "reward", None)
    regularization = getattr(cfg, "policy_regularization", None)
    model = getattr(cfg, "model", None)
    algo = getattr(cfg, "algo", None)

    reward_payload = _dataclass_payload(
        reward,
        include=(
            "mode",
            "delay",
            "energy",
            "queue",
            "violation",
            "infeasible",
            "load_balance",
            "lyapunov_v",
            "offload_gain",
            "local_queue_pressure",
            "remote_feasible_bonus",
            "action_balance_bonus",
            "selected_when_visible_bonus",
            "cost_normalization_enabled",
            "per_tier_cost_normalization",
            "ground_congestion_penalty",
            "geo_delay_penalty",
            "local_queue_penalty",
            "neighbor_link_penalty",
            "remote_bonus",
            "local_penalty",
            "neighbor_penalty",
            "geo_penalty",
            "ground_penalty",
            "use_oracle_cost_components",
            "use_lower_effect_in_upper_reward",
            "include_energy",
            "delay_weight",
            "queue_weight",
            "transmission_weight",
            "compute_weight",
            "feasibility_weight",
            "include_failure_risk",
            "failure_penalty_weight",
        ),
    )

    return {
        "software_schema_version": SOFTWARE_SCHEMA_VERSION,
        "metric_schema_version": METRIC_SCHEMA_VERSION,
        "observation": {
            "mode": str(getattr(observation, "mode", "")),
            "include_oracle_cost": bool(getattr(observation, "include_oracle_cost", False)),
            "include_cost_prior_features": bool(getattr(observation, "include_cost_prior_features", False)),
        },
        "reward": reward_payload,
        "policy_regularization": {
            "enabled": bool(getattr(regularization, "enabled", False)),
            "mode": str(getattr(regularization, "mode", "")),
            "weight": float(getattr(regularization, "weight", 0.0)),
        },
        "environment": {
            "action_mask": {
                "enabled": bool(getattr(scenario, "action_mask_enabled", True)),
                "legacy_mode": str(getattr(scenario, "action_mask_mode", "")),
                "mode": str(getattr(scenario, "action_mask_layer_mode", "")),
                "enable_visibility_mask": bool(getattr(scenario, "enable_visibility_mask", False)),
                "enable_completion_safe_mask": bool(getattr(scenario, "enable_completion_safe_mask", False)),
                "enable_mobility_risk_mask": bool(getattr(scenario, "enable_mobility_risk_mask", False)),
            },
        },
        "trace": {
            "topology_mode": str(getattr(scenario, "topology_mode", "")),
            "path": str(getattr(scenario, "topology_trace_path", "") or ""),
            "sha256": str(trace_sha256),
        },
        "scenario": {
            "n_leo": int(getattr(scenario, "n_leo", 0)),
            "steps": int(getattr(scenario, "episode_len", 0)),
            "success_profile": str(getattr(scenario, "success_profile", "")),
            "action_space_architecture": str(getattr(scenario, "action_space_architecture", "")),
        },
        "physical_units": {
            "delay_s_per_unit": float(getattr(scenario, "delay_s_per_unit", 1.0)),
            "energy_j_per_unit": float(getattr(scenario, "energy_j_per_unit", 1.0)),
            "queue_cycles_per_unit": float(getattr(scenario, "queue_cycles_per_unit", 1.0)),
            "queue_tasks_per_unit": 1.0,
            "cpu_ghz_per_unit": float(getattr(scenario, "cpu_ghz_per_unit", 1.0)),
            "rate_mbps_per_unit": float(getattr(scenario, "rate_mbps_per_unit", 1.0)),
            "bandwidth_mbps_per_unit": float(getattr(scenario, "bandwidth_mbps_per_unit", 1.0)),
            "power_w_per_unit": float(getattr(scenario, "power_w_per_unit", 1.0)),
            "task_size_bits_per_unit": float(getattr(scenario, "task_size_bits_per_unit", 1.0)),
            "workload_cycles_per_unit": float(getattr(scenario, "workload_cycles_per_unit", 1.0)),
        },
        "encoder": {
            "topology_encoder": str(getattr(model, "topology_encoder", "")),
            "temporal": _dataclass_payload(getattr(model, "temporal", None)),
            "algo_encoder_mode": str(getattr(algo, "encoder_mode", "")),
            "gnn_hidden_dim": int(getattr(algo, "gnn_hidden_dim", 0)),
            "policy_head": str(getattr(algo, "policy_head", "")),
        },
    }


def contract_sha256(payload: dict[str, Any]) -> str:
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def assert_paper_safe(cfg: Any) -> None:
    errors: list[str] = []
    observation = getattr(cfg, "observation", None)
    reward = getattr(cfg, "reward", None)
    regularization = getattr(cfg, "policy_regularization", None)

    if str(getattr(observation, "mode", "")).strip().lower() != "safe_observable":
        errors.append("observation.mode must be safe_observable")
    if bool(getattr(observation, "include_oracle_cost", False)):
        errors.append("observation.include_oracle_cost must be false")
    if bool(getattr(observation, "include_cost_prior_features", False)):
        errors.append("observation.include_cost_prior_features must be false")
    if str(getattr(reward, "mode", "")).strip().lower() != "physical_weighted":
        errors.append("reward.mode must be physical_weighted")
    if bool(getattr(reward, "use_oracle_cost_components", False)):
        errors.append("reward.use_oracle_cost_components must be false")
    if bool(getattr(regularization, "enabled", False)):
        errors.append("policy_regularization.enabled must be false")
    if str(getattr(regularization, "mode", "")).strip().lower() != "none":
        errors.append("policy_regularization.mode must be none")
    if abs(float(getattr(regularization, "weight", 0.0))) > 1.0e-12:
        errors.append("policy_regularization.weight must be 0.0")

    if errors:
        raise ValueError("Config is not paper-safe:\n- " + "\n- ".join(errors))


def assert_same_contract(lhs: dict[str, Any], rhs: dict[str, Any], allowed_diff_paths: Iterable[str] = ()) -> None:
    allowed = {str(path).strip(".") for path in allowed_diff_paths}
    diffs = [path for path in _diff_paths(lhs, rhs) if not _path_allowed(path, allowed)]
    if diffs:
        raise ValueError("Experiment contracts differ:\n- " + "\n- ".join(sorted(diffs)))


def contract_diff_paths(lhs: dict[str, Any], rhs: dict[str, Any]) -> list[str]:
    return sorted(_diff_paths(lhs, rhs))


def _dataclass_payload(obj: Any, include: tuple[str, ...] | None = None) -> dict[str, Any]:
    if obj is None:
        return {}
    payload = asdict(obj) if is_dataclass(obj) else dict(obj)
    if include is None:
        return payload
    return {key: payload.get(key) for key in include if key in payload}


def _diff_paths(lhs: Any, rhs: Any, prefix: str = "") -> list[str]:
    if isinstance(lhs, dict) and isinstance(rhs, dict):
        paths: list[str] = []
        for key in sorted(set(lhs) | set(rhs)):
            child = f"{prefix}.{key}" if prefix else str(key)
            paths.extend(_diff_paths(lhs.get(key), rhs.get(key), child))
        return paths
    if isinstance(lhs, list) and isinstance(rhs, list):
        max_len = max(len(lhs), len(rhs))
        paths = []
        for idx in range(max_len):
            child = f"{prefix}[{idx}]"
            left_item = lhs[idx] if idx < len(lhs) else None
            right_item = rhs[idx] if idx < len(rhs) else None
            paths.extend(_diff_paths(left_item, right_item, child))
        return paths
    return [] if lhs == rhs else [prefix]


def _path_allowed(path: str, allowed: set[str]) -> bool:
    if path in allowed:
        return True
    return any(path.startswith(f"{item}.") or path.startswith(f"{item}[") for item in allowed)
