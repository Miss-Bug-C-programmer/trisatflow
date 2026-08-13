from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from trisatflow.baselines.lower_allocators import build_lower_allocator, lower_allocator_metadata
from trisatflow.baselines.registry import baseline_metadata, validate_baseline_for_formal
from trisatflow.config import load_config
from trisatflow.config_validation import is_formal_or_paper_config
from trisatflow.data.trace_manifest import audit_manifest_records, load_manifest_dir, load_manifest_file
from trisatflow.formal_gates import validate_formal_training_seed_count

SATEDGESIM_RECEIPT_FILES = [
    "SatEdgeSim/edu/weijunyong/satedgesim/server/ExecutionReceipt.java",
    "SatEdgeSim/edu/weijunyong/satedgesim/server/RlCompletionReceipt.java",
]
SATEDGESIM_RESOURCE_FILES = [
    "SatEdgeSim/edu/weijunyong/satedgesim/server/RlResourceProfile.java",
    "SatEdgeSim/edu/weijunyong/satedgesim/server/RlResourceBindingAudit.java",
    "SatEdgeSim/edu/weijunyong/satedgesim/server/RlNativeResourceBindingManager.java",
    "SatEdgeSim/edu/weijunyong/satedgesim/Network/DefaultNetworkModel.java",
    "SatEdgeSim/edu/weijunyong/satedgesim/DataCentersManager/DefaultEnergyModel.java",
]
VALID_ENCODER_MODES = {"shared_upper_detached_lower", "shared_joint", "separate_lower_encoder", "shared_frozen"}


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value in (None, ""):
        return False
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _check(name: str, status: str, message: str, *, details: Mapping[str, Any] | None = None) -> Dict[str, Any]:
    return {
        "name": name,
        "status": status,
        "message": message,
        "details": dict(details or {}),
    }


def _fail(name: str, message: str, *, details: Mapping[str, Any] | None = None) -> Dict[str, Any]:
    return _check(name, "failed", message, details=details)


def _pass(name: str, message: str, *, details: Mapping[str, Any] | None = None) -> Dict[str, Any]:
    return _check(name, "passed", message, details=details)


def _warn(name: str, message: str, *, details: Mapping[str, Any] | None = None) -> Dict[str, Any]:
    return _check(name, "warning", message, details=details)


def _skip(name: str, message: str, *, details: Mapping[str, Any] | None = None) -> Dict[str, Any]:
    return _check(name, "skipped", message, details=details)


def _resolve_root(raw: str | None, *, fallback: Path) -> Path:
    if raw:
        return Path(raw).expanduser().resolve()
    return fallback.resolve()


def _resolve_config_path(root: Path, raw: str) -> Path:
    path = Path(raw).expanduser()
    if path.is_absolute():
        return path
    return (root / path).resolve()


def _output_path(root: Path) -> Path:
    return root / "outputs" / "preflight" / "formal_readiness_report.json"


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _check_config(cfg: Any, config_path: Path, run_mode: str) -> List[Dict[str, Any]]:
    checks: List[Dict[str, Any]] = []
    scenario = cfg.scenario
    reward = cfg.reward
    observation = cfg.observation
    algo = cfg.algo
    split = cfg.experiment.split

    physical_enabled = bool(getattr(scenario.physical, "enabled", False))
    checks.append(
        _pass("scenario_physical_enabled", "scenario.physical.enabled=true")
        if physical_enabled
        else _fail("scenario_physical_enabled", "formal config requires scenario.physical.enabled=true")
    )

    reward_mode = str(getattr(reward, "mode", "")).strip().lower()
    checks.append(
        _pass("reward_physical_gate", "reward/config validation passed", details={"reward_mode": reward_mode})
        if reward_mode == "physical_weighted" and physical_enabled
        else _fail(
            "reward_physical_gate",
            "formal preflight requires reward.mode=physical_weighted with scenario.physical.enabled=true",
            details={"reward_mode": reward_mode, "scenario_physical_enabled": physical_enabled},
        )
    )

    formalish = is_formal_or_paper_config(cfg, config_path)
    checks.append(
        _pass("formal_config_identity", "config is marked paper/formal", details={"is_formal_or_paper_config": formalish})
        if formalish
        else (
            _warn("formal_config_identity", "smoke config is not marked paper/formal", details={"is_formal_or_paper_config": formalish})
            if run_mode == "smoke"
            else _fail("formal_config_identity", "formal run requires experiment.paper_ready=true or formal/paper config path")
        )
    )

    oracle_risks = {
        "scenario.mask_source": str(getattr(scenario, "mask_source", "")).strip().lower(),
        "observation.mode": str(getattr(observation, "mode", "")).strip().lower(),
        "observation.include_oracle_cost": bool(getattr(observation, "include_oracle_cost", False)),
        "observation.include_cost_prior_features": bool(getattr(observation, "include_cost_prior_features", False)),
        "reward.use_oracle_cost_components": bool(getattr(reward, "use_oracle_cost_components", False)),
    }
    has_oracle_mask = oracle_risks["scenario.mask_source"] == "oracle_trace"
    has_oracle_obs = (
        oracle_risks["observation.mode"] == "oracle_debug"
        or oracle_risks["observation.include_oracle_cost"]
        or oracle_risks["reward.use_oracle_cost_components"]
    )
    if has_oracle_mask or has_oracle_obs:
        checks.append(
            _warn("oracle_privileged_fields", "oracle/privileged fields allowed only for diagnostic smoke", details=oracle_risks)
            if run_mode == "smoke"
            else _fail("oracle_privileged_fields", "formal run cannot use oracle trace masks or privileged observation/reward fields", details=oracle_risks)
        )
    else:
        checks.append(_pass("oracle_privileged_fields", "no oracle trace mask or privileged oracle observation fields", details=oracle_risks))

    train_seeds = list(getattr(split, "train_seeds", []) or [])
    try:
        seed_metadata = validate_formal_training_seed_count(train_seeds, run_mode=run_mode)
        status = "warning" if run_mode == "smoke" else "passed"
        message = "smoke preflight does not assert formal seed count" if run_mode == "smoke" else "formal seed count is sufficient"
        checks.append(_check("formal_train_seed_count", status, message, details=seed_metadata))
    except ValueError as exc:
        checks.append(_fail("formal_train_seed_count", str(exc), details={"train_seeds": train_seeds}))

    encoder_mode = str(getattr(algo, "encoder_mode", "")).strip().lower()
    encoder_details = {
        "encoder_mode": encoder_mode,
        "lower_observation_mode": str(getattr(algo, "lower_observation_mode", "")),
        "stop_gradient_to_encoder_from_lower": bool(getattr(algo, "stop_gradient_to_encoder_from_lower", False)),
    }
    checks.append(
        _pass("encoder_mode_metadata", "encoder mode metadata is valid", details=encoder_details)
        if encoder_mode in VALID_ENCODER_MODES
        else _fail("encoder_mode_metadata", "encoder mode metadata is missing or invalid", details=encoder_details)
    )
    return checks


def _check_baseline(name: str | None, checkpoint: str | None, root: Path, run_mode: str) -> List[Dict[str, Any]]:
    if not name:
        return [_skip("baseline_formal_registry", "no baseline requested")]
    checkpoint_path = Path(checkpoint).expanduser() if checkpoint else None
    if checkpoint_path is not None and not checkpoint_path.is_absolute():
        checkpoint_path = (root / checkpoint_path).resolve()
    checkpoint_loaded = bool(checkpoint_path and checkpoint_path.is_file())
    try:
        meta = validate_baseline_for_formal(name, checkpoint_loaded=checkpoint_loaded)
        return [_pass("baseline_formal_registry", "baseline is allowed for formal evaluation", details=meta.to_dict())]
    except Exception as exc:  # noqa: BLE001
        meta = baseline_metadata(name, checkpoint_loaded=checkpoint_loaded)
        details = meta.to_dict()
        details["checkpoint"] = str(checkpoint_path) if checkpoint_path else ""
        details["checkpoint_exists"] = checkpoint_loaded
        check = _warn("baseline_formal_registry", str(exc), details=details) if run_mode == "smoke" else _fail("baseline_formal_registry", str(exc), details=details)
        return [check]


def _check_lower_allocator(cfg: Any, checkpoint: str | None, root: Path, run_mode: str) -> List[Dict[str, Any]]:
    lower_mode = str(getattr(cfg, "lower_action_mode", "") or "").strip().lower()
    if lower_mode not in {"same_learned", "same_learned_lower"}:
        return [_skip("same_learned_lower_allocator", "same learned lower allocator not requested", details={"lower_action_mode": lower_mode})]
    checkpoint_path = Path(checkpoint).expanduser() if checkpoint else None
    if checkpoint_path is not None and not checkpoint_path.is_absolute():
        checkpoint_path = (root / checkpoint_path).resolve()
    try:
        allocator = build_lower_allocator("same_learned", checkpoint=checkpoint_path, formal=(run_mode == "formal"), cfg=cfg)
        meta = lower_allocator_metadata(allocator)
        status = "passed" if bool(meta.get("formal_claim_allowed")) else ("warning" if run_mode == "smoke" else "failed")
        return [_check("same_learned_lower_allocator", status, "same learned lower allocator ABI checked", details=meta)]
    except Exception as exc:  # noqa: BLE001
        return [
            (_warn if run_mode == "smoke" else _fail)(
                "same_learned_lower_allocator",
                str(exc),
                details={"checkpoint": str(checkpoint_path) if checkpoint_path else "", "lower_action_mode": lower_mode},
            )
        ]


def _load_manifest_records(path: Path) -> List[Dict[str, Any]]:
    if path.is_dir():
        return load_manifest_dir(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return [dict(item) for item in data if isinstance(item, Mapping)]
    if isinstance(data, Mapping) and isinstance(data.get("records"), list):
        return [dict(item) for item in data["records"] if isinstance(item, Mapping)]
    return [load_manifest_file(path)]


def _check_trace_manifest(raw: str | None, root: Path, run_mode: str) -> List[Dict[str, Any]]:
    if not raw:
        return [_warn("trace_manifest_split_safety", "trace manifest not provided; formal trace leakage cannot be proven") if run_mode == "smoke" else _fail("trace_manifest_split_safety", "formal preflight requires --trace-manifest")]
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = (root / path).resolve()
    if not path.exists():
        return [_fail("trace_manifest_split_safety", f"trace manifest path does not exist: {path}")]
    try:
        records = _load_manifest_records(path)
        audit = audit_manifest_records(records)
    except Exception as exc:  # noqa: BLE001
        return [_fail("trace_manifest_split_safety", f"trace manifest audit failed: {type(exc).__name__}: {exc}")]
    if audit.get("audit_status") == "passed" and audit.get("leakage_risk") in {"none", "low"}:
        return [_pass("trace_manifest_split_safety", "trace manifest audit passed", details=audit)]
    status = "warning" if run_mode == "smoke" else "failed"
    return [_check("trace_manifest_split_safety", status, "trace manifest audit is not formal-safe", details=audit)]


def _check_smoke_outputs(cfg: Any, root: Path, run_mode: str) -> Dict[str, Any]:
    output_dir = Path(str(getattr(cfg, "output_dir", "outputs") or "outputs"))
    if not output_dir.is_absolute():
        output_dir = (root / output_dir).resolve()
    text = output_dir.as_posix().lower()
    if run_mode == "formal" and ("smoke" in text or "tiny" in text or "debug" in text):
        return _fail("smoke_outputs_not_consumed", "formal run output_dir must not point at smoke/tiny/debug outputs", details={"output_dir": str(output_dir)})
    return _pass("smoke_outputs_not_consumed", "output directory does not look like a smoke/tiny source", details={"output_dir": str(output_dir)})


def _check_satedgesim_semantics(root: Path) -> List[Dict[str, Any]]:
    checks: List[Dict[str, Any]] = []
    missing = [rel for rel in [*SATEDGESIM_RECEIPT_FILES, *SATEDGESIM_RESOURCE_FILES] if not (root / rel).is_file()]
    if missing:
        return [_warn("satedgesim_semantics_files", "SatEdgeSim semantic source files were not found at the supplied root; rerun preflight with --satedgesim-root on the deployment host", details={"missing": missing})]

    execution_text = (root / SATEDGESIM_RECEIPT_FILES[0]).read_text(encoding="utf-8")
    completion_text = (root / SATEDGESIM_RECEIPT_FILES[1]).read_text(encoding="utf-8")
    receipt_ok = all(token in execution_text for token in ("receiptStage", "Boolean taskCompleted", "Boolean taskSucceeded")) and all(
        token in completion_text for token in ("receiptStage", "decisionId", "taskCompleted", "taskSucceeded", "simlog_final_energy_wh")
    )
    checks.append(
        _pass("satedgesim_receipt_schema", "scheduling/completion receipt schema is present")
        if receipt_ok
        else _fail("satedgesim_receipt_schema", "SatEdgeSim receipt schema does not expose required scheduling/completion fields")
    )

    profile_text = (root / SATEDGESIM_RESOURCE_FILES[0]).read_text(encoding="utf-8")
    audit_text = (root / SATEDGESIM_RESOURCE_FILES[1]).read_text(encoding="utf-8")
    native_manager_text = (root / SATEDGESIM_RESOURCE_FILES[2]).read_text(encoding="utf-8")
    network_text = (root / SATEDGESIM_RESOURCE_FILES[3]).read_text(encoding="utf-8")
    energy_text = (root / SATEDGESIM_RESOURCE_FILES[4]).read_text(encoding="utf-8")
    native_profile_ok = (
        "native_scheduler_bound" in profile_text
        and "continuousApplied" in profile_text
        and "native_scheduler_bound is not implemented" not in profile_text
    )
    native_manager_ok = all(
        token in native_manager_text
        for token in (
            "vm_mips_scoped_min_active_share",
            "bindTask",
            "releaseTask",
            "setMips",
            "bandwidthShareForTask",
            "txPowerRatioForTask",
        )
    )
    native_network_ok = "RlNativeResourceBindingManager.attachToTransfer" in network_text and "getBandwidthShareClamped" in network_text
    native_power_ok = "getTxPowerRatioClamped" in energy_text and "transmissionEnergyConsumption" in energy_text
    metadata_ok = "full_hybrid_closed_loop_claim_allowed" in audit_text and "p.nativeSchedulerBound()" in audit_text
    native_ok = native_profile_ok and native_manager_ok and native_network_ok and native_power_ok and metadata_ok
    checks.append(
        _pass("satedgesim_native_binding_claim_guard", "native VM MIPS/network bandwidth/tx power binding hooks are present and metadata is guarded")
        if native_ok
        else _fail("satedgesim_native_binding_claim_guard", "native scheduler binding hooks or metadata are incomplete")
    )
    return checks


def _check_energy_source(cfg: Any) -> Dict[str, Any]:
    del cfg
    return _pass(
        "satedgesim_energy_source_semantics",
        "SatEdgeSim replay tooling requires explicit unit-qualified energy source",
        details={"allowed_sources": ["receipt_delta_wh", "simlog_final_wh", "estimator_expected_j"]},
    )


def _check_oracle_privileged_fields_from_raw_config(config_path: Path, run_mode: str) -> Dict[str, Any] | None:
    try:
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    except Exception:
        return None
    if not isinstance(payload, Mapping):
        return None
    scenario = payload.get("scenario") if isinstance(payload.get("scenario"), Mapping) else {}
    observation = payload.get("observation") if isinstance(payload.get("observation"), Mapping) else {}
    reward = payload.get("reward") if isinstance(payload.get("reward"), Mapping) else {}
    oracle_risks = {
        "scenario.mask_source": str(scenario.get("mask_source", "")).strip().lower(),
        "observation.mode": str(observation.get("mode", "")).strip().lower(),
        "observation.include_oracle_cost": bool(observation.get("include_oracle_cost", False)),
        "observation.include_cost_prior_features": bool(observation.get("include_cost_prior_features", False)),
        "reward.use_oracle_cost_components": bool(reward.get("use_oracle_cost_components", False)),
    }
    has_oracle = (
        oracle_risks["scenario.mask_source"] == "oracle_trace"
        or oracle_risks["observation.mode"] == "oracle_debug"
        or oracle_risks["observation.include_oracle_cost"]
        or oracle_risks["reward.use_oracle_cost_components"]
    )
    if not has_oracle:
        return None
    return (
        _warn("oracle_privileged_fields", "oracle/privileged fields allowed only for diagnostic smoke", details=oracle_risks)
        if run_mode == "smoke"
        else _fail("oracle_privileged_fields", "formal run cannot use oracle trace masks or privileged observation/reward fields", details=oracle_risks)
    )


def run_preflight(args: argparse.Namespace) -> Dict[str, Any]:
    trisatflow_root = _resolve_root(args.trisatflow_root, fallback=REPO_ROOT)
    satedgesim_root = _resolve_root(args.satedgesim_root, fallback=trisatflow_root.parent / "satedgeSimv2")
    config_path = _resolve_config_path(trisatflow_root, args.config)
    run_mode = str(args.run_mode).strip().lower()
    checks: List[Dict[str, Any]] = []

    try:
        cfg = load_config(config_path)
        checks.append(_pass("config_load", "config loaded and TrainConfig validation passed", details={"config": str(config_path)}))
        checks.extend(_check_config(cfg, config_path, run_mode))
        checks.extend(_check_baseline(args.baseline, args.checkpoint, trisatflow_root, run_mode))
        checks.extend(_check_lower_allocator(cfg, args.checkpoint, trisatflow_root, run_mode))
        checks.extend(_check_trace_manifest(args.trace_manifest, trisatflow_root, run_mode))
        checks.append(_check_smoke_outputs(cfg, trisatflow_root, run_mode))
        checks.append(_check_energy_source(cfg))
        config_sha256 = _sha256_file(config_path) if config_path.is_file() else ""
        config_payload = asdict(cfg)
    except Exception as exc:  # noqa: BLE001
        checks.append(_fail("config_load", f"config loading or validation failed: {type(exc).__name__}: {exc}", details={"config": str(config_path)}))
        oracle_check = _check_oracle_privileged_fields_from_raw_config(config_path, run_mode)
        if oracle_check is not None:
            checks.append(oracle_check)
        config_sha256 = _sha256_file(config_path) if config_path.is_file() else ""
        config_payload = {}

    checks.extend(_check_satedgesim_semantics(satedgesim_root))

    failed = [check for check in checks if check["status"] == "failed"]
    warnings = [check for check in checks if check["status"] == "warning"]
    formal_ready = run_mode == "formal" and not failed
    report = {
        "schema_version": "trisatflow_formal_readiness_preflight_v1",
        "run_mode": run_mode,
        "formal_ready": bool(formal_ready),
        "outputs_are_smoke_only": run_mode == "smoke",
        "formal_claim_allowed": bool(formal_ready),
        "trisatflow_root": str(trisatflow_root),
        "satedgesim_root": str(satedgesim_root),
        "config": str(config_path),
        "config_sha256": config_sha256,
        "baseline": args.baseline or "",
        "checkpoint": args.checkpoint or "",
        "trace_manifest": args.trace_manifest or "",
        "num_checks": len(checks),
        "num_failed": len(failed),
        "num_warnings": len(warnings),
        "checks": checks,
        "config_summary": {
            "output_dir": config_payload.get("output_dir", ""),
            "reward_mode": (config_payload.get("reward") or {}).get("mode"),
            "scenario_physical_enabled": ((config_payload.get("scenario") or {}).get("physical") or {}).get("enabled"),
            "train_seeds": (((config_payload.get("experiment") or {}).get("split") or {}).get("train_seeds")),
            "encoder_mode": (config_payload.get("algo") or {}).get("encoder_mode"),
        },
    }
    return report


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preflight formal readiness without launching training.")
    parser.add_argument("--trisatflow-root", type=str, default="")
    parser.add_argument("--satedgesim-root", type=str, default="")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument("--trace-manifest", type=str, default="")
    parser.add_argument("--baseline", type=str, default="")
    parser.add_argument("--run-mode", type=str, default="smoke", choices=["smoke", "formal"])
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    report = run_preflight(args)
    root = _resolve_root(args.trisatflow_root, fallback=REPO_ROOT)
    output_path = _output_path(root)
    _write_json(output_path, report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    print(f"formal_readiness_report={output_path}")
    if args.run_mode == "formal" and not bool(report.get("formal_ready", False)):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
