from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping

import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_stress_suite import run_suite
from trisatflow.baselines.registry import baseline_metadata
from trisatflow.config import load_config
from trisatflow.reporting.formal_input_guard import validate_formal_result_payload


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_mapping(path: Path) -> Dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"matrix/config file must contain a mapping: {path}")
    return data


def _resolve_path(raw: str | Path, *, base: Path = PROJECT_ROOT) -> Path:
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = base / path
    return path.resolve()


def _stress_configs_from_matrix(config: str, matrix_config: str) -> List[Path]:
    configs: List[Path] = []
    if matrix_config:
        path = _resolve_path(matrix_config)
        payload = _load_mapping(path)
        raw_configs = payload.get("stress_configs") or payload.get("configs") or payload.get("matrix") or []
        if not isinstance(raw_configs, list):
            raise ValueError("matrix-config stress_configs must be a list")
        for item in raw_configs:
            if isinstance(item, Mapping):
                raw = item.get("config") or item.get("path")
            else:
                raw = item
            if not raw:
                continue
            configs.append(_resolve_path(str(raw)))
    if config:
        configs.append(_resolve_path(config))
    if not configs:
        raise ValueError("provide --config or --matrix-config with stress_configs")
    return sorted(dict.fromkeys(configs))


def _stress_metadata(config_path: Path) -> Dict[str, Any]:
    raw = _load_mapping(config_path)
    stress = dict(raw.get("stress") or {})
    cfg = load_config(config_path)
    return {
        "config": str(config_path),
        "config_sha256": _sha256_file(config_path),
        "stress_name": str(stress.get("stress_name") or config_path.stem),
        "train_n_leo": int(stress.get("train_n_leo", 16)),
        "test_n_leo": int(cfg.scenario.n_leo),
        "n_geo": int(cfg.scenario.n_geo),
        "n_ground": int(cfg.scenario.n_ground),
        "mask_source": str(getattr(cfg.scenario, "mask_source", "")),
        "observation_mode": str(getattr(cfg.observation, "mode", "")),
        "diagnostic_oracle_allowed": bool(getattr(cfg.experiment, "diagnostic_oracle_allowed", False)),
        "gateway_visibility_mode": str(stress.get("gateway_visibility_mode", "unknown")),
        "burst_prob": float(stress.get("burst_prob", getattr(cfg.scenario, "burst_prob", 0.0))),
        "deadline_mode": str(stress.get("deadline_mode", "unknown")),
        "mask_noise_level": float(stress.get("mask_noise_level", 0.0)),
        "domain_shift": bool(stress.get("domain_shift", False)),
        "topology_trace_path": str(getattr(cfg.scenario, "topology_trace_path", "")),
    }


def _load_checkpoint_metadata(checkpoint: Path) -> Dict[str, Any]:
    sidecar = checkpoint.with_suffix(checkpoint.suffix + ".metadata.json")
    if sidecar.is_file():
        data = json.loads(sidecar.read_text(encoding="utf-8"))
        return dict(data) if isinstance(data, dict) else {}
    if checkpoint.suffix.lower() == ".json":
        data = json.loads(checkpoint.read_text(encoding="utf-8"))
        return dict(data) if isinstance(data, dict) else {}
    try:
        payload = torch.load(checkpoint, map_location="cpu")
    except Exception:
        return {}
    if isinstance(payload, dict):
        for key in ("metadata", "formal_metadata", "checkpoint_metadata"):
            value = payload.get(key)
            if isinstance(value, dict):
                return dict(value)
    return {}


def checkpoint_supports_variable_n(checkpoint: Path, *, train_n_leo: int, test_n_leo: int) -> bool:
    if int(train_n_leo) == int(test_n_leo):
        return True
    metadata = _load_checkpoint_metadata(checkpoint)
    return bool(
        metadata.get("variable_n_transfer_supported")
        or metadata.get("supports_variable_constellation_size")
        or metadata.get("supports_variable_n")
    )


def _validate_formal_inputs(
    *,
    stress_rows: List[Dict[str, Any]],
    checkpoint: Path | None,
    trace_manifest: Path | None,
    baseline: str,
    max_episodes: int | None,
    max_steps: int | None,
    allow_diagnostic_inputs: bool,
) -> None:
    if max_episodes is not None or max_steps is not None:
        raise ValueError("formal stress runner does not allow smoke --max-episodes/--max-steps caps")
    if allow_diagnostic_inputs:
        raise ValueError("formal stress runner cannot use --allow-diagnostic-inputs")
    if checkpoint is None or not checkpoint.is_file():
        raise ValueError("formal stress runner requires a real --checkpoint so checkpoint sha256 can be recorded")
    if trace_manifest is None or not trace_manifest.is_file():
        raise ValueError("formal stress runner requires --trace-manifest so trace manifest sha256 can be recorded")
    if baseline:
        meta = baseline_metadata(baseline)
        if not bool(meta.paper_ready):
            raise ValueError(f"formal stress runner rejects paper_ready=false baseline: {baseline}")
    for row in stress_rows:
        if str(row.get("mask_source", "")).strip().lower() == "oracle_trace" or bool(row.get("diagnostic_oracle_allowed")):
            raise ValueError("formal stress runner rejects diagnostic oracle mask inputs")
        if not checkpoint_supports_variable_n(
            checkpoint,
            train_n_leo=int(row["train_n_leo"]),
            test_n_leo=int(row["test_n_leo"]),
        ):
            raise ValueError("checkpoint architecture does not support variable constellation size")


def run_formal_stress_matrix(args: argparse.Namespace) -> Dict[str, Any]:
    configs = _stress_configs_from_matrix(args.config, args.matrix_config)
    checkpoint = _resolve_path(args.checkpoint) if args.checkpoint else None
    trace_manifest = _resolve_path(args.trace_manifest) if args.trace_manifest else None
    output_dir = _resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stress_rows = [_stress_metadata(path) for path in configs]
    run_mode = str(args.run_mode).strip().lower()
    checkpoint_sha256 = _sha256_file(checkpoint) if checkpoint and checkpoint.is_file() else ""
    trace_manifest_sha256 = _sha256_file(trace_manifest) if trace_manifest and trace_manifest.is_file() else ""
    matrix_sha256 = _sha256_file(_resolve_path(args.matrix_config)) if args.matrix_config else ""

    if run_mode == "formal":
        _validate_formal_inputs(
            stress_rows=stress_rows,
            checkpoint=checkpoint,
            trace_manifest=trace_manifest,
            baseline=str(args.baseline or ""),
            max_episodes=args.max_episodes,
            max_steps=args.max_steps,
            allow_diagnostic_inputs=bool(args.allow_diagnostic_inputs),
        )

    summary: Dict[str, Any] = {
        "schema_version": "trisatflow_formal_stress_matrix_v1",
        "run_mode": run_mode,
        "outputs_are_smoke_only": run_mode == "smoke",
        "tiny_results_are_not_paper_results": run_mode == "smoke",
        "formal_claim_allowed": False,
        "paper_ready": run_mode == "formal",
        "status": "FORMAL_STRESS_PREFLIGHT_READY" if run_mode == "formal" else "SMOKE_STRESS_COMPLETED",
        "config": args.config,
        "matrix_config": args.matrix_config,
        "matrix_config_sha256": matrix_sha256,
        "checkpoint": str(checkpoint) if checkpoint else "",
        "checkpoint_sha256": checkpoint_sha256,
        "trace_manifest": str(trace_manifest) if trace_manifest else "",
        "trace_manifest_sha256": trace_manifest_sha256,
        "baseline": str(args.baseline or ""),
        "stress_configs": stress_rows,
        "max_episodes": args.max_episodes,
        "max_steps": args.max_steps,
        "allow_diagnostic_inputs": bool(args.allow_diagnostic_inputs),
        "variable_n_transfer_guard": "fail-fast unless checkpoint metadata declares variable_n_transfer_supported=true",
    }

    if run_mode == "smoke":
        episodes = int(args.max_episodes or 1)
        steps = int(args.max_steps or 2)
        suite_summary = run_suite(
            configs,
            policy="checkpoint" if checkpoint else "random_visible",
            checkpoint=str(checkpoint) if checkpoint else None,
            episodes=max(1, episodes),
            steps=max(1, steps),
            device=str(args.device),
            output_dir=output_dir,
        )
        summary.update(suite_summary)

    validate_formal_result_payload(summary, source="stress_runner_self_check", allow_diagnostic_inputs=True, output_path=output_dir / "diagnostic_stress_summary.json")
    out = output_dir / "stress_summary.json"
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run formal/smoke TriSatFlow stress-transfer matrix gates.")
    parser.add_argument("--config", type=str, default="")
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument("--matrix-config", type=str, default="")
    parser.add_argument("--run-mode", type=str, required=True, choices=["smoke", "formal"])
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--allow-diagnostic-inputs", action="store_true")
    parser.add_argument("--trace-manifest", type=str, default="")
    parser.add_argument("--baseline", type=str, default="")
    parser.add_argument("--device", type=str, default="cpu")
    return parser.parse_args(argv)


def main(argv: List[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        summary = run_formal_stress_matrix(args)
    except Exception as exc:  # noqa: BLE001
        print(f"FORMAL_STRESS_MATRIX_FAILED {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
