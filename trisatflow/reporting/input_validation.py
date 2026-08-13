from __future__ import annotations

import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from statistical_tests import build_significance_rows
from trisatflow.baselines.registry import baseline_metadata


DEPRECATED_METRICS = {"mean_system_cost", "final_mean_system_cost", "tail_mean_system_cost"}
ACCEPTED_COST_METRICS = ("final_normalized_system_cost", "normalized_system_cost")
REQUIRED_STATS_COLUMNS = {
    "method_a",
    "method_b",
    "p_value_raw",
    "p_value_holm",
    "n_independent_train_seeds",
    "status",
}
MIN_FORMAL_TRAIN_SEEDS = 3


@dataclass(frozen=True)
class ReportingInput:
    input_root: Path
    rows: List[Dict[str, Any]]
    significance_rows: List[Dict[str, Any]]
    contract_sha256: str
    metric_schema_version: str
    smoke_mode: bool


class ReportingInputError(ValueError):
    pass


def load_reporting_input(
    input_root: str | Path,
    *,
    allow_smoke_small_n: bool = False,
    formal: bool = False,
    min_train_seeds: int = MIN_FORMAL_TRAIN_SEEDS,
    primary_semantic_class: str = "",
) -> ReportingInput:
    if formal and allow_smoke_small_n:
        raise ReportingInputError("--allow-smoke-small-n is not allowed in formal mode")
    root = Path(input_root)
    if not root.is_dir():
        raise ReportingInputError(f"input root does not exist: {root}")
    rows = _load_summary_rows(root)
    smoke_mode = False
    if not rows:
        if not allow_smoke_small_n:
            raise ReportingInputError(f"no audited summary rows found under {root}")
        rows = _load_smoke_rows(root)
        smoke_mode = True
    if not rows:
        raise ReportingInputError(f"no reportable rows found under {root}")

    normalized = [_normalize_row(row, root=root) for row in rows]
    _validate_primary_semantic_class(normalized, primary_semantic_class)
    _validate_rows(
        normalized,
        allow_smoke_small_n=allow_smoke_small_n,
        formal=formal,
        min_train_seeds=min_train_seeds,
    )
    significance_rows = _load_or_build_significance_rows(root, normalized)
    contract = _single_value(normalized, "experiment_contract_sha256", "mixed experiment_contract_sha256")
    schema = _single_value(normalized, "metric_schema_version", "mixed metric schema")
    return ReportingInput(
        input_root=root,
        rows=normalized,
        significance_rows=significance_rows,
        contract_sha256=contract,
        metric_schema_version=schema,
        smoke_mode=smoke_mode,
    )


def _load_summary_rows(root: Path) -> List[Dict[str, str]]:
    paths = [
        root / "sweep" / "sweep_summary.csv",
        root / "learning" / "sweep_summary.csv",
        root / "sweep_summary.csv",
        root / "ablation_runs.csv",
        root / "baseline_episode_metrics.csv",
        root / "main_actual" / "sweep_summary.csv",
        root / "rules_actual" / "baseline_episode_metrics.csv",
        root / "learning_baselines" / "sweep_summary.csv",
        root / "formal_learning" / "sweep_summary.csv",
        root / "formal_ablation" / "ablation_runs.csv",
        root / "stress" / "ablation_runs.csv",
    ]
    rows: List[Dict[str, str]] = []
    for path in paths:
        rows.extend(_read_csv(path))
    return rows


def _load_smoke_rows(root: Path) -> List[Dict[str, Any]]:
    metrics = root / "train" / "metrics.csv"
    manifest = root / "train" / "manifest.json"
    metadata_path = root / "train" / "run_metadata.json"
    metric_rows = _read_csv(metrics)
    if not metric_rows or not manifest.is_file():
        return []
    manifest_payload = _read_json(manifest)
    metadata = _read_json(metadata_path) if metadata_path.is_file() else {}
    last = metric_rows[-1]
    contract = str(manifest_payload.get("experiment_contract_sha256") or metadata.get("experiment_contract_sha256") or "").strip()
    schema = str(
        last.get("metric_schema_version")
        or manifest_payload.get("metric_schema_version")
        or metadata.get("metric_schema_version")
        or ""
    ).strip()
    return [
        {
            **last,
            "status": "ok",
            "phase": "train",
            "protocol_role": "train",
            "seed": "0",
            "train_seed": "0",
            "upper_algo": last.get("upper_algo", "mappo"),
            "lower_algo": last.get("lower_algo", "maddpg"),
            "baseline": "",
            "observation_ablation": "",
            "experiment_contract_sha256": contract,
            "metric_schema_version": schema,
            "observation_mode": metadata.get("observation_mode", last.get("observation_mode", "safe_observable")),
            "include_oracle_cost": last.get("include_oracle_cost", "0"),
            "include_cost_prior_features": last.get("include_cost_prior_features", "0"),
            "metrics_csv": str(metrics),
            "checkpoint": str(root / "train" / "smoke_checkpoint.pt"),
        }
    ]


def _normalize_row(row: Dict[str, Any], *, root: Path) -> Dict[str, Any]:
    status = str(row.get("status", "ok") or "ok").strip().lower()
    if status not in {"ok", "success", ""}:
        raise ReportingInputError(f"non-ok row is not reportable: status={status!r}")
    metric = _metric_value(row)
    metrics_payload = _metrics_payload_for_row(row, root=root)
    contract = str(
        row.get("experiment_contract_sha256")
        or metrics_payload.get("experiment_contract_sha256")
        or row.get("fairness_contract_sha256")
        or ""
    ).strip()
    schema = str(row.get("metric_schema_version") or metrics_payload.get("metric_schema_version") or "").strip()
    out = dict(row)
    out["method"] = _method_label(row)
    out["metric_value"] = metric
    out["experiment_contract_sha256"] = contract
    out["metric_schema_version"] = schema
    out["phase"] = str(row.get("phase", "train") or "train").strip().lower()
    out["train_seed"] = _to_int(row.get("train_seed", row.get("seed", 0)), default=0)
    out["seed"] = _to_int(row.get("seed", out["train_seed"]), default=out["train_seed"])
    out["eval_seed"] = str(row.get("eval_seed", "") or "").strip()
    out["n_eval_seeds"] = _to_int(row.get("n_eval_seeds", 0), default=0)
    out["eval_seed_bank"] = str(row.get("eval_seed_bank", "") or "").strip()
    out["observation_mode"] = str(row.get("observation_mode", metrics_payload.get("observation_mode", "")) or "").strip()
    out["include_oracle_cost"] = _to_bool(row.get("include_oracle_cost", metrics_payload.get("include_oracle_cost", False)))
    out["include_cost_prior_features"] = _to_bool(row.get("include_cost_prior_features", metrics_payload.get("include_cost_prior_features", False)))
    out["trace_semantic_class"] = str(
        row.get("trace_semantic_class", metrics_payload.get("trace_semantic_class", "")) or ""
    ).strip()
    return out


def _metrics_payload_for_row(row: Dict[str, Any], *, root: Path) -> Dict[str, Any]:
    metrics_raw = str(row.get("metrics_csv", "") or "").strip()
    payload: Dict[str, Any] = {}
    if metrics_raw:
        path = Path(metrics_raw)
        if not path.is_absolute() and not path.exists():
            path = root / metrics_raw
        metric_rows = _read_csv(path)
        if metric_rows:
            payload.update(metric_rows[-1])
    output_raw = str(row.get("output_dir", "") or "").strip()
    if output_raw:
        output_dir = Path(output_raw)
        if not output_dir.is_absolute() and not output_dir.exists():
            output_dir = root / output_raw
        metadata = output_dir / "run_metadata.json"
        manifest = output_dir / "manifest.json"
        for path in (metadata, manifest):
            if path.is_file():
                payload.update(_read_json(path))
    return payload


def _metric_value(row: Dict[str, Any]) -> float:
    for key in ACCEPTED_COST_METRICS:
        value = _to_float(row.get(key))
        if value is not None:
            return value
    present_deprecated = [key for key in DEPRECATED_METRICS if str(row.get(key, "") or "").strip()]
    if present_deprecated:
        raise ReportingInputError(f"deprecated metric field is not allowed for paper reporting: {present_deprecated[0]}")
    raise ReportingInputError("missing normalized cost metric; expected final_normalized_system_cost or normalized_system_cost")


def _validate_rows(
    rows: List[Dict[str, Any]],
    *,
    allow_smoke_small_n: bool,
    formal: bool,
    min_train_seeds: int,
) -> None:
    if any(not str(row.get("experiment_contract_sha256", "")).strip() for row in rows):
        raise ReportingInputError("missing experiment_contract_sha256")
    if any(not str(row.get("metric_schema_version", "")).strip() for row in rows):
        raise ReportingInputError("missing metric_schema_version")
    _single_value(rows, "experiment_contract_sha256", "mixed experiment_contract_sha256")
    _single_value(rows, "metric_schema_version", "mixed metric schema")

    for row in rows:
        if str(row.get("observation_mode", "")).strip().lower() != "safe_observable":
            raise ReportingInputError("oracle/debug experiments are not allowed in paper reporting")
        if bool(row.get("include_oracle_cost")) or bool(row.get("include_cost_prior_features")):
            raise ReportingInputError("oracle/debug or privileged observation rows are not allowed")
        baseline = str(row.get("baseline", "") or "").strip().lower()
        if baseline:
            if _to_bool(row.get("placeholder", False)):
                raise ReportingInputError(f"placeholder baseline is not allowed: {baseline}")
            try:
                meta = baseline_metadata(baseline)
            except ValueError:
                meta = None
            if meta is not None and (meta.type == "placeholder" or not bool(meta.paper_ready)):
                raise ReportingInputError(f"placeholder/non-paper-ready baseline is not allowed: {baseline}")

    train_seeds = {int(row["train_seed"]) for row in rows if str(row.get("phase")) in {"train", "validation", "test"}}
    if len(train_seeds) < int(min_train_seeds) and not allow_smoke_small_n:
        raise ReportingInputError(
            f"too few independent train seeds: {len(train_seeds)} < {int(min_train_seeds)}; "
            "--allow-smoke-small-n is only for Stage 12 smoke"
        )
    has_test_bank = any(
        str(row.get("phase")) == "test"
        and (int(row.get("n_eval_seeds", 0) or 0) > 0 or bool(str(row.get("eval_seed_bank", "")).strip()))
        for row in rows
    )
    has_test_eval_rows = any(str(row.get("phase")) == "test" and str(row.get("eval_seed", "")).strip() for row in rows)
    if not (has_test_bank or has_test_eval_rows) and not allow_smoke_small_n:
        raise ReportingInputError("missing test bank aggregation")
    if formal and allow_smoke_small_n:
        raise ReportingInputError("--allow-smoke-small-n cannot be used in formal mode")


def _validate_primary_semantic_class(rows: List[Dict[str, Any]], primary_semantic_class: str) -> None:
    primary = str(primary_semantic_class or "").strip()
    if not primary:
        return
    classes = {str(row.get("trace_semantic_class", "") or "").strip() for row in rows}
    if "" in classes:
        raise ReportingInputError(f"missing trace_semantic_class for primary semantic class filter: {primary}")
    if classes != {primary}:
        raise ReportingInputError(f"mixed or non-primary trace_semantic_class input: expected={primary} actual={sorted(classes)}")


def _load_or_build_significance_rows(root: Path, rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    candidates = [
        root / "significance_tests.csv",
        root / "learning" / "significance_tests.csv",
        root / "tables" / "table_s2_statistical_tests.csv",
        root / "main_actual_summary" / "significance_tests.csv",
    ]
    for path in candidates:
        if path.is_file():
            stat_rows = _read_csv(path)
            _validate_stats_columns(stat_rows, path)
            return [dict(row) for row in stat_rows]
    sig_input = [
        {
            "phase": row.get("phase", "train"),
            "seed": row.get("train_seed", row.get("seed", 0)),
            "value": row["metric_value"],
            "upper_algo": row.get("upper_algo", ""),
            "lower_algo": row.get("lower_algo", ""),
            "baseline": row.get("baseline", ""),
            "observation_ablation": row.get("observation_ablation", ""),
        }
        for row in rows
    ]
    return build_significance_rows(sig_input, metric="final_normalized_system_cost")


def _validate_stats_columns(rows: List[Dict[str, Any]], path: Path) -> None:
    fieldnames = set(rows[0].keys()) if rows else set()
    if not rows:
        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            fieldnames = set(reader.fieldnames or [])
    missing = sorted(REQUIRED_STATS_COLUMNS - fieldnames)
    if missing:
        raise ReportingInputError(f"significance input missing required statistical columns {missing}: {path}")


def _method_label(row: Dict[str, Any]) -> str:
    baseline = str(row.get("baseline", "") or "").strip()
    if baseline:
        return baseline
    upper = str(row.get("upper_algo", "") or "").strip()
    lower = str(row.get("lower_algo", "") or "").strip()
    if upper or lower:
        return f"{upper}+{lower}".strip("+")
    return "unknown"


def _single_value(rows: Iterable[Dict[str, Any]], key: str, message: str) -> str:
    values = {str(row.get(key, "") or "").strip() for row in rows}
    values.discard("")
    if len(values) != 1:
        raise ReportingInputError(f"{message}: {sorted(values)}")
    return next(iter(values))


def _read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return [dict(row) for row in csv.DictReader(f)]


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return dict(payload) if isinstance(payload, dict) else {}


def _to_float(value: Any) -> float | None:
    try:
        if value in (None, "", "NA"):
            return None
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _to_int(value: Any, *, default: int = 0) -> int:
    try:
        if value in (None, ""):
            return int(default)
        return int(float(value))
    except (TypeError, ValueError):
        return int(default)


def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    return text in {"1", "1.0", "true", "yes", "y"}


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: List[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
