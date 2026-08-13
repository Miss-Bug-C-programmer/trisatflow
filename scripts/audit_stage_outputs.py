from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def _fail(message: str) -> None:
    raise SystemExit(f"audit_stage_outputs failed: {message}")


def _require_file(path: Path, *, label: str, nonempty: bool = True) -> Path:
    if not path.is_file():
        _fail(f"missing {label}: {path}")
    if nonempty and path.stat().st_size <= 0:
        _fail(f"empty {label}: {path}")
    return path


def _resolve_artifact(raw: str, *, input_root: Path) -> Path:
    path = Path(raw)
    if path.is_absolute() or path.exists():
        return path
    candidate = input_root / raw
    if candidate.exists():
        return candidate
    return path


def _read_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        _fail(f"invalid JSON in {path}: {exc}")
    if not isinstance(data, dict):
        _fail(f"expected JSON object in {path}")
    return data


def _audit_test_shards(input_root: Path) -> list[dict[str, Any]]:
    status_path = _require_file(input_root / "tests" / "shard_status.tsv", label="test shard status")
    rows: list[dict[str, Any]] = []
    with status_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            shard = str(row.get("shard", "")).strip()
            status = str(row.get("status", "")).strip()
            log_raw = str(row.get("log", "")).strip()
            if not shard:
                _fail(f"blank shard name in {status_path}")
            if status != "0":
                _fail(f"test shard failed: {shard} status={status}")
            log_path = _resolve_artifact(log_raw, input_root=input_root)
            _require_file(log_path, label=f"test shard log for {shard}")
            rows.append({"shard": shard, "status": int(status), "log": str(log_path)})
    if not rows:
        _fail(f"no test shard rows in {status_path}")
    return rows


def _audit_smoke_train(input_root: Path, *, subdir: str = "train") -> dict[str, Any]:
    train_root = input_root / subdir
    smoke_log = _require_file(train_root / "smoke_test.log", label="smoke test log")
    if "SMOKE_TEST_OK" not in smoke_log.read_text(encoding="utf-8"):
        _fail(f"SMOKE_TEST_OK marker not found in {smoke_log}")

    manifest_path = _require_file(train_root / "manifest.json", label="smoke manifest")
    manifest = _read_json(manifest_path)
    if manifest.get("status") != "ok":
        _fail(f"manifest status is not ok: {manifest_path}")

    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict):
        _fail(f"manifest artifacts object missing: {manifest_path}")

    required_artifacts = ("metrics_csv", "checkpoint", "run_metadata", "resolved_config")
    resolved: dict[str, str] = {}
    for key in required_artifacts:
        raw = artifacts.get(key)
        if not isinstance(raw, str) or not raw.strip():
            _fail(f"manifest artifact missing: {key}")
        artifact_path = _resolve_artifact(raw, input_root=input_root)
        _require_file(artifact_path, label=f"manifest artifact {key}")
        resolved[key] = str(artifact_path)

    with Path(resolved["metrics_csv"]).open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = set(reader.fieldnames or [])
    if not rows:
        _fail(f"metrics CSV has no data rows: {resolved['metrics_csv']}")
    for field in ("mean_delay_s", "mean_energy_j", "normalized_system_cost", "reward_mean"):
        if field not in fieldnames:
            _fail(f"metrics CSV missing required field {field}: {resolved['metrics_csv']}")

    metadata = _read_json(Path(resolved["run_metadata"]))
    for key in ("requested_device", "actual_device", "uses_privileged_info", "observation_mode"):
        if key not in metadata:
            _fail(f"run metadata missing {key}: {resolved['run_metadata']}")
    if metadata.get("uses_privileged_info") is not False:
        _fail("run metadata must mark uses_privileged_info=false")

    return {
        "manifest": str(manifest_path),
        "smoke_log": str(smoke_log),
        "artifacts": resolved,
        "metrics_rows": len(rows),
    }


def _audit_seed_protocol(summary_csv: Path) -> dict[str, Any]:
    if not summary_csv.is_file():
        return {"status": "not_present", "summary_csv": str(summary_csv)}
    required = {
        "train_seed",
        "eval_seed",
        "checkpoint_id",
        "checkpoint_selection_mode",
        "protocol_role",
        "experiment_contract_sha256",
    }
    test_rows: list[dict[str, str]] = []
    with summary_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = set(reader.fieldnames or [])
        missing = sorted(required - fieldnames)
        if missing:
            _fail(f"sweep summary missing seed protocol fields: {missing}")
        for row in reader:
            if str(row.get("phase", "")).strip().lower() == "test":
                test_rows.append(dict(row))
    if not test_rows:
        return {"status": "no_test_rows", "summary_csv": str(summary_csv)}

    banks: dict[str, set[int]] = {}
    for row in test_rows:
        if not str(row.get("train_seed", "")).strip():
            _fail("test row missing train_seed")
        if not str(row.get("eval_seed", "")).strip():
            _fail("test row missing eval_seed")
        if str(row.get("protocol_role", "")).strip().lower() != "test":
            _fail("test row protocol_role must be test")
        method = "::".join(
            [
                str(row.get("upper_algo", "")).strip(),
                str(row.get("lower_algo", "")).strip(),
                str(row.get("baseline", "")).strip(),
            ]
        )
        banks.setdefault(method, set()).add(int(row["eval_seed"]))
    frozen = {method: tuple(sorted(bank)) for method, bank in banks.items()}
    if len(set(frozen.values())) > 1:
        _fail(f"inconsistent test seed bank across algorithms: {frozen}")
    return {
        "status": "ok",
        "summary_csv": str(summary_csv),
        "test_rows": len(test_rows),
        "test_seed_banks": {method: list(bank) for method, bank in frozen.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit Stage smoke gate outputs before GATE_OK is created.")
    parser.add_argument("--stage", required=True)
    parser.add_argument("--input-root", required=True)
    args = parser.parse_args()

    stage = str(args.stage).strip()
    if not stage:
        _fail("--stage must be non-empty")
    input_root = Path(args.input_root)
    if not input_root.is_dir():
        _fail(f"input root does not exist: {input_root}")

    smoke_train = _audit_smoke_train(input_root)
    report = {
        "stage": stage,
        "input_root": str(input_root),
        "test_shards": _audit_test_shards(input_root),
        "smoke_train": smoke_train,
        "seed_protocol": _audit_seed_protocol(input_root / "sweep" / "sweep_summary.csv"),
    }
    if stage == "stage_09_policy_adaptivity":
        report["policy_adaptivity"] = {
            "actual_train": smoke_train,
            "controlled_stress_train": _audit_smoke_train(input_root, subdir="train_stress"),
        }
    report_path = input_root / "audit_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"AUDIT_STAGE_OUTPUTS_OK stage={stage} report={report_path}")


if __name__ == "__main__":
    main()
