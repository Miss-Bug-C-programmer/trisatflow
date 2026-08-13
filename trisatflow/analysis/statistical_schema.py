from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping


REQUIRED_COLUMNS = ("method", "metric", "value", "train_seed", "checkpoint_id")
OPTIONAL_COLUMNS = ("test_seed", "online_seed", "scenario_id", "split", "statistical_unit", "source_file")


class StatisticalSchemaError(ValueError):
    """Raised when statistical input would make the independence unit ambiguous."""


@dataclass(frozen=True)
class StatisticalRecord:
    method: str
    metric: str
    value: float
    train_seed: str
    checkpoint_id: str
    test_seed: str = ""
    online_seed: str = ""
    scenario_id: str = ""
    split: str = "offline"
    statistical_unit: str = "train_seed_checkpoint"
    source_file: str = ""

    @property
    def cluster_id(self) -> str:
        return f"{self.train_seed}::{self.checkpoint_id}"


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _as_float(value: Any, *, column: str) -> float:
    try:
        x = float(value)
    except (TypeError, ValueError) as exc:
        raise StatisticalSchemaError(f"column '{column}' must be numeric, got {value!r}") from exc
    if not math.isfinite(x):
        raise StatisticalSchemaError(f"column '{column}' must be finite, got {value!r}")
    return x


def infer_statistical_unit(row: Mapping[str, Any]) -> str:
    explicit = _as_text(row.get("statistical_unit"))
    if explicit:
        return explicit
    split = _as_text(row.get("split")).lower()
    if _as_text(row.get("online_seed")) or split in {"online", "satedgesim", "replay"}:
        return "train_seed_checkpoint_cluster"
    return "train_seed_checkpoint"


def normalize_record(row: Mapping[str, Any], *, row_index: int = 0, source_file: str = "") -> StatisticalRecord:
    missing = [name for name in REQUIRED_COLUMNS if not _as_text(row.get(name))]
    if missing:
        raise StatisticalSchemaError(
            "statistical input is missing required independence fields "
            f"{missing} at row {row_index}; provide method, metric, value, train_seed, and checkpoint_id. "
            "Do not use test_seed or online_seed as independent samples."
        )
    split = _as_text(row.get("split")) or ("online" if _as_text(row.get("online_seed")) else "offline")
    return StatisticalRecord(
        method=_as_text(row.get("method")),
        metric=_as_text(row.get("metric")),
        value=_as_float(row.get("value"), column="value"),
        train_seed=_as_text(row.get("train_seed")),
        checkpoint_id=_as_text(row.get("checkpoint_id")),
        test_seed=_as_text(row.get("test_seed")),
        online_seed=_as_text(row.get("online_seed")),
        scenario_id=_as_text(row.get("scenario_id")),
        split=split,
        statistical_unit=infer_statistical_unit({**dict(row), "split": split}),
        source_file=_as_text(row.get("source_file")) or source_file,
    )


def normalize_records(rows: Iterable[Mapping[str, Any]], *, source_file: str = "") -> List[StatisticalRecord]:
    return [normalize_record(row, row_index=i + 1, source_file=source_file) for i, row in enumerate(rows)]


def read_records(path: Path) -> List[StatisticalRecord]:
    if not path.exists():
        raise StatisticalSchemaError(f"statistical input file does not exist: {path}")
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            rows = payload.get("rows", payload.get("records", []))
        else:
            rows = payload
        if not isinstance(rows, list):
            raise StatisticalSchemaError(f"JSON input must be a list of records or contain rows/records: {path}")
        return normalize_records(rows, source_file=str(path))
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    return normalize_records(rows, source_file=str(path))


def write_standard_csv(path: Path, records: Iterable[StatisticalRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(REQUIRED_COLUMNS + OPTIONAL_COLUMNS)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow({name: getattr(record, name) for name in fieldnames})
