from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping
import json


SMOKE_BLOCK_KEYS = (
    "outputs_are_smoke_only",
    "tiny_results_are_not_paper_results",
)

FORMAL_FALSE_KEYS = (
    "formal_claim_allowed",
    "paper_ready",
)


class FormalInputError(ValueError):
    pass


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    return text in {"1", "true", "yes", "y"}


def formal_block_reasons(payload: Mapping[str, Any]) -> list[str]:
    reasons: list[str] = []
    for key in SMOKE_BLOCK_KEYS:
        if _as_bool(payload.get(key, False)):
            reasons.append(f"{key}=true")
    for key in FORMAL_FALSE_KEYS:
        if key in payload and not _as_bool(payload.get(key, False)):
            reasons.append(f"{key}=false")
    rows = payload.get("rows")
    if isinstance(rows, list):
        for idx, row in enumerate(rows):
            if not isinstance(row, Mapping):
                continue
            for key in SMOKE_BLOCK_KEYS:
                if _as_bool(row.get(key, False)):
                    reasons.append(f"rows[{idx}].{key}=true")
            for key in FORMAL_FALSE_KEYS:
                if key in row and not _as_bool(row.get(key, False)):
                    reasons.append(f"rows[{idx}].{key}=false")
    return reasons


def ensure_diagnostic_output_path(path: str | Path) -> Path:
    out = Path(path)
    text = out.name.lower()
    if "diagnostic" not in text:
        raise FormalInputError(
            "diagnostic inputs require an output path containing 'diagnostic' so they cannot overwrite formal paper outputs"
        )
    return out


def validate_formal_result_payload(
    payload: Mapping[str, Any],
    *,
    source: str | Path,
    allow_diagnostic_inputs: bool = False,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    reasons = formal_block_reasons(payload)
    if reasons and not allow_diagnostic_inputs:
        raise FormalInputError(f"formal collector rejected diagnostic/smoke input {source}: {', '.join(reasons)}")
    diagnostic = bool(reasons)
    if diagnostic and output_path is not None:
        ensure_diagnostic_output_path(output_path)
    return {
        "source": str(source),
        "diagnostic_input": diagnostic,
        "diagnostic_reasons": reasons,
        "allow_diagnostic_inputs": bool(allow_diagnostic_inputs),
    }


def validate_summary_tree(
    root: str | Path,
    *,
    allow_diagnostic_inputs: bool = False,
    output_path: str | Path | None = None,
) -> list[dict[str, Any]]:
    root_path = Path(root)
    summaries = []
    for pattern in ("summary.json", "summary_compact.json", "strong_baseline_summary.json", "stress_summary.json"):
        summaries.extend(root_path.rglob(pattern))
    guards: list[dict[str, Any]] = []
    for path in sorted(set(summaries)):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        guards.append(
            validate_formal_result_payload(
                payload,
                source=path,
                allow_diagnostic_inputs=allow_diagnostic_inputs,
                output_path=output_path if allow_diagnostic_inputs else None,
            )
        )
    return guards
