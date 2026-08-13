from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any


QUEUE_KEYS = ("queue", "leo_queue", "physical_queue_cycles")
VIRTUAL_QUEUE_KEYS = ("virtual_delay_queue", "mean_virtual_deadline_queue")
DRIFT_KEYS = ("lyapunov_drift", "drift")
PENALTY_KEYS = ("penalty", "system_cost", "normalized_system_cost", "immediate_cost")
DRIFT_PLUS_PENALTY_KEYS = ("drift_plus_penalty", "lyapunov_reward_shaping_cost")
OVERFLOW_KEYS = ("finite_buffer_overflow_count", "overflow_count")
OVERFLOW_RISK_KEYS = ("overflow_risk",)


def _flatten(value: Any) -> list[float]:
    if value is None:
        return []
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, Mapping):
        out: list[float] = []
        for item in value.values():
            out.extend(_flatten(item))
        return out
    if isinstance(value, (str, bytes)):
        return []
    if isinstance(value, Iterable):
        out = []
        for item in value:
            out.extend(_flatten(item))
        return out
    try:
        return [float(value)]
    except (TypeError, ValueError):
        return []


def _collect(records: Iterable[Mapping[str, Any]], keys: tuple[str, ...]) -> list[float]:
    values: list[float] = []
    for record in records:
        for key in keys:
            if key in record:
                values.extend(_flatten(record.get(key)))
                break
    return values


def _mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = int(round((len(ordered) - 1) * pct))
    return float(ordered[max(0, min(len(ordered) - 1, index))])


def _pairwise_sum(left: list[float], right: list[float]) -> list[float]:
    if not left:
        return list(right)
    if not right:
        return list(left)
    n = min(len(left), len(right))
    return [float(left[i] + right[i]) for i in range(n)]


def compute_lyapunov_diagnostics(
    episode_records: Iterable[Mapping[str, Any]],
    *,
    queue_cap_mode: str = "finite_buffer",
    queue_cap: float | None = None,
) -> dict[str, Any]:
    """Summarize queue/drift diagnostics without claiming Lyapunov stability.

    The returned fields are empirical diagnostics for finite episodes. They are
    intentionally named as reward-shaping diagnostics, not theorem outcomes.
    """

    records = list(episode_records)
    queues = _collect(records, QUEUE_KEYS)
    virtual_queues = _collect(records, VIRTUAL_QUEUE_KEYS)
    drifts = _collect(records, DRIFT_KEYS)
    penalties = _collect(records, PENALTY_KEYS)
    drift_plus_penalty = _collect(records, DRIFT_PLUS_PENALTY_KEYS)
    if not drift_plus_penalty:
        drift_plus_penalty = _pairwise_sum(drifts, penalties)
    overflow_counts = _collect(records, OVERFLOW_KEYS)
    overflow_risks = _collect(records, OVERFLOW_RISK_KEYS)

    mode = str(queue_cap_mode or "finite_buffer").strip().lower()
    if mode not in {"finite_buffer", "unbounded_eval"}:
        mode = "finite_buffer"
    positive_drifts = [item for item in drifts if item > 0.0]
    derived_overflow_count = 0
    derived_overflow_risk = 0.0
    if queue_cap is not None and queues:
        cap = float(queue_cap)
        derived_flags = [1.0 if item > cap else 0.0 for item in queues]
        derived_overflow_count = int(sum(derived_flags))
        derived_overflow_risk = _mean(derived_flags)
    explicit_overflow_count = int(round(sum(max(0.0, item) for item in overflow_counts))) if overflow_counts else 0

    return {
        "mean_queue": _mean(queues),
        "p95_queue": _percentile(queues, 0.95),
        "p99_queue": _percentile(queues, 0.99),
        "max_queue": float(max(queues)) if queues else 0.0,
        "mean_virtual_deadline_queue": _mean(virtual_queues),
        "positive_drift_ratio": float(len(positive_drifts) / len(drifts)) if drifts else 0.0,
        "mean_drift": _mean(drifts),
        "mean_drift_plus_penalty": _mean(drift_plus_penalty),
        "finite_buffer_overflow_count": explicit_overflow_count if overflow_counts else derived_overflow_count,
        "overflow_risk": _mean(overflow_risks) if overflow_risks else derived_overflow_risk,
        "queue_cap_mode": mode,
        "queue_cap": None if queue_cap is None else float(queue_cap),
        "queue_stability_claim_allowed": False,
        "lyapunov_semantics": "reward_shaping_no_stability_theorem",
    }
