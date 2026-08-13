from __future__ import annotations

from dataclasses import dataclass, field
import random
import warnings
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Mapping

import torch

from trisatflow.baselines.registry import (
    ACTION_NAMES,
    FORMAL_BASELINE_NAMES,
    LEGACY_BASELINE_ALIASES,
    BaselinePolicy,
    build_baseline_policy,
)

if TYPE_CHECKING:
    from trisatflow.envs import GeoLeoGroundEnv


@dataclass
class OfflineDecisionBatch:
    upper_action: torch.Tensor
    lower_action: torch.Tensor
    decision_info: List[Dict[str, Any]]


@dataclass
class OfflineDecisionStats:
    decision_count: int = 0
    fallback_count: int = 0
    invalid_attempt_count: int = 0
    requested_counts: List[int] = field(default_factory=lambda: [0, 0, 0, 0])
    selected_counts: List[int] = field(default_factory=lambda: [0, 0, 0, 0])

    def snapshot(self) -> Dict[str, Any]:
        return {
            "decision_count": int(self.decision_count),
            "fallback_count": int(self.fallback_count),
            "invalid_attempt_count": int(self.invalid_attempt_count),
            "requested_counts": list(self.requested_counts),
            "selected_counts": list(self.selected_counts),
        }


def normalize_baseline_name(name: str, *, warn: bool = True) -> str:
    key = str(name or "").strip().lower()
    if key in LEGACY_BASELINE_ALIASES:
        canonical = LEGACY_BASELINE_ALIASES[key]
        if warn:
            warnings.warn(
                f"baseline name {key!r} is deprecated; use {canonical!r}",
                UserWarning,
                stacklevel=2,
            )
        return canonical
    return key


def offline_baseline_registry(*, include_legacy_aliases: bool = True) -> Dict[str, BaselinePolicy]:
    names = list(FORMAL_BASELINE_NAMES)
    if include_legacy_aliases:
        names.extend(LEGACY_BASELINE_ALIASES)
    return {name: build_baseline_policy(normalize_baseline_name(name, warn=False)) for name in names}


def build_offline_baseline_policy(name: str) -> BaselinePolicy:
    return build_baseline_policy(normalize_baseline_name(name, warn=True))


class OfflineBaselineAdapter:
    def __init__(self, policy: BaselinePolicy, *, rng: random.Random | None = None) -> None:
        self.policy = policy
        self.rng = rng or random.Random()
        self.stats = OfflineDecisionStats()

    def select_actions(self, env: "GeoLeoGroundEnv") -> OfflineDecisionBatch:
        contexts = env.baseline_contexts()
        selected: List[int] = []
        lower_rows: List[List[float]] = []
        decisions: List[Dict[str, Any]] = []
        for agent_idx, ctx in enumerate(contexts):
            mask = [bool(item) for item in ctx["mask"]]  # type: ignore[index]
            decision = self.policy.select_action(
                obs=ctx["obs"],
                state=ctx["state"],  # type: ignore[arg-type]
                mask=mask,
                candidate_info=ctx["candidate_info"],  # type: ignore[arg-type]
                rng=self.rng,
            )
            info = dict(decision.get("decision_info") or {})
            requested_action = int(info.get("requested_action", info.get("target_action", decision.get("upper_action", 0))))
            selected_action = int(decision.get("upper_action", info.get("selected_action", 0)))
            if not (0 <= selected_action < len(mask) and bool(mask[selected_action])):
                raise RuntimeError(
                    f"baseline {getattr(self.policy, 'name', '<unknown>')} selected masked action "
                    f"{selected_action} for agent {agent_idx}; mask={mask}"
                )
            invalid_attempt = not (0 <= requested_action < len(mask) and bool(mask[requested_action]))
            fallback_used = bool(info.get("fallback_used", False)) or bool(invalid_attempt)
            self._record(requested_action, selected_action, fallback_used=fallback_used, invalid_attempt=invalid_attempt)
            info.update(
                {
                    "agent_index": int(agent_idx),
                    "requested_action": int(requested_action),
                    "requested_action_name": ACTION_NAMES[max(0, min(3, requested_action))],
                    "selected_action": int(selected_action),
                    "selected_action_name": ACTION_NAMES[selected_action],
                    "fallback_used": bool(fallback_used),
                    "invalid_attempt": bool(invalid_attempt),
                    "mask": [1 if item else 0 for item in mask],
                }
            )
            selected.append(selected_action)
            raw_lower = decision.get("lower_action", [1.0, 1.0, 1.0])
            if not isinstance(raw_lower, list) or len(raw_lower) < env.LOWER_ACTION_DIM:
                raw_lower = [1.0, 1.0, 1.0]
            lower_rows.append([float(raw_lower[idx]) for idx in range(env.LOWER_ACTION_DIM)])
            decisions.append(info)
        upper = torch.tensor(selected, dtype=torch.long, device=env.device)
        lower = torch.tensor(lower_rows, dtype=torch.float32, device=env.device).clamp(0.0, 1.0)
        return OfflineDecisionBatch(upper_action=upper, lower_action=lower, decision_info=decisions)

    def _record(self, requested: int, selected: int, *, fallback_used: bool, invalid_attempt: bool) -> None:
        self.stats.decision_count += 1
        if 0 <= requested < 4:
            self.stats.requested_counts[requested] += 1
        if 0 <= selected < 4:
            self.stats.selected_counts[selected] += 1
        if fallback_used:
            self.stats.fallback_count += 1
        if invalid_attempt:
            self.stats.invalid_attempt_count += 1


def stats_delta(after: Mapping[str, Any], before: Mapping[str, Any]) -> Dict[str, Any]:
    requested_after = list(after.get("requested_counts") or [0, 0, 0, 0])
    requested_before = list(before.get("requested_counts") or [0, 0, 0, 0])
    selected_after = list(after.get("selected_counts") or [0, 0, 0, 0])
    selected_before = list(before.get("selected_counts") or [0, 0, 0, 0])
    return {
        "decision_count": int(after.get("decision_count", 0)) - int(before.get("decision_count", 0)),
        "fallback_count": int(after.get("fallback_count", 0)) - int(before.get("fallback_count", 0)),
        "invalid_attempt_count": int(after.get("invalid_attempt_count", 0)) - int(before.get("invalid_attempt_count", 0)),
        "requested_counts": [int(a) - int(b) for a, b in zip(requested_after, requested_before)],
        "selected_counts": [int(a) - int(b) for a, b in zip(selected_after, selected_before)],
    }


def ratio_fields(prefix: str, counts: Iterable[int]) -> Dict[str, float]:
    values = [int(v) for v in counts]
    total = float(max(1, sum(values)))
    return {f"{prefix}_{name}_ratio": float(values[idx] / total) for idx, name in enumerate(ACTION_NAMES)}
