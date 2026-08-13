from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

import numpy as np


ALLOCATOR_ORDER = "bandwidth_share,tx_power_ratio,cpu_share"
ENV_LOWER_ACTION_ORDER = "cpu_share,bandwidth_share,tx_power_ratio"


class LowerAllocatorCheckpointError(RuntimeError):
    """Raised when a formal same-learned lower allocator cannot load a valid checkpoint."""


class LowerAllocator(Protocol):
    name: str
    mode: str

    def allocate(
        self,
        obs: Any,
        state: Mapping[str, Any],
        upper_action: int,
        candidate_info: Mapping[int, Mapping[str, Any]],
    ) -> np.ndarray:
        """Return [bandwidth_share, tx_power_ratio, cpu_share] in [0, 1]."""
        ...


def clamp_allocator_action(values: Sequence[float]) -> np.ndarray:
    arr = np.asarray(list(values), dtype=np.float32).reshape(-1)
    if arr.size != 3:
        raise ValueError(f"lower allocator must return exactly 3 values, got shape={arr.shape}")
    return np.clip(arr, 0.0, 1.0).astype(np.float32)


def allocator_to_env_lower_action(values: Sequence[float]) -> list[float]:
    bw_share, tx_power_ratio, cpu_share = clamp_allocator_action(values).tolist()
    return [float(cpu_share), float(bw_share), float(tx_power_ratio)]


class NeutralAllocator:
    name = "neutral"
    mode = "fixed"

    def __init__(self, values: Sequence[float] | None = None) -> None:
        self.values = clamp_allocator_action(values or [1.0, 1.0, 1.0])

    def allocate(
        self,
        obs: Any,
        state: Mapping[str, Any],
        upper_action: int,
        candidate_info: Mapping[int, Mapping[str, Any]],
    ) -> np.ndarray:
        del obs, state, upper_action, candidate_info
        return self.values.copy()


class SameLearnedLowerAllocator:
    name = "same_learned"
    mode = "checkpoint"

    def __init__(
        self,
        checkpoint: str | Path | None = None,
        fallback: Sequence[float] | None = None,
        *,
        formal: bool = False,
    ) -> None:
        self.checkpoint = Path(checkpoint) if checkpoint else None
        self.fallback = NeutralAllocator(fallback)
        self.formal = bool(formal)
        self.available = bool(self.checkpoint and self.checkpoint.exists())
        self.skip_reason = "" if self.available else "checkpoint_not_provided_or_missing"
        self._loaded: Any = None
        if not self.available:
            if self.formal:
                if self.checkpoint is None:
                    raise LowerAllocatorCheckpointError("checkpoint_not_provided_or_missing for same_learned lower allocator")
                raise LowerAllocatorCheckpointError(f"same_learned lower checkpoint missing: {self.checkpoint}")
            return
        try:
            import torch

            self._loaded = torch.load(str(self.checkpoint), map_location="cpu")
        except Exception as exc:  # pragma: no cover - depends on external checkpoints
            self.available = False
            self.skip_reason = f"checkpoint_load_failed:{type(exc).__name__}"
            self._loaded = None
            if self.formal:
                raise LowerAllocatorCheckpointError(self.skip_reason) from exc
        if self.available and not self._has_valid_abi(self._loaded):
            self.available = False
            self.skip_reason = "checkpoint_missing_lower_policy_act_abi"
            if self.formal:
                raise LowerAllocatorCheckpointError(self.skip_reason)

    @staticmethod
    def _has_valid_abi(obj: Any) -> bool:
        if hasattr(obj, "act") and callable(getattr(obj, "act")):
            return True
        if isinstance(obj, Mapping):
            for key in ("lower_policy", "lower_actor", "policy"):
                candidate = obj.get(key)
                if hasattr(candidate, "act") and callable(getattr(candidate, "act")):
                    return True
        return False

    def _policy(self) -> Any:
        loaded = self._loaded
        if hasattr(loaded, "act") and callable(getattr(loaded, "act")):
            return loaded
        if isinstance(loaded, Mapping):
            for key in ("lower_policy", "lower_actor", "policy"):
                candidate = loaded.get(key)
                if hasattr(candidate, "act") and callable(getattr(candidate, "act")):
                    return candidate
        return None

    def allocate(
        self,
        obs: Any,
        state: Mapping[str, Any],
        upper_action: int,
        candidate_info: Mapping[int, Mapping[str, Any]],
    ) -> np.ndarray:
        policy = self._policy()
        if self.available and policy is not None:
            try:  # pragma: no cover - external learned agent ABI
                return clamp_allocator_action(policy.act(obs, state, upper_action, candidate_info))
            except Exception as exc:
                self.available = False
                self.skip_reason = f"checkpoint_act_failed:{type(exc).__name__}"
                if self.formal:
                    raise LowerAllocatorCheckpointError(self.skip_reason) from exc
        return self.fallback.allocate(obs, state, upper_action, candidate_info)


class OptimizedGreedyLowerAllocator:
    name = "optimized_greedy"
    mode = "grid_low"

    def __init__(
        self,
        grid: Sequence[float] | None = None,
        *,
        delay_weight: float = 1.0,
        energy_weight: float = 0.05,
        violation_weight: float = 1.0,
    ) -> None:
        self.grid = tuple(float(v) for v in (grid or [0.25, 0.5, 0.75, 1.0]))
        self.delay_weight = float(delay_weight)
        self.energy_weight = float(energy_weight)
        self.violation_weight = float(violation_weight)

    def allocate(
        self,
        obs: Any,
        state: Mapping[str, Any],
        upper_action: int,
        candidate_info: Mapping[int, Mapping[str, Any]],
    ) -> np.ndarray:
        del obs
        info = candidate_info.get(int(upper_action), {})
        best: tuple[float, tuple[float, float, float]] | None = None
        for bw_share in self.grid:
            for tx_power_ratio in self.grid:
                for cpu_share in self.grid:
                    objective = self._objective(
                        state=state,
                        info=info,
                        upper_action=int(upper_action),
                        bw_share=bw_share,
                        tx_power_ratio=tx_power_ratio,
                        cpu_share=cpu_share,
                    )
                    candidate = (objective, (bw_share, tx_power_ratio, cpu_share))
                    if best is None or candidate < best:
                        best = candidate
        assert best is not None
        return clamp_allocator_action(best[1])

    def _objective(
        self,
        *,
        state: Mapping[str, Any],
        info: Mapping[str, Any],
        upper_action: int,
        bw_share: float,
        tx_power_ratio: float,
        cpu_share: float,
    ) -> float:
        delay = max(0.0, _to_float(info.get("estimated_delay"), _to_float(info.get("estimated_cost"), 0.0)))
        energy = max(0.0, _to_float(info.get("estimated_energy_j"), 0.0))
        deadline = max(1.0e-6, _to_float(state.get("deadline_threshold", state.get("deadlineThreshold", 1.0)), 1.0))
        if upper_action == 0:
            adjusted_delay = delay / max(cpu_share, 1.0e-6)
            adjusted_energy = energy + 0.01 * cpu_share * cpu_share
        else:
            tx_delay = delay / max(bw_share, 1.0e-6)
            compute_delay = delay / max(cpu_share, 1.0e-6)
            adjusted_delay = 0.5 * tx_delay + 0.5 * compute_delay
            adjusted_energy = energy + tx_power_ratio * tx_delay + 0.01 * cpu_share * cpu_share
        violation = max(0.0, adjusted_delay - deadline)
        return float(
            self.delay_weight * adjusted_delay
            + self.energy_weight * adjusted_energy
            + self.violation_weight * violation
        )


class OracleLowerAllocator(OptimizedGreedyLowerAllocator):
    name = "oracle_grid"
    mode = "oracle_like_grid_diagnostic_only"

    def __init__(self, grid: Sequence[float] | None = None) -> None:
        super().__init__(grid or [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])


def build_lower_allocator(
    name: str,
    *,
    checkpoint: str | Path | None = None,
    neutral_values: Sequence[float] | None = None,
    formal: bool = False,
    cfg: Any | None = None,
) -> LowerAllocator:
    key = str(name or "neutral").strip().lower()
    if key == "neutral":
        return NeutralAllocator(neutral_values)
    if key in {"same_learned", "same_learned_lower"}:
        del cfg
        return SameLearnedLowerAllocator(checkpoint=checkpoint, fallback=neutral_values, formal=formal)
    if key == "optimized_greedy":
        return OptimizedGreedyLowerAllocator()
    if key in {"oracle", "oracle_grid"}:
        return OracleLowerAllocator()
    raise ValueError("unsupported lower allocator={!r}; choose neutral, same_learned, optimized_greedy, oracle_grid".format(name))


def lower_allocator_metadata(allocator: LowerAllocator) -> dict[str, Any]:
    requested = str(getattr(allocator, "name", "unknown"))
    same_loaded = bool(getattr(allocator, "available", True))
    fallback_allocator = ""
    effective = requested
    formal_claim_allowed = True
    if requested == "same_learned" and not same_loaded:
        fallback_allocator = "neutral"
        effective = "neutral"
        formal_claim_allowed = False
    return {
        "requested_allocator": requested,
        "effective_lower_allocator": effective,
        "lower_allocator_name": requested,
        "lower_allocator_mode": str(getattr(allocator, "mode", "unknown")),
        "lower_allocator_order": ALLOCATOR_ORDER,
        "env_lower_action_order": ENV_LOWER_ACTION_ORDER,
        "same_lower_available": same_loaded,
        "same_lower_skip_reason": str(getattr(allocator, "skip_reason", "")),
        "same_learned_lower_loaded": same_loaded if requested == "same_learned" else False,
        "fallback_allocator": fallback_allocator,
        "formal_claim_allowed": formal_claim_allowed,
    }


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        if value in (None, ""):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default
