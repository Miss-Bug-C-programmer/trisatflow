from __future__ import annotations

import csv
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence

import torch

LOGGER = logging.getLogger(__name__)


@dataclass
class TraceTopologySnapshot:
    """Per-step topology slice aligned with SatEdgeSim abstract action semantics.

    Each row in the trace describes one decision step for one source LEO and may
    explicitly provide:
      * `abstract_action_mask = [local, neighbor, geo, ground]`
      * visibility/rate summaries for the four tiers
      * candidate counts / distance / queue / delay summaries

    `provided[leo]` indicates whether the trace contains a concrete row for that
    `(step, leo_id)` pair. Missing rows deliberately fall back to the analytic
    topology inside ``GeoLeoGroundEnv`` so training can still proceed while the
    provider keeps counters about trace coverage quality.
    """

    local_visible: torch.Tensor
    neighbor_visible: torch.Tensor
    geo_visible: torch.Tensor
    ground_visible: torch.Tensor
    local_rate: torch.Tensor
    neighbor_rate: torch.Tensor
    geo_rate: torch.Tensor
    ground_rate: torch.Tensor
    local_delay: torch.Tensor
    neighbor_delay: torch.Tensor
    geo_delay: torch.Tensor
    ground_delay: torch.Tensor
    local_prop_delay: torch.Tensor
    neighbor_prop_delay: torch.Tensor
    geo_prop_delay: torch.Tensor
    ground_prop_delay: torch.Tensor
    local_tx_delay: torch.Tensor
    neighbor_tx_delay: torch.Tensor
    geo_tx_delay: torch.Tensor
    ground_tx_delay: torch.Tensor
    local_compute_delay: torch.Tensor
    neighbor_compute_delay: torch.Tensor
    geo_compute_delay: torch.Tensor
    ground_compute_delay: torch.Tensor
    local_queue_delay: torch.Tensor
    neighbor_queue_delay: torch.Tensor
    geo_queue_delay: torch.Tensor
    ground_queue_delay: torch.Tensor
    local_queue: torch.Tensor
    neighbor_queue: torch.Tensor
    geo_queue: torch.Tensor
    ground_queue: torch.Tensor
    abstract_action_mask: torch.Tensor
    abstract_action_mask_visible: torch.Tensor
    abstract_action_mask_mobility_safe: torch.Tensor
    abstract_action_mask_completion_safe: torch.Tensor
    abstract_action_mask_final: torch.Tensor
    mask_field_presence: torch.Tensor
    delay_semantic_code: torch.Tensor
    local_mobility_risk: torch.Tensor
    neighbor_mobility_risk: torch.Tensor
    geo_mobility_risk: torch.Tensor
    ground_mobility_risk: torch.Tensor
    local_link_lifetime: torch.Tensor
    neighbor_link_lifetime: torch.Tensor
    geo_link_lifetime: torch.Tensor
    ground_link_lifetime: torch.Tensor
    local_link_margin_to_completion: torch.Tensor
    neighbor_link_margin_to_completion: torch.Tensor
    geo_link_margin_to_completion: torch.Tensor
    ground_link_margin_to_completion: torch.Tensor
    local_handover_required: torch.Tensor
    neighbor_handover_required: torch.Tensor
    geo_handover_required: torch.Tensor
    ground_handover_required: torch.Tensor
    local_candidate_count: torch.Tensor
    neighbor_candidate_count: torch.Tensor
    geo_candidate_count: torch.Tensor
    ground_candidate_count: torch.Tensor
    provided: torch.Tensor


class TraceTopologyMissingError(RuntimeError):
    """Raised when strict trace mode cannot provide a requested (step, leo_id)."""


class TopologyTraceProvider:
    """Offline trace loader for SatEdgeSim-aligned topology supervision."""

    def __init__(self, path: str | Path, *, n_leo: int, device: torch.device, repeat: bool = True, strict: bool = False):
        self.path = Path(path)
        self.n_leo = int(n_leo)
        self.device = device
        self.repeat = bool(repeat)
        self.strict = bool(strict)
        if not self.path.exists():
            raise FileNotFoundError(f"topology_trace_path does not exist: {self.path}")
        self._by_step = self._load(self.path)
        if not self._by_step:
            raise ValueError(f"topology trace has no usable rows: {self.path}")
        self.steps = sorted(self._by_step)
        self.max_step = max(self.steps)
        self.num_rows = sum(len(by_leo) for by_leo in self._by_step.values())
        self._warnings_emitted = 0
        # Snapshots are immutable trace slices.  Episodes repeatedly revisit the
        # same horizon, so rebuilding dozens of tensors for every step causes a
        # large avoidable slowdown, especially on CUDA.
        self._snapshot_cache: Dict[int, tuple[TraceTopologySnapshot, int, int]] = {}
        self.reset_counters()

    def reset_counters(self) -> None:
        self.missing_leo_rows = 0
        self.missing_step_queries = 0
        self.trace_pair_queries = 0
        self.trace_pair_hits = 0
        self.trace_pair_missing = 0
        self.trace_fallback_count = 0
        self.missing_mask_field_count = 0
        self.missing_visible_mask_field_count = 0
        self.missing_completion_mask_field_count = 0
        self.missing_mobility_mask_field_count = 0
        self.missing_final_mask_field_count = 0
        self.delay_semantic_counts = {key: 0 for key in _DELAY_SEMANTIC_TO_CODE}

    def snapshot(self, step: int) -> TraceTopologySnapshot:
        requested_step = int(step)
        key, used_fallback_step = self._resolve_step(requested_step)
        cached = self._snapshot_cache.get(key)
        if cached is not None:
            snapshot, hits, missing = cached
            self.trace_pair_queries += self.n_leo
            self.trace_pair_hits += hits
            self.trace_pair_missing += missing
            if missing > 0:
                self.missing_leo_rows += missing
                self.trace_fallback_count += missing
            if used_fallback_step and not self.repeat:
                self.trace_fallback_count += self.n_leo
            return snapshot
        rows = self._by_step.get(key, {})
        if not rows:
            self.missing_step_queries += 1
            self.trace_pair_queries += self.n_leo
            self.trace_pair_missing += self.n_leo
            if self.strict:
                raise TraceTopologyMissingError(
                    f"strict topology trace missing all rows for requested_step={requested_step} resolved_step={key} path={self.path}"
                )
            self.trace_fallback_count += self.n_leo
            self._warn_once(f"topology trace has no rows for resolved step={key}; analytic fallback will be used")

        zeros = torch.zeros(self.n_leo, dtype=torch.float32, device=self.device)
        local_visible = zeros.clone()
        neighbor_visible = zeros.clone()
        geo_visible = zeros.clone()
        ground_visible = zeros.clone()
        local_rate = zeros.clone()
        neighbor_rate = zeros.clone()
        geo_rate = zeros.clone()
        ground_rate = zeros.clone()
        local_delay = zeros.clone()
        neighbor_delay = zeros.clone()
        geo_delay = zeros.clone()
        ground_delay = zeros.clone()
        local_prop_delay = zeros.clone()
        neighbor_prop_delay = zeros.clone()
        geo_prop_delay = zeros.clone()
        ground_prop_delay = zeros.clone()
        local_tx_delay = zeros.clone()
        neighbor_tx_delay = zeros.clone()
        geo_tx_delay = zeros.clone()
        ground_tx_delay = zeros.clone()
        local_compute_delay = zeros.clone()
        neighbor_compute_delay = zeros.clone()
        geo_compute_delay = zeros.clone()
        ground_compute_delay = zeros.clone()
        local_queue_delay = zeros.clone()
        neighbor_queue_delay = zeros.clone()
        geo_queue_delay = zeros.clone()
        ground_queue_delay = zeros.clone()
        local_queue = zeros.clone()
        neighbor_queue = zeros.clone()
        geo_queue = zeros.clone()
        ground_queue = zeros.clone()
        local_candidate_count = zeros.clone()
        neighbor_candidate_count = zeros.clone()
        geo_candidate_count = zeros.clone()
        ground_candidate_count = zeros.clone()
        abstract_action_mask = torch.zeros((self.n_leo, 4), dtype=torch.bool, device=self.device)
        abstract_action_mask_visible = torch.zeros((self.n_leo, 4), dtype=torch.bool, device=self.device)
        abstract_action_mask_mobility_safe = torch.zeros((self.n_leo, 4), dtype=torch.bool, device=self.device)
        abstract_action_mask_completion_safe = torch.zeros((self.n_leo, 4), dtype=torch.bool, device=self.device)
        abstract_action_mask_final = torch.zeros((self.n_leo, 4), dtype=torch.bool, device=self.device)
        mask_field_presence = torch.zeros((self.n_leo, 4), dtype=torch.bool, device=self.device)
        delay_semantic_code = torch.full(
            (self.n_leo,),
            float(_DELAY_SEMANTIC_TO_CODE["legacy_unknown"]),
            dtype=torch.float32,
            device=self.device,
        )
        local_mobility_risk = zeros.clone()
        neighbor_mobility_risk = zeros.clone()
        geo_mobility_risk = zeros.clone()
        ground_mobility_risk = zeros.clone()
        local_link_lifetime = zeros.clone()
        neighbor_link_lifetime = zeros.clone()
        geo_link_lifetime = zeros.clone()
        ground_link_lifetime = zeros.clone()
        local_link_margin_to_completion = zeros.clone()
        neighbor_link_margin_to_completion = zeros.clone()
        geo_link_margin_to_completion = zeros.clone()
        ground_link_margin_to_completion = zeros.clone()
        local_handover_required = zeros.clone()
        neighbor_handover_required = zeros.clone()
        geo_handover_required = zeros.clone()
        ground_handover_required = zeros.clone()
        provided = torch.zeros(self.n_leo, dtype=torch.bool, device=self.device)

        if used_fallback_step and not self.repeat:
            self.trace_fallback_count += self.n_leo

        hits = 0
        missing = 0
        for leo in range(self.n_leo):
            row = rows.get(leo)
            if row is None:
                missing += 1
                continue
            provided[leo] = True
            presence = _row_mask_field_presence(row)
            mask_field_presence[leo] = torch.tensor(presence, dtype=torch.bool, device=self.device)
            missing_layers = [name for name, present in zip(_MASK_FIELD_ORDER, presence) if not present]
            if missing_layers:
                self.missing_mask_field_count += len(missing_layers)
                self.missing_visible_mask_field_count += int("visible" in missing_layers)
                self.missing_completion_mask_field_count += int("completion_safe" in missing_layers)
                self.missing_mobility_mask_field_count += int("mobility_safe" in missing_layers)
                self.missing_final_mask_field_count += int("final" in missing_layers)
            visible_mask = _row_mask(row, "visible", default=_row_abstract_mask(row))
            mobility_mask = _row_mask(row, "mobility_safe", default=visible_mask)
            completion_mask = _row_mask(row, "completion_safe", default=mobility_mask)
            action_mode = str(row.get("action_mask_mode", row.get("actionMaskMode", "visible_only"))).strip().lower()
            if action_mode in {"full", "full_mask", "completion_safe"}:
                mask = completion_mask
            elif action_mode in {"mobility_safe", "mobility_risk"}:
                mask = mobility_mask
            elif action_mode in {"none", "no_mask"}:
                mask = [True, True, True, True]
            else:
                mask = visible_mask
            mask = _row_mask(row, "final", default=mask)
            if not any(mask):
                mask = visible_mask if any(visible_mask) else [True, False, False, False]
            abstract_action_mask[leo] = torch.tensor(mask, dtype=torch.bool, device=self.device)
            abstract_action_mask_visible[leo] = torch.tensor(visible_mask, dtype=torch.bool, device=self.device)
            abstract_action_mask_mobility_safe[leo] = torch.tensor(mobility_mask, dtype=torch.bool, device=self.device)
            abstract_action_mask_completion_safe[leo] = torch.tensor(completion_mask, dtype=torch.bool, device=self.device)
            abstract_action_mask_final[leo] = torch.tensor(mask, dtype=torch.bool, device=self.device)
            semantic = _row_delay_semantic(row)
            self.delay_semantic_counts[semantic] += 1
            delay_semantic_code[leo] = float(_DELAY_SEMANTIC_TO_CODE[semantic])
            local_visible[leo] = float(visible_mask[0])
            neighbor_visible[leo] = float(visible_mask[1])
            geo_visible[leo] = float(visible_mask[2])
            ground_visible[leo] = float(visible_mask[3])
            local_rate[leo] = max(0.0, _to_float(row.get("local_rate"), 0.0))
            neighbor_rate[leo] = max(0.0, _to_float(row.get("neighbor_rate"), 0.0))
            geo_rate[leo] = max(0.0, _to_float(row.get("geo_rate"), 0.0))
            ground_rate[leo] = max(0.0, _to_float(row.get("ground_rate"), 0.0))
            local_delay[leo] = max(0.0, _to_float(row.get("local_delay", row.get("local_best_delay")), 0.0))
            neighbor_delay[leo] = max(0.0, _to_float(row.get("neighbor_delay", row.get("neighbor_best_delay")), 0.0))
            geo_delay[leo] = max(0.0, _to_float(row.get("geo_delay", row.get("geo_best_delay")), 0.0))
            ground_delay[leo] = max(0.0, _to_float(row.get("ground_delay", row.get("ground_best_delay")), 0.0))
            local_prop_delay[leo] = max(0.0, _to_float(row.get("local_prop_delay"), 0.0))
            neighbor_prop_delay[leo] = max(0.0, _to_float(row.get("neighbor_prop_delay"), 0.0))
            geo_prop_delay[leo] = max(0.0, _to_float(row.get("geo_prop_delay"), 0.0))
            ground_prop_delay[leo] = max(0.0, _to_float(row.get("ground_prop_delay"), 0.0))
            local_tx_delay[leo] = max(0.0, _to_float(row.get("local_tx_delay"), 0.0))
            neighbor_tx_delay[leo] = max(0.0, _to_float(row.get("neighbor_tx_delay"), 0.0))
            geo_tx_delay[leo] = max(0.0, _to_float(row.get("geo_tx_delay"), 0.0))
            ground_tx_delay[leo] = max(0.0, _to_float(row.get("ground_tx_delay"), 0.0))
            local_compute_delay[leo] = max(0.0, _to_float(row.get("local_compute_delay"), 0.0))
            neighbor_compute_delay[leo] = max(0.0, _to_float(row.get("neighbor_compute_delay"), 0.0))
            geo_compute_delay[leo] = max(0.0, _to_float(row.get("geo_compute_delay"), 0.0))
            ground_compute_delay[leo] = max(0.0, _to_float(row.get("ground_compute_delay"), 0.0))
            local_queue_delay[leo] = max(0.0, _to_float(row.get("local_queue_delay"), 0.0))
            neighbor_queue_delay[leo] = max(0.0, _to_float(row.get("neighbor_queue_delay"), 0.0))
            geo_queue_delay[leo] = max(0.0, _to_float(row.get("geo_queue_delay"), 0.0))
            ground_queue_delay[leo] = max(0.0, _to_float(row.get("ground_queue_delay"), 0.0))
            local_queue[leo] = max(0.0, _to_float(row.get("local_queue", row.get("local_best_queue")), 0.0))
            neighbor_queue[leo] = max(0.0, _to_float(row.get("neighbor_queue", row.get("neighbor_best_queue")), 0.0))
            geo_queue[leo] = max(0.0, _to_float(row.get("geo_queue", row.get("geo_best_queue")), 0.0))
            ground_queue[leo] = max(0.0, _to_float(row.get("ground_queue", row.get("ground_best_queue")), 0.0))
            local_candidate_count[leo] = max(0.0, _to_float(row.get("local_candidate_count"), 0.0))
            neighbor_candidate_count[leo] = max(0.0, _to_float(row.get("neighbor_candidate_count"), 0.0))
            geo_candidate_count[leo] = max(0.0, _to_float(row.get("geo_candidate_count"), 0.0))
            ground_candidate_count[leo] = max(0.0, _to_float(row.get("ground_candidate_count"), 0.0))
            local_mobility_risk[leo] = _tier_float(row, "local", "mobility_risk", default=0.0)
            neighbor_mobility_risk[leo] = _tier_float(row, "neighbor", "mobility_risk", default=1.0 if not mask[1] else 0.0)
            geo_mobility_risk[leo] = _tier_float(row, "geo", "mobility_risk", default=1.0 if not mask[2] else 0.0)
            ground_mobility_risk[leo] = _tier_float(row, "ground", "mobility_risk", default=1.0 if not mask[3] else 0.0)
            local_link_lifetime[leo] = _tier_float(row, "local", "link_lifetime_sec", default=0.0)
            neighbor_link_lifetime[leo] = _tier_float(row, "neighbor", "link_lifetime_sec", default=0.0)
            geo_link_lifetime[leo] = _tier_float(row, "geo", "link_lifetime_sec", default=0.0)
            ground_link_lifetime[leo] = _tier_float(row, "ground", "link_lifetime_sec", default=0.0)
            local_link_margin_to_completion[leo] = _tier_float(row, "local", "link_survival_margin_to_completion_sec", default=0.0)
            neighbor_link_margin_to_completion[leo] = _tier_float(row, "neighbor", "link_survival_margin_to_completion_sec", default=0.0)
            geo_link_margin_to_completion[leo] = _tier_float(row, "geo", "link_survival_margin_to_completion_sec", default=0.0)
            ground_link_margin_to_completion[leo] = _tier_float(row, "ground", "link_survival_margin_to_completion_sec", default=0.0)
            local_handover_required[leo] = 1.0 if _tier_bool(row, "local", "handover_required", default=False) else 0.0
            neighbor_handover_required[leo] = 1.0 if _tier_bool(row, "neighbor", "handover_required", default=False) else 0.0
            geo_handover_required[leo] = 1.0 if _tier_bool(row, "geo", "handover_required", default=False) else 0.0
            ground_handover_required[leo] = 1.0 if _tier_bool(row, "ground", "handover_required", default=False) else 0.0
            hits += 1

        self.trace_pair_queries += self.n_leo
        self.trace_pair_hits += hits
        self.trace_pair_missing += missing
        if missing > 0:
            self.missing_leo_rows += missing
            if self.strict:
                missing_leos = [str(leo) for leo in range(self.n_leo) if not bool(provided[leo].item())]
                raise TraceTopologyMissingError(
                    f"strict topology trace missing rows for requested_step={requested_step} resolved_step={key} "
                    f"leo_ids={','.join(missing_leos)} path={self.path}"
                )
            self.trace_fallback_count += missing
            self._warn_once(
                f"topology trace step={key} is missing {missing}/{self.n_leo} leo rows; "
                "GeoLeoGroundEnv will fall back to analytic topology for those agents"
            )

        snapshot = TraceTopologySnapshot(
            local_visible=local_visible.bool(),
            neighbor_visible=neighbor_visible.bool(),
            geo_visible=geo_visible.bool(),
            ground_visible=ground_visible.bool(),
            local_rate=local_rate,
            neighbor_rate=neighbor_rate,
            geo_rate=geo_rate,
            ground_rate=ground_rate,
            local_delay=local_delay,
            neighbor_delay=neighbor_delay,
            geo_delay=geo_delay,
            ground_delay=ground_delay,
            local_prop_delay=local_prop_delay,
            neighbor_prop_delay=neighbor_prop_delay,
            geo_prop_delay=geo_prop_delay,
            ground_prop_delay=ground_prop_delay,
            local_tx_delay=local_tx_delay,
            neighbor_tx_delay=neighbor_tx_delay,
            geo_tx_delay=geo_tx_delay,
            ground_tx_delay=ground_tx_delay,
            local_compute_delay=local_compute_delay,
            neighbor_compute_delay=neighbor_compute_delay,
            geo_compute_delay=geo_compute_delay,
            ground_compute_delay=ground_compute_delay,
            local_queue_delay=local_queue_delay,
            neighbor_queue_delay=neighbor_queue_delay,
            geo_queue_delay=geo_queue_delay,
            ground_queue_delay=ground_queue_delay,
            local_queue=local_queue,
            neighbor_queue=neighbor_queue,
            geo_queue=geo_queue,
            ground_queue=ground_queue,
            abstract_action_mask=abstract_action_mask,
            abstract_action_mask_visible=abstract_action_mask_visible,
            abstract_action_mask_mobility_safe=abstract_action_mask_mobility_safe,
            abstract_action_mask_completion_safe=abstract_action_mask_completion_safe,
            abstract_action_mask_final=abstract_action_mask_final,
            mask_field_presence=mask_field_presence,
            delay_semantic_code=delay_semantic_code,
            local_mobility_risk=local_mobility_risk,
            neighbor_mobility_risk=neighbor_mobility_risk,
            geo_mobility_risk=geo_mobility_risk,
            ground_mobility_risk=ground_mobility_risk,
            local_link_lifetime=local_link_lifetime,
            neighbor_link_lifetime=neighbor_link_lifetime,
            geo_link_lifetime=geo_link_lifetime,
            ground_link_lifetime=ground_link_lifetime,
            local_link_margin_to_completion=local_link_margin_to_completion,
            neighbor_link_margin_to_completion=neighbor_link_margin_to_completion,
            geo_link_margin_to_completion=geo_link_margin_to_completion,
            ground_link_margin_to_completion=ground_link_margin_to_completion,
            local_handover_required=local_handover_required,
            neighbor_handover_required=neighbor_handover_required,
            geo_handover_required=geo_handover_required,
            ground_handover_required=ground_handover_required,
            local_candidate_count=local_candidate_count,
            neighbor_candidate_count=neighbor_candidate_count,
            geo_candidate_count=geo_candidate_count,
            ground_candidate_count=ground_candidate_count,
            provided=provided,
        )
        self._snapshot_cache[key] = (snapshot, hits, missing)
        return snapshot

    def stats(self) -> Dict[str, int]:
        return {
            "missing_leo_rows": int(self.missing_leo_rows),
            "missing_step_queries": int(self.missing_step_queries),
            "trace_pair_queries": int(self.trace_pair_queries),
            "trace_pair_hits": int(self.trace_pair_hits),
            "trace_pair_missing": int(self.trace_pair_missing),
            "trace_fallback_count": int(self.trace_fallback_count),
            "missing_mask_field_count": int(self.missing_mask_field_count),
            "missing_visible_mask_field_count": int(self.missing_visible_mask_field_count),
            "missing_completion_mask_field_count": int(self.missing_completion_mask_field_count),
            "missing_mobility_mask_field_count": int(self.missing_mobility_mask_field_count),
            "missing_final_mask_field_count": int(self.missing_final_mask_field_count),
            "delay_semantic_physical_seconds_actual_count": int(self.delay_semantic_counts["physical_seconds_actual"]),
            "delay_semantic_physical_seconds_controlled_estimate_count": int(
                self.delay_semantic_counts["physical_seconds_controlled_estimate"]
            ),
            "delay_semantic_normalized_score_count": int(self.delay_semantic_counts["normalized_score"]),
            "delay_semantic_legacy_unknown_count": int(self.delay_semantic_counts["legacy_unknown"]),
            "trace_hit_ratio": float(self.trace_pair_hits / max(1, self.trace_pair_queries)),
            "num_trace_steps": len(self.steps),
            "num_trace_rows": int(self.num_rows),
        }

    def _resolve_step(self, step: int) -> tuple[int, bool]:
        if self.repeat:
            return self.steps[step % len(self.steps)], False
        if step in self._by_step:
            return step, False
        if step > self.max_step:
            if self.strict:
                raise TraceTopologyMissingError(
                    f"strict topology trace requested step={step} beyond max_step={self.max_step} path={self.path}"
                )
            self.missing_step_queries += 1
            resolved = self.max_step
            self._warn_once(f"topology trace step={step} missing; reusing max available step={resolved}")
            return resolved, True
        if step in self._by_step:
            return step, False
        lower = [candidate for candidate in self.steps if candidate <= step]
        if lower:
            if self.strict:
                raise TraceTopologyMissingError(f"strict topology trace missing exact step={step} path={self.path}")
            self.missing_step_queries += 1
            resolved = lower[-1]
            self._warn_once(f"topology trace step={step} missing; reusing nearest earlier step={resolved}")
            return resolved, True
        if self.strict:
            raise TraceTopologyMissingError(f"strict topology trace missing exact step={step} path={self.path}")
        self.missing_step_queries += 1
        resolved = self.steps[0]
        self._warn_once(f"topology trace step={step} missing; reusing first available step={resolved}")
        return resolved, True

    def _warn_once(self, message: str) -> None:
        if self._warnings_emitted >= 8:
            return
        LOGGER.warning(message)
        self._warnings_emitted += 1

    @staticmethod
    def _load(path: Path) -> Dict[int, Dict[int, Dict[str, Any]]]:
        if path.suffix.lower() == ".jsonl":
            rows: List[Dict[str, Any]] = []
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        rows.append(json.loads(line))
        elif path.suffix.lower() == ".json":
            raw = json.loads(path.read_text(encoding="utf-8"))
            rows = raw if isinstance(raw, list) else list(raw.get("rows", raw.get("snapshots", [])))
        else:
            with path.open("r", encoding="utf-8", newline="") as f:
                rows = list(csv.DictReader(f))

        by_step: Dict[int, Dict[int, Dict[str, Any]]] = {}
        for row in rows:
            step = int(_to_float(row.get("step", row.get("decision_step", row.get("t", 0))), 0.0))
            leo = int(_to_float(row.get("leo_id", row.get("source_index", 0)), 0.0))
            by_step.setdefault(step, {})[leo] = row
        return by_step


def _parse_mask_value(raw: Any) -> List[bool] | None:
    if isinstance(raw, str) and raw.strip().startswith("["):
        try:
            raw = json.loads(raw)
        except ValueError:
            raw = None
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)) and len(raw) >= 4:
        return [bool(raw[i]) for i in range(4)]
    return None


_MASK_FIELD_ORDER = ("visible", "completion_safe", "mobility_safe", "final")

_DELAY_SEMANTIC_TO_CODE = {
    "physical_seconds_actual": 0,
    "physical_seconds_controlled_estimate": 1,
    "normalized_score": 2,
    "legacy_unknown": 3,
}


def _row_delay_semantic(row: Dict[str, Any]) -> str:
    explicit = str(row.get("delay_semantic") or "").strip()
    if explicit in _DELAY_SEMANTIC_TO_CODE:
        return explicit
    queue_source = str(row.get("queue_estimate_source", row.get("queueEstimateSource", ""))).strip().lower()
    trace_class = str(row.get("trace_semantic_class", "")).strip().lower()
    if queue_source == "controlled_estimate" or "controlled" in trace_class or _to_bool(row.get("is_controlled_rl_scenario"), False):
        return "physical_seconds_controlled_estimate"
    origin = str(row.get("trace_origin", row.get("traceOrigin", ""))).strip().lower()
    if origin in {"satedgesim", "satedgesim_real_dense"}:
        return "physical_seconds_actual"
    return "legacy_unknown"


def _row_mask(row: Dict[str, Any], mode: str, *, default: Sequence[bool] | None = None) -> List[bool]:
    mode = str(mode).strip().lower()
    candidates = []
    if mode == "visible":
        candidates = ["abstract_action_mask_visible", "abstractActionMaskVisible", "abstract_action_mask", "abstractActionMask"]
    elif mode == "mobility_safe":
        candidates = ["abstract_action_mask_mobility_safe", "abstractActionMaskMobilitySafe"]
    elif mode == "completion_safe":
        candidates = ["abstract_action_mask_completion_safe", "abstractActionMaskCompletionSafe"]
    elif mode == "final":
        candidates = ["abstract_action_mask_final", "abstractActionMaskFinal", "abstract_action_mask", "abstractActionMask"]
    for key in candidates:
        parsed = _parse_mask_value(row.get(key))
        if parsed is not None:
            return parsed
    if mode == "mobility_safe":
        keys = ("local_mobility_safe", "neighbor_mobility_safe", "geo_mobility_safe", "ground_mobility_safe")
        camel = ("localMobilitySafe", "neighborMobilitySafe", "geoMobilitySafe", "groundMobilitySafe")
        if any(k in row for k in keys) or any(k in row for k in camel):
            return [_to_bool(row.get(keys[i], row.get(camel[i])), False) for i in range(4)]
    if mode == "completion_safe":
        keys = ("local_completion_safe", "neighbor_completion_safe", "geo_completion_safe", "ground_completion_safe")
        camel = ("localCompletionSafe", "neighborCompletionSafe", "geoCompletionSafe", "groundCompletionSafe")
        if any(k in row for k in keys) or any(k in row for k in camel):
            return [_to_bool(row.get(keys[i], row.get(camel[i])), False) for i in range(4)]
    if default is not None:
        return [bool(x) for x in list(default)[:4]]
    return [True, False, False, False]


def _row_mask_field_presence(row: Dict[str, Any]) -> List[bool]:
    raw = row.get("mask_field_presence", row.get("maskFieldPresence"))
    if isinstance(raw, str):
        text = raw.strip()
        if text.startswith("{") or text.startswith("["):
            try:
                raw = json.loads(text)
            except ValueError:
                raw = None
    if isinstance(raw, dict):
        aliases = {
            "visible": ("visible", "abstract_action_mask_visible", "abstractActionMaskVisible"),
            "completion_safe": ("completion_safe", "completion", "abstract_action_mask_completion_safe", "abstractActionMaskCompletionSafe"),
            "mobility_safe": ("mobility_safe", "mobility", "mobility_risk", "abstract_action_mask_mobility_safe", "abstractActionMaskMobilitySafe"),
            "final": ("final", "abstract_action_mask_final", "abstractActionMaskFinal"),
        }
        return [any(_to_bool(raw.get(alias), False) for alias in aliases[name]) for name in _MASK_FIELD_ORDER]
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)) and len(raw) >= 4:
        return [bool(raw[i]) for i in range(4)]
    return [
        any(key in row for key in ("abstract_action_mask_visible", "abstractActionMaskVisible")),
        any(key in row for key in ("abstract_action_mask_completion_safe", "abstractActionMaskCompletionSafe")),
        any(key in row for key in ("abstract_action_mask_mobility_safe", "abstractActionMaskMobilitySafe")),
        any(key in row for key in ("abstract_action_mask_final", "abstractActionMaskFinal")),
    ]


def _tier_key_candidates(tier: str, metric: str) -> List[str]:
    camel_tier = tier[0].upper() + tier[1:]
    camel_metric = "".join(part.capitalize() for part in metric.split("_"))
    return [
        f"{tier}_{metric}",
        f"{tier}_best_{metric}",
        f"{tier}{camel_metric}",
        f"{camel_tier}{camel_metric}",
        f"{camel_tier}Best{camel_metric}",
    ]


def _tier_float(row: Dict[str, Any], tier: str, metric: str, *, default: float = 0.0) -> float:
    aliases = {
        "mobility_risk": [f"{tier}_mobility_risk_mean", f"{tier}MobilityRiskMean", f"{tier[0].upper() + tier[1:]}MobilityRiskMean"],
        "link_lifetime_sec": [f"{tier}_best_link_lifetime_sec", f"{tier[0].upper() + tier[1:]}BestLinkLifetimeSec"],
        "link_survival_margin_to_completion_sec": [
            f"{tier}_best_link_survival_margin_to_completion_sec",
            f"{tier}_best_link_survival_margin_sec",
            f"{tier[0].upper() + tier[1:]}BestLinkSurvivalMarginToCompletionSec",
            f"{tier[0].upper() + tier[1:]}BestLinkSurvivalMarginSec",
        ],
    }
    for key in _tier_key_candidates(tier, metric) + aliases.get(metric, []):
        if row.get(key) not in (None, ""):
            return _to_float(row.get(key), default)
    return default


def _tier_bool(row: Dict[str, Any], tier: str, metric: str, *, default: bool = False) -> bool:
    for key in _tier_key_candidates(tier, metric):
        if row.get(key) not in (None, ""):
            return _to_bool(row.get(key), default)
    return default


def _row_abstract_mask(row: Dict[str, Any]) -> List[bool]:
    parsed = _parse_mask_value(row.get("abstract_action_mask", row.get("abstractActionMask")))
    if parsed is not None:
        return parsed
    return [
        _to_bool(row.get("local_visible"), True),
        _to_bool(row.get("neighbor_visible", row.get("neighbor_available")), False),
        _to_bool(row.get("geo_visible", row.get("geo_available")), False),
        _to_bool(row.get("ground_visible", row.get("ground_available")), False),
    ]


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        if value in (None, ""):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _to_bool(value: Any, default: bool = False) -> bool:
    if value in (None, ""):
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "available", "feasible"}
    return bool(value)
