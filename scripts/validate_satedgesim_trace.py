from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Sequence


LAYER_FIELDS = {
    "visible": ("abstract_action_mask_visible", "abstractActionMaskVisible"),
    "completion_safe": ("abstract_action_mask_completion_safe", "abstractActionMaskCompletionSafe"),
    "mobility_safe": ("abstract_action_mask_mobility_safe", "abstractActionMaskMobilitySafe"),
    "final": ("abstract_action_mask_final", "abstractActionMaskFinal", "abstract_action_mask", "abstractActionMask"),
}


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


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _parse_mask(raw: Any) -> List[int] | None:
    if isinstance(raw, str) and raw.strip().startswith("["):
        try:
            raw = json.loads(raw)
        except ValueError:
            raw = None
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)) and len(raw) >= 4:
        return [1 if bool(raw[i]) else 0 for i in range(4)]
    return None


def _mask(row: Dict[str, Any], layer: str = "final") -> List[int]:
    for key in LAYER_FIELDS.get(layer, ()):
        parsed = _parse_mask(row.get(key))
        if parsed is not None:
            return parsed
    if layer == "visible":
        return [
            1 if _to_bool(row.get("local_visible"), True) else 0,
            1 if _to_bool(row.get("neighbor_visible")) else 0,
            1 if _to_bool(row.get("geo_visible")) else 0,
            1 if _to_bool(row.get("ground_visible")) else 0,
        ]
    if layer in {"completion_safe", "mobility_safe"}:
        prefix = "completion" if layer == "completion_safe" else "mobility"
        return [
            1 if _to_bool(row.get(f"local_{prefix}_safe"), False) else 0,
            1 if _to_bool(row.get(f"neighbor_{prefix}_safe"), False) else 0,
            1 if _to_bool(row.get(f"geo_{prefix}_safe"), False) else 0,
            1 if _to_bool(row.get(f"ground_{prefix}_safe"), False) else 0,
        ]
    return [
        1 if _to_bool(row.get("local_visible"), True) else 0,
        1 if _to_bool(row.get("neighbor_visible")) else 0,
        1 if _to_bool(row.get("geo_visible")) else 0,
        1 if _to_bool(row.get("ground_visible")) else 0,
    ]


def _presence(row: Dict[str, Any]) -> Dict[str, bool]:
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
        return {name: any(_to_bool(raw.get(alias), False) for alias in names) for name, names in aliases.items()}
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)) and len(raw) >= 4:
        return {
            "visible": bool(raw[0]),
            "completion_safe": bool(raw[1]),
            "mobility_safe": bool(raw[2]),
            "final": bool(raw[3]),
        }
    return {name: any(key in row for key in keys) for name, keys in LAYER_FIELDS.items()}


def _mask_key(mask: Sequence[int]) -> str:
    return "".join(str(int(bool(x))) for x in list(mask)[:4])


def _prune_ratio(before_masks: List[List[int]], after_masks: List[List[int]], *, remote_only: bool = True) -> float:
    indexes = (1, 2, 3) if remote_only else (0, 1, 2, 3)
    opportunities = sum(int(before[i]) for before in before_masks for i in indexes)
    pruned = sum(int(before[i] and not after[i]) for before, after in zip(before_masks, after_masks) for i in indexes)
    return pruned / max(1, opportunities)


def _visibility_prune_ratio(visible_masks: List[List[int]]) -> float:
    opportunities = len(visible_masks) * 3
    pruned = sum(int(not mask[i]) for mask in visible_masks for i in (1, 2, 3))
    return pruned / max(1, opportunities)


def _transition_counts(rows: List[Dict[str, Any]], final_masks: List[List[int]]) -> Dict[str, int]:
    indexed = []
    for row, mask in zip(rows, final_masks):
        indexed.append((int(_to_float(row.get("leo_id"), 0.0)), int(_to_float(row.get("step"), 0.0)), mask))
    grouped: Dict[int, List[tuple[int, List[int]]]] = {}
    for leo, step, mask in indexed:
        grouped.setdefault(leo, []).append((step, mask))
    counts = {"neighbor": 0, "geo": 0, "ground": 0}
    for items in grouped.values():
        items.sort(key=lambda item: item[0])
        for (_, prev), (_, curr) in zip(items, items[1:]):
            counts["neighbor"] += int(prev[1] != curr[1])
            counts["geo"] += int(prev[2] != curr[2])
            counts["ground"] += int(prev[3] != curr[3])
    return counts


def _selected_layer(action_mask_mode: str) -> str:
    mode = str(action_mask_mode or "visible_only").strip().lower()
    if mode in {"visible_only", "visibility", "visibility_only"}:
        return "visible"
    if mode in {"mobility_safe", "mobility_risk"}:
        return "mobility_safe"
    if mode in {"completion_safe", "full", "full_mask"}:
        return "completion_safe"
    return "visible"


def _subset_count(lhs_masks: List[List[int]], rhs_masks: List[List[int]]) -> int:
    return sum(
        int(any(bool(lhs[i]) and not bool(rhs[i]) for i in range(4)))
        for lhs, rhs in zip(lhs_masks, rhs_masks)
    )


def _final_layer_mismatch_count(
    rows: List[Dict[str, Any]],
    *,
    visible_masks: List[List[int]],
    mobility_masks: List[List[int]],
    completion_masks: List[List[int]],
    final_masks: List[List[int]],
) -> int:
    layer_to_masks = {
        "visible": visible_masks,
        "mobility_safe": mobility_masks,
        "completion_safe": completion_masks,
    }
    mismatches = 0
    for idx, row in enumerate(rows):
        selected = _selected_layer(str(row.get("action_mask_mode", row.get("actionMaskMode", "visible_only"))))
        expected = layer_to_masks[selected][idx]
        if final_masks[idx] != expected:
            mismatches += 1
    return mismatches


def _row_text(row: Dict[str, Any], *keys: str, default: str = "") -> str:
    for key in keys:
        value = row.get(key)
        if value not in (None, ""):
            return str(value)
    return default


def _ratio(rows: List[Dict[str, Any]], key: str) -> float:
    if not rows:
        return 0.0
    return sum(1 for row in rows if _to_bool(row.get(key))) / len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate SatEdgeSim-aligned topology traces.")
    parser.add_argument("--trace", type=str, required=True)
    parser.add_argument("--min-rows", type=int, default=100)
    parser.add_argument("--expected-n-leo", type=int, default=None)
    parser.add_argument("--expected-min-steps", type=int, default=None)
    parser.add_argument("--require-dense", action="store_true")
    parser.add_argument("--min-geo-visible-ratio", type=float, default=0.0)
    parser.add_argument("--min-ground-visible-ratio", type=float, default=0.0)
    parser.add_argument("--min-neighbor-visible-ratio", type=float, default=0.0)
    parser.add_argument("--min-remote-visible-ratio", type=float, default=0.20)
    parser.add_argument("--paper-strict", action="store_true")
    parser.add_argument("--trace-semantic-class", type=str, default="")
    parser.add_argument("--require-origin-satedgesim", action="store_true")
    parser.add_argument("--require-explicit-layered-masks", action="store_true")
    parser.add_argument("--require-success-profile", type=str, default="")
    parser.add_argument("--require-action-mask-mode", type=str, default="")
    parser.add_argument("--require-dense-projection", type=str, default="")
    parser.add_argument("--max-fully-open-ratio", type=float, default=None, help="Deprecated alias for --max-fully-open-final-ratio.")
    parser.add_argument("--max-fully-open-visible-ratio", type=float, default=None)
    parser.add_argument("--max-fully-open-final-ratio", type=float, default=None)
    parser.add_argument("--min-unique-visible-masks", type=int, default=None)
    parser.add_argument("--min-unique-final-masks", type=int, default=None)
    parser.add_argument("--min-remote-tier-transition-count", type=int, default=None)
    parser.add_argument("--min-neighbor-transition-count", type=int, default=None)
    parser.add_argument("--min-geo-transition-count", type=int, default=None)
    parser.add_argument("--min-ground-transition-count", type=int, default=None)
    parser.add_argument("--min-visibility-prune-ratio", type=float, default=None)
    parser.add_argument("--min-completion-prune-ratio", type=float, default=None)
    parser.add_argument("--min-mobility-prune-ratio", type=float, default=None)
    parser.add_argument("--min-phase-count", type=int, default=None)
    parser.add_argument("--output", type=str, default="")
    args = parser.parse_args()

    if args.paper_strict:
        args.require_origin_satedgesim = True
        args.require_explicit_layered_masks = True
        args.require_success_profile = args.require_success_profile or "paper_strict"
        args.require_action_mask_mode = args.require_action_mask_mode or "completion_safe"
        args.require_dense_projection = args.require_dense_projection or "source_projection"
        args.max_fully_open_visible_ratio = 0.95 if args.max_fully_open_visible_ratio is None else args.max_fully_open_visible_ratio
        args.max_fully_open_final_ratio = 0.95 if args.max_fully_open_final_ratio is None else args.max_fully_open_final_ratio
        if args.max_fully_open_ratio is not None and args.max_fully_open_final_ratio is None:
            args.max_fully_open_final_ratio = args.max_fully_open_ratio
        args.min_unique_visible_masks = 2 if args.min_unique_visible_masks is None else args.min_unique_visible_masks
        args.min_unique_final_masks = 2 if args.min_unique_final_masks is None else args.min_unique_final_masks
        args.min_neighbor_transition_count = 1 if args.min_neighbor_transition_count is None else args.min_neighbor_transition_count
        args.min_geo_transition_count = 1 if args.min_geo_transition_count is None else args.min_geo_transition_count
        args.min_ground_transition_count = 1 if args.min_ground_transition_count is None else args.min_ground_transition_count
        args.min_phase_count = 2 if args.min_phase_count is None else args.min_phase_count
        if args.min_rows == 100:
            args.min_rows = 1
    if args.max_fully_open_ratio is not None and args.max_fully_open_final_ratio is None:
        args.max_fully_open_final_ratio = args.max_fully_open_ratio

    path = Path(args.trace)
    rows = _read_jsonl(path)
    visible_masks = [_mask(row, "visible") for row in rows]
    completion_masks = [_mask(row, "completion_safe") for row in rows]
    mobility_masks = [_mask(row, "mobility_safe") for row in rows]
    final_masks = [_mask(row, "final") for row in rows]
    mask_sizes = [sum(mask) for mask in final_masks]
    contradictions = 0
    trace_pairs = set()
    steps = set()
    leos = set()
    per_leo_counts: Counter[int] = Counter()
    remote_visible_count = 0
    missing_layered_rows = 0
    phases = set()
    for row, visible_mask in zip(rows, visible_masks):
        visible = [
            _to_bool(row.get("local_visible"), bool(visible_mask[0])),
            _to_bool(row.get("neighbor_visible"), bool(visible_mask[1])),
            _to_bool(row.get("geo_visible"), bool(visible_mask[2])),
            _to_bool(row.get("ground_visible"), bool(visible_mask[3])),
        ]
        contradictions += sum(int(int(visible_mask[i]) != int(visible[i])) for i in range(4))
        step = int(_to_float(row.get("step"), 0.0))
        leo = int(_to_float(row.get("leo_id"), 0.0))
        phase = row.get("phase_id", row.get("phaseId", row.get("scenario_phase", row.get("scenarioPhase", ""))))
        phases.add(str(phase))
        trace_pairs.add((step, leo))
        steps.add(step)
        leos.add(leo)
        per_leo_counts[leo] += 1
        if visible[1] or visible[2] or visible[3]:
            remote_visible_count += 1
        row_presence = _presence(row)
        if not all(row_presence.get(name, False) for name in ("visible", "completion_safe", "mobility_safe", "final")):
            missing_layered_rows += 1

    expected_dense_rows = None
    dense_coverage_ratio = None
    trace_missing_pairs = None
    if args.expected_n_leo is not None and args.expected_min_steps is not None:
        expected_dense_rows = args.expected_n_leo * args.expected_min_steps
        expected_pairs = {(step, leo) for step in range(args.expected_min_steps) for leo in range(args.expected_n_leo)}
        trace_missing_pairs = len(expected_pairs - trace_pairs)
        dense_coverage_ratio = len(trace_pairs) / max(1, len(expected_pairs))

    transition_counts = _transition_counts(rows, final_masks)
    fully_open_visible_ratio = sum(1 for mask in visible_masks if mask == [1, 1, 1, 1]) / max(1, len(visible_masks))
    fully_open_final_ratio = sum(1 for mask in final_masks if mask == [1, 1, 1, 1]) / max(1, len(final_masks))
    visibility_prune_ratio = _visibility_prune_ratio(visible_masks)
    completion_prune_ratio = _prune_ratio(visible_masks, completion_masks)
    mobility_prune_ratio = _prune_ratio(completion_masks, mobility_masks)
    fallback_due_missing_field_ratio = missing_layered_rows / max(1, len(rows))
    mobility_not_subset_visible = _subset_count(mobility_masks, visible_masks)
    completion_not_subset_mobility = _subset_count(completion_masks, mobility_masks)
    completion_not_subset_visible = _subset_count(completion_masks, visible_masks)
    completion_relation_violations = sum(
        int(
            any(bool(completion[i]) and not bool(mobility[i]) for i in range(4))
            and any(bool(completion[i]) and not bool(visible[i]) for i in range(4))
        )
        for completion, mobility, visible in zip(completion_masks, mobility_masks, visible_masks)
    )
    final_layer_mismatch_count = _final_layer_mismatch_count(
        rows,
        visible_masks=visible_masks,
        mobility_masks=mobility_masks,
        completion_masks=completion_masks,
        final_masks=final_masks,
    )
    trace_origins = Counter(_row_text(row, "trace_origin", "traceOrigin", default="") for row in rows)
    semantic_classes = Counter(_row_text(row, "trace_semantic_class", "traceSemanticClass", default="") for row in rows)
    success_profiles = Counter(_row_text(row, "success_profile", "successProfile", default="") for row in rows)
    action_mask_modes = Counter(_row_text(row, "action_mask_mode", "actionMaskMode", default="visible_only") for row in rows)
    dense_projection_modes = Counter(_row_text(row, "dense_projection_mode", "denseProjectionMode", default="") for row in rows)
    synthetic_count = sum(1 for row in rows if _to_bool(row.get("synthetic"), False))
    summary: Dict[str, Any] = {
        "trace": str(path),
        "num_rows": len(rows),
        "num_unique_leo": len(leos),
        "num_unique_steps": len(steps),
        "expected_dense_rows": expected_dense_rows,
        "dense_coverage_ratio": dense_coverage_ratio,
        "local_visible_ratio": sum(mask[0] for mask in visible_masks) / max(1, len(visible_masks)),
        "neighbor_visible_ratio": sum(mask[1] for mask in visible_masks) / max(1, len(visible_masks)),
        "geo_visible_ratio": sum(mask[2] for mask in visible_masks) / max(1, len(visible_masks)),
        "ground_visible_ratio": sum(mask[3] for mask in visible_masks) / max(1, len(visible_masks)),
        "remote_visible_ratio": remote_visible_count / max(1, len(rows)),
        "mean_mask_size": mean(mask_sizes) if mask_sizes else 0.0,
        "fully_open_ratio": fully_open_final_ratio,
        "fully_open_visible_ratio": fully_open_visible_ratio,
        "fully_open_final_ratio": fully_open_final_ratio,
        "unique_visible_masks": len({_mask_key(mask) for mask in visible_masks}),
        "unique_completion_safe_masks": len({_mask_key(mask) for mask in completion_masks}),
        "unique_mobility_safe_masks": len({_mask_key(mask) for mask in mobility_masks}),
        "unique_final_masks": len({_mask_key(mask) for mask in final_masks}),
        "neighbor_transition_count": transition_counts["neighbor"],
        "geo_transition_count": transition_counts["geo"],
        "ground_transition_count": transition_counts["ground"],
        "visibility_prune_ratio": visibility_prune_ratio,
        "completion_prune_ratio": completion_prune_ratio,
        "mobility_prune_ratio": mobility_prune_ratio,
        "fallback_due_missing_field_ratio": fallback_due_missing_field_ratio,
        "mobility_not_subset_visible_count": mobility_not_subset_visible,
        "completion_not_subset_mobility_count": completion_not_subset_mobility,
        "completion_not_subset_visible_count": completion_not_subset_visible,
        "completion_relation_violation_count": completion_relation_violations,
        "final_layer_mismatch_count": final_layer_mismatch_count,
        "trace_origins": dict(trace_origins),
        "trace_semantic_classes": dict(semantic_classes),
        "success_profiles": dict(success_profiles),
        "action_mask_modes": dict(action_mask_modes),
        "dense_projection_modes": dict(dense_projection_modes),
        "synthetic_ratio": synthetic_count / max(1, len(rows)),
        "num_phase": len(phases),
        "trace_missing_pairs": trace_missing_pairs,
        "per_leo_row_count_min": min(per_leo_counts.values()) if per_leo_counts else 0,
        "per_leo_row_count_max": max(per_leo_counts.values()) if per_leo_counts else 0,
        "per_leo_row_count_mean": mean(per_leo_counts.values()) if per_leo_counts else 0.0,
        "mask_visible_contradictions": contradictions,
    }

    failures: List[str] = []
    if summary["num_rows"] < args.min_rows:
        failures.append(f"num_rows={summary['num_rows']} < min_rows={args.min_rows}")
    if summary["local_visible_ratio"] <= 0.0:
        failures.append("local_visible_ratio==0")
    if summary["neighbor_visible_ratio"] < args.min_neighbor_visible_ratio:
        failures.append(
            f"neighbor_visible_ratio={summary['neighbor_visible_ratio']:.6f} < min_neighbor_visible_ratio={args.min_neighbor_visible_ratio}"
        )
    if summary["geo_visible_ratio"] < args.min_geo_visible_ratio:
        failures.append(
            f"geo_visible_ratio={summary['geo_visible_ratio']:.6f} < min_geo_visible_ratio={args.min_geo_visible_ratio}"
        )
    if summary["ground_visible_ratio"] < args.min_ground_visible_ratio:
        failures.append(
            f"ground_visible_ratio={summary['ground_visible_ratio']:.6f} < min_ground_visible_ratio={args.min_ground_visible_ratio}"
        )
    if summary["remote_visible_ratio"] < args.min_remote_visible_ratio:
        failures.append(
            f"remote_visible_ratio={summary['remote_visible_ratio']:.6f} < min_remote_visible_ratio={args.min_remote_visible_ratio}"
        )
    if summary["geo_visible_ratio"] <= 0.0:
        failures.append("geo_visible_ratio==0")
    if summary["ground_visible_ratio"] <= 0.0:
        failures.append("ground_visible_ratio==0")
    if contradictions > 0:
        failures.append(f"mask_visible_contradictions={contradictions}")
    if mobility_not_subset_visible > 0:
        failures.append(f"mobility_not_subset_visible_count={mobility_not_subset_visible}")
    if completion_relation_violations > 0:
        failures.append(f"completion_relation_violation_count={completion_relation_violations}")
    if final_layer_mismatch_count > 0:
        failures.append(f"final_layer_mismatch_count={final_layer_mismatch_count}")
    if args.require_origin_satedgesim and (set(trace_origins) != {"satedgesim"} or synthetic_count > 0):
        failures.append(f"trace_origin_not_satedgesim={dict(trace_origins)} synthetic_count={synthetic_count}")
    if args.trace_semantic_class:
        bad = sum(count for value, count in semantic_classes.items() if value != args.trace_semantic_class)
        if bad > 0:
            failures.append(f"trace_semantic_class_mismatch={bad} expected={args.trace_semantic_class}")
    if args.require_success_profile:
        bad = sum(count for value, count in success_profiles.items() if value != args.require_success_profile)
        if bad > 0:
            failures.append(f"success_profile_mismatch={bad} expected={args.require_success_profile}")
    if args.require_action_mask_mode:
        bad = sum(count for value, count in action_mask_modes.items() if value != args.require_action_mask_mode)
        if bad > 0:
            failures.append(f"action_mask_mode_mismatch={bad} expected={args.require_action_mask_mode}")
    if args.require_dense_projection:
        bad = sum(count for value, count in dense_projection_modes.items() if value != args.require_dense_projection)
        if bad > 0:
            failures.append(f"dense_projection_mode_mismatch={bad} expected={args.require_dense_projection}")
    if args.require_explicit_layered_masks and missing_layered_rows > 0:
        failures.append(f"missing_explicit_layered_mask_rows={missing_layered_rows}")
    if args.max_fully_open_visible_ratio is not None and summary["fully_open_visible_ratio"] > args.max_fully_open_visible_ratio:
        failures.append(
            f"fully_open_visible_ratio={summary['fully_open_visible_ratio']:.6f} > "
            f"max_fully_open_visible_ratio={args.max_fully_open_visible_ratio}"
        )
    if args.max_fully_open_final_ratio is not None and summary["fully_open_final_ratio"] > args.max_fully_open_final_ratio:
        failures.append(
            f"fully_open_final_ratio={summary['fully_open_final_ratio']:.6f} > "
            f"max_fully_open_final_ratio={args.max_fully_open_final_ratio}"
        )
    if args.min_unique_visible_masks is not None and summary["unique_visible_masks"] < args.min_unique_visible_masks:
        failures.append(
            f"unique_visible_masks={summary['unique_visible_masks']} < min_unique_visible_masks={args.min_unique_visible_masks}"
        )
    if args.min_unique_final_masks is not None and summary["unique_final_masks"] < args.min_unique_final_masks:
        failures.append(
            f"unique_final_masks={summary['unique_final_masks']} < min_unique_final_masks={args.min_unique_final_masks}"
        )
    remote_transition_count = (
        summary["neighbor_transition_count"] + summary["geo_transition_count"] + summary["ground_transition_count"]
    )
    if args.min_remote_tier_transition_count is not None and remote_transition_count < args.min_remote_tier_transition_count:
        failures.append(
            f"remote_tier_transition_count={remote_transition_count} < "
            f"min_remote_tier_transition_count={args.min_remote_tier_transition_count}"
        )
    if args.min_neighbor_transition_count is not None and summary["neighbor_transition_count"] < args.min_neighbor_transition_count:
        failures.append(
            f"neighbor_transition_count={summary['neighbor_transition_count']} < "
            f"min_neighbor_transition_count={args.min_neighbor_transition_count}"
        )
    if args.min_geo_transition_count is not None and summary["geo_transition_count"] < args.min_geo_transition_count:
        failures.append(
            f"geo_transition_count={summary['geo_transition_count']} < min_geo_transition_count={args.min_geo_transition_count}"
        )
    if args.min_ground_transition_count is not None and summary["ground_transition_count"] < args.min_ground_transition_count:
        failures.append(
            f"ground_transition_count={summary['ground_transition_count']} < "
            f"min_ground_transition_count={args.min_ground_transition_count}"
        )
    if args.min_visibility_prune_ratio is not None and summary["visibility_prune_ratio"] < args.min_visibility_prune_ratio:
        failures.append(
            f"visibility_prune_ratio={summary['visibility_prune_ratio']:.6f} < "
            f"min_visibility_prune_ratio={args.min_visibility_prune_ratio}"
        )
    if args.min_completion_prune_ratio is not None and summary["completion_prune_ratio"] < args.min_completion_prune_ratio:
        failures.append(
            f"completion_prune_ratio={summary['completion_prune_ratio']:.6f} < "
            f"min_completion_prune_ratio={args.min_completion_prune_ratio}"
        )
    if args.min_mobility_prune_ratio is not None and summary["mobility_prune_ratio"] < args.min_mobility_prune_ratio:
        failures.append(
            f"mobility_prune_ratio={summary['mobility_prune_ratio']:.6f} < "
            f"min_mobility_prune_ratio={args.min_mobility_prune_ratio}"
        )
    if args.min_phase_count is not None and summary["num_phase"] < args.min_phase_count:
        failures.append(f"num_phase={summary['num_phase']} < min_phase_count={args.min_phase_count}")

    if args.expected_n_leo is not None and summary["num_unique_leo"] < args.expected_n_leo:
        failures.append(f"num_unique_leo={summary['num_unique_leo']} < expected_n_leo={args.expected_n_leo}")
    if args.expected_min_steps is not None and summary["num_unique_steps"] < args.expected_min_steps:
        failures.append(f"num_unique_steps={summary['num_unique_steps']} < expected_min_steps={args.expected_min_steps}")
    if args.require_dense and expected_dense_rows is not None and dense_coverage_ratio is not None:
        if summary["num_rows"] < expected_dense_rows:
            failures.append(f"num_rows={summary['num_rows']} < expected_dense_rows={expected_dense_rows}")
        if dense_coverage_ratio < 0.99:
            failures.append(f"dense_coverage_ratio={dense_coverage_ratio:.6f} < 0.99")
        for leo in range(args.expected_n_leo):
            if per_leo_counts.get(leo, 0) < args.expected_min_steps:
                failures.append(
                    f"leo_id={leo} row_count={per_leo_counts.get(leo, 0)} < expected_min_steps={args.expected_min_steps}"
                )
                break

    summary["status"] = "TRACE_VALIDATION_OK" if not failures else "TRACE_VALIDATION_FAILED"
    summary["failures"] = failures
    text = json.dumps(summary, ensure_ascii=False, indent=2)
    print(text)
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
