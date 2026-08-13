from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import torch

from trisatflow.envs.obs_builder import build_shared_observation, canonical_row, dense_rows_from_state, ring_edge_index
from trisatflow.envs.obs_schema import ACTION_NAMES, FIELD_NAMES
from trisatflow.satedgesim_eval.action_mapper import abstract_action_mask_from_state, map_upper_to_target_vm_with_trace
from trisatflow.satedgesim_eval.client import SatEdgeSimClient, SatEdgeSimClientError
from trisatflow.satedgesim_eval.state_adapter import source_leo_id_from_state


def load_trace_groups(trace_path: str | Path, *, n_leo: int, num_states: int) -> List[List[Dict[str, Any]]]:
    by_step: Dict[int, Dict[int, Dict[str, Any]]] = {}
    path = Path(trace_path)
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            step = int(row.get("step", 0))
            leo_id = int(row.get("leo_id", row.get("sourceDeviceId", 0)))
            by_step.setdefault(step, {})[leo_id] = row
    groups: List[List[Dict[str, Any]]] = []
    for step in sorted(by_step):
        rows = by_step[step]
        if len(rows) < n_leo:
            continue
        group = [rows[leo] for leo in sorted(rows)[:n_leo]]
        groups.append(group)
        if len(groups) >= num_states:
            break
    return groups


def load_trace_rows(trace_path: str | Path, *, num_rows: int) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    path = Path(trace_path)
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if len(rows) >= num_rows:
                break
    return rows


def shared_batch_from_trace_group(
    group: Sequence[Mapping[str, Any]],
    *,
    node_feature_dim: int,
    normalization_mode: str = "legacy",
    normalization_stats: Mapping[str, Any] | None = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch = build_shared_observation(
        group,
        source_index=0,
        node_feature_dim=node_feature_dim,
        normalization_mode=normalization_mode,
        normalization_stats=normalization_stats,
    )
    edge_index, edge_attr = ring_edge_index(batch.obs.shape[0])
    return batch.obs, edge_index, edge_attr


def collect_live_states(
    *,
    base_url: str,
    scenario_profile: str,
    task_source_mode: str,
    num_states: int,
    devices_count: int = 20,
    seed: int = 13,
    poll_sleep_sec: float = 0.05,
    request_timeout: float = 30.0,
) -> List[Dict[str, Any]]:
    client = SatEdgeSimClient(base_url, timeout=request_timeout)
    states: List[Dict[str, Any]] = []
    round_robin_index = 0
    while len(states) < num_states:
        state = client.reset(
            devices_count=devices_count,
            algorithm_index=0,
            architecture_index=0,
            seed=seed + len(states),
            clean_output_folder=False,
            wait_for_first_decision=True,
            wait_timeout_ms=30000,
            extra={
                "scenarioProfile": scenario_profile,
                "taskSourceMode": task_source_mode,
                "maxDecisions": max(32, min(1024, num_states - len(states) + 8)),
            },
        )
        polls = 0
        while polls < 300 and state.get("status") not in {"WAITING_FOR_ACTION", "FINISHED", "CLOSED", "FAILED", "ERROR"}:
            time.sleep(poll_sleep_sec)
            state = client.get_state()
            polls += 1
        while len(states) < num_states and state.get("status") == "WAITING_FOR_ACTION":
            states.append(dict(state))
            mask = list(abstract_action_mask_from_state(state))
            visible_actions = [idx for idx, visible in enumerate(mask) if bool(visible)]
            if not visible_actions:
                break
            action = visible_actions[round_robin_index % len(visible_actions)]
            round_robin_index += 1
            target_index, target_trace = map_upper_to_target_vm_with_trace(state, action, require_visible=True)
            task = state.get("task") or {}
            client.apply_action(
                {
                    "decisionId": state.get("decisionId"),
                    "requestId": state.get("decisionId"),
                    "taskId": task.get("id", state.get("taskId")),
                    "policyUpperAction": action,
                    "policyUpperActionName": ACTION_NAMES[action],
                    "abstractAction": action,
                    "abstractActionName": ACTION_NAMES[action],
                    "targetVmIndex": target_index,
                    "targetVmId": target_trace.get("selected_vm_id", -1),
                    "selectedVmId": target_trace.get("selected_vm_id", -1),
                }
            )
            polls = 0
            state = client.get_state()
            while polls < 300 and state.get("status") not in {"WAITING_FOR_ACTION", "FINISHED", "CLOSED", "FAILED", "ERROR"}:
                time.sleep(poll_sleep_sec)
                state = client.get_state()
                polls += 1
        if state.get("status") in {"FAILED", "ERROR"}:
            raise SatEdgeSimClientError(f"live collection failed: {state.get('message', state.get('status'))}")
        if state.get("status") in {"FINISHED", "CLOSED"} and len(states) < num_states:
            continue
        if state.get("status") != "WAITING_FOR_ACTION" and len(states) < num_states:
            break
    return states


def shared_batch_from_live_state(
    state: Mapping[str, Any],
    *,
    node_feature_dim: int,
    normalization_mode: str = "legacy",
    normalization_stats: Mapping[str, Any] | None = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, List[Dict[str, Any]]]:
    rows = dense_rows_from_state(state)
    if not rows:
        raise ValueError("state has no denseSourceSummaries")
    source_leo = source_leo_id_from_state(dict(state), fallback_n=max(1, len(rows)))
    source_index = 0
    for idx, row in enumerate(rows):
        if int(row.get("leo_id", idx)) == source_leo:
            source_index = idx
            break
    batch = build_shared_observation(
        rows,
        source_index=source_index,
        node_feature_dim=node_feature_dim,
        normalization_mode=normalization_mode,
        normalization_stats=normalization_stats,
    )
    edge_index, edge_attr = ring_edge_index(batch.obs.shape[0])
    return batch.obs, edge_index, edge_attr, batch.source_index, batch.raw_rows


def current_dense_row_from_state(state: Mapping[str, Any]) -> Dict[str, Any]:
    rows = dense_rows_from_state(state)
    if not rows:
        raise ValueError("state has no denseSourceSummaries")
    source_leo = source_leo_id_from_state(dict(state), fallback_n=max(1, len(rows)))
    for idx, row in enumerate(rows):
        if int(row.get("leo_id", idx)) == source_leo:
            return row
    return rows[0]


def describe_mask_distribution(masks: torch.Tensor) -> Dict[str, float]:
    if masks.numel() == 0:
        return {}
    counts: Dict[str, int] = {}
    for row in masks.detach().cpu().int().tolist():
        key = "".join(str(int(bit)) for bit in row[:4])
        counts[key] = counts.get(key, 0) + 1
    total = float(sum(counts.values()))
    return {key: value / max(1.0, total) for key, value in sorted(counts.items())}


def summarize_field_stats(values: Sequence[float]) -> Dict[str, float]:
    if not values:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "p05": 0.0, "p50": 0.0, "p95": 0.0}
    tensor = torch.tensor(list(values), dtype=torch.float32)
    quantiles = torch.quantile(tensor, torch.tensor([0.05, 0.50, 0.95], dtype=torch.float32))
    return {
        "mean": float(tensor.mean()),
        "std": float(tensor.std(unbiased=False)),
        "min": float(tensor.min()),
        "max": float(tensor.max()),
        "p05": float(quantiles[0]),
        "p50": float(quantiles[1]),
        "p95": float(quantiles[2]),
    }


def raw_field_series(rows: Iterable[Mapping[str, Any]]) -> Dict[str, List[float]]:
    out = {field: [] for field in FIELD_NAMES}
    for row in rows:
        canonical = canonical_row(row)
        for field in FIELD_NAMES:
            out[field].append(float(canonical.get(field, 0.0)))
    return out
