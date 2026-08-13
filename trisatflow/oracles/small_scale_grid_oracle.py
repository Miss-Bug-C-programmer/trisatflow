from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import torch

from trisatflow.envs.geo_leo_ground_env import GeoLeoGroundEnv


@dataclass
class OracleResult:
    oracle_action: torch.Tensor
    oracle_lower_action: torch.Tensor
    oracle_cost: float
    oracle_mode: str
    evaluated_candidates: int
    exact: bool
    metadata: Dict[str, Any]


def compute_oracle_gap(method_cost: float, oracle_cost: float, eps: float = 1.0e-8) -> float:
    denom = max(abs(float(oracle_cost)), float(eps))
    return float((float(method_cost) - float(oracle_cost)) / denom)


class SmallScaleGridOracle:
    """Small-scale grid oracle using the same env.step cost estimator.

    The oracle is exact only when the joint action/resource grid is below
    ``max_exact_candidates``. Larger settings are explicitly marked
    ``beam_grid_approx`` and must not be described as exact/MINLP.
    """

    def __init__(
        self,
        *,
        resource_grid: Sequence[float] = (0.25, 0.5, 0.75, 1.0),
        max_exact_candidates: int = 20000,
        beam_width: int = 8,
        max_n_leo: int = 4,
    ) -> None:
        self.resource_grid = tuple(float(v) for v in resource_grid)
        self.max_exact_candidates = int(max_exact_candidates)
        self.beam_width = int(beam_width)
        self.max_n_leo = int(max_n_leo)

    @property
    def metadata(self) -> Dict[str, Any]:
        return {
            "method": "small_scale_grid_oracle",
            "baseline_family": "small_scale_grid_oracle",
            "trainable": False,
            "update_implemented": False,
            "mask_supported": True,
            "action_mask_supported": True,
            "continuous_action_supported": True,
            "paper_ready": False,
            "smoke_training_passed": False,
            "full_experiment_required": True,
            "oracle_name_guard": "grid_oracle_not_minlp",
        }

    def solve_one_step(self, env: GeoLeoGroundEnv) -> OracleResult:
        if env.cfg.n_leo > self.max_n_leo:
            raise ValueError(f"SmallScaleGridOracle supports n_leo <= {self.max_n_leo}; got {env.cfg.n_leo}")
        mask = env._upper_action_mask_at_step(env.t).detach().bool()
        per_agent = self._candidate_lists(mask)
        total = 1
        for candidates in per_agent:
            total *= max(1, len(candidates))
        if total <= self.max_exact_candidates:
            return self._solve_exact(env, per_agent, total)
        return self._solve_beam(env, per_agent, total)

    def _candidate_lists(self, mask: torch.Tensor) -> List[List[Tuple[int, Tuple[float, float, float]]]]:
        resource_rows = list(itertools.product(self.resource_grid, repeat=3))
        out: List[List[Tuple[int, Tuple[float, float, float]]]] = []
        for row in mask.detach().cpu().tolist():
            actions = [idx for idx, bit in enumerate(row) if bool(bit)]
            if not actions:
                actions = [0]
            out.append([(int(action), tuple(float(v) for v in lower)) for action in actions for lower in resource_rows])
        return out

    def _snapshot(self, env: GeoLeoGroundEnv) -> Dict[str, Any]:
        return {
            "t": env.t,
            "queue": env.queue.clone(),
            "virtual_delay_queue": env.virtual_delay_queue.clone(),
            "energy": env.energy.clone(),
            "last_arrivals": env.last_arrivals.clone(),
            "last_service": env.last_service.clone(),
            "last_task_bits": env.last_task_bits.clone(),
            "last_cycles_per_bit": env.last_cycles_per_bit.clone(),
            "episode_action_counts": env.episode_action_counts.clone(),
            "generator_state": env.generator.get_state(),
        }

    def _restore(self, env: GeoLeoGroundEnv, state: Dict[str, Any]) -> None:
        env.t = int(state["t"])
        env.queue = state["queue"].clone()
        env.virtual_delay_queue = state["virtual_delay_queue"].clone()
        env.energy = state["energy"].clone()
        env.last_arrivals = state["last_arrivals"].clone()
        env.last_service = state["last_service"].clone()
        env.last_task_bits = state["last_task_bits"].clone()
        env.last_cycles_per_bit = state["last_cycles_per_bit"].clone()
        env.episode_action_counts = state["episode_action_counts"].clone()
        env.generator.set_state(state["generator_state"])
        env._trace_snapshot_cache_step = None
        env._trace_snapshot_cache_value = None
        env._action_mask_cache_step = None
        env._action_mask_cache_value = None

    def _evaluate_joint(self, env: GeoLeoGroundEnv, joint: Sequence[Tuple[int, Tuple[float, float, float]]], snapshot: Dict[str, Any]) -> float:
        self._restore(env, snapshot)
        upper = torch.tensor([item[0] for item in joint], dtype=torch.long, device=env.device)
        lower = torch.tensor([item[1] for item in joint], dtype=torch.float32, device=env.device)
        step = env.step(upper, lower, minimal_info=True)
        cost = step.info.get("normalized_system_cost")
        if torch.is_tensor(cost):
            return float(cost.float().mean().detach().cpu().item())
        return float(cost)

    def _solve_exact(
        self,
        env: GeoLeoGroundEnv,
        per_agent: List[List[Tuple[int, Tuple[float, float, float]]]],
        total: int,
    ) -> OracleResult:
        snapshot = self._snapshot(env)
        best_cost = float("inf")
        best_joint: Sequence[Tuple[int, Tuple[float, float, float]]] | None = None
        evaluated = 0
        for joint in itertools.product(*per_agent):
            cost = self._evaluate_joint(env, joint, snapshot)
            evaluated += 1
            if cost < best_cost:
                best_cost = cost
                best_joint = joint
        assert best_joint is not None
        self._restore(env, snapshot)
        return self._result_from_joint(env, best_joint, best_cost, "exact_grid", evaluated, True, total)

    def _solve_beam(
        self,
        env: GeoLeoGroundEnv,
        per_agent: List[List[Tuple[int, Tuple[float, float, float]]]],
        total: int,
    ) -> OracleResult:
        snapshot = self._snapshot(env)
        neutral = [(0, (1.0, 1.0, 1.0)) for _ in range(env.cfg.n_leo)]
        reduced: List[List[Tuple[int, Tuple[float, float, float]]]] = []
        evaluated = 0
        for agent_idx, candidates in enumerate(per_agent):
            scored = []
            for candidate in candidates:
                joint = list(neutral)
                joint[agent_idx] = candidate
                scored.append((self._evaluate_joint(env, joint, snapshot), candidate))
                evaluated += 1
            scored.sort(key=lambda x: x[0])
            reduced.append([candidate for _, candidate in scored[: max(1, self.beam_width)]])
        best_cost = float("inf")
        best_joint = None
        for joint in itertools.product(*reduced):
            cost = self._evaluate_joint(env, joint, snapshot)
            evaluated += 1
            if cost < best_cost:
                best_cost = cost
                best_joint = joint
        assert best_joint is not None
        self._restore(env, snapshot)
        return self._result_from_joint(env, best_joint, best_cost, "beam_grid_approx", evaluated, False, total)

    def _result_from_joint(
        self,
        env: GeoLeoGroundEnv,
        joint: Sequence[Tuple[int, Tuple[float, float, float]]],
        cost: float,
        mode: str,
        evaluated: int,
        exact: bool,
        total: int,
    ) -> OracleResult:
        upper = torch.tensor([item[0] for item in joint], dtype=torch.long, device=env.device)
        lower = torch.tensor([item[1] for item in joint], dtype=torch.float32, device=env.device).clamp(0.0, 1.0)
        return OracleResult(
            oracle_action=upper,
            oracle_lower_action=lower,
            oracle_cost=float(cost),
            oracle_mode=mode,
            evaluated_candidates=int(evaluated),
            exact=bool(exact),
            metadata={
                "oracle_mode": mode,
                "candidate_space_size": int(total),
                "evaluated_candidates": int(evaluated),
                "uses_same_env_step_cost_estimator": True,
                "oracle_name_guard": "grid_oracle_not_minlp",
            },
        )

