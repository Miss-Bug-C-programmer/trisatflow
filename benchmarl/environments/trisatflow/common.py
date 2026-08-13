#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

import copy
from typing import Callable, Dict, List, Optional

from torchrl.data import Composite
from torchrl.envs import EnvBase

from benchmarl.environments.common import Task, TaskClass
from benchmarl.utils import DEVICE_TYPING
from trisatflow.benchmarl_adapter import TriSatFlowBenchMARLEnv
from trisatflow.config import RewardWeights, ScenarioConfig


class TriSatFlowClass(TaskClass):
    def get_env_fun(
        self,
        num_envs: int,
        continuous_actions: bool,
        seed: Optional[int],
        device: DEVICE_TYPING,
    ) -> Callable[[], EnvBase]:
        del num_envs, continuous_actions
        config = copy.deepcopy(self.config or {})
        scenario_cfg = dict(config.get("scenario") or {})
        reward_cfg = dict(config.get("reward") or {})
        include_graph_specs = bool(config.get("include_graph_specs", True))
        group_name = str(config.get("group_name", "leo"))
        graph_max_edges = config.get("graph_max_edges", None)
        if seed is not None:
            scenario_cfg["seed"] = int(seed)
        scenario = ScenarioConfig(**scenario_cfg)
        reward = RewardWeights(**reward_cfg)
        return lambda: TriSatFlowBenchMARLEnv(
            scenario=scenario,
            reward_weights=reward,
            device=device,
            group_name=group_name,
            include_graph_specs=include_graph_specs,
            graph_max_edges=graph_max_edges,
        )

    def supports_continuous_actions(self) -> bool:
        return True

    def supports_discrete_actions(self) -> bool:
        return True

    def has_render(self, env: EnvBase) -> bool:
        del env
        return False

    def max_steps(self, env: EnvBase) -> int:
        inner = getattr(env, "inner", None)
        if inner is not None and hasattr(inner, "cfg"):
            return int(getattr(inner.cfg, "episode_len", 32))
        return int((self.config or {}).get("scenario", {}).get("episode_len", 32))

    def group_map(self, env: EnvBase) -> Dict[str, List[str]]:
        if hasattr(env, "group_map"):
            return dict(getattr(env, "group_map"))
        scenario_cfg = dict((self.config or {}).get("scenario") or {})
        n_leo = int(scenario_cfg.get("n_leo", 6))
        group = str((self.config or {}).get("group_name", "leo"))
        return {group: [f"leo_{i}" for i in range(n_leo)]}

    def state_spec(self, env: EnvBase) -> Optional[Composite]:
        return env.state_spec.clone()

    def action_mask_spec(self, env: EnvBase) -> Optional[Composite]:
        del env
        return None

    def observation_spec(self, env: EnvBase) -> Composite:
        return env.observation_spec.clone()

    def info_spec(self, env: EnvBase) -> Optional[Composite]:
        del env
        return None

    def action_spec(self, env: EnvBase) -> Composite:
        return env.action_spec.clone()

    @staticmethod
    def env_name() -> str:
        return "trisatflow"


class TriSatFlowTask(Task):
    """Enum for TriSatFlow tasks."""

    MIXED_SMALL = None

    @staticmethod
    def associated_class():
        return TriSatFlowClass
