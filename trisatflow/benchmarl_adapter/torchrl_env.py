from __future__ import annotations

"""Experimental TorchRL EnvBase adapters for TriSatFlow."""

from dataclasses import replace
from typing import Any, Dict, Optional

import torch
from tensordict import TensorDict
from torchrl.data.tensor_specs import Bounded, Categorical, Composite, Unbounded
from torchrl.envs import EnvBase

from trisatflow.config import RewardWeights, ScenarioConfig
from trisatflow.envs import GeoLeoGroundEnv


class TriSatFlowTorchRLEnv(EnvBase):
    """Minimal runnable EnvBase wrapper around :class:`GeoLeoGroundEnv`.

    This class supports:
    - flat action layout (default): ``upper_action`` + ``lower_action``;
    - optional grouped layout for BenchMARL-style keys via ``group_name``;
    - optional padded graph tensors in specs: ``edge_index``, ``edge_attr``, ``edge_mask``.
    """

    def __init__(
        self,
        scenario: Optional[ScenarioConfig] = None,
        reward_weights: Optional[RewardWeights] = None,
        *,
        device: str | torch.device = "cpu",
        group_name: str | None = None,
        include_graph_specs: bool = True,
        graph_max_edges: int | None = None,
    ) -> None:
        self._scenario = replace(scenario) if scenario is not None else ScenarioConfig()
        self._reward = replace(reward_weights) if reward_weights is not None else RewardWeights()
        self._scenario.seed = int(self._scenario.seed)
        self.group_name = str(group_name).strip() if group_name else None
        self.include_graph_specs = bool(include_graph_specs)

        super().__init__(device=torch.device(device), batch_size=torch.Size([]))

        self.inner = GeoLeoGroundEnv(self._scenario, self._reward, device=self.device)
        self.n_leo = int(self.inner.cfg.n_leo)
        self.node_feature_dim = int(self.inner.cfg.node_feature_dim)
        self.edge_feature_dim = int(self.inner.cfg.edge_feature_dim)
        default_max_edges = max(1, self.n_leo * max(1, self.n_leo - 1))
        self.graph_max_edges = int(graph_max_edges) if graph_max_edges is not None else int(default_max_edges)
        if self.graph_max_edges <= 0:
            raise ValueError(f"graph_max_edges must be > 0, got {self.graph_max_edges}")

        if self.group_name is not None:
            self.group_map = {self.group_name: [f"leo_{i}" for i in range(self.n_leo)]}

        self._build_specs()

    def _build_specs(self) -> None:
        obs_fields: Dict[str, Any] = {
            "observation": Unbounded(
                shape=(self.n_leo, self.node_feature_dim),
                dtype=torch.float32,
                device=self.device,
            )
        }
        if self.include_graph_specs:
            obs_fields.update(
                {
                    "edge_index": Unbounded(
                        shape=(2, self.graph_max_edges),
                        dtype=torch.int64,
                        device=self.device,
                    ),
                    "edge_attr": Unbounded(
                        shape=(self.graph_max_edges, self.edge_feature_dim),
                        dtype=torch.float32,
                        device=self.device,
                    ),
                    "edge_mask": Unbounded(
                        shape=(self.graph_max_edges,),
                        dtype=torch.bool,
                        device=self.device,
                    ),
                    "edge_count": Unbounded(
                        shape=(1,),
                        dtype=torch.int64,
                        device=self.device,
                    ),
                }
            )

        obs_composite = Composite(
            **obs_fields,
            shape=torch.Size([]),
            device=self.device,
        )
        if self.group_name is None:
            self.observation_spec = obs_composite
        else:
            self.observation_spec = Composite(
                **{
                    self.group_name: obs_composite,
                },
                shape=torch.Size([]),
                device=self.device,
            )
        self.state_spec = self.observation_spec.clone()

        action_struct = Composite(
            upper_action=Categorical(
                n=GeoLeoGroundEnv.N_UPPER_ACTIONS,
                shape=(self.n_leo,),
                dtype=torch.int64,
                device=self.device,
            ),
            lower_action=Bounded(
                low=torch.zeros((self.n_leo, GeoLeoGroundEnv.LOWER_ACTION_DIM), device=self.device, dtype=torch.float32),
                high=torch.ones((self.n_leo, GeoLeoGroundEnv.LOWER_ACTION_DIM), device=self.device, dtype=torch.float32),
                shape=(self.n_leo, GeoLeoGroundEnv.LOWER_ACTION_DIM),
                dtype=torch.float32,
                device=self.device,
            ),
            shape=torch.Size([]),
            device=self.device,
        )
        if self.group_name is None:
            self.action_spec = action_struct
        else:
            self.action_spec = Composite(
                **{
                    self.group_name: Composite(
                        action=action_struct,
                        shape=torch.Size([]),
                        device=self.device,
                    )
                },
                shape=torch.Size([]),
                device=self.device,
            )

        reward_struct = Unbounded(
            shape=(self.n_leo, 1),
            dtype=torch.float32,
            device=self.device,
        )
        if self.group_name is None:
            self.reward_spec = reward_struct
        else:
            self.reward_spec = Composite(
                **{
                    self.group_name: Composite(
                        reward=reward_struct,
                        shape=torch.Size([]),
                        device=self.device,
                    )
                },
                shape=torch.Size([]),
                device=self.device,
            )

    def _set_seed(self, seed: Optional[int]) -> Optional[int]:
        if seed is None:
            return None
        seed = int(seed)
        self.inner.cfg.seed = seed
        self.inner.generator.manual_seed(seed)
        return seed

    def _pack_observation(self, obs: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> Dict[str, torch.Tensor]:
        payload: Dict[str, torch.Tensor] = {
            "observation": obs.to(self.device, dtype=torch.float32),
        }
        if self.include_graph_specs:
            payload.update(self._pad_graph(edge_index=edge_index, edge_attr=edge_attr))
        return payload

    def _pad_graph(self, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> Dict[str, torch.Tensor]:
        edge_index = edge_index.to(self.device, dtype=torch.int64)
        edge_attr = edge_attr.to(self.device, dtype=torch.float32)

        edge_index_padded = torch.zeros((2, self.graph_max_edges), dtype=torch.int64, device=self.device)
        edge_attr_padded = torch.zeros((self.graph_max_edges, self.edge_feature_dim), dtype=torch.float32, device=self.device)
        edge_mask = torch.zeros((self.graph_max_edges,), dtype=torch.bool, device=self.device)

        n_edges = int(min(edge_index.shape[1], self.graph_max_edges))
        if n_edges > 0:
            edge_index_padded[:, :n_edges] = edge_index[:, :n_edges]
            edge_attr_padded[:n_edges] = edge_attr[:n_edges, : self.edge_feature_dim]
            edge_mask[:n_edges] = True
        return {
            "edge_index": edge_index_padded,
            "edge_attr": edge_attr_padded,
            "edge_mask": edge_mask,
            "edge_count": torch.tensor([n_edges], dtype=torch.int64, device=self.device),
        }

    def _format_output(self, payload: Dict[str, torch.Tensor], *, done: torch.Tensor, reward: torch.Tensor | None) -> TensorDict:
        out: Dict[str, Any] = {
            "done": done,
            "terminated": done.clone(),
        }
        if self.group_name is None:
            out.update(payload)
            if reward is not None:
                out["reward"] = reward
        else:
            group_payload: Dict[str, Any] = dict(payload)
            if reward is not None:
                group_payload["reward"] = reward
            out[self.group_name] = group_payload
        return TensorDict(out, batch_size=self.batch_size, device=self.device)

    def _extract_action(self, tensordict: TensorDict) -> tuple[torch.Tensor, torch.Tensor]:
        if self.group_name is None:
            upper = tensordict.get("upper_action")
            lower = tensordict.get("lower_action")
        else:
            upper = tensordict.get((self.group_name, "action", "upper_action"))
            lower = tensordict.get((self.group_name, "action", "lower_action"))
        return (
            upper.to(device=self.device, dtype=torch.long).view(self.n_leo),
            lower.to(device=self.device, dtype=torch.float32).view(self.n_leo, GeoLeoGroundEnv.LOWER_ACTION_DIM),
        )

    def _reset(self, tensordict: Optional[TensorDict] = None) -> TensorDict:
        obs, edge_index, edge_attr = self.inner.reset()
        done = torch.zeros((1,), dtype=torch.bool, device=self.device)
        payload = self._pack_observation(obs, edge_index, edge_attr)
        return self._format_output(payload=payload, done=done, reward=None)

    def _step(self, tensordict: TensorDict) -> TensorDict:
        upper, lower = self._extract_action(tensordict)
        step = self.inner.step(upper_action=upper, lower_action=lower)
        done = torch.tensor([bool(step.done)], dtype=torch.bool, device=self.device)
        reward = step.upper_reward.to(self.device, dtype=torch.float32).unsqueeze(-1)
        payload = self._pack_observation(step.obs, step.edge_index, step.edge_attr)
        return self._format_output(payload=payload, done=done, reward=reward)


class TriSatFlowBenchMARLEnv(TriSatFlowTorchRLEnv):
    """BenchMARL-oriented grouped-key wrapper."""

    def __init__(
        self,
        scenario: Optional[ScenarioConfig] = None,
        reward_weights: Optional[RewardWeights] = None,
        *,
        device: str | torch.device = "cpu",
        group_name: str = "leo",
        include_graph_specs: bool = True,
        graph_max_edges: int | None = None,
    ) -> None:
        super().__init__(
            scenario=scenario,
            reward_weights=reward_weights,
            device=device,
            group_name=group_name,
            include_graph_specs=include_graph_specs,
            graph_max_edges=graph_max_edges,
        )
