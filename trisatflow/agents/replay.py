from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

import random
import torch


@dataclass
class RolloutBuffer:
    obs: List[torch.Tensor] = field(default_factory=list)
    edge_index: List[torch.Tensor] = field(default_factory=list)
    edge_attr: List[torch.Tensor] = field(default_factory=list)
    upper_action: List[torch.Tensor] = field(default_factory=list)
    log_prob: List[torch.Tensor] = field(default_factory=list)
    value: List[torch.Tensor] = field(default_factory=list)
    reward: List[torch.Tensor] = field(default_factory=list)
    done: List[bool] = field(default_factory=list)
    cost_prior: List[torch.Tensor] = field(default_factory=list)
    oracle_action: List[torch.Tensor] = field(default_factory=list)
    old_action_probs: List[torch.Tensor] = field(default_factory=list)
    scenario_phase: List[List[str]] = field(default_factory=list)
    task_type: List[List[str]] = field(default_factory=list)
    step_index: List[int] = field(default_factory=list)
    extras: List[Dict[str, Any]] = field(default_factory=list)

    def clear(self) -> None:
        self.obs.clear(); self.edge_index.clear(); self.edge_attr.clear(); self.upper_action.clear()
        self.log_prob.clear(); self.value.clear(); self.reward.clear(); self.done.clear()
        self.cost_prior.clear(); self.oracle_action.clear(); self.old_action_probs.clear()
        self.scenario_phase.clear(); self.task_type.clear(); self.step_index.clear(); self.extras.clear()

    def __len__(self) -> int:
        return len(self.obs)


class ReplayBuffer:
    def __init__(self, capacity: int = 20000):
        self.capacity = capacity
        self.storage: List[Dict[str, torch.Tensor]] = []
        self.pos = 0

    def add(self, **transition: torch.Tensor) -> None:
        clean = {k: v.detach().cpu() if torch.is_tensor(v) else v for k, v in transition.items()}
        if len(self.storage) < self.capacity:
            self.storage.append(clean)
        else:
            self.storage[self.pos] = clean
            self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size: int, device: torch.device) -> Dict[str, torch.Tensor]:
        batch = random.sample(self.storage, batch_size)
        out: Dict[str, torch.Tensor] = {}
        for key in batch[0].keys():
            values = [item[key] for item in batch]
            if torch.is_tensor(values[0]):
                out[key] = torch.stack(values, dim=0).to(device)
            else:
                out[key] = values
        return out

    def __len__(self) -> int:
        return len(self.storage)
