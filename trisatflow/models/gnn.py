from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class GraphSAGELayer(nn.Module):
    """Dependency-light GraphSAGE layer with edge-feature gates.

    It intentionally avoids torch_scatter / torch_geometric so the prototype can
    run in a bare PyTorch environment. On a GPU server, this can be replaced by
    BenchMARL's torch_geometric-backed GNN model with the same inputs.
    """

    def __init__(self, in_dim: int, edge_dim: int, out_dim: int):
        super().__init__()
        self.self_proj = nn.Linear(in_dim, out_dim)
        self.neigh_proj = nn.Linear(in_dim, out_dim)
        self.edge_gate = nn.Sequential(nn.Linear(edge_dim, out_dim), nn.Sigmoid())
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        n = x.shape[0]
        if edge_index.numel() == 0:
            return self.norm(self.self_proj(x))
        src, dst = edge_index
        msg = self.neigh_proj(x[src])
        if edge_attr is not None and edge_attr.numel() > 0:
            msg = msg * self.edge_gate(edge_attr)
        agg = torch.zeros(n, msg.shape[-1], device=x.device, dtype=x.dtype)
        agg.index_add_(0, dst, msg)
        deg = torch.zeros(n, 1, device=x.device, dtype=x.dtype)
        deg.index_add_(0, dst, torch.ones_like(dst, dtype=x.dtype).unsqueeze(-1))
        agg = agg / deg.clamp_min(1.0)
        return F.relu(self.norm(self.self_proj(x) + agg))


class TopologyEncoder(nn.Module):
    """Dynamic topology encoder used by both upper and lower policies."""

    def __init__(self, node_dim: int, edge_dim: int, hidden_dim: int = 64, n_layers: int = 2):
        super().__init__()
        layers = []
        in_dim = node_dim
        for _ in range(n_layers):
            layers.append(GraphSAGELayer(in_dim, edge_dim, hidden_dim))
            in_dim = hidden_dim
        self.layers = nn.ModuleList(layers)
        self.out_norm = nn.LayerNorm(hidden_dim)

    temporal_enabled: bool = False
    history_len: int = 1

    def forward(
        self,
        obs: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        update_state: bool = True,
    ) -> torch.Tensor:
        x = obs
        for layer in self.layers:
            x = layer(x, edge_index, edge_attr)
        return self.out_norm(x)

    def reset_temporal_state(self) -> None:
        return None

    def detach_temporal_state(self) -> None:
        return None

class FeatureEncoder(nn.Module):
    """A no-message-passing encoder used for the w/o GNN ablation.

    It preserves the same call signature as TopologyEncoder so the trainer can
    toggle graph reasoning without changing upper/lower policy code.
    """

    def __init__(self, node_dim: int, edge_dim: int, hidden_dim: int = 64, n_layers: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
        )

    temporal_enabled: bool = False
    history_len: int = 1

    def forward(
        self,
        obs: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        update_state: bool = True,
    ) -> torch.Tensor:
        return self.net(obs)

    def reset_temporal_state(self) -> None:
        return None

    def detach_temporal_state(self) -> None:
        return None


class TemporalTopologyEncoder(nn.Module):
    """Wraps a per-step topology encoder with temporal aggregation via GRU.

    The wrapped encoder keeps the old `obs/edge_index/edge_attr -> embedding`
    API. Temporal state is managed per episode through `reset_temporal_state`.
    """

    temporal_enabled: bool = True

    def __init__(
        self,
        base_encoder: TopologyEncoder | FeatureEncoder,
        *,
        base_dim: int,
        history_len: int = 4,
        temporal_hidden_dim: int = 128,
    ):
        super().__init__()
        self.base_encoder = base_encoder
        self.history_len = max(1, int(history_len))
        self.temporal_hidden_dim = max(1, int(temporal_hidden_dim))
        self.gru = nn.GRU(input_size=int(base_dim), hidden_size=self.temporal_hidden_dim, num_layers=1, batch_first=False)
        self.out_proj = nn.Linear(self.temporal_hidden_dim, int(base_dim))
        self.out_norm = nn.LayerNorm(int(base_dim))
        self._history: list[torch.Tensor] = []
        self._hidden_state: torch.Tensor | None = None

    def _base_forward(self, obs: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        try:
            return self.base_encoder(obs, edge_index, edge_attr, update_state=False)
        except TypeError:
            return self.base_encoder(obs, edge_index, edge_attr)

    def reset_temporal_state(self) -> None:
        self._history.clear()
        self._hidden_state = None

    def detach_temporal_state(self) -> None:
        self._history = [item.detach() for item in self._history]
        if self._hidden_state is not None:
            self._hidden_state = self._hidden_state.detach()

    def temporal_state(self) -> torch.Tensor | None:
        if self._hidden_state is None:
            return None
        return self._hidden_state.detach().clone()

    def forward(
        self,
        obs: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        update_state: bool = True,
    ) -> torch.Tensor:
        base = self._base_forward(obs, edge_index, edge_attr)
        past = self._history[-(self.history_len - 1) :] if self.history_len > 1 else []
        seq_items = [item.to(device=base.device, dtype=base.dtype) for item in past] + [base]
        seq = torch.stack(seq_items, dim=0)  # [L, N, D]
        out, h_n = self.gru(seq)
        embed = self.out_norm(self.out_proj(out[-1]))
        if update_state:
            self._history.append(base.detach())
            if len(self._history) > self.history_len:
                self._history = self._history[-self.history_len :]
            self._hidden_state = h_n.detach()
        return embed
