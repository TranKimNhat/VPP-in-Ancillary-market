from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class GraphObservation:
    node_features: np.ndarray | torch.Tensor
    adjacency: np.ndarray | torch.Tensor


@dataclass(frozen=True)
class GATEncoderConfig:
    in_dim: int = 6
    hidden_dim: int = 32
    output_dim: int = 64
    heads_l1: int = 4
    dropout: float = 0.1
    add_self_loops: bool = True


class _DenseGATLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.heads = heads
        self.dropout = dropout

        self.proj = nn.Linear(in_dim, out_dim * heads, bias=False)
        self.attn_src = nn.Parameter(torch.empty(heads, out_dim))
        self.attn_dst = nn.Parameter(torch.empty(heads, out_dim))
        self.bias = nn.Parameter(torch.zeros(heads * out_dim))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.xavier_uniform_(self.attn_src)
        nn.init.xavier_uniform_(self.attn_dst)
        nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        n = x.shape[0]
        h = self.proj(x).view(n, self.heads, self.out_dim)

        e_src = (h * self.attn_src.unsqueeze(0)).sum(dim=-1)
        e_dst = (h * self.attn_dst.unsqueeze(0)).sum(dim=-1)

        logits = e_src.unsqueeze(1) + e_dst.unsqueeze(0)
        logits = F.leaky_relu(logits, negative_slope=0.2)

        mask = adjacency > 0
        very_neg = torch.full_like(logits, -1e9)
        logits = torch.where(mask.unsqueeze(-1), logits, very_neg)

        alpha = torch.softmax(logits, dim=1)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        out = torch.einsum("ijh,jhd->ihd", alpha, h).reshape(n, self.heads * self.out_dim)
        out = out + self.bias
        return out


class GATEncoder(nn.Module):
    """Trainable two-layer dense GAT encoder for power-grid graphs."""

    def __init__(self, config: GATEncoderConfig | None = None) -> None:
        super().__init__()
        self.config = config or GATEncoderConfig()

        self.gat1 = _DenseGATLayer(
            in_dim=self.config.in_dim,
            out_dim=self.config.hidden_dim,
            heads=self.config.heads_l1,
            dropout=self.config.dropout,
        )
        self.gat2 = _DenseGATLayer(
            in_dim=self.config.hidden_dim * self.config.heads_l1,
            out_dim=self.config.output_dim,
            heads=1,
            dropout=self.config.dropout,
        )

    @staticmethod
    def _to_tensor(x: np.ndarray | torch.Tensor) -> torch.Tensor:
        if isinstance(x, torch.Tensor):
            return x.float()
        return torch.tensor(np.asarray(x, dtype=np.float32), dtype=torch.float32)

    def encode(self, obs: GraphObservation) -> torch.Tensor:
        x = self._to_tensor(obs.node_features)
        a = self._to_tensor(obs.adjacency)

        if x.ndim != 2 or a.ndim != 2:
            raise ValueError("node_features and adjacency must be 2D arrays/tensors.")
        if a.shape[0] != a.shape[1] or a.shape[0] != x.shape[0]:
            raise ValueError("adjacency must be square and aligned with node_features.")

        if self.config.add_self_loops:
            a = a.clone()
            eye = torch.eye(a.shape[0], device=a.device, dtype=a.dtype)
            a = torch.maximum(a, eye)

        h1 = F.elu(self.gat1(x, a))
        h1 = F.dropout(h1, p=self.config.dropout, training=self.training)
        h2 = self.gat2(h1, a)
        return h2

    def forward(self, obs: GraphObservation) -> torch.Tensor:
        return self.encode(obs)
