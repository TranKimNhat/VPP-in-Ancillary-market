from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class _GraphSAGELayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, activation: nn.Module | None = None) -> None:
        super().__init__()
        self.w_self = nn.Linear(in_dim, out_dim)
        self.w_neigh = nn.Linear(in_dim, out_dim)
        self.activation = activation

    @staticmethod
    def _mean_aggregate(x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        n_nodes = x.shape[0]
        src = edge_index[0]
        dst = edge_index[1]

        neigh_sum = torch.zeros_like(x)
        neigh_sum.index_add_(0, dst, x[src])

        degree = torch.zeros(n_nodes, device=x.device, dtype=x.dtype)
        ones = torch.ones(dst.shape[0], device=x.device, dtype=x.dtype)
        degree.index_add_(0, dst, ones)
        degree = degree.clamp_min(1.0).unsqueeze(-1)

        return neigh_sum / degree

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        neigh_mean = self._mean_aggregate(x, edge_index)
        out = self.w_self(x) + self.w_neigh(neigh_mean)
        if self.activation is not None:
            out = self.activation(out)
        return out


class GraphSAGEEncoder(nn.Module):
    def __init__(self, in_dim: int = 6, hidden_dim: int = 64, out_dim: int = 64) -> None:
        super().__init__()
        self.layer1 = _GraphSAGELayer(in_dim=in_dim, out_dim=hidden_dim, activation=nn.ReLU())
        self.layer2 = _GraphSAGELayer(in_dim=hidden_dim, out_dim=out_dim, activation=None)

    @staticmethod
    def _to_tensor(x: np.ndarray | torch.Tensor, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        if isinstance(x, torch.Tensor):
            return x.to(dtype=dtype)
        return torch.as_tensor(x, dtype=dtype)

    @staticmethod
    def _normalize_edge_index(
        edge_index: np.ndarray | torch.Tensor,
        n_nodes: int,
        device: torch.device,
    ) -> torch.Tensor:
        ei = GraphSAGEEncoder._to_tensor(edge_index, dtype=torch.long).to(device=device)
        if ei.ndim != 2 or ei.shape[0] != 2:
            raise ValueError("edge_index must have shape (2, E)")
        if ei.numel() > 0:
            if int(ei.min()) < 0 or int(ei.max()) >= int(n_nodes):
                raise ValueError("edge_index contains out-of-range node indices")
        return ei

    def encode(self, x: np.ndarray | torch.Tensor, edge_index: np.ndarray | torch.Tensor) -> torch.Tensor:
        x_t = self._to_tensor(x, dtype=torch.float32)
        if x_t.ndim != 2:
            raise ValueError("encode expects x with shape (N, F)")

        edge_t = self._normalize_edge_index(edge_index, n_nodes=x_t.shape[0], device=x_t.device)

        h = self.layer1(x_t, edge_t)
        h = self.layer2(h, edge_t)
        return h

    def encode_batch(
        self,
        x_batch: np.ndarray | torch.Tensor,
        edge_index: np.ndarray | torch.Tensor,
    ) -> torch.Tensor:
        x_t = self._to_tensor(x_batch, dtype=torch.float32)
        if x_t.ndim != 3:
            raise ValueError("encode_batch expects x_batch with shape (B, N, F)")

        batch_size, n_nodes, _ = x_t.shape
        edge_t = self._normalize_edge_index(edge_index, n_nodes=n_nodes, device=x_t.device)

        out = torch.empty(
            (batch_size, n_nodes, self.layer2.w_self.out_features),
            device=x_t.device,
            dtype=x_t.dtype,
        )
        for b in range(batch_size):
            h = self.layer1(x_t[b], edge_t)
            out[b] = self.layer2(h, edge_t)
        return out

    def forward(self, x: np.ndarray | torch.Tensor, edge_index: np.ndarray | torch.Tensor) -> torch.Tensor:
        x_t = self._to_tensor(x, dtype=torch.float32)
        if x_t.ndim == 2:
            return self.encode(x_t, edge_index)
        if x_t.ndim == 3:
            return self.encode_batch(x_t, edge_index)
        raise ValueError("x must have shape (N, F) or (B, N, F)")
