"""MLP encoder that ignores graph structure (ablation baseline for GraphSAGE)."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class MLPEncoder(nn.Module):
    """MLP encoder that processes node features without message passing.

    This encoder has the same interface as GraphSAGEEncoder but ignores
    edge_index entirely. Used as ablation baseline to isolate the
    contribution of graph message passing.
    """

    def __init__(self, in_dim: int = 6, hidden_dim: int = 128, out_dim: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )
        self.out_dim = out_dim

    @staticmethod
    def _to_tensor(x: np.ndarray | torch.Tensor, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        if isinstance(x, torch.Tensor):
            return x.to(dtype=dtype)
        return torch.as_tensor(x, dtype=dtype)

    def encode(self, x: np.ndarray | torch.Tensor, edge_index: np.ndarray | torch.Tensor | None = None) -> torch.Tensor:
        """Encode node features. edge_index is ignored (no message passing).

        Args:
            x: Node features with shape (N, F)
            edge_index: Ignored - kept for API compatibility with GraphSAGEEncoder

        Returns:
            Node embeddings with shape (N, out_dim)
        """
        x_t = self._to_tensor(x, dtype=torch.float32)
        if x_t.ndim != 2:
            raise ValueError("encode expects x with shape (N, F)")
        return self.net(x_t)

    def encode_batch(
        self,
        x_batch: np.ndarray | torch.Tensor,
        edge_index: np.ndarray | torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode batched node features. edge_index is ignored.

        Args:
            x_batch: Batched node features with shape (B, N, F)
            edge_index: Ignored - kept for API compatibility

        Returns:
            Batched node embeddings with shape (B, N, out_dim)
        """
        x_t = self._to_tensor(x_batch, dtype=torch.float32)
        if x_t.ndim != 3:
            raise ValueError("encode_batch expects x_batch with shape (B, N, F)")
        batch_size, n_nodes, _ = x_t.shape
        x_flat = x_t.reshape(batch_size * n_nodes, -1)
        out_flat = self.net(x_flat)
        return out_flat.reshape(batch_size, n_nodes, self.out_dim)

    def forward(self, x: np.ndarray | torch.Tensor, edge_index: np.ndarray | torch.Tensor | None = None) -> torch.Tensor:
        """Forward pass - dispatches to encode or encode_batch based on input shape."""
        x_t = self._to_tensor(x, dtype=torch.float32)
        if x_t.ndim == 2:
            return self.encode(x_t, edge_index)
        if x_t.ndim == 3:
            return self.encode_batch(x_t, edge_index)
        raise ValueError("x must have shape (N, F) or (B, N, F)")
