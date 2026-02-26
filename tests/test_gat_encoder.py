from __future__ import annotations

import numpy as np
import torch

from src.layer2_control.gat_encoder import GATEncoder, GATEncoderConfig, GraphObservation


def test_gat_encoder_output_shape() -> None:
    cfg = GATEncoderConfig(in_dim=6, hidden_dim=8, output_dim=16, heads_l1=2)
    encoder = GATEncoder(cfg)

    node_features = np.random.rand(10, 6).astype(np.float32)
    adjacency = np.eye(10, dtype=np.float32)
    obs = GraphObservation(node_features=node_features, adjacency=adjacency)

    out = encoder.encode(obs)
    assert isinstance(out, torch.Tensor)
    assert out.shape == (10, 16)
    assert torch.isfinite(out).all()
