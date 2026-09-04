from __future__ import annotations

import copy

import numpy as np
import torch

from src.env.IEEE123bus import build_ieee123_net
from src.environment.topology_manager import build_edge_index
from src.layer2_control.actor_critic import ActorCritic, ActorCriticConfig
from src.layer2_control.graph_sage_encoder import GraphSAGEEncoder
from src.layer2_control.mappo_policy import MappoPolicy


def _node_features(net) -> np.ndarray:
    n = len(net.bus.index)
    idx = np.arange(n, dtype=np.float32)
    x = np.stack(
        [
            np.ones(n, dtype=np.float32),
            idx / max(float(n - 1), 1.0),
            np.sin(idx / 5.0).astype(np.float32),
            np.cos(idx / 7.0).astype(np.float32),
            (idx % 3) / 2.0,
            (idx % 5) / 4.0,
        ],
        axis=1,
    )
    return x.astype(np.float32)


def test_graphsage_forward_shapes_and_finite() -> None:
    net = build_ieee123_net(mode="matpower", balanced=True, convert_switches=True, slack_zones=None, source_mode="publish")
    x = _node_features(net)
    edge_index = build_edge_index(net)

    encoder = GraphSAGEEncoder(in_dim=6, hidden_dim=16, out_dim=32)
    actor_critic = ActorCritic(ActorCriticConfig(local_state_dim=6, graph_emb_dim=32, global_state_dim=7, action_dim=2))
    policy = MappoPolicy(encoder=encoder, actor_critic=actor_critic)

    embeddings = policy.encode(x, edge_index)
    assert embeddings.shape == (x.shape[0], 32)
    assert torch.isfinite(embeddings).all()


def test_graphsage_backward_produces_gradients() -> None:
    net = build_ieee123_net(mode="matpower", balanced=True, convert_switches=True, slack_zones=None, source_mode="publish")
    x = _node_features(net)
    edge_index = build_edge_index(net)

    encoder = GraphSAGEEncoder(in_dim=6, hidden_dim=16, out_dim=32)
    embeddings = encoder.encode(x, edge_index)
    loss = embeddings.pow(2).mean()
    loss.backward()

    grad_norm = 0.0
    for p in encoder.parameters():
        if p.grad is not None:
            grad_norm += float(p.grad.norm().item())
    assert grad_norm > 0.0


def test_graphsage_embedding_changes_with_topology_change() -> None:
    net = build_ieee123_net(mode="matpower", balanced=True, convert_switches=True, slack_zones=None, source_mode="publish")
    x = _node_features(net)

    edge_a = build_edge_index(net)

    net_mut = copy.deepcopy(net)
    if not net_mut.switch.empty:
        sw_idx = int(net_mut.switch.index[0])
        current = bool(net_mut.switch.at[sw_idx, "closed"])
        net_mut.switch.at[sw_idx, "closed"] = not current
    elif not net_mut.line.empty:
        line_idx = int(net_mut.line.index[0])
        current = bool(net_mut.line.at[line_idx, "in_service"])
        net_mut.line.at[line_idx, "in_service"] = not current

    edge_b = build_edge_index(net_mut)

    encoder = GraphSAGEEncoder(in_dim=6, hidden_dim=16, out_dim=32)
    emb_a = encoder.encode(x, edge_a)
    emb_b = encoder.encode(x, edge_b)

    diff = torch.norm(emb_a - emb_b).item()
    assert diff > 1e-3
