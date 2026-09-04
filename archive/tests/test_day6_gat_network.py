from __future__ import annotations

import numpy as np
import torch

from src.rl.networks import GATEncoder, TypeConditionedActor, VPPCritic


def test_day6_gat_network_gate() -> None:
    device = "cpu"
    encoder = GATEncoder().to(device)
    actor = TypeConditionedActor().to(device)
    critic = VPPCritic().to(device)

    x_full = torch.randn(123, 24, device=device)
    edge_index = torch.randint(0, 123, (2, 200), device=device)
    agent_obs = torch.randn(30, 24, device=device)
    global_state = torch.randn(10, device=device)

    embeddings = encoder(x_full, edge_index)
    agent_types = [
        "EVCS_PV",
        "EVCS_PV",
        "EVCS_PV",
        "EVCS_BESS",
        "EVCS_BESS",
        "EVCS_BESS",
        "EVCS_V2G",
        "EVCS_V2G",
        "EVCS_V2G",
        "EVCS_V2G",
        "EVCS_V2G",
        "EVCS_V2G",
        "DPV",
        "DPV",
        "DPV",
    ] * 2

    actor_out = actor(embeddings[:30], agent_obs, agent_types)
    actions = actor_out.actions
    log_probs = actor_out.log_probs
    values = critic(embeddings[:41], global_state)

    assert actions.shape == (30, 2)
    assert log_probs.shape == (30,)
    assert values.shape == (3,)

    loss = -log_probs.mean() + values.sum()
    loss.backward()
    assert any(p.grad is not None for p in encoder.parameters())

    flat = TypeConditionedActor.get_flat_actions(actions, agent_types)
    assert flat.shape == (1, 48)

    print(f"GATE DAY 6: PASS (device={device})")
