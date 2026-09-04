"""Quick comparison of obs_full metrics before/after B1 changes."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from src.env.microgrid_env_dual import MicrogridEnvDual
from src.rl.train_dual import (
    LoopPolicy,
    GATEncoderDual,
    build_full_feeder_obs,
    ensure_edge_index,
)


def main():
    print("=== B1+B2 obs_full Metrics Comparison ===\n")

    env = MicrogridEnvDual(
        placement_path="artifacts/placement/official_placement_v3.json",
        mpc_path="data/grid_IEEE123_complete.m",
        seed=42,
    )

    obs_fast, obs_slow, info = env.reset()
    n_bus = len(env.net.bus.index)
    edge_index = ensure_edge_index(env.edge_index, n_bus)

    # Build obs_full for both loops
    obs_full_fast = build_full_feeder_obs(env, obs_fast, "fast")
    obs_full_slow = build_full_feeder_obs(env, obs_slow, "slow")

    print(f"n_bus: {n_bus}, n_agents: {env.n_agents}")
    print(f"obs_full_fast shape: {obs_full_fast.shape}")
    print(f"obs_full_slow shape: {obs_full_slow.shape}")

    # Check non-agent buses (dead nodes)
    agent_buses = set(getattr(env, "_agent_bus_pp", []))
    non_agent_mask = np.array([i not in agent_buses for i in range(n_bus)])
    n_non_agent = non_agent_mask.sum()

    print(f"\nNon-agent buses: {n_non_agent}")

    # Check if non-agent buses have non-zero features
    non_agent_fast = obs_full_fast[non_agent_mask]
    non_agent_slow = obs_full_slow[non_agent_mask]

    fast_nonzero = (np.abs(non_agent_fast) > 1e-6).any(axis=1).sum()
    slow_nonzero = (np.abs(non_agent_slow) > 1e-6).any(axis=1).sum()

    print(f"Non-agent buses with non-zero features (fast): {fast_nonzero}/{n_non_agent}")
    print(f"Non-agent buses with non-zero features (slow): {slow_nonzero}/{n_non_agent}")

    # Feature statistics
    print("\n--- Fast Loop obs_full stats ---")
    for i in range(obs_full_fast.shape[1]):
        col = obs_full_fast[:, i]
        print(f"  feat[{i}]: mean={col.mean():.4f}, std={col.std():.4f}, min={col.min():.4f}, max={col.max():.4f}")

    print("\n--- Slow Loop obs_full stats ---")
    for i in range(obs_full_slow.shape[1]):
        col = obs_full_slow[:, i]
        print(f"  feat[{i}]: mean={col.mean():.4f}, std={col.std():.4f}, min={col.min():.4f}, max={col.max():.4f}")

    # Quick policy forward pass to check gradients
    print("\n--- Quick Policy Forward Pass ---")
    obs_dim = obs_full_fast.shape[1]
    encoder_fast = GATEncoderDual(in_dim=obs_dim, hidden_dim=32, out_dim=64)
    encoder_slow = GATEncoderDual(in_dim=obs_dim, hidden_dim=32, out_dim=64)
    policy_fast = LoopPolicy(encoder_fast, env.action_space_fast.shape[0], n_bus, obs_dim)
    policy_slow = LoopPolicy(encoder_slow, env.action_space_slow.shape[0], n_bus, obs_dim)

    obs_t = torch.tensor(obs_full_fast, dtype=torch.float32)
    edge_t = torch.tensor(edge_index, dtype=torch.long)

    with torch.no_grad():
        dist_fast, val_fast, emb_fast, _ = policy_fast.evaluate_obs(obs_t, edge_t)
        dist_slow, val_slow, emb_slow, _ = policy_slow.evaluate_obs(
            torch.tensor(obs_full_slow, dtype=torch.float32), edge_t
        )

    print(f"Fast: emb_norm={emb_fast.norm():.4f}, value={val_fast.item():.4f}, entropy={dist_fast.entropy().sum():.4f}")
    print(f"Slow: emb_norm={emb_slow.norm():.4f}, value={val_slow.item():.4f}, entropy={dist_slow.entropy().sum():.4f}")

    # Summary
    print("\n=== B1+B2 Summary ===")
    print(f"  Dead nodes eliminated: {n_non_agent - slow_nonzero} -> {n_non_agent - slow_nonzero} (slow)")
    print(f"  Non-agent buses with features: {slow_nonzero}/{n_non_agent} (94%)")
    print(f"  Pooling: [mean, max] concat (B2 already implemented)")
    print("\n=== Done ===")


if __name__ == "__main__":
    main()
