"""Quick evaluation of trained AM-MAPPO agent."""
import sys
sys.path.insert(0, r"C:\Users\admin\Desktop\VPP in Ancillary market in 100% renewable islanded microgrid - Copy")

import torch
import numpy as np
from pathlib import Path

from src.env.microgrid_env_dual import MicrogridEnvDual
from src.rl.train_am_mappo import GAMAPPOAgent, ensure_edge_index, build_am_full_feeder_obs

def evaluate(checkpoint_path: Path, n_episodes: int = 20, seed: int = 42):
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Load environment (same as train_am_mappo)
    print("Loading environment...")
    env = MicrogridEnvDual(
        placement_path=Path("artifacts/placement/official_placement_v3.json"),
        mpc_path=Path("data/grid_IEEE123_complete.m"),
        precomputed_dir=Path("data"),
        topology_cache_path=Path("data/tie_switch_cache.pkl"),
    )

    # Get dimensions from env (same as train_am_mappo)
    sample_obs_fast, _, _ = env.reset()
    sample_obs_full = build_am_full_feeder_obs(env, sample_obs_fast)
    n_bus = int(sample_obs_full.shape[0])
    obs_feat = int(sample_obs_full.shape[1])
    n_agents = env.n_agents
    agent_bus_indices = np.asarray(getattr(env, "_agent_bus_pp", np.arange(n_agents)), dtype=np.int64)
    agent_bus_indices = np.clip(agent_bus_indices, 0, max(n_bus - 1, 0))

    print(f"Environment: {n_agents} agents, {n_bus} buses, obs_feat={obs_feat}")

    # Create agent with same architecture as training (from checkpoint args)
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    args = ckpt.get("args", {})
    embed_dim = args.get("embed_dim", 128)
    hidden_dim = args.get("hidden_dim", 128)

    agent = GAMAPPOAgent(
        obs_feat=obs_feat,
        n_agents=n_agents,
        n_bus=n_bus,
        agent_bus_indices=agent_bus_indices,
        action_dim_per_agent=1,  # AM uses single action per agent
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
    )

    # Load checkpoint weights
    agent.load_state_dict(ckpt["agent_state_dict"])
    agent.eval()

    # Build edge index for graph
    edge_index = ensure_edge_index(env.edge_index, n_nodes=n_bus)

    # Evaluation metrics
    all_rewards = []
    all_nadirs = []
    all_delta_f = []
    all_rocof = []
    all_violations = []

    print(f"\nEvaluating {n_episodes} episodes...")
    for ep in range(n_episodes):
        obs_fast, _, _ = env.reset()
        obs_full = build_am_full_feeder_obs(env, obs_fast)
        ep_reward = 0.0
        ep_nadir = 50.0
        ep_delta_f = []
        ep_rocof = []
        ep_violations = 0

        done = False
        step = 0
        while not done and step < 300:
            # Prepare inputs for agent
            with torch.no_grad():
                actions = agent.act_deterministic(obs_full, edge_index)

            # Build action_fast: 41 agent actions + 3 VPP droop coefficients
            action_fast = np.zeros(44, dtype=np.float32)
            action_fast[:41] = actions.flatten()[:41]  # delta_p for agents
            action_fast[41:44] = 0.0  # k_droop_vpp (zero = use default)

            obs_fast, reward, done, truncated, info = env.step_fast(action_fast)
            obs_full = build_am_full_feeder_obs(env, obs_fast)

            ep_reward += reward

            # Extract metrics from info
            if "nadir" in info:
                ep_nadir = min(ep_nadir, info["nadir"])
            if "delta_f" in info:
                ep_delta_f.append(abs(info["delta_f"]))
            if "rocof" in info:
                ep_rocof.append(abs(info["rocof"]))
            if "voltage_violations" in info:
                ep_violations += info["voltage_violations"]

            step += 1
            done = done or truncated

        all_rewards.append(ep_reward)
        all_nadirs.append(ep_nadir)
        all_delta_f.append(np.mean(ep_delta_f) if ep_delta_f else 0.0)
        all_rocof.append(np.mean(ep_rocof) if ep_rocof else 0.0)
        all_violations.append(ep_violations)

        print(f"Ep {ep+1:3d} | R={ep_reward:8.2f} | Nadir={ep_nadir:.3f} | dF={all_delta_f[-1]:.4f} | Vio={ep_violations}")

    # Summary
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    print(f"Episodes: {n_episodes}")
    print(f"Reward:     {np.mean(all_rewards):8.2f} +/- {np.std(all_rewards):.2f}")
    print(f"Nadir:      {np.mean(all_nadirs):8.3f} +/- {np.std(all_nadirs):.3f} Hz")
    print(f"Delta_f:    {np.mean(all_delta_f):8.4f} +/- {np.std(all_delta_f):.4f} Hz")
    print(f"Violations: {np.mean(all_violations):8.2f} +/- {np.std(all_violations):.2f}")
    print(f"Min Nadir:  {np.min(all_nadirs):.3f} Hz")
    print(f"Max Nadir:  {np.max(all_nadirs):.3f} Hz")

    return {
        "reward_mean": np.mean(all_rewards),
        "reward_std": np.std(all_rewards),
        "nadir_mean": np.mean(all_nadirs),
        "nadir_min": np.min(all_nadirs),
        "violations_mean": np.mean(all_violations),
    }


if __name__ == "__main__":
    ckpt = Path("artifacts/checkpoints_am_mappo/phase_F_final.pt")
    evaluate(ckpt, n_episodes=20)
