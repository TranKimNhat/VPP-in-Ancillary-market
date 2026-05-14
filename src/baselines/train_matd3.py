"""Training script for MATD3 baseline."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch

from src.baselines.matd3 import MATD3Agent, MATD3Config
from src.env.microgrid_env_dual import MicrogridEnvDual


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_obs_from_env(env: MicrogridEnvDual, obs_fast: np.ndarray) -> np.ndarray:
    """Build per-agent observations from environment."""
    n_agents = env.n_agents
    obs_dim = 24

    obs = np.zeros((n_agents, obs_dim), dtype=np.float32)
    freq_state = env.freq_dyn.get_state()

    for i in range(n_agents):
        agent_obs = np.zeros(obs_dim, dtype=np.float32)
        agent_obs[0] = freq_state.delta_f_hz
        agent_obs[1] = freq_state.rocof_hz_s
        if i < len(obs_fast):
            agent_obs[2:min(obs_dim, 2 + len(obs_fast[i]))] = obs_fast[i][:obs_dim - 2]
        obs[i] = agent_obs

    return obs


def train_matd3(
    placement_path: Path,
    mpc_path: Path,
    output_dir: Path,
    n_episodes: int = 500,
    episode_length: int = 200,
    warmup_steps: int = 1000,
    update_freq: int = 1,
    eval_interval: int = 50,
    seed: int = 42,
) -> dict:
    set_seed(seed)

    env = MicrogridEnvDual(
        placement_path=str(placement_path),
        mpc_path=str(mpc_path),
        seed=seed,
    )

    config = MATD3Config(
        obs_dim=24,
        action_dim=2,
        n_agents=env.n_agents,
        buffer_size=100_000,
        batch_size=256,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    agent = MATD3Agent(config, device=device)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    total_steps = 0
    best_reward = float("-inf")
    rewards_history = []

    for episode in range(1, n_episodes + 1):
        obs_fast, _, _ = env.reset()
        obs = build_obs_from_env(env, obs_fast)
        episode_reward = 0.0

        for step in range(episode_length):
            explore = total_steps >= warmup_steps
            actions = agent.act(obs, explore=explore)

            actions_flat = actions[:, 0]
            n_vpps = len(env._vpp_droop_agents)
            full_action = np.zeros(env.n_agents + n_vpps, dtype=np.float32)
            full_action[:env.n_agents] = actions_flat

            obs_fast_next, reward, done, _, info = env.step_fast(full_action)
            next_obs = build_obs_from_env(env, obs_fast_next)

            rewards = np.full(env.n_agents, reward, dtype=np.float32)
            agent.buffer.add(obs, actions, rewards, next_obs, done)

            obs = next_obs
            episode_reward += reward
            total_steps += 1

            if total_steps >= warmup_steps and total_steps % update_freq == 0:
                agent.update()

            if done:
                break

        rewards_history.append(episode_reward)

        if episode % eval_interval == 0:
            mean_reward = np.mean(rewards_history[-eval_interval:])
            print(f"Episode {episode}/{n_episodes} | Mean reward: {mean_reward:.4f} | Noise: {agent.exploration_noise:.4f}")

            if mean_reward > best_reward:
                best_reward = mean_reward
                agent.save(output_dir / "matd3_best.pt")

        if episode % 100 == 0:
            agent.save(output_dir / f"matd3_ep{episode}.pt")

    agent.save(output_dir / "matd3_final.pt")

    result = {
        "seed": seed,
        "n_episodes": n_episodes,
        "best_reward": float(best_reward),
        "final_reward": float(np.mean(rewards_history[-50:])),
        "checkpoint": str(output_dir / "matd3_final.pt"),
    }

    with open(output_dir / "train_result.json", "w") as f:
        json.dump(result, f, indent=2)

    print(f"Training complete. Best reward: {best_reward:.4f}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Train MATD3 baseline")
    parser.add_argument("--placement", type=Path, default=Path("artifacts/placement/official_placement_v3.json"))
    parser.add_argument("--mpc-path", type=Path, default=Path("data/grid_IEEE123_complete.m"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/checkpoints_matd3"))
    parser.add_argument("--n-episodes", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train_matd3(
        placement_path=args.placement,
        mpc_path=args.mpc_path,
        output_dir=args.output_dir,
        n_episodes=args.n_episodes,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
