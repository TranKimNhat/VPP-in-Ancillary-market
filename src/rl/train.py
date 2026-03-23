from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable

import numpy as np
import torch

from src.env.microgrid_env import MicrogridEnv
from src.env.make_env import make_vec_envs
from src.layer2_control.mappo_policy import MappoPolicy, MappoPolicyConfig, RolloutBuffer
from src.rl.networks import AGENT_TYPES, V2G_AGENT_INDICES, build_edge_index

N_AGENTS = 41
ACTION_DIM_FLAT = 73
ALPHA_P2P = 0.2


@dataclass(frozen=True)
class PhaseConfig:
    fixed_topology: bool
    freq_events: bool
    n_episodes: int
    learning_rate: float


PHASES: Dict[str, PhaseConfig] = {
    "A": PhaseConfig(fixed_topology=True, freq_events=False, n_episodes=2000, learning_rate=3e-4),
    "B": PhaseConfig(fixed_topology=False, freq_events=False, n_episodes=3000, learning_rate=3e-4),
    "C": PhaseConfig(fixed_topology=False, freq_events=True, n_episodes=10000, learning_rate=3e-4),
    "D": PhaseConfig(fixed_topology=False, freq_events=True, n_episodes=15000, learning_rate=3e-4),
    "E": PhaseConfig(fixed_topology=False, freq_events=True, n_episodes=20000, learning_rate=1e-4),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Day 12 MAPPO training loop")
    parser.add_argument("--phase", default="A", choices=sorted(PHASES.keys()))
    parser.add_argument("--n-episodes", type=int, default=None)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--seeds", type=int, nargs="*", default=[42])
    parser.add_argument("--placement", type=str, required=True)
    parser.add_argument("--precomputed-dir", type=str, required=True)
    parser.add_argument("--checkpoint-dir", type=str, required=True)
    parser.add_argument("--log-interval", type=int, default=100)
    return parser.parse_args()


def load_placement(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def build_vpp_of_agent(placement: dict) -> Dict[int, int]:
    vpp_of_agent: Dict[int, int] = {}
    for i, ev in enumerate(placement.get("evcs", [])):
        vpp_idx = int(str(ev["vpp"])[-1]) - 1
        vpp_of_agent[i] = vpp_idx
        vpp_of_agent[i + 9] = vpp_idx
        vpp_of_agent[i + 18] = vpp_idx
    for j, pv in enumerate(placement.get("dpv", [])):
        vpp_idx = int(str(pv["vpp"])[-1]) - 1
        vpp_of_agent[27 + j] = vpp_idx
    return vpp_of_agent


def build_global_state(day_row: dict, vpp_idx: int) -> np.ndarray:
    vpp_zone = {0: 1, 1: 2, 2: 4}
    zone = vpp_zone[vpp_idx]
    lambda_p2p = float(day_row.get(f"lambda_p2p_z{zone}", day_row.get("lambda_p2p", 0.0)))
    lambda_as = float(day_row.get(f"lambda_as_z{zone}", day_row.get("lambda_as_ffr", 0.0)))
    delta_f = float(day_row.get("delta_f", 0.0))
    hour_sin = float(day_row.get("hour_sin", 0.0))
    hour_cos = float(day_row.get("hour_cos", 0.0))
    total_load = float(day_row.get("total_load", 0.0))
    return np.array([delta_f, total_load, hour_sin, hour_cos, lambda_p2p, lambda_as], dtype=np.float32)


def build_agent_obs(
    node_features: np.ndarray,
    adjacency: np.ndarray,
    local_state: np.ndarray,
    day_row: dict,
    agent_bus_idx: int,
    vpp_idx: int,
) -> dict:
    return {
        "node_features": node_features.astype(np.float32),
        "adjacency": adjacency.astype(np.float32),
        "local_state": local_state.astype(np.float32),
        "global_state": build_global_state(day_row, vpp_idx),
        "agent_index": int(agent_bus_idx),
    }


def build_agent_bus_indices(env: MicrogridEnv) -> list[int]:
    bus_indices: list[int] = []
    for agent_idx in range(len(env.agent_bus_map)):
        mpc_bus = env.agent_bus_map[agent_idx][0]
        bus_indices.append(env.pp_idx(mpc_bus))
    return bus_indices


def build_bus_adjacency(env: MicrogridEnv) -> np.ndarray:
    bus_ids = list(env.net.bus.index)
    bus_id_map = {bus_id: idx for idx, bus_id in enumerate(bus_ids)}
    n_buses = len(bus_ids)
    adjacency = np.zeros((n_buses, n_buses), dtype=np.float32)
    for _, row in env.net.line.iterrows():
        from_bus = int(row["from_bus"])
        to_bus = int(row["to_bus"])
        adjacency[bus_id_map[from_bus], bus_id_map[to_bus]] = 1.0
        adjacency[bus_id_map[to_bus], bus_id_map[from_bus]] = 1.0
    return adjacency


def flatten_actions(actions: np.ndarray) -> np.ndarray:
    flat_parts: list[np.ndarray] = []
    for idx, agent_type in enumerate(AGENT_TYPES):
        if agent_type == "EVCS_V2G":
            flat_parts.append(actions[idx, :1])
        else:
            flat_parts.append(actions[idx, :2])
    return np.concatenate(flat_parts, axis=0)


def rollout_episode(
    envs,
    policy: MappoPolicy,
    buffer: RolloutBuffer,
    vpp_of_agent: Dict[int, int],
    n_steps: int,
) -> float:
    obs = envs.reset()
    episode_rewards = np.zeros(envs.num_envs, dtype=np.float32)

    agent_bus_indices = build_agent_bus_indices(envs.envs[0])
    bus_adjacency = build_bus_adjacency(envs.envs[0])
    n_buses = bus_adjacency.shape[0]

    for _ in range(n_steps):
        actions_batch = []
        for env_idx in range(envs.num_envs):
            obs_env = obs[env_idx]
            if obs_env.shape[0] != N_AGENTS:
                raise ValueError(f"Expected {N_AGENTS} agents, got {obs_env.shape[0]}")
            day_row = envs.envs[env_idx].current_row.to_dict() if envs.envs[env_idx].current_row is not None else {}

            node_features = np.zeros((n_buses, obs_env.shape[1]), dtype=np.float32)
            for agent_idx, bus_idx in enumerate(agent_bus_indices):
                node_features[bus_idx] = obs_env[agent_idx]

            embeddings = policy.encode(node_features, bus_adjacency)
            embeddings_np = embeddings.detach().cpu().numpy()

            agent_bus_idx_arr = np.asarray(agent_bus_indices, dtype=np.int64)
            emb_per_agent = embeddings_np[agent_bus_idx_arr]
            local_states = obs_env.astype(np.float32)
            global_states = np.stack(
                [build_global_state(day_row, vpp_of_agent.get(i, 0)) for i in range(obs_env.shape[0])],
                axis=0,
            ).astype(np.float32)

            actions, log_probs, values = policy.act_batch(
                emb_per_agent,
                local_states,
                global_states,
                agent_bus_idx_arr,
            )

            agent_actions = actions.astype(np.float32)
            for agent_idx in range(obs_env.shape[0]):
                agent_bus_idx = agent_bus_indices[agent_idx]
                buffer.add(
                    embeddings=embeddings_np[agent_bus_idx],
                    agent_index=agent_bus_idx,
                    local_state=local_states[agent_idx],
                    global_state=global_states[agent_idx],
                    action=agent_actions[agent_idx],
                    log_prob=float(log_probs[agent_idx].item()),
                    reward=0.0,
                    value=float(values[agent_idx].item()),
                    done=False,
                )

            flat_action = flatten_actions(agent_actions)
            if flat_action.shape[0] != ACTION_DIM_FLAT:
                raise ValueError(
                    f"Expected flat action dim {ACTION_DIM_FLAT}, got {flat_action.shape[0]}"
                )
            actions_batch.append(flat_action)

        next_obs, rewards, dones, infos = envs.step(np.asarray(actions_batch, dtype=np.float32))
        episode_rewards += rewards

        for env_idx in range(envs.num_envs):
            env_reward = float(rewards[env_idx])
            info = infos[env_idx] if isinstance(infos, (list, tuple)) else infos
            p_p2p = np.asarray(info.get("P_p2p", np.zeros(N_AGENTS, dtype=np.float32)), dtype=np.float32)
            if p_p2p.shape[0] != N_AGENTS:
                raise ValueError(f"Expected P_p2p shape ({N_AGENTS},), got {p_p2p.shape}")
            zone_prices = next_obs[env_idx][:, 11].astype(np.float32)
            r_p2p_zone = float(ALPHA_P2P * np.sum(zone_prices * p_p2p))
            r_p2p_old = float(info.get("reward_breakdown", {}).get("r_p2p", 0.0))
            reward = env_reward - r_p2p_old + r_p2p_zone

            buffer.rewards[env_idx * N_AGENTS : (env_idx + 1) * N_AGENTS] = [
                reward
            ] * N_AGENTS
            done = bool(dones[env_idx])
            buffer.dones[env_idx * N_AGENTS : (env_idx + 1) * N_AGENTS] = [done] * N_AGENTS

        obs = next_obs

    return float(episode_rewards.mean())


def main() -> None:
    args = parse_args()
    phase_cfg = PHASES[args.phase]
    n_episodes = args.n_episodes or phase_cfg.n_episodes

    placement = load_placement(args.placement)
    vpp_of_agent = build_vpp_of_agent(placement)

    env_kwargs = {
        "placement_path": args.placement,
        "mpc_path": "data/grid_IEEE123_complete.m",
        "precomputed_dir": args.precomputed_dir,
    }

    envs = make_vec_envs(n_envs=args.n_envs, env_kwargs=env_kwargs, seed=args.seeds[0], use_dummy=True)

    policy_config = MappoPolicyConfig(learning_rate=phase_cfg.learning_rate)
    policy = MappoPolicy(config=policy_config)

    buffer = RolloutBuffer()
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"phase_{args.phase.lower()}_latest.pt"
    if checkpoint_path.exists():
        policy.load_checkpoint(checkpoint_path)


    reward_log = []
    for episode in range(n_episodes):
        buffer.clear()
        avg_reward = rollout_episode(envs, policy, buffer, vpp_of_agent, n_steps=96)
        last_value = 0.0
        metrics = policy.update(buffer, last_value=last_value)

        reward_log.append(avg_reward)
        if (episode + 1) % args.log_interval == 0 or episode == 0:
            mean_recent = float(np.mean(reward_log[-args.log_interval :]))
            print(
                f"ep={episode:4d} reward={avg_reward:8.3f} mean({args.log_interval})={mean_recent:8.3f} "
                f"loss={metrics['total_loss']:.3f}"
            )
            policy.save_checkpoint(checkpoint_path)

    policy.save_checkpoint(checkpoint_path)


if __name__ == "__main__":
    main()
