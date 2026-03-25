from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import numpy as np

from src.env.microgrid_env import MicrogridEnv
from src.env.make_env import make_vec_envs
from src.layer2_control.mappo_policy import MappoPolicy, MappoPolicyConfig, RolloutBuffer
from src.rl.networks import AGENT_TYPES

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
PHASE_ORDER = ["A", "B", "C", "D", "E"]


class EarlyStopping:
    def __init__(self, patience: int = 500, min_delta: float = 0.5) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.best = -np.inf
        self.counter = 0

    def reset(self) -> None:
        self.best = -np.inf
        self.counter = 0

    def step(self, mean_reward: float) -> bool:
        if mean_reward > self.best + self.min_delta:
            self.best = mean_reward
            self.counter = 0
            return False
        self.counter += 1
        return self.counter >= self.patience



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Day 12 MAPPO training loop")
    parser.add_argument("--phase", default="A", choices=[*PHASE_ORDER, "full"])
    parser.add_argument("--n-episodes", type=int, default=None)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--placement", type=str, required=True)
    parser.add_argument("--precomputed-dir", type=str, required=True)
    parser.add_argument("--checkpoint-dir", type=str, required=True)
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--minibatch", type=int, default=1024)
    parser.add_argument("--update-epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--early-stop", action="store_true")
    parser.add_argument("--patience", type=int, default=500)
    parser.add_argument("--min-delta", type=float, default=0.5)
    parser.add_argument("--early-stop-warmup", type=int, default=1000)
    parser.add_argument("--early-stop-window", type=int, default=100)
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
    n_steps: int,
    agent_bus_indices: list[int],
    bus_adjacency: np.ndarray,
) -> float:
    obs = envs.reset()
    episode_rewards = np.zeros(envs.num_envs, dtype=np.float32)

    n_envs = envs.num_envs
    n_agents = N_AGENTS
    n_buses = bus_adjacency.shape[0]
    agent_bus_idx_arr = np.asarray(agent_bus_indices, dtype=np.int64)

    for _ in range(n_steps):
        if obs.shape[1] != N_AGENTS:
            raise ValueError(f"Expected {N_AGENTS} agents, got {obs.shape[1]}")

        node_features_batch = np.zeros((n_envs, n_buses, obs.shape[2]), dtype=np.float32)
        node_features_batch[:, agent_bus_idx_arr, :] = obs.astype(np.float32)

        adj_batch = np.broadcast_to(bus_adjacency, (n_envs, n_buses, n_buses)).astype(np.float32)
        embeddings = policy.encode(node_features_batch, adj_batch)
        embeddings_np = embeddings.detach().cpu().numpy().astype(np.float32)

        emb_per_agent = embeddings_np[:, agent_bus_idx_arr, :]
        local_states = obs.astype(np.float32)
        global_states = np.zeros((n_envs, n_agents, 6), dtype=np.float32)
        global_states[:, :, 2] = local_states[:, :, 13]
        global_states[:, :, 3] = local_states[:, :, 14]
        global_states[:, :, 4] = local_states[:, :, 11]
        global_states[:, :, 5] = local_states[:, :, 12]

        actions, log_probs, values = policy.act_batch(
            emb_per_agent.reshape(n_envs * n_agents, -1),
            local_states.reshape(n_envs * n_agents, -1),
            global_states.reshape(n_envs * n_agents, -1),
            np.tile(np.arange(n_agents, dtype=np.int64), n_envs),
        )

        actions = actions.reshape(n_envs, n_agents, -1).astype(np.float32)
        log_probs = log_probs.reshape(n_envs, n_agents)
        values = values.reshape(n_envs, n_agents)

        actions_batch = np.zeros((n_envs, ACTION_DIM_FLAT), dtype=np.float32)
        for env_idx in range(n_envs):
            for agent_idx in range(n_agents):
                agent_bus_idx = agent_bus_indices[agent_idx]
                buffer.add(
                    embeddings=embeddings_np[env_idx, agent_bus_idx],
                    agent_index=agent_bus_idx,
                    local_state=local_states[env_idx, agent_idx],
                    global_state=global_states[env_idx, agent_idx],
                    action=actions[env_idx, agent_idx],
                    log_prob=float(log_probs[env_idx, agent_idx].item()),
                    reward=0.0,
                    value=float(values[env_idx, agent_idx].item()),
                    done=False,
                )

            flat_action = flatten_actions(actions[env_idx])
            if flat_action.shape[0] != ACTION_DIM_FLAT:
                raise ValueError(
                    f"Expected flat action dim {ACTION_DIM_FLAT}, got {flat_action.shape[0]}"
                )
            actions_batch[env_idx] = flat_action

        next_obs, rewards, dones, infos = envs.step(actions_batch)
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

    preloaded_days = MicrogridEnv.load_all_days(args.precomputed_dir)

    env_kwargs = {
        "placement_path": args.placement,
        "mpc_path": "data/grid_IEEE123_complete.m",
        "precomputed_dir": args.precomputed_dir,
        "preloaded_days": preloaded_days,
    }

    probe_env = MicrogridEnv(**env_kwargs, seed=args.seed)
    agent_bus_indices = build_agent_bus_indices(probe_env)
    bus_adjacency = build_bus_adjacency(probe_env)
    probe_env.close()

    envs = make_vec_envs(n_envs=args.n_envs, env_kwargs=env_kwargs, seed=args.seed, use_dummy=True)

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    phases_to_run = PHASE_ORDER if args.phase == "full" else [args.phase]

    current_lr = args.lr if args.lr is not None else PHASES[phases_to_run[0]].learning_rate
    policy_config = MappoPolicyConfig(
        learning_rate=current_lr,
        minibatch_size=args.minibatch,
        update_epochs=args.update_epochs,
    )
    policy = MappoPolicy(config=policy_config)
    buffer = RolloutBuffer()
    early_stopper = EarlyStopping(patience=args.patience, min_delta=args.min_delta)

    for phase_idx, phase in enumerate(phases_to_run):
        print(f"\n=== Starting Phase {phase} ===")
        phase_cfg = PHASES[phase]
        n_episodes = args.n_episodes or phase_cfg.n_episodes

        phase_lr = args.lr if args.lr is not None else phase_cfg.learning_rate
        for param_group in policy.optimizer.param_groups:
            param_group["lr"] = phase_lr

        latest_ckpt = checkpoint_dir / f"phase_{phase.lower()}_latest.pt"
        final_ckpt = checkpoint_dir / f"phase_{phase}_final.pt"

        if phase_idx == 0 and latest_ckpt.exists():
            policy.load_checkpoint(latest_ckpt)
        elif phase_idx > 0:
            prev_phase = phases_to_run[phase_idx - 1]
            prev_final_ckpt = checkpoint_dir / f"phase_{prev_phase}_final.pt"
            if prev_final_ckpt.exists():
                policy.load_checkpoint(prev_final_ckpt)
                print(f"Loaded checkpoint from {prev_final_ckpt}")

        reward_log = []
        early_stopper.reset()
        apply_early_stop = args.early_stop and phase != "A"

        avg_reward = float("nan")
        for episode in range(n_episodes):
            buffer.clear()
            avg_reward = rollout_episode(
                envs,
                policy,
                buffer,
                n_steps=96,
                agent_bus_indices=agent_bus_indices,
                bus_adjacency=bus_adjacency,
            )
            last_value = 0.0
            metrics = policy.update(buffer, last_value=last_value)

            reward_log.append(avg_reward)
            if (episode + 1) % args.log_interval == 0 or episode == 0:
                mean_recent = float(np.mean(reward_log[-args.log_interval :]))
                print(
                    f"phase={phase} ep={episode:5d} reward={avg_reward:8.3f} mean({args.log_interval})={mean_recent:8.3f} "
                    f"loss={metrics['loss']:.3f} entropy={metrics['entropy']:.3f}"
                )
                policy.save_checkpoint(latest_ckpt)

            if apply_early_stop and episode >= args.early_stop_warmup:
                window = min(args.early_stop_window, len(reward_log))
                mean_window = float(np.mean(reward_log[-window:]))
                if early_stopper.step(mean_window):
                    print(
                        f"Phase {phase} early stopped at ep={episode}, "
                        f"best={early_stopper.best:.3f} — moving to next phase"
                    )
                    break

        policy.save_checkpoint(latest_ckpt)
        policy.save_checkpoint(final_ckpt)
        print(f"Phase {phase} complete. Best reward: {early_stopper.best:.3f}")

    envs.close()


if __name__ == "__main__":
    main()
