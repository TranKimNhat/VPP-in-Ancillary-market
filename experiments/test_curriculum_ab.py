"""A/B test: Curriculum vs No-Curriculum for AM-MAPPO (100 episodes each)."""
from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.env.microgrid_env_dual import MicrogridEnvDual
from src.opt.tie_switch_reconfig import TieSwitchReconfiguration
from src.rl.train_am_mappo import (
    AMBuffer,
    AMRewardConfig,
    GAMAPPOAgent,
    RewardNormalizer,
    RunningNormalizer,
    build_am_full_feeder_obs,
    compute_am_reward,
    ensure_edge_index,
)


@dataclass
class CurriculumPhase:
    name: str
    episodes: int
    event_prob: float
    max_delta_p_mw: float
    lr_factor: float = 1.0


CURRICULUM_PHASES = [
    CurriculumPhase("MILD", 25, 0.5, 2.5, 1.0),
    CurriculumPhase("MODERATE", 25, 0.7, 4.0, 0.8),
    CurriculumPhase("SEVERE", 25, 0.9, 5.5, 0.6),
    CurriculumPhase("FULL", 25, 0.9, 6.3, 0.4),
]

NO_CURRICULUM_PHASE = CurriculumPhase("FULL", 100, 0.9, 6.3, 1.0)


def train_with_config(
    env: MicrogridEnvDual,
    phases: list[CurriculumPhase],
    seed: int,
    base_lr: float = 3e-4,
    steps_per_episode: int = 200,
) -> dict:
    """Train AM-MAPPO with given curriculum phases."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    obs_fast0, _, _ = env.reset(seed=seed)
    obs_full0 = build_am_full_feeder_obs(env, obs_fast0)
    n_bus = int(obs_full0.shape[0])
    obs_feat = int(obs_full0.shape[1])
    agent_bus_indices = np.asarray(getattr(env, "_agent_bus_pp", np.arange(env.n_agents)), dtype=np.int64)
    agent_bus_indices = np.clip(agent_bus_indices, 0, max(n_bus - 1, 0))

    agent = GAMAPPOAgent(
        obs_feat=obs_feat,
        n_agents=env.n_agents,
        n_bus=n_bus,
        agent_bus_indices=agent_bus_indices,
        action_dim_per_agent=1,
        hidden_dim=64,
        embed_dim=64,
        lr=base_lr,
        gamma=0.99,
        gae_lambda=0.95,
        entropy_coef=0.01,
        value_coef=0.5,
    )

    buffer = AMBuffer()
    obs_normalizer = RunningNormalizer(obs_full0.shape)
    reward_normalizer = RewardNormalizer(gamma=agent.gamma)
    reward_cfg = AMRewardConfig()

    history = {
        "episode_reward": [],
        "delta_f_mean": [],
        "max_abs_delta_f": [],
        "violation_fraction": [],
        "entropy": [],
        "phase": [],
    }

    global_ep = 0
    for phase in phases:
        # Adjust event injector for this phase
        env.event_injector.event_prob = phase.event_prob
        env.event_injector.max_delta_p = phase.max_delta_p_mw

        # Adjust learning rate
        phase_lr = base_lr * phase.lr_factor
        for pg in agent.optimizer.param_groups:
            pg["lr"] = phase_lr

        for _ep in range(phase.episodes):
            obs_fast, _, _ = env.reset()
            n_bus = int(len(env.net.bus.index))
            edge_index = ensure_edge_index(env.edge_index, n_nodes=n_bus)
            obs_full = build_am_full_feeder_obs(env, obs_fast)
            obs_normalizer.update(obs_full)
            obs_norm = obs_normalizer.normalize(obs_full)

            ep_reward = 0.0
            ep_delta_f = []
            ep_violation = []
            prev_action = None

            for _t in range(steps_per_episode):
                policy_actions, log_probs, values, entropy = agent.act(obs_norm, edge_index)
                control_actions = -policy_actions

                full_action = np.zeros(44, dtype=np.float32)
                full_action[:41] = control_actions.flatten()[:41]
                for vpp_idx, (_vpp_id, member_agents) in enumerate(env._vpp_droop_agents.items()):
                    vpp_action = np.mean([control_actions[ai, 0] for ai in member_agents if ai < len(control_actions)])
                    full_action[41 + vpp_idx] = vpp_action

                pre_freq_state = env.freq_dyn.get_state()
                pre_delta_f = float(pre_freq_state.delta_f_hz)
                pre_rocof = float(pre_freq_state.rocof_hz_s)
                pre_freq_hz = 50.0 + pre_delta_f

                reward, _ = compute_am_reward(
                    delta_f=pre_delta_f,
                    rocof=pre_rocof,
                    action=control_actions.flatten(),
                    prev_action=prev_action,
                    freq_hz=pre_freq_hz,
                    cfg=reward_cfg,
                )

                next_obs_fast, _, done, _, info = env.step_fast(full_action)
                n_bus = int(len(env.net.bus.index))
                edge_index = ensure_edge_index(info.get("edge_index", edge_index), n_nodes=n_bus)

                freq_state = env.freq_dyn.get_state()
                reward_norm = reward_normalizer.normalize(reward, done=done)

                buffer.add(
                    obs=obs_norm.copy(),
                    edge_index=edge_index.copy(),
                    action=policy_actions.copy(),
                    log_prob=float(log_probs.mean()),
                    value=float(values.mean()),
                    reward=reward_norm,
                    done=done,
                )

                next_obs = build_am_full_feeder_obs(env, next_obs_fast)
                obs_normalizer.update(next_obs)
                obs_norm = obs_normalizer.normalize(next_obs)
                prev_action = control_actions.flatten().copy()

                ep_reward += reward
                ep_delta_f.append(abs(freq_state.delta_f_hz))
                ep_violation.append(1.0 if freq_state.delta_f_hz < -reward_cfg.f_limit else 0.0)

            # Update policy
            if len(buffer) >= 10:
                rewards = np.array(buffer.rewards, dtype=np.float32)
                values_arr = np.array(buffer.values, dtype=np.float32)
                dones = np.array(buffer.dones, dtype=np.float32)
                advantages, returns = agent.compute_gae(rewards, values_arr, dones)

                obs_batch = torch.tensor(np.stack(buffer.obs), dtype=torch.float32)
                edge_batch = [torch.tensor(e, dtype=torch.long) for e in buffer.edge_index]
                actions_batch = torch.tensor(np.stack(buffer.actions), dtype=torch.float32)
                old_log_probs = torch.tensor(np.array(buffer.log_probs), dtype=torch.float32)
                returns_t = torch.tensor(returns, dtype=torch.float32)
                advantages_t = torch.tensor(advantages, dtype=torch.float32)

                agent.update(
                    obs_batch, edge_batch, actions_batch,
                    old_log_probs, returns_t, advantages_t,
                    n_epochs=4, mini_batch_size=32,
                )
                buffer.clear()

            history["episode_reward"].append(ep_reward)
            history["delta_f_mean"].append(float(np.mean(ep_delta_f)))
            history["max_abs_delta_f"].append(float(np.max(ep_delta_f)))
            history["violation_fraction"].append(float(np.mean(ep_violation)))
            history["entropy"].append(float(entropy.mean()) if hasattr(entropy, "mean") else float(entropy))
            history["phase"].append(phase.name)

            global_ep += 1

    return history


class CurriculumEventWrapper:
    """Wraps EventInjector to control event probability and magnitude."""

    def __init__(self, injector, event_prob: float = 0.9, max_delta_p: float = 6.3):
        self._injector = injector
        self.event_prob = event_prob
        self.max_delta_p = max_delta_p
        self._rng = np.random.default_rng()

    def sample(self):
        event = self._injector.sample()
        # Probability gate
        if self._rng.random() > self.event_prob:
            event.delta_P_mw = 0.0
            event.t_inject = 1e9
        # Magnitude clipping
        elif abs(event.delta_P_mw) > self.max_delta_p:
            sign = np.sign(event.delta_P_mw)
            event.delta_P_mw = float(sign * self.max_delta_p)
        return event

    def __getattr__(self, name):
        return getattr(self._injector, name)


def run_ab_test(seed: int = 42) -> dict:
    """Run A/B test comparing curriculum vs no-curriculum."""
    print("Initializing environment...")
    env = MicrogridEnvDual(
        placement_path="artifacts/placement/official_placement_v3.json",
        mpc_path="data/grid_IEEE123_complete.m",
        seed=seed,
    )

    # Precompute topologies
    tie_reconfig = TieSwitchReconfiguration(env.base_net, seed=seed)
    if not tie_reconfig.load_cache("data/tie_switch_cache.pkl"):
        tie_reconfig.generate_scenarios(n=20)
        tie_reconfig.save_cache("data/tie_switch_cache.pkl")

    # Wrap event injector for curriculum control
    curriculum_wrapper = CurriculumEventWrapper(env.event_injector)
    env.event_injector = curriculum_wrapper

    print("\n" + "=" * 60)
    print("TEST A: NO CURRICULUM (100 episodes, full difficulty)")
    print("=" * 60)
    t0 = time.time()
    results_no_curriculum = train_with_config(
        env=env,
        phases=[NO_CURRICULUM_PHASE],
        seed=seed,
    )
    time_no_curriculum = time.time() - t0

    # Reset env state
    env.event_injector.event_prob = 0.9
    env.event_injector.max_delta_p = 6.3

    print("\n" + "=" * 60)
    print("TEST B: WITH CURRICULUM (4 phases x 25 episodes)")
    print("=" * 60)
    t0 = time.time()
    results_curriculum = train_with_config(
        env=env,
        phases=CURRICULUM_PHASES,
        seed=seed + 1000,  # Different seed for fair comparison
    )
    time_curriculum = time.time() - t0

    return {
        "no_curriculum": results_no_curriculum,
        "curriculum": results_curriculum,
        "time_no_curriculum": time_no_curriculum,
        "time_curriculum": time_curriculum,
    }


def analyze_results(results: dict) -> None:
    """Print comparison analysis."""
    nc = results["no_curriculum"]
    c = results["curriculum"]

    def stats(arr, last_n=25):
        arr = np.array(arr)
        return {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "last_n_mean": float(np.mean(arr[-last_n:])),
            "last_n_std": float(np.std(arr[-last_n:])),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
        }

    print("\n" + "=" * 70)
    print("A/B TEST RESULTS: Curriculum vs No-Curriculum")
    print("=" * 70)

    metrics = ["episode_reward", "delta_f_mean", "max_abs_delta_f", "violation_fraction", "entropy"]
    labels = ["Reward", "Mean |dF| (Hz)", "Max |dF| (Hz)", "Violation %", "Entropy"]

    print(f"\n{'Metric':<20} | {'No Curriculum':>20} | {'Curriculum':>20} | {'Delta':>10}")
    print("-" * 75)

    for metric, label in zip(metrics, labels):
        nc_stats = stats(nc[metric])
        c_stats = stats(c[metric])
        delta = c_stats["last_n_mean"] - nc_stats["last_n_mean"]
        delta_pct = (delta / abs(nc_stats["last_n_mean"]) * 100) if nc_stats["last_n_mean"] != 0 else 0

        print(f"{label:<20} | {nc_stats['last_n_mean']:>9.4f} +/- {nc_stats['last_n_std']:>6.4f} | "
              f"{c_stats['last_n_mean']:>9.4f} +/- {c_stats['last_n_std']:>6.4f} | "
              f"{delta_pct:>+8.1f}%")

    print("-" * 75)
    print(f"{'Training time (s)':<20} | {results['time_no_curriculum']:>20.1f} | {results['time_curriculum']:>20.1f} |")

    # Learning stability (reward variance in last 25 episodes)
    nc_stability = np.std(nc["episode_reward"][-25:])
    c_stability = np.std(c["episode_reward"][-25:])
    print(f"\n{'Stability (reward std)':<20} | {nc_stability:>20.4f} | {c_stability:>20.4f} |")

    # Final verdict
    print("\n" + "=" * 70)
    print("VERDICT:")
    nc_final = np.mean(nc["episode_reward"][-25:])
    c_final = np.mean(c["episode_reward"][-25:])
    nc_viol = np.mean(nc["violation_fraction"][-25:])
    c_viol = np.mean(c["violation_fraction"][-25:])

    if c_final > nc_final and c_viol <= nc_viol:
        print("-> CURRICULUM WINS: Higher reward AND lower/equal violation")
    elif nc_final > c_final and nc_viol <= c_viol:
        print("-> NO-CURRICULUM WINS: Higher reward AND lower/equal violation")
    elif c_viol < nc_viol:
        print("-> CURRICULUM WINS: Lower violation (safety priority)")
    elif nc_viol < c_viol:
        print("-> NO-CURRICULUM WINS: Lower violation (safety priority)")
    else:
        print("-> INCONCLUSIVE: Mixed results, need more episodes or seeds")
    print("=" * 70)


def main():
    results = run_ab_test(seed=42)

    # Save raw results
    output_path = Path("artifacts/curriculum_ab_test.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert to serializable format
    def convert_list(v):
        if not isinstance(v, list):
            return v
        if len(v) > 0 and isinstance(v[0], str):
            return v  # Keep string lists as-is
        return [float(x) for x in v]

    save_data = {
        "no_curriculum": {k: convert_list(v) for k, v in results["no_curriculum"].items()},
        "curriculum": {k: convert_list(v) for k, v in results["curriculum"].items()},
        "time_no_curriculum": results["time_no_curriculum"],
        "time_curriculum": results["time_curriculum"],
    }
    output_path.write_text(json.dumps(save_data, indent=2), encoding="utf-8")
    print(f"\nRaw results saved to {output_path}")

    analyze_results(results)


if __name__ == "__main__":
    main()
