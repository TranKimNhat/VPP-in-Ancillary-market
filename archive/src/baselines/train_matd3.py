"""Training script for MATD3 baseline."""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

# Ensure src is importable
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import torch

from src.baselines.matd3 import MATD3Agent, MATD3Config
from src.env.microgrid_env_dual import MicrogridEnvDual

# Import shared Markov process components from main trainer
from src.rl.train_am_mappo import (
    AM_PHASES,
    AMRewardConfig,
    compute_am_reward,
    ensure_edge_index,
    build_am_full_feeder_obs,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# Explorer noise strategies (EIE): each episode is assigned one, cycling through.
EXPLORER_STRATEGIES = ["gaussian", "ou", "eps_greedy", "max_entropy"]
# Every Nth episode is a demonstrator (classical-droop expert) episode -> Pool 2.
DEMO_EVERY = 4


def build_demonstrator_a_k(env) -> np.ndarray:
    """Per-agent demonstrator action component a_K in [-1,1] (a_P is held at 0).

    The demonstrator is the classical droop expert in OUR action space: each
    FFR-capable DER holds a nominal droop gain K_nom = k_coef * P_rated (from
    droop.DroopFFRConfig), inverse-mapped through the env's per-agent affine
    K range [K_min, K_max] to the normalized a_K. Non-FFR agents (PV/wind) get
    a_K=-1 (K=K_min=0, i.e. no droop). This generates high-quality imitation
    transitions for Pool 2.
    """
    from src.baselines.droop import DroopFFRConfig
    cfg = DroopFFRConfig()
    n = env.n_agents
    p_rated = np.asarray(getattr(env, "_agent_p_rated", np.ones(n)), dtype=np.float32)
    bess = set(int(i) for i in getattr(env, "_bess_indices", []))
    v2g = set(int(i) for i in getattr(env, "_v2g_indices", []))
    k_nom = np.zeros(n, dtype=np.float32)
    for i in range(n):
        if i in bess:
            k_nom[i] = cfg.k_coef_bess * p_rated[i]
        elif i in v2g:
            k_nom[i] = cfg.k_coef_v2g * p_rated[i]
    kmin = np.asarray(getattr(env, "_k_droop_min_per_agent", np.zeros(n)), dtype=np.float32)
    kmax = np.asarray(getattr(env, "_k_droop_max_per_agent", np.ones(n)), dtype=np.float32)
    span = np.maximum(kmax - kmin, 1e-6)
    a_k = 2.0 * (k_nom - kmin) / span - 1.0
    return np.clip(a_k, -1.0, 1.0).astype(np.float32)


def train_matd3(
    placement_path: Path,
    mpc_path: Path,
    output_dir: Path,
    n_episodes: int = 500,
    episode_length: int = 300,
    warmup_steps: int = 1000,
    update_freq: int = 1,
    eval_interval: int = 50,
    seed: int = 42,
    curriculum: bool = False,
    resume_from: Path | None = None,
    device: str = "auto",
    fixed_base_topology: bool = False,
    skip_to_phase: str | None = None,
    start_global_episode: int = 0,
) -> dict:
    """Train MATD3 with synchronized Markov process (same as train_am_mappo.py).

    Resume support:
      resume_from         load weights + buffer state from this checkpoint
      device              "auto" | "cpu" | "cuda". CPU avoids CUDA memory-leak
                          crashes seen in long runs (>3800 ep).
      skip_to_phase       e.g. "E" to start training from Phase E onward; phases
                          before it are not iterated.
      start_global_episode initial value of global_episode counter (so logs
                          and checkpoint filenames stay aligned with the
                          original run).
    """
    set_seed(seed)

    env = MicrogridEnvDual(
        placement_path=str(placement_path),
        mpc_path=str(mpc_path),
        seed=seed,
        ffr_mode="mappo_dual",  # same MDP as proposed; exercises the (a_P, a_K) dual action
    )
    env.fixed_base_topology = fixed_base_topology
    print(f"fixed_base_topology={env.fixed_base_topology}")

    # Get observation dimension from build_am_full_feeder_obs (same as train_am_mappo.py)
    sample_obs_fast, _, _ = env.reset()
    sample_obs_full = build_am_full_feeder_obs(env, sample_obs_fast)
    obs_feat = int(sample_obs_full.shape[1])
    n_bus = int(sample_obs_full.shape[0])

    # MATD3 uses per-agent observations, flatten full feeder obs per agent
    # obs_dim = obs_feat (features per node, agent extracts its node's features)
    config = MATD3Config(
        obs_dim=obs_feat,
        action_dim=2,  # (ctrl_p, ctrl_k) same as train_am_mappo.py
        n_agents=env.n_agents,
        buffer_size=100_000,
        batch_size=256,
    )

    if device == "auto":
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device_str = device
    agent = MATD3Agent(config, device=device_str)
    print(f"[matd3] device={device_str}")

    if resume_from is not None:
        agent.load(Path(resume_from))
        print(f"[matd3] resumed from {resume_from} (exploration_noise={agent.exploration_noise:.4f})")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    reward_cfg = AMRewardConfig()

    # Setup curriculum or single-phase training
    if curriculum:
        phase_items = list(AM_PHASES.items())
        if skip_to_phase is not None:
            phase_items = [(name, cfg) for (name, cfg) in phase_items if name >= skip_to_phase]
            print(f"[matd3] skipping to phase {skip_to_phase}; running phases: {[n for n,_ in phase_items]}")
    else:
        phase_items = [
            ("FULL", {
                "n_episodes": n_episodes,
                "event_probs": {"load_step": 0.3, "gen_trip": 0.3, "line_trip": 0.2, "high_ren": 0.2},
                "max_delta_p_mw": 6.3,
            })
        ]

    total_steps = 0
    # If we skipped phases, warmup is already done; mark steps past the warmup threshold
    if resume_from is not None:
        total_steps = max(warmup_steps, 1)
    global_episode = int(start_global_episode)
    best_reward = float("-inf")
    rewards_history = []

    # EIE demonstrator action (classical-droop expert) in our action space.
    a_k_demo = build_demonstrator_a_k(env)

    for phase_name, phase_cfg in phase_items:
        n_phase_eps = phase_cfg.get("n_episodes", n_episodes)

        # Configure environment for current phase
        env.event_injector.set_max_delta_p_mw(phase_cfg.get("max_delta_p_mw", 6.3))
        env.event_injector.set_probs(phase_cfg["event_probs"])

        print(f"\n=== Phase {phase_name} ({n_phase_eps} episodes) ===")

        for ep in range(n_phase_eps):
            global_episode += 1

            # Reset and build full feeder observation (same as train_am_mappo.py)
            obs_fast, _, _ = env.reset()
            n_bus = int(len(env.net.bus.index))
            edge_index = ensure_edge_index(env.edge_index, n_nodes=n_bus)
            obs_full = build_am_full_feeder_obs(env, obs_fast)

            # Extract per-agent observations from full feeder obs
            agent_bus_idx = np.asarray(getattr(env, "_agent_bus_pp", np.arange(env.n_agents)), dtype=np.int64)
            agent_bus_idx = np.clip(agent_bus_idx, 0, n_bus - 1)
            obs = obs_full[agent_bus_idx]  # (n_agents, obs_feat)

            # --- EIE role assignment for this episode ---
            # Demonstrator episode -> Pool 2 (imitation); otherwise an explorer
            # episode with one of the diverse noise strategies -> Pool 1.
            is_demo = (global_episode % DEMO_EVERY == 0)
            strategy = EXPLORER_STRATEGIES[ep % len(EXPLORER_STRATEGIES)]
            agent.reset_noise()
            target_buffer = agent.demo_buffer if is_demo else agent.buffer

            episode_reward = 0.0
            prev_action: np.ndarray | None = None
            prev_k_droop = np.zeros(env.n_agents, dtype=np.float32)

            for step in range(episode_length):
                explore = total_steps >= warmup_steps
                if is_demo:
                    # Classical-droop expert: a_P=0, a_K=nominal droop (per DER).
                    actions = np.zeros((env.n_agents, 2), dtype=np.float32)
                    actions[:, 1] = a_k_demo
                else:
                    actions = agent.act(obs, explore=explore, strategy=strategy)  # (n_agents, 2)

                # Build full action (same logic as train_am_mappo.py)
                ctrl_p = -actions[:, 0]  # Sign flip for frequency response
                ctrl_k = actions[:, 1]

                n_vpps = len(env._vpp_droop_agents)
                if env.ffr_mode == "mappo_dual":
                    full_action = np.zeros(2 * env.n_agents + n_vpps, dtype=np.float32)
                    full_action[: env.n_agents] = ctrl_p
                    full_action[env.n_agents : 2 * env.n_agents] = ctrl_k
                else:
                    full_action = np.zeros(env.n_agents + n_vpps, dtype=np.float32)
                    full_action[: env.n_agents] = ctrl_p
                    for vpp_idx, (_vpp_id, member_agents) in enumerate(env._vpp_droop_agents.items()):
                        vpp_action = np.mean([ctrl_p[ai] for ai in member_agents if ai < len(ctrl_p)])
                        full_action[env.n_agents + vpp_idx] = vpp_action

                control_actions = np.stack([ctrl_p, ctrl_k], axis=1)

                # Step fast timescale
                next_obs_fast, _r_env, done, _, info = env.step_fast(full_action)

                # Compute reward using compute_am_reward (same as train_am_mappo.py),
                # on the COI frequency deviation: the resolvable system-frequency
                # observable at the 1 s control step, identical across all methods.
                post_freq_state = env.freq_dyn_lti.get_state()
                post_delta_f = float(post_freq_state.delta_f_hz)
                post_rocof = float(post_freq_state.rocof_hz_s)
                post_freq_hz = 50.0 + post_delta_f

                soc_vec = np.asarray(next_obs_fast[:, 3], dtype=np.float32) if next_obs_fast.ndim == 2 else None
                zone_lmp_vec = (
                    np.asarray(next_obs_fast[:, 4], dtype=np.float32) if (next_obs_fast.ndim == 2 and next_obs_fast.shape[1] > 4) else None
                )
                k_droop_now = getattr(env, "_k_droop_last", None)
                k_droop_max = getattr(env, "_k_droop_max_per_agent", None)
                p_ref_now = getattr(env, "_p_ref_last", None)

                am_reward, _ = compute_am_reward(
                    delta_f=post_delta_f,
                    rocof=post_rocof,
                    action=control_actions.flatten(),
                    prev_action=prev_action,
                    freq_hz=post_freq_hz,
                    cfg=reward_cfg,
                    soc=soc_vec,
                    zone_lmp=zone_lmp_vec,
                    k_droop_now=k_droop_now,
                    k_droop_prev=prev_k_droop,
                    p_ref_now=p_ref_now,
                    k_droop_max=k_droop_max,
                    lambda_as_ffr=getattr(env, "lambda_as_ffr", None),
                )

                if k_droop_now is not None:
                    prev_k_droop = np.asarray(k_droop_now, dtype=np.float32).copy()

                # Build next observation
                n_bus = int(len(env.net.bus.index))
                next_obs_full = build_am_full_feeder_obs(env, next_obs_fast)
                next_obs = next_obs_full[agent_bus_idx]

                # Store transition with per-agent rewards into the correct pool
                # (Pool 2 for demonstrator episodes, Pool 1 for explorer episodes).
                rewards = np.full(env.n_agents, am_reward, dtype=np.float32)
                target_buffer.add(obs, actions, rewards, next_obs, done)

                obs = next_obs
                prev_action = control_actions.flatten().copy()
                episode_reward += am_reward
                total_steps += 1

                if total_steps >= warmup_steps and total_steps % update_freq == 0:
                    agent.update()

                if done:
                    break

            rewards_history.append(episode_reward)

            if global_episode % eval_interval == 0:
                mean_reward = np.mean(rewards_history[-eval_interval:])
                print(f"Episode {global_episode} | Phase {phase_name} | Mean reward: {mean_reward:.4f} | Noise: {agent.exploration_noise:.4f}")

                if mean_reward > best_reward:
                    best_reward = mean_reward
                    agent.save(output_dir / "matd3_best.pt")

            if global_episode % 100 == 0:
                agent.save(output_dir / f"matd3_ep{global_episode}.pt")

    agent.save(output_dir / "matd3_final.pt")

    result = {
        "seed": seed,
        "n_episodes": global_episode,
        "best_reward": float(best_reward),
        "final_reward": float(np.mean(rewards_history[-50:])) if rewards_history else 0.0,
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
    parser.add_argument("--n-episodes", type=int, default=6000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--curriculum", action="store_true", help="Use AM_PHASES curriculum (6000 ep total)")
    parser.add_argument("--fixed-base-topology", action="store_true",
                        help="Train only on the nominal base feeder (no reconfiguration). "
                             "Used for the train-on-base, eval-on-all-reconfig generalization protocol.")
    parser.add_argument("--resume-from", type=Path, default=None, help="Load weights+buffer from this checkpoint")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"],
                        help="Force device. CPU avoids CUDA illegal-memory crashes seen after ~3800 ep.")
    parser.add_argument("--skip-to-phase", type=str, default=None,
                        help="Start training from this phase name (e.g. 'E'); earlier phases are skipped.")
    parser.add_argument("--start-global-episode", type=int, default=0,
                        help="Initial value of global_episode counter (so log/save filenames continue the original run).")
    args = parser.parse_args()

    train_matd3(
        placement_path=args.placement,
        mpc_path=args.mpc_path,
        output_dir=args.output_dir,
        n_episodes=args.n_episodes,
        seed=args.seed,
        curriculum=args.curriculum,
        resume_from=args.resume_from,
        device=args.device,
        skip_to_phase=args.skip_to_phase,
        start_global_episode=args.start_global_episode,
        fixed_base_topology=args.fixed_base_topology,
    )


if __name__ == "__main__":
    main()
