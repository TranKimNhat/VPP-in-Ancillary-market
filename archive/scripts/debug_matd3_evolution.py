"""Debug: is MATD3's constant max-effort a bug, saturation, or a training stage?

Collects real observations from one S2 episode (env driven by fixed droop),
then evaluates MATD3 actors from successive checkpoints on the SAME obs set.
Reports per-checkpoint action statistics: magnitude, saturation, and
state-sensitivity (does the action actually vary with the state?).

Usage: PYTHONPATH=. python scripts/debug_matd3_evolution.py
"""
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch

from src.env.microgrid_env_dual import MicrogridEnvDual
from src.rl.train_am_mappo import build_am_full_feeder_obs
from src.baselines.matd3 import MATD3Agent, MATD3Config

CKPTS = ["matd3_ep100", "matd3_ep500", "matd3_ep1000", "matd3_ep1500",
         "matd3_ep2000", "matd3_ep2700", "matd3_best"]


def collect_obs(env, n_steps: int = 90) -> np.ndarray:
    """Run one S2 episode under the env's built-in droop; save per-agent obs rows."""
    from src.eval.eval_ffr_topology import EventConfig
    env.ffr_mode = "droop"
    env.nadir_safety_enabled = False
    prev = env.fixed_base_topology
    env.fixed_base_topology = True
    obs_fast, _, _ = env.reset(options={"force_event": EventConfig(
        type="gen_trip", delta_P_mw=-3.9, location=67, t_inject=30.0)})
    env.fixed_base_topology = prev

    agent_bus_idx = np.clip(
        np.asarray(getattr(env, "_agent_bus_pp", np.arange(env.n_agents)), dtype=np.int64),
        0, 122,
    )
    rows = []
    n_vpps = len(env._vpp_droop_agents)
    action = np.zeros(env.n_agents + n_vpps, dtype=np.float32)
    for _ in range(n_steps):
        obs_full = build_am_full_feeder_obs(env, obs_fast)
        rows.append(obs_full[agent_bus_idx].copy())
        obs_fast, _, _, _, _ = env.step_fast(action)
    return np.stack(rows)  # (T, n_agents, obs_feat)


def main():
    env = MicrogridEnvDual(
        placement_path="artifacts/placement/official_placement_v3.json",
        mpc_path="data/grid_IEEE123_complete.m",
        seed=42,
        ffr_mode="droop",
        day_split="eval",
    )
    obs_set = collect_obs(env)
    T, n_agents, obs_feat = obs_set.shape
    pre, post = obs_set[:30].reshape(-1, obs_feat), obs_set[30:45].reshape(-1, obs_feat)
    print(f"obs set: {T} steps x {n_agents} agents x {obs_feat} feats "
          f"(pre-event {pre.shape[0]} rows, post-event {post.shape[0]} rows)")

    print(f"\n{'ckpt':>14} | {'aP_pre':>7} {'aP_post':>7} | {'aK_pre':>7} {'aK_post':>7} | "
          f"{'sat%|a|>.9':>10} | {'state-sens aP':>13} {'aK':>6}")
    for name in CKPTS:
        path = Path(f"artifacts/ckpt_matd3/{name}.pt")
        if not path.exists():
            continue
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        cfg = ckpt.get("config")
        obs_dim = int(getattr(cfg, "obs_dim", obs_feat))
        hidden = int(getattr(cfg, "hidden_dim", 128))
        agent = MATD3Agent(MATD3Config(obs_dim=obs_dim, hidden_dim=hidden,
                                       action_dim=2, n_agents=n_agents), device="cpu")
        agent.load(path)
        agent.eval()

        def act(rows: np.ndarray) -> np.ndarray:
            out = []
            for t in range(0, rows.shape[0], n_agents):
                chunk = rows[t:t + n_agents]
                if chunk.shape[0] != n_agents:
                    break
                out.append(agent.act_deterministic(chunk))
            return np.concatenate(out)

        a_pre, a_post = act(pre), act(post)
        a_all = np.concatenate([a_pre, a_post])
        sat = float(np.mean(np.abs(a_all) > 0.9)) * 100
        # state-sensitivity: std of action across different states (per channel)
        sens_p = float(np.std(a_all[:, 0]))
        sens_k = float(np.std(a_all[:, 1]))
        print(f"{name:>14} | {np.mean(np.abs(a_pre[:, 0])):7.3f} {np.mean(np.abs(a_post[:, 0])):7.3f} | "
              f"{np.mean(a_pre[:, 1]):7.3f} {np.mean(a_post[:, 1]):7.3f} | {sat:9.1f}% | "
              f"{sens_p:13.3f} {sens_k:6.3f}")


if __name__ == "__main__":
    main()
