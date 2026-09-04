"""Decisive test: how much of the frequency response is the hidden env-side MPC?

Runs S2 (base topology) for GraphSAGE-MAPPO, MATD3 and Fixed Droop with the
env's battery mpc_correction enabled vs monkeypatched to zero, and compares
nadir / iae_post(50s) / iae_total.

Usage: PYTHONPATH=. python scripts/debug_mpc_contribution.py
"""
from copy import deepcopy
from pathlib import Path

import numpy as np

import src.env.microgrid_env_dual as med
from src.eval.eval_ffr_topology import FFRTopologyEvaluator, EventConfig
from src.rl.train_am_mappo import build_am_full_feeder_obs, ensure_edge_index

EVENT = EventConfig(type="gen_trip", delta_P_mw=-3.9, location=67, t_inject=30.0)
_orig_mpc = med.mpc_correction


def run_episode(ev, policy, n_steps: int = 300) -> dict:
    env = ev.env
    env.ffr_mode = getattr(policy, "ffr_mode", "droop")
    env.nadir_safety_enabled = bool(getattr(policy, "nadir_safety", True))
    prev = env.fixed_base_topology
    env.fixed_base_topology = True
    obs_fast, _, _ = env.reset(options={"force_event": deepcopy(EVENT)})
    env.fixed_base_topology = prev

    n_bus = len(env.net.bus.index)
    edge_index = ensure_edge_index(env.edge_index, n_nodes=n_bus)
    obs_full = build_am_full_feeder_obs(env, obs_fast)

    df = []
    for _ in range(n_steps):
        action = policy.act(obs_full, edge_index, env, obs_fast=obs_fast)
        obs_fast, _, _, _, info = env.step_fast(action)
        edge_index = ensure_edge_index(info.get("edge_index", edge_index), n_nodes=n_bus)
        obs_full = build_am_full_feeder_obs(env, obs_fast)
        df.append(env.freq_dyn_lti.get_state().delta_f_hz)
    df = np.asarray(df)
    return {
        "nadir": 50.0 + df.min(),
        "iae_post50": float(np.sum(np.abs(df[30:80]))),
        "iae_total": float(np.sum(np.abs(df))),
    }


def main():
    env_config = {
        "placement_path": "artifacts/placement/official_placement_v3.json",
        "mpc_path": "data/grid_IEEE123_complete.m",
        "seed": 42,
        "ffr_mode": "mappo_dual",
        "day_split": "eval",
    }
    ev = FFRTopologyEvaluator(
        env_config=env_config,
        checkpoint_path=Path("artifacts/ckpt_proposed_s42/am_mappo_ep400.pt"),
        matd3_checkpoint=Path("artifacts/ckpt_matd3/matd3_ep2700.pt"),
        output_dir=Path("results/debug_mpc"),
        base_reference=True,
    )

    names = ["GraphSAGE-MAPPO", "MATD3", "Fixed Droop", "No FFR"]
    print(f"\n{'policy':>16} | {'MPC':>3} | {'nadir':>7} | {'iae_post50':>10} | {'iae_total':>9}")
    for name in names:
        policy = ev.policies[name]
        for mpc_on in (True, False):
            med.mpc_correction = _orig_mpc if mpc_on else (
                lambda evcs_list, **kw: np.zeros(len(evcs_list)))
            r = run_episode(ev, policy)
            print(f"{name:>16} | {'ON' if mpc_on else 'OFF':>3} | {r['nadir']:7.3f} | "
                  f"{r['iae_post50']:10.3f} | {r['iae_total']:9.3f}")
    med.mpc_correction = _orig_mpc


if __name__ == "__main__":
    main()
