"""Debug: why does MATD3 beat GraphSAGE-MAPPO on iae_post?

Runs one base-topology episode per (policy, scenario), logging per-step
delta_f, a_P power, droop K, and nadir-safety activity, then prints where
the IAE accumulates (windows after the 30 s event).

Usage: PYTHONPATH=. python scripts/debug_iae_vs_matd3.py
"""
from copy import deepcopy
from pathlib import Path

import numpy as np

from src.eval.eval_ffr_topology import (
    FFRTopologyEvaluator,
    EventConfig,
)
from src.rl.train_am_mappo import build_am_full_feeder_obs, ensure_edge_index

CKPT_PROPOSED = Path("artifacts/ckpt_proposed_s42/am_mappo_ep400.pt")
CKPT_MATD3 = Path("artifacts/ckpt_matd3/matd3_ep2700.pt")

SCENARIOS = {
    "S1_load_step": EventConfig(type="load_step", delta_P_mw=2.5, location=45, t_inject=30.0),
    "S2_gen_trip": EventConfig(type="gen_trip", delta_P_mw=-3.9, location=67, t_inject=30.0),
}

WINDOWS = [(30, 35), (35, 45), (45, 60), (60, 120), (120, 300)]


def run_traced_episode(ev: FFRTopologyEvaluator, policy, event: EventConfig, n_steps: int = 300):
    env = ev.env
    env.ffr_mode = getattr(policy, "ffr_mode", "droop")
    env.nadir_safety_enabled = bool(getattr(policy, "nadir_safety", True))

    prev_flag = env.fixed_base_topology
    env.fixed_base_topology = True
    obs_fast, _, _ = env.reset(options={"force_event": deepcopy(event)})
    env.fixed_base_topology = prev_flag

    n_bus = len(env.net.bus.index)
    edge_index = ensure_edge_index(env.edge_index, n_nodes=n_bus)
    obs_full = build_am_full_feeder_obs(env, obs_fast)

    p_rated = np.asarray(env._agent_p_rated, dtype=np.float64)

    rec = {k: [] for k in ["delta_f", "p_mw", "k_sum", "ns_active", "ns_dist", "abs_aP"]}
    for _ in range(n_steps):
        action = policy.act(obs_full, edge_index, env, obs_fast=obs_fast)
        obs_fast, _, done, _, info = env.step_fast(action)
        edge_index = ensure_edge_index(info.get("edge_index", edge_index), n_nodes=n_bus)
        obs_full = build_am_full_feeder_obs(env, obs_fast)

        fs = env.freq_dyn_lti.get_state()
        a_p = np.asarray(env._p_ref_last, dtype=np.float64)
        rec["delta_f"].append(fs.delta_f_hz)
        rec["p_mw"].append(float(np.sum(a_p * 0.1 * p_rated)))  # signed commanded ref MW
        rec["abs_aP"].append(float(np.mean(np.abs(a_p))))
        rec["k_sum"].append(float(np.sum(env._k_droop_last)))   # MW/Hz, post SoC mask
        rec["ns_active"].append(bool(info.get("nadir_safety_active", False)))
        rec["ns_dist"].append(float(info.get("nadir_safety_dist", 0.0)))
    return {k: np.asarray(v) for k, v in rec.items()}


def report(name: str, tr: dict, event_step: int = 30):
    df = tr["delta_f"]
    iae_post = float(np.sum(np.abs(df[event_step:])))
    print(f"\n--- {name} ---")
    print(f"iae_post={iae_post:.3f} Hz·s | nadir={50 + df.min():.3f} | zenith={50 + df.max():.3f}")
    print(f"{'window':>10} | {'IAE':>7} | {'mean|df|':>8} | {'mean P_cmd MW':>13} | {'mean|a_P|':>9} | "
          f"{'K_sum MW/Hz':>11} | {'NS act%':>7} | {'NS dist':>8}")
    for a, b in WINDOWS:
        sl = slice(a, b)
        iae_w = float(np.sum(np.abs(df[sl])))
        print(f"{a:>4}-{b:<5} | {iae_w:7.3f} | {np.mean(np.abs(df[sl])):8.4f} | {np.mean(tr['p_mw'][sl]):13.3f} | "
              f"{np.mean(tr['abs_aP'][sl]):9.3f} | {np.mean(tr['k_sum'][sl]):11.2f} | "
              f"{100 * np.mean(tr['ns_active'][sl]):6.1f}% | {np.mean(tr['ns_dist'][sl]):8.4f}")
    # tail bias: signed mean delta_f in the tail reveals steady offset vs oscillation
    tail = df[120:]
    print(f"tail signed mean df={np.mean(tail):+.4f} Hz (|mean|~mean|.| => offset; <<: oscillation)")


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
        checkpoint_path=CKPT_PROPOSED,
        matd3_checkpoint=CKPT_MATD3,
        output_dir=Path("results/debug_iae_traces"),
        base_reference=True,
    )
    for sc_name, event in SCENARIOS.items():
        print(f"\n{'=' * 70}\nSCENARIO {sc_name} (dP={event.delta_P_mw} MW @ t=30s)\n{'=' * 70}")
        for pname in ["GraphSAGE-MAPPO", "MATD3"]:
            policy = ev.policies[pname]
            tr = run_traced_episode(ev, policy, event)
            report(f"{pname} [{sc_name}]", tr)
            np.savez(Path("results/debug_iae_traces") / f"{sc_name}_{pname.replace('-', '_')}.npz", **tr)


if __name__ == "__main__":
    Path("results/debug_iae_traces").mkdir(parents=True, exist_ok=True)
    main()
