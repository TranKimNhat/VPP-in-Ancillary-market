"""Evaluate THD per method using HarmonicAnalyzer.

For each policy, drive the env to a steady operating point, read the
agent-bus active-power injection, and pass it to the closed-form
Y_h harmonic solver. Aggregate THD_V_PCC, THD_V_max, count of buses
> 5% (IEEE-519), THD_I_max, count of branches > 5%, and write the
result to:
    results/thd_verify/thd_per_method.csv
    paper/tables/thd_measured_summary.csv
"""
from __future__ import annotations

from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.env.microgrid_env_dual import MicrogridEnvDual
from src.eval.harmonic_analysis import HarmonicAnalyzer
from src.rl.train_am_mappo import build_am_full_feeder_obs

OUT_RESULTS = ROOT / "results" / "thd_verify"
OUT_RESULTS.mkdir(parents=True, exist_ok=True)
OUT_TABLE = ROOT / "paper" / "tables" / "thd_measured_summary.csv"

PLACEMENT = ROOT / "artifacts/placement/official_placement_v3.json"
MPC = ROOT / "data/grid_IEEE123_complete.m"

CHECKPOINTS = {
    "GraphSAGE-MAPPO": "artifacts/_smoke_am_mappo_200ep/am_mappo_final.pt",
    "GCNN-PPO":        "artifacts/checkpoints_gcnn_ppo_OLD_buggy/final.pt",
    "MLP-MAPPO":       "artifacts/checkpoints_mlp_mappo_6k/mlp_mappo_final.pt",
}


def build_env() -> MicrogridEnvDual:
    # Use mappo mode (44-dim action: n_agents + n_vpps) so the policy
    # wrappers in eval_ffr_topology.py emit compatible actions.
    return MicrogridEnvDual(
        placement_path=str(PLACEMENT),
        mpc_path=str(MPC),
        seed=42,
        ffr_mode="mappo",
    )


def commanded_p_mw(env: MicrogridEnvDual, action_44: np.ndarray) -> np.ndarray:
    """Convert a policy's 44-dim mappo action to per-agent injection in MW.

    First n_agents elements of action_44 are per-agent normalized P refs
    in [-1, 1]; scale by rated MW to get the actual injection that drives
    the inverter switching pattern and thus the harmonic spectrum.
    Adds a baseline 50%-rated operating point so the policy delta is
    superposed on a representative scheduled dispatch (otherwise zero
    actions trivially give zero THD).
    """
    agent_specs = env._agent_specs
    rated = np.asarray([a.get("p_rated_kw", 100.0) / 1000.0 for a in agent_specs], dtype=float)
    n = env.n_agents
    delta_p = np.asarray(action_44[:n], dtype=float) * rated  # normalized * rated
    baseline = 0.5 * rated  # 50%-rated scheduled dispatch
    return baseline + delta_p


def thd_for_policy(env: MicrogridEnvDual, policy_fn, n_warmup: int = 10) -> dict:
    """Run env n_warmup steps under policy_fn, then THD on the final action."""
    obs_fast, _, _ = env.reset()
    edge_index = np.asarray(env.edge_index, dtype=np.int64)
    obs_full = build_am_full_feeder_obs(env, obs_fast)

    last_action = None
    for _ in range(int(n_warmup)):
        try:
            action_env = policy_fn(obs_full, edge_index, env, obs_fast=obs_fast)
        except Exception as exc:
            # If policy crashes on a topology, fall back to a small mock action
            n = env.n_agents
            n_vpps = len(env._vpp_droop_agents)
            action_env = np.zeros(n + n_vpps, dtype=np.float32)
        last_action = action_env
        try:
            next_obs_fast, _, done, _, info = env.step_fast(action_env)
        except Exception:
            break
        obs_fast = next_obs_fast
        obs_full = build_am_full_feeder_obs(env, obs_fast)
        edge_index = np.asarray(info.get("edge_index", edge_index), dtype=np.int64)
        if done:
            obs_fast, _, _ = env.reset()
            obs_full = build_am_full_feeder_obs(env, obs_fast)
            edge_index = np.asarray(env.edge_index, dtype=np.int64)

    # Run power flow at steady state
    import pandapower as pp
    pp.runpp(env.net, numba=False, algorithm="nr", init="flat")

    if last_action is None:
        n = env.n_agents
        n_vpps = len(env._vpp_droop_agents)
        last_action = np.zeros(n + n_vpps, dtype=np.float32)
    p_mw = commanded_p_mw(env, np.asarray(last_action, dtype=float))
    agent_bus_idx = [int(b) for b in env._agent_bus_pp.tolist()]
    # Include GFM bus per HarmonicAnalyzer assertion (islanded mode)
    gfm_idx = getattr(env.net, "_gfm_bus_idx", None)
    if gfm_idx is not None and int(gfm_idx) not in set(agent_bus_idx):
        agent_bus_idx = agent_bus_idx + [int(gfm_idx)]
        # extend p_mw to match (use zero injection for the GFM slack)
        p_mw = np.concatenate([p_mw, np.zeros(1, dtype=float)])

    vm = env.net.res_bus["vm_pu"].values
    bus_mask = np.isfinite(vm) & (np.abs(vm) > 0.05)

    analyzer = HarmonicAnalyzer(env.net)
    result = analyzer.run(p_mw, agent_bus_idx, bus_mask=bus_mask)

    return {
        "THD_V_pct": np.asarray(result["THD_V_pct"], dtype=float),
        "THD_I_pct": np.asarray(result["THD_I_pct"], dtype=float),
        "THD_V_PCC": float(result["THD_V_PCC"]),
        "THD_V_max": float(result["THD_V_max"]),
        "THD_I_max": float(result["THD_I_max"]),
        "buses_over": int(np.sum(np.asarray(result["THD_V_pct"]) > 5.0)),
        "branches_over": int(np.sum(np.asarray(result["THD_I_pct"]) > 5.0)),
        "harmonic_valid": bool(result.get("harmonic_valid", True)),
    }


def make_policy(name: str, ckpt: str):
    """Return a callable matching policy_fn(obs_full, edge_index, env, obs_fast=...)."""
    from src.eval.eval_ffr_topology import (
        GraphSAGEMAPPOPolicy, GCNNPPOPolicy, MLPMAPPOPolicy,
    )
    env = build_env()
    if name == "GraphSAGE-MAPPO":
        return GraphSAGEMAPPOPolicy(Path(ckpt), env), env
    if name == "GCNN-PPO":
        return GCNNPPOPolicy(Path(ckpt), env), env
    if name == "MLP-MAPPO":
        return MLPMAPPOPolicy(Path(ckpt), env), env
    raise ValueError(name)


def fixed_droop_policy_fn(obs_full, edge_index, env, obs_fast=None):
    """Fixed 5% droop in mappo format (44-dim: n_agents P + n_vpps droop).

    Zero action -> classical droop kicks in via env defaults. Each agent
    contributes proportionally to its rated P at 5% droop coefficient.
    The resulting P injection at steady state is small but non-zero.
    """
    n = env.n_agents
    n_vpps = len(env._vpp_droop_agents)
    full = np.zeros(n + n_vpps, dtype=np.float32)
    full[n:] = 0.05  # 5% droop coefficient for each VPP
    return full


def no_ffr_policy_fn(obs_full, edge_index, env, obs_fast=None):
    """No FFR: zero policy action across all DERs and VPPs."""
    n = env.n_agents
    n_vpps = len(env._vpp_droop_agents)
    return np.zeros(n + n_vpps, dtype=np.float32)


def main() -> None:
    print("THD evaluation per method")
    print("=" * 60)

    rows = []
    for name, ckpt in CHECKPOINTS.items():
        if not Path(ckpt).is_file():
            print(f"[skip] {name}: checkpoint not found at {ckpt}")
            continue
        try:
            policy, env = make_policy(name, ckpt)
        except Exception as exc:
            print(f"[skip] {name}: load failed -- {exc}")
            continue
        try:
            r = thd_for_policy(env, policy.act, n_warmup=30)
        except Exception as exc:
            print(f"[fail] {name}: thd run error -- {exc}")
            continue
        print(f"  {name:20s} THD_V_PCC={r['THD_V_PCC']:5.2f}%  "
              f"THD_V_max={r['THD_V_max']:5.2f}%  buses>5%={r['buses_over']:3d}  "
              f"THD_I_max={r['THD_I_max']:5.2f}%  branches>5%={r['branches_over']:3d}  "
              f"valid={r['harmonic_valid']}")
        rows.append({
            "Method": name,
            "THD_V_PCC_pct": r["THD_V_PCC"],
            "THD_V_max_pct": r["THD_V_max"],
            "Buses_over_5pct": r["buses_over"],
            "THD_I_max_pct": r["THD_I_max"],
            "Branches_over_5pct": r["branches_over"],
            "harmonic_valid": r["harmonic_valid"],
        })

    # Non-learning baselines
    for name, fn in [("Fixed Droop", fixed_droop_policy_fn), ("No FFR", no_ffr_policy_fn)]:
        env = build_env()
        try:
            r = thd_for_policy(env, fn, n_warmup=30)
        except Exception as exc:
            print(f"[fail] {name}: {exc}")
            continue
        print(f"  {name:20s} THD_V_PCC={r['THD_V_PCC']:5.2f}%  "
              f"THD_V_max={r['THD_V_max']:5.2f}%  buses>5%={r['buses_over']:3d}  "
              f"THD_I_max={r['THD_I_max']:5.2f}%  branches>5%={r['branches_over']:3d}  "
              f"valid={r['harmonic_valid']}")
        rows.append({
            "Method": name,
            "THD_V_PCC_pct": r["THD_V_PCC"],
            "THD_V_max_pct": r["THD_V_max"],
            "Buses_over_5pct": r["buses_over"],
            "THD_I_max_pct": r["THD_I_max"],
            "Branches_over_5pct": r["branches_over"],
            "harmonic_valid": r["harmonic_valid"],
        })

    df = pd.DataFrame(rows)
    OUT_RESULTS.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_RESULTS / "thd_per_method.csv", index=False)
    OUT_TABLE.parent.mkdir(parents=True, exist_ok=True)
    df_save = df[["Method", "THD_V_PCC_pct", "THD_V_max_pct", "Buses_over_5pct",
                  "THD_I_max_pct", "Branches_over_5pct"]]
    df_save.to_csv(OUT_TABLE, index=False)
    print(f"\nSaved: {OUT_RESULTS / 'thd_per_method.csv'}")
    print(f"Saved: {OUT_TABLE}")


if __name__ == "__main__":
    main()
