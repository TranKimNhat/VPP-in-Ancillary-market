"""Measure THD while FFR is actively responding to a generation trip (S2).

Procedure per method:
  1. Inject S2_gen_trip event
  2. Run env for T_peak steps (~5 s) so FFR is at peak activation
  3. Call pandapower runpp + HarmonicAnalyzer on the live network state
  4. Also measure at T_settle (~30 s) when system is settling

Output: THD at peak vs. settle vs. steady-state (no event) for each method.
"""
from __future__ import annotations
import copy
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
import sys; sys.path.insert(0, str(ROOT))

from src.eval.eval_ffr_topology import (
    FFRTopologyEvaluator, ensure_edge_index,
    GraphSAGEMAPPOPolicy, MLPMAPPOPolicy, MATD3Policy, GCNNPPOPolicy,
)
from src.eval.harmonic_analysis import HarmonicAnalyzer, IEEE519_THD_V_LIMIT
from src.rl.train_am_mappo import build_am_full_feeder_obs
from scripts.eval_thd import commanded_p_mw, agent_rated_mw
import pandapower as pp

PLACEMENT = "artifacts/placement/official_placement_v3.json"
MPC = "data/grid_IEEE123_complete.m"

ev = FFRTopologyEvaluator(
    env_config={"placement_path": PLACEMENT, "mpc_path": MPC, "seed": 42},
    checkpoint_path=Path("artifacts/ckpt_proposed_s42/am_mappo_final.pt"),
    mlp_mappo_checkpoint=Path("artifacts/ckpt_mlp_mappo/mlp_mappo_final.pt"),
    gcnn_checkpoint=Path("artifacts/ckpt_gcnn_ppo/final.pt"),
    matd3_checkpoint=Path("artifacts/ckpt_matd3/matd3_ep3800.pt"),
    output_dir=Path("results/_thd_ffr"),
    base_reference=True,
)

# Use S2 (gen trip) — the harshest event that triggers full FFR activation
S2_EVENT = ev.scenarios["S2_gen_trip"]

T_PEAK   = 5    # steps after event (~0.5 s) — nadir region, peak FFR current
T_MID    = 30   # steps (~3 s) — mid-response
T_SETTLE = 200  # steps (~20 s) — settling phase

CKPTS = {
    "GraphSAGE-MAPPO": ("artifacts/ckpt_proposed_s42/am_mappo_final.pt",   GraphSAGEMAPPOPolicy),
    "MLP-MAPPO":       ("artifacts/ckpt_mlp_mappo/mlp_mappo_final.pt",     MLPMAPPOPolicy),
    "MATD3":           ("artifacts/ckpt_matd3/matd3_ep3800.pt",            MATD3Policy),
    "GCNN-PPO":        ("artifacts/ckpt_gcnn_ppo/final.pt",                GCNNPPOPolicy),
}

def measure_thd(env, action, i_l_a=None):
    """Run runpp + HarmonicAnalyzer on current env state with given action.

    i_l_a: frozen pre-event demand-current reference for TDD (IEEE 519 uses
    a fixed max-demand I_L, not the instantaneous fundamental during the
    event -- a gen trip depresses I_1 at the PCC and would inflate TDD).
    """
    try:
        pp.runpp(env.net, numba=False, algorithm="nr", init="flat")
    except Exception:
        pass
    n = env.n_agents
    p_mw = commanded_p_mw(env, np.asarray(action, dtype=float))
    rated = agent_rated_mw(env)
    agent_bus_idx = [int(b) for b in env._agent_bus_pp.tolist()]
    gfm_idx = getattr(env.net, "_gfm_bus_idx", None)
    if gfm_idx is not None and int(gfm_idx) not in set(agent_bus_idx):
        agent_bus_idx = agent_bus_idx + [int(gfm_idx)]
        p_mw = np.concatenate([p_mw, np.zeros(1, dtype=float)])
        rated = np.concatenate([rated, np.ones(1, dtype=float)])
    vm = env.net.res_bus["vm_pu"].values
    bus_mask = np.isfinite(vm) & (np.abs(vm) > 0.05)
    analyzer = HarmonicAnalyzer(env.net)
    result = analyzer.run(p_mw, agent_bus_idx, bus_mask=bus_mask,
                          agent_p_rated_mw=rated, i_l_a=i_l_a)
    thd_v = np.asarray(result["THD_V_pct"])
    return {
        "THD_V_PCC":  float(result["THD_V_PCC"]),
        "THD_V_max":  float(result["THD_V_max"]),
        "buses_over": int(np.sum(thd_v[np.isfinite(thd_v)] > IEEE519_THD_V_LIMIT)),
        "TDD_I_PCC":  float(result["TDD_I_PCC"]),
        "TDD_I_max":  float(result["TDD_I_max"]),
        "branches_over_tdd": int(result["branches_over_tdd"]),
        "a_P_mean":   float(np.mean(action[:n])),
        "p_mw_mean":  float(np.mean(p_mw)),
        "loading_mean": float(np.mean(np.clip(np.abs(p_mw) / np.maximum(rated, 1e-9), 0, 1))),
        "freq_hz":    50.0 + float(env.freq_dyn_lti.get_state().delta_f_hz),
        "I1_branch_A": np.asarray(result["I1_branch_A"], dtype=float),
    }

rows = []
for name, (ckpt, cls) in CKPTS.items():
    print(f"\n{'='*60}\n{name}")
    env = ev.env
    env.ffr_mode = "mappo_dual"
    env.nadir_safety_enabled = True
    env.fixed_base_topology = True
    pol = cls(Path(ckpt), env)

    # ---- Pre-event reference: freeze the demand current I_L for TDD ----
    obs, _, _ = env.reset()
    nb = len(env.net.bus.index)
    edge = ensure_edge_index(env.edge_index, n_nodes=nb)
    obs_full = build_am_full_feeder_obs(env, obs)
    action = None
    for _ in range(10):
        action = pol.act(obs_full, edge, env, obs_fast=obs)
        obs, _, _, _, info = env.step_fast(action)
        edge = ensure_edge_index(info.get("edge_index", edge), n_nodes=nb)
        obs_full = build_am_full_feeder_obs(env, obs)
    r0 = measure_thd(env, action)
    i_l_ref = r0["I1_branch_A"]
    print(f"  pre-event       freq={r0['freq_hz']:.3f} Hz  "
          f"a_P={r0['a_P_mean']:+.3f}  loading={r0['loading_mean']:.3f}  "
          f"THD_PCC={r0['THD_V_PCC']:.2f}%  TDD_PCC={r0['TDD_I_PCC']:.2f}%")

    env.fixed_base_topology = True
    obs, _, _ = env.reset(options={"force_event": copy.deepcopy(S2_EVENT)})
    env.fixed_base_topology = False
    edge = ensure_edge_index(env.edge_index, n_nodes=nb)
    obs_full = build_am_full_feeder_obs(env, obs)

    snapshots = {}
    for t in range(T_SETTLE + 1):
        action = pol.act(obs_full, edge, env, obs_fast=obs)
        if t in (T_PEAK, T_MID, T_SETTLE):
            r = measure_thd(env, action, i_l_a=i_l_ref)
            r["step"] = t
            snapshots[t] = r
            print(f"  t={t:3d} ({t*0.1:.1f}s)  freq={r['freq_hz']:.3f} Hz  "
                  f"a_P={r['a_P_mean']:+.3f}  loading={r['loading_mean']:.3f}  "
                  f"THD_PCC={r['THD_V_PCC']:.2f}%  buses>{IEEE519_THD_V_LIMIT}%: {r['buses_over']}/123  "
                  f"TDD_PCC={r['TDD_I_PCC']:.2f}%  TDD_max={r['TDD_I_max']:.2f}%  "
                  f"br>TDD: {r['branches_over_tdd']}")
        obs, _, _, _, info = env.step_fast(action)
        edge = ensure_edge_index(info.get("edge_index", edge), n_nodes=nb)
        obs_full = build_am_full_feeder_obs(env, obs)

    r0.pop("I1_branch_A", None)
    rows.append({"method": name, "phase": "pre-event", "step": -1, **r0})
    for t, r in snapshots.items():
        r.pop("I1_branch_A", None)
        rows.append({"method": name, "phase": f"t={t}s({t*0.1:.0f}s)", **r})

df = pd.DataFrame(rows)
out = ROOT / "results" / "thd_during_ffr"
out.mkdir(parents=True, exist_ok=True)
df.to_csv(out / "thd_during_ffr.csv", index=False)
print(f"\nSaved -> {out/'thd_during_ffr.csv'}")
print("\nSummary:")
print(df[["method","phase","freq_hz","a_P_mean","loading_mean","THD_V_PCC","buses_over",
          "TDD_I_PCC","TDD_I_max","branches_over_tdd"]].to_string(index=False))
