"""DEFINITIVE check: is GraphSAGE's nonzero pre-event a_P legitimate economic
dispatch (rewarded by r_market = w_market * LMP*p_ref), or a spurious bias?

Test: across many base-topology steps WITHOUT any event, record per-agent a_P and
the economic target the env exposes (p_ref_target_vpp from precompute, mapped to
each agent's VPP). If a_P tracks the sign/direction of the economic signal, the
behaviour is by-design dual-product VPP dispatch — NOT a bug, and iae_total simply
conflates economics with FFR. Also quantify energy 'revenue proxy' = mean(LMP*p_ref)
that baselines (a_P=0) forgo.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np

from src.eval.eval_ffr_topology import FFRTopologyEvaluator
from src.rl.train_am_mappo import build_am_full_feeder_obs, ensure_edge_index

ev = FFRTopologyEvaluator(
    env_config={"placement_path": "artifacts/placement/official_placement_v3.json",
                "mpc_path": "data/grid_IEEE123_complete.m", "seed": 42},
    checkpoint_path=Path("artifacts/checkpoints_am_mappo_base/am_mappo_final.pt"),
    output_dir=Path("results/_diag"),
    base_reference=True,
)
env = ev.env
# map each agent -> its VPP index (0..2) via _vpp_droop_agents
agent_vpp = np.full(env.n_agents, -1, dtype=int)
for vi, (vpp_id, idxs) in enumerate(env._vpp_droop_agents.items()):
    for a in idxs:
        agent_vpp[a] = vi

pol = ev.policies["GraphSAGE-MAPPO"]
env.fixed_base_topology = True
env.ffr_mode = "mappo_dual"

aP_by_vpp = {0: [], 1: [], 2: []}
sig_by_vpp = {0: [], 1: [], 2: []}
energy_rev = []

# NO event: pure steady-state behaviour over many resets/steps
for ep in range(8):
    obs_fast, _, _ = env.reset()   # no force_event
    n_bus = len(env.net.bus.index)
    edge = ensure_edge_index(env.edge_index, n_nodes=n_bus)
    obs_full = build_am_full_feeder_obs(env, obs_fast)
    for t in range(40):
        a = pol.act(obs_full, edge, env, obs_fast=obs_fast)
        aP = np.asarray(a[:env.n_agents], dtype=np.float32)  # raw a_P per agent
        sig = np.asarray(env.p_ref_target_vpp, dtype=np.float32)  # per-VPP econ target
        for vi in range(3):
            mask = agent_vpp == vi
            if mask.any():
                aP_by_vpp[vi].append(float(aP[mask].mean()))
                sig_by_vpp[vi].append(float(sig[vi]))
        energy_rev.append(float(np.mean(np.abs(env._p_ref_last))))  # |p_ref| dispatched
        obs_fast, _, done, _, info = env.step_fast(a)
        edge = ensure_edge_index(info.get("edge_index", edge), n_nodes=n_bus)
        obs_full = build_am_full_feeder_obs(env, obs_fast)

print("Per-VPP: does a_P track the economic target p_ref_target_vpp?")
print(f"{'VPP':>4} {'mean_aP':>9} {'mean_sig':>9} {'corr(aP,sig)':>13} {'sign_match%':>11}")
for vi in range(3):
    aP = np.array(aP_by_vpp[vi]); sig = np.array(sig_by_vpp[vi])
    if len(aP) > 2 and sig.std() > 1e-9 and aP.std() > 1e-9:
        corr = float(np.corrcoef(aP, sig)[0, 1])
    else:
        corr = float("nan")
    sign_match = float(np.mean(np.sign(aP) == np.sign(sig)) * 100) if len(aP) else float("nan")
    print(f"{vi:>4} {aP.mean():9.4f} {sig.mean():9.4f} {corr:13.3f} {sign_match:11.1f}")

print(f"\nMean |p_ref| dispatched by GraphSAGE at steady-state (no event): {np.mean(energy_rev) if energy_rev else float('nan'):.5f} pu")
print("Baselines (No-FFR, Fixed-Droop) hold a_P=0 -> |p_ref|=0 (they dispatch NO energy product at all).")
