"""Diagnostic: are No-FFR / Fixed-Droop legit, or buggy-advantaged?

For each policy run ONE base-topology S2 episode and split the |Δf| integral into
PRE-event (steps <30) and POST-event (>=30). Also record the mean |a_P| (power-
reference action) and mean K_droop the policy actually injects. If the learned
method perturbs the steady state pre-event (a_P != 0) while baselines stay quiet
(a_P == 0), that explains why 'doing nothing' wins on total IAE.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np

from src.eval.eval_ffr_topology import (
    FFRTopologyEvaluator, NoFFRPolicy, FixedDroopPolicy, GraphSAGEMAPPOPolicy,
)
from src.rl.train_am_mappo import build_am_full_feeder_obs, ensure_edge_index

ev = FFRTopologyEvaluator(
    env_config={"placement_path": "artifacts/placement/official_placement_v3.json",
                "mpc_path": "data/grid_IEEE123_complete.m", "seed": 42},
    checkpoint_path=Path("artifacts/checkpoints_am_mappo_base/am_mappo_final.pt"),
    output_dir=Path("results/_diag"),
    base_reference=True,
)
env = ev.env
event = ev.scenarios["S2_gen_trip"]
EV_STEP = int(event.t_inject)  # 30

pols = {
    "No FFR": ev.policies["No FFR"],
    "Fixed Droop": ev.policies["Fixed Droop"],
    "GraphSAGE": ev.policies["GraphSAGE-MAPPO"],
}

print(f"{'policy':14s} {'IAE_pre':>9s} {'IAE_post':>9s} {'IAE_tot':>9s} {'|aP|_pre':>9s} {'|aP|_post':>10s} {'K_pre':>8s}")
for name, pol in pols.items():
    env.fixed_base_topology = True
    env.ffr_mode = getattr(pol, "ffr_mode", "droop")
    obs_fast, _, _ = env.reset(options={"force_event": __import__("copy").deepcopy(event)})
    env.fixed_base_topology = False
    n_bus = len(env.net.bus.index)
    edge = ensure_edge_index(env.edge_index, n_nodes=n_bus)
    obs_full = build_am_full_feeder_obs(env, obs_fast)
    df_pre, df_post, ap_pre, ap_post, k_pre = [], [], [], [], []
    for t in range(300):
        a = pol.act(obs_full, edge, env, obs_fast=obs_fast)
        obs_fast, _, done, _, info = env.step_fast(a)
        edge = ensure_edge_index(info.get("edge_index", edge), n_nodes=n_bus)
        obs_full = build_am_full_feeder_obs(env, obs_fast)
        df = abs(float(env.freq_dyn_lti.get_state().delta_f_hz))
        ap = float(np.mean(np.abs(env._p_ref_last)))      # mean |a_P| over agents
        kk = float(np.mean(env._k_droop_last))            # mean K_droop injected
        if t < EV_STEP:
            df_pre.append(df); ap_pre.append(ap); k_pre.append(kk)
        else:
            df_post.append(df); ap_post.append(ap)
    print(f"{name:14s} {np.sum(df_pre):9.3f} {np.sum(df_post):9.3f} "
          f"{np.sum(df_pre)+np.sum(df_post):9.3f} {np.mean(ap_pre):9.4f} "
          f"{np.mean(ap_post):10.4f} {np.mean(k_pre):8.4f}")
