"""A/B: does fixing the connected-mask (h_sys 0.97 -> 1.34) change any reported metric?

h_sys feeds ONLY mpc_correction's internal predictor. mpc_correction edits
delta_p_set, which feeds the LTI freq model, so the effect (if any) is indirect.
Run the proposed policy on S2 with (A) the default mask and (B) all-GFM-connected
(h_sys=1.34), comparing nadir / IAE / RoCoF-500ms / settling.
"""
import types
from pathlib import Path
import numpy as np
from src.eval.eval_ffr_topology import FFRTopologyEvaluator

ev = FFRTopologyEvaluator(
    env_config={"placement_path": "artifacts/placement/official_placement_v3.json",
                "mpc_path": "data/grid_IEEE123_complete.m", "seed": 42},
    checkpoint_path=Path("artifacts/ckpt_proposed_s42/am_mappo_final.pt"),
    mlp_mappo_checkpoint=Path("artifacts/ckpt_mlp_mappo/mlp_mappo_final.pt"),
    gcnn_checkpoint=Path("artifacts/ckpt_gcnn_ppo/final.pt"),
    matd3_checkpoint=Path("artifacts/ckpt_matd3/matd3_ep5700.pt"),
    output_dir=Path("results/_ab_hsys"),
    base_reference=True,
)
ev.env.hires_substeps = 10
pol = ev.policies["GraphSAGE-MAPPO"]
event = ev.scenarios["S2_gen_trip"]
lti = ev.env.freq_dyn_lti
N = 10


# Use a RECONFIG (unseen) topology where the mask is actually 2/6 (h_sys~0.97),
# not the base feeder (6/6). This is where fixing the mask could move metrics.
RECONFIG_TOPO = ev.test_topologies[0] if ev.test_topologies else 0
print(f"reconfig topology under test: {RECONFIG_TOPO}")


def run(label):
    h_seen, nad, iae_p, iae_t, settle, rocof = [], [], [], [], [], []
    for _ in range(N):
        topo = RECONFIG_TOPO
        m = ev.run_episode(pol, event=event, topology_idx=topo)
        h_seen.append(lti.h_sys)
        nad.append(m.nadir_hz); iae_p.append(m.iae_post); iae_t.append(m.iae_total)
        settle.append(m.settling_time_s); rocof.append(m.rocof_max_500ms_hz_s)
    f = lambda x: (float(np.mean(x)), float(np.std(x)))
    print(f"[{label}] h_sys={np.mean(h_seen):.3f}  "
          f"nadir={f(nad)[0]:.4f}+/-{f(nad)[1]:.4f}  "
          f"IAE_post={f(iae_p)[0]:.4f}  IAE_tot={f(iae_t)[0]:.4f}  "
          f"settle={f(settle)[0]:.3f}  rocof500={f(rocof)[0]:.4f}")


# A: default mask (the bug -> ~0.97)
run("A default mask")

# B: force all GFM connected every reset -> h_sys = 1.34
def _all(self, ids=None):
    self._connected_mask = np.ones(self.n_gfm, dtype=bool)
lti.update_topology = types.MethodType(_all, lti)
run("B all-connected ")
