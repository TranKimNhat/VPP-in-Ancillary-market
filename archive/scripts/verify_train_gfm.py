"""Verify how many GFMs the training env had, via two independent signals:
  (1) GFM entries in the placement file the checkpoint args point to.
  (2) Frequency-channel fingerprint: the MLP checkpoint's obs_normalizer
      mean/std over the per-bus delta_f / rocof channels, compared with a
      fresh rollout of the CURRENT 2-GFM env. A 6-GFM env would damp the
      frequency excursions differently, shifting these stats.
"""
from __future__ import annotations
import sys, json
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import numpy as np
import torch

# ---- (1) Placement file the checkpoint was trained with -------------------
pl = json.loads((ROOT / "artifacts/placement/official_placement_v3.json").read_text())
gfm = pl.get("gfm", {})
print("=" * 60)
print("(1) Placement file official_placement_v3.json")
print(f"    GFM entries: {len(gfm)}")
for gid, g in gfm.items():
    print(f"      {gid}: bus={g.get('bus')}, type={g.get('type', g.get('mode','?'))}, "
          f"S={g.get('s_mva', g.get('mva','?'))}")

# ---- (2a) Checkpoint frequency fingerprint --------------------------------
ck = torch.load(ROOT / "artifacts/ckpt_mlp_mappo/mlp_mappo_final.pt",
                map_location="cpu", weights_only=False)
on = ck["obs_normalizer"]
mean = np.asarray(on["mean"]); var = np.asarray(on["var"])
std = np.sqrt(var)
# channels 0,1 = delta_f_norm, rocof_norm (freq skip-connection convention)
print("\n(2) MLP checkpoint obs_normalizer (training fingerprint)")
print(f"    shape={mean.shape}, count={on['count']:.0f}")
print(f"    delta_f channel (col0): mean over buses={mean[:,0].mean():+.4f}, "
      f"std={std[:,0].mean():.4f}")
print(f"    rocof   channel (col1): mean over buses={mean[:,1].mean():+.4f}, "
      f"std={std[:,1].mean():.4f}")

# ---- (2b) Fresh rollout of current 2-GFM env ------------------------------
from src.env.microgrid_env_dual import MicrogridEnvDual
from src.rl.train_am_mappo import build_am_full_feeder_obs

env = MicrogridEnvDual(
    placement_path="artifacts/placement/official_placement_v3.json",
    mpc_path="data/grid_IEEE123_complete.m",
    seed=42, ffr_mode="mappo_dual", day_split="train",
)
from src.env.events import EventConfig
acc = []
rng = np.random.default_rng(0)
n_ep = 12
for ep in range(n_ep):
    obs_fast, _, _ = env.reset(options={"force_event": EventConfig(
        type="gen_trip", delta_P_mw=-3.9, location=67, t_inject=30.0)})
    for t in range(120):
        a = rng.uniform(-1, 1, size=2*env.n_agents + len(env._vpp_droop_agents)).astype(np.float32)
        obs_fast, _, d, _, info = env.step_fast(a)
        of = build_am_full_feeder_obs(env, obs_fast)
        acc.append(np.asarray(of)[:, :2])
A = np.stack(acc, 0)  # (T, 123, 2)
print(f"\n    Current 2-GFM env rollout ({n_ep} eps, gen_trip):")
print(f"    delta_f channel: mean over buses={A[:,:,0].mean():+.4f}, std={A[:,:,0].std():.4f}")
print(f"    rocof   channel: mean over buses={A[:,:,1].mean():+.4f}, std={A[:,:,1].std():.4f}")
print(f"    GFM ext_grid roots in this env: "
      f"{sorted(set(int(b) for b in env.net.ext_grid['bus'])) if not env.net.ext_grid.empty else 'none'}")
print("\n  -> If the checkpoint freq-channel stats are consistent with this "
      "2-GFM rollout, training was on 2 GFM.")
