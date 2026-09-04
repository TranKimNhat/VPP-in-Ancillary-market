"""Forensic: what environment structure do the trained checkpoints encode?
Goal: find evidence of how many GFMs the training env had."""
from __future__ import annotations
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import torch
import numpy as np

CKPTS = [
    ("PROPOSED", "artifacts/ckpt_proposed_s42/am_mappo_final.pt"),
    ("MATD3",    "artifacts/ckpt_matd3/matd3_final.pt"),
    ("MLP",      "artifacts/ckpt_mlp_mappo/mlp_mappo_final.pt"),
    ("GCNN",     "artifacts/ckpt_gcnn_ppo/final.pt"),
]

for name, rel in CKPTS:
    p = ROOT / rel
    print("=" * 60)
    print(name, "->", rel)
    if not p.exists():
        print("  MISSING"); continue
    ck = torch.load(p, map_location="cpu", weights_only=False)
    if not isinstance(ck, dict):
        print("  (not a dict, type=%s)" % type(ck)); continue
    print("  top-level keys:", list(ck.keys()))
    # any embedded config / placement / args
    for k, v in ck.items():
        kl = k.lower()
        if any(t in kl for t in ["config", "placement", "env", "arg", "gfm", "meta"]):
            if hasattr(v, "shape"):
                print("   [%s] tensor%s" % (k, tuple(v.shape)))
            elif hasattr(v, "__dict__"):
                print("   [%s] obj attrs: %s" % (k, {a: getattr(v, a)
                      for a in vars(v)} if hasattr(v, "__dict__") else v))
            else:
                print("   [%s] = %s" % (k, v))
    # obs_normalizer fingerprint (shape encodes n_bus x obs_feat)
    on = ck.get("obs_normalizer")
    if isinstance(on, dict) and "mean" in on:
        m = np.asarray(on["mean"])
        print("  obs_normalizer mean shape:", m.shape, " count:", on.get("count"))
    # config dataclass
    cfg = ck.get("config")
    if cfg is not None and hasattr(cfg, "__dict__"):
        print("  config:", vars(cfg))
    # infer n_agents from actor/encoder shapes
    sd = ck.get("agent_state_dict", ck.get("model_state_dict", ck.get("actors", ck)))
    if isinstance(sd, dict):
        for key in list(sd.keys())[:6]:
            val = sd[key]
            if hasattr(val, "shape"):
                print("   wkey %s -> %s" % (key, tuple(val.shape)))
