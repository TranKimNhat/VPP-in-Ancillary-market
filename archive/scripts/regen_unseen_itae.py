"""Focused: table2 + fig6 (fixed ITAE) -> results_260620, then draw the
unseen-ITAE / unseen-IAE figures. Reuses the already-computed table1.

Keeps the heavy per-scenario radar sweep OUT — only what fig_unseen_itae needs.
"""
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

PROPOSED  = ROOT / "artifacts/ckpt_proposed_s42/am_mappo_final.pt"
GCNN      = ROOT / "artifacts/ckpt_gcnn_ppo/final.pt"
MATD3     = ROOT / "artifacts/ckpt_matd3/matd3_final.pt"
MLP       = ROOT / "artifacts/ckpt_mlp_mappo/mlp_mappo_final.pt"
PLACEMENT = ROOT / "artifacts/placement/official_placement_v3.json"
MPC_PATH  = ROOT / "data/grid_IEEE123_complete.m"

OUT_DIR = ROOT / "results" / "results_260620"
OUT_DIR.mkdir(parents=True, exist_ok=True)

np.random.seed(42); torch.manual_seed(42)

from src.eval.eval_ffr_topology import FFRTopologyEvaluator

ev = FFRTopologyEvaluator(
    env_config={
        "placement_path": str(PLACEMENT),
        "mpc_path":       str(MPC_PATH),
        "seed": 42,
        "ffr_mode": "mappo_dual",
        "day_split": "eval",
    },
    checkpoint_path=PROPOSED,
    gcnn_checkpoint=GCNN,
    matd3_checkpoint=MATD3,
    mlp_mappo_checkpoint=MLP,
    output_dir=OUT_DIR,
    base_reference=True,
)

print("[1/3] Table 2 (topology adaptation, n_runs=10)...", flush=True)
ev.build_table2_topology_adaptation(n_runs=10)

print("[2/3] Fig.6 (IAE/ITAE vs d_E, n_runs=5)...", flush=True)
ev.plot_iae_degradation_vs_distance(n_runs=5)

print("[3/3] Drawing fig_unseen_itae + fig_unseen_iae...", flush=True)
import matplotlib; matplotlib.use("Agg")
from src.eval.figures_style import apply_paper_style
import scripts.make_paper_figures as F

apply_paper_style()
F.OUT = ROOT / "results" / "paper_figures"
F.OUT.mkdir(parents=True, exist_ok=True)
F.fig_unseen_itae(OUT_DIR)
F.fig_unseen_iae(OUT_DIR)

print("\nDone.", flush=True)
print(f"  -> {F.OUT / 'fig_unseen_itae.pdf'}", flush=True)
print(f"  -> IEEE_Journal_Paper_Template/img/fig_unseen_itae.pdf", flush=True)
