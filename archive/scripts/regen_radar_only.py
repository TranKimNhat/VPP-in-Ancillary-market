"""Quick radar regeneration: mini table1 (n_runs=5) + fig1_radar only.

Uses fixed ITAE (post_window 50s) and eval_radar_final_n20 topology fallback.
Output overwrites results/paper_figures/fig1_radar.pdf and the LaTeX copy.
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

RADAR_DIR = ROOT / "results" / "_radar_quick"
RADAR_DIR.mkdir(parents=True, exist_ok=True)

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
    output_dir=RADAR_DIR,
    base_reference=True,
)

print("[1/3] Mini table1 (n_runs=5, fixed ITAE)...")
ev.build_table1_ffr_comparison(n_runs=5)

print("[2/3] Per-scenario topology score (n_runs=3)...")
ev.build_topology_score_per_scenario(n_runs=3)

print("[3/3] Drawing fig1_radar...")
import matplotlib; matplotlib.use("Agg")
from src.eval.figures_style import apply_paper_style
import scripts.make_paper_figures as F

apply_paper_style()
F.OUT = ROOT / "results" / "paper_figures"
F.OUT.mkdir(parents=True, exist_ok=True)
F.fig_radar(RADAR_DIR)

print(f"\nDone -> {F.OUT / 'fig1_radar.pdf'}")
print(f"       -> IEEE_Journal_Paper_Template/img/fig1_radar.pdf")
