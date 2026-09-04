"""Full evaluation pipeline -> results/results_260620.

Steps:
  1. eval_ffr_topology run_all (all 6 policies, 24 unseen topologies, n_runs=20)
  2. make_paper_figures (radar / severity / unseen-ITAE / pareto)
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

OUT_DIR = ROOT / "results" / "results_260620"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PROPOSED  = ROOT / "artifacts/ckpt_proposed_s42/am_mappo_final.pt"
GCNN      = ROOT / "artifacts/ckpt_gcnn_ppo/final.pt"
MATD3     = ROOT / "artifacts/ckpt_matd3/matd3_final.pt"
MLP       = ROOT / "artifacts/ckpt_mlp_mappo/mlp_mappo_final.pt"
PLACEMENT = ROOT / "artifacts/placement/official_placement_v3.json"
MPC_PATH  = ROOT / "data/grid_IEEE123_complete.m"

N_RUNS = 20
SEED   = 42

# ── 1. Evaluation ────────────────────────────────────────────────────────────
print("=" * 60)
print("STEP 1 / 2 — FFR + topology evaluation")
print("=" * 60)

np.random.seed(SEED)
torch.manual_seed(SEED)

from src.eval.eval_ffr_topology import FFRTopologyEvaluator

env_config = {
    "placement_path": str(PLACEMENT),
    "mpc_path":       str(MPC_PATH),
    "seed":           SEED,
    "ffr_mode":       "mappo_dual",
    "day_split":      "eval",
}

ev = FFRTopologyEvaluator(
    env_config=env_config,
    checkpoint_path=PROPOSED,
    gcnn_checkpoint=GCNN,
    matd3_checkpoint=MATD3,
    mlp_mappo_checkpoint=MLP,
    output_dir=OUT_DIR,
    base_reference=True,   # all 24 reconfig topologies treated as unseen
)

ev.run_all(n_runs=N_RUNS)

# ── 2. Figures ───────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 2 / 2 — Paper figures")
print("=" * 60)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.eval.figures_style import apply_paper_style
import scripts.make_paper_figures as F

apply_paper_style()
print(f"Reading {OUT_DIR}")
F.OUT = ROOT / "results" / "paper_figures"
F.OUT.mkdir(parents=True, exist_ok=True)

F.fig_radar(OUT_DIR)
F.fig_severity(OUT_DIR)
F.fig_unseen_iae(OUT_DIR)
try:
    F.fig_unseen_itae(OUT_DIR)
except (KeyError, FileNotFoundError) as exc:
    print(f"  [skip] fig_unseen_itae: {exc}")

DSO_CSV = ROOT / "results/ffr_topology_baseref_final/dso_cost_per_event.csv"
F.fig_pareto(DSO_CSV)

print(f"\nDone. Figures -> {F.OUT}")
print(f"CSVs       -> {OUT_DIR}")
