"""One-off: re-run the 24-topology generalization sweep exporting ITAE alongside
IAE_post (build_table2 + fig6 only), into results/_itae_probe, using the canonical
ep5700 checkpoints. Validates iae_post against eval_final3 before trusting ITAE.
"""
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.eval.eval_ffr_topology import FFRTopologyEvaluator  # noqa: E402

env_config = {
    "placement_path": "artifacts/placement/official_placement_v3.json",
    "mpc_path": "data/grid_IEEE123_complete.m",
    "seed": 42,
    "ffr_mode": "mappo_dual",
    "day_split": "eval",
}

ev = FFRTopologyEvaluator(
    env_config=env_config,
    checkpoint_path=ROOT / "artifacts/ckpt_proposed_s42/am_mappo_final.pt",
    output_dir=ROOT / "results/_itae_probe",
    gcnn_checkpoint=ROOT / "artifacts/ckpt_gcnn_ppo/final.pt",
    matd3_checkpoint=ROOT / "artifacts/ckpt_matd3/matd3_ep5700.pt",
    mlp_mappo_checkpoint=ROOT / "artifacts/ckpt_mlp_mappo/mlp_mappo_final.pt",
    base_reference=True,
)

print(">>> build_table2 (n_runs=10) ...")
ev.build_table2_topology_adaptation(n_runs=10)
print(">>> fig6 iae/itae vs distance (n_runs=5) ...")
ev.plot_iae_degradation_vs_distance(n_runs=5)
print(">>> done -> results/_itae_probe")
