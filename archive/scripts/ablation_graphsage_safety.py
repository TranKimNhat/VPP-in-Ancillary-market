"""Ablation: decompose GraphSAGE iae_post into RL-policy vs nadir-safety-layer.

Runs the trained GraphSAGE policy on the S2 gen-trip over the base topology and
all 24 unseen reconfig topologies, with the nadir safety layer ON vs OFF. The gap
isolates how much of the proposed system's frequency performance comes from the
learned policy itself vs the in-the-loop safety projection.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np

from src.eval.eval_ffr_topology import FFRTopologyEvaluator

ev = FFRTopologyEvaluator(
    env_config={"placement_path": "artifacts/placement/official_placement_v3.json",
                "mpc_path": "data/grid_IEEE123_complete.m", "seed": 42},
    checkpoint_path=Path("artifacts/checkpoints_am_mappo_base/am_mappo_final.pt"),
    output_dir=Path("results/_ablation_safety"),
    base_reference=True,
)
event = ev.scenarios["S2_gen_trip"]
pol = ev.policies["GraphSAGE-MAPPO"]
N = 5  # runs per topology

print(f"{'safety':>8} {'split':>7} {'nadir_hz':>9} {'iae_post':>9}")
for safety in [True, False]:
    pol.nadir_safety = safety
    for split, topos in [("base", ev.train_topologies), ("unseen", ev.test_topologies)]:
        nad, iae = [], []
        for ti in topos:
            for _ in range(N):
                m = ev.run_episode(pol, event=event, topology_idx=ti)
                nad.append(m.nadir_hz); iae.append(m.iae_post)
        print(f"{str(safety):>8} {split:>7} {np.mean(nad):9.4f} {np.mean(iae):9.4f}")
# restore default
pol.nadir_safety = True
