"""Final RoCoF 500 ms table: build_table1 at n_runs=20, mean+/-std per scenario/method."""
from pathlib import Path
import pandas as pd
from src.eval.eval_ffr_topology import FFRTopologyEvaluator

ev = FFRTopologyEvaluator(
    env_config={"placement_path": "artifacts/placement/official_placement_v3.json",
                "mpc_path": "data/grid_IEEE123_complete.m", "seed": 42},
    checkpoint_path=Path("artifacts/ckpt_proposed_s42/am_mappo_final.pt"),
    mlp_mappo_checkpoint=Path("artifacts/ckpt_mlp_mappo/mlp_mappo_final.pt"),
    gcnn_checkpoint=Path("artifacts/ckpt_gcnn_ppo/final.pt"),
    matd3_checkpoint=Path("artifacts/ckpt_matd3/matd3_ep5700.pt"),
    output_dir=Path("results/_rocof_final"),
    base_reference=True,
)

df = ev.build_table1_ffr_comparison(n_runs=20)

sub = df[df["metric"] == "rocof_max_500ms_hz_s"].copy()
sub["cell"] = sub["mean"].round(2).astype(str) + " ± " + sub["std"].round(2).astype(str)
piv = sub.pivot_table(index="scenario", columns="method", values="mean").round(3)
print("=== rocof_max_500ms_hz_s (mean Hz/s) ===")
print(piv.to_string())
print("\n=== mean +/- std ===")
piv2 = sub.pivot_table(index="scenario", columns="method", values="cell", aggfunc="first")
print(piv2.to_string())
print("\nPeak 500ms RoCoF (max mean):",
      round(float(sub["mean"].max()), 3), "Hz/s")
sub.to_csv("results/_rocof_final/rocof_500ms_table.csv", index=False)
print("Saved -> results/_rocof_final/rocof_500ms_table.csv")
