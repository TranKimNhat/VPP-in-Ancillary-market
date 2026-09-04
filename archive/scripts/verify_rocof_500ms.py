"""Quick verification: build_table1 with n_runs=3, compare 500ms vs 1s RoCoF per policy."""
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
    output_dir=Path("results/_rocof_verify"),
    base_reference=True,
)

df = ev.build_table1_ffr_comparison(n_runs=3)

sub = df[df["metric"].isin(["rocof_max_500ms_hz_s", "rocof_max_coi_hz_s", "rocof_max_hz_s"])].copy()
piv = sub.pivot_table(index=["scenario", "method"], columns="metric", values="mean").round(3)
pd.set_option("display.width", 200); pd.set_option("display.max_rows", 200)
print(piv.to_string())
print("\nPeak 500ms RoCoF across all scenarios/methods:",
      round(float(sub[sub["metric"] == "rocof_max_500ms_hz_s"]["mean"].max()), 3), "Hz/s")
