"""Export precomputed_v3 to signal CSVs consumed by GridEnvironment.

Source  : data/precomputed_v3  (seed=42, 200 days, 96 steps/day @ 15-min)
Outputs (written once, shared identically by BOTH training arms):
  data/oedisi-ieee123-main/profiles/layer1_vpp/layer1_pref.csv
  data/oedisi-ieee123-main/profiles/layer0_hourly/layer0_zone_prices.csv
  data/oedisi-ieee123-main/profiles/layer0_hourly/layer0_dlmp_per_bus.csv

Aggregation rules (pre-registered, not changed after viewing output):
  - P_ref / R_commit / Q_ref : sum of VPP1+VPP2+VPP3, then mean over all 200 days per step
  - lambda_e per bus         : lambda_p2p (system mean over 200 days) — zone-level prices
                               vary per zone (z1/z2/z4), but bus→zone mapping is absent in
                               env_config so system mean is used to avoid injecting zone bias
  - lambda_q per bus         : lambda_q (constant 2.5 in precomputed_v3, same for all buses)
  - DLMP decomposition       : lambda_p_total = lambda_p_energy (no network decomp available);
                               loss/congestion/voltage components = 0.0
  - zone_prices              : offpeak/median/peak map to z1/z2/z4 price columns respectively;
                               reserve_price from corresponding lambda_as_z* column
  - hour column              : uses `step` (0-95), NOT fractional hour, to align with env
                               max_steps=96 array indexing

Fairness note: this script produces deterministic output given fixed input parquet files.
Both arms (baseline and method) point to the same output files via env_config.yaml.
No arm-specific processing is applied.
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np
import pandas as pd

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "data" / "precomputed_v3"
OUT_L1 = REPO_ROOT / "data" / "oedisi-ieee123-main" / "profiles" / "layer1_vpp"
OUT_L0 = REPO_ROOT / "data" / "oedisi-ieee123-main" / "profiles" / "layer0_hourly"


def load_all_days() -> pd.DataFrame:
    frames = []
    parquets = sorted(SRC.glob("day_*.parquet"))
    if not parquets:
        sys.exit(f"ERROR: no parquet files found in {SRC}")
    for p in parquets:
        df = pd.read_parquet(p)
        df["_day_idx"] = int(p.stem.split("_")[1])
        frames.append(df)
    all_days = pd.concat(frames, ignore_index=True)
    print(f"  Loaded {all_days['_day_idx'].nunique()} days, {len(all_days)} total rows")
    return all_days


def export_layer1_pref(all_days: pd.DataFrame) -> None:
    OUT_L1.mkdir(parents=True, exist_ok=True)

    all_days = all_days.copy()
    all_days["P_ref_total"] = all_days["p_ref_vpp1"] + all_days["p_ref_vpp2"] + all_days["p_ref_vpp3"]
    all_days["R_commit_total"] = all_days["r_as_vpp1"] + all_days["r_as_vpp2"] + all_days["r_as_vpp3"]
    all_days["Q_ref_total"] = all_days["q_commit_vpp1"] + all_days["q_commit_vpp2"] + all_days["q_commit_vpp3"]

    per_step = (
        all_days.groupby("step", as_index=False)
        .agg(
            P_ref=("P_ref_total", "mean"),
            R_commit=("R_commit_total", "mean"),
            Q_ref=("Q_ref_total", "mean"),
            lambda_q_expected=("lambda_q", "mean"),
            price_energy_expected=("lambda_p2p", "mean"),
            price_reserve_expected=("lambda_as_ffr", "mean"),
        )
        .sort_values("step")
        .rename(columns={"step": "hour"})
    )
    per_step["price_energy_robust"] = per_step["price_energy_expected"] * 0.98
    per_step["price_reserve_robust"] = per_step["price_reserve_expected"] * 0.98
    per_step["SoC"] = 0.5
    per_step["solver_status"] = "precomputed_v3"
    per_step["curtailment_ratio"] = float("nan")
    per_step["reoptimize_next_cycle"] = False

    out = OUT_L1 / "layer1_pref.csv"
    per_step.to_csv(out, index=False)
    print(f"  -> {out}  ({len(per_step)} rows)")
    print(f"     P_ref mean={per_step['P_ref'].mean():.3f} MW, "
          f"R_commit mean={per_step['R_commit'].mean():.3f} MW, "
          f"Q_ref mean={per_step['Q_ref'].mean():.3f} MVAR")


def export_zone_prices(all_days: pd.DataFrame) -> None:
    OUT_L0.mkdir(parents=True, exist_ok=True)

    # offpeak=z1, median=z2, peak=z4 — pre-registered mapping; not chosen post-results
    zone_map = [
        ("offpeak", "lambda_p2p_z1", "lambda_as_z1"),
        ("median",  "lambda_p2p_z2", "lambda_as_z2"),
        ("peak",    "lambda_p2p_z4", "lambda_as_z4"),
    ]
    rows = []
    for day_label, col_e, col_r in zone_map:
        per_step = (
            all_days.groupby("step", as_index=False)
            .agg(energy_price=(col_e, "mean"), reserve_price=(col_r, "mean"))
            .sort_values("step")
        )
        for _, row in per_step.iterrows():
            rows.append({
                "day": day_label,
                "hour": int(row["step"]),
                "zone": day_label,
                "zone_id": day_label,
                "energy_price": float(row["energy_price"]),
                "reserve_price": float(row["reserve_price"]),
            })

    out = OUT_L0 / "layer0_zone_prices.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"  -> {out}  ({len(rows)} rows)")


def export_dlmp_per_bus(all_days: pd.DataFrame) -> None:
    OUT_L0.mkdir(parents=True, exist_ok=True)

    # per-step system-mean prices (averaged over all 200 days)
    per_step = (
        all_days.groupby("step", as_index=False)
        .agg(
            lambda_p_total=("lambda_p2p", "mean"),
            lambda_q_total=("lambda_q", "mean"),
        )
        .sort_values("step")
    )

    # load bus list from IEEE 123 net
    sys.path.insert(0, str(REPO_ROOT))
    from src.env.IEEE123bus import build_ieee123_net
    net = build_ieee123_net(
        mode="matpower", balanced=True, convert_switches=True,
        slack_zones=None, source_mode="publish",
    )
    bus_ids = [int(b) for b in net.bus.index]

    rows = []
    lp_arr = per_step["lambda_p_total"].to_numpy(dtype=float)
    lq_arr = per_step["lambda_q_total"].to_numpy(dtype=float)
    steps = per_step["step"].to_numpy(dtype=int)

    for bus_id in bus_ids:
        for i, step in enumerate(steps):
            lp = float(lp_arr[i])
            lq = float(lq_arr[i])
            rows.append({
                "day": "expected",
                "hour": int(step),
                "bus_id": bus_id,
                "lambda_p_total": lp,
                "lambda_q_total": lq,
                "lambda_p_energy": lp,   # all residual in energy; no network decomp available
                "lambda_p_loss": 0.0,
                "lambda_p_congestion": 0.0,
                "lambda_p_voltage": 0.0,
            })

    out = OUT_L0 / "layer0_dlmp_per_bus.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"  -> {out}  ({len(rows)} rows, {len(bus_ids)} buses × {len(steps)} steps)")
    print(f"     lambda_e mean={lp_arr.mean():.3f}, lambda_q mean={lq_arr.mean():.3f}")


def main() -> None:
    print(f"Source: {SRC}")
    all_days = load_all_days()
    print("Exporting layer1_pref.csv ...")
    export_layer1_pref(all_days)
    print("Exporting layer0_zone_prices.csv ...")
    export_zone_prices(all_days)
    print("Exporting layer0_dlmp_per_bus.csv ...")
    export_dlmp_per_bus(all_days)
    print("Done.")


if __name__ == "__main__":
    main()
