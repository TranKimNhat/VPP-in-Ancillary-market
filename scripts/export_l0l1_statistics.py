from __future__ import annotations

import glob
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


PRECOMPUTED_DIR = "data/precomputed"
OUTPUT_DIR = "artifacts/l0l1_stats"
DT_HOURS = 0.25


def load_all_parquet(precomputed_dir: str) -> pd.DataFrame:
    """Load và concat 200 files thành 1 DataFrame lớn."""
    files = sorted(glob.glob(f"{precomputed_dir}/day_*.parquet"))
    dfs = []
    for i, f in enumerate(files):
        df = pd.read_parquet(f)
        df["day"] = i
        dfs.append(df)
    if not dfs:
        raise RuntimeError(f"No parquet files found in {precomputed_dir}")
    return pd.concat(dfs, ignore_index=True)


def compute_summary_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """Tính percentile statistics cho tất cả numeric columns."""
    metrics: List[Dict[str, object]] = []
    cols_of_interest: List[Tuple[str, str]] = [
        ("p_ref_vpp1", "MW"),
        ("p_ref_vpp2", "MW"),
        ("p_ref_vpp3", "MW"),
        ("r_as_vpp1", "MW"),
        ("r_as_vpp2", "MW"),
        ("r_as_vpp3", "MW"),
        ("q_commit_vpp1", "Mvar"),
        ("q_commit_vpp2", "Mvar"),
        ("q_commit_vpp3", "Mvar"),
        ("lambda_p2p", "$/MWh"),
        ("lambda_as_ffr", "$/MWh"),
        ("lambda_as_pfr", "$/MWh"),
        ("lambda_as_sfr", "$/MWh"),
        ("lambda_q", "$/Mvar-period"),
        ("pv_pu", "p.u."),
        ("wind_mw", "MW"),
        ("load_z1", "MW"),
        ("load_z2", "MW"),
        ("load_z3", "MW"),
        ("load_z4", "MW"),
        ("delta_p_cont", "MW"),
    ]
    for col, unit in cols_of_interest:
        if col not in df.columns:
            continue
        s = df[col]
        metrics.append(
            {
                "metric": col,
                "unit": unit,
                "mean": round(float(s.mean()), 4),
                "std": round(float(s.std()), 4),
                "min": round(float(s.min()), 4),
                "p25": round(float(s.quantile(0.25)), 4),
                "p50": round(float(s.quantile(0.50)), 4),
                "p75": round(float(s.quantile(0.75)), 4),
                "max": round(float(s.max()), 4),
            }
        )

    if "freq_event_flag" in df.columns:
        freq_rate = float(df["freq_event_flag"].mean())
        metrics.append(
            {
                "metric": "freq_event_rate",
                "unit": "fraction",
                "mean": round(freq_rate, 4),
                "std": 0,
                "min": 0,
                "p25": 0,
                "p50": 1,
                "p75": 1,
                "max": 1,
            }
        )

    return pd.DataFrame(metrics)


def compute_daily_summary(df: pd.DataFrame) -> pd.DataFrame:
    daily = df.groupby("day").agg(
        season=("season", "first"),
        day_type=("day_type", "first"),
        p_ref_vpp1_mean=("p_ref_vpp1", "mean"),
        p_ref_vpp2_mean=("p_ref_vpp2", "mean"),
        p_ref_vpp3_mean=("p_ref_vpp3", "mean"),
        r_as_total_mean=(
            "r_as_vpp1",
            "mean",
        ),
        q_commit_total_mean=(
            "q_commit_vpp1",
            "mean",
        ),
        lambda_p2p_mean=("lambda_p2p", "mean"),
        lambda_as_ffr_mean=("lambda_as_ffr", "mean"),
        pv_peak_pu=("pv_pu", "max"),
        wind_mean_mw=("wind_mw", "mean"),
        load_z1_mean=("load_z1", "mean"),
        load_z2_mean=("load_z2", "mean"),
        load_z3_mean=("load_z3", "mean"),
        load_z4_mean=("load_z4", "mean"),
        freq_event_count=("freq_event_flag", "sum"),
        p_ref_vpp1_std=("p_ref_vpp1", "std"),
        p_ref_vpp2_std=("p_ref_vpp2", "std"),
        p_ref_vpp3_std=("p_ref_vpp3", "std"),
    )

    daily["r_as_total_mean"] = (
        df.groupby("day")["r_as_vpp1"].mean()
        + df.groupby("day")["r_as_vpp2"].mean()
        + df.groupby("day")["r_as_vpp3"].mean()
    )
    daily["q_commit_total_mean"] = (
        df.groupby("day")["q_commit_vpp1"].mean()
        + df.groupby("day")["q_commit_vpp2"].mean()
        + df.groupby("day")["q_commit_vpp3"].mean()
    )

    daily["total_load_mean_mw"] = (
        daily["load_z1_mean"]
        + daily["load_z2_mean"]
        + daily["load_z3_mean"]
        + daily["load_z4_mean"]
    )

    daily = daily.reset_index()
    return daily[
        [
            "day",
            "season",
            "day_type",
            "p_ref_vpp1_mean",
            "p_ref_vpp2_mean",
            "p_ref_vpp3_mean",
            "r_as_total_mean",
            "q_commit_total_mean",
            "lambda_p2p_mean",
            "lambda_as_ffr_mean",
            "pv_peak_pu",
            "wind_mean_mw",
            "total_load_mean_mw",
            "freq_event_count",
            "p_ref_vpp1_std",
            "p_ref_vpp2_std",
            "p_ref_vpp3_std",
        ]
    ]


def compute_diurnal_profile(df: pd.DataFrame) -> pd.DataFrame:
    diurnal = df.groupby("step").agg(
        hour=("hour", "mean"),
        p_ref_vpp1_mean=("p_ref_vpp1", "mean"),
        p_ref_vpp1_std=("p_ref_vpp1", "std"),
        p_ref_vpp2_mean=("p_ref_vpp2", "mean"),
        p_ref_vpp2_std=("p_ref_vpp2", "std"),
        p_ref_vpp3_mean=("p_ref_vpp3", "mean"),
        p_ref_vpp3_std=("p_ref_vpp3", "std"),
        r_as_vpp1_mean=("r_as_vpp1", "mean"),
        r_as_vpp2_mean=("r_as_vpp2", "mean"),
        r_as_vpp3_mean=("r_as_vpp3", "mean"),
        pv_pu_mean=("pv_pu", "mean"),
        wind_mw_mean=("wind_mw", "mean"),
        load_total_mean=("load_z1", "mean"),
        lambda_p2p_mean=("lambda_p2p", "mean"),
    )

    diurnal["load_total_mean"] = (
        df.groupby("step")["load_z1"].mean()
        + df.groupby("step")["load_z2"].mean()
        + df.groupby("step")["load_z3"].mean()
        + df.groupby("step")["load_z4"].mean()
    )

    diurnal = diurnal.reset_index()
    return diurnal[
        [
            "step",
            "hour",
            "p_ref_vpp1_mean",
            "p_ref_vpp1_std",
            "p_ref_vpp2_mean",
            "p_ref_vpp2_std",
            "p_ref_vpp3_mean",
            "p_ref_vpp3_std",
            "r_as_vpp1_mean",
            "r_as_vpp2_mean",
            "r_as_vpp3_mean",
            "pv_pu_mean",
            "wind_mw_mean",
            "load_total_mean",
            "lambda_p2p_mean",
        ]
    ]


def estimate_vpp_revenue(df: pd.DataFrame) -> pd.DataFrame:
    dT = DT_HOURS
    results = []
    for vpp_id in [1, 2, 3]:
        p2p_rev = (
            df["lambda_p2p"] * df[f"p_ref_vpp{vpp_id}"].clip(lower=0) * dT
        ).mean() * 96
        as_rev = (df["lambda_as_ffr"] * df[f"r_as_vpp{vpp_id}"] * dT).mean() * 96
        q_rev = (df["lambda_q"] * df[f"q_commit_vpp{vpp_id}"] * dT).mean() * 96
        deg_cost = 0.02 * df[f"p_ref_vpp{vpp_id}"].abs().mean() * dT * 1000 * 96
        results.append(
            {
                "vpp": f"VPP_{vpp_id}",
                "p2p_revenue_usd": round(float(p2p_rev), 2),
                "as_revenue_usd": round(float(as_rev), 2),
                "q_revenue_usd": round(float(q_rev), 2),
                "total_revenue_usd": round(float(p2p_rev + as_rev + q_rev), 2),
                "bess_deg_cost_usd": round(float(deg_cost), 2),
                "net_revenue_usd": round(float(p2p_rev + as_rev + q_rev - deg_cost), 2),
            }
        )
    return pd.DataFrame(results)


def compute_solve_performance(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for solver in ["l0", "l1"]:
        time_col = f"{solver}_solve_time_ms"
        status_col = f"{solver}_status"
        if time_col not in df.columns or status_col not in df.columns:
            continue
        times = df[time_col].dropna()
        if times.empty:
            continue
        statuses = df[status_col].fillna("")
        rows.append(
            {
                "solver": solver,
                "n_solves": int(len(times)),
                "time_mean_ms": round(float(times.mean()), 2),
                "time_std_ms": round(float(times.std()), 2),
                "time_p95_ms": round(float(times.quantile(0.95)), 2),
                "status_optimal_rate": round(float((statuses == "optimal").mean()), 4),
                "status_inaccurate_rate": round(
                    float((statuses == "optimal_inaccurate").mean()), 4
                ),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading 200 parquet files...")
    df = load_all_parquet(PRECOMPUTED_DIR)
    print(f"Total rows: {len(df)} (expect 19,200 = 200x96)")

    print("Computing statistics...")
    summary = compute_summary_statistics(df)
    daily = compute_daily_summary(df)
    diurnal = compute_diurnal_profile(df)
    revenue = estimate_vpp_revenue(df)
    performance = compute_solve_performance(df)

    summary.to_csv(os.path.join(OUTPUT_DIR, "summary_statistics.csv"), index=False)
    daily.to_csv(os.path.join(OUTPUT_DIR, "daily_summary.csv"), index=False)
    diurnal.to_csv(os.path.join(OUTPUT_DIR, "diurnal_profile.csv"), index=False)
    revenue.to_csv(os.path.join(OUTPUT_DIR, "vpp_revenue_estimate.csv"), index=False)
    performance.to_csv(os.path.join(OUTPUT_DIR, "solve_performance.csv"), index=False)

    print("\n=== KEY NUMBERS FOR PAPER ===")
    print("\nL1 Dispatch References (mean +/- std):")
    for v in [1, 2, 3]:
        col = f"p_ref_vpp{v}"
        print(
            f"  VPP_{v}: {df[col].mean():.3f} +/- {df[col].std():.3f} MW "
            f"[{df[col].min():.3f}, {df[col].max():.3f}]"
        )

    print("\nAS Reserve (mean +/- std):")
    for v in [1, 2, 3]:
        col = f"r_as_vpp{v}"
        print(f"  VPP_{v}: {df[col].mean():.3f} +/- {df[col].std():.3f} MW")

    print("\nMarket Prices:")
    for col, unit in [
        ("lambda_p2p", "$/MWh"),
        ("lambda_as_ffr", "$/MWh"),
        ("lambda_q", "$/Mvar"),
    ]:
        print(f"  {col}: {df[col].mean():.2f} +/- {df[col].std():.2f} {unit}")

    print("\nFrequency Events:")
    freq_rate = df["freq_event_flag"].mean() * 100
    print(f"  Event rate: {freq_rate:.1f}% of steps")
    ev = df[df["freq_event_flag"] == 1]["delta_p_cont"]
    if not ev.empty:
        print(
            f"  delta_p_cont: {ev.mean():.3f} +/- {ev.std():.3f} MW (when event occurs)"
        )

    print("\nEstimated VPP Revenue ($/day average):")
    print(revenue[["vpp", "net_revenue_usd"]].to_string(index=False))

    print("\nFiles saved to artifacts/l0l1_stats/")


if __name__ == "__main__":
    main()
