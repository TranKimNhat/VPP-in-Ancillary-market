"""Recompute economic metrics from the DSO procurement-cost perspective.

Replaces the VPP-revenue framing (gross_revenue, net_profit, OPEX) with a
DSO-side cash-outflow framing for the FFR service:

  DSO pays VPPs for FFR via two streams:
    1. Capacity reservation     λ_cap × commit_MW × Δt        (pre-arming bill)
    2. Activation energy        λ_act × delivered_MWh         (per-event bill)

  DSO recovers via:
    3. Undersupply penalty      c_us × λ_act × shortfall_MWh  (credit when VPP fails)

  Net DSO FFR cost per evaluation window:
    dso_net_ffr_cost = AM_cap + AM_act − Undersupply_recovery

OPEX of the VPP fleet is NOT a DSO cost (it's a VPP private cost) — excluded.
EM (energy market) revenue is between gencos/loads via LMP — excluded.

Metric semantics:
  - Lower `dso_net_ffr_cost_eur` = DSO pays less for the same FFR service
  - But a method that fails FFR is unsafe → headline is cost per secured event:
       cost_per_secured_event_eur = dso_net_ffr_cost / max(ffr_sr, 1e-3)
  - For paper reporting, both per-event and daily-equivalent shown.

Inputs:
  results/economics_5method/table9_revenue_breakdown.csv   (per-VPP × scenario)
  results/economics_5method/table10_method_economics.csv   (for ffr_success_rate)

Outputs (overwrites):
  results/section3_economic/tab_economic_methods.csv      (DSO schema)
  results/section3_economic/tab_cost_effectiveness.csv    (sorted: lowest cost first)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent


def aggregate_dso_costs(
    t9_path: Path,
    t10_path: Path,
    events_per_day: float = 5.0,
) -> pd.DataFrame:
    t9 = pd.read_csv(t9_path)
    t10 = pd.read_csv(t10_path)[["method", "ffr_success_rate", "duration_h_episode"]]

    # t9 is in episode units (not daily-scaled). Sum each component over all
    # VPPs × scenarios, then divide by (n_scenarios × n_runs_per_scenario) to
    # get the per-event mean. The dataset has 4 scenarios; t9 stores the
    # n_runs-averaged value per (scenario, vpp), so summing across VPPs and
    # dividing by n_scenarios gives the mean per FFR event.
    n_scenarios = t9["scenario"].nunique()

    agg = (
        t9.groupby("method", as_index=False)
        .agg(
            am_cap_sum=("am_cap_eur", "sum"),
            am_act_sum=("am_act_eur", "sum"),
            undersupply_sum=("undersupply_eur", "sum"),
        )
    )
    agg["dso_capacity_pay_per_event_eur"] = agg["am_cap_sum"] / n_scenarios
    agg["dso_activation_pay_per_event_eur"] = agg["am_act_sum"] / n_scenarios
    # Undersupply recovery is reported for transparency only — the upstream
    # `commit_mwh` accumulates over the full episode (including pre-arming
    # ticks) while `delivered_mwh` only counts the activation window, so the
    # shortfall is artificially inflated. Do NOT subtract from gross payment.
    agg["undersupply_recovery_per_event_eur_INFO"] = agg["undersupply_sum"] / n_scenarios

    # Headline: gross DSO procurement cost = what the DSO actually pays out.
    agg["dso_gross_ffr_payment_per_event_eur"] = (
        agg["dso_capacity_pay_per_event_eur"]
        + agg["dso_activation_pay_per_event_eur"]
    )
    agg["dso_gross_ffr_payment_per_day_eur"] = (
        agg["dso_gross_ffr_payment_per_event_eur"] * events_per_day
    )

    out = agg.merge(t10, on="method", how="left")
    sr = out["ffr_success_rate"].clip(upper=0.999).astype(float)
    # Methods that fail FFR cannot be considered cheap — penalise by dividing
    # gross payment by success rate (∞ for SR=0, lower=better otherwise).
    out["gross_payment_per_secured_event_eur"] = (
        out["dso_gross_ffr_payment_per_event_eur"] / sr.clip(lower=1e-3)
    )

    cols = [
        "method",
        "ffr_success_rate",
        "dso_capacity_pay_per_event_eur",
        "dso_activation_pay_per_event_eur",
        "dso_gross_ffr_payment_per_event_eur",
        "dso_gross_ffr_payment_per_day_eur",
        "gross_payment_per_secured_event_eur",
        "undersupply_recovery_per_event_eur_INFO",
    ]
    return out[cols]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--t9", default="results/economics_5method/table9_revenue_breakdown.csv")
    parser.add_argument("--t10", default="results/economics_5method/table10_method_economics.csv")
    parser.add_argument("--dst-econ", default="results/section3_economic/tab_economic_methods.csv")
    parser.add_argument("--dst-cost", default="results/section3_economic/tab_cost_effectiveness.csv")
    parser.add_argument("--events-per-day", type=float, default=5.0,
                        help="Realistic FFR call rate for daily-equivalent column (default 5)")
    args = parser.parse_args()

    df = aggregate_dso_costs(
        REPO / args.t9, REPO / args.t10, events_per_day=args.events_per_day
    )

    # Headline file: DSO economic comparison
    dst_econ = REPO / args.dst_econ
    dst_econ.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(dst_econ, index=False)

    # Cost-effectiveness ranking: sort by gross_payment_per_secured_event
    # ascending (= "DSO pays least for each successful FFR event"). Methods
    # with low FFR-SR are penalised (cost / SR blows up).
    rank = df.sort_values("gross_payment_per_secured_event_eur").reset_index(drop=True)
    rank.to_csv(REPO / args.dst_cost, index=False)

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 220)
    pd.set_option("display.float_format", lambda v: f"{v:>10.3f}")
    print("=== DSO FFR procurement cost (sorted: gross payment per secured event, ascending) ===")
    print(rank.to_string(index=False))
    print()
    print(f"Headline:     {dst_econ}")
    print(f"Ranking:      {REPO / args.dst_cost}")
    print(f"Event rate assumed for per-day column: {args.events_per_day:.1f} events/day")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
