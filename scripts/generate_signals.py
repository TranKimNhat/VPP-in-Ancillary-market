"""Generate signal CSVs for GridEnvironment from the full L0 SOCP + L1 DRO + Q-OPF pipeline.

Runs all 365 profile days through L0 SOCP reconfiguration to obtain real per-bus DLMPs
and zone prices, then runs L1 Wasserstein DRO to produce P_ref, R_commit, Q_ref.

Signal CSVs are averaged across all 365 days per step (0-95). Both training arms
(baseline and method) share the same output files via env_config.yaml.

Pre-registered zone→scenario mapping (not changed after viewing output):
  zone 1 → "offpeak", zone 2 → "median", zone 4 → "peak"

Outputs:
  data/oedisi-ieee123-main/profiles/layer1_vpp/layer1_pref.csv
  data/oedisi-ieee123-main/profiles/layer0_hourly/layer0_zone_prices.csv
  data/oedisi-ieee123-main/profiles/layer0_hourly/layer0_dlmp_per_bus.csv

Checkpoint: per-day L0 results cached in data/signal_cache/ for resume support.

Usage:
  python scripts/generate_signals.py [--n-days N] [--pricing-method METHOD] [--no-q-opf]
"""
from __future__ import annotations

import argparse
import logging
import pathlib
import pickle
import sys
import time

import numpy as np
import pandas as pd

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.env.IEEE123bus import build_ieee123_net
from src.layer0_dso.layer0_dso import (
    Layer0HourlyResult,
    build_hourly_profiles,
    run_layer0_dso_hourly,
)
from src.layer0_dso.reconfiguration import switch_edge_map
from src.layer1_vpp.dro_bidding import DroConfig, solve_wasserstein_dro
from src.layer1_vpp.scenario_generator import build_price_scenarios
from src.layer1_vpp.virtual_battery import VirtualBatteryConfig

# ── output paths (must match env_config.yaml signals block) ─────────────────
OUT_L1 = REPO_ROOT / "data" / "oedisi-ieee123-main" / "profiles" / "layer1_vpp"
OUT_L0 = REPO_ROOT / "data" / "oedisi-ieee123-main" / "profiles" / "layer0_hourly"
CACHE_DIR = REPO_ROOT / "data" / "signal_cache"

STEPS_PER_DAY = 96
TOTAL_DAYS = 365

# Pre-registered; do not change after first run.
ZONE_TO_SCENARIO: dict[int, str] = {1: "offpeak", 2: "median", 4: "peak"}

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


# ── helper types ─────────────────────────────────────────────────────────────

def _new_step_lists() -> list[list[float]]:
    return [[] for _ in range(STEPS_PER_DAY)]


# ── per-day L0 runner with checkpoint support ────────────────────────────────

def _run_day(
    day_index: int,
    base_net: object,
    load_profiles: dict,
    pv_profiles: dict | None,
    wind_profiles: dict | None,
    pricing_method: str,
) -> list[Layer0HourlyResult] | None:
    cache_path = CACHE_DIR / f"day_{day_index:03d}.pkl"
    if cache_path.exists():
        try:
            with open(cache_path, "rb") as fh:
                return pickle.load(fh)  # type: ignore[return-value]
        except Exception:
            cache_path.unlink(missing_ok=True)

    common_kwargs: dict = dict(
        day_label=f"day_{day_index:03d}",
        day_index=day_index,
        base_net=base_net,  # type: ignore[arg-type]
        load_profiles=load_profiles,
        pv_profiles=pv_profiles,
        wind_profiles=wind_profiles,
        pricing_method=pricing_method,
    )
    # Try full reconfiguration first; fall back to fixed topology on MISOCP infeasibility.
    for attempt, force_closed in enumerate([False, True]):
        try:
            results = run_layer0_dso_hourly(**common_kwargs, force_switch_closed=force_closed)
            with open(cache_path, "wb") as fh:
                pickle.dump(results, fh)
            if attempt > 0:
                log.info("Day %03d: fixed-topology fallback succeeded", day_index)
            return results
        except Exception as exc:
            if attempt == 0:
                log.info("Day %03d: reconfiguration failed (%s) — retrying with fixed topology", day_index, exc)
            else:
                log.warning("Day %03d: both attempts failed — skipped: %s", day_index, exc)
    return None


# ── main accumulation loop ───────────────────────────────────────────────────

def collect_signals(
    n_days: int,
    pricing_method: str,
    base_net: object,
    load_profiles: dict,
    pv_profiles: dict | None,
    wind_profiles: dict | None,
) -> tuple[dict, dict, list]:
    """Run L0 SOCP for *n_days* days and accumulate raw per-step signal lists.

    Returns:
      dlmp_acc  : {bus_id: {field: list-of-lists[step][day_values]}}
      zone_acc  : {zone_id: {"energy": list[step][day_values],
                             "reserve": list[step][day_values]}}
      switch_records: list of {day, hour, edge_id, alpha} dicts
    """
    dlmp_acc: dict[int, dict[str, list[list[float]]]] = {}
    zone_acc: dict[int, dict[str, list[list[float]]]] = {
        z: {"energy": _new_step_lists(), "reserve": _new_step_lists()}
        for z in ZONE_TO_SCENARIO
    }
    switch_records: list[dict] = []

    t0 = time.time()
    ok = 0
    for day_index in range(n_days):
        results = _run_day(
            day_index, base_net, load_profiles, pv_profiles, wind_profiles, pricing_method
        )
        if not results:
            continue
        ok += 1
        for hourly in results:
            step = int(hourly.hour)
            # per-bus DLMP
            if hourly.dlmp_components:
                for bus_id, dlmp in hourly.dlmp_components.items():
                    if bus_id not in dlmp_acc:
                        dlmp_acc[bus_id] = {
                            "lambda_p_total": _new_step_lists(),
                            "lambda_q_total": _new_step_lists(),
                            "lambda_p_energy": _new_step_lists(),
                            "lambda_p_loss": _new_step_lists(),
                            "lambda_p_congestion": _new_step_lists(),
                            "lambda_p_voltage": _new_step_lists(),
                        }
                    dlmp_acc[bus_id]["lambda_p_total"][step].append(dlmp.lambda_p_total)
                    dlmp_acc[bus_id]["lambda_q_total"][step].append(dlmp.lambda_q_total)
                    dlmp_acc[bus_id]["lambda_p_energy"][step].append(dlmp.lambda_p_energy)
                    dlmp_acc[bus_id]["lambda_p_loss"][step].append(dlmp.lambda_p_loss)
                    dlmp_acc[bus_id]["lambda_p_congestion"][step].append(dlmp.lambda_p_congestion)
                    dlmp_acc[bus_id]["lambda_p_voltage"][step].append(dlmp.lambda_p_voltage)
            # alpha_star (tie-switch decisions)
            if hourly.alpha_star:
                for edge_id, alpha_val in hourly.alpha_star.items():
                    switch_records.append({
                        "day": day_index,
                        "hour": step,
                        "edge_id": int(edge_id),
                        "alpha": int(alpha_val),
                    })
            # per-zone prices (market_signals keys are zone id strings)
            for zone_str, sig in hourly.market_signals.items():
                try:
                    zone_id = int(zone_str)
                except ValueError:
                    continue
                if zone_id not in zone_acc:
                    continue
                e = float(sig.get("energy_price", 0.0))
                r = float(sig.get("reserve_price", 0.0))
                zone_acc[zone_id]["energy"][step].append(e)
                zone_acc[zone_id]["reserve"][step].append(r)

        # per-day summary log (use last step's result as quick sanity values)
        last = results[-1]
        elapsed = time.time() - t0
        eta = elapsed / (day_index + 1) * (n_days - day_index - 1)
        n_buses_today = len(last.dlmp_components) if last.dlmp_components else 0
        lp_vals = [d.lambda_p_total for d in last.dlmp_components.values()] if last.dlmp_components else []
        lp_mean = float(np.mean(lp_vals)) if lp_vals else float("nan")
        last_step = int(last.hour)
        zone_e_str = "  ".join(
            f"z{z}={np.mean(zone_acc[z]['energy'][last_step]) if zone_acc[z]['energy'][last_step] else float('nan'):.3f}"
            for z in ZONE_TO_SCENARIO
        )
        log.info(
            "Day %3d/%d | ok=%d | buses=%d | λp_mean=%.3f €/MWh | %s | ETA %.0fs",
            day_index + 1, n_days, ok, n_buses_today, lp_mean, zone_e_str, eta,
        )

    log.info("Collection done: %d/%d days contributed data", ok, n_days)
    return dlmp_acc, zone_acc, switch_records


# ── safe mean helper ─────────────────────────────────────────────────────────

def _safe_mean(vals: list[float]) -> float:
    arr = [v for v in vals if np.isfinite(v)]
    return float(np.mean(arr)) if arr else 0.0


# ── export helpers ───────────────────────────────────────────────────────────

def export_dlmp_csv(dlmp_acc: dict[int, dict[str, list[list[float]]]]) -> None:
    OUT_L0.mkdir(parents=True, exist_ok=True)
    rows = []
    for bus_id in sorted(dlmp_acc):
        bd = dlmp_acc[bus_id]
        for step in range(STEPS_PER_DAY):
            lp = _safe_mean(bd["lambda_p_total"][step])
            lq = _safe_mean(bd["lambda_q_total"][step])
            rows.append({
                "day": "expected",
                "hour": step,
                "bus_id": bus_id,
                "lambda_p_total": lp,
                "lambda_q_total": lq,
                "lambda_p_energy": _safe_mean(bd["lambda_p_energy"][step]),
                "lambda_p_loss": _safe_mean(bd["lambda_p_loss"][step]),
                "lambda_p_congestion": _safe_mean(bd["lambda_p_congestion"][step]),
                "lambda_p_voltage": _safe_mean(bd["lambda_p_voltage"][step]),
            })
    out = OUT_L0 / "layer0_dlmp_per_bus.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    n_buses = len(dlmp_acc)
    log.info("  -> %s  (%d rows, %d buses × %d steps)", out, len(rows), n_buses, STEPS_PER_DAY)


def export_zone_prices_csv(
    zone_acc: dict[int, dict[str, list[list[float]]]],
) -> pd.DataFrame:
    OUT_L0.mkdir(parents=True, exist_ok=True)
    rows = []
    for zone_id, scenario in ZONE_TO_SCENARIO.items():
        za = zone_acc[zone_id]
        for step in range(STEPS_PER_DAY):
            e = _safe_mean(za["energy"][step])
            r = _safe_mean(za["reserve"][step])
            rows.append({
                "day": scenario,
                "hour": step,
                "zone": scenario,
                "zone_id": scenario,
                "energy_price": e,
                "reserve_price": r,
            })
    df = pd.DataFrame(rows)
    out = OUT_L0 / "layer0_zone_prices.csv"
    df.to_csv(out, index=False)
    log.info("  -> %s  (%d rows)", out, len(rows))
    return df


def export_switches_csv(switch_records: list[dict], base_net: object) -> None:
    OUT_L0.mkdir(parents=True, exist_ok=True)

    # Export alpha (all edges with alpha decisions)
    alpha_path = OUT_L0 / "layer0_alpha.csv"
    alpha_df = pd.DataFrame(switch_records)
    alpha_df.to_csv(alpha_path, index=False)
    log.info("  -> %s  (%d rows)", alpha_path, len(switch_records))

    # Export switches (only tie-switch edges with bus info)
    sw_map = switch_edge_map(base_net)
    switch_rows = []
    for rec in switch_records:
        edge_id = int(rec["edge_id"])
        if edge_id in sw_map:
            switch_rows.append({
                **rec,
                "from_bus": sw_map[edge_id]["from_bus"],
                "to_bus": sw_map[edge_id]["to_bus"],
            })
    switch_path = OUT_L0 / "layer0_switches.csv"
    pd.DataFrame(switch_rows).to_csv(switch_path, index=False)
    log.info("  -> %s  (%d rows)", switch_path, len(switch_rows))


def export_layer1_pref_csv(zone_prices_df: pd.DataFrame, q_opf_enabled: bool) -> None:
    OUT_L1.mkdir(parents=True, exist_ok=True)

    cols = ["day", "hour", "energy_price", "reserve_price"]
    agg = pd.DataFrame({c: zone_prices_df[c].to_numpy() for c in cols})
    scenarios = build_price_scenarios(
        aggregated_prices=agg,
        scenario_weights={"offpeak": 1.0, "median": 1.0, "peak": 1.0},
    )

    # System-level virtual battery (aggregate of all VPP storage in IEEE 123)
    battery = VirtualBatteryConfig(
        p_discharge_max=2.0,
        p_charge_max=2.0,
        energy_capacity_mwh=8.0,
        inverter_s_mva=3.0,
    )
    dro_cfg = DroConfig(
        wasserstein_radius=0.02,
        q_opf_enabled=q_opf_enabled,
        q_opf_mode="aggregate",
    )
    result = solve_wasserstein_dro(scenarios, battery, dro_cfg)

    per_step = pd.DataFrame({
        "hour": np.arange(STEPS_PER_DAY, dtype=int),
        "P_ref": result.schedule.p_ref,
        "R_commit": result.schedule.r_commit,
        "Q_ref": result.q_ref,
        "lambda_q_expected": result.lambda_q_expected,
        "price_energy_expected": result.expected_energy_price,
        "price_reserve_expected": result.expected_reserve_price,
    })
    per_step["price_energy_robust"] = per_step["price_energy_expected"] * 0.98
    per_step["price_reserve_robust"] = per_step["price_reserve_expected"] * 0.98
    per_step["SoC"] = result.schedule.soc[:STEPS_PER_DAY]
    per_step["solver_status"] = result.solver_status
    per_step["curtailment_ratio"] = float("nan")
    per_step["reoptimize_next_cycle"] = False

    out = OUT_L1 / "layer1_pref.csv"
    per_step.to_csv(out, index=False)
    log.info("  -> %s  (%d rows)", out, len(per_step))
    log.info(
        "     P_ref mean=%.3f MW, R_commit mean=%.3f MW, Q_ref mean=%.3f MVAR",
        float(per_step["P_ref"].mean()),
        float(per_step["R_commit"].mean()),
        float(per_step["Q_ref"].mean()),
    )


# ── entry point ──────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate signal CSVs from full L0 SOCP + L1 DRO pipeline"
    )
    parser.add_argument(
        "--n-days", type=int, default=TOTAL_DAYS,
        help=f"Days to process (default: {TOTAL_DAYS})",
    )
    parser.add_argument(
        "--pricing-method", default="load_weighted",
        choices=sorted(["load_weighted", "max_dlmp", "congestion_weighted"]),
    )
    parser.add_argument(
        "--no-q-opf", action="store_true",
        help="Disable Q-OPF for faster run (Q_ref will be 0)",
    )
    args = parser.parse_args()

    n_days = min(max(1, args.n_days), TOTAL_DAYS)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    log.info("Source: data/profiles/  |  days=%d  |  pricing=%s  |  Q-OPF=%s",
             n_days, args.pricing_method, not args.no_q_opf)

    log.info("Loading profiles ...")
    load_profiles, pv_profiles, wind_profiles = build_hourly_profiles()

    log.info("Building IEEE 123 base net ...")
    base_net = build_ieee123_net(
        mode="matpower", balanced=True, convert_switches=True,
        slack_zones=None, source_mode="publish",
    )

    log.info("Running L0 SOCP for %d days ...", n_days)
    dlmp_acc, zone_acc, switch_records = collect_signals(
        n_days, args.pricing_method, base_net,
        load_profiles, pv_profiles, wind_profiles,
    )

    log.info("Exporting layer0_dlmp_per_bus.csv ...")
    export_dlmp_csv(dlmp_acc)

    log.info("Exporting layer0_zone_prices.csv ...")
    zone_prices_df = export_zone_prices_csv(zone_acc)

    log.info("Exporting layer0_switches.csv ...")
    export_switches_csv(switch_records, base_net)

    log.info("Solving L1 DRO%s and exporting layer1_pref.csv ...",
             " + Q-OPF" if not args.no_q_opf else "")
    export_layer1_pref_csv(zone_prices_df, q_opf_enabled=not args.no_q_opf)

    log.info("Done.")


if __name__ == "__main__":
    main()
