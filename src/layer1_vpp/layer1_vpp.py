from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import argparse
import sys

import pandas as pd

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.layer1_vpp.dro_bidding import DroConfig, solve_wasserstein_dro
from src.layer1_vpp.scenario_generator import SCENARIOS, aggregate_zone_prices, build_price_scenarios
from src.layer1_vpp.virtual_battery import VirtualBatteryConfig, reserve_limit_from_state


PROFILES_PER_DAY = 96
DEFAULT_WEIGHTS = {"offpeak": 0.5, "median": 0.3, "peak": 0.2}


@dataclass(frozen=True)
class Layer1Config:
    input_csv: Path
    output_csv: Path
    weights: dict[str, float]
    sign: str
    wasserstein_radius: float
    degradation_cost: float
    curtailment_ratio: float | None = None
    feedback_threshold: float = 0.05
    vpp_mode: bool = False
    mapping_bus_to_vpp_csv: Path | None = None
    mapping_vpp_to_zone_csv: Path | None = None
    legacy_output_csv: Path | None = None



def _default_input() -> Path:
    return Path(__file__).resolve().parents[2] / "data" / "oedisi-ieee123-main" / "profiles" / "layer0_hourly" / "layer0_zone_prices.csv"



def _default_output() -> Path:
    return Path(__file__).resolve().parents[2] / "data" / "oedisi-ieee123-main" / "profiles" / "layer1_vpp" / "layer1_pref.csv"



def _parse_weights(text: str) -> dict[str, float]:
    raw = {}
    for item in text.split(","):
        key, value = item.split("=")
        key = key.strip().lower()
        if key not in DEFAULT_WEIGHTS:
            raise argparse.ArgumentTypeError(f"Unknown scenario '{key}'.")
        raw[key] = float(value)
    if set(raw) != set(DEFAULT_WEIGHTS):
        missing = set(DEFAULT_WEIGHTS) - set(raw)
        raise argparse.ArgumentTypeError(f"Weights missing scenarios: {sorted(missing)}")
    return raw



def _validate_prices(df: pd.DataFrame) -> pd.DataFrame:
    required = {"day", "hour", "energy_price"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Input CSV missing columns: {sorted(missing)}")

    if "zone_id" not in df.columns and "zone" not in df.columns:
        raise ValueError("Input CSV missing zone identifier columns ('zone_id' or 'zone').")

    df = df.copy()
    if "zone_id" not in df.columns:
        df["zone_id"] = df["zone"].astype(str)
    else:
        df["zone_id"] = df["zone_id"].astype(str)

    df["day"] = df["day"].astype(str).str.lower()
    df = df[df["day"].isin(SCENARIOS)]
    df = df[df["hour"].between(0, PROFILES_PER_DAY - 1)]
    if "reserve_price" not in df.columns:
        df["reserve_price"] = 0.0
    if df.empty:
        raise ValueError("Input CSV has no rows for offpeak/median/peak and hour 0-95.")
    return df



def _feedback_flags(curtailment_ratio: float | None, threshold: float) -> tuple[float, bool]:
    if curtailment_ratio is None:
        return float("nan"), False
    ratio = float(curtailment_ratio)
    return ratio, bool(ratio > threshold)



def _read_vpp_zone_map(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"vpp_id", "zone_id"}
    if not required.issubset(df.columns):
        raise ValueError(f"{path} must contain columns {sorted(required)}")
    out = df[list(required)].copy()
    out["vpp_id"] = out["vpp_id"].astype(str)
    out["zone_id"] = out["zone_id"].astype(str)
    return out.drop_duplicates().sort_values(["zone_id", "vpp_id"]).reset_index(drop=True)



def _read_bus_vpp_counts(path: Path) -> dict[str, int]:
    df = pd.read_csv(path)
    required = {"bus", "vpp_id"}
    if not required.issubset(df.columns):
        raise ValueError(f"{path} must contain columns {sorted(required)}")

    counted = df.dropna(subset=["vpp_id"]).copy()
    counted["vpp_id"] = counted["vpp_id"].astype(str)
    return counted.groupby("vpp_id")["bus"].nunique().astype(int).to_dict()



def _solve_schedule_for_prices(
    prices_agg: pd.DataFrame,
    config: Layer1Config,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    scenarios = build_price_scenarios(prices_agg, config.weights)

    battery_cfg = VirtualBatteryConfig()
    dro_cfg = DroConfig(
        wasserstein_radius=config.wasserstein_radius,
        degradation_cost=config.degradation_cost,
    )

    dro_result = solve_wasserstein_dro(scenarios, battery_cfg, dro_cfg)

    sign_value = 1.0 if config.sign == "inject" else -1.0
    p_ref_signed = dro_result.schedule.p_ref * sign_value
    reserve_limit = reserve_limit_from_state(p_ref_signed, dro_result.schedule.soc, battery_cfg)
    r_commit = pd.Series(dro_result.schedule.r_commit).clip(lower=0.0)
    r_commit = r_commit.combine(pd.Series(reserve_limit), min).to_numpy(dtype=float)

    curtail_ratio, reopt_flag = _feedback_flags(config.curtailment_ratio, config.feedback_threshold)

    base = pd.DataFrame(
        {
            "hour": scenarios.hours,
            "P_ref": p_ref_signed,
            "R_commit": r_commit,
            "price_energy_expected": dro_result.expected_energy_price,
            "price_reserve_expected": dro_result.expected_reserve_price,
            "price_energy_robust": dro_result.robust_energy_margin,
            "price_reserve_robust": dro_result.robust_reserve_margin,
            "SoC": dro_result.schedule.soc[:-1],
            "solver_status": dro_result.solver_status,
            "curtailment_ratio": curtail_ratio,
            "reoptimize_next_cycle": reopt_flag,
        }
    )
    return base, pd.DataFrame({"hour": scenarios.hours})



def run_layer1(config: Layer1Config) -> Path:
    prices_raw = pd.read_csv(config.input_csv)
    prices_valid = _validate_prices(prices_raw)

    output_path = config.output_csv
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not config.vpp_mode or config.mapping_vpp_to_zone_csv is None:
        prices_agg = aggregate_zone_prices(prices_valid)
        output, _ = _solve_schedule_for_prices(prices_agg, config)
        output.to_csv(output_path, index=False)
        return output_path

    vpp_zone_df = _read_vpp_zone_map(config.mapping_vpp_to_zone_csv)
    bus_count_by_vpp: dict[str, int] = {}
    if config.mapping_bus_to_vpp_csv is not None:
        bus_count_by_vpp = _read_bus_vpp_counts(config.mapping_bus_to_vpp_csv)
    zone_prices = aggregate_zone_prices(prices_valid, preserve_zone=True)

    long_rows: list[pd.DataFrame] = []
    legacy_parts: list[pd.DataFrame] = []

    for row in vpp_zone_df.itertuples(index=False):
        zone_id = str(row.zone_id)
        vpp_id = str(row.vpp_id)
        zone_slice = zone_prices[zone_prices["zone_id"] == zone_id]
        if zone_slice.empty:
            continue

        prices_agg = zone_slice[["day", "hour", "energy_price", "reserve_price"]].copy()
        solved, _ = _solve_schedule_for_prices(prices_agg, config)
        solved["zone_id"] = zone_id
        solved["vpp_id"] = vpp_id
        solved["vpp_bus_count"] = int(bus_count_by_vpp.get(vpp_id, 0))

        scenario_energy = (
            prices_agg.groupby("hour", as_index=False)["energy_price"]
            .mean()
            .sort_values("hour")
            .rename(columns={"energy_price": "zone_energy_price_mean"})
        )
        scenario_reserve = (
            prices_agg.groupby("hour", as_index=False)["reserve_price"]
            .mean()
            .sort_values("hour")
            .rename(columns={"reserve_price": "zone_reserve_price_mean"})
        )
        solved = solved.merge(scenario_energy, on="hour", how="left").merge(scenario_reserve, on="hour", how="left")
        solved["day"] = "expected"

        long_rows.append(
            solved[
                [
                    "day",
                    "hour",
                    "zone_id",
                    "vpp_id",
                    "vpp_bus_count",
                    "P_ref",
                    "R_commit",
                    "price_energy_expected",
                    "price_reserve_expected",
                    "price_energy_robust",
                    "price_reserve_robust",
                    "zone_energy_price_mean",
                    "zone_reserve_price_mean",
                    "SoC",
                    "solver_status",
                    "curtailment_ratio",
                    "reoptimize_next_cycle",
                ]
            ]
        )

        legacy_parts.append(solved[["hour", "P_ref", "R_commit"]].copy())

    if not long_rows:
        raise ValueError("No per-VPP schedules were produced. Check vpp_to_zone mapping and Layer0 zones.")

    long_out = pd.concat(long_rows, ignore_index=True).sort_values(["zone_id", "vpp_id", "hour"])
    long_out.to_csv(output_path, index=False)

    legacy_out_path = config.legacy_output_csv or output_path.with_name(f"{output_path.stem}_legacy.csv")
    legacy_out_path.parent.mkdir(parents=True, exist_ok=True)
    legacy_agg = pd.concat(legacy_parts, ignore_index=True).groupby("hour", as_index=False)[["P_ref", "R_commit"]].sum()
    legacy_agg.to_csv(legacy_out_path, index=False)

    return output_path



def _parse_args() -> Layer1Config:
    parser = argparse.ArgumentParser(description="Generate Layer 1 VPP schedule with Wasserstein DRO.")
    parser.add_argument("--input", type=Path, default=_default_input(), help="Path to layer0_zone_prices.csv")
    parser.add_argument("--output", type=Path, default=_default_output(), help="Output CSV path for layer1 output")
    parser.add_argument(
        "--weights",
        type=_parse_weights,
        default=DEFAULT_WEIGHTS,
        help="Scenario weights as offpeak=0.5,median=0.3,peak=0.2",
    )
    parser.add_argument(
        "--sign",
        choices=["inject", "consume"],
        default="inject",
        help="Sign convention for P_ref (inject=+1, consume=-1)",
    )
    parser.add_argument(
        "--wasserstein-radius",
        type=float,
        default=0.02,
        help="Wasserstein radius epsilon for DRO approximation.",
    )
    parser.add_argument(
        "--degradation-cost",
        type=float,
        default=1.0,
        help="Linear battery degradation penalty coefficient.",
    )
    parser.add_argument(
        "--curtailment-ratio",
        type=float,
        default=None,
        help="Optional Layer2 curtailment ratio feedback (0..1).",
    )
    parser.add_argument(
        "--feedback-threshold",
        type=float,
        default=0.05,
        help="Trigger re-optimization when curtailment ratio exceeds this threshold.",
    )
    args = parser.parse_args()

    return Layer1Config(
        input_csv=args.input,
        output_csv=args.output,
        weights=args.weights,
        sign=args.sign,
        wasserstein_radius=args.wasserstein_radius,
        degradation_cost=args.degradation_cost,
        curtailment_ratio=args.curtailment_ratio,
        feedback_threshold=args.feedback_threshold,
    )



def main() -> None:
    config = _parse_args()
    output = run_layer1(config)
    print(f"Layer 1 output saved to {output}")


if __name__ == "__main__":
    main()
