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
    required = {"day", "hour", "zone", "energy_price"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Input CSV missing columns: {sorted(missing)}")

    df = df.copy()
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


def run_layer1(config: Layer1Config) -> Path:
    prices_raw = pd.read_csv(config.input_csv)
    prices_valid = _validate_prices(prices_raw)
    prices_agg = aggregate_zone_prices(prices_valid)
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

    output = pd.DataFrame(
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

    output_path = config.output_csv
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False)
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
