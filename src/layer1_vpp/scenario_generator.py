from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


SCENARIOS = ("offpeak", "median", "peak")


@dataclass(frozen=True)
class PriceScenarioSet:
    hours: np.ndarray
    scenario_names: tuple[str, ...]
    scenario_weights: np.ndarray
    energy_prices: np.ndarray
    reserve_prices: np.ndarray


def aggregate_zone_prices(df: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        df.groupby(["day", "hour"], as_index=False)[["energy_price", "reserve_price"]]
        .mean()
        .sort_values(["hour", "day"])
    )
    return grouped


def _complete_hours(df: pd.DataFrame, scenario: str, hours: np.ndarray) -> pd.DataFrame:
    scenario_df = df[df["day"] == scenario].copy()
    if scenario_df.empty:
        scenario_df = pd.DataFrame({
            "day": scenario,
            "hour": hours,
            "energy_price": np.nan,
            "reserve_price": np.nan,
        })
    else:
        scenario_df = (
            pd.DataFrame({"hour": hours})
            .merge(scenario_df, on="hour", how="left")
            .assign(day=scenario)
        )
    return scenario_df


def build_price_scenarios(
    aggregated_prices: pd.DataFrame,
    scenario_weights: dict[str, float],
) -> PriceScenarioSet:
    scenario_names = tuple(name.lower() for name in scenario_weights)
    weights = np.array([float(scenario_weights[name]) for name in scenario_names], dtype=float)
    weight_sum = float(weights.sum())
    if weight_sum <= 0:
        raise ValueError("Scenario weights must sum to a positive value.")
    weights = weights / weight_sum

    hours = np.arange(96, dtype=int)
    completed = [
        _complete_hours(aggregated_prices, scenario=name, hours=hours)
        for name in scenario_names
    ]
    full_df = pd.concat(completed, ignore_index=True)

    for col in ("energy_price", "reserve_price"):
        global_mean = float(full_df[col].mean()) if full_df[col].notna().any() else 0.0
        full_df[col] = full_df[col].fillna(global_mean)

    energy_matrix = np.zeros((len(scenario_names), len(hours)), dtype=float)
    reserve_matrix = np.zeros((len(scenario_names), len(hours)), dtype=float)

    for idx, scenario in enumerate(scenario_names):
        scenario_df = full_df[full_df["day"] == scenario].sort_values("hour")
        energy_matrix[idx, :] = scenario_df["energy_price"].to_numpy(dtype=float)
        reserve_matrix[idx, :] = scenario_df["reserve_price"].to_numpy(dtype=float)

    return PriceScenarioSet(
        hours=hours,
        scenario_names=scenario_names,
        scenario_weights=weights,
        energy_prices=energy_matrix,
        reserve_prices=reserve_matrix,
    )
