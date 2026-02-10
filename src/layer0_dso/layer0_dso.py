from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import copy
import re
import warnings

import numpy as np
import pandas as pd
import pandapower as pp

from src.env.IEEE123bus import IEEE123_ZONE_BUS_MAP, build_ieee123_net, validate_ieee123_net
from src.layer0_dso.reconfiguration import ModelWeights, SolverOptions, run_reconfiguration, switch_edge_map
from src.layer0_dso.zonalpricing import generate_market_signals
from scripts.partition_zones import partition_zones
from scripts.hourly_profiles import build_hourly_profiles, build_hourly_totals, day_slice, select_representative_days


@dataclass(frozen=True)
class Layer0Result:
    lambda_dlmp: dict[int, float]
    alpha_star: dict[int, int]
    market_signals: dict[str, dict[str, float]]


@dataclass(frozen=True)
class Layer0HourlyResult:
    day_label: str
    hour: int
    lambda_dlmp: dict[int, float]
    alpha_star: dict[int, int]
    market_signals: dict[str, dict[str, float]]


def run_layer0_dso(
    net: pp.pandapowerNet | None = None,
    weights: ModelWeights | None = None,
    switch_cost: float = 0.01,
    solver_opts: SolverOptions | None = None,
    n_zones: int = 4,
) -> Layer0Result:
    if net is None:
        net = build_ieee123_net(mode="feeder123", balanced=True, convert_switches=True)

    if not net.trafo.empty and not net.ext_grid.empty:
        hv_buses = set(int(bus) for bus in net.trafo.hv_bus.values)
        filtered = net.ext_grid[~net.ext_grid.bus.isin(hv_buses)].copy()
        if not filtered.empty:
            net.ext_grid = filtered.reset_index(drop=True)

    validate_ieee123_net(net)

    lambda_dlmp, alpha_star = run_reconfiguration(
        net=net,
        weights=weights,
        switch_cost=switch_cost,
        solver_opts=solver_opts,
    )

    zone_map = partition_zones(net, n_zones=n_zones, zone_bus_map=IEEE123_ZONE_BUS_MAP)
    market_signals = generate_market_signals(net, lambda_dlmp=lambda_dlmp, zone_map=zone_map)

    return Layer0Result(
        lambda_dlmp=lambda_dlmp,
        alpha_star=alpha_star,
        market_signals=market_signals,
    )


def _apply_hourly_profiles(
    net: pp.pandapowerNet,
    load_profiles: dict[str, dict[str, np.ndarray]],
    pv_profiles: dict[str, dict[str, np.ndarray]] | None,
    wind_profiles: dict[str, dict[str, np.ndarray]] | None,
    hour_index: int,
    name_map: dict[str, str] | None = None,
) -> tuple[int, int]:
    updated_loads = 0
    updated_sgens = 0
    if not net.load.empty:
        for idx, row in net.load.iterrows():
            name = str(row.get("name", ""))
            profile_key = name_map.get(name, name) if name_map else name
            profile = load_profiles.get(profile_key)
            if profile is None:
                continue
            net.load.at[idx, "p_mw"] = float(profile["p_mw"][hour_index])
            net.load.at[idx, "q_mvar"] = float(profile["q_mvar"][hour_index])
            updated_loads += 1

    pv_output = 0.0
    wind_output = 0.0
    baseline_pv = 0.0
    baseline_wind = 0.0
    mean_wind_factor = 0.0

    if pv_profiles:
        total_profile = pv_profiles.get("total_pv_mw")
        if total_profile is not None:
            total_series = total_profile.get("p_mw")
            if total_series is not None:
                pv_output = float(total_series[hour_index])
                baseline_pv = float(np.mean(total_series))

    if wind_profiles:
        wind_profile = wind_profiles.get("profile")
        if wind_profile is not None:
            wind_series = wind_profile.get("p_mw")
            if wind_series is not None:
                mean_wind_factor = float(np.mean(wind_series))

    if not net.sgen.empty:
        for idx, row in net.sgen.iterrows():
            sgen_type = str(row.get("type", ""))
            if sgen_type == "pv":
                if not pv_profiles:
                    continue
                bus_idx = row.get("bus")
                if bus_idx is None:
                    continue
                bus_name = str(net.bus.at[bus_idx, "name"]) if bus_idx in net.bus.index else None
                if not bus_name:
                    continue
                profile = pv_profiles.get(bus_name)
                if profile is None:
                    continue
                net.sgen.at[idx, "p_mw"] = float(profile["p_mw"][hour_index])
                updated_sgens += 1
            elif sgen_type == "wind":
                if not wind_profiles:
                    continue
                profile = wind_profiles.get("profile")
                if profile is None:
                    continue
                profile_series = profile.get("p_mw")
                if profile_series is None:
                    continue
                base_p = float(row.get("p_mw", 0.0))
                factor = float(profile_series[hour_index])
                net.sgen.at[idx, "p_mw"] = base_p * factor
                wind_output += base_p * factor
                baseline_wind += base_p * mean_wind_factor

    renewable_deviation = (pv_output + wind_output) - (baseline_pv + baseline_wind)

    if not net.sgen.empty:
        for idx, row in net.sgen.iterrows():
            sgen_type = str(row.get("type", ""))
            if sgen_type not in {"bess", "storage"}:
                continue
            nameplate = abs(float(row.get("p_mw", 0.0)))
            if nameplate <= 0.0:
                net.sgen.at[idx, "p_mw"] = 0.0
                continue
            bess_setpoint = -renewable_deviation
            bess_setpoint = max(-nameplate, min(nameplate, bess_setpoint))
            net.sgen.at[idx, "p_mw"] = bess_setpoint

    thermal_min = 0.2
    if not net.sgen.empty:
        for idx, row in net.sgen.iterrows():
            sgen_type = str(row.get("type", ""))
            if sgen_type != "thermal":
                continue
            nameplate = abs(float(row.get("p_mw", 0.0)))
            if nameplate <= 0.0:
                net.sgen.at[idx, "p_mw"] = 0.0
                continue
            min_output = thermal_min * nameplate
            thermal_setpoint = max(min_output, nameplate - renewable_deviation)
            thermal_setpoint = min(nameplate, thermal_setpoint)
            net.sgen.at[idx, "p_mw"] = thermal_setpoint

    return updated_loads, updated_sgens


def _build_profile_name_map(
    net: pp.pandapowerNet,
    load_profiles: dict[str, dict[str, np.ndarray]],
) -> dict[str, str]:
    mapping: dict[str, str] = {}
    if not load_profiles:
        return mapping
    bus_to_profiles: dict[str, list[str]] = {}
    pattern = re.compile(r"^S(?P<bus>\d+)", re.IGNORECASE)
    for name in load_profiles.keys():
        match = pattern.match(name)
        if not match:
            continue
        bus = match.group("bus")
        bus_to_profiles.setdefault(bus, []).append(name)
    for bus, names in bus_to_profiles.items():
        if names:
            mapping[f"load_{bus}"] = names[0]
    agg_pattern = re.compile(r"^load_agg_(?P<bus_idx>\d+)$", re.IGNORECASE)
    for name in net.load["name"].astype(str).tolist():
        match = agg_pattern.match(name)
        if not match:
            continue
        bus_idx = int(match.group("bus_idx"))
        if bus_idx not in net.bus.index:
            continue
        bus_name = str(net.bus.at[bus_idx, "name"])
        profile_names = bus_to_profiles.get(bus_name)
        if profile_names:
            mapping[name] = profile_names[0]
    return mapping


def _warn_if_profile_mismatch(updated: int, total: int, label: str, kind: str) -> None:
    if total == 0:
        return
    ratio = updated / max(total, 1)
    if ratio < 0.5:
        warnings.warn(
            f"Hourly {kind} profiles matched {updated}/{total} entries for {label}. "
            "Check name mapping between pandapower net and DSS profiles.",
            RuntimeWarning,
        )


def _zone_price_records(day_label: str, hour: int, market_signals: dict[str, dict[str, float]]) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for zone, data in market_signals.items():
        records.append(
            {
                "day": day_label,
                "hour": hour,
                "zone": zone,
                "energy_price": float(data.get("energy_price", float("nan"))),
            }
        )
    return records


def _alpha_records(day_label: str, hour: int, alpha_star: dict[int, int]) -> list[dict[str, object]]:
    return [
        {
            "day": day_label,
            "hour": hour,
            "edge_id": int(edge_id),
            "alpha": int(value),
        }
        for edge_id, value in alpha_star.items()
    ]


def run_layer0_dso_hourly(
    day_label: str,
    day_index: int,
    base_net: pp.pandapowerNet,
    load_profiles: dict[str, dict[str, np.ndarray]],
    pv_profiles: dict[str, dict[str, np.ndarray]] | None = None,
    wind_profiles: dict[str, dict[str, np.ndarray]] | None = None,
    weights: ModelWeights | None = None,
    switch_cost: float = 0.01,
    solver_opts: SolverOptions | None = None,
    n_zones: int = 4,
) -> list[Layer0HourlyResult]:
    results: list[Layer0HourlyResult] = []
    if not base_net.trafo.empty and not base_net.ext_grid.empty:
        hv_buses = set(int(bus) for bus in base_net.trafo.hv_bus.values)
        filtered = base_net.ext_grid[~base_net.ext_grid.bus.isin(hv_buses)].copy()
        if not filtered.empty:
            base_net.ext_grid = filtered.reset_index(drop=True)

    validate_ieee123_net(base_net)
    name_map = _build_profile_name_map(base_net, load_profiles)

    slice_ = day_slice(day_index)
    for hour_offset, hour_index in enumerate(range(slice_.start, slice_.stop)):
        hour_net = copy.deepcopy(base_net)
        updated_loads, updated_sgens = _apply_hourly_profiles(
            hour_net,
            load_profiles,
            pv_profiles,
            wind_profiles,
            hour_index,
            name_map=name_map,
        )
        _warn_if_profile_mismatch(updated_loads, len(hour_net.load), f"{day_label} hour {hour_offset}", "load")
        _warn_if_profile_mismatch(updated_sgens, len(hour_net.sgen), f"{day_label} hour {hour_offset}", "pv")
        lambda_dlmp, alpha_star = run_reconfiguration(
            net=hour_net,
            weights=weights,
            switch_cost=switch_cost,
            solver_opts=solver_opts,
            debug=False,
            force_switch_closed=False,
            apply_voltage_bounds=True,
        )
        zone_map = partition_zones(hour_net, n_zones=n_zones, zone_bus_map=IEEE123_ZONE_BUS_MAP)
        market_signals = generate_market_signals(hour_net, lambda_dlmp=lambda_dlmp, zone_map=zone_map)
        results.append(
            Layer0HourlyResult(
                day_label=day_label,
                hour=hour_offset,
                lambda_dlmp=lambda_dlmp,
                alpha_star=alpha_star,
                market_signals=market_signals,
            )
        )
    return results


def export_layer0_csvs(output_dir: Path, hourly_results: list[Layer0HourlyResult]) -> tuple[Path, Path, Path]:
    zone_rows: list[dict[str, object]] = []
    alpha_rows: list[dict[str, object]] = []
    switch_rows: list[dict[str, object]] = []
    switch_map = switch_edge_map(build_ieee123_net(mode="feeder123", balanced=True, convert_switches=True))
    for result in hourly_results:
        zone_rows.extend(_zone_price_records(result.day_label, result.hour, result.market_signals))
        alpha_records = _alpha_records(result.day_label, result.hour, result.alpha_star)
        alpha_rows.extend(alpha_records)
        for record in alpha_records:
            edge_id = int(record["edge_id"])
            switch = switch_map.get(edge_id)
            if switch is None:
                continue
            switch_rows.append(
                {
                    **record,
                    "from_bus": switch["from_bus"],
                    "to_bus": switch["to_bus"],
                }
            )

    zone_path = output_dir / "layer0_zone_prices.csv"
    alpha_path = output_dir / "layer0_alpha.csv"
    switch_path = output_dir / "layer0_switches.csv"

    pd.DataFrame(zone_rows).to_csv(zone_path, index=False)
    pd.DataFrame(alpha_rows).to_csv(alpha_path, index=False)
    pd.DataFrame(switch_rows).to_csv(switch_path, index=False)

    return zone_path, alpha_path, switch_path


def main() -> None:
    output_dir = Path(__file__).resolve().parents[2] / "data" / "oedisi-ieee123-main" / "profiles" / "layer0_hourly"
    output_dir.mkdir(parents=True, exist_ok=True)

    total_p_mw, _, _ = build_hourly_totals()
    day_map = select_representative_days(total_p_mw)
    load_profiles, pv_profiles, wind_profiles = build_hourly_profiles()

    net = build_ieee123_net(mode="feeder123", balanced=True, convert_switches=True)
    validate_ieee123_net(net)

    all_results: list[Layer0HourlyResult] = []
    for label, day_index in day_map.items():
        all_results.extend(
            run_layer0_dso_hourly(
                day_label=label,
                day_index=day_index,
                base_net=net,
                load_profiles=load_profiles,
                pv_profiles=pv_profiles,
                wind_profiles=wind_profiles,
            )
        )

    zone_path, alpha_path, switch_path = export_layer0_csvs(output_dir, all_results)
    print("Exported Layer 0 hourly CSVs:")
    print(f"- Zone prices: {zone_path}")
    print(f"- Alpha decisions: {alpha_path}")
    print(f"- Switch decisions: {switch_path}")


if __name__ == "__main__":
    main()
