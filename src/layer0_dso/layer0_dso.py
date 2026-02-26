from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import argparse
import copy
import hashlib
import logging
import re
import sys
import warnings

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
import pandapower as pp

from src.env.IEEE123bus import IEEE123_ZONE_BUS_MAP, build_ieee123_net, validate_ieee123_net
from src.layer0_dso.reconfiguration import (
    ModelWeights,
    SolverOptions,
    run_reconfiguration_detailed,
    switch_edge_map,
)
from src.layer0_dso.socp_validator import validate_socp_against_ac
from src.layer0_dso.zonal_pricing import PRICING_METHODS, generate_market_signals


PROFILES_PER_DAY = 96
PROFILES_PER_YEAR = 365 * PROFILES_PER_DAY


def _profile_base_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "data" / "profiles"


def _wind_profile_dir() -> Path:
    return _profile_base_dir() / "wind_profiles"


def day_slice(day_index: int) -> slice:
    start = int(day_index) * PROFILES_PER_DAY
    stop = start + PROFILES_PER_DAY
    return slice(start, stop)


def _load_profile_series(path: Path) -> np.ndarray:
    series = pd.read_csv(path, header=None).iloc[:, 0].to_numpy(dtype=float)
    return series


def _wind_seed(bus_name: str) -> int:
    digest = hashlib.sha256(bus_name.encode("utf-8")).hexdigest()[:8]
    return int(digest, 16)


def _generate_wind_profile(bus_name: str, length: int = PROFILES_PER_YEAR) -> np.ndarray:
    rng = np.random.default_rng(_wind_seed(bus_name))
    t = np.arange(length, dtype=float)
    daily = 2 * np.pi * (t % PROFILES_PER_DAY) / PROFILES_PER_DAY
    yearly = 2 * np.pi * t / length
    base = 0.55 + 0.25 * np.sin(daily) + 0.15 * np.sin(yearly)
    noise = rng.normal(loc=0.0, scale=0.08, size=length)
    series = np.clip(base + noise, 0.0, 1.0)
    return series


def _ensure_wind_profiles(bus_names: list[str]) -> dict[str, dict[str, np.ndarray]]:
    wind_dir = _wind_profile_dir()
    wind_dir.mkdir(parents=True, exist_ok=True)
    profiles: dict[str, dict[str, np.ndarray]] = {}
    for bus_name in bus_names:
        path = wind_dir / f"wind_profile_{bus_name}.csv"
        if path.exists():
            series = _load_profile_series(path)
        else:
            series = _generate_wind_profile(bus_name)
            pd.DataFrame(series).to_csv(path, index=False, header=False)
        profiles[bus_name] = {"p_mw": series}
    return profiles


def _derive_load_name(file_name: str) -> str:
    name = Path(file_name).stem
    prefix = "loadshape_"
    if name.lower().startswith(prefix):
        return name[len(prefix) :]
    return name


def _derive_pv_name(file_name: str) -> str:
    name = Path(file_name).stem
    prefix = "pvshape_"
    if name.lower().startswith(prefix):
        return name[len(prefix) :]
    return name


def build_hourly_profiles(
    *,
    load_dir: Path | None = None,
    pv_dir: Path | None = None,
    wind_dir: Path | None = None,
) -> tuple[
    dict[str, dict[str, np.ndarray]],
    dict[str, dict[str, np.ndarray]] | None,
    dict[str, dict[str, np.ndarray]] | None,
]:
    base_dir = _profile_base_dir()
    load_dir = load_dir or (base_dir / "load_profiles")
    pv_dir = pv_dir or (base_dir / "pv_profiles")
    wind_dir = wind_dir or _wind_profile_dir()

    load_profiles: dict[str, dict[str, np.ndarray]] = {}
    if load_dir.exists():
        for path in sorted(load_dir.glob("*.csv")):
            series = _load_profile_series(path)
            load_profiles[_derive_load_name(path.name)] = {"p_mw": series, "q_mvar": series * 0.2}

    pv_profiles: dict[str, dict[str, np.ndarray]] = {}
    if pv_dir.exists():
        total_series: list[np.ndarray] = []
        for path in sorted(pv_dir.glob("*.csv")):
            if path.name.lower() == "temperature.csv":
                continue
            series = _load_profile_series(path)
            pv_profiles[_derive_pv_name(path.name)] = {"p_mw": series}
            total_series.append(series)
        if total_series:
            pv_profiles["total_pv_mw"] = {"p_mw": np.sum(total_series, axis=0)}

    wind_profiles: dict[str, dict[str, np.ndarray]] = {}
    if wind_dir.exists():
        for path in sorted(wind_dir.glob("wind_profile_*.csv")):
            series = _load_profile_series(path)
            name = Path(path.name).stem.replace("wind_profile_", "")
            wind_profiles[name] = {"p_mw": series}

    return load_profiles, (pv_profiles or None), (wind_profiles or None)


def build_hourly_totals(
    load_profiles: dict[str, dict[str, np.ndarray]] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if load_profiles is None:
        load_profiles, _, _ = build_hourly_profiles()
    series_list = [profile["p_mw"] for profile in load_profiles.values() if "p_mw" in profile]
    if not series_list:
        total_p = np.zeros(PROFILES_PER_YEAR, dtype=float)
    else:
        total_p = np.sum(series_list, axis=0)
    total_q = total_p * 0.2
    return total_p, total_q, total_p.copy()


def select_representative_days(total_p_mw: np.ndarray) -> dict[str, int]:
    if total_p_mw.size < PROFILES_PER_YEAR:
        raise ValueError("Total load profile length is shorter than expected.")
    daily_totals = total_p_mw[: PROFILES_PER_YEAR].reshape(365, PROFILES_PER_DAY).mean(axis=1)
    order = np.argsort(daily_totals)
    return {
        "offpeak": int(order[0]),
        "median": int(order[len(order) // 2]),
        "peak": int(order[-1]),
    }


def partition_zones(
    net: pp.pandapowerNet,
    n_zones: int,
    zone_bus_map: dict[int, list[str]],
) -> dict[int, int]:
    bus_name_to_zone: dict[str, int] = {}
    for zone, names in zone_bus_map.items():
        for name in names:
            bus_name_to_zone[str(name)] = int(zone)

    zone_map: dict[int, int] = {}
    for bus_idx, row in net.bus.iterrows():
        name = str(row.get("name", ""))
        zone = bus_name_to_zone.get(name, 1)
        zone_map[int(bus_idx)] = zone

    return zone_map


@dataclass(frozen=True)
class Layer0Result:
    lambda_dlmp: dict[int, float]
    alpha_star: dict[int, int]
    market_signals: dict[str, dict[str, float]]
    pricing_method: str
    socp_ac_gap_max: float
    ac_valid: bool


@dataclass(frozen=True)
class Layer0HourlyResult:
    day_label: str
    hour: int
    lambda_dlmp: dict[int, float]
    alpha_star: dict[int, int]
    market_signals: dict[str, dict[str, float]]
    pricing_method: str
    socp_ac_gap_max: float
    ac_valid: bool


@dataclass(frozen=True)
class Layer0CsvBundle:
    zone_prices_csv: Path
    alpha_csv: Path
    switches_csv: Path


def run_layer0_dso(
    net: pp.pandapowerNet | None = None,
    weights: ModelWeights | None = None,
    switch_cost: float = 0.01,
    solver_opts: SolverOptions | None = None,
    n_zones: int = 4,
    debug_reconfig: bool = False,
    force_switch_closed: bool = False,
    apply_voltage_bounds: bool = True,
    drop_isolated_loads: bool = False,
    fix_switch_status: bool = False,
    penalize_switch_changes: bool = True,
    enforce_radiality: bool = False,
    radiality_slack: int = 3,
    pricing_method: str = "load_weighted",
    ac_tolerance: float = 0.2,
) -> Layer0Result:
    if net is None:
        net = build_ieee123_net(mode="feeder123", balanced=True, convert_switches=True, slack_zones=None)

    validate_ieee123_net(net)

    if pricing_method not in PRICING_METHODS:
        raise ValueError(f"pricing_method must be one of {sorted(PRICING_METHODS)}")

    if drop_isolated_loads:
        _drop_isolated_loads(net)

    result = run_reconfiguration_detailed(
        net=net,
        weights=weights,
        switch_cost=switch_cost,
        solver_opts=solver_opts,
        debug=debug_reconfig,
        force_switch_closed=force_switch_closed,
        apply_voltage_bounds=apply_voltage_bounds,
        fix_switch_status=fix_switch_status,
        penalize_switch_changes=penalize_switch_changes,
        enforce_radiality=enforce_radiality,
        radiality_slack=radiality_slack,
    )
    validation = validate_socp_against_ac(
        net,
        result.alpha_star,
        result.socp_voltage_squared,
        tolerance=ac_tolerance,
    )

    zone_map = partition_zones(net, n_zones=n_zones, zone_bus_map=IEEE123_ZONE_BUS_MAP)
    market_signals = generate_market_signals(
        net,
        lambda_dlmp=result.lambda_dlmp,
        zone_map=zone_map,
        pricing_method=pricing_method,
    )

    return Layer0Result(
        lambda_dlmp=result.lambda_dlmp,
        alpha_star=result.alpha_star,
        market_signals=market_signals,
        pricing_method=pricing_method,
        socp_ac_gap_max=validation.socp_ac_gap_max,
        ac_valid=validation.ac_valid,
    )


def _apply_hourly_profiles(
    net: pp.pandapowerNet,
    load_profiles: dict[str, dict[str, np.ndarray]],
    pv_profiles: dict[str, dict[str, np.ndarray]] | None,
    wind_profiles: dict[str, dict[str, np.ndarray]] | None,
    hour_index: int,
    name_map: dict[str, str] | None = None,
    bess_soc: dict[int, float] | None = None,
    bess_prev: dict[int, float] | None = None,
    thermal_prev: dict[int, float] | None = None,
    timestep_hours: float = 1.0,
    bess_energy_hours: float = 4.0,
    bess_efficiency: float = 0.9,
    ramp_fraction: float = 0.1,
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
            base_p = float(row.get("p_mw", 0.0))
            base_q = float(row.get("q_mvar", 0.0))
            shape = float(profile["p_mw"][hour_index])
            net.load.at[idx, "p_mw"] = base_p * shape
            net.load.at[idx, "q_mvar"] = base_q * shape
            updated_loads += 1

    pv_output = 0.0
    wind_output = 0.0
    baseline_pv = 0.0
    baseline_wind = 0.0

    if pv_profiles:
        total_profile = pv_profiles.get("total_pv_mw")
        if total_profile is not None:
            total_series = total_profile.get("p_mw")
            if total_series is not None:
                pv_output = float(total_series[hour_index])
                baseline_pv = float(np.mean(total_series))

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
                bus_idx = row.get("bus")
                if bus_idx is None or bus_idx not in net.bus.index:
                    continue
                bus_name = str(net.bus.at[bus_idx, "name"])
                profile = wind_profiles.get(bus_name)
                if profile is None:
                    continue
                profile_series = profile.get("p_mw")
                if profile_series is None:
                    continue
                base_p = float(row.get("p_mw", 0.0))
                factor = float(profile_series[hour_index])
                net.sgen.at[idx, "p_mw"] = base_p * factor
                wind_output += base_p * factor
                baseline_wind += base_p * float(np.mean(profile_series))

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
            soc = 0.5
            if bess_soc is not None:
                soc = bess_soc.get(idx, soc)
            energy_mwh = nameplate * bess_energy_hours
            max_delta = nameplate * timestep_hours
            requested = -renewable_deviation
            prev = 0.0 if bess_prev is None else bess_prev.get(idx, 0.0)
            requested = max(prev - max_delta, min(prev + max_delta, requested))
            if requested >= 0:
                discharge_limit = soc * energy_mwh / max(timestep_hours, 1e-6)
                bess_setpoint = min(requested, nameplate, discharge_limit)
                soc -= (bess_setpoint * timestep_hours) / energy_mwh
            else:
                charge_limit = (1.0 - soc) * energy_mwh / max(timestep_hours, 1e-6)
                bess_setpoint = max(requested, -nameplate, -charge_limit)
                soc -= (bess_setpoint * timestep_hours) * bess_efficiency / energy_mwh
            soc = max(0.0, min(1.0, soc))
            net.sgen.at[idx, "p_mw"] = bess_setpoint
            if bess_soc is not None:
                bess_soc[idx] = soc
            if bess_prev is not None:
                bess_prev[idx] = bess_setpoint

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
            target = max(min_output, nameplate - renewable_deviation)
            target = min(nameplate, target)
            prev = target if thermal_prev is None else thermal_prev.get(idx, target)
            ramp = ramp_fraction * nameplate
            thermal_setpoint = max(prev - ramp, min(prev + ramp, target))
            net.sgen.at[idx, "p_mw"] = thermal_setpoint
            if thermal_prev is not None:
                thermal_prev[idx] = thermal_setpoint

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


def _zone_price_records(
    day_label: str,
    hour: int,
    market_signals: dict[str, dict[str, float]],
    pricing_method: str,
    socp_ac_gap_max: float,
    ac_valid: bool,
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for zone, data in market_signals.items():
        records.append(
            {
                "day": day_label,
                "hour": hour,
                "zone": zone,
                "energy_price": float(data.get("energy_price", float("nan"))),
                "reserve_price": float(data.get("reserve_price", float("nan"))),
                "pricing_method": pricing_method,
                "socp_ac_gap_max": float(socp_ac_gap_max),
                "ac_valid": bool(ac_valid),
            }
        )
    return records


def _keep_hv_slack_only(net: pp.pandapowerNet) -> None:
    if net.ext_grid.empty:
        return
    if net.trafo.empty:
        return
    hv_buses = set(int(bus) for bus in net.trafo.hv_bus.values)
    hv_ext = net.ext_grid[net.ext_grid.bus.isin(hv_buses)].copy()
    if not hv_ext.empty:
        net.ext_grid = hv_ext.reset_index(drop=True)


def _drop_isolated_loads(net: pp.pandapowerNet) -> list[int]:
    if net.load.empty:
        return []
    connected_buses: set[int] = set()
    if not net.line.empty:
        connected_buses.update(net.line["from_bus"].astype(int).tolist())
        connected_buses.update(net.line["to_bus"].astype(int).tolist())
    if not net.trafo.empty:
        connected_buses.update(net.trafo["hv_bus"].astype(int).tolist())
        connected_buses.update(net.trafo["lv_bus"].astype(int).tolist())
    if not net.switch.empty:
        connected_buses.update(net.switch["bus"].astype(int).tolist())
        bus_switches = net.switch[net.switch["et"].astype(str) == "b"]
        if not bus_switches.empty:
            connected_buses.update(bus_switches["element"].astype(int).tolist())

    dropped_buses: list[int] = []
    for idx, row in net.load.iterrows():
        bus = int(row.get("bus"))
        if bus in connected_buses:
            continue
        if abs(float(row.get("p_mw", 0.0))) <= 1e-9 and abs(float(row.get("q_mvar", 0.0))) <= 1e-9:
            continue
        net.load.at[idx, "p_mw"] = 0.0
        net.load.at[idx, "q_mvar"] = 0.0
        dropped_buses.append(bus)
    return sorted(set(dropped_buses))


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
    debug_reconfig: bool = False,
    force_switch_closed: bool = False,
    apply_voltage_bounds: bool = True,
    drop_isolated_loads: bool = False,
    fix_switch_status: bool = False,
    penalize_switch_changes: bool = True,
    enforce_radiality: bool = False,
    radiality_slack: int = 3,
    pricing_method: str = "load_weighted",
    ac_tolerance: float = 0.2,
) -> list[Layer0HourlyResult]:
    if pricing_method not in PRICING_METHODS:
        raise ValueError(f"pricing_method must be one of {sorted(PRICING_METHODS)}")

    results: list[Layer0HourlyResult] = []
    bess_soc: dict[int, float] = {}
    bess_prev: dict[int, float] = {}
    thermal_prev: dict[int, float] = {}
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
            bess_soc=bess_soc,
            bess_prev=bess_prev,
            thermal_prev=thermal_prev,
        )
        _warn_if_profile_mismatch(updated_loads, len(hour_net.load), f"{day_label} hour {hour_offset}", "load")
        _warn_if_profile_mismatch(updated_sgens, len(hour_net.sgen), f"{day_label} hour {hour_offset}", "pv")
        if drop_isolated_loads:
            _drop_isolated_loads(hour_net)
        reconfig_result = run_reconfiguration_detailed(
            net=hour_net,
            weights=weights,
            switch_cost=switch_cost,
            solver_opts=solver_opts,
            debug=debug_reconfig,
            force_switch_closed=force_switch_closed,
            apply_voltage_bounds=apply_voltage_bounds,
            fix_switch_status=fix_switch_status,
            penalize_switch_changes=penalize_switch_changes,
            enforce_radiality=enforce_radiality,
            radiality_slack=radiality_slack,
        )
        validation = validate_socp_against_ac(
            hour_net,
            reconfig_result.alpha_star,
            reconfig_result.socp_voltage_squared,
            tolerance=ac_tolerance,
        )
        zone_map = partition_zones(hour_net, n_zones=n_zones, zone_bus_map=IEEE123_ZONE_BUS_MAP)
        market_signals = generate_market_signals(
            hour_net,
            lambda_dlmp=reconfig_result.lambda_dlmp,
            zone_map=zone_map,
            pricing_method=pricing_method,
        )
        results.append(
            Layer0HourlyResult(
                day_label=day_label,
                hour=hour_offset,
                lambda_dlmp=reconfig_result.lambda_dlmp,
                alpha_star=reconfig_result.alpha_star,
                market_signals=market_signals,
                pricing_method=pricing_method,
                socp_ac_gap_max=validation.socp_ac_gap_max,
                ac_valid=validation.ac_valid,
            )
        )
    return results


def run_layer0_pipeline(
    output_dir: Path,
    pricing_method: str = "load_weighted",
    ac_tolerance: float = 0.2,
    debug_reconfig: bool = False,
    force_switch_closed: bool = True,
    apply_voltage_bounds: bool = True,
    drop_isolated_loads: bool = False,
) -> Layer0CsvBundle:
    output_dir.mkdir(parents=True, exist_ok=True)

    net = build_ieee123_net(mode="feeder123", balanced=True, convert_switches=True, slack_zones=None)
    validate_ieee123_net(net)

    wind_bus_names: list[str] = []
    if not net.sgen.empty:
        for _, row in net.sgen.iterrows():
            if str(row.get("type", "")).lower() != "wind":
                continue
            bus_idx = row.get("bus")
            if bus_idx is None or bus_idx not in net.bus.index:
                continue
            bus_name = str(net.bus.at[bus_idx, "name"])
            if bus_name:
                wind_bus_names.append(bus_name)
    wind_bus_names = sorted(set(wind_bus_names))
    wind_profiles = _ensure_wind_profiles(wind_bus_names) if wind_bus_names else None

    load_profiles, pv_profiles, _ = build_hourly_profiles()
    total_p_mw, _, _ = build_hourly_totals(load_profiles)
    day_map = select_representative_days(total_p_mw)

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
                debug_reconfig=debug_reconfig,
                force_switch_closed=force_switch_closed,
                apply_voltage_bounds=apply_voltage_bounds,
                drop_isolated_loads=drop_isolated_loads,
                pricing_method=pricing_method,
                ac_tolerance=ac_tolerance,
            )
        )

    return export_layer0_csvs(output_dir, all_results)


def export_layer0_csvs(output_dir: Path, hourly_results: list[Layer0HourlyResult]) -> Layer0CsvBundle:
    zone_rows: list[dict[str, object]] = []
    alpha_rows: list[dict[str, object]] = []
    switch_rows: list[dict[str, object]] = []
    switch_map = switch_edge_map(build_ieee123_net(mode="feeder123", balanced=True, convert_switches=True, slack_zones=None))
    for result in hourly_results:
        zone_rows.extend(
            _zone_price_records(
                result.day_label,
                result.hour,
                result.market_signals,
                pricing_method=result.pricing_method,
                socp_ac_gap_max=result.socp_ac_gap_max,
                ac_valid=result.ac_valid,
            )
        )
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

    return Layer0CsvBundle(
        zone_prices_csv=zone_path,
        alpha_csv=alpha_path,
        switches_csv=switch_path,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Layer 0 DSO reconfiguration.")
    parser.add_argument("--debug-reconfig", action="store_true", help="Enable reconfiguration infeasibility diagnostics.")
    parser.add_argument("--no-voltage-bounds", action="store_true", help="Disable voltage magnitude bounds.")
    parser.add_argument(
        "--no-force-switches-closed",
        action="store_true",
        help="Do not force CB switches closed before solving.",
    )
    parser.add_argument(
        "--drop-isolated-loads",
        action="store_true",
        help="Zero loads on buses with no incident edges.",
    )
    parser.add_argument(
        "--pricing-method",
        choices=sorted(PRICING_METHODS),
        default="load_weighted",
        help="Zonal aggregation method for energy prices.",
    )
    parser.add_argument(
        "--ac-tolerance",
        type=float,
        default=0.2,
        help="SOCP vs AC validation tolerance in p.u.",
    )
    args = parser.parse_args()

    if args.debug_reconfig:
        logging.basicConfig(level=logging.INFO)

    output_dir = Path(__file__).resolve().parents[2] / "data" / "oedisi-ieee123-main" / "profiles" / "layer0_hourly"
    csv_bundle = run_layer0_pipeline(
        output_dir=output_dir,
        pricing_method=args.pricing_method,
        ac_tolerance=args.ac_tolerance,
        debug_reconfig=args.debug_reconfig,
        force_switch_closed=not args.no_force_switches_closed,
        apply_voltage_bounds=not args.no_voltage_bounds,
        drop_isolated_loads=args.drop_isolated_loads,
    )
    print("Exported Layer 0 hourly CSVs:")
    print(f"- Zone prices: {csv_bundle.zone_prices_csv}")
    print(f"- Alpha decisions: {csv_bundle.alpha_csv}")
    print(f"- Switch decisions: {csv_bundle.switches_csv}")


if __name__ == "__main__":
    main()
