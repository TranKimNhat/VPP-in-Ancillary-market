from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
import argparse
import copy
import hashlib
import logging
import re
import sys
import warnings
from typing import Mapping, Sequence

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
class ZoneScoringConfig:
    min_buses_per_zone: int = 8
    max_buses_per_zone: int = 25
    max_bus_imbalance_ratio: float = 2.5
    min_load_buses_per_zone: int = 5
    min_load_share: float = 0.08
    max_load_share: float = 0.35
    max_load_imbalance_ratio: float = 3.0
    min_der_penetration: float = 0.10
    max_der_penetration: float = 0.60
    max_boundary_edges_per_zone: int = 3
    weights: Mapping[str, float] = (
        ("connectivity", 20.0),
        ("bus_size", 15.0),
        ("load_balance", 20.0),
        ("der_penetration", 15.0),
        ("load_bus_count", 10.0),
        ("boundary_cut", 10.0),
        ("zone_count", 10.0),
    )


@dataclass(frozen=True)
class ZoneScoreResult:
    total_score: float
    metrics: dict[str, float]
    penalties: dict[str, float]


def _linear_score(value: float, low: float, high: float) -> float:
    if low > high:
        low, high = high, low
    if low <= value <= high:
        return 1.0
    if value < low:
        scale = max(abs(low), 1e-9)
        return max(0.0, 1.0 - (low - value) / scale)
    scale = max(abs(high), 1e-9)
    return max(0.0, 1.0 - (value - high) / scale)


def _build_bus_adjacency(net: pp.pandapowerNet) -> dict[int, set[int]]:
    adjacency: dict[int, set[int]] = {int(bus): set() for bus in net.bus.index}

    if not net.line.empty:
        for _, row in net.line.iterrows():
            u = int(row["from_bus"])
            v = int(row["to_bus"])
            adjacency.setdefault(u, set()).add(v)
            adjacency.setdefault(v, set()).add(u)

    if not net.switch.empty:
        for _, row in net.switch.iterrows():
            if str(row.get("et", "")) != "b":
                continue
            if not bool(row.get("closed", True)):
                continue
            u = int(row["bus"])
            v = int(row["element"])
            adjacency.setdefault(u, set()).add(v)
            adjacency.setdefault(v, set()).add(u)

    return adjacency


def _zone_bus_groups(zone_map: Mapping[int, int]) -> dict[int, list[int]]:
    groups: dict[int, list[int]] = defaultdict(list)
    for bus, zone in zone_map.items():
        groups[int(zone)].append(int(bus))
    return dict(groups)


def _connected_fraction(buses: Sequence[int], adjacency: Mapping[int, set[int]]) -> float:
    bus_set = {int(bus) for bus in buses}
    if not bus_set:
        return 0.0
    visited: set[int] = set()
    stack = [next(iter(bus_set))]
    while stack:
        node = stack.pop()
        if node in visited:
            continue
        visited.add(node)
        for nbr in adjacency.get(node, set()):
            if nbr in bus_set and nbr not in visited:
                stack.append(nbr)
    return float(len(visited)) / float(len(bus_set))


def score_zone_partition(
    net: pp.pandapowerNet,
    zone_map: Mapping[int, int],
    config: ZoneScoringConfig | None = None,
) -> ZoneScoreResult:
    cfg = config or ZoneScoringConfig()
    zone_groups = _zone_bus_groups(zone_map)
    if not zone_groups:
        return ZoneScoreResult(total_score=0.0, metrics={}, penalties={"empty_partition": 1.0})

    adjacency = _build_bus_adjacency(net)

    load_by_bus: dict[int, float] = defaultdict(float)
    if not net.load.empty:
        for _, row in net.load.iterrows():
            load_by_bus[int(row["bus"])] += float(row.get("p_mw", 0.0))

    der_by_bus: dict[int, float] = defaultdict(float)
    if not net.sgen.empty:
        for _, row in net.sgen.iterrows():
            der_by_bus[int(row["bus"])] += max(float(row.get("p_mw", 0.0)), 0.0)
    if not net.gen.empty:
        for _, row in net.gen.iterrows():
            der_by_bus[int(row["bus"])] += max(float(row.get("p_mw", 0.0)), 0.0)

    zone_bus_counts = np.array([len(buses) for buses in zone_groups.values()], dtype=float)
    zone_loads = np.array([sum(load_by_bus.get(bus, 0.0) for bus in buses) for buses in zone_groups.values()], dtype=float)
    zone_ders = np.array([sum(der_by_bus.get(bus, 0.0) for bus in buses) for buses in zone_groups.values()], dtype=float)
    zone_load_bus_counts = np.array([
        sum(1 for bus in buses if load_by_bus.get(bus, 0.0) > 1e-9) for buses in zone_groups.values()
    ], dtype=float)

    total_load = float(np.sum(zone_loads))
    load_shares = zone_loads / total_load if total_load > 1e-9 else np.zeros_like(zone_loads)
    der_penetration = np.divide(zone_ders, np.maximum(zone_loads, 1e-9))

    boundary_edges: dict[int, int] = defaultdict(int)
    for bus, nbrs in adjacency.items():
        zone_u = int(zone_map.get(bus, -1))
        for nbr in nbrs:
            if bus >= nbr:
                continue
            zone_v = int(zone_map.get(nbr, -1))
            if zone_u != zone_v:
                boundary_edges[zone_u] += 1
                boundary_edges[zone_v] += 1

    connectivity_scores = [
        _connected_fraction(buses, adjacency)
        for buses in zone_groups.values()
    ]

    bus_ratio = float(np.max(zone_bus_counts) / max(np.min(zone_bus_counts), 1.0))
    load_ratio = float(np.max(zone_loads) / max(np.min(zone_loads[zone_loads > 1e-9], default=1.0), 1e-9))

    metrics = {
        "n_zones": float(len(zone_groups)),
        "bus_count_min": float(np.min(zone_bus_counts)),
        "bus_count_max": float(np.max(zone_bus_counts)),
        "bus_imbalance_ratio": bus_ratio,
        "load_share_min": float(np.min(load_shares)) if load_shares.size else 0.0,
        "load_share_max": float(np.max(load_shares)) if load_shares.size else 0.0,
        "load_imbalance_ratio": load_ratio,
        "der_penetration_min": float(np.min(der_penetration)) if der_penetration.size else 0.0,
        "der_penetration_max": float(np.max(der_penetration)) if der_penetration.size else 0.0,
        "load_bus_count_min": float(np.min(zone_load_bus_counts)) if zone_load_bus_counts.size else 0.0,
        "boundary_edges_max": float(max(boundary_edges.values()) if boundary_edges else 0.0),
        "connectivity_min": float(min(connectivity_scores) if connectivity_scores else 0.0),
    }

    weight_map = dict(cfg.weights)

    components = {
        "connectivity": metrics["connectivity_min"],
        "bus_size": float(np.mean([
            _linear_score(v, cfg.min_buses_per_zone, cfg.max_buses_per_zone)
            for v in zone_bus_counts
        ])),
        "load_balance": float(np.mean([
            _linear_score(v, cfg.min_load_share, cfg.max_load_share)
            for v in load_shares
        ])) * _linear_score(metrics["load_imbalance_ratio"], 1.0, cfg.max_load_imbalance_ratio),
        "der_penetration": float(np.mean([
            _linear_score(v, cfg.min_der_penetration, cfg.max_der_penetration)
            for v in der_penetration
        ])),
        "load_bus_count": float(np.mean([
            _linear_score(v, cfg.min_load_buses_per_zone, max(cfg.min_load_buses_per_zone, v))
            for v in zone_load_bus_counts
        ])),
        "boundary_cut": _linear_score(metrics["boundary_edges_max"], 0.0, cfg.max_boundary_edges_per_zone),
        "zone_count": _linear_score(float(len(zone_groups)), max(2.0, float(cfg.min_buses_per_zone // 4)), 8.0),
    }

    penalties = {
        "bus_imbalance_penalty": 1.0 - _linear_score(metrics["bus_imbalance_ratio"], 1.0, cfg.max_bus_imbalance_ratio),
        "load_imbalance_penalty": 1.0 - _linear_score(metrics["load_imbalance_ratio"], 1.0, cfg.max_load_imbalance_ratio),
    }

    total_weight = float(sum(weight_map.get(k, 0.0) for k in components))
    weighted_sum = sum(weight_map.get(k, 0.0) * components[k] for k in components)
    total_score = 100.0 * (weighted_sum / total_weight) if total_weight > 0 else 0.0
    total_score *= max(0.0, 1.0 - 0.5 * penalties["bus_imbalance_penalty"] - 0.5 * penalties["load_imbalance_penalty"])

    return ZoneScoreResult(
        total_score=float(np.clip(total_score, 0.0, 100.0)),
        metrics=metrics,
        penalties=penalties,
    )


def select_best_zone_partition(
    net: pp.pandapowerNet,
    candidate_zone_maps: Sequence[Mapping[int, int]],
    config: ZoneScoringConfig | None = None,
) -> tuple[dict[int, int], ZoneScoreResult, list[ZoneScoreResult]]:
    if not candidate_zone_maps:
        raise ValueError("candidate_zone_maps must not be empty.")

    scored = [score_zone_partition(net, zone_map=candidate, config=config) for candidate in candidate_zone_maps]
    best_idx = int(np.argmax([result.total_score for result in scored]))
    best_map = {int(bus): int(zone) for bus, zone in candidate_zone_maps[best_idx].items()}
    return best_map, scored[best_idx], scored


@dataclass(frozen=True)
class Layer0Result:
    lambda_dlmp: dict[int, float]
    alpha_star: dict[int, int]
    market_signals: dict[str, dict[str, float]]
    pricing_method: str
    socp_ac_gap_max: float
    socp_ac_gap_p95: float
    socp_ac_gap_p50: float
    socp_ac_worst_bus: int | None
    socp_ac_compared_bus_count: int
    ac_converged: bool
    ac_valid: bool
    soc_slack_max: float
    soc_slack_sum: float
    voltage_drop_slack_max: float
    voltage_drop_slack_sum: float


@dataclass(frozen=True)
class Layer0HourlyResult:
    day_label: str
    hour: int
    lambda_dlmp: dict[int, float]
    alpha_star: dict[int, int]
    market_signals: dict[str, dict[str, float]]
    pricing_method: str
    socp_ac_gap_max: float
    socp_ac_gap_p95: float
    socp_ac_gap_p50: float
    socp_ac_worst_bus: int | None
    socp_ac_compared_bus_count: int
    ac_converged: bool
    ac_valid: bool
    soc_slack_max: float
    soc_slack_sum: float
    voltage_drop_slack_max: float
    voltage_drop_slack_sum: float


@dataclass(frozen=True)
class Layer0CsvBundle:
    zone_prices_csv: Path
    alpha_csv: Path
    switches_csv: Path
    diagnostics_csv: Path
    valid_for_layer1: bool


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
    enforce_radiality: bool = True,
    radiality_slack: int = 0,
    pricing_method: str = "load_weighted",
    ac_tolerance: float = 0.01,
    soc_relax: float = 1.001,
    soc_slack_cap: float = 0.0,
    voltage_drop_slack_cap: float = 0.0,
    voltage_reference_upper_band: float = 0.01,
) -> Layer0Result:
    if net is None:
        net = build_ieee123_net(
            mode="feeder123",
            balanced=True,
            convert_switches=True,
            slack_zones={1},
            source_mode="publish",
        )
        _keep_hv_slack_only(net)

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
        soc_relax=soc_relax,
        soc_slack_cap=soc_slack_cap,
        voltage_drop_slack_cap=voltage_drop_slack_cap,
        voltage_reference_upper_band=voltage_reference_upper_band,
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
        socp_ac_gap_p95=validation.socp_ac_gap_p95,
        socp_ac_gap_p50=validation.socp_ac_gap_p50,
        socp_ac_worst_bus=validation.worst_bus,
        socp_ac_compared_bus_count=validation.compared_bus_count,
        ac_converged=validation.converged,
        ac_valid=validation.ac_valid,
        soc_slack_max=result.soc_slack_max,
        soc_slack_sum=result.soc_slack_sum,
        voltage_drop_slack_max=result.voltage_drop_slack_max,
        voltage_drop_slack_sum=result.voltage_drop_slack_sum,
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
    timestep_hours: float = 0.25,
    bess_energy_hours: float = 4.0,
    bess_efficiency: float = 0.9,
    ramp_fraction: float = 0.1,
) -> tuple[int, int, int]:
    def _safe_nameplate(row: pd.Series) -> float:
        p_raw = row.get("p_mw", 0.0)
        try:
            p_fallback = float(p_raw)
        except (TypeError, ValueError):
            p_fallback = 0.0
        if np.isnan(p_fallback):
            p_fallback = 0.0

        raw = row.get("sn_mva", p_fallback)
        try:
            value = float(raw)
        except (TypeError, ValueError):
            value = p_fallback
        if np.isnan(value) or value <= 0.0:
            value = p_fallback
        if np.isnan(value):
            return 0.0
        return abs(value)

    updated_loads = 0
    updated_sgens = 0
    pv_total = 0
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

    pv_total_series = None
    pv_total_scale = 1.0
    if pv_profiles:
        total_profile = pv_profiles.get("total_pv_mw")
        if total_profile is not None:
            total_series = total_profile.get("p_mw")
            if total_series is not None:
                pv_total_series = np.asarray(total_series, dtype=float)
                if pv_total_series.size > 0:
                    pv_total_scale = max(float(np.max(np.clip(pv_total_series, 0.0, None))), 1e-9)

    if not net.sgen.empty:
        for idx, row in net.sgen.iterrows():
            sgen_type = str(row.get("type", "")).lower()
            if sgen_type == "pv":
                pv_total += 1
            nameplate = _safe_nameplate(row)
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
                profile_series = profile.get("p_mw") if profile is not None else pv_total_series
                if profile_series is None or nameplate <= 0.0:
                    continue
                profile_arr = np.asarray(profile_series, dtype=float)
                if profile_arr.size <= hour_index:
                    continue
                if profile is None:
                    factor_series = np.clip(profile_arr / pv_total_scale, 0.0, 1.0)
                else:
                    factor_series = np.clip(profile_arr, 0.0, 1.0)
                factor = float(factor_series[hour_index])
                net.sgen.at[idx, "p_mw"] = nameplate * factor
                pv_output += nameplate * factor
                baseline_pv += nameplate * float(np.mean(factor_series))
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
                if profile_series is None or nameplate <= 0.0:
                    continue
                factor = float(np.clip(profile_series[hour_index], 0.0, 1.0))
                net.sgen.at[idx, "p_mw"] = nameplate * factor
                wind_output += nameplate * factor
                baseline_wind += nameplate * float(np.mean(np.clip(profile_series, 0.0, 1.0)))

    renewable_deviation = (pv_output + wind_output) - (baseline_pv + baseline_wind)

    if not net.sgen.empty:
        for idx, row in net.sgen.iterrows():
            sgen_type = str(row.get("type", "")).lower()
            if sgen_type not in {"bess", "storage"}:
                continue
            nameplate = _safe_nameplate(row)
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
            sgen_type = str(row.get("type", "")).lower()
            if sgen_type != "thermal":
                continue
            nameplate = _safe_nameplate(row)
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

    return updated_loads, updated_sgens, pv_total


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


def _safe_metric(value: object) -> float:
    try:
        metric = float(value)
    except (TypeError, ValueError):
        return float("inf")
    if not np.isfinite(metric):
        return float("inf")
    return metric


def _zone_price_records(
    day_label: str,
    hour: int,
    market_signals: dict[str, dict[str, float]],
    pricing_method: str,
    socp_ac_gap_max: float,
    socp_ac_gap_p95: float,
    socp_ac_gap_p50: float,
    socp_ac_worst_bus: int | None,
    socp_ac_compared_bus_count: int,
    ac_converged: bool,
    ac_valid: bool,
    soc_slack_max: float,
    soc_slack_sum: float,
    voltage_drop_slack_max: float,
    voltage_drop_slack_sum: float,
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
                "socp_ac_gap_max": _safe_metric(socp_ac_gap_max),
                "socp_ac_gap_p95": _safe_metric(socp_ac_gap_p95),
                "socp_ac_gap_p50": _safe_metric(socp_ac_gap_p50),
                "socp_ac_worst_bus": socp_ac_worst_bus,
                "socp_ac_compared_bus_count": int(socp_ac_compared_bus_count),
                "ac_converged": bool(ac_converged),
                "ac_valid": bool(ac_valid),
                "soc_slack_max": float(soc_slack_max),
                "soc_slack_sum": float(soc_slack_sum),
                "voltage_drop_slack_max": float(voltage_drop_slack_max),
                "voltage_drop_slack_sum": float(voltage_drop_slack_sum),
            }
        )
    return records


def _keep_hv_slack_only(net: pp.pandapowerNet) -> None:
    if net.ext_grid.empty:
        return
    if net.trafo.empty:
        net.ext_grid = net.ext_grid.iloc[:1].copy().reset_index(drop=True)
        return
    hv_buses = set(int(bus) for bus in net.trafo.hv_bus.values)
    hv_ext = net.ext_grid[net.ext_grid.bus.isin(hv_buses)].copy()
    if not hv_ext.empty:
        net.ext_grid = hv_ext.iloc[:1].copy().reset_index(drop=True)
        return
    net.ext_grid = net.ext_grid.iloc[:1].copy().reset_index(drop=True)


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
    enforce_radiality: bool = True,
    radiality_slack: int = 0,
    pricing_method: str = "load_weighted",
    ac_tolerance: float = 0.01,
    soc_relax: float = 1.001,
    soc_slack_cap: float = 0.0,
    voltage_drop_slack_cap: float = 0.0,
    voltage_reference_upper_band: float = 0.01,
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
        updated_loads, updated_sgens, pv_total = _apply_hourly_profiles(
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
        if pv_profiles is not None:
            _warn_if_profile_mismatch(updated_sgens, pv_total, f"{day_label} hour {hour_offset}", "pv")
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
            soc_relax=soc_relax,
            soc_slack_cap=soc_slack_cap,
            voltage_drop_slack_cap=voltage_drop_slack_cap,
            voltage_reference_upper_band=voltage_reference_upper_band,
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
                socp_ac_gap_p95=validation.socp_ac_gap_p95,
                socp_ac_gap_p50=validation.socp_ac_gap_p50,
                socp_ac_worst_bus=validation.worst_bus,
                socp_ac_compared_bus_count=validation.compared_bus_count,
                ac_converged=validation.converged,
                ac_valid=validation.ac_valid,
                soc_slack_max=reconfig_result.soc_slack_max,
                soc_slack_sum=reconfig_result.soc_slack_sum,
                voltage_drop_slack_max=reconfig_result.voltage_drop_slack_max,
                voltage_drop_slack_sum=reconfig_result.voltage_drop_slack_sum,
            )
        )
    return results


def run_layer0_pipeline(
    output_dir: Path,
    pricing_method: str = "load_weighted",
    ac_tolerance: float = 0.01,
    debug_reconfig: bool = False,
    force_switch_closed: bool = True,
    apply_voltage_bounds: bool = True,
    drop_isolated_loads: bool = False,
    enforce_radiality: bool = True,
    radiality_slack: int = 0,
    soc_relax: float = 1.001,
    soc_slack_cap: float = 0.0,
    voltage_drop_slack_cap: float = 0.0,
    voltage_reference_upper_band: float = 0.01,
    diagnostics_only_on_fail: bool = True,
) -> Layer0CsvBundle:
    output_dir.mkdir(parents=True, exist_ok=True)

    net = build_ieee123_net(
        mode="feeder123",
        balanced=True,
        convert_switches=True,
        slack_zones={1},
        source_mode="publish",
    )
    _keep_hv_slack_only(net)
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
                enforce_radiality=enforce_radiality,
                radiality_slack=radiality_slack,
                pricing_method=pricing_method,
                ac_tolerance=ac_tolerance,
                soc_relax=soc_relax,
                soc_slack_cap=soc_slack_cap,
                voltage_drop_slack_cap=voltage_drop_slack_cap,
                voltage_reference_upper_band=voltage_reference_upper_band,
            )
        )

    soc_slack_limit = max(float(soc_slack_cap), 0.0) + 1e-6
    voltage_drop_slack_limit = max(float(voltage_drop_slack_cap), 0.0) + 1e-6
    valid_for_layer1 = bool(
        all(
            result.ac_valid
            and result.ac_converged
            and float(result.soc_slack_max) <= soc_slack_limit
            and float(result.voltage_drop_slack_max) <= voltage_drop_slack_limit
            for result in all_results
        )
    )
    diagnostics_path = output_dir / "layer0_diagnostics.csv"
    if diagnostics_only_on_fail and not valid_for_layer1:
        return export_layer0_csvs(output_dir, all_results, valid_for_layer1=False, diagnostics_csv=diagnostics_path)
    return export_layer0_csvs(output_dir, all_results, valid_for_layer1=valid_for_layer1, diagnostics_csv=diagnostics_path)


def export_layer0_csvs(
    output_dir: Path,
    hourly_results: list[Layer0HourlyResult],
    *,
    valid_for_layer1: bool,
    diagnostics_csv: Path,
) -> Layer0CsvBundle:
    zone_rows: list[dict[str, object]] = []
    alpha_rows: list[dict[str, object]] = []
    switch_rows: list[dict[str, object]] = []
    switch_map = switch_edge_map(
        build_ieee123_net(
            mode="feeder123",
            balanced=True,
            convert_switches=True,
            slack_zones={1},
            source_mode="publish",
        )
    )
    for result in hourly_results:
        zone_rows.extend(
            _zone_price_records(
                result.day_label,
                result.hour,
                result.market_signals,
                pricing_method=result.pricing_method,
                socp_ac_gap_max=result.socp_ac_gap_max,
                socp_ac_gap_p95=result.socp_ac_gap_p95,
                socp_ac_gap_p50=result.socp_ac_gap_p50,
                socp_ac_worst_bus=result.socp_ac_worst_bus,
                socp_ac_compared_bus_count=result.socp_ac_compared_bus_count,
                ac_converged=result.ac_converged,
                ac_valid=result.ac_valid,
                soc_slack_max=result.soc_slack_max,
                soc_slack_sum=result.soc_slack_sum,
                voltage_drop_slack_max=result.voltage_drop_slack_max,
                voltage_drop_slack_sum=result.voltage_drop_slack_sum,
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

    diagnostics_rows = []
    for result in hourly_results:
        diagnostics_rows.append(
            {
                "day": result.day_label,
                "hour": result.hour,
                "socp_ac_gap_max": _safe_metric(result.socp_ac_gap_max),
                "socp_ac_gap_p95": _safe_metric(result.socp_ac_gap_p95),
                "socp_ac_gap_p50": _safe_metric(result.socp_ac_gap_p50),
                "socp_ac_worst_bus": result.socp_ac_worst_bus,
                "socp_ac_compared_bus_count": int(result.socp_ac_compared_bus_count),
                "ac_converged": bool(result.ac_converged),
                "ac_valid": bool(result.ac_valid),
                "soc_slack_max": float(result.soc_slack_max),
                "soc_slack_sum": float(result.soc_slack_sum),
                "voltage_drop_slack_max": float(result.voltage_drop_slack_max),
                "voltage_drop_slack_sum": float(result.voltage_drop_slack_sum),
            }
        )

    pd.DataFrame(diagnostics_rows).to_csv(diagnostics_csv, index=False)

    if valid_for_layer1:
        pd.DataFrame(zone_rows).to_csv(zone_path, index=False)
        pd.DataFrame(alpha_rows).to_csv(alpha_path, index=False)
        pd.DataFrame(switch_rows).to_csv(switch_path, index=False)
    else:
        pd.DataFrame(
            columns=[
                "day",
                "hour",
                "zone",
                "energy_price",
                "reserve_price",
                "pricing_method",
                "socp_ac_gap_max",
                "socp_ac_gap_p95",
                "socp_ac_gap_p50",
                "socp_ac_worst_bus",
                "socp_ac_compared_bus_count",
                "ac_converged",
                "ac_valid",
                "soc_slack_max",
                "soc_slack_sum",
                "voltage_drop_slack_max",
                "voltage_drop_slack_sum",
            ]
        ).to_csv(zone_path, index=False)
        pd.DataFrame(columns=["day", "hour", "edge_id", "alpha"]).to_csv(alpha_path, index=False)
        pd.DataFrame(columns=["day", "hour", "edge_id", "alpha", "from_bus", "to_bus"]).to_csv(
            switch_path, index=False
        )

    return Layer0CsvBundle(
        zone_prices_csv=zone_path,
        alpha_csv=alpha_path,
        switches_csv=switch_path,
        diagnostics_csv=diagnostics_csv,
        valid_for_layer1=valid_for_layer1,
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
        default=0.01,
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
    print(f"- Diagnostics: {csv_bundle.diagnostics_csv}")
    print(f"- Valid for Layer1: {csv_bundle.valid_for_layer1}")


if __name__ == "__main__":
    main()
