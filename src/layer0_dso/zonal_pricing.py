from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandapower as pp


PRICING_METHODS = {"load_weighted", "max_dlmp", "congestion_weighted"}


def _as_int_key_map(mapping: Mapping[Any, Any]) -> dict[int, Any]:
    converted: dict[int, Any] = {}
    for key, value in mapping.items():
        converted[int(key)] = value
    return converted


def _safe_mean(values: list[float]) -> float:
    if not values:
        return float("nan")
    return float(np.mean(values))


def _collect_loads_by_bus(net: pp.pandapowerNet) -> dict[int, float]:
    loads: dict[int, float] = {int(bus): 0.0 for bus in net.bus.index}
    if net.load.empty:
        return loads
    for _, row in net.load.iterrows():
        bus = int(row["bus"])
        loads[bus] = loads.get(bus, 0.0) + float(row.get("p_mw", 0.0))
    return loads


def _collect_congestion_weights(net: pp.pandapowerNet) -> dict[int, float]:
    weights: dict[int, float] = {int(bus): 0.0 for bus in net.bus.index}
    if not hasattr(net, "res_line") or net.res_line.empty or net.line.empty:
        return weights

    for line_idx, row in net.line.iterrows():
        if line_idx not in net.res_line.index:
            continue
        max_i = float(row.get("max_i_ka", 0.0) or 0.0)
        if max_i <= 0:
            continue
        line_loading = abs(float(net.res_line.at[line_idx, "i_ka"])) / max_i
        from_bus = int(row["from_bus"])
        to_bus = int(row["to_bus"])
        weights[from_bus] = max(weights.get(from_bus, 0.0), line_loading)
        weights[to_bus] = max(weights.get(to_bus, 0.0), line_loading)

    return weights


def _weighted_average(prices: dict[int, float], weights: dict[int, float], buses: list[int]) -> float:
    num = 0.0
    den = 0.0
    for bus in buses:
        weight = float(weights.get(bus, 0.0))
        if weight <= 0:
            continue
        if bus not in prices:
            continue
        num += weight * float(prices[bus])
        den += weight
    if den == 0.0:
        return float("nan")
    return num / den


def _max_dlmp(prices: dict[int, float], buses: list[int]) -> float:
    candidates = [float(prices[bus]) for bus in buses if bus in prices and not np.isnan(float(prices[bus]))]
    if not candidates:
        return float("nan")
    return float(max(candidates))


def _zone_energy_price(
    *,
    net: pp.pandapowerNet,
    lambda_dlmp_int: dict[int, float],
    buses: list[int],
    method: str,
    loads_by_bus: dict[int, float],
) -> float:
    if method == "load_weighted":
        return _weighted_average(lambda_dlmp_int, loads_by_bus, buses)
    if method == "max_dlmp":
        return _max_dlmp(lambda_dlmp_int, buses)
    if method == "congestion_weighted":
        congestion_weights = _collect_congestion_weights(net)
        merged_weights: dict[int, float] = {}
        for bus in buses:
            load_weight = float(loads_by_bus.get(bus, 0.0))
            congestion = float(congestion_weights.get(bus, 0.0))
            merged_weights[bus] = load_weight * (1.0 + congestion)
        return _weighted_average(lambda_dlmp_int, merged_weights, buses)
    raise ValueError(f"Unsupported pricing method '{method}'.")


def _reserve_market_signals(
    zones: dict[int, list[int]],
    reserve_duals: Mapping[int, float],
    base_reserve_price: float | None,
) -> dict[str, dict[str, float]]:
    market_signals: dict[str, dict[str, float]] = {}
    for zone, buses in zones.items():
        premiums = [float(reserve_duals.get(bus, 0.0)) for bus in buses]
        premium = _safe_mean(premiums)
        reserve_price = premium if base_reserve_price is None else float(base_reserve_price) + premium
        market_signals[str(zone)] = {"reserve_price": reserve_price}
    return market_signals


def generate_market_signals(
    net: pp.pandapowerNet,
    lambda_dlmp: Mapping[int, float],
    zone_map: Mapping[int, int],
    reserve_duals: Mapping[int, float] | None = None,
    base_reserve_price: float | None = None,
    pricing_method: str = "load_weighted",
) -> dict[str, dict[str, float]]:
    if pricing_method not in PRICING_METHODS:
        raise ValueError(f"pricing_method must be one of {sorted(PRICING_METHODS)}")

    zone_map_int = _as_int_key_map(zone_map)
    lambda_dlmp_int = _as_int_key_map(lambda_dlmp)
    reserve_duals_int = _as_int_key_map(reserve_duals) if reserve_duals is not None else {}

    loads_by_bus = _collect_loads_by_bus(net)
    zones: dict[int, list[int]] = {}
    for bus, zone in zone_map_int.items():
        zones.setdefault(int(zone), []).append(int(bus))

    market_signals = _reserve_market_signals(zones, reserve_duals_int, base_reserve_price)

    for zone, buses in zones.items():
        energy_price = _zone_energy_price(
            net=net,
            lambda_dlmp_int=lambda_dlmp_int,
            buses=buses,
            method=pricing_method,
            loads_by_bus=loads_by_bus,
        )
        zone_signal = market_signals.setdefault(str(zone), {})
        zone_signal["energy_price"] = energy_price
        zone_signal["pricing_method"] = pricing_method

    return market_signals
