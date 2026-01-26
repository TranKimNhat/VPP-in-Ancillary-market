from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandapower as pp


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
) -> dict[str, dict[str, float]]:
    zone_map_int = _as_int_key_map(zone_map)
    lambda_dlmp_int = _as_int_key_map(lambda_dlmp)
    reserve_duals_int = _as_int_key_map(reserve_duals) if reserve_duals is not None else {}

    loads_by_bus = _collect_loads_by_bus(net)
    zones: dict[int, list[int]] = {}
    for bus, zone in zone_map_int.items():
        zones.setdefault(int(zone), []).append(int(bus))

    market_signals = _reserve_market_signals(zones, reserve_duals_int, base_reserve_price)

    for zone, buses in zones.items():
        energy_price = _weighted_average(lambda_dlmp_int, loads_by_bus, buses)
        market_signals.setdefault(str(zone), {})["energy_price"] = energy_price

    return market_signals


if __name__ == "__main__":
    from ieee123bus import build_ieee123_net

    net = build_ieee123_net()
    bus_indices = list(net.bus.index)

    lambda_dlmp = {int(bus): 10.0 + (idx % 3) for idx, bus in enumerate(bus_indices)}
    zone_map = {int(bus): 1 + (idx % 3) for idx, bus in enumerate(bus_indices)}
    reserve_duals = {int(bus): 0.1 + 0.05 * (idx % 3) for idx, bus in enumerate(bus_indices)}

    signals = generate_market_signals(
        net,
        lambda_dlmp=lambda_dlmp,
        zone_map=zone_map,
        reserve_duals=reserve_duals,
        base_reserve_price=5.0,
    )
    print(signals)
