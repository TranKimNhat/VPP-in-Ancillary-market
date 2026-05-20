from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import pandapower as pp


PRICING_METHODS = {"load_weighted", "max_dlmp", "congestion_weighted"}


@dataclass(frozen=True)
class DLMPComponents:
    lambda_p_total: float   # total active DLMP at bus (€/MWh)
    lambda_p_energy: float  # reference bus (slack) active price
    lambda_p_loss: float    # active DLMP loss component
    lambda_p_congestion: float  # active DLMP congestion component
    lambda_p_voltage: float  # active DLMP voltage component


def compute_dlmp(
    net: pp.pandapowerNet,
    lambda_dlmp: Mapping[int, float],
) -> dict[int, DLMPComponents]:
    """Compute per-bus active DLMP from SOCP reconfiguration duals (AM-only build).

    Active DLMP: taken directly from lambda_dlmp (SOCP dual variables).
    lambda_p_energy = slack-bus active price (reference node LMP).
    Remaining active spread is decomposed into loss, congestion, voltage using
    OPF-derived local congestion/voltage stress weights when available.
    Reactive (Q) DLMP has been dropped — system is AM-focused (energy + FFR only).
    """
    lambda_p = _as_int_key_map(lambda_dlmp)

    ref_price = 0.0
    if not net.ext_grid.empty:
        slack_bus = int(net.ext_grid.iloc[0]["bus"])
        ref_price = float(lambda_p.get(slack_bus, 0.0))
    elif not net.gen.empty and "slack" in net.gen.columns:
        slack_gens = net.gen[net.gen["slack"].astype(bool)]
        if not slack_gens.empty:
            try:
                ref_price = float(lambda_p.get(int(slack_gens.iloc[0]["bus"]), 0.0))
            except (TypeError, ValueError):
                pass

    congestion_weights: dict[int, float] = {int(bus): 0.0 for bus in net.bus.index}
    voltage_weights: dict[int, float] = {int(bus): 0.0 for bus in net.bus.index}
    try:
        net_copy = copy.deepcopy(net)
        if net_copy.poly_cost.empty:
            for gen_idx in net_copy.gen.index:
                pp.create_poly_cost(net_copy, gen_idx, "gen", cp1_eur_per_mw=1.0)
            for eg_idx in net_copy.ext_grid.index:
                pp.create_poly_cost(net_copy, eg_idx, "ext_grid", cp1_eur_per_mw=1.0)
        pp.runopp(net_copy, numba=False)
        congestion_weights = _collect_congestion_weights(net_copy)
        if hasattr(net_copy, "res_bus") and "vm_pu" in net_copy.res_bus.columns:
            for bus_id in net_copy.bus.index:
                if bus_id in net_copy.res_bus.index:
                    vm = float(net_copy.res_bus.at[bus_id, "vm_pu"])
                    voltage_weights[int(bus_id)] = abs(vm - 1.0)
    except Exception:
        pass

    result: dict[int, DLMPComponents] = {}
    for bus_id in net.bus.index:
        bus_int = int(bus_id)
        lp = float(lambda_p.get(bus_int, 0.0))
        residual = lp - ref_price

        congestion_w = max(float(congestion_weights.get(bus_int, 0.0)), 0.0)
        voltage_w = max(float(voltage_weights.get(bus_int, 0.0)), 0.0)
        denom = congestion_w + voltage_w
        if denom > 0.0:
            lambda_p_congestion = residual * (congestion_w / denom)
            lambda_p_voltage = residual * (voltage_w / denom)
        else:
            lambda_p_congestion = 0.0
            lambda_p_voltage = 0.0
        lambda_p_loss = residual - lambda_p_congestion - lambda_p_voltage

        result[bus_int] = DLMPComponents(
            lambda_p_total=lp,
            lambda_p_energy=ref_price,
            lambda_p_loss=lambda_p_loss,
            lambda_p_congestion=lambda_p_congestion,
            lambda_p_voltage=lambda_p_voltage,
        )
    return result


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
