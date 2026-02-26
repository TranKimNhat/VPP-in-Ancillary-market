from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable
import copy

REACTIVE_CONTROL_TYPES = {"wind", "bess", "storage"}
REACTIVE_PF = 0.95

import numpy as np
import pandapower as pp
import pyomo.environ as pyo
import networkx as nx

from src.layer0_dso.dlmp_calculator import (
    extract_dlmp as _extract_dlmp,
    fix_and_solve_socp as _fix_and_solve_socp,
    run_dlmp_calculation,
    solve_misocp as _solve_misocp,
)


@dataclass(frozen=True)
class EdgeData:
    edge_id: int
    from_bus: int
    to_bus: int
    r_ohm: float
    x_ohm: float
    r_pu: float
    x_pu: float
    tap_ratio: float
    max_i_ka: float
    rating_mva: float
    is_switch: bool
    switch_index: int | None
    element_type: str


@dataclass(frozen=True)
class NetworkData:
    buses: list[int]
    slack_buses: list[int]
    edges: list[EdgeData]
    loads_p: dict[int, float]
    loads_q: dict[int, float]
    reactive_caps: dict[int, float]
    base_kv: float
    base_mva: float
    bus_nominal_kv: dict[int, float]
    switch_closed: dict[int, bool]


@dataclass(frozen=True)
class ModelWeights:
    loss: float = 1.0
    switch: float = 1.0
    voltage: float = 1.0


@dataclass(frozen=True)
class SolverOptions:
    mipgap: float | None = None
    timelimit: float | None = None


def _slack_buses(net: pp.pandapowerNet) -> list[int]:
    if net.ext_grid.empty:
        raise ValueError("No ext_grid found in network.")
    return [int(bus) for bus in net.ext_grid.bus]


def _collect_loads(
    net: pp.pandapowerNet,
) -> tuple[dict[int, float], dict[int, float], dict[int, float]]:
    def _safe_float(value: object) -> float:
        try:
            result = float(value)
        except (TypeError, ValueError):
            return 0.0
        if np.isnan(result):
            return 0.0
        return result

    loads_p = {int(bus): 0.0 for bus in net.bus.index}
    loads_q = {int(bus): 0.0 for bus in net.bus.index}
    reactive_caps = {int(bus): 0.0 for bus in net.bus.index}
    if not net.load.empty:
        for _, row in net.load.iterrows():
            bus = int(row["bus"])
            loads_p[bus] = loads_p.get(bus, 0.0) + _safe_float(row.get("p_mw", 0.0))
            loads_q[bus] = loads_q.get(bus, 0.0) + _safe_float(row.get("q_mvar", 0.0))
    if not net.sgen.empty:
        for _, row in net.sgen.iterrows():
            in_service = row.get("in_service")
            if in_service is False:
                continue
            bus = int(row["bus"])
            p_mw = _safe_float(row.get("p_mw", 0.0))
            q_mvar = _safe_float(row.get("q_mvar", 0.0))
            loads_p[bus] = loads_p.get(bus, 0.0) - p_mw
            loads_q[bus] = loads_q.get(bus, 0.0) - q_mvar
            sgen_type = str(row.get("type", "")).lower()
            if sgen_type in REACTIVE_CONTROL_TYPES and abs(p_mw) > 1e-9:
                max_q = abs(p_mw) * (max(1.0 / (REACTIVE_PF**2) - 1.0, 0.0) ** 0.5)
                reactive_caps[bus] = reactive_caps.get(bus, 0.0) + max_q
    if not net.shunt.empty:
        for _, row in net.shunt.iterrows():
            in_service = row.get("in_service")
            if in_service is False:
                continue
            bus = int(row["bus"])
            q_mvar = _safe_float(row.get("q_mvar", 0.0))
            p_mw = _safe_float(row.get("p_mw", 0.0))
            loads_p[bus] = loads_p.get(bus, 0.0) + p_mw
            loads_q[bus] = loads_q.get(bus, 0.0) + q_mvar
    return loads_p, loads_q, reactive_caps


def _edge_resistance(line: pp.Series) -> float:
    return float(line["r_ohm_per_km"] * line["length_km"])


def _edge_reactance(line: pp.Series) -> float:
    return float(line["x_ohm_per_km"] * line["length_km"])


def _edge_max_i_ka(line: pp.Series) -> float:
    value = line.get("max_i_ka", 0.0)
    if value is None:
        return 0.0
    return float(value)


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    if not np.isfinite(result):
        return default
    return result


def _trafo_tap_ratio(trafo: pp.Series, vn_bus_hv: float, vn_bus_lv: float) -> float:
    vn_hv_kv = _safe_float(trafo.get("vn_hv_kv", 0.0), default=0.0)
    vn_lv_kv = _safe_float(trafo.get("vn_lv_kv", 0.0), default=0.0)
    if vn_hv_kv <= 0.0 or vn_lv_kv <= 0.0 or vn_bus_hv <= 0.0 or vn_bus_lv <= 0.0:
        return 1.0

    tap_step_percent = _safe_float(trafo.get("tap_step_percent", 0.0), default=0.0)
    tap_pos = _safe_float(trafo.get("tap_pos", 0.0), default=0.0)
    tap_neutral = _safe_float(trafo.get("tap_neutral", 0.0), default=0.0)
    tap_side = str(trafo.get("tap_side", "")).lower()
    if tap_side not in {"hv", "lv"}:
        tap_side = "hv"

    tap_factor = 1.0 + ((tap_pos - tap_neutral) * tap_step_percent / 100.0)
    if tap_factor <= 0.0:
        tap_factor = 1.0

    effective_vn_hv = vn_hv_kv
    effective_vn_lv = vn_lv_kv
    if tap_side == "hv":
        effective_vn_hv = vn_hv_kv * tap_factor
    else:
        effective_vn_lv = vn_lv_kv * tap_factor

    if effective_vn_hv <= 0.0 or effective_vn_lv <= 0.0:
        return 1.0

    nominal_ratio = effective_vn_hv / effective_vn_lv
    bus_ratio = vn_bus_hv / vn_bus_lv
    tap_ratio = nominal_ratio / bus_ratio
    if not np.isfinite(tap_ratio) or tap_ratio <= 0.0:
        return 1.0
    return float(tap_ratio)


def extract_network_data(net: pp.pandapowerNet) -> NetworkData:
    buses = [int(bus) for bus in net.bus.index]
    slack_buses = _slack_buses(net)
    base_bus = int(slack_buses[0]) if slack_buses else None
    if base_bus is not None and not net.bus.empty and base_bus in net.bus.index:
        base_kv = float(net.bus.loc[base_bus, "vn_kv"])
    else:
        base_kv = float(net.bus["vn_kv"].mean()) if not net.bus.empty else 1.0

    base_mva = 1.0
    loads_p, loads_q, reactive_caps = _collect_loads(net)

    edges: list[EdgeData] = []
    edge_id = 0
    for _, line in net.line.iterrows():
        from_bus = int(line["from_bus"])
        to_bus = int(line["to_bus"])
        r_ohm = _edge_resistance(line)
        x_ohm = _edge_reactance(line)
        vn_from = float(net.bus.at[from_bus, "vn_kv"])
        z_base_from = (vn_from**2) / base_mva if vn_from > 0 else 1.0
        r_pu = r_ohm / z_base_from
        x_pu = x_ohm / z_base_from
        max_i_ka = _edge_max_i_ka(line)
        rating_mva = 0.0
        if max_i_ka > 0.0 and vn_from > 0.0:
            rating_mva = np.sqrt(3.0) * vn_from * max_i_ka
        edges.append(
            EdgeData(
                edge_id=edge_id,
                from_bus=from_bus,
                to_bus=to_bus,
                r_ohm=r_ohm,
                x_ohm=x_ohm,
                r_pu=r_pu,
                x_pu=x_pu,
                tap_ratio=1.0,
                max_i_ka=max_i_ka,
                rating_mva=rating_mva,
                is_switch=False,
                switch_index=None,
                element_type="line",
            )
        )
        edge_id += 1

    if not net.trafo.empty:
        for _, trafo in net.trafo.iterrows():
            hv_bus = int(trafo["hv_bus"])
            lv_bus = int(trafo["lv_bus"])
            sn_mva = _safe_float(trafo.get("sn_mva", 0.0), default=0.0)
            vn_hv_kv = _safe_float(trafo.get("vn_hv_kv", 0.0), default=0.0)
            vk_percent = _safe_float(trafo.get("vk_percent", 0.0), default=0.0)
            vkr_percent = _safe_float(trafo.get("vkr_percent", 0.0), default=0.0)

            z_pu_nameplate = max(vk_percent / 100.0, 0.0)
            r_pu_nameplate = max(vkr_percent / 100.0, 0.0)
            x_sq = max(z_pu_nameplate**2 - r_pu_nameplate**2, 0.0)
            x_pu_nameplate = x_sq**0.5
            if sn_mva > 0.0:
                scale = base_mva / sn_mva
                r_pu = r_pu_nameplate * scale
                x_pu = x_pu_nameplate * scale
            else:
                r_pu = 0.0
                x_pu = 0.0

            z_base_hv = (vn_hv_kv**2) / base_mva if vn_hv_kv > 0 else 1.0
            r_ohm = r_pu * z_base_hv
            x_ohm = x_pu * z_base_hv

            vn_bus_hv = _safe_float(net.bus.at[hv_bus, "vn_kv"], default=0.0)
            vn_bus_lv = _safe_float(net.bus.at[lv_bus, "vn_kv"], default=0.0)
            tap_ratio = _trafo_tap_ratio(trafo, vn_bus_hv, vn_bus_lv)

            max_i_ka = 0.0
            if sn_mva > 0.0 and vn_bus_hv > 0.0:
                max_i_ka = sn_mva / (np.sqrt(3.0) * vn_bus_hv)

            edges.append(
                EdgeData(
                    edge_id=edge_id,
                    from_bus=hv_bus,
                    to_bus=lv_bus,
                    r_ohm=r_ohm,
                    x_ohm=x_ohm,
                    r_pu=r_pu,
                    x_pu=x_pu,
                    tap_ratio=tap_ratio,
                    max_i_ka=max_i_ka,
                    rating_mva=sn_mva,
                    is_switch=False,
                    switch_index=None,
                    element_type="trafo",
                )
            )
            edge_id += 1

    switch_closed: dict[int, bool] = {}
    if not net.switch.empty:
        candidate = net.switch[(net.switch["et"].astype(str) == "b") & (net.switch["type"].astype(str) == "CB")]
        for sw_idx, sw in candidate.iterrows():
            switch_closed[int(sw_idx)] = bool(sw.get("closed", True))
            edges.append(
                EdgeData(
                    edge_id=edge_id,
                    from_bus=int(sw["bus"]),
                    to_bus=int(sw["element"]),
                    r_ohm=0.0,
                    x_ohm=0.0,
                    r_pu=0.0,
                    x_pu=0.0,
                    tap_ratio=1.0,
                    max_i_ka=0.0,
                    rating_mva=0.0,
                    is_switch=True,
                    switch_index=int(sw_idx),
                    element_type="switch",
                )
            )
            edge_id += 1

    bus_nominal_kv = {
        int(bus_idx): float(net.bus.at[bus_idx, "vn_kv"])
        for bus_idx in net.bus.index
    }

    return NetworkData(
        buses=buses,
        slack_buses=slack_buses,
        edges=edges,
        loads_p=loads_p,
        loads_q=loads_q,
        reactive_caps=reactive_caps,
        base_kv=base_kv,
        base_mva=base_mva,
        bus_nominal_kv=bus_nominal_kv,
        switch_closed=switch_closed,
    )


def switch_edge_map(net: pp.pandapowerNet) -> dict[int, dict[str, int]]:
    data = extract_network_data(net)
    return {
        edge.edge_id: {"from_bus": edge.from_bus, "to_bus": edge.to_bus}
        for edge in data.edges
        if edge.is_switch
    }


def _apply_switch_solution(net: pp.pandapowerNet, data: NetworkData, alpha_star: dict[int, int]) -> pp.pandapowerNet:
    switched = copy.deepcopy(net)
    for edge in data.edges:
        if not edge.is_switch or edge.switch_index is None:
            continue
        if edge.edge_id not in alpha_star:
            continue
        switched.switch.at[edge.switch_index, "closed"] = bool(alpha_star[edge.edge_id])
    return switched


def build_misocp_model(
    data: NetworkData,
    weights: ModelWeights,
    switch_cost: float,
    voltage_bounds: tuple[float, float] | None = None,
    soc_relax: float = 1.001,
    soc_slack_weight: float = 1000.0,
    voltage_drop_slack_weight: float = 10000.0,
    trafo_voltage_drop_slack_weight: float = 200000.0,
    soc_slack_cap: float = 0.0,
    voltage_drop_slack_cap: float = 0.0,
    soc_reference_v2: dict[int, float] | None = None,
    voltage_reference_upper_band: float = 0.01,
    fix_switch_status: bool = False,
    fixed_switch_indices: set[int] | None = None,
    switch_initial: dict[int, int] | None = None,
    enforce_radiality: bool = True,
    radiality_slack: int = 0,
) -> tuple[pyo.ConcreteModel, dict[int, pyo.Constraint]]:
    model = pyo.ConcreteModel("ieee123_reconfiguration")

    bus_set = sorted(set(data.buses))
    edge_set = [edge.edge_id for edge in data.edges]
    switch_edges = [edge.edge_id for edge in data.edges if edge.is_switch]

    model.BUS = pyo.Set(initialize=bus_set)
    model.EDGE = pyo.Set(initialize=edge_set)
    model.SWITCH = pyo.Set(initialize=switch_edges)

    model.P = pyo.Var(model.EDGE, domain=pyo.Reals)
    model.Q = pyo.Var(model.EDGE, domain=pyo.Reals)
    model.V2 = pyo.Var(model.BUS, domain=pyo.NonNegativeReals)
    model.I2 = pyo.Var(model.EDGE, domain=pyo.NonNegativeReals)
    model.Qcap = pyo.Var(model.BUS, domain=pyo.Reals)
    model.alpha = pyo.Var(model.SWITCH, domain=pyo.Binary)
    model.SwitchChange = pyo.Var(model.SWITCH, domain=pyo.NonNegativeReals)
    model.F = pyo.Var(model.EDGE, domain=pyo.Reals)
    model.Pslack = pyo.Var(data.slack_buses, domain=pyo.Reals)
    model.Qslack = pyo.Var(data.slack_buses, domain=pyo.Reals)
    model.SOCSlack = pyo.Var(model.EDGE, domain=pyo.NonNegativeReals)
    model.VoltageDropSlack = pyo.Var(model.EDGE, domain=pyo.NonNegativeReals)

    base_mva = max(float(data.base_mva), 1e-9)

    def _bus_vbase_kv(bus: int) -> float:
        return max(float(data.bus_nominal_kv.get(int(bus), data.base_kv)), 1e-9)

    edge_map = {edge.edge_id: edge for edge in data.edges}
    model._bus_nominal_kv = {int(bus): float(v) for bus, v in data.bus_nominal_kv.items()}
    model._voltage_in_pu = True

    bus_voltage_bounds: dict[int, tuple[float, float]] = {}
    if voltage_bounds is not None:
        voltage_min, voltage_max = voltage_bounds
        v_base_global = max(float(data.base_kv), 1e-9)
        vm_min_pu = (max(voltage_min, 0.0) ** 0.5) / v_base_global
        vm_max_pu = (max(voltage_max, 0.0) ** 0.5) / v_base_global

        for bus in bus_set:
            bus_voltage_bounds[int(bus)] = (vm_min_pu**2, vm_max_pu**2)

        def _voltage_bounds_rule(m: pyo.ConcreteModel, bus: int) -> pyo.Constraint:
            v_min_bus, v_max_bus = bus_voltage_bounds[int(bus)]
            return pyo.inequality(v_min_bus, m.V2[bus], v_max_bus)

        model.VoltageBounds = pyo.Constraint(model.BUS, rule=_voltage_bounds_rule)

    if voltage_bounds is not None:
        def _slack_voltage_rule(m: pyo.ConcreteModel, bus: int) -> pyo.Constraint:
            if bus not in bus_set:
                return pyo.Constraint.Skip
            return m.V2[bus] == 1.0

        model.SlackVoltage = pyo.Constraint(data.slack_buses, rule=_slack_voltage_rule)

    balance_constraints: dict[int, pyo.Constraint] = {}

    incident_in = {bus: [edge.edge_id for edge in data.edges if edge.to_bus == bus] for bus in bus_set}
    incident_out = {bus: [edge.edge_id for edge in data.edges if edge.from_bus == bus] for bus in bus_set}
    incident_in_line = {bus: [edge.edge_id for edge in data.edges if not edge.is_switch and edge.to_bus == bus] for bus in bus_set}
    incident_out_line = {bus: [edge.edge_id for edge in data.edges if not edge.is_switch and edge.from_bus == bus] for bus in bus_set}
    slack_set = set(data.slack_buses)

    def _balance_rule(m: pyo.ConcreteModel, bus: int, reactive: bool) -> pyo.Constraint:
        var = m.Q if reactive else m.P
        incoming = incident_in.get(bus, [])
        outgoing = incident_out.get(bus, [])
        load_mw = data.loads_p.get(bus, 0.0)
        load_q = data.loads_q.get(bus, 0.0)
        load = (load_q if reactive else load_mw) / base_mva
        if not incoming and not outgoing:
            if abs(load) <= 1e-9:
                return pyo.Constraint.Feasible
            if bus in slack_set:
                slack_var = m.Qslack[bus] if reactive else m.Pslack[bus]
                return slack_var == load
            return pyo.Constraint.Infeasible
        inflow = pyo.quicksum(var[e] for e in incoming)
        outflow = pyo.quicksum(var[e] for e in outgoing)
        if reactive:
            loss_term = pyo.quicksum(edge_map[e].x_pu * m.I2[e] for e in incoming)
            reactive_support = m.Qcap[bus]
        else:
            loss_term = pyo.quicksum(edge_map[e].r_pu * m.I2[e] for e in incoming)
            reactive_support = 0.0
        if bus in slack_set:
            slack_var = m.Qslack[bus] if reactive else m.Pslack[bus]
            return inflow - outflow + slack_var == load + loss_term - reactive_support
        return inflow - outflow == load + loss_term - reactive_support

    model.PBalance = pyo.Constraint(model.BUS, rule=lambda m, b: _balance_rule(m, b, reactive=False))
    model.QBalance = pyo.Constraint(model.BUS, rule=lambda m, b: _balance_rule(m, b, reactive=True))

    for bus in data.buses:
        if bus not in slack_set:
            balance_constraints[bus] = model.PBalance[bus]

    active_buses = [
        bus
        for bus in bus_set
        if bus in slack_set
        or incident_in_line.get(bus, [])
        or incident_out_line.get(bus, [])
        or abs(data.loads_p.get(bus, 0.0) / base_mva) > 1e-9
        or abs(data.loads_q.get(bus, 0.0) / base_mva) > 1e-9
    ]
    active_bus_set = set(active_buses)
    flow_limit = float(len(active_buses))
    root_balance = len(active_buses) - len(slack_set)

    if enforce_radiality:
        fixed_edges_count = sum(1 for edge in data.edges if not edge.is_switch)
        target_edges = len(active_buses) - len(slack_set) + radiality_slack
        model.Radiality = pyo.Constraint(
            expr=pyo.quicksum(model.alpha[e] for e in model.SWITCH) + fixed_edges_count == target_edges
        )

    def _connectivity_rule(m: pyo.ConcreteModel, bus: int) -> pyo.Constraint:
        if bus not in active_bus_set:
            return pyo.Constraint.Skip
        incoming = incident_in.get(bus, [])
        outgoing = incident_out.get(bus, [])
        if not incoming and not outgoing:
            if abs(data.loads_p.get(bus, 0.0)) <= 1e-9 and abs(data.loads_q.get(bus, 0.0)) <= 1e-9:
                return pyo.Constraint.Skip
            return pyo.Constraint.Infeasible
        if bus in slack_set:
            return pyo.Constraint.Skip
        return pyo.quicksum(m.F[e] for e in incoming) - pyo.quicksum(m.F[e] for e in outgoing) == -1

    model.Connectivity = pyo.Constraint(model.BUS, rule=_connectivity_rule)

    def _connectivity_root_rule(m: pyo.ConcreteModel) -> pyo.Constraint:
        slack_balance = []
        for bus in slack_set:
            incoming = incident_in.get(bus, [])
            outgoing = incident_out.get(bus, [])
            slack_balance.append(pyo.quicksum(m.F[e] for e in incoming) - pyo.quicksum(m.F[e] for e in outgoing))
        return pyo.quicksum(slack_balance) == root_balance

    model.ConnectivityRoot = pyo.Constraint(rule=_connectivity_root_rule)

    if voltage_bounds is not None:
        v_base_global = max(float(data.base_kv), 1e-9)
        vm_min_pu = (max(voltage_bounds[0], 0.0) ** 0.5) / v_base_global
        vm_max_pu = (max(voltage_bounds[1], 0.0) ** 0.5) / v_base_global
        big_m = float(max(vm_max_pu**2 - vm_min_pu**2, 1e-3))
    else:
        big_m = 4.0
    non_switch_edges = [edge for edge in data.edges if not edge.is_switch]
    estimated_flow_bound = sum(abs(data.loads_p.get(int(bus), 0.0)) for bus in data.buses) / base_mva
    estimated_q_bound = sum(abs(data.loads_q.get(int(bus), 0.0)) for bus in data.buses) / base_mva
    rated_s_bound = max((edge.rating_mva for edge in non_switch_edges if edge.rating_mva > 0.0), default=base_mva) / base_mva
    if non_switch_edges:
        estimated_flow_bound += 0.5 * sum(abs(edge.r_pu) for edge in non_switch_edges)
        estimated_q_bound += 0.5 * sum(abs(edge.x_pu) for edge in non_switch_edges)
    big_m_p = max(1e-3, estimated_flow_bound, rated_s_bound)
    big_m_q = max(1e-3, estimated_q_bound, rated_s_bound)
    big_m_i = max(1e-6, (big_m_p**2 + big_m_q**2))

    def _voltage_drop_upper_rule(m: pyo.ConcreteModel, edge_id: int) -> pyo.Constraint:
        edge = edge_map[edge_id]
        from_bus = edge.from_bus
        to_bus = edge.to_bus
        r = edge.r_pu
        x = edge.x_pu
        tap_sq = edge.tap_ratio**2 if edge.tap_ratio > 0 else 1.0
        drop = -2 * (r * m.P[edge_id] + x * m.Q[edge_id]) + (r**2 + x**2) * m.I2[edge_id]
        residual = m.V2[to_bus] - tap_sq * m.V2[from_bus] - drop
        if edge.is_switch:
            return residual <= big_m * (1 - m.alpha[edge_id]) + m.VoltageDropSlack[edge_id]
        return residual <= m.VoltageDropSlack[edge_id]

    def _voltage_drop_lower_rule(m: pyo.ConcreteModel, edge_id: int) -> pyo.Constraint:
        edge = edge_map[edge_id]
        from_bus = edge.from_bus
        to_bus = edge.to_bus
        r = edge.r_pu
        x = edge.x_pu
        tap_sq = edge.tap_ratio**2 if edge.tap_ratio > 0 else 1.0
        drop = -2 * (r * m.P[edge_id] + x * m.Q[edge_id]) + (r**2 + x**2) * m.I2[edge_id]
        residual = m.V2[to_bus] - tap_sq * m.V2[from_bus] - drop
        if edge.is_switch:
            return -residual <= big_m * (1 - m.alpha[edge_id]) + m.VoltageDropSlack[edge_id]
        return -residual <= m.VoltageDropSlack[edge_id]

    model.VoltageDropUpper = pyo.Constraint(model.EDGE, rule=_voltage_drop_upper_rule)
    model.VoltageDropLower = pyo.Constraint(model.EDGE, rule=_voltage_drop_lower_rule)

    def _soc_slack_cap_rule(m: pyo.ConcreteModel, edge_id: int) -> pyo.Constraint:
        if soc_slack_cap <= 0.0:
            return m.SOCSlack[edge_id] == 0.0
        return m.SOCSlack[edge_id] <= soc_slack_cap

    def _voltage_drop_slack_cap_rule(m: pyo.ConcreteModel, edge_id: int) -> pyo.Constraint:
        if voltage_drop_slack_cap <= 0.0:
            return m.VoltageDropSlack[edge_id] == 0.0
        return m.VoltageDropSlack[edge_id] <= voltage_drop_slack_cap

    model.SOCSlackCap = pyo.Constraint(model.EDGE, rule=_soc_slack_cap_rule)
    model.VoltageDropSlackCap = pyo.Constraint(model.EDGE, rule=_voltage_drop_slack_cap_rule)

    if fix_switch_status:
        fixed_switches = fixed_switch_indices or set()

        def _switch_status_rule(m: pyo.ConcreteModel, edge_id: int) -> pyo.Constraint:
            edge = edge_map[edge_id]
            if edge.switch_index is None or edge.switch_index not in fixed_switches:
                return pyo.Constraint.Skip
            closed = data.switch_closed.get(edge.switch_index, True)
            if closed:
                return m.alpha[edge_id] == 1
            return m.alpha[edge_id] == 0

        model.SwitchStatus = pyo.Constraint(model.SWITCH, rule=_switch_status_rule)

    initial_alpha = switch_initial or {}

    def _switch_change_pos_rule(m: pyo.ConcreteModel, edge_id: int) -> pyo.Constraint:
        return m.SwitchChange[edge_id] >= m.alpha[edge_id] - initial_alpha.get(int(edge_id), 0)

    def _switch_change_neg_rule(m: pyo.ConcreteModel, edge_id: int) -> pyo.Constraint:
        return m.SwitchChange[edge_id] >= initial_alpha.get(int(edge_id), 0) - m.alpha[edge_id]

    model.SwitchChangePos = pyo.Constraint(model.SWITCH, rule=_switch_change_pos_rule)
    model.SwitchChangeNeg = pyo.Constraint(model.SWITCH, rule=_switch_change_neg_rule)

    default_v2_max = max((bounds[1] for bounds in bus_voltage_bounds.values()), default=1.5**2)
    default_soc_reference_v2 = {int(bus): bus_voltage_bounds.get(int(bus), (0.0, default_v2_max))[1] for bus in bus_set}
    calibration_bounds: dict[int, tuple[float, float]] = {}
    if soc_reference_v2:
        band = max(float(voltage_reference_upper_band), 0.0)
        band_upper = (1.0 + band) ** 2
        band_lower = (max(1.0 - band, 1e-3)) ** 2
        for bus, value in soc_reference_v2.items():
            bus_i = int(bus)
            if bus_i not in default_soc_reference_v2:
                continue
            try:
                v2_ref = float(value)
            except (TypeError, ValueError):
                continue
            if not np.isfinite(v2_ref) or v2_ref <= 0.0:
                continue
            default_soc_reference_v2[bus_i] = min(default_soc_reference_v2[bus_i], v2_ref)
            base_lower, base_upper = bus_voltage_bounds.get(bus_i, (0.0, default_v2_max))
            cal_lower = max(base_lower, v2_ref * band_lower)
            cal_upper = min(base_upper, v2_ref * band_upper)
            if cal_upper >= cal_lower:
                calibration_bounds[bus_i] = (cal_lower, cal_upper)

    model._soc_reference_v2 = copy.deepcopy(default_soc_reference_v2)

    if calibration_bounds:
        def _reference_voltage_band_rule(m: pyo.ConcreteModel, bus: int) -> pyo.Constraint:
            bounds = calibration_bounds.get(int(bus))
            if bounds is None:
                return pyo.Constraint.Skip
            return pyo.inequality(bounds[0], m.V2[bus], bounds[1])

        model.ReferenceVoltageBand = pyo.Constraint(model.BUS, rule=_reference_voltage_band_rule)

    def _soc_rule(m: pyo.ConcreteModel, edge_id: int) -> pyo.Constraint:
        edge = edge_map[edge_id]
        tap_sq = edge.tap_ratio**2 if edge.tap_ratio > 0 else 1.0
        v2_from_ref = default_soc_reference_v2.get(int(edge.from_bus), default_v2_max)
        v2_to_ref = default_soc_reference_v2.get(int(edge.to_bus), default_v2_max)
        current_side_v2_ref = min(tap_sq * v2_from_ref, v2_to_ref)
        current_side_v2_ref = max(float(current_side_v2_ref), 1e-6)
        return m.P[edge_id] ** 2 + m.Q[edge_id] ** 2 <= soc_relax * current_side_v2_ref * m.I2[edge_id] + m.SOCSlack[edge_id]

    model.SOC = pyo.Constraint(model.EDGE, rule=_soc_rule)

    def _reactive_cap_rule(m: pyo.ConcreteModel, bus: int) -> pyo.Constraint:
        cap = data.reactive_caps.get(bus, 0.0) / base_mva
        if cap <= 0:
            return m.Qcap[bus] == 0.0
        return pyo.inequality(-cap, m.Qcap[bus], cap)

    model.ReactiveCap = pyo.Constraint(model.BUS, rule=_reactive_cap_rule)

    def _switch_flow_p_ub_rule(m: pyo.ConcreteModel, edge_id: int) -> pyo.Constraint:
        return m.P[edge_id] <= big_m_p * m.alpha[edge_id]

    def _switch_flow_p_lb_rule(m: pyo.ConcreteModel, edge_id: int) -> pyo.Constraint:
        return m.P[edge_id] >= -big_m_p * m.alpha[edge_id]

    def _switch_flow_q_ub_rule(m: pyo.ConcreteModel, edge_id: int) -> pyo.Constraint:
        return m.Q[edge_id] <= big_m_q * m.alpha[edge_id]

    def _switch_flow_q_lb_rule(m: pyo.ConcreteModel, edge_id: int) -> pyo.Constraint:
        return m.Q[edge_id] >= -big_m_q * m.alpha[edge_id]

    def _switch_flow_i_rule(m: pyo.ConcreteModel, edge_id: int) -> pyo.Constraint:
        return m.I2[edge_id] <= big_m_i * m.alpha[edge_id]

    model.SwitchFlowPUpper = pyo.Constraint(model.SWITCH, rule=_switch_flow_p_ub_rule)
    model.SwitchFlowPLower = pyo.Constraint(model.SWITCH, rule=_switch_flow_p_lb_rule)
    model.SwitchFlowQUpper = pyo.Constraint(model.SWITCH, rule=_switch_flow_q_ub_rule)
    model.SwitchFlowQLower = pyo.Constraint(model.SWITCH, rule=_switch_flow_q_lb_rule)
    model.SwitchFlowI = pyo.Constraint(model.SWITCH, rule=_switch_flow_i_rule)

    def _connectivity_lb_rule(m: pyo.ConcreteModel, edge_id: int) -> pyo.Constraint:
        edge = edge_map[edge_id]
        if edge.is_switch:
            return m.F[edge_id] >= -flow_limit * m.alpha[edge_id]
        return m.F[edge_id] >= -flow_limit

    def _connectivity_ub_rule(m: pyo.ConcreteModel, edge_id: int) -> pyo.Constraint:
        edge = edge_map[edge_id]
        if edge.is_switch:
            return m.F[edge_id] <= flow_limit * m.alpha[edge_id]
        return m.F[edge_id] <= flow_limit

    model.ConnectivityLower = pyo.Constraint(model.EDGE, rule=_connectivity_lb_rule)
    model.ConnectivityUpper = pyo.Constraint(model.EDGE, rule=_connectivity_ub_rule)

    # No explicit radiality constraints here; connectivity flow constraints enforce a tree
    # when combined with a fixed number of energized edges (see earlier iterations).

    def _current_limit_rule(m: pyo.ConcreteModel, edge_id: int) -> pyo.Constraint:
        edge = edge_map[edge_id]
        if edge.rating_mva <= 0:
            return pyo.Constraint.Skip
        limit_pu = edge.rating_mva / base_mva
        return m.I2[edge_id] <= limit_pu**2

    model.CurrentLimit = pyo.Constraint(model.EDGE, rule=_current_limit_rule)

    def _min_current_rule(m: pyo.ConcreteModel, edge_id: int) -> pyo.Constraint:
        edge = edge_map[edge_id]
        if edge.max_i_ka <= 0:
            return pyo.Constraint.Skip
        return m.I2[edge_id] >= 0.0

    model.MinCurrent = pyo.Constraint(model.EDGE, rule=_min_current_rule)

    loss_term = sum(edge_map[e].r_pu * model.I2[e] for e in model.EDGE)
    switch_term = switch_cost * sum(model.SwitchChange[e] for e in model.SWITCH)
    voltage_term = sum((model.V2[b] - 1.0) ** 2 for b in model.BUS)
    soc_slack_term = soc_slack_weight * sum(model.SOCSlack[e] for e in model.EDGE)

    voltage_drop_slack_term = 0.0
    if hasattr(model, "VoltageDropSlack"):
        voltage_drop_slack_term = pyo.quicksum(
            (trafo_voltage_drop_slack_weight if edge_map[e].element_type == "trafo" else voltage_drop_slack_weight)
            * model.VoltageDropSlack[e]
            for e in model.EDGE
        )

    model.Objective = pyo.Objective(
        expr=weights.loss * loss_term
        + weights.switch * switch_term
        + weights.voltage * voltage_term
        + soc_slack_term
        + voltage_drop_slack_term,
        sense=pyo.minimize,
    )

    return model, balance_constraints


def solve_misocp(
    model: pyo.ConcreteModel,
    solver_opts: SolverOptions | None = None,
    debug: bool = False,
    debug_context: dict[str, object] | None = None,
) -> dict[int, int]:
    return _solve_misocp(model, solver_opts=solver_opts, debug=debug, debug_context=debug_context)


def fix_and_solve_socp(
    model: pyo.ConcreteModel,
    alpha_star: dict[int, int],
) -> pyo.ConcreteModel:
    return _fix_and_solve_socp(model, alpha_star)


def extract_dlmp(
    model: pyo.ConcreteModel,
    balance_constraints: dict[int, pyo.Constraint],
) -> dict[int, float]:
    return _extract_dlmp(model, balance_constraints)


@dataclass(frozen=True)
class ReconfigurationResult:
    lambda_dlmp: dict[int, float]
    alpha_star: dict[int, int]
    socp_voltage_squared: dict[int, float]
    soc_slack_max: float
    soc_slack_sum: float
    voltage_drop_slack_max: float
    voltage_drop_slack_sum: float


def run_reconfiguration_detailed(
    net: pp.pandapowerNet,
    weights: ModelWeights | None = None,
    switch_cost: float = 0.01,
    solver_opts: SolverOptions | None = None,
    debug: bool = False,
    force_switch_closed: bool = False,
    apply_voltage_bounds: bool = True,
    fix_switch_status: bool = False,
    penalize_switch_changes: bool = True,
    enforce_radiality: bool = True,
    radiality_slack: int = 0,
    soc_relax: float = 1.001,
    soc_slack_cap: float = 0.0,
    voltage_drop_slack_cap: float = 0.0,
    soc_reference_iterations: int = 4,
    voltage_reference_upper_band: float = 0.01,
) -> ReconfigurationResult:
    weights = weights or ModelWeights()

    if not net.switch.empty:
        candidate_mask = (net.switch["et"].astype(str) == "b") & (net.switch["type"].astype(str) == "CB")
        net.switch.loc[~candidate_mask, "closed"] = net.switch.loc[~candidate_mask, "closed"].fillna(True)

    fixed_pairs = {("250", "251"), ("95", "195"), ("300", "350"), ("450", "451"), ("61", "610")}
    fixed_indices: list[int] = []
    if not net.switch.empty:
        bus_names = net.bus["name"].astype(str)
        name_by_bus = bus_names.to_dict()
        for idx, row in net.switch.iterrows():
            if str(row["et"]) != "b":
                continue
            bus = int(row["bus"])
            element = int(row["element"])
            name_pair = tuple(sorted((name_by_bus.get(bus, ""), name_by_bus.get(element, ""))))
            if name_pair in fixed_pairs:
                fixed_indices.append(idx)
        if fixed_indices:
            net.switch.loc[fixed_indices, "closed"] = True

    data = extract_network_data(net)
    if force_switch_closed:
        for edge in data.edges:
            if edge.is_switch and edge.switch_index is not None:
                net.switch.at[edge.switch_index, "closed"] = True
        data = extract_network_data(net)
    else:
        for edge in data.edges:
            if edge.is_switch and edge.switch_index is not None:
                net.switch.at[edge.switch_index, "closed"] = bool(net.switch.at[edge.switch_index, "closed"])
        data = extract_network_data(net)

    voltage_bounds = None
    if apply_voltage_bounds:
        voltage_bounds = ((0.85 * data.base_kv) ** 2, (1.15 * data.base_kv) ** 2)

    switch_initial = None
    if penalize_switch_changes:
        switch_initial = {
            edge.edge_id: int(data.switch_closed.get(edge.switch_index, True))
            for edge in data.edges
            if edge.is_switch
        }

    debug_context = None
    if debug:
        isolated_load_buses = [
            bus
            for bus in data.buses
            if not any(edge.from_bus == bus or edge.to_bus == bus for edge in data.edges)
            and (abs(data.loads_p.get(bus, 0.0)) > 1e-9 or abs(data.loads_q.get(bus, 0.0)) > 1e-9)
        ]
        switch_edges = [
            (edge.edge_id, edge.from_bus, edge.to_bus)
            for edge in data.edges
            if edge.is_switch
        ]
        switch_status: list[tuple[int, int | None, bool]] = []
        for edge in data.edges:
            if not edge.is_switch:
                continue
            switch_index = edge.switch_index
            closed = data.switch_closed.get(switch_index, True) if switch_index is not None else True
            switch_status.append((edge.edge_id, switch_index, closed))
        switch_closed = [edge_id for edge_id, _, closed in switch_status if closed]
        switch_open = [edge_id for edge_id, _, closed in switch_status if not closed]
        edge_with_limit = [edge.edge_id for edge in data.edges if edge.max_i_ka > 0]
        active_buses = [
            bus
            for bus in data.buses
            if bus in data.slack_buses
            or any(edge.to_bus == bus or edge.from_bus == bus for edge in data.edges if not edge.is_switch)
            or abs(data.loads_p.get(bus, 0.0)) > 1e-9
            or abs(data.loads_q.get(bus, 0.0)) > 1e-9
        ]
        graph = pp.topology.create_nxgraph(net, include_lines=True, include_switches=True)
        components = list(nx.connected_components(graph))
        slack_buses = set(int(bus) for bus in net.ext_grid.bus) if not net.ext_grid.empty else set()
        slack_component: set[int] = set()
        island_summaries: list[dict[str, object]] = []
        for idx, comp in enumerate(components, start=1):
            comp_set = {int(bus) for bus in comp}
            has_slack = bool(slack_buses & comp_set)
            if has_slack:
                slack_component = comp_set
            island_summaries.append({"id": idx, "bus_count": len(comp_set), "has_slack": has_slack})
        buses_outside_slack = sorted(set(data.buses) - slack_component) if slack_component else sorted(data.buses)
        debug_context = {
            "bus_count": len(data.buses),
            "edge_count": len(data.edges),
            "switch_edge_count": len(switch_edges),
            "switch_closed_count": len(switch_closed),
            "switch_open_count": len(switch_open),
            "active_bus_count": len(active_buses),
            "isolated_load_bus_count": len(isolated_load_buses),
            "edges_with_current_limits": len(edge_with_limit),
            "island_count": len(components),
            "islands": island_summaries,
            "buses_outside_slack": buses_outside_slack,
            "isolated_load_buses": isolated_load_buses,
            "switch_edges": switch_edges,
            "switch_status": switch_status,
            "voltage_bounds": voltage_bounds,
            "base_kv": data.base_kv,
        }

    soc_reference_v2: dict[int, float] | None = None
    solve_result = None
    model = None
    balance_constraints = None

    total_iterations = max(int(soc_reference_iterations), 1)
    for iteration in range(total_iterations):
        model, balance_constraints = build_misocp_model(
            data,
            weights=weights,
            switch_cost=switch_cost,
            voltage_bounds=voltage_bounds,
            soc_relax=soc_relax,
            soc_slack_weight=1000.0,
            voltage_drop_slack_weight=10000.0,
            trafo_voltage_drop_slack_weight=200000.0,
            soc_slack_cap=soc_slack_cap,
            voltage_drop_slack_cap=voltage_drop_slack_cap,
            soc_reference_v2=soc_reference_v2,
            voltage_reference_upper_band=voltage_reference_upper_band,
            fix_switch_status=fix_switch_status,
            fixed_switch_indices=set(fixed_indices),
            switch_initial=switch_initial,
            enforce_radiality=enforce_radiality,
            radiality_slack=radiality_slack,
        )

        solve_result = run_dlmp_calculation(
            model,
            balance_constraints,
            solver_opts=solver_opts,
            debug=debug,
            debug_context=debug_context,
        )

        if iteration >= total_iterations - 1:
            break

        try:
            ac_net = _apply_switch_solution(net, data, solve_result.alpha_star)
            pp.runpp(ac_net, algorithm="nr", init="auto", calculate_voltage_angles=False)
            updated_reference: dict[int, float] = {}
            for bus in data.buses:
                if bus not in ac_net.res_bus.index:
                    continue
                vm = float(ac_net.res_bus.at[bus, "vm_pu"])
                if not np.isfinite(vm) or vm <= 0.0:
                    continue
                updated_reference[int(bus)] = vm**2
            if updated_reference:
                soc_reference_v2 = updated_reference
        except Exception:
            break

    if solve_result is None or model is None:
        raise RuntimeError("Failed to solve reconfiguration model.")

    soc_slack_values = [float(pyo.value(model.SOCSlack[e])) for e in model.EDGE]
    voltage_drop_slack_values = [float(pyo.value(model.VoltageDropSlack[e])) for e in model.EDGE]

    return ReconfigurationResult(
        lambda_dlmp=solve_result.lambda_dlmp,
        alpha_star=solve_result.alpha_star,
        socp_voltage_squared=solve_result.socp_voltage_squared,
        soc_slack_max=float(max(soc_slack_values)) if soc_slack_values else 0.0,
        soc_slack_sum=float(sum(soc_slack_values)) if soc_slack_values else 0.0,
        voltage_drop_slack_max=float(max(voltage_drop_slack_values)) if voltage_drop_slack_values else 0.0,
        voltage_drop_slack_sum=float(sum(voltage_drop_slack_values)) if voltage_drop_slack_values else 0.0,
    )


def run_reconfiguration(
    net: pp.pandapowerNet,
    weights: ModelWeights | None = None,
    switch_cost: float = 0.01,
    solver_opts: SolverOptions | None = None,
    debug: bool = False,
    force_switch_closed: bool = False,
    apply_voltage_bounds: bool = True,
    fix_switch_status: bool = False,
    penalize_switch_changes: bool = True,
    enforce_radiality: bool = True,
    radiality_slack: int = 0,
) -> tuple[dict[int, float], dict[int, int]]:
    result = run_reconfiguration_detailed(
        net=net,
        weights=weights,
        switch_cost=switch_cost,
        solver_opts=solver_opts,
        debug=debug,
        force_switch_closed=force_switch_closed,
        apply_voltage_bounds=apply_voltage_bounds,
        fix_switch_status=fix_switch_status,
        penalize_switch_changes=penalize_switch_changes,
        enforce_radiality=enforce_radiality,
        radiality_slack=radiality_slack,
    )
    return result.lambda_dlmp, result.alpha_star
