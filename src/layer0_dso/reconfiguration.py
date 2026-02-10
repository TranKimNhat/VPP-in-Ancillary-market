from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import logging
import numpy as np
import pandapower as pp
import pyomo.environ as pyo
from pyomo.opt import SolverFactory
from pyomo.util.infeasible import log_infeasible_constraints



@dataclass(frozen=True)
class EdgeData:
    edge_id: int
    from_bus: int
    to_bus: int
    r_ohm: float
    x_ohm: float
    max_i_ka: float
    is_switch: bool
    switch_index: int | None


@dataclass(frozen=True)
class NetworkData:
    buses: list[int]
    slack_buses: list[int]
    edges: list[EdgeData]
    loads_p: dict[int, float]
    loads_q: dict[int, float]
    base_kv: float
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


def _collect_loads(net: pp.pandapowerNet) -> tuple[dict[int, float], dict[int, float]]:
    loads_p = {int(bus): 0.0 for bus in net.bus.index}
    loads_q = {int(bus): 0.0 for bus in net.bus.index}
    if net.load.empty:
        return loads_p, loads_q
    for _, row in net.load.iterrows():
        bus = int(row["bus"])
        loads_p[bus] = loads_p.get(bus, 0.0) + float(row.get("p_mw", 0.0))
        loads_q[bus] = loads_q.get(bus, 0.0) + float(row.get("q_mvar", 0.0))
    return loads_p, loads_q


def _edge_resistance(line: pp.Series) -> float:
    return float(line["r_ohm_per_km"] * line["length_km"])


def _edge_reactance(line: pp.Series) -> float:
    return float(line["x_ohm_per_km"] * line["length_km"])


def _edge_max_i_ka(line: pp.Series) -> float:
    value = line.get("max_i_ka", 0.0)
    if value is None:
        return 0.0
    return float(value)


def extract_network_data(net: pp.pandapowerNet) -> NetworkData:
    buses = [int(bus) for bus in net.bus.index]
    slack_buses = _slack_buses(net)
    base_bus = None
    if not net.trafo.empty:
        lv_candidates = net.trafo[net.trafo.hv_bus.isin(slack_buses)]
        if not lv_candidates.empty:
            base_bus = int(lv_candidates.lv_bus.iloc[0])
    if base_bus is None and slack_buses:
        base_bus = int(slack_buses[0])
    if base_bus is not None and not net.bus.empty and base_bus in net.bus.index:
        base_kv = float(net.bus.loc[base_bus, "vn_kv"])
    else:
        base_kv = float(net.bus["vn_kv"].mean()) if not net.bus.empty else 1.0

    loads_p, loads_q = _collect_loads(net)

    edges: list[EdgeData] = []
    edge_id = 0
    for _, line in net.line.iterrows():
        edges.append(
            EdgeData(
                edge_id=edge_id,
                from_bus=int(line["from_bus"]),
                to_bus=int(line["to_bus"]),
                r_ohm=_edge_resistance(line),
                x_ohm=_edge_reactance(line),
                max_i_ka=_edge_max_i_ka(line),
                is_switch=False,
                switch_index=None,
            )
        )
        edge_id += 1

    if not net.trafo.empty:
        for _, trafo in net.trafo.iterrows():
            hv_bus = int(trafo["hv_bus"])
            lv_bus = int(trafo["lv_bus"])
            sn_mva = float(trafo.get("sn_mva", 0.0) or 0.0)
            vn_hv_kv = float(trafo.get("vn_hv_kv", 0.0) or 0.0)
            vn_lv_kv = float(trafo.get("vn_lv_kv", 0.0) or 0.0)
            vk_percent = float(trafo.get("vk_percent", 0.0) or 0.0)
            vkr_percent = float(trafo.get("vkr_percent", 0.0) or 0.0)
            z_pu = vk_percent / 100.0
            r_pu = vkr_percent / 100.0
            x_pu = (max(z_pu**2 - r_pu**2, 0.0)) ** 0.5
            z_base_kv = vn_lv_kv if vn_lv_kv > 0 else vn_hv_kv
            if sn_mva > 0 and z_base_kv > 0:
                z_base = (z_base_kv**2) / sn_mva
                r_ohm = r_pu * z_base
                x_ohm = x_pu * z_base
            else:
                r_ohm = 0.0
                x_ohm = 0.0
            edges.append(
                EdgeData(
                    edge_id=edge_id,
                    from_bus=hv_bus,
                    to_bus=lv_bus,
                    r_ohm=r_ohm,
                    x_ohm=x_ohm,
                    max_i_ka=0.0,
                    is_switch=False,
                    switch_index=None,
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
                    max_i_ka=0.0,
                    is_switch=True,
                    switch_index=int(sw_idx),
                )
            )
            edge_id += 1

    return NetworkData(
        buses=buses,
        slack_buses=slack_buses,
        edges=edges,
        loads_p=loads_p,
        loads_q=loads_q,
        base_kv=base_kv,
        switch_closed=switch_closed,
    )


def switch_edge_map(net: pp.pandapowerNet) -> dict[int, dict[str, int]]:
    data = extract_network_data(net)
    return {
        edge.edge_id: {"from_bus": edge.from_bus, "to_bus": edge.to_bus}
        for edge in data.edges
        if edge.is_switch
    }


def build_misocp_model(
    data: NetworkData,
    weights: ModelWeights,
    switch_cost: float,
    voltage_bounds: tuple[float, float] | None = None,
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
    model.alpha = pyo.Var(model.SWITCH, domain=pyo.Binary)
    model.F = pyo.Var(model.EDGE, domain=pyo.Reals)

    edge_map = {edge.edge_id: edge for edge in data.edges}

    if voltage_bounds is not None:
        voltage_min, voltage_max = voltage_bounds

        def _voltage_bounds_rule(m: pyo.ConcreteModel, bus: int) -> pyo.Constraint:
            return pyo.inequality(voltage_min, m.V2[bus], voltage_max)

        model.VoltageBounds = pyo.Constraint(model.BUS, rule=_voltage_bounds_rule)

    if voltage_bounds is not None:
        def _slack_voltage_rule(m: pyo.ConcreteModel, bus: int) -> pyo.Constraint:
            return m.V2[bus] == data.base_kv**2

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
        if bus in slack_set:
            return pyo.Constraint.Skip
        load = data.loads_p.get(bus, 0.0) if not reactive else data.loads_q.get(bus, 0.0)
        if not incoming and not outgoing:
            if abs(load) <= 1e-9:
                return pyo.Constraint.Feasible
            return pyo.Constraint.Infeasible
        inflow = pyo.quicksum(var[e] for e in incoming)
        outflow = pyo.quicksum(var[e] for e in outgoing)
        return inflow - outflow == load

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
        or abs(data.loads_p.get(bus, 0.0)) > 1e-9
        or abs(data.loads_q.get(bus, 0.0)) > 1e-9
    ]
    active_bus_set = set(active_buses)
    flow_limit = float(len(active_buses))
    root_balance = len(active_buses) - len(slack_set)

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
        big_m = float(voltage_bounds[1] - voltage_bounds[0])
    else:
        big_m = float((2 * data.base_kv) ** 2)
    big_m_p = 10.0
    big_m_q = 10.0
    big_m_i = 10.0

    def _voltage_drop_rule(m: pyo.ConcreteModel, edge_id: int) -> pyo.Constraint:
        edge = edge_map[edge_id]
        from_bus = edge.from_bus
        to_bus = edge.to_bus
        r = edge.r_ohm
        x = edge.x_ohm
        drop = 2 * (r * m.P[edge_id] + x * m.Q[edge_id]) - (r**2 + x**2) * m.I2[edge_id]
        if edge.is_switch:
            return m.V2[to_bus] - m.V2[from_bus] <= drop + big_m * (1 - m.alpha[edge_id])
        return m.V2[to_bus] - m.V2[from_bus] == drop

    model.VoltageDrop = pyo.Constraint(model.EDGE, rule=_voltage_drop_rule)

    def _voltage_drop_switch_rule(m: pyo.ConcreteModel, edge_id: int) -> pyo.Constraint:
        edge = edge_map[edge_id]
        from_bus = edge.from_bus
        to_bus = edge.to_bus
        r = edge.r_ohm
        x = edge.x_ohm
        drop = 2 * (r * m.P[edge_id] + x * m.Q[edge_id]) - (r**2 + x**2) * m.I2[edge_id]
        return m.V2[to_bus] - m.V2[from_bus] >= drop - big_m * (1 - m.alpha[edge_id])

    model.VoltageDropSwitch = pyo.Constraint(model.SWITCH, rule=_voltage_drop_switch_rule)

    def _switch_status_rule(m: pyo.ConcreteModel, edge_id: int) -> pyo.Constraint:
        edge = edge_map[edge_id]
        if edge.switch_index is None:
            return pyo.Constraint.Skip
        closed = data.switch_closed.get(edge.switch_index, True)
        if closed:
            return m.alpha[edge_id] == 1
        return m.alpha[edge_id] == 0

    model.SwitchStatus = pyo.Constraint(model.SWITCH, rule=_switch_status_rule)

    v_base_sq = data.base_kv**2

    def _soc_rule(m: pyo.ConcreteModel, edge_id: int) -> pyo.Constraint:
        return m.P[edge_id] ** 2 + m.Q[edge_id] ** 2 <= v_base_sq * m.I2[edge_id]

    model.SOC = pyo.Constraint(model.EDGE, rule=_soc_rule)

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
        if edge.max_i_ka <= 0:
            return pyo.Constraint.Skip
        return m.I2[edge_id] <= edge.max_i_ka**2

    model.CurrentLimit = pyo.Constraint(model.EDGE, rule=_current_limit_rule)

    def _min_current_rule(m: pyo.ConcreteModel, edge_id: int) -> pyo.Constraint:
        edge = edge_map[edge_id]
        if edge.max_i_ka <= 0:
            return pyo.Constraint.Skip
        return m.I2[edge_id] >= 0.0

    model.MinCurrent = pyo.Constraint(model.EDGE, rule=_min_current_rule)

    loss_term = sum(edge_map[e].r_ohm * model.I2[e] for e in model.EDGE)
    switch_term = switch_cost * sum(model.alpha[e] for e in model.SWITCH)
    voltage_term = sum((model.V2[b] - 1.0) ** 2 for b in model.BUS)

    model.Objective = pyo.Objective(
        expr=weights.loss * loss_term + weights.switch * switch_term + weights.voltage * voltage_term,
        sense=pyo.minimize,
    )

    return model, balance_constraints


def solve_misocp(
    model: pyo.ConcreteModel,
    solver_opts: SolverOptions | None = None,
    debug: bool = False,
    debug_context: dict[str, object] | None = None,
) -> dict[int, int]:
    solver = SolverFactory("mosek")
    if solver_opts:
        if solver_opts.mipgap is not None:
            solver.options["mio_tol_rel_gap"] = solver_opts.mipgap
        if solver_opts.timelimit is not None:
            solver.options["mio_max_time"] = solver_opts.timelimit

    result = solver.solve(model, tee=True)
    if result.solver.termination_condition not in {pyo.TerminationCondition.optimal, pyo.TerminationCondition.feasible}:
        if debug:
            if debug_context:
                debug_logger = logging.getLogger(__name__)
                for key, value in debug_context.items():
                    debug_logger.info("%s: %s", key, value)
            logging.getLogger("pyomo.util.infeasible").setLevel(logging.INFO)
            log_infeasible_constraints(model, log_expression=True, log_variables=True)
        raise RuntimeError(f"MISOCP solve failed: {result.solver.termination_condition}")

    alpha_star: dict[int, int] = {}
    for edge_id in model.SWITCH:
        value = pyo.value(model.alpha[edge_id])
        alpha_star[int(edge_id)] = 1 if value >= 0.5 else 0
    return alpha_star


def fix_and_solve_socp(
    model: pyo.ConcreteModel,
    alpha_star: dict[int, int],
) -> pyo.ConcreteModel:
    for edge_id, value in alpha_star.items():
        if edge_id in model.SWITCH:
            model.alpha[edge_id].fix(value)

    pyo.TransformationFactory("core.relax_integer_vars").apply_to(model)
    model.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)
    solver = SolverFactory("mosek")
    result = solver.solve(model, tee=True)
    if result.solver.termination_condition != pyo.TerminationCondition.optimal:
        raise RuntimeError(f"SOCP solve failed: {result.solver.termination_condition}")
    return model


def extract_dlmp(
    model: pyo.ConcreteModel,
    balance_constraints: dict[int, pyo.Constraint],
) -> dict[int, float]:
    dlmp: dict[int, float] = {}
    for bus, constraint in balance_constraints.items():
        if constraint not in model.dual:
            dlmp[int(bus)] = float("nan")
        else:
            dlmp[int(bus)] = float(model.dual[constraint])
    return dlmp


def run_reconfiguration(
    net: pp.pandapowerNet,
    weights: ModelWeights | None = None,
    switch_cost: float = 0.01,
    solver_opts: SolverOptions | None = None,
    debug: bool = False,
    force_switch_closed: bool = False,
    apply_voltage_bounds: bool = True,
) -> tuple[dict[int, float], dict[int, int]]:
    weights = weights or ModelWeights()

    if not net.switch.empty:
        candidate_mask = (net.switch["et"].astype(str) == "b") & (net.switch["type"].astype(str) == "CB")
        net.switch.loc[~candidate_mask, "closed"] = net.switch.loc[~candidate_mask, "closed"].fillna(True)

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
        if debug:
            voltage_bounds = ((0.90 * data.base_kv) ** 2, (1.10 * data.base_kv) ** 2)
        else:
            voltage_bounds = ((0.95 * data.base_kv) ** 2, (1.05 * data.base_kv) ** 2)

    model, balance_constraints = build_misocp_model(
        data,
        weights=weights,
        switch_cost=switch_cost,
        voltage_bounds=voltage_bounds,
    )

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
        debug_context = {
            "isolated_load_buses": isolated_load_buses,
            "switch_edges": switch_edges,
            "voltage_bounds": voltage_bounds,
            "base_kv": data.base_kv,
        }

    alpha_star = solve_misocp(model, solver_opts=solver_opts, debug=debug, debug_context=debug_context)

    fix_and_solve_socp(model, alpha_star)
    lambda_dlmp = extract_dlmp(model, balance_constraints)

    return lambda_dlmp, alpha_star
