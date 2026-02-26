from __future__ import annotations

from dataclasses import dataclass
import logging

import numpy as np
import pyomo.environ as pyo
from pyomo.opt import SolverFactory
from pyomo.util.infeasible import log_infeasible_constraints


@dataclass(frozen=True)
class DlmpSolveResult:
    lambda_dlmp: dict[int, float]
    alpha_star: dict[int, int]
    socp_voltage_squared: dict[int, float]


def solve_misocp(
    model: pyo.ConcreteModel,
    solver_opts: object | None = None,
    debug: bool = False,
    debug_context: dict[str, object] | None = None,
) -> dict[int, int]:
    solver = SolverFactory("mosek")
    if solver_opts is not None:
        mipgap = getattr(solver_opts, "mipgap", None)
        timelimit = getattr(solver_opts, "timelimit", None)
        if mipgap is not None:
            solver.options["mio_tol_rel_gap"] = mipgap
        if timelimit is not None:
            solver.options["mio_max_time"] = timelimit

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
    termination = result.solver.termination_condition
    if termination in {pyo.TerminationCondition.optimal, pyo.TerminationCondition.feasible}:
        return model

    raise RuntimeError(f"SOCP solve failed: {termination}")


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


def extract_socp_voltage_squared(model: pyo.ConcreteModel) -> dict[int, float]:
    if getattr(model, "_voltage_in_pu", False):
        bus_nominal_kv = getattr(model, "_bus_nominal_kv", {})
        return {
            int(bus): float(pyo.value(model.V2[bus])) * (float(bus_nominal_kv.get(int(bus), 1.0)) ** 2)
            for bus in model.BUS
        }
    return {int(bus): float(pyo.value(model.V2[bus])) for bus in model.BUS}


def run_dlmp_calculation(
    model: pyo.ConcreteModel,
    balance_constraints: dict[int, pyo.Constraint],
    solver_opts: object | None = None,
    debug: bool = False,
    debug_context: dict[str, object] | None = None,
) -> DlmpSolveResult:
    alpha_star = solve_misocp(model, solver_opts=solver_opts, debug=debug, debug_context=debug_context)
    fix_and_solve_socp(model, alpha_star)
    lambda_dlmp = extract_dlmp(model, balance_constraints)
    socp_voltage_squared = extract_socp_voltage_squared(model)
    return DlmpSolveResult(
        lambda_dlmp=lambda_dlmp,
        alpha_star=alpha_star,
        socp_voltage_squared=socp_voltage_squared,
    )
