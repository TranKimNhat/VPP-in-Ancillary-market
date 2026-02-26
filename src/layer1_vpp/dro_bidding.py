from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.layer1_vpp.scenario_generator import PriceScenarioSet
from src.layer1_vpp.virtual_battery import VirtualBatteryConfig, VirtualBatterySchedule, simulate_soc


@dataclass(frozen=True)
class DroConfig:
    wasserstein_radius: float = 0.02
    degradation_cost: float = 1.0


@dataclass(frozen=True)
class DroSolveResult:
    schedule: VirtualBatterySchedule
    expected_energy_price: np.ndarray
    expected_reserve_price: np.ndarray
    robust_energy_margin: np.ndarray
    robust_reserve_margin: np.ndarray
    solver_status: str


def _price_statistics(scenarios: PriceScenarioSet) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    w = scenarios.scenario_weights
    energy = scenarios.energy_prices
    reserve = scenarios.reserve_prices

    mu_energy = np.einsum("s,st->t", w, energy)
    mu_reserve = np.einsum("s,st->t", w, reserve)

    mad_energy = np.einsum("s,st->t", w, np.abs(energy - mu_energy[None, :]))
    mad_reserve = np.einsum("s,st->t", w, np.abs(reserve - mu_reserve[None, :]))
    return mu_energy, mu_reserve, mad_energy, mad_reserve


def _solve_lp_schedule(
    scenarios: PriceScenarioSet,
    battery: VirtualBatteryConfig,
    dro: DroConfig,
) -> tuple[np.ndarray, np.ndarray, str]:
    try:
        from scipy.optimize import linprog
    except Exception:
        return _solve_greedy_schedule(scenarios, battery), np.zeros(len(scenarios.hours), dtype=float), "greedy_fallback"

    mu_e, mu_r, mad_e, mad_r = _price_statistics(scenarios)
    n = len(scenarios.hours)
    p_idx = np.arange(0, n)
    m_idx = np.arange(n, 2 * n)
    r_idx = np.arange(2 * n, 3 * n)

    c = np.zeros(3 * n, dtype=float)
    c[p_idx] = -mu_e + dro.degradation_cost + dro.wasserstein_radius * mad_e
    c[m_idx] = mu_e + dro.degradation_cost + dro.wasserstein_radius * mad_e
    c[r_idx] = -mu_r + dro.wasserstein_radius * mad_r

    bounds = []
    for _ in range(n):
        bounds.append((0.0, battery.p_discharge_max))
    for _ in range(n):
        bounds.append((0.0, battery.p_charge_max))
    for _ in range(n):
        bounds.append((0.0, None))

    rows: list[np.ndarray] = []
    rhs: list[float] = []

    for t in range(n):
        row = np.zeros(3 * n, dtype=float)
        row[p_idx[t]] = 1.0
        row[m_idx[t]] = -1.0
        row[r_idx[t]] = 1.0
        rows.append(row)
        rhs.append(battery.p_discharge_max)

        row2 = np.zeros(3 * n, dtype=float)
        row2[p_idx[t]] = 1.0
        row2[m_idx[t]] = 1.0
        row2[r_idx[t]] = 1.0
        rows.append(row2)
        rhs.append(battery.inverter_s_mva)

    a_dis = battery.dt_hours / max(battery.eta_dis * battery.energy_capacity_mwh, 1e-9)
    a_ch = battery.eta_ch * battery.dt_hours / max(battery.energy_capacity_mwh, 1e-9)

    for t in range(n):
        upper = np.zeros(3 * n, dtype=float)
        lower = np.zeros(3 * n, dtype=float)
        for k in range(t + 1):
            upper[p_idx[k]] += -a_dis
            upper[m_idx[k]] += a_ch
            lower[p_idx[k]] += a_dis
            lower[m_idx[k]] += -a_ch
        rows.append(upper)
        rhs.append(battery.soc_max - battery.soc_initial)
        rows.append(lower)
        rhs.append(battery.soc_initial - battery.soc_min)

    a_ub = np.vstack(rows)
    b_ub = np.array(rhs, dtype=float)

    result = linprog(c=c, A_ub=a_ub, b_ub=b_ub, bounds=bounds, method="highs")
    if not result.success:
        return _solve_greedy_schedule(scenarios, battery), np.zeros(n, dtype=float), f"greedy_fallback:{result.message}"

    x = result.x
    p_plus = x[p_idx]
    p_minus = x[m_idx]
    r_commit = np.maximum(x[r_idx], 0.0)
    p_ref = p_plus - p_minus
    return p_ref, r_commit, "optimal"


def _solve_greedy_schedule(
    scenarios: PriceScenarioSet,
    battery: VirtualBatteryConfig,
) -> np.ndarray:
    mu_e, _, _, _ = _price_statistics(scenarios)
    p_ref = np.zeros(len(mu_e), dtype=float)
    soc = battery.soc_initial
    for t, price in enumerate(mu_e):
        if price >= float(np.median(mu_e)):
            candidate = battery.p_discharge_max
        else:
            candidate = -battery.p_charge_max

        delta = (battery.eta_ch * max(-candidate, 0.0) - max(candidate, 0.0) / max(battery.eta_dis, 1e-9))
        soc_next = soc + delta * battery.dt_hours / max(battery.energy_capacity_mwh, 1e-9)
        if soc_next > battery.soc_max:
            candidate = 0.0
            soc_next = soc
        if soc_next < battery.soc_min:
            candidate = 0.0
            soc_next = soc

        p_ref[t] = candidate
        soc = soc_next
    return p_ref


def solve_wasserstein_dro(
    scenarios: PriceScenarioSet,
    battery: VirtualBatteryConfig,
    dro: DroConfig,
) -> DroSolveResult:
    mu_e, mu_r, mad_e, mad_r = _price_statistics(scenarios)
    p_ref, r_commit, status = _solve_lp_schedule(scenarios, battery, dro)

    soc = simulate_soc(p_ref, battery)
    robust_energy_margin = mu_e - dro.wasserstein_radius * mad_e
    robust_reserve_margin = mu_r - dro.wasserstein_radius * mad_r

    schedule = VirtualBatterySchedule(
        p_ref=p_ref,
        r_commit=r_commit,
        soc=soc,
    )
    return DroSolveResult(
        schedule=schedule,
        expected_energy_price=mu_e,
        expected_reserve_price=mu_r,
        robust_energy_margin=robust_energy_margin,
        robust_reserve_margin=robust_reserve_margin,
        solver_status=status,
    )
