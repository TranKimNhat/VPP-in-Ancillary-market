from __future__ import annotations

from typing import Any

import numpy as np
from scipy.integrate import solve_ivp


def simulate_freq_response(
    delta_P_mw: float,
    P_bess_mw: float,
    P_v2g_mw: float,
    H_sys: float = 2.5,
    D_sys: float = 1.5,
    f0: float = 50.0,
    S_base: float = 15.705,
    t_end: float = 30.0,
    dt: float = 0.01,
    t_ramp: float = 0.5,
) -> dict[str, Any]:
    """
    Simulate frequency response after contingency using swing equation:
      2H/f0 × df/dt = ΔP_net(t) - D × Δf/f0

    ΔP_net(t) = delta_P_pu + P_response(t)
    P_response(t): ramp up from 0 to (P_bess+P_v2g)/S_base in t_ramp seconds.
    """
    dP_pu = float(delta_P_mw) / float(S_base)
    p_resp_pu = (float(P_bess_mw) + float(P_v2g_mw)) / float(S_base)

    def dp_net(t: float) -> float:
        ramp = min(t / t_ramp, 1.0) if t_ramp > 0 else 1.0
        return dP_pu + p_resp_pu * ramp

    def swing_eq(t: float, y: np.ndarray) -> list[float]:
        df_dt = (f0 / (2.0 * H_sys)) * (dp_net(t) - D_sys * y[0] / f0)
        return [float(df_dt)]

    t_eval = np.arange(0.0, float(t_end), float(dt), dtype=float)
    sol = solve_ivp(
        swing_eq,
        (0.0, float(t_end)),
        [0.0],
        t_eval=t_eval,
        method="RK45",
    )

    delta_f = np.asarray(sol.y[0], dtype=float)
    f = f0 + delta_f
    rocof = np.gradient(f, float(dt))

    nadir_idx = int(np.argmin(f))
    nadir = float(f[nadir_idx])
    t_nadir = float(t_eval[nadir_idx])
    f_ss = float(f[-1])

    return {
        "t": t_eval,
        "f": f,
        "rocof": rocof,
        "nadir": nadir,
        "t_nadir": t_nadir,
        "f_ss": f_ss,
        "nadir_violation": bool(nadir < 49.5),
    }


def simulate_scenarios_comparison(
    delta_P_mw: float = -1.0,
    policies_response: dict[str, dict[str, float]] | None = None,
    H_sys: float = 2.5,
) -> dict[str, dict[str, Any]]:
    """Simulate f(t) for multiple policies under the same contingency."""
    if policies_response is None:
        policies_response = {
            "Proposed (MAPPO)": {"P_bess": 0.85, "P_v2g": 0.35},
            "Rule-based": {"P_bess": 0.60, "P_v2g": 0.20},
            "Random": {"P_bess": 0.15, "P_v2g": 0.08},
            "No BESS": {"P_bess": 0.00, "P_v2g": 0.00},
        }

    results: dict[str, dict[str, Any]] = {}
    for policy, response in policies_response.items():
        results[policy] = simulate_freq_response(
            delta_P_mw=delta_P_mw,
            P_bess_mw=float(response.get("P_bess", 0.0)),
            P_v2g_mw=float(response.get("P_v2g", 0.0)),
            H_sys=H_sys,
        )
    return results
