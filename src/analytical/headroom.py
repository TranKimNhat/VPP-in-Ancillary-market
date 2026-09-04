"""SoC-derated upward headroom, and the droop-sharing margin it implies.

Two independent limits cap a BESS GFM's upward active power:

  * a low-SoC taper on the converter/cell current, and
  * the energy actually left above the reserve floor, divided by how long the
    unit is expected to hold the response.

Below `soc_taper` the taper dominates; the energy limit dominates only when
the required hold time is long relative to the unit's E/P ratio.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DeratingModel:
    soc_min: float = 0.10  # reserve floor, not dischargeable
    soc_taper: float = 0.20  # full rating recovered at or above this SoC
    hold_time_h: float = 0.25  # window the primary response must be sustained


def p_head_mw(
    soc: float | np.ndarray,
    p_rated_mw: np.ndarray,
    e_rated_mwh: np.ndarray,
    model: DeratingModel,
) -> np.ndarray:
    """Upward headroom per unit, (…, n_gfm), for an idle (zero-dispatch) unit."""
    soc = np.atleast_1d(np.asarray(soc, dtype=float))[..., None]
    usable = np.clip(soc - model.soc_min, 0.0, None)

    taper = np.clip(usable / (model.soc_taper - model.soc_min), 0.0, 1.0) * p_rated_mw
    energy = usable * e_rated_mwh / model.hold_time_h
    return np.squeeze(np.minimum(taper, energy))


def droop_shares(s_rated_mva: np.ndarray, droop_r: np.ndarray | float) -> np.ndarray:
    """Fraction of a power imbalance each unit picks up under P-f droop.

    Steady-state droop sharing is proportional to S_g / R_g; with a uniform R
    it reduces to capacity sharing.
    """
    gain = np.asarray(s_rated_mva, dtype=float) / np.asarray(droop_r, dtype=float)
    return gain / gain.sum()


def df_ss_hz(dp_mw: float, s_rated_mva: np.ndarray, droop_r: float, f_nom_hz: float) -> float:
    """Steady-state frequency deviation of the unsaturated droop fleet.

    Note what is absent: P_head. In the unconstrained regime the frequency
    response is set by droop gain and converter rating alone, so SoC cannot
    reach it except by pushing a unit into saturation.
    """
    return f_nom_hz * dp_mw * droop_r / float(np.sum(s_rated_mva))


def dp_critical_mw(p_head: np.ndarray, shares: np.ndarray) -> float:
    """Largest imbalance before the first unit hits its headroom limit."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return float(np.min(np.where(shares > 0, p_head / shares, np.inf)))
