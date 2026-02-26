from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class VirtualBatteryConfig:
    dt_hours: float = 0.25
    soc_min: float = 0.1
    soc_max: float = 0.9
    soc_initial: float = 0.5
    p_charge_max: float = 0.5
    p_discharge_max: float = 0.5
    eta_ch: float = 0.95
    eta_dis: float = 0.95
    energy_capacity_mwh: float = 2.0
    inverter_s_mva: float = 0.8


@dataclass(frozen=True)
class VirtualBatterySchedule:
    p_ref: np.ndarray
    r_commit: np.ndarray
    soc: np.ndarray


def simulate_soc(
    p_ref: np.ndarray,
    config: VirtualBatteryConfig,
) -> np.ndarray:
    soc = np.zeros(len(p_ref) + 1, dtype=float)
    soc[0] = config.soc_initial
    for t, pref in enumerate(p_ref):
        p_dis = max(pref, 0.0)
        p_ch = max(-pref, 0.0)
        delta = (config.eta_ch * p_ch - p_dis / max(config.eta_dis, 1e-9)) * config.dt_hours
        soc_next = soc[t] + delta / max(config.energy_capacity_mwh, 1e-9)
        soc[t + 1] = float(np.clip(soc_next, config.soc_min, config.soc_max))
    return soc


def reserve_limit_from_state(p_ref: np.ndarray, soc: np.ndarray, config: VirtualBatteryConfig) -> np.ndarray:
    limits = np.zeros_like(p_ref, dtype=float)
    for t, pref in enumerate(p_ref):
        p_dis = max(pref, 0.0)
        p_ch = max(-pref, 0.0)
        headroom_power = config.p_discharge_max - p_dis + p_ch
        soc_headroom = max(soc[t] - config.soc_min, 0.0) * config.energy_capacity_mwh / max(config.dt_hours, 1e-9)
        inverter_headroom = max(config.inverter_s_mva - abs(pref), 0.0)
        limits[t] = max(min(headroom_power, soc_headroom, inverter_headroom), 0.0)
    return limits
