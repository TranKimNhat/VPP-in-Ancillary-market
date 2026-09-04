"""Per-DER Droop FFR Baseline.

This module provides a classical droop-based Fast Frequency Response (FFR)
controller as a baseline for comparison with MAPPO-based control.

Each DER (BESS, V2G) responds proportionally to frequency deviation:
    P_ffr_i = -k_droop_i * delta_f / S_BASE

where k_droop_i = coef * P_rated_i (MW/Hz).

Reference:
- Kerdphol et al. (2019), "Robust Virtual Inertia Control of an Islanded
  Microgrid Considering High Penetration of Renewable Energy"
- Bevrani et al. (2014), "Power System Frequency Control"
"""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np


@dataclass
class DroopFFRConfig:
    """Configuration for per-DER droop FFR controller.

    Attributes:
        k_coef_bess: Droop coefficient multiplier for BESS (MW/Hz per MW capacity).
                     Higher = more aggressive response. Default 0.15.
        k_coef_v2g: Droop coefficient multiplier for V2G. Default 0.10 (less than BESS).
        tau_bess_s: First-order lag time constant for BESS (s). Default 0.1s (100ms).
        tau_v2g_s: First-order lag time constant for V2G (s). Default 0.3s (300ms).
        s_base_mw: System base power (MW). Default 15.7 (IEEE 123-bus modified).
        ffr_threshold_hz: Activation threshold |delta_f| (Hz). Default 0.15.
        ffr_deactivation_hz: Deactivation threshold (hysteresis). Default 0.05.
    """
    k_coef_bess: float = 0.15
    k_coef_v2g: float = 0.10
    tau_bess_s: float = 0.1
    tau_v2g_s: float = 0.3
    s_base_mw: float = 15.7
    ffr_threshold_hz: float = 0.15
    ffr_deactivation_hz: float = 0.05


class DroopFFRBaseline:
    """Per-DER droop-based FFR controller (baseline for MAPPO comparison).

    Usage:
        bess_caps = [0.325, 0.325, 0.325, 0.275, 0.275, 0.275, 0.225, 0.225, 0.225]
        v2g_caps  = [0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.075, 0.075, 0.075]
        ctrl = DroopFFRBaseline(bess_caps, v2g_caps)

        for t in time_steps:
            p_bess_pu, p_v2g_pu = ctrl.step(delta_f_hz=freq_state.delta_f_hz, dt=0.01)
            freq_state = freq_dyn.step(dt, delta_P_pu, P_bess_pu=p_bess_pu, P_v2g_pu=p_v2g_pu)
    """

    def __init__(
        self,
        bess_capacities_mw: list[float] | np.ndarray,
        v2g_capacities_mw: list[float] | np.ndarray,
        config: DroopFFRConfig | None = None,
    ):
        self.cfg = config if config is not None else DroopFFRConfig()
        self._bess_caps = np.asarray(bess_capacities_mw, dtype=np.float32)
        self._v2g_caps = np.asarray(v2g_capacities_mw, dtype=np.float32)

        # Per-DER droop coefficients (MW/Hz)
        self._k_bess = self.cfg.k_coef_bess * self._bess_caps  # MW/Hz
        self._k_v2g = self.cfg.k_coef_v2g * self._v2g_caps    # MW/Hz

        # Per-DER state for first-order lag (pu)
        self._p_bess_state = np.zeros_like(self._bess_caps)
        self._p_v2g_state = np.zeros_like(self._v2g_caps)

        self._active = False

    @property
    def n_bess(self) -> int:
        return int(self._bess_caps.size)

    @property
    def n_v2g(self) -> int:
        return int(self._v2g_caps.size)

    @property
    def k_droop_bess(self) -> np.ndarray:
        """Per-DER BESS droop coefficients (MW/Hz)."""
        return self._k_bess.copy()

    @property
    def k_droop_v2g(self) -> np.ndarray:
        """Per-DER V2G droop coefficients (MW/Hz)."""
        return self._k_v2g.copy()

    @property
    def is_active(self) -> bool:
        return self._active

    def reset(self) -> None:
        """Reset internal state."""
        self._p_bess_state.fill(0.0)
        self._p_v2g_state.fill(0.0)
        self._active = False

    def step(
        self,
        delta_f_hz: float,
        dt: float,
        force_active: bool | None = None,
    ) -> tuple[float, float]:
        """Compute per-DER FFR power for one time step.

        Args:
            delta_f_hz: Frequency deviation (Hz). Negative = under-frequency.
            dt: Time step (s).
            force_active: If provided, override automatic activation logic.

        Returns:
            p_bess_pu: Aggregate BESS FFR power (pu, positive = inject).
            p_v2g_pu: Aggregate V2G FFR power (pu, positive = inject).
        """
        # Activation logic with hysteresis
        if force_active is not None:
            self._active = bool(force_active)
        else:
            if not self._active and abs(delta_f_hz) > self.cfg.ffr_threshold_hz:
                self._active = True
            elif self._active and abs(delta_f_hz) < self.cfg.ffr_deactivation_hz:
                self._active = False

        if self._active:
            delta_f_clamped = float(np.clip(delta_f_hz, -1.0, 1.0))

            # Per-DER droop with first-order lag
            alpha_bess = dt / (self.cfg.tau_bess_s + dt)
            alpha_v2g = dt / (self.cfg.tau_v2g_s + dt)

            for i in range(self.n_bess):
                p_target = -self._k_bess[i] * delta_f_clamped / self.cfg.s_base_mw
                self._p_bess_state[i] = (1.0 - alpha_bess) * self._p_bess_state[i] + alpha_bess * p_target

            for i in range(self.n_v2g):
                p_target = -self._k_v2g[i] * delta_f_clamped / self.cfg.s_base_mw
                self._p_v2g_state[i] = (1.0 - alpha_v2g) * self._p_v2g_state[i] + alpha_v2g * p_target
        else:
            # Decay to zero when inactive
            alpha_bess = dt / (self.cfg.tau_bess_s + dt)
            alpha_v2g = dt / (self.cfg.tau_v2g_s + dt)
            self._p_bess_state *= (1.0 - alpha_bess)
            self._p_v2g_state *= (1.0 - alpha_v2g)

        return float(self._p_bess_state.sum()), float(self._p_v2g_state.sum())

    def get_per_der_power(self) -> tuple[np.ndarray, np.ndarray]:
        """Return current per-DER power state (pu)."""
        return self._p_bess_state.copy(), self._p_v2g_state.copy()


def from_placement(placement: dict, config: DroopFFRConfig | None = None) -> DroopFFRBaseline:
    """Build DroopFFRBaseline from placement JSON dict (e.g., official_placement_v3.json).

    Args:
        placement: Loaded placement dict containing 'evcs' list with 'bess_mw' and 'v2g_mw'.
        config: Optional DroopFFRConfig override.
    """
    evcs_list = placement.get("evcs", [])
    bess_caps = [float(ev["bess_mw"]) for ev in evcs_list]
    v2g_caps = [float(ev["v2g_mw"]) for ev in evcs_list]
    return DroopFFRBaseline(bess_caps, v2g_caps, config)


__all__ = ["DroopFFRConfig", "DroopFFRBaseline", "from_placement"]
