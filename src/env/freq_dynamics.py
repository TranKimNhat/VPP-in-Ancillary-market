from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass
class FrequencyState:
    t: float
    f_hz: float
    delta_f_hz: float
    rocof_hz_s: float
    p_gov_pu: float
    p_ref_pu: float
    agc_integral_hz_s: float


class FrequencyDynamics:
    def __init__(
        self,
        H: float = 2.5,
        D: float = 1.5,
        R: float = 0.05,
        Tg: float = 0.2,
        f0: float = 50.0,
        Kp: float = 0.2,
        Ki: float = 0.1,
        deadband: float = 0.1,
        pref_min_pu: float = -0.05,
        pref_max_pu: float = 0.05,
        max_abs_delta_f_hz: float = 5.0,
    ) -> None:
        self.H = float(H)
        self.D = float(D)
        self.R = float(R)
        self.Tg = float(Tg)
        self.f0 = float(f0)
        self.Kp = float(Kp)
        self.Ki = float(Ki)
        self.deadband = float(deadband)
        self.pref_min_pu = float(pref_min_pu)
        self.pref_max_pu = float(pref_max_pu)
        self.max_abs_delta_f_hz = float(max_abs_delta_f_hz)

        # GFM units with individual inertia (H) and damping (D) parameters
        # LOW-INERTIA ISLANDED MICROGRID configuration for FFR evaluation
        #
        # Literature basis for H values:
        #   - Bevrani et al. (2016, 406 citations): H∞ control for islanded MG frequency
        #   - Kerdphol et al. (2017, 218 citations): H < 3s for high RES penetration MG
        #   - Rafiee et al. (2021, 61 citations): VSG inertia 1-3s typical
        #   - Paturet et al. (2019, 187 citations): RoCoF constraints with low H
        #   - Ali et al. (2019, 104 citations): CDM control for low-inertia islanded MG
        #
        # H_SYS = Σ(H_i * S_i) / S_BASE ≈ 1.2s (very low - stress test for FFR)
        # This simulates 100% inverter-based system with synthetic inertia only
        #
        # G1 @ bus114 (main VSG anchor), G2 @ bus60 (droop+PV)
        # G4 @ bus67, G5 @ bus36, G6 @ bus101 (support units)
        self._gfm_params = {
            "G1": {"S": 6.0, "H": 1.5, "D": 1.0, "R": 0.04},  # Main VSG anchor
            "G2": {"S": 3.0, "H": 1.2, "D": 0.8, "R": 0.05},  # Droop + PV hybrid
            "G4": {"S": 2.0, "H": 1.0, "D": 0.5, "R": 0.05},  # Distributed support
            "G5": {"S": 2.0, "H": 1.0, "D": 0.5, "R": 0.05},  # Distributed support
            "G6": {"S": 2.0, "H": 1.0, "D": 0.5, "R": 0.05},  # Distributed support
        }
        self.S_BASE = 15.7  # System base MVA
        self.F0 = self.f0
        self._update_system_params()

        self.reset(f0=self.f0)

    def _update_system_params(self, soc_weights: dict[str, float] | None = None) -> None:
        """Calculate aggregated system parameters from individual GFM units.

        H_SYS = Σ(H_i * S_i * w_i) / S_BASE
        D_SYS = Σ(D_i * S_i * w_i) / S_BASE
        1/R_SYS = Σ(S_i * w_i / R_i) / S_BASE

        Args:
            soc_weights: Optional dict mapping GFM id to weight (0-1) based on SOC.
                         Lower SOC -> lower weight to reduce contribution.
        """
        sum_HS = 0.0
        sum_DS = 0.0
        sum_S_over_R = 0.0

        for gfm_id, params in self._gfm_params.items():
            w = 1.0 if soc_weights is None else soc_weights.get(gfm_id, 1.0)
            S_eff = params["S"] * w
            sum_HS += params["H"] * S_eff
            sum_DS += params["D"] * S_eff
            sum_S_over_R += S_eff / params["R"]

        self.H_SYS = sum_HS / self.S_BASE
        self.D_SYS = sum_DS / self.S_BASE
        self.R_SYS = self.S_BASE / sum_S_over_R if sum_S_over_R > 0 else self.R

        # Update instance D and R to use aggregated values
        self.D = self.D_SYS
        self.R = self.R_SYS

    def update_soc_weights(self, soc_dict: dict[str, float]) -> None:
        """Dynamically adjust GFM contributions based on SOC levels.

        Args:
            soc_dict: Dict mapping GFM id to SOC (0-1).
                      SOC < 0.2 -> reduced contribution
                      SOC > 0.8 -> full contribution
        """
        weights = {}
        for gfm_id, soc in soc_dict.items():
            if soc < 0.2:
                weights[gfm_id] = 0.5 + soc * 2.5  # 0.5 at SOC=0, 1.0 at SOC=0.2
            elif soc > 0.8:
                weights[gfm_id] = 1.0
            else:
                weights[gfm_id] = 1.0
        self._update_system_params(soc_weights=weights)

    def update_topology(self, connected_gfm_ids: set[str] | None = None) -> None:
        """Update H_SYS based on electrically connected GFMs only.

        When topology changes (tie-switch reconfiguration), some GFMs may become
        electrically isolated from the main system. Only connected GFMs contribute
        to system inertia and frequency response.

        Reference: Farmer et al. (2020) "Optimising power system frequency stability
        using virtual inertia" - distributed inertia must consider network topology.

        Args:
            connected_gfm_ids: Set of GFM IDs (e.g., {"G1", "G2", "G4"}) that are
                               electrically connected. If None, all GFMs contribute.
        """
        if connected_gfm_ids is None:
            self._update_system_params(soc_weights=None)
            return

        weights = {
            gfm_id: 1.0 if gfm_id in connected_gfm_ids else 0.0
            for gfm_id in self._gfm_params
        }
        self._update_system_params(soc_weights=weights)

    def reset(self, f0: float = 50.0) -> None:
        self.f0 = float(f0)
        self.t = 0.0
        self._x = 0.0  # pu frequency deviation: x = delta_f_hz / f0
        self._p_gov = 0.0
        self._p_ref = 0.0
        self._agc_integral = 0.0
        self._rocof = 0.0
        self._f_prev = 0.0  # Track delta_f, not absolute f

    def step(self, dt: float, delta_P_pu: float, P_bess_pu: float = 0.0) -> FrequencyState:
        dt_val = float(dt)
        delta_p = float(delta_P_pu)
        p_bess = float(P_bess_pu)

        assert math.isfinite(dt_val) and dt_val > 0.0, f"Invalid dt={dt_val}"
        assert math.isfinite(delta_p), f"delta_P_pu non-finite: {delta_p}"
        assert math.isfinite(p_bess), f"P_bess_pu non-finite: {p_bess}"
        assert abs(p_bess) <= 1.0, f"P_bess_pu too large: {p_bess}"

        delta_f_hz = self._x * self.f0
        assert math.isfinite(delta_f_hz), f"delta_f_hz non-finite before update: x={self._x}, f0={self.f0}"
        if abs(delta_f_hz) <= self.deadband:
            delta_f_for_agc = 0.0
        else:
            delta_f_for_agc = delta_f_hz

        self._agc_integral += delta_f_for_agc * dt_val
        self._p_ref += (self.Kp * delta_f_for_agc + self.Ki * self._agc_integral) * dt_val
        self._p_ref = float(min(self.pref_max_pu, max(self.pref_min_pu, self._p_ref)))

        p_gov_dot = (-self._p_gov + self._p_ref - (self._x / self.R)) / self.Tg
        self._p_gov += p_gov_dot * dt_val

        x_dot = (self._p_gov - delta_p - self.D * self._x + p_bess) / (2.0 * self.H_SYS)
        self._x += x_dot * dt_val

        delta_f_after_update = self._x * self.f0
        self._rocof = (delta_f_after_update - self._f_prev) / dt_val
        self._f_prev = delta_f_after_update
        self.t += dt_val

        if not math.isfinite(self._x):
            self._x = math.copysign(self.max_abs_delta_f_hz / self.f0, x_dot if math.isfinite(x_dot) and x_dot != 0.0 else 1.0)
            self._p_gov = 0.0
            self._p_ref = 0.0
            self._agc_integral = 0.0
            self._rocof = 0.0

        delta_f_after = self._x * self.f0
        if not math.isfinite(delta_f_after):
            delta_f_after = math.copysign(self.max_abs_delta_f_hz, delta_f_hz if math.isfinite(delta_f_hz) and delta_f_hz != 0.0 else 1.0)

        if abs(delta_f_after) > self.max_abs_delta_f_hz:
            self._x = math.copysign(self.max_abs_delta_f_hz / self.f0, delta_f_after)
            self._p_gov = 0.0
            self._p_ref = 0.0
            self._agc_integral = 0.0
            self._rocof = 0.0

        assert math.isfinite(self._x), (
            f"x exploded: x={self._x}, x_dot={x_dot}, p_gov={self._p_gov}, "
            f"delta_p={delta_p}, p_bess={p_bess}, dt={dt_val}"
        )
        assert math.isfinite(self._rocof), f"rocof non-finite: rocof={self._rocof}, x_dot={x_dot}"

        return self.get_state()

    def compute_rocof(self, delta_P_mw: float) -> float:
        delta_P_pu = float(delta_P_mw) / self.S_BASE
        rocof = -self.F0 * delta_P_pu / (2.0 * self.H_SYS)
        return float(rocof)

    @property
    def h_sys(self) -> float:
        return float(self.H_SYS)

    @property
    def rocof(self) -> float:
        return float(self._rocof)

    def get_state(self) -> FrequencyState:
        delta_f_hz = self._x * self.f0
        return FrequencyState(
            t=float(self.t),
            f_hz=float(self.f0 + delta_f_hz),
            delta_f_hz=float(delta_f_hz),
            rocof_hz_s=float(self._rocof),
            p_gov_pu=float(self._p_gov),
            p_ref_pu=float(self._p_ref),
            agc_integral_hz_s=float(self._agc_integral),
        )
