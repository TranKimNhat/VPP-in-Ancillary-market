"""
Topology-aware linearized frequency dynamics for all-GFM islanded microgrid.

This module implements the Kron-reduced state-space model described in the
reviewer response (Trujillo et al. framework adapted for 100% inverter-based
resources). The system matrix A_f depends on both topology G_t (via the
reduced Jacobian J_r) and the learnable droop gains K_droop (via M_p).

Key features vs legacy FrequencyDynamics:
  1. Per-bus frequency deviation (not scalar COI)
  2. Implicit active losses via AC Jacobian (R retained)
  3. Topology enters system matrix through J_r(G_t)
  4. Learnable droop K_droop_i enters M_p diagonal
  5. Matrix exponential integration for speed (cached per topology/K bin)

Reference:
  Trujillo et al., "Reduced-order frequency dynamics for multi-machine
  power systems with high shares of grid-forming inverters," 2024.
"""
from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandapower as pp
import scipy.linalg
import scipy.sparse as sp

logger = logging.getLogger(__name__)


@dataclass
class FrequencyStateLTI:
    """State output from LTI topology-aware frequency dynamics."""
    t: float = 0.0
    f_hz: float = 50.0
    delta_f_hz: float = 0.0
    rocof_hz_s: float = 0.0
    delta_f_per_bus: np.ndarray = field(default_factory=lambda: np.zeros(6))
    rocof_per_bus: np.ndarray = field(default_factory=lambda: np.zeros(6))
    delta_f_worst: float = 0.0
    rocof_worst: float = 0.0
    p_ref_pu: float = 0.0
    agc_integral_hz_s: float = 0.0


@dataclass
class GFMParams:
    """Parameters for a single GFM inverter."""
    gfm_id: str
    bus_id: int
    pp_bus_idx: int
    rating_mva: float
    tau_filter: float = 0.1
    H_virt: float = 0.0
    mode: str = "Droop"


class LTITopologyFreqDynamics:
    """
    Linearized topology-aware frequency dynamics with Kron reduction.

    The state vector is x = [Δδ_rel (n_gfm-1), Δω (n_gfm)]^T.
    Relative angles are w.r.t. the reference GFM (largest rating).

    System:
        dx/dt = A_f(G_t, K_droop) x + B_ref ΔP_ref + B_net ΔP_L
    where:
        A_f = [[0, ω0 * 1^{-1}], [-T_c^{-1} M_p J̃_r, -T_c^{-1}]]
        1^{-1} = relative-angle transform (n_gfm-1 × n_gfm)
    """

    def __init__(
        self,
        placement: dict[str, Any],
        base_net: pp.pandapowerNet,
        bus_map: dict[int, int],
        f0: float = 50.0,
        dt_fast: float = 1.0,
        tau_default: float = 0.1,
        agc_ki: float = 0.05,
        cache_k_bins: int = 5,
        use_pseudoinverse: bool = True,
    ):
        """
        Args:
            placement: Dict with "gfm" key containing GFM specs.
            base_net: Base pandapower network (will be deep-copied per topology).
            bus_map: MATPOWER bus_id → pandapower bus index mapping.
            f0: Nominal frequency (Hz).
            dt_fast: Fast-step duration (s) for matrix exponential.
            tau_default: Default power filter time constant (s).
            agc_ki: AGC integral gain (pu/Hz/s).
            cache_k_bins: Number of K_droop bins for caching Φ.
            use_pseudoinverse: Use Moore-Penrose if B_LL singular.
        """
        self.f0 = f0
        self.omega0 = 2.0 * np.pi * f0
        self.dt_fast = dt_fast
        self.tau_default = tau_default
        self.agc_ki = agc_ki
        self.cache_k_bins = cache_k_bins
        self.use_pseudoinverse = use_pseudoinverse

        self._bus_map = bus_map
        self._base_net = base_net

        self._gfm_params: list[GFMParams] = []
        self._parse_gfm_placement(placement)
        self.n_gfm = len(self._gfm_params)

        if self.n_gfm < 2:
            raise ValueError(f"LTI freq dynamics requires ≥2 GFMs, got {self.n_gfm}")

        self._gfm_pp_idx = np.array([g.pp_bus_idx for g in self._gfm_params], dtype=int)
        self._gfm_ratings = np.array([g.rating_mva for g in self._gfm_params], dtype=float)
        self._tau_c = np.array([g.tau_filter for g in self._gfm_params], dtype=float)

        self._ref_idx = int(np.argmax(self._gfm_ratings))
        self._ref_gfm_id = self._gfm_params[self._ref_idx].gfm_id
        logger.info(f"LTI freq_dyn: reference GFM = {self._ref_gfm_id} (bus {self._gfm_params[self._ref_idx].bus_id}, {self._gfm_ratings[self._ref_idx]:.1f} MVA)")

        self._build_relative_angle_transform()

        self._n_state = (self.n_gfm - 1) + self.n_gfm

        self._x = np.zeros(self._n_state, dtype=float)
        self._x_prev = np.zeros(self._n_state, dtype=float)
        self._t = 0.0
        self._agc_integral = 0.0
        self._p_ref_pu = 0.0

        self._J_r: np.ndarray | None = None
        self._J_L: np.ndarray | None = None
        self._current_topology_id: int | None = None
        self._current_k_bin: int | None = None

        self._phi_cache: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}
        self._A_f_cache: dict[tuple[int, int], np.ndarray] = {}
        self._B_ref: np.ndarray | None = None
        self._B_net: np.ndarray | None = None

        self._singular_count = 0

    def _parse_gfm_placement(self, placement: dict[str, Any]) -> None:
        """Extract GFM parameters from placement dict."""
        gfm_dict = placement.get("gfm", {})
        for gfm_id, spec in gfm_dict.items():
            bus_id = int(spec.get("bus", 0))
            pp_idx = self._bus_map.get(bus_id, -1)
            if pp_idx < 0:
                logger.warning(f"GFM {gfm_id} bus {bus_id} not in bus_map, skipping")
                continue
            rating = float(spec.get("inverter_mva", spec.get("bess_mw", 1.0)))
            tau = float(spec.get("tau_filter", self.tau_default))
            mode = str(spec.get("mode", "Droop"))
            H_virt = float(spec.get("H_virt", 0.0))
            self._gfm_params.append(GFMParams(
                gfm_id=gfm_id,
                bus_id=bus_id,
                pp_bus_idx=pp_idx,
                rating_mva=rating,
                tau_filter=tau,
                H_virt=H_virt,
                mode=mode,
            ))
        self._gfm_params.sort(key=lambda g: g.pp_bus_idx)

    def _build_relative_angle_transform(self) -> None:
        """Build 1^{-1} matrix (n_gfm-1 × n_gfm) for relative angles."""
        n = self.n_gfm
        one_inv = np.zeros((n - 1, n), dtype=float)
        col = 0
        for i in range(n):
            if i == self._ref_idx:
                continue
            one_inv[col, i] = 1.0
            one_inv[col, self._ref_idx] = -1.0
            col += 1
        self._one_inv = one_inv
        self._non_ref_idx = [i for i in range(n) if i != self._ref_idx]

    def bind_operating_point(self, net: pp.pandapowerNet, topology_id: int = 0) -> None:
        """
        Extract Jacobian from converged power flow and Kron-reduce.

        Call this after pp.runpp(net) on a topology sample.
        """
        if not hasattr(net, "_ppc") or net._ppc is None:
            pp.runpp(net, algorithm="nr", init="auto", calculate_voltage_angles=True)

        ppc = net._ppc
        if ppc is None:
            raise RuntimeError("net._ppc is None after runpp; cannot extract Jacobian")

        # Translate pandas indices (from _bus_map) to row positions in J_full.
        # base_net.bus.index can have gaps (e.g., 0..449 with len=123 after MATPOWER
        # conversion); Y_bus and J_full are indexed by row position (0..n_bus-1).
        pandas_to_row = {int(p_idx): row for row, p_idx in enumerate(net.bus.index)}
        self._gfm_row_idx = np.array(
            [pandas_to_row.get(int(p_idx), -1) for p_idx in self._gfm_pp_idx],
            dtype=int,
        )

        J_full = self._build_jacobian(net, ppc)
        self._J_r, self._J_L = self._kron_reduce(J_full, net)
        self._current_topology_id = topology_id
        self._phi_cache.clear()
        self._A_f_cache.clear()
        self._build_input_matrices()

    def _build_jacobian(self, net: pp.pandapowerNet, ppc: dict) -> np.ndarray:
        """
        Build linearized AC power flow Jacobian ∂P/∂θ at operating point.

        Uses the admittance matrix Y_bus and operating voltages.
        For distribution feeders (high R/X), this retains loss sensitivity.
        """
        from pandapower.pypower.makeYbus import makeYbus

        bus = ppc["bus"]
        branch = ppc["branch"]
        baseMVA = ppc["baseMVA"]

        Y_bus, _, _ = makeYbus(baseMVA, bus, branch)
        if sp.issparse(Y_bus):
            Y_bus = Y_bus.toarray()
        Y_bus = np.asarray(Y_bus, dtype=complex)

        n_bus = Y_bus.shape[0]

        V_mag = bus[:, 7].copy()
        V_ang_rad = np.deg2rad(bus[:, 8])

        # Isolated buses (NaN from runpp) -> nominal V=1.0, angle=0
        nan_mask = ~np.isfinite(V_mag) | ~np.isfinite(V_ang_rad)
        if nan_mask.any():
            logger.debug(f"Replacing NaN at {int(nan_mask.sum())} isolated buses with V=1.0, angle=0")
            V_mag[nan_mask] = 1.0
            V_ang_rad[nan_mask] = 0.0

        V = V_mag * np.exp(1j * V_ang_rad)

        J_Ptheta = np.zeros((n_bus, n_bus), dtype=float)

        for i in range(n_bus):
            for j in range(n_bus):
                if i == j:
                    S_i = V[i] * np.conj(np.sum(Y_bus[i, :] * V))
                    J_Ptheta[i, i] = -np.imag(S_i) - (V_mag[i] ** 2) * np.imag(Y_bus[i, i])
                else:
                    Gij = np.real(Y_bus[i, j])
                    Bij = np.imag(Y_bus[i, j])
                    theta_ij = V_ang_rad[i] - V_ang_rad[j]
                    J_Ptheta[i, j] = V_mag[i] * V_mag[j] * (Gij * np.sin(theta_ij) - Bij * np.cos(theta_ij))

        return J_Ptheta

    def _kron_reduce(self, J_full: np.ndarray, net: pp.pandapowerNet) -> tuple[np.ndarray, np.ndarray]:
        """
        Kron-reduce Jacobian to dynamic (GFM) buses.

        Returns (J_r, J_L) where:
            J_r = J_II - J_IL @ inv(J_LL) @ J_LI
            J_L = J_IL @ inv(J_LL)
        """
        n_bus = J_full.shape[0]
        # Use row positions translated in bind_operating_point, falling back to
        # _gfm_pp_idx if bind_operating_point wasn't called (legacy path).
        gfm_indices = getattr(self, "_gfm_row_idx", self._gfm_pp_idx)
        dyn_mask = np.zeros(n_bus, dtype=bool)
        for row_idx in gfm_indices:
            if 0 <= row_idx < n_bus:
                dyn_mask[row_idx] = True

        dyn_idx = np.where(dyn_mask)[0]
        pas_idx = np.where(~dyn_mask)[0]

        J_II = J_full[np.ix_(dyn_idx, dyn_idx)]
        J_IL = J_full[np.ix_(dyn_idx, pas_idx)]
        J_LI = J_full[np.ix_(pas_idx, dyn_idx)]
        J_LL = J_full[np.ix_(pas_idx, pas_idx)]

        # Clean NaN/Inf before inversion
        if not np.isfinite(J_LL).all():
            logger.warning("J_LL contains NaN/Inf, replacing with zeros")
            J_LL = np.nan_to_num(J_LL, nan=0.0, posinf=0.0, neginf=0.0)

        # Add small regularization for numerical stability
        J_LL_reg = J_LL + np.eye(J_LL.shape[0]) * 1e-8

        try:
            if self.use_pseudoinverse:
                J_LL_inv = scipy.linalg.pinv(J_LL_reg, rtol=1e-6)
            else:
                J_LL_inv = np.linalg.inv(J_LL_reg)
        except (np.linalg.LinAlgError, ValueError):
            self._singular_count += 1
            logger.warning(f"J_LL singular (topology {self._current_topology_id}), using regularized pinv")
            try:
                J_LL_inv = scipy.linalg.pinv(J_LL_reg, rtol=1e-4)
            except (np.linalg.LinAlgError, ValueError):
                logger.warning("scipy.linalg.pinv failed, using identity fallback")
                J_LL_inv = np.eye(J_LL.shape[0]) * 1e-6

        J_r = J_II - J_IL @ J_LL_inv @ J_LI
        J_L = J_IL @ J_LL_inv

        gfm_indices = getattr(self, "_gfm_row_idx", self._gfm_pp_idx)
        gfm_order = np.argsort(gfm_indices)
        reorder = np.zeros(len(dyn_idx), dtype=int)
        for i, g_ord in enumerate(gfm_order):
            row_idx = gfm_indices[g_ord]
            pos_in_dyn = np.where(dyn_idx == row_idx)[0]
            if len(pos_in_dyn) > 0:
                reorder[i] = pos_in_dyn[0]

        J_r = J_r[np.ix_(reorder, reorder)]
        J_L = J_L[reorder, :]

        return J_r, J_L

    def _build_input_matrices(self) -> None:
        """Build B_ref and B_net input matrices."""
        n_omega = self.n_gfm
        T_c_inv = np.diag(1.0 / self._tau_c)

        self._B_ref = np.zeros((self._n_state, n_omega), dtype=float)
        self._B_ref[(self.n_gfm - 1):, :] = T_c_inv

        if self._J_L is not None:
            n_passive = self._J_L.shape[1]
            self._B_net = np.zeros((self._n_state, n_passive), dtype=float)
        else:
            self._B_net = np.zeros((self._n_state, 1), dtype=float)

    def _assemble_A_f(self, K_droop: np.ndarray) -> np.ndarray:
        """
        Assemble system matrix A_f for given droop gains.

        A_f = [[0, ω0 * 1^{-1}],
               [-T_c^{-1} M_p J̃_r, -T_c^{-1}]]

        where M_p = diag(1/K_droop_i) and J̃_r is J_r with reference column removed.
        """
        n_delta = self.n_gfm - 1
        n_omega = self.n_gfm

        A_f = np.zeros((self._n_state, self._n_state), dtype=float)

        A_f[:n_delta, n_delta:] = self.omega0 * self._one_inv

        T_c_inv = np.diag(1.0 / self._tau_c)

        K_safe = np.maximum(np.abs(K_droop), 1e-6)
        M_p = np.diag(1.0 / K_safe)

        if self._J_r is not None:
            J_r_tilde = self._J_r[:, self._non_ref_idx]
            A_f[n_delta:, :n_delta] = -T_c_inv @ M_p @ J_r_tilde
        else:
            A_f[n_delta:, :n_delta] = np.zeros((n_omega, n_delta))

        A_f[n_delta:, n_delta:] = -T_c_inv

        return A_f

    def _get_k_bin(self, K_droop: np.ndarray) -> int:
        """Quantize K_droop to bin index for caching."""
        k_avg = float(np.mean(np.abs(K_droop)))
        return int(np.clip(np.round(k_avg * self.cache_k_bins), 0, self.cache_k_bins * 5))

    def _get_phi(self, K_droop: np.ndarray, topology_id: int) -> tuple[np.ndarray, np.ndarray]:
        """Get cached or compute (Φ, M) where Φ = exp(A_f·dt), M = ∫₀^dt exp(A_f·τ)dτ.

        M is the ZOH input integral applied to u_input; using M avoids the
        forward-Euler overshoot when dt ≫ system time constants. Computed via
        the augmented exponential trick (robust to singular A_f):
            exp([[A_f, I], [0, 0]]·dt) = [[Φ, M], [0, I]]
        """
        k_bin = self._get_k_bin(K_droop)
        key = (topology_id, k_bin)

        if key in self._phi_cache:
            return self._phi_cache[key]

        A_f = self._assemble_A_f(K_droop)

        n = self._n_state
        aug = np.zeros((2 * n, 2 * n), dtype=float)
        aug[:n, :n] = A_f
        aug[:n, n:] = np.eye(n)
        aug_exp = scipy.linalg.expm(aug * self.dt_fast)
        Phi = aug_exp[:n, :n]
        M = aug_exp[:n, n:]

        self._phi_cache[key] = (Phi, M)
        self._A_f_cache[key] = A_f
        self._current_k_bin = k_bin

        return Phi, M

    def reset(self, f0: float = 50.0) -> None:
        """Reset state to equilibrium."""
        self.f0 = f0
        self.omega0 = 2.0 * np.pi * f0
        self._x.fill(0.0)
        self._x_prev.fill(0.0)
        self._t = 0.0
        self._agc_integral = 0.0
        self._p_ref_pu = 0.0

    def step(
        self,
        dt: float,
        delta_P_ref: np.ndarray,
        delta_P_L: float | np.ndarray,
        K_droop: np.ndarray,
        topology_id: int,
        ffr_active: bool = False,
    ) -> FrequencyStateLTI:
        """
        Integrate one fast-step using matrix exponential.

        Args:
            dt: Time step (s). Should equal self.dt_fast.
            delta_P_ref: Per-GFM power reference deviation (pu), shape (n_gfm,).
            delta_P_L: Passive bus disturbance (pu), scalar or vector.
            K_droop: Per-GFM droop gains, shape (n_gfm,).
            topology_id: Current topology cache index.
            ffr_active: Whether FFR is active (freezes AGC).

        Returns:
            FrequencyStateLTI with per-bus and scalar outputs.
        """
        self._x_prev = self._x.copy()

        Phi, M = self._get_phi(K_droop, topology_id)

        u_input = np.zeros(self._n_state, dtype=float)
        rating_total = float(np.sum(self._gfm_ratings))
        share = self._gfm_ratings / rating_total
        if self._B_ref is not None:
            M_p = np.diag(1.0 / np.maximum(np.abs(K_droop), 1e-6))
            T_c_inv = np.diag(1.0 / self._tau_c)

            # AGC closure: distribute previous-step integral as P_ref adjustment
            # proportional to rating share. Sign: under-frequency (Δf<0) makes
            # integral negative, so feedback (-integral·share) adds positive power
            # to restore f→50 Hz. Uses one-step-delayed integral to avoid algebraic
            # loop (integral updated below from this step's Δf).
            delta_P_ref_eff = np.asarray(delta_P_ref, dtype=float) - self._agc_integral * share
            u_input[(self.n_gfm - 1):] = T_c_inv @ M_p @ delta_P_ref_eff

            # Passive-bus disturbance: scalar imbalance distributed across GFMs by
            # rating share. Positive delta_P_L = generation deficit → Δω decreases.
            # TODO: replace with J_L-based per-passive-bus injection for full
            # topology-aware response (matches V-G derivation in bigupdate.md).
            dP_L = np.asarray(delta_P_L, dtype=float)
            dP_L_total = float(dP_L.sum()) if dP_L.ndim > 0 else float(dP_L)
            u_input[(self.n_gfm - 1):] -= T_c_inv @ M_p @ (share * dP_L_total)

        delta_omega = self._x[(self.n_gfm - 1):]
        delta_f_coi = float(np.sum(self._gfm_ratings * delta_omega) / np.sum(self._gfm_ratings)) * self.f0

        # Secondary control (AGC) runs CONCURRENTLY with primary droop/FFR — the
        # textbook two-loop design: primary responds fast and leaves a steady-state
        # offset, secondary slowly integrates the offset away. The previous code
        # froze AGC for the ENTIRE ffr_active window, which is correct only for a
        # one-shot imbalance: for a SUSTAINED disturbance (e.g. high_ren surplus)
        # where the policy cannot drive |Δf| below the FFR-deactivation threshold,
        # ffr_active stayed True forever, AGC never engaged, and the frequency got
        # stuck off-nominal (the S4 bimodal failure — confirmed: stuck topos had
        # ffr_active 29/30 steps & agc_integral ~0). AGC is slow (agc_ki small) and
        # anti-windup-clipped, so concurrent operation does not disturb the fast
        # primary transient. Always integrate.
        self._agc_integral += self.agc_ki * delta_f_coi * dt
        self._agc_integral = float(np.clip(self._agc_integral, -0.35, 0.35))

        self._p_ref_pu = float(np.mean(delta_P_ref))

        self._x = Phi @ self._x + M @ u_input

        self._t += dt

        return self.get_state()

    def nadir_safe_projection(
        self,
        delta_P_ref: np.ndarray,
        delta_P_L: float | np.ndarray,
        K_droop: np.ndarray,
        topology_id: int,
        delta_f_under: float = 0.5,
        delta_f_over: float = 0.5,
    ) -> tuple[np.ndarray, bool, float, float]:
        """Closed-form minimal-perturbation safety projection on delta_P_ref.

        Predicts the next-step COI Δf as an affine function of delta_P_ref using
        the SAME ZOH dynamics as step(), then — if the prediction would breach the
        nadir/zenith band [-delta_f_under, +delta_f_over] — Euclidean-projects
        delta_P_ref onto the active half-space (single linear constraint, so the
        projection is closed-form; no QP solver needed). This is the nadir
        "Safety Layer": minimal intervention, near-zero compute, runs in-the-loop.

        Δf_pred = a + bᵀ·ΔP_ref, with a,b built to mirror step() exactly
        (AGC one-step-delay + rating-share disturbance). Assumes the refinement
        does not hit the power cap (closed-form regime); downstream rating/reserve
        clipping handles the rare cap case.

        Returns (delta_P_ref_safe, activated, projection_distance, df_pred).
        """
        n = self.n_gfm
        dPref0 = np.asarray(delta_P_ref, dtype=float).copy()
        if self._B_ref is None:
            return dPref0, False, 0.0, 0.0

        Phi, M = self._get_phi(K_droop, topology_id)
        rating_total = float(np.sum(self._gfm_ratings))
        w = self._gfm_ratings / rating_total                       # COI weights
        share = w                                                  # rating share
        M_p = np.diag(1.0 / np.maximum(np.abs(K_droop), 1e-6))
        T_c_inv = np.diag(1.0 / self._tau_c)
        G = T_c_inv @ M_p                                          # ΔP_ref → u_omega
        dP_L = np.asarray(delta_P_L, dtype=float)
        dP_L_total = float(dP_L.sum()) if dP_L.ndim > 0 else float(dP_L)
        c_u = -(T_c_inv @ M_p @ share) * (self._agc_integral + dP_L_total)

        M_om_om = M[(n - 1):, (n - 1):]                            # omega-block of M
        Phi_x_om = (Phi @ self._x)[(n - 1):]
        a = self.f0 * float(w @ (Phi_x_om + M_om_om @ c_u))
        b = self.f0 * (G.T @ (M_om_om.T @ w))                      # ∂Δf_pred/∂ΔP_ref
        df_pred = a + float(b @ dPref0)

        activated = False
        bb = float(b @ b)
        if bb > 1e-12:
            if df_pred < -delta_f_under:
                dPref0 = dPref0 + b * ((-delta_f_under - df_pred) / bb)
                activated = True
            elif df_pred > delta_f_over:
                dPref0 = dPref0 + b * ((delta_f_over - df_pred) / bb)
                activated = True
        dist = float(np.linalg.norm(dPref0 - np.asarray(delta_P_ref, dtype=float)))
        return dPref0, activated, dist, df_pred

    def get_state(self) -> FrequencyStateLTI:
        """Return current frequency state."""
        delta_omega = self._x[(self.n_gfm - 1):]
        delta_f_per_bus = delta_omega * self.f0

        delta_omega_prev = self._x_prev[(self.n_gfm - 1):]
        delta_f_prev = delta_omega_prev * self.f0
        rocof_per_bus = (delta_f_per_bus - delta_f_prev) / max(self.dt_fast, 1e-6)

        delta_f_coi = float(np.sum(self._gfm_ratings * delta_f_per_bus) / np.sum(self._gfm_ratings))
        rocof_coi = float(np.sum(self._gfm_ratings * rocof_per_bus) / np.sum(self._gfm_ratings))

        delta_f_worst = float(np.max(np.abs(delta_f_per_bus)))
        rocof_worst = float(np.max(np.abs(rocof_per_bus)))

        return FrequencyStateLTI(
            t=self._t,
            f_hz=self.f0 + delta_f_coi,
            delta_f_hz=delta_f_coi,
            rocof_hz_s=rocof_coi,
            delta_f_per_bus=delta_f_per_bus.copy(),
            rocof_per_bus=rocof_per_bus.copy(),
            delta_f_worst=delta_f_worst,
            rocof_worst=rocof_worst,
            p_ref_pu=self._p_ref_pu,
            agc_integral_hz_s=self._agc_integral,
        )

    def get_gfm_bus_idx(self, agent_bus_pp_idx: int) -> int:
        """
        Map agent's pandapower bus index to GFM index.

        Returns GFM index if agent is at a GFM bus, else nearest GFM.
        """
        if agent_bus_pp_idx in self._gfm_pp_idx:
            return int(np.where(self._gfm_pp_idx == agent_bus_pp_idx)[0][0])
        distances = np.abs(self._gfm_pp_idx - agent_bus_pp_idx)
        return int(np.argmin(distances))

    @property
    def n_states(self) -> int:
        return self._n_state

    @property
    def gfm_bus_indices(self) -> np.ndarray:
        return self._gfm_pp_idx.copy()

    @property
    def gfm_ratings(self) -> np.ndarray:
        return self._gfm_ratings.copy()

    @property
    def reference_gfm_id(self) -> str:
        return self._ref_gfm_id

    @property
    def singular_count(self) -> int:
        return self._singular_count
