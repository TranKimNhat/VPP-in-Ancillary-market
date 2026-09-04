"""
Topology-aware linearized frequency dynamics for all-GFM islanded microgrid.

This module implements the Kron-reduced state-space model described in the
reviewer response (Trujillo et al. framework adapted for 100% inverter-based
resources), with per-GFM Virtual-Synchronous-Generator (VSG) swing dynamics:
2H_i·dΔω_i/dt = ΔP_set,i − ΔP_e,i − D_i·Δω_i. The system matrix A_f depends on
topology G_t (via the reduced Jacobian J_r), the virtual inertia H_i, and the
learnable droop gains K_droop (realised as the swing damping D_i = K_i).

Key features vs the legacy scalar-COI SG model (removed):
  1. Per-bus frequency deviation (not scalar COI)
  2. Implicit active losses via AC Jacobian (R retained)
  3. Topology enters system matrix through J_r(G_t)
  4. Virtual inertia H_i gives a genuine inertial response (RoCoF = ΔP/2H);
     learnable droop K_droop_i enters as the swing damping D_i
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

    System (per-GFM VSG swing, D_i = K_droop_i):
        dx/dt = A_f(G_t, K_droop) x + B_ref ΔP_ref + B_net ΔP_L
    where:
        A_f = [[0, ω0 * 1^{-1}], [-diag(1/2H) J̃_r, -diag(1/2H) D]]
        B_ref|_ω = diag(1/2H)
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
        h_virt_vsg: float = 2.0,
        h_virt_floor: float = 0.5,
        s_base_mva: float = 15.7,
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
        self.h_virt_vsg = float(h_virt_vsg)
        self.h_virt_floor = float(h_virt_floor)
        self.s_base_mva = float(s_base_mva)

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
        # Virtual inertia per GFM (VSG swing): 2H·dΔω/dt = ΔP_set − ΔP_e − D·Δω.
        # H must be > 0 for all units (else 1/2H singular); floor enforced in parse.
        self._H_virt = np.array([g.H_virt for g in self._gfm_params], dtype=float)
        self._inv2H = 1.0 / (2.0 * self._H_virt)  # per-GFM input/coupling gain
        self._connected_mask = np.ones(self.n_gfm, dtype=bool)  # all connected by default

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
        self._pas_row_idx: np.ndarray | None = None
        self._ppidx_to_Jcol: dict[int, int] = {}
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
            # Virtual inertia: explicit spec wins; else default by mode. A VSG-mode
            # unit emulates inertia (h_virt_vsg); a Droop-mode unit still needs a
            # small positive H (floor) so the 2H·dΔω/dt swing stays well-posed —
            # in that limit it behaves as quasi-instantaneous droop.
            if "H_virt" in spec:
                H_virt = max(float(spec["H_virt"]), self.h_virt_floor)
            elif mode.upper() == "VSG":
                H_virt = self.h_virt_vsg
            else:
                H_virt = self.h_virt_floor
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

        # Translate pandas bus indices to row positions in Y_bus / J_full.
        # CRITICAL: pandapower fuses buses joined by closed switches / zero-impedance
        # branches in its internal ppc, so net.bus.index is NOT 1:1 with ppc rows --
        # a fused bus is collapsed onto a representative ppc node while its own leftover
        # row is left ISOLATED (NaN voltage, zero admittance). A naive positional map
        # (enumerate over net.bus.index) points such a bus -- notably the reference GFM,
        # which connects through a fused node -- at its isolated leftover row, zeroing
        # that GFM's Jacobian row/column. The result is a rank-deficient J~_r and a
        # marginal (~0) eigenvalue in A_f: a collective inter-machine mode that never
        # synchronizes within the evaluation window. Use pandapower's fusion-aware
        # lookup so every GFM/passive bus maps to its TRUE electrical ppc node.
        lookups = getattr(net, "_pd2ppc_lookups", None)
        bus_lookup = lookups.get("bus") if isinstance(lookups, dict) else None
        if bus_lookup is not None:
            def _row_of(p_idx: int) -> int:
                return int(bus_lookup[int(p_idx)])
        else:  # fallback: positional map (no fusion info available)
            _naive = {int(p_idx): row for row, p_idx in enumerate(net.bus.index)}
            def _row_of(p_idx: int) -> int:
                return int(_naive.get(int(p_idx), -1))

        self._gfm_row_idx = np.array(
            [_row_of(p_idx) for p_idx in self._gfm_pp_idx], dtype=int,
        )

        J_full = self._build_jacobian(net, ppc)
        self._J_r, self._J_L, pas_idx = self._kron_reduce(J_full, net)
        self._pas_row_idx = np.asarray(pas_idx, dtype=int)

        # Map pandapower bus index -> column in J_L for located disturbance injection.
        # J_L column c corresponds to passive ppc row pas_idx[c]; map each net bus to
        # its (fused) ppc row, then to that column. Fused buses share a column.
        ppc_row_to_col = {int(r): c for c, r in enumerate(self._pas_row_idx)}
        self._ppidx_to_Jcol = {}
        for p_idx in net.bus.index:
            r = _row_of(int(p_idx))
            if r in ppc_row_to_col:
                self._ppidx_to_Jcol[int(p_idx)] = ppc_row_to_col[r]

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

    def _kron_reduce(self, J_full: np.ndarray, net: pp.pandapowerNet) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Kron-reduce Jacobian to dynamic (GFM) buses.

        Returns (J_r, J_L, pas_idx) where:
            J_r = J_II - J_IL @ inv(J_LL) @ J_LI
            J_L = J_IL @ inv(J_LL)
            pas_idx = J_full row positions of the passive buses (J_L column order)
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

        # J_L rows are reordered to GFM order above; its COLUMNS stay in pas_idx
        # order (J_IL columns were taken in pas_idx order and J_LL_inv is square),
        # so pas_idx maps each J_L column back to a J_full row position. Returned
        # so the caller can build a pp-bus -> J_L-column map for located injection.
        return J_r, J_L, pas_idx

    def _build_input_matrices(self) -> None:
        """Build B_ref and B_net input matrices."""
        n_omega = self.n_gfm

        # Power reference enters the swing through the inertia gain diag(1/2H).
        self._B_ref = np.zeros((self._n_state, n_omega), dtype=float)
        self._B_ref[(self.n_gfm - 1):, :] = np.diag(self._inv2H)

        # Passive-bus disturbance map: a per-passive-bus injection ΔP_L_pas enters
        # the swing through diag(1/2H)·J_L (Kron sensitivity of GFM power mismatch
        # to passive-bus injections). J_L columns here sum to ≈ −1 (the Jacobian sign
        # convention), i.e. J_L plays the role of a NEGATED, spatially-distributed
        # rating share. So B_net = +diag(1/2H)·J_L makes a positive ΔP_L (generation
        # deficit) push Δω DOWN — matching the legacy scalar path's sign. B_net is
        # topology-only (no K dependence); rebuilt here per topology right after
        # _kron_reduce, so the Phi/M (k_bin) cache is unaffected.
        if self._J_L is not None:
            n_passive = self._J_L.shape[1]
            self._B_net = np.zeros((self._n_state, n_passive), dtype=float)
            self._B_net[(self.n_gfm - 1):, :] = self._inv2H[:, None] * self._J_L
        else:
            self._B_net = np.zeros((self._n_state, 1), dtype=float)

    def _assemble_A_f(self, K_droop: np.ndarray) -> np.ndarray:
        """
        Assemble system matrix A_f for given droop gains (VSG swing dynamics).

        Per-GFM swing equation: 2H_i·dΔω_i/dt = ΔP_set,i − ΔP_e,i − D_i·Δω_i,
        with ΔP_e = J̃_r·Δδ and damping D_i = K_droop_i (learned droop realised as
        swing damping, preserving the steady-state droop Δω_ss = ΔP/K). Hence:

        A_f = [[0,                    ω0 * 1^{-1}      ],
               [-diag(1/2H) J̃_r,     -diag(1/2H) D    ]]

        The immediate inertial response is RoCoF = ΔP/(2H), independent of K — the
        defining VSG behaviour. (Legacy first-order droop used -T_c^{-1} M_p J̃_r
        and -T_c^{-1}; replaced by the inertia-scaled blocks above.)
        """
        n_delta = self.n_gfm - 1
        n_omega = self.n_gfm

        A_f = np.zeros((self._n_state, self._n_state), dtype=float)

        A_f[:n_delta, n_delta:] = self.omega0 * self._one_inv

        inv2H = self._inv2H  # per-GFM 1/(2H_i), shape (n_gfm,)
        D = np.maximum(np.abs(K_droop), 1e-6)  # damping = droop gain

        if self._J_r is not None:
            J_r_tilde = self._J_r[:, self._non_ref_idx]
            A_f[n_delta:, :n_delta] = -(inv2H[:, None] * J_r_tilde)
        else:
            A_f[n_delta:, :n_delta] = np.zeros((n_omega, n_delta))

        A_f[n_delta:, n_delta:] = -np.diag(inv2H * D)

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

    def _disturbance_omega_forcing(
        self,
        delta_P_L: float | np.ndarray,
        event_location_pp: int | None = None,
    ) -> np.ndarray:
        """Omega-block forcing (shape (n_gfm,)) from the passive-bus disturbance.

        Four cases (plan §1), in priority order:
          1. event at a GFM bus -> inject directly into that GFM's swing (bypass J_L).
          2. event maps to a J_L column -> located one-hot injection via B_net.
          3. delta_P_L is a length-n_passive vector -> use directly via B_net.
          4. else (scalar / unmappable, e.g. line_trip) -> legacy rating-share scalar.
        Sign: positive delta_P_L = generation deficit -> Δω down (negative forcing).
        Shared by step() and nadir_safe_projection() so the in-the-loop guard
        predicts exactly what step() applies.
        """
        inv2H = self._inv2H
        share = self._gfm_ratings / float(np.sum(self._gfm_ratings))
        dP_L = np.asarray(delta_P_L, dtype=float)
        dP_L_total = float(dP_L.sum()) if dP_L.ndim > 0 else float(dP_L)
        B_net_om = self._B_net[(self.n_gfm - 1):, :] if self._B_net is not None else None

        # Case 1: event at a GFM bus -> direct swing injection.
        if event_location_pp is not None and event_location_pp in self._gfm_pp_idx:
            gfm_i = int(np.where(self._gfm_pp_idx == event_location_pp)[0][0])
            f = np.zeros(self.n_gfm, dtype=float)
            f[gfm_i] = -inv2H[gfm_i] * dP_L_total
            return f

        # Case 2: located passive-bus injection via J_L one-hot column.
        if (event_location_pp is not None and B_net_om is not None
                and event_location_pp in self._ppidx_to_Jcol):
            col = int(self._ppidx_to_Jcol[event_location_pp])
            return B_net_om[:, col] * dP_L_total

        # Case 3: explicit per-passive-bus vector.
        if (B_net_om is not None and dP_L.ndim == 1 and dP_L.size == B_net_om.shape[1]):
            return B_net_om @ dP_L

        # Case 4: legacy rating-share scalar (unlocated / line_trip / scalar).
        return -inv2H * (share * dP_L_total)

    def step(
        self,
        dt: float,
        delta_P_ref: np.ndarray,
        delta_P_L: float | np.ndarray,
        K_droop: np.ndarray,
        topology_id: int,
        ffr_active: bool = False,
        event_location_pp: int | None = None,
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
            event_location_pp: Pandapower bus index of the disturbance location.
                Routes the disturbance through J_L (located injection) when known;
                None falls back to the legacy rating-share scalar.

        Returns:
            FrequencyStateLTI with per-bus and scalar outputs.
        """
        self._x_prev = self._x.copy()

        Phi, M = self._get_phi(K_droop, topology_id)

        u_input = np.zeros(self._n_state, dtype=float)
        rating_total = float(np.sum(self._gfm_ratings))
        share = self._gfm_ratings / rating_total
        if self._B_ref is not None:
            inv2H = self._inv2H  # input enters the swing via diag(1/2H)

            # AGC closure: distribute previous-step integral as P_ref adjustment
            # proportional to rating share. Sign: under-frequency (Δf<0) makes
            # integral negative, so feedback (-integral·share) adds positive power
            # to restore f→50 Hz. Uses one-step-delayed integral to avoid algebraic
            # loop (integral updated below from this step's Δf).
            delta_P_ref_eff = np.asarray(delta_P_ref, dtype=float) - self._agc_integral * share
            u_input[(self.n_gfm - 1):] = inv2H * delta_P_ref_eff

            # Passive-bus disturbance: topology-aware per-bus injection via J_L when
            # the event location is known (located one-hot through B_net), falling
            # back to the legacy rating-share scalar otherwise. See
            # _disturbance_omega_forcing for the four resolution cases.
            u_input[(self.n_gfm - 1):] += self._disturbance_omega_forcing(
                delta_P_L, event_location_pp
            )

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

    def simulate_hires(
        self,
        dt: float,
        delta_P_ref: np.ndarray,
        delta_P_L: float | np.ndarray,
        K_droop: np.ndarray,
        topology_id: int,
        n_sub: int,
        event_location_pp: int | None = None,
    ) -> list[float]:
        """Non-destructive sub-step COI Δf trace over one fast-step (plotting only).

        step() advances the state with a single ZOH jump over dt (=dt_fast), which
        lands on the quasi-steady value and SKIPS the sub-second nadir transient.
        For figures we re-simulate the SAME fast-step at micro_dt = dt/n_sub using
        the group property exp(A_f·dt) = exp(A_f·micro_dt)^n_sub, so the trace's last
        sample coincides with step()'s post-state (identical up to round-off) while
        the intermediate samples reveal the true under-damped nadir.

        This is READ-ONLY: it copies self._x and reuses the current AGC integral,
        mutating nothing. step() still performs the real (training-identical)
        propagation; this only produces extra observation points. Returns n_sub COI
        Δf samples (Hz deviation) at spacing micro_dt.
        """
        if self._B_ref is None or int(n_sub) <= 1:
            return []
        micro_dt = float(dt) / int(n_sub)
        A_f = self._assemble_A_f(np.asarray(K_droop, dtype=float))
        n = self._n_state
        aug = np.zeros((2 * n, 2 * n), dtype=float)
        aug[:n, :n] = A_f
        aug[:n, n:] = np.eye(n)
        aug_exp = scipy.linalg.expm(aug * micro_dt)
        Phi_s, M_s = aug_exp[:n, :n], aug_exp[:n, n:]

        # Mirror step()'s u_input EXACTLY (same delta_P_ref_eff with one-step-delayed
        # AGC integral + located disturbance forcing).
        share = self._gfm_ratings / float(np.sum(self._gfm_ratings))
        u_input = np.zeros(self._n_state, dtype=float)
        delta_P_ref_eff = np.asarray(delta_P_ref, dtype=float) - self._agc_integral * share
        u_input[(self.n_gfm - 1):] = self._inv2H * delta_P_ref_eff
        u_input[(self.n_gfm - 1):] += self._disturbance_omega_forcing(
            delta_P_L, event_location_pp
        )

        x = self._x.copy()
        rating = self._gfm_ratings
        rating_sum = float(np.sum(rating))
        out: list[float] = []
        for _ in range(int(n_sub)):
            x = Phi_s @ x + M_s @ u_input
            delta_omega = x[(self.n_gfm - 1):]
            out.append(float(np.sum(rating * delta_omega) / rating_sum) * self.f0)
        return out

    def nadir_safe_projection(
        self,
        delta_P_ref: np.ndarray,
        delta_P_L: float | np.ndarray,
        K_droop: np.ndarray,
        topology_id: int,
        delta_f_under: float = 0.5,
        delta_f_over: float = 0.5,
        event_location_pp: int | None = None,
    ) -> tuple[np.ndarray, bool, float, float]:
        """Closed-form minimal-perturbation safety projection on delta_P_ref (COI).

        Predicts the next-step COI Δf as an affine function of delta_P_ref using the
        SAME ZOH dynamics + located disturbance forcing as step(); if the prediction
        would breach the band [-delta_f_under, +delta_f_over], Euclidean-projects
        delta_P_ref onto the active half-space (single linear constraint -> closed
        form, no QP solver). The COI is the system-frequency observable resolvable at
        the 1 s FFR/SFR control step (Anderson 1990); per-GFM inter-machine deviations
        are a sub-second phenomenon outside this control timescale, so the guard acts
        on the COI -- the quantity the controller is rewarded and evaluated on.

        Δf_pred = a + bᵀ·ΔP_ref, with a,b built to mirror step() (AGC one-step-delay +
        located disturbance forcing via _disturbance_omega_forcing).
        Returns (delta_P_ref_safe, activated, projection_distance, df_pred).
        """
        n = self.n_gfm
        dPref0 = np.asarray(delta_P_ref, dtype=float).copy()
        if self._B_ref is None:
            return dPref0, False, 0.0, 0.0

        Phi, M = self._get_phi(K_droop, topology_id)
        w = self._gfm_ratings / float(np.sum(self._gfm_ratings))   # COI (rating) weights
        G = np.diag(self._inv2H)                                   # ΔP_ref → u_omega
        # c_u: the ΔP_ref-independent part of u_omega — AGC one-step-delay + the SAME
        # located disturbance forcing step() applies (keeps guard ⇄ plant in lockstep).
        c_u = -(self._inv2H * w) * self._agc_integral \
            + self._disturbance_omega_forcing(delta_P_L, event_location_pp)

        M_om_om = M[(n - 1):, (n - 1):]                            # omega-block of M
        Phi_x_om = (Phi @ self._x)[(n - 1):]
        a = self.f0 * float(w @ (Phi_x_om + M_om_om @ c_u))
        b = self.f0 * (G.T @ (M_om_om.T @ w))                      # ∂Δf_coi/∂ΔP_ref
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

    def update_topology(self, connected_gfm_ids: set[str] | None = None) -> None:
        """Mark which GFMs are electrically connected (affects aggregate h_sys).

        When a tie-switch
        reconfiguration isolates a GFM, it no longer contributes virtual inertia.
        Kron reduction already handles the network coupling; this only gates the
        rating-weighted inertia sum exposed via the h_sys property.
        """
        if connected_gfm_ids is None:
            self._connected_mask = np.ones(self.n_gfm, dtype=bool)
            return
        self._connected_mask = np.array(
            [g.gfm_id in connected_gfm_ids for g in self._gfm_params], dtype=bool
        )

    @property
    def h_sys(self) -> float:
        """Aggregate system inertia constant H_sys = Σ(H_i·S_i)/S_BASE [s].

        Rating-weighted over connected GFMs only. This is the virtual inertia the
        VSG fleet emulates — there is no rotating mass in a 100% IBR microgrid —
        and is consumed by the MPC battery correction (evcs_model.mpc_correction).
        """
        mask = self._connected_mask
        hs = float(np.sum(self._H_virt[mask] * self._gfm_ratings[mask]) / self.s_base_mva)
        return hs

    @property
    def reference_gfm_id(self) -> str:
        return self._ref_gfm_id

    @property
    def singular_count(self) -> int:
        return self._singular_count
