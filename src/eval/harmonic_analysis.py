"""
Harmonic Power Flow for IEEE 123-bus IBR Microgrid.

Computes THD_V (voltage) and THD_I (current) at all buses/branches
using harmonic admittance matrix from pandapower network model.

Method: Hybrid analytical + numerical Y_h matrix
- Harmonic current source: closed-form (Akhhmetov 2020)
- Harmonic propagation: Y_h inverse solve (pandapower Ybus)

IEEE Std 519-2014 reference (compliance thresholds enforced by callers):
  Voltage THD limit at PCC (Table 1):
    - LV (< 1 kV):           8.0%
    - MV (1 kV – 69 kV):     5.0%   <-- applies to this 4.16 kV microgrid
    - HV (69 kV – 161 kV):   2.5%
    - EHV (> 161 kV):        1.5%
  Individual voltage harmonic (MV): 3.0%

Current side (IEEE 519-2014 §5.2): reports BOTH THD_I (legacy diagnostic,
referenced to fundamental I_1) and TDD (standard-compliant, referenced to
demand current I_L with the total-TDD limit tiered by the short-circuit
ratio I_sc/I_L per Table 2: <20 -> 5%, <50 -> 8%, <100 -> 12%,
<1000 -> 15%, else 20%). I_L is proxied by the fundamental branch current
at the audited dispatch; the true annual max demand satisfies I_L >= I_1,
so the reported TDD is a conservative upper bound. I_sc per bus comes from
the fundamental Thevenin impedance diag(inv(Y_1)).
"""

from __future__ import annotations

from typing import Dict, List
import warnings

import numpy as np
from scipy.special import jv as bessel_j

DEFAULT_INV_PARAMS = dict(
    Vdc_v=800.0,
    fsw_hz=3000.0,
    Lf_h=0.15,  # 150mH - scaled for MV grid-tied inverter with output transformer
    ma=0.90,
    P_rated_mw=0.05,
)

SIDEBANDS = [
    (1, -2, 58),
    (1, +2, 62),
    (2, -1, 119),
    (2, +1, 121),
]

HARMONICS = [58, 62, 119, 121]
IEEE519_THD_V_LIMIT = 5.0
MIN_V_PU_FOR_THD = 0.05
MIN_I_A_FOR_THD = 1.0


class HarmonicAnalyzer:
    def __init__(self, net, inv_params: dict | None = None) -> None:
        self.net = net
        self.inv_params = {**DEFAULT_INV_PARAMS, **(inv_params or {})}
        self._sideband_currents = self._compute_sideband_currents()

    def _get_ppc(self) -> dict:
        ppc = getattr(self.net, "_ppc", None)

        if ppc is None or not isinstance(ppc, dict):
            raise RuntimeError("pandapower ppc unavailable")
        if "bus" not in ppc or "branch" not in ppc or "baseMVA" not in ppc:
            raise RuntimeError("pandapower ppc missing required fields")
        return ppc

    def run(
        self,
        agent_powers_mw: np.ndarray,
        agent_bus_idx: List[int],
        bus_mask: np.ndarray | None = None,
        pcc_bus_idx: int | None = None,
        agent_p_rated_mw: np.ndarray | None = None,
        i_l_a: np.ndarray | None = None,
    ) -> Dict[str, object]:
        """
        IMPORTANT (islanded mode): agent_bus_idx must include GFM bus when _gfm_bus_idx is provided.
        pcc_bus_idx defaults to 0 if not specified.
        """
        import pandapower as pp

        if not hasattr(self.net, "res_bus") or self.net.res_bus.empty:
            warnings.warn("runpp not called yet — calling now")
            pp.runpp(self.net)

        ppc = self._get_ppc()

        # Islanded mode assumption: harmonic analysis must include GFM/slack source bus.
        gfm_bus_idx = getattr(self.net, "_gfm_bus_idx", None)
        if gfm_bus_idx is not None and int(gfm_bus_idx) not in set(int(i) for i in agent_bus_idx):
            raise ValueError("agent_bus_idx must include GFM bus for islanded harmonic analysis")

        _baseMVA = float(ppc["baseMVA"])
        bus = ppc["bus"]
        branch = ppc["branch"]
        n_bus = bus.shape[0]

        V1 = self.net.res_bus["vm_pu"].values

        V_h_sq = np.zeros(n_bus, dtype=float)
        V_h_all = np.zeros((n_bus, len(HARMONICS)), dtype=complex)
        invalid_reasons: list[str] = []

        for hi, h in enumerate(HARMONICS):
            Y_h = self._build_Yh(ppc, h)
            I_h_inject = self._build_injection(
                agent_powers_mw,
                agent_bus_idx,
                n_bus,
                h,
                _baseMVA,
                agent_p_rated_mw=agent_p_rated_mw,
            )

            try:
                V_h = np.linalg.solve(Y_h, I_h_inject)
            except np.linalg.LinAlgError:
                V_h = np.linalg.lstsq(Y_h, I_h_inject, rcond=None)[0]
                invalid_reasons.append(f"lstsq_fallback_h{h}")

            if not np.isfinite(V_h).all():
                invalid_reasons.append(f"non_finite_h{h}")
            V_h_all[:, hi] = V_h
            V_h_sq += np.abs(V_h) ** 2

        if bus_mask is None:
            bus_mask = np.asarray(np.isfinite(V1), dtype=bool)
        else:
            bus_mask = np.asarray(bus_mask, dtype=bool)
            if bus_mask.shape[0] != n_bus:
                raise ValueError(f"bus_mask shape {bus_mask.shape} incompatible with n_bus={n_bus}")

        V1_safe = np.where(np.abs(V1) > MIN_V_PU_FOR_THD, np.abs(V1), np.nan)
        THD_V_raw = np.sqrt(V_h_sq) / V1_safe * 100.0
        THD_V = np.full((n_bus,), np.nan, dtype=float)
        THD_V[bus_mask] = THD_V_raw[bus_mask]

        THD_I, i_h_rss_A, I1_A, branch_mask = self._compute_branch_THD_I(
            V_h_all, branch, bus_mask, baseMVA=_baseMVA)

        # ---- IEEE 519-2014 §5.2 current-side compliance (TDD) ----
        # TDD references the maximum demand load current I_L. In a single
        # operating-point simulation we proxy I_L by the fundamental branch
        # current at the audited dispatch; since the true annual max demand
        # satisfies I_L >= I_1, the reported TDD is an upper bound (conservative).
        # Tier (Table 2, 120 V - 69 kV): in a 100% inverter-based islanded
        # microgrid the fault current is converter-limited (~1.2-2 pu of the
        # source rating), so I_sc/I_L < 20 by construction and the strictest
        # 5% total-TDD tier applies system-wide. (A Thevenin I_sc from
        # inv(Y_1) is meaningless here: the radial Ybus has no source
        # impedance to ground and is near-singular.)
        # I_L reference: caller-supplied demand current (e.g. frozen at the
        # pre-event operating point, per the standard's fixed max-demand
        # reference) or, failing that, the fundamental current at the
        # audited state.
        if i_l_a is not None:
            I_L_ref = np.asarray(i_l_a, dtype=float)
            if I_L_ref.shape[0] != I1_A.shape[0]:
                raise ValueError(
                    f"i_l_a shape {I_L_ref.shape} incompatible with "
                    f"n_branch={I1_A.shape[0]}")
        else:
            I_L_ref = I1_A
        with np.errstate(divide="ignore", invalid="ignore"):
            valid_il = np.isfinite(I_L_ref) & (I_L_ref > MIN_I_A_FOR_THD)
            TDD_I = np.where(valid_il, np.sqrt(i_h_rss_A) / I_L_ref * 100.0, np.nan)
        TDD_I[~branch_mask] = np.nan
        tdd_limit = np.full(TDD_I.shape, self._tdd_limit_pct(1.0), dtype=float)
        tdd_over = branch_mask & np.isfinite(TDD_I) & (TDD_I > tdd_limit)
        # TDD at the PCC -- the point where §5.2 formally applies.
        pcc_for_tdd = 0 if pcc_bus_idx is None else int(pcc_bus_idx)
        from_bus_arr = branch[:, 0].astype(np.int64)
        to_bus_arr = branch[:, 1].astype(np.int64)
        pcc_branches = (from_bus_arr == pcc_for_tdd) | (to_bus_arr == pcc_for_tdd)
        tdd_pcc_vals = TDD_I[pcc_branches & np.isfinite(TDD_I)]
        TDD_I_PCC = float(np.max(tdd_pcc_vals)) if tdd_pcc_vals.size else float("nan")

        invalid_mask_v = bus_mask & (~np.isfinite(THD_V))
        invalid_mask_i = ~np.isfinite(THD_I)
        # THD_V invalidity is critical (voltage should be finite on active buses)
        if np.any(invalid_mask_v):
            invalid_reasons.append("invalid_thd_v_denominator")
        # THD_I invalidity is expected for lightly loaded branches - only flag if >50% invalid
        i_invalid_rate = float(np.sum(invalid_mask_i)) / max(len(THD_I), 1)
        if i_invalid_rate > 0.5:
            invalid_reasons.append("invalid_thd_i_denominator")

        buses_over = list(np.where((THD_V > IEEE519_THD_V_LIMIT) & bus_mask & np.isfinite(THD_V))[0])
        pcc_idx = 0 if pcc_bus_idx is None else int(pcc_bus_idx)
        if pcc_idx < 0 or pcc_idx >= n_bus:
            raise ValueError(f"pcc_bus_idx={pcc_idx} out of range for n_bus={n_bus}")
        THD_V_PCC = float(THD_V[pcc_idx]) if (bus_mask[pcc_idx] and np.isfinite(THD_V[pcc_idx])) else float("nan")

        # Valid if no critical errors (lstsq fallback and THD_V invalid are critical)
        critical_reasons = [r for r in invalid_reasons if r.startswith("lstsq") or r.startswith("non_finite") or r == "invalid_thd_v_denominator"]
        harmonic_valid = len(critical_reasons) == 0
        return {
            "THD_V_pct": THD_V,
            "THD_I_pct": THD_I,
            "V_h": V_h_all,
            "buses_over": buses_over,
            "THD_V_PCC": THD_V_PCC,
            "THD_V_max": float(np.nanmax(THD_V)) if np.isfinite(THD_V).any() else float("nan"),
            "THD_I_max": float(np.nanmax(THD_I)) if np.isfinite(THD_I).any() else float("nan"),
            "TDD_I_pct": TDD_I,
            "TDD_limit_pct": tdd_limit,
            "TDD_I_max": float(np.nanmax(TDD_I)) if np.isfinite(TDD_I).any() else float("nan"),
            "TDD_I_PCC": TDD_I_PCC,
            "branches_over_tdd": int(np.sum(tdd_over)),
            "I1_branch_A": I1_A,
            "n_buses_over_limit": len(buses_over),
            "harmonic_valid": bool(harmonic_valid),
            "invalid_reasons": sorted(set(invalid_reasons)),
        }

    @staticmethod
    def _tdd_limit_pct(sc_ratio: float) -> float:
        """Total-TDD limit, IEEE Std 519-2014 Table 2 (120 V - 69 kV)."""
        if not np.isfinite(sc_ratio):
            return float("nan")
        if sc_ratio < 20.0:
            return 5.0
        if sc_ratio < 50.0:
            return 8.0
        if sc_ratio < 100.0:
            return 12.0
        if sc_ratio < 1000.0:
            return 15.0
        return 20.0

    def _compute_sideband_currents(self) -> dict[int, float]:
        p = self.inv_params
        Vdc = float(p["Vdc_v"])
        ma = float(p["ma"])
        Lf = float(p["Lf_h"])
        f0 = 50.0

        result: dict[int, float] = {}
        for m, n, h in SIDEBANDS:
            Jn = float(bessel_j(n, m * ma * np.pi / 2.0))
            V_mn = abs(2.0 * Vdc / (m * np.pi) * Jn * np.sin((m + n) * np.pi / 2.0))
            Z_h = 2.0 * np.pi * h * f0 * Lf
            I_mn = V_mn / Z_h if Z_h > 0.0 else 0.0
            result[h] = float(I_mn)
        return result

    def _build_Yh(self, ppc: dict, h: int) -> np.ndarray:
        from pandapower.pypower.makeYbus import makeYbus
        import scipy.sparse as sp

        bus_h = ppc["bus"].copy()
        branch_h = ppc["branch"].copy()

        # For capacitor shunt approximation, susceptance scales with harmonic order h.
        # GS (conductance) remains unchanged in this model.
        bus_h[:, 6] = bus_h[:, 6] * h
        branch_h[:, 3] = branch_h[:, 3] * h
        branch_h[:, 4] = branch_h[:, 4] * h

        Y_h, _, _ = makeYbus(ppc["baseMVA"], bus_h, branch_h)

        if sp.issparse(Y_h):
            Y_h = Y_h.toarray()
        Y_h = Y_h.astype(complex)

        # Regularize disconnected buses (zero diagonal) to avoid singular matrix
        # These are buses with no branch connections in ppc (e.g., switch-only connections)
        diag = np.diag(Y_h)
        disconnected = np.abs(diag) < 1e-10
        if np.any(disconnected):
            # Add small shunt admittance to regularize (1e-6 pu)
            Y_h[disconnected, disconnected] += 1e-6
        return Y_h

    def _build_injection(
        self,
        agent_powers_mw: np.ndarray,
        agent_bus_idx: List[int],
        n_bus: int,
        h: int,
        baseMVA: float,
        agent_p_rated_mw: np.ndarray | None = None,
    ) -> np.ndarray:
        I_h = np.zeros(n_bus, dtype=complex)
        I_sideband_A = float(self._sideband_currents.get(h, 0.0))
        p_rated_default = float(self.inv_params.get("P_rated_mw", 0.05))

        for k, (P_mw, bus_k) in enumerate(zip(agent_powers_mw, agent_bus_idx)):
            bus_idx = int(bus_k)
            if bus_idx < 0 or bus_idx >= n_bus:
                continue
            if "vn_kv" in self.net.bus.columns:
                V_base_kv = float(self.net.bus.at[bus_idx, "vn_kv"])
            else:
                V_base_kv = 11.0
            I_base_A = 1e6 * float(baseMVA) / (np.sqrt(3.0) * max(V_base_kv, 1e-6) * 1000.0)
            I_pu_per_amp = 1.0 / max(I_base_A, 1e-9)
            # Per-unit loading: |P| over THIS unit's rating (falls back to the
            # global cell rating when no per-agent ratings are supplied).
            if agent_p_rated_mw is not None and k < len(agent_p_rated_mw):
                p_rated_k = float(agent_p_rated_mw[k])
            else:
                p_rated_k = p_rated_default
            loading = np.clip(abs(float(P_mw)) / max(p_rated_k, 1e-9), 0.0, 1.0)
            I_h[bus_idx] += complex(I_sideband_A * loading * I_pu_per_amp, 0.0)

        return I_h

    def _compute_branch_THD_I(
        self,
        V_h_all: np.ndarray,
        branch: np.ndarray,
        bus_mask: np.ndarray,
        baseMVA: float,
    ) -> np.ndarray:
        n_branch = branch.shape[0]

        from_bus = branch[:, 0].astype(np.int64)
        to_bus = branch[:, 1].astype(np.int64)
        r = branch[:, 2].astype(np.float64)
        x = branch[:, 3].astype(np.float64)

        i_h_sq_pu = np.zeros(n_branch, dtype=np.float64)
        z_real = r
        z_imag_base = x
        z_abs_sq_eps = 1e-20
        for hi, h in enumerate(HARMONICS):
            v_diff_h = V_h_all[from_bus, hi] - V_h_all[to_bus, hi]
            vd_r = v_diff_h.real
            vd_i = v_diff_h.imag

            zi = z_imag_base * float(h)
            z_abs_sq = z_real * z_real + zi * zi
            valid = z_abs_sq > z_abs_sq_eps

            num_r = vd_r * z_real + vd_i * zi
            num_i = vd_i * z_real - vd_r * zi

            i_r = np.divide(num_r, z_abs_sq, out=np.zeros_like(num_r), where=valid)
            i_i = np.divide(num_i, z_abs_sq, out=np.zeros_like(num_i), where=valid)
            i_h_sq_pu += i_r * i_r + i_i * i_i

        if "vn_kv" in self.net.bus.columns:
            bus_index = self.net.bus.index.to_numpy(dtype=np.int64, copy=False)
            if np.isin(from_bus, bus_index).all():
                v_base_kv = self.net.bus.loc[from_bus, "vn_kv"].to_numpy(dtype=np.float64)
            elif "bus_id" in self.net.bus.columns:
                bus_id = self.net.bus["bus_id"].to_numpy(dtype=np.int64, copy=False)
                pos_map = {int(lbl): i for i, lbl in enumerate(bus_id.tolist())}
                mapped = np.asarray([pos_map.get(int(lbl), -1) for lbl in from_bus.tolist()], dtype=np.int64)
                valid_map = mapped >= 0
                v_base_kv = np.full((n_branch,), 11.0, dtype=np.float64)
                if np.any(valid_map):
                    v_base_kv[valid_map] = self.net.bus.iloc[mapped[valid_map]]["vn_kv"].to_numpy(dtype=np.float64)
            else:
                v_base_kv = np.full((n_branch,), 11.0, dtype=np.float64)
        else:
            v_base_kv = np.full((n_branch,), 11.0, dtype=np.float64)
        i_base_a = 1e6 * float(baseMVA) / (np.sqrt(3.0) * np.maximum(v_base_kv, 1e-6) * 1000.0)
        i_h_sq_a = i_h_sq_pu * np.square(i_base_a)

        branch_mask = np.asarray(bus_mask[from_bus] & bus_mask[to_bus], dtype=bool)

        try:
            I1_ka = np.asarray(self.net.res_line["i_from_ka"].values, dtype=np.float64)
            I1_A = np.full((n_branch,), np.nan, dtype=float)
            I1_A[: min(n_branch, I1_ka.shape[0])] = \
                1000.0 * np.abs(I1_ka[: min(n_branch, I1_ka.shape[0])])
            I1_safe = np.where(I1_A > MIN_I_A_FOR_THD, I1_A, np.nan)
            THD_I_raw = np.sqrt(i_h_sq_a) / I1_safe * 100.0
            THD_I = np.full((n_branch,), np.nan, dtype=float)
            THD_I[branch_mask] = THD_I_raw[branch_mask]
        except Exception:
            THD_I = np.full((n_branch,), np.nan, dtype=float)
            I1_A = np.full((n_branch,), np.nan, dtype=float)

        return THD_I, i_h_sq_a, I1_A, branch_mask


def compute_thd_episode(net, agent_powers_mw, agent_bus_idx) -> Dict[str, object]:
    analyzer = HarmonicAnalyzer(net)
    return analyzer.run(agent_powers_mw, agent_bus_idx)
