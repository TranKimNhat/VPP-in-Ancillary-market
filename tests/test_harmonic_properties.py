from __future__ import annotations

from typing import cast

import numpy as np
import pandapower as pp

from src.eval.harmonic_analysis import HARMONICS, HarmonicAnalyzer


def _make_three_bus_net() -> pp.pandapowerNet:
    net = pp.create_empty_network(sn_mva=1.0)
    b0 = pp.create_bus(net, vn_kv=22.0)
    b1 = pp.create_bus(net, vn_kv=22.0)
    b2 = pp.create_bus(net, vn_kv=22.0)
    pp.create_ext_grid(net, bus=b0, vm_pu=1.0)
    pp.create_line_from_parameters(
        net,
        from_bus=b0,
        to_bus=b1,
        length_km=1.0,
        r_ohm_per_km=0.12,
        x_ohm_per_km=0.05,
        c_nf_per_km=0.0,
        max_i_ka=1.0,
        in_service=True,
    )
    pp.create_line_from_parameters(
        net,
        from_bus=b1,
        to_bus=b2,
        length_km=1.0,
        r_ohm_per_km=0.08,
        x_ohm_per_km=0.04,
        c_nf_per_km=0.0,
        max_i_ka=1.0,
        in_service=True,
    )
    pp.create_load(net, bus=b1, p_mw=0.08, q_mvar=0.01)
    pp.create_load(net, bus=b2, p_mw=0.06, q_mvar=0.01)
    return net


def _legacy_branch_thd_i(
    analyzer: HarmonicAnalyzer,
    v_h_all: np.ndarray,
    branch: np.ndarray,
    bus_mask: np.ndarray,
    base_mva: float,
) -> np.ndarray:
    n_branch = branch.shape[0]
    from_bus = branch[:, 0].astype(np.int64)
    to_bus = branch[:, 1].astype(np.int64)
    r = branch[:, 2].astype(np.float64)
    x = branch[:, 3].astype(np.float64)

    v_from_h = v_h_all[from_bus, :]
    v_to_h = v_h_all[to_bus, :]
    v_diff = v_from_h - v_to_h

    h_vec = np.asarray(HARMONICS, dtype=np.float64)[None, :]
    z_h = r[:, None] + 1j * (x[:, None] * h_vec)
    valid_z = np.abs(z_h) > 1e-10

    i_h_br_pu = np.zeros_like(v_diff, dtype=np.complex128)
    i_h_br_pu[valid_z] = v_diff[valid_z] / z_h[valid_z]

    if "vn_kv" in analyzer.net.bus.columns:
        v_base_kv = analyzer.net.bus.loc[from_bus, "vn_kv"].to_numpy(dtype=np.float64)
    else:
        v_base_kv = np.full((n_branch,), 11.0, dtype=np.float64)
    i_base_a = 1e6 * float(base_mva) / (np.sqrt(3.0) * np.maximum(v_base_kv, 1e-6) * 1000.0)
    i_h_br_a = np.abs(i_h_br_pu) * i_base_a[:, None]
    i_h_sq = np.sum(i_h_br_a**2, axis=1)

    branch_mask = np.asarray(
        [
            bool(bus_mask[int(branch[bi, 0])]) and bool(bus_mask[int(branch[bi, 1])])
            for bi in range(n_branch)
        ],
        dtype=bool,
    )

    i1_ka = np.asarray(analyzer.net.res_line["i_from_ka"].values, dtype=np.float64)
    i1_a = 1000.0 * np.abs(i1_ka)
    i1_safe = np.where(i1_a > 1.0, i1_a, np.nan)
    thd_i_raw = np.sqrt(i_h_sq) / i1_safe * 100.0
    thd_i = np.full((n_branch,), np.nan, dtype=float)
    thd_i[branch_mask] = thd_i_raw[branch_mask]
    return thd_i


def test_branch_thd_i_vectorized_matches_legacy_reference() -> None:
    rng = np.random.default_rng(123)
    net = _make_three_bus_net()
    pp.runpp(net, numba=False)
    net.res_line.loc[:, "i_from_ka"] = np.array([0.16, 0.21], dtype=np.float64)

    analyzer = HarmonicAnalyzer(net)
    out = analyzer.run(
        agent_powers_mw=np.array([0.04, 0.03], dtype=np.float64),
        agent_bus_idx=[1, 2],
    )

    ppc = analyzer._get_ppc()
    branch = cast(np.ndarray, ppc["branch"])
    base_mva = float(ppc["baseMVA"])
    v_h_all = cast(np.ndarray, out["V_h"])

    bus_mask = rng.choice([False, True], size=v_h_all.shape[0], replace=True)
    ref = _legacy_branch_thd_i(analyzer, v_h_all, branch, bus_mask, base_mva)
    got = analyzer._compute_branch_THD_I(v_h_all, branch, bus_mask, baseMVA=base_mva)

    assert np.allclose(got, ref, rtol=1e-10, atol=1e-12, equal_nan=True)


def test_branch_thd_i_returns_nan_when_branch_mask_false() -> None:
    net = _make_three_bus_net()
    pp.runpp(net, numba=False)
    net.res_line.loc[:, "i_from_ka"] = np.array([0.18, 0.19], dtype=np.float64)

    analyzer = HarmonicAnalyzer(net)
    out = analyzer.run(
        agent_powers_mw=np.array([0.04, 0.03], dtype=np.float64),
        agent_bus_idx=[1, 2],
    )

    ppc = analyzer._get_ppc()
    branch = cast(np.ndarray, ppc["branch"])
    base_mva = float(ppc["baseMVA"])
    v_h_all = cast(np.ndarray, out["V_h"])

    bus_mask = np.array([True, False, False], dtype=bool)
    got = analyzer._compute_branch_THD_I(v_h_all, branch, bus_mask, baseMVA=base_mva)
    assert np.isnan(got).all()
