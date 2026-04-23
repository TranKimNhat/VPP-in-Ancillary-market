from __future__ import annotations

from typing import cast

import numpy as np
import pandapower as pp

from src.eval.harmonic_analysis import HarmonicAnalyzer


def _make_two_bus_resistive_net() -> pp.pandapowerNet:
    net = pp.create_empty_network(sn_mva=1.0)
    b0 = pp.create_bus(net, vn_kv=22.0)
    b1 = pp.create_bus(net, vn_kv=22.0)
    pp.create_ext_grid(net, bus=b0, vm_pu=1.0)
    pp.create_line_from_parameters(
        net,
        from_bus=b0,
        to_bus=b1,
        length_km=1.0,
        r_ohm_per_km=0.1,
        x_ohm_per_km=1e-4,
        c_nf_per_km=0.0,
        max_i_ka=1.0,
        in_service=True,
    )
    pp.create_load(net, bus=b1, p_mw=0.1, q_mvar=0.0)
    return net


def test_thd_i_unit_consistency_with_resistive_two_bus() -> None:
    net = _make_two_bus_resistive_net()
    pp.runpp(net, numba=False)

    # Make denominator deterministic and non-trivial.
    net.res_line.loc[:, "i_from_ka"] = 0.2

    analyzer = HarmonicAnalyzer(net)
    out = analyzer.run(
        agent_powers_mw=np.array([0.05], dtype=np.float64),
        agent_bus_idx=[1],
    )

    thd_i = np.asarray(out["THD_I_pct"], dtype=np.float64)
    assert thd_i.shape[0] == 1
    assert np.isfinite(thd_i[0])

    ppc = net._ppc
    assert ppc is not None
    base_mva = float(ppc["baseMVA"])
    branch = ppc["branch"]
    from_bus = int(branch[0, 0])
    to_bus = int(branch[0, 1])
    r = float(branch[0, 2])
    x = float(branch[0, 3])

    i_h_sq = 0.0
    for h in [58, 62, 119, 121]:
        y_h = analyzer._build_Yh(ppc, h)
        i_h_inj = analyzer._build_injection(
            agent_powers_mw=np.array([0.05], dtype=np.float64),
            agent_bus_idx=[1],
            n_bus=ppc["bus"].shape[0],
            h=h,
            baseMVA=base_mva,
        )
        try:
            v_h = np.linalg.solve(y_h, i_h_inj)
        except np.linalg.LinAlgError:
            v_h = np.linalg.lstsq(y_h, i_h_inj, rcond=None)[0]
        z_h = complex(r, x * h)
        i_h_pu = (v_h[from_bus] - v_h[to_bus]) / z_h
        v_base_kv = float(net.bus.at[from_bus, "vn_kv"])
        i_base_a = 1e6 * base_mva / (np.sqrt(3.0) * v_base_kv * 1000.0)
        i_h_a = abs(i_h_pu) * i_base_a
        i_h_sq += float(i_h_a**2)

    i1_a = 1000.0 * 0.2
    expected = np.sqrt(i_h_sq) / i1_a * 100.0

    rel_err = abs(float(thd_i[0]) - float(expected)) / max(abs(float(expected)), 1e-12)
    assert rel_err <= 0.01, f"THD_I mismatch: got={thd_i[0]:.6f}, expected={expected:.6f}, rel_err={rel_err:.4%}"


def test_thd_v_pcc_uses_configured_bus_index() -> None:
    net = _make_two_bus_resistive_net()
    pp.runpp(net, numba=False)

    analyzer = HarmonicAnalyzer(net)
    out = analyzer.run(
        agent_powers_mw=np.array([0.05], dtype=np.float64),
        agent_bus_idx=[1],
        pcc_bus_idx=1,
    )

    thd_v = np.asarray(out["THD_V_pct"], dtype=np.float64)
    thd_v_pcc = float(cast(float, out["THD_V_PCC"]))
    assert np.isfinite(thd_v_pcc)
    assert np.isclose(thd_v_pcc, thd_v[1], rtol=0.0, atol=0.0)


def test_build_injection_uses_per_bus_vn_kv_scaling() -> None:
    net = pp.create_empty_network(sn_mva=1.0)
    bus_hv = pp.create_bus(net, vn_kv=22.0)
    bus_lv = pp.create_bus(net, vn_kv=0.4)

    analyzer = HarmonicAnalyzer(net)
    i_h = analyzer._build_injection(
        agent_powers_mw=np.array([0.05, 0.05], dtype=np.float64),
        agent_bus_idx=[int(bus_hv), int(bus_lv)],
        n_bus=2,
        h=58,
        baseMVA=1.0,
    )

    assert np.isfinite(i_h[bus_hv])
    assert np.isfinite(i_h[bus_lv])
    assert np.isclose(i_h[bus_hv].imag, 0.0, rtol=0.0, atol=0.0)
    assert np.isclose(i_h[bus_lv].imag, 0.0, rtol=0.0, atol=0.0)

    ratio = float(i_h[bus_hv].real / i_h[bus_lv].real)
    expected_ratio = float(net.bus.at[bus_hv, "vn_kv"] / net.bus.at[bus_lv, "vn_kv"])
    assert np.isclose(ratio, expected_ratio, rtol=1e-12, atol=0.0)
