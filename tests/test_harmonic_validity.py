from __future__ import annotations

from typing import cast

import numpy as np
import pandapower as pp

from src.eval.harmonic_analysis import HarmonicAnalyzer


def _make_small_net() -> pp.pandapowerNet:
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
        x_ohm_per_km=0.1,
        c_nf_per_km=0.0,
        max_i_ka=1.0,
        in_service=True,
    )
    pp.create_load(net, bus=b1, p_mw=0.1, q_mvar=0.02)
    return net


def test_harmonic_run_flags_invalid_when_denominator_not_finite() -> None:
    net = _make_small_net()
    pp.runpp(net, numba=False)
    net.res_bus.loc[:, "vm_pu"] = np.nan

    analyzer = HarmonicAnalyzer(net)
    out = analyzer.run(
        agent_powers_mw=np.array([0.05], dtype=np.float64),
        agent_bus_idx=[1],
        bus_mask=np.array([True, True], dtype=bool),
    )

    reasons = cast(list[str], out["invalid_reasons"])
    thd_v_pcc = cast(float, out["THD_V_PCC"])
    assert out["harmonic_valid"] is False
    assert "invalid_thd_v_denominator" in reasons
    assert np.isnan(thd_v_pcc)


def test_harmonic_run_returns_valid_shapes() -> None:
    net = _make_small_net()
    pp.runpp(net, numba=False)

    analyzer = HarmonicAnalyzer(net)
    out = analyzer.run(
        agent_powers_mw=np.array([0.05], dtype=np.float64),
        agent_bus_idx=[1],
    )

    thd_v = cast(np.ndarray, out["THD_V_pct"])
    thd_i = cast(np.ndarray, out["THD_I_pct"])
    assert thd_v.shape[0] == len(net.bus)
    assert thd_i.shape[0] == len(net.line)
    assert "harmonic_valid" in out
    assert "invalid_reasons" in out
