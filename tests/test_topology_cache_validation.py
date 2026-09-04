from __future__ import annotations

from copy import deepcopy

import numpy as np
import pandapower as pp

from src.opt.tie_switch_reconfig import TieSwitchReconfiguration


def _make_two_bus_connected() -> pp.pandapowerNet:
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


def test_sanitize_cache_drops_disconnected_entry() -> None:
    net_ok = _make_two_bus_connected()
    net_bad = deepcopy(net_ok)
    net_bad.line.loc[:, "in_service"] = False

    reconf = TieSwitchReconfiguration(net_ok, seed=42)
    edge_ok = reconf._build_edge_index(net_ok)
    edge_bad = reconf._build_edge_index(net_bad)

    cleaned, dropped = reconf._sanitize_cache_entries(
        [
            (net_ok, edge_ok, set()),
            (net_bad, edge_bad, set()),
        ]
    )

    assert dropped == 1
    assert len(cleaned) == 1
    assert reconf._is_topology_valid(deepcopy(cleaned[0][0]), run_power_flow=True)


def test_load_cache_accepts_legacy_list_and_sanitizes(tmp_path) -> None:
    net_ok = _make_two_bus_connected()
    net_bad = deepcopy(net_ok)
    net_bad.line.loc[:, "in_service"] = False

    reconf = TieSwitchReconfiguration(net_ok, seed=1)
    payload = [
        (net_ok, reconf._build_edge_index(net_ok), set()),
        (net_bad, reconf._build_edge_index(net_bad), set()),
    ]

    cache_path = tmp_path / "legacy_cache.pkl"
    import pickle

    with cache_path.open("wb") as f:
        pickle.dump(payload, f)

    loaded = reconf.load_cache(cache_path)
    assert loaded is True
    assert len(reconf._cache) == 1
    net_loaded, edge_loaded, _ = reconf._cache[0]
    assert reconf._is_topology_valid(deepcopy(net_loaded), run_power_flow=True)
    assert isinstance(edge_loaded, np.ndarray)
