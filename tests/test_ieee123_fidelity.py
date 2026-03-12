from __future__ import annotations

import numpy as np
import pandapower as pp

from src.env.IEEE123bus import build_ieee123_net, validate_ieee123_net


def _build_matpower_publish_net(convert_switches: bool) -> pp.pandapowerNet:
    return build_ieee123_net(
        mode="matpower",
        balanced=True,
        convert_switches=convert_switches,
        slack_zones=None,
        source_mode="publish",
    )


def _runpp(net: pp.pandapowerNet) -> None:
    pp.runpp(net, init="flat", max_iteration=100, tolerance_mva=1e-6)
    assert net.converged


def _near_zero_line_count(net: pp.pandapowerNet) -> int:
    mask = (net.line["r_ohm_per_km"].abs() <= 2e-7) & (net.line["x_ohm_per_km"].abs() <= 2e-6)
    return int(mask.sum())


def test_matpower_publish_counts_baseline() -> None:
    net = _build_matpower_publish_net(convert_switches=True)

    assert len(net.bus) == 123
    assert len(net.line) == 114
    assert len(net.load) == 87
    assert len(net.sgen) == 31
    assert len(net.switch) == 5
    assert len(net.ext_grid) == 1


def test_publish_mode_slack_bus_is_114() -> None:
    net = _build_matpower_publish_net(convert_switches=True)

    assert len(net.ext_grid) == 1
    slack_bus_idx = int(net.ext_grid.iloc[0]["bus"])
    slack_bus_name = str(net.bus.at[slack_bus_idx, "name"])
    assert slack_bus_name == "114"


def test_near_zero_branches_present_when_not_converted() -> None:
    net = _build_matpower_publish_net(convert_switches=False)

    assert _near_zero_line_count(net) > 0


def test_validate_topology_loads_connected_to_slack_component() -> None:
    net = _build_matpower_publish_net(convert_switches=True)

    summary = validate_ieee123_net(net)
    assert summary["disconnected_load_buses"] == 0


def test_switch_conversion_reduces_or_preserves_near_zero_line_count() -> None:
    net_before = _build_matpower_publish_net(convert_switches=False)
    net_after = _build_matpower_publish_net(convert_switches=True)

    near_zero_before = _near_zero_line_count(net_before)
    near_zero_after = _near_zero_line_count(net_after)

    assert near_zero_after <= near_zero_before
    assert near_zero_before > 0


def test_powerflow_converges_and_has_finite_results() -> None:
    net = _build_matpower_publish_net(convert_switches=True)

    _runpp(net)

    load_buses = net.load["bus"].astype(int).unique().tolist()
    slack_buses = net.ext_grid["bus"].astype(int).unique().tolist()
    buses_to_check = sorted(set(load_buses + slack_buses))

    assert np.isfinite(net.res_bus.loc[buses_to_check, "vm_pu"]).all()
    assert np.isfinite(net.res_line["loading_percent"]).all()


def test_switch_conversion_electrical_equivalence_envelope() -> None:
    net_no_convert = _build_matpower_publish_net(convert_switches=False)
    net_convert = _build_matpower_publish_net(convert_switches=True)

    _runpp(net_no_convert)
    _runpp(net_convert)

    vm_no_convert = net_no_convert.res_bus[["vm_pu"]].copy()
    vm_no_convert["name"] = net_no_convert.bus["name"].astype(str).values

    vm_convert = net_convert.res_bus[["vm_pu"]].copy()
    vm_convert["name"] = net_convert.bus["name"].astype(str).values

    merged = vm_no_convert.merge(vm_convert, on="name", suffixes=("_no_convert", "_convert"))
    max_vm_delta = float((merged["vm_pu_no_convert"] - merged["vm_pu_convert"]).abs().max())

    assert max_vm_delta < 1e-2

    loss_no_convert = float(net_no_convert.res_line["pl_mw"].sum())
    loss_convert = float(net_convert.res_line["pl_mw"].sum())
    rel_loss_delta = abs(loss_no_convert - loss_convert) / (abs(loss_no_convert) + 1e-12)

    assert rel_loss_delta < 5e-2
