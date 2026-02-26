from __future__ import annotations

import inspect
from pathlib import Path

import pandas as pd
import pandapower as pp
import pytest

from src.env.IEEE123bus import _aggregate_power_elements
from src.layer0_dso import layer0_dso as layer0_module
from src.layer0_dso.layer0_dso import Layer0HourlyResult, export_layer0_csvs, run_layer0_dso, run_layer0_dso_hourly, run_layer0_pipeline
from src.layer0_dso.reconfiguration import _collect_loads
from src.layer0_dso.socp_validator import validate_socp_against_ac


def _tiny_net() -> pp.pandapowerNet:
    net = pp.create_empty_network()
    bus0 = pp.create_bus(net, vn_kv=20.0, name="0")
    bus1 = pp.create_bus(net, vn_kv=20.0, name="1")
    pp.create_ext_grid(net, bus=bus0, vm_pu=1.0)
    pp.create_line_from_parameters(
        net,
        from_bus=bus0,
        to_bus=bus1,
        length_km=1.0,
        r_ohm_per_km=0.2,
        x_ohm_per_km=0.4,
        c_nf_per_km=0.0,
        max_i_ka=0.4,
        name="l01",
    )
    pp.create_load(net, bus=bus1, p_mw=0.5, q_mvar=0.1)
    return net


def test_socp_validator_reports_publish_grade_metrics() -> None:
    net = _tiny_net()
    pp.runpp(net)

    socp_voltage_squared = {
        int(bus): float((net.res_bus.at[bus, "vm_pu"] * net.bus.at[bus, "vn_kv"]) ** 2)
        for bus in net.bus.index
    }

    result = validate_socp_against_ac(net, alpha_star={}, socp_voltage_squared=socp_voltage_squared, tolerance=1e-4)

    assert result.converged
    assert result.ac_valid
    assert result.socp_ac_gap_max <= 1e-4
    assert result.socp_ac_gap_p95 <= result.socp_ac_gap_max
    assert result.socp_ac_gap_p50 <= result.socp_ac_gap_p95
    assert result.compared_bus_count == len(net.bus)
    assert result.worst_bus is not None


def test_aggregate_sgen_keeps_type_split() -> None:
    net = pp.create_empty_network()
    bus = pp.create_bus(net, vn_kv=4.16, name="10")
    pp.create_sgen(net, bus=bus, p_mw=1.0, q_mvar=0.0, type="pv", name="pv_a")
    pp.create_sgen(net, bus=bus, p_mw=0.5, q_mvar=0.0, type="pv", name="pv_b")
    pp.create_sgen(net, bus=bus, p_mw=2.0, q_mvar=0.0, type="wind", name="w_a")

    _aggregate_power_elements(net, "sgen")

    assert set(net.sgen["type"].astype(str)) == {"pv", "wind"}
    pv_p = float(net.sgen.loc[net.sgen["type"] == "pv", "p_mw"].sum())
    wind_p = float(net.sgen.loc[net.sgen["type"] == "wind", "p_mw"].sum())
    assert pv_p == pytest.approx(1.5)
    assert wind_p == pytest.approx(2.0)


def test_collect_loads_includes_shunt_q() -> None:
    net = pp.create_empty_network()
    bus = pp.create_bus(net, vn_kv=4.16, name="11")
    pp.create_load(net, bus=bus, p_mw=1.0, q_mvar=0.3)
    pp.create_shunt(net, bus=bus, p_mw=0.0, q_mvar=-0.2)

    loads_p, loads_q, _ = _collect_loads(net)

    assert loads_p[int(bus)] == pytest.approx(1.0)
    assert loads_q[int(bus)] == pytest.approx(0.1)


def test_publish_defaults_are_strict_and_radial() -> None:
    sig_single = inspect.signature(run_layer0_dso)
    assert sig_single.parameters["enforce_radiality"].default is True
    assert sig_single.parameters["radiality_slack"].default == 0
    assert sig_single.parameters["ac_tolerance"].default == pytest.approx(0.01)

    sig_hourly = inspect.signature(run_layer0_dso_hourly)
    assert sig_hourly.parameters["enforce_radiality"].default is True
    assert sig_hourly.parameters["radiality_slack"].default == 0
    assert sig_hourly.parameters["ac_tolerance"].default == pytest.approx(0.01)

    sig_pipeline = inspect.signature(run_layer0_pipeline)
    assert sig_pipeline.parameters["ac_tolerance"].default == pytest.approx(0.01)


def test_export_fail_closed_outputs_diagnostics_only(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(layer0_module, "build_ieee123_net", lambda **kwargs: pp.create_empty_network())
    monkeypatch.setattr(layer0_module, "switch_edge_map", lambda _net: {})

    result = Layer0HourlyResult(
        day_label="peak",
        hour=0,
        lambda_dlmp={1: 1.0},
        alpha_star={10: 1},
        market_signals={"zone_1": {"energy_price": 100.0, "reserve_price": 5.0}},
        pricing_method="load_weighted",
        socp_ac_gap_max=0.2,
        socp_ac_gap_p95=0.15,
        socp_ac_gap_p50=0.1,
        socp_ac_worst_bus=12,
        socp_ac_compared_bus_count=50,
        ac_converged=True,
        ac_valid=False,
        soc_slack_max=1e-3,
        soc_slack_sum=1e-2,
        voltage_drop_slack_max=2e-3,
        voltage_drop_slack_sum=2e-2,
    )

    bundle = export_layer0_csvs(
        output_dir=tmp_path,
        hourly_results=[result],
        valid_for_layer1=False,
        diagnostics_csv=tmp_path / "layer0_diagnostics.csv",
    )

    assert not bundle.valid_for_layer1
    zone_df = pd.read_csv(bundle.zone_prices_csv)
    diag_df = pd.read_csv(bundle.diagnostics_csv)
    assert zone_df.empty
    assert not diag_df.empty
    assert diag_df["ac_valid"].eq(False).all()
