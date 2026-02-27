from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.env.IEEE123bus import build_ieee123_net
from src.layer0_dso.layer0_dso import run_layer0_pipeline
from src.layer1_vpp.layer1_vpp import Layer1Config, run_layer1


def test_layer0_layer1_pipeline_io(tmp_path: Path) -> None:
    net = build_ieee123_net(mode="feeder123", balanced=True, convert_switches=True, slack_zones=None)
    bus_to_zone_csv = tmp_path / "bus_to_zone.csv"
    vpp_to_zone_csv = tmp_path / "vpp_to_zone.csv"
    pd.DataFrame({"bus": [int(b) for b in net.bus.index], "zone_id": [1] * len(net.bus.index)}).to_csv(bus_to_zone_csv, index=False)
    pd.DataFrame(columns=["vpp_id", "zone_id"]).to_csv(vpp_to_zone_csv, index=False)

    layer0_dir = tmp_path / "layer0"
    layer0_bundle = run_layer0_pipeline(
        output_dir=layer0_dir,
        mapping_config={
            "bus_to_zone_csv": str(bus_to_zone_csv),
            "vpp_to_zone_csv": str(vpp_to_zone_csv),
        },
    )

    assert layer0_bundle.zone_prices_csv.exists()
    assert layer0_bundle.alpha_csv.exists()
    assert layer0_bundle.diagnostics_csv.exists()

    zone_df = pd.read_csv(layer0_bundle.zone_prices_csv)
    assert {"day", "hour", "zone", "zone_id", "energy_price"}.issubset(zone_df.columns)

    diagnostics_df = pd.read_csv(layer0_bundle.diagnostics_csv)
    assert {"day", "hour", "socp_ac_gap_max", "ac_valid"}.issubset(diagnostics_df.columns)

    if not layer0_bundle.valid_for_layer1:
        assert zone_df.empty
        assert not diagnostics_df.empty
        assert diagnostics_df["ac_valid"].eq(False).any()
        return

    layer1_out = tmp_path / "layer1_pref.csv"
    cfg = Layer1Config(
        input_csv=layer0_bundle.zone_prices_csv,
        output_csv=layer1_out,
        weights={"offpeak": 0.5, "median": 0.3, "peak": 0.2},
        sign="inject",
        wasserstein_radius=0.02,
        degradation_cost=1.0,
    )
    out_path = run_layer1(cfg)
    assert out_path.exists()

    df = pd.read_csv(out_path)
    assert {"hour", "P_ref", "R_commit"}.issubset(df.columns)


def test_layer1_vpp_mode_dual_output(tmp_path: Path) -> None:
    layer0_csv = tmp_path / "layer0_zone_prices.csv"
    rows = []
    for day in ["offpeak", "median", "peak"]:
        for hour in range(4):
            rows.append({"day": day, "hour": hour, "zone_id": "1", "energy_price": 10.0 + hour, "reserve_price": 1.0})
            rows.append({"day": day, "hour": hour, "zone_id": "2", "energy_price": 20.0 + hour, "reserve_price": 2.0})
    pd.DataFrame(rows).to_csv(layer0_csv, index=False)

    vpp_to_zone_csv = tmp_path / "vpp_to_zone.csv"
    pd.DataFrame(
        {
            "vpp_id": ["vpp_1a", "vpp_1b", "vpp_2a"],
            "zone_id": ["1", "1", "2"],
        }
    ).to_csv(vpp_to_zone_csv, index=False)

    out_long = tmp_path / "layer1_vpp_long.csv"
    out_legacy = tmp_path / "layer1_vpp_legacy.csv"

    cfg = Layer1Config(
        input_csv=layer0_csv,
        output_csv=out_long,
        weights={"offpeak": 0.5, "median": 0.3, "peak": 0.2},
        sign="inject",
        wasserstein_radius=0.02,
        degradation_cost=1.0,
        vpp_mode=True,
        mapping_vpp_to_zone_csv=vpp_to_zone_csv,
        legacy_output_csv=out_legacy,
    )
    out_path = run_layer1(cfg)
    assert out_path.exists()
    assert out_legacy.exists()

    long_df = pd.read_csv(out_path)
    legacy_df = pd.read_csv(out_legacy)

    assert {"day", "hour", "zone_id", "vpp_id", "vpp_bus_count", "P_ref", "R_commit"}.issubset(long_df.columns)
    assert set(long_df["vpp_id"].unique()) == {"vpp_1a", "vpp_1b", "vpp_2a"}
    assert long_df["vpp_bus_count"].ge(0).all()
    assert {"hour", "P_ref", "R_commit"}.issubset(legacy_df.columns)
