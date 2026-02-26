from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.layer0_dso.layer0_dso import run_layer0_pipeline
from src.layer1_vpp.layer1_vpp import Layer1Config, run_layer1


def test_layer0_layer1_pipeline_io(tmp_path: Path) -> None:
    layer0_dir = tmp_path / "layer0"
    layer0_bundle = run_layer0_pipeline(output_dir=layer0_dir)

    assert layer0_bundle.zone_prices_csv.exists()
    assert layer0_bundle.alpha_csv.exists()
    assert layer0_bundle.diagnostics_csv.exists()

    zone_df = pd.read_csv(layer0_bundle.zone_prices_csv)
    assert {"day", "hour", "zone", "energy_price"}.issubset(zone_df.columns)

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
