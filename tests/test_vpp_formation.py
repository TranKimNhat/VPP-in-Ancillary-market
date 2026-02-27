from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.layer0_dso.vpp_formation import VppFormationConfig, run_vpp_formation


def test_vpp_formation_generates_expected_artifacts(tmp_path: Path) -> None:
    bus_to_zone_csv = tmp_path / "bus_to_zone.csv"
    buses = list(range(0, 130))
    zones = [1 if i < 45 else 2 if i < 90 else 3 for i in buses]
    pd.DataFrame({"bus": buses, "zone_id": zones}).to_csv(bus_to_zone_csv, index=False)

    out_dir = tmp_path / "vpp_assignments"
    artifacts = run_vpp_formation(
        output_dir=out_dir,
        mapping_config={"bus_to_zone_csv": str(bus_to_zone_csv)},
        formation_config=VppFormationConfig(p_min_der_kw=5.0, p_min_vpp_kw=10.0, n_min_buses=1, n_max_buses=6),
    )

    assert artifacts.bus_to_vpp_csv.exists()
    assert artifacts.vpp_to_zone_csv.exists()
    assert artifacts.vpp_summary_csv.exists()

    bus_to_vpp = pd.read_csv(artifacts.bus_to_vpp_csv)
    vpp_to_zone = pd.read_csv(artifacts.vpp_to_zone_csv)
    vpp_summary = pd.read_csv(artifacts.vpp_summary_csv)

    assert {"bus", "zone_id", "vpp_id", "der_type", "p_cap_kw", "controllable", "vpp_role"}.issubset(bus_to_vpp.columns)
    assert {"vpp_id", "zone_id"}.issubset(vpp_to_zone.columns)
    assert {"vpp_id", "zone_id", "bus_count", "der_count", "load_mw", "der_mw", "buses"}.issubset(vpp_summary.columns)

    mapped_vpps = set(vpp_to_zone["vpp_id"].astype(str).tolist())
    assigned_vpps = set(bus_to_vpp["vpp_id"].dropna().astype(str).tolist())
    assert assigned_vpps.issubset(mapped_vpps)

    merged = bus_to_vpp.dropna(subset=["vpp_id"]).merge(vpp_to_zone, on="vpp_id", suffixes=("_bus", "_vpp"))
    assert (merged["zone_id_bus"].astype(int) == merged["zone_id_vpp"].astype(int)).all()
