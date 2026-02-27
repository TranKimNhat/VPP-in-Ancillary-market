from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping
import warnings

import pandas as pd
import pandapower as pp


@dataclass(frozen=True)
class CanonicalMappings:
    bus_to_zone: dict[int, int]
    bus_to_vpp: dict[int, str | None]
    vpp_to_zone: dict[str, int]
    der_to_vpp: dict[int, str | None]
    legacy_mode: bool


@dataclass(frozen=True)
class MappingSummary:
    rows: list[dict[str, object]]


def _resolve_path(path_like: str | Path | None) -> Path | None:
    if not path_like:
        return None
    path = Path(path_like)
    if not path.is_absolute():
        path = (Path(__file__).resolve().parents[2] / path).resolve()
    return path


def _legacy_bus_to_zone(net: pp.pandapowerNet, zone_bus_map: Mapping[int, list[str]] | None) -> dict[int, int]:
    bus_name_to_zone: dict[str, int] = {}
    if zone_bus_map is not None:
        for zone, names in zone_bus_map.items():
            for name in names:
                bus_name_to_zone[str(name)] = int(zone)

    zone_map: dict[int, int] = {}
    for bus_idx, row in net.bus.iterrows():
        bus_name = str(row.get("name", ""))
        zone_map[int(bus_idx)] = int(bus_name_to_zone.get(bus_name, 1))
    return zone_map


def _read_bus_zone_csv(path: Path) -> dict[int, int]:
    df = pd.read_csv(path)
    required = {"bus", "zone_id"}
    if not required.issubset(df.columns):
        raise ValueError(f"{path} must contain columns {sorted(required)}")
    return {int(row.bus): int(row.zone_id) for row in df.itertuples(index=False)}


def _read_vpp_zone_csv(path: Path) -> dict[str, int]:
    df = pd.read_csv(path)
    required = {"vpp_id", "zone_id"}
    if not required.issubset(df.columns):
        raise ValueError(f"{path} must contain columns {sorted(required)}")
    return {str(row.vpp_id): int(row.zone_id) for row in df.itertuples(index=False)}


def _read_bus_vpp_csv(path: Path) -> dict[int, str | None]:
    df = pd.read_csv(path)
    required = {"bus", "vpp_id"}
    if not required.issubset(df.columns):
        raise ValueError(f"{path} must contain columns {sorted(required)}")

    mapping: dict[int, str | None] = {}
    for row in df.itertuples(index=False):
        value = None if pd.isna(row.vpp_id) else str(row.vpp_id)
        if value == "":
            value = None
        mapping[int(row.bus)] = value
    return mapping


def _read_der_vpp_csv(path: Path) -> dict[int, str | None]:
    df = pd.read_csv(path)
    required = {"der", "vpp_id"}
    if not required.issubset(df.columns):
        raise ValueError(f"{path} must contain columns {sorted(required)}")

    mapping: dict[int, str | None] = {}
    for row in df.itertuples(index=False):
        value = None if pd.isna(row.vpp_id) else str(row.vpp_id)
        if value == "":
            value = None
        mapping[int(row.der)] = value
    return mapping


def load_canonical_mappings(
    net: pp.pandapowerNet,
    mapping_config: Mapping[str, Any] | None,
    *,
    zone_bus_map: Mapping[int, list[str]] | None = None,
) -> CanonicalMappings:
    cfg = dict(mapping_config or {})
    bus_to_zone_path = _resolve_path(cfg.get("bus_to_zone_csv"))
    vpp_to_zone_path = _resolve_path(cfg.get("vpp_to_zone_csv"))
    bus_to_vpp_path = _resolve_path(cfg.get("bus_to_vpp_csv"))
    der_to_vpp_path = _resolve_path(cfg.get("der_to_vpp_csv"))

    if bus_to_zone_path is None or vpp_to_zone_path is None:
        warnings.warn(
            "Using legacy zone mapping fallback because canonical CSV mappings are incomplete. "
            "Provide both 'bus_to_zone_csv' and 'vpp_to_zone_csv' to enforce static canonical partitions.",
            RuntimeWarning,
            stacklevel=2,
        )
        return CanonicalMappings(
            bus_to_zone=_legacy_bus_to_zone(net, zone_bus_map),
            bus_to_vpp={int(bus): None for bus in net.bus.index},
            vpp_to_zone={},
            der_to_vpp={},
            legacy_mode=True,
        )

    bus_to_zone = _read_bus_zone_csv(bus_to_zone_path)
    bus_to_vpp_raw = _read_bus_vpp_csv(bus_to_vpp_path) if bus_to_vpp_path is not None else {}
    vpp_to_zone = _read_vpp_zone_csv(vpp_to_zone_path)
    der_to_vpp = _read_der_vpp_csv(der_to_vpp_path) if der_to_vpp_path is not None else {}

    missing_buses = [int(bus) for bus in net.bus.index if int(bus) not in bus_to_zone]
    if missing_buses:
        raise ValueError(
            f"bus_to_zone mapping missing buses: {missing_buses[:10]}"
            + ("..." if len(missing_buses) > 10 else "")
        )

    bus_to_vpp = {int(bus): bus_to_vpp_raw.get(int(bus)) for bus in net.bus.index}

    for bus, vpp in bus_to_vpp.items():
        if vpp is None:
            continue
        if vpp not in vpp_to_zone:
            raise ValueError(f"bus {bus} references unknown vpp_id '{vpp}'")

    for der, vpp in der_to_vpp.items():
        if vpp is None:
            continue
        if vpp not in vpp_to_zone:
            raise ValueError(f"der {der} references unknown vpp_id '{vpp}'")

    for bus, vpp in bus_to_vpp.items():
        if vpp is None:
            continue
        if vpp_to_zone[vpp] != bus_to_zone[bus]:
            raise ValueError(
                f"bus {bus} in zone {bus_to_zone[bus]} conflicts with vpp '{vpp}' in zone {vpp_to_zone[vpp]}"
            )

    return CanonicalMappings(
        bus_to_zone={int(k): int(v) for k, v in bus_to_zone.items()},
        bus_to_vpp=bus_to_vpp,
        vpp_to_zone={str(k): int(v) for k, v in vpp_to_zone.items()},
        der_to_vpp={int(k): v for k, v in der_to_vpp.items()},
        legacy_mode=False,
    )


def resolve_der_vpp(mappings: CanonicalMappings, der_idx: int, bus_idx: int) -> str | None:
    if der_idx in mappings.der_to_vpp:
        return mappings.der_to_vpp[der_idx]
    return mappings.bus_to_vpp.get(bus_idx)


def build_partition_report(net: pp.pandapowerNet, mappings: CanonicalMappings) -> MappingSummary:
    load_by_bus: dict[int, float] = {int(bus): 0.0 for bus in net.bus.index}
    for _, row in net.load.iterrows():
        bus = int(row["bus"])
        load_by_bus[bus] = load_by_bus.get(bus, 0.0) + float(row.get("p_mw", 0.0))

    der_by_bus: dict[int, float] = {int(bus): 0.0 for bus in net.bus.index}
    der_count_by_bus: dict[int, int] = {int(bus): 0 for bus in net.bus.index}
    for sgen_idx, row in net.sgen.iterrows():
        if row.get("in_service") is False:
            continue
        bus = int(row["bus"])
        der_by_bus[bus] = der_by_bus.get(bus, 0.0) + float(row.get("p_mw", 0.0))
        der_count_by_bus[bus] = der_count_by_bus.get(bus, 0) + 1

    rows: list[dict[str, object]] = []
    all_vpps = sorted(mappings.vpp_to_zone)
    for vpp_id in all_vpps:
        zone_id = mappings.vpp_to_zone[vpp_id]
        buses = [bus for bus, owner in mappings.bus_to_vpp.items() if owner == vpp_id]
        der_count = sum(der_count_by_bus.get(bus, 0) for bus in buses)
        rows.append(
            {
                "scope": "vpp",
                "zone_id": zone_id,
                "vpp_id": vpp_id,
                "bus_count": len(buses),
                "der_count": int(der_count),
                "load_mw": float(sum(load_by_bus.get(bus, 0.0) for bus in buses)),
                "der_mw": float(sum(der_by_bus.get(bus, 0.0) for bus in buses)),
                "buses": ";".join(str(bus) for bus in sorted(buses)),
            }
        )

    non_vpp_buses = sorted([bus for bus, owner in mappings.bus_to_vpp.items() if owner is None])
    rows.append(
        {
            "scope": "non_vpp",
            "zone_id": "mixed",
            "vpp_id": "",
            "bus_count": len(non_vpp_buses),
            "der_count": int(sum(der_count_by_bus.get(bus, 0) for bus in non_vpp_buses)),
            "load_mw": float(sum(load_by_bus.get(bus, 0.0) for bus in non_vpp_buses)),
            "der_mw": float(sum(der_by_bus.get(bus, 0.0) for bus in non_vpp_buses)),
            "buses": ";".join(str(bus) for bus in non_vpp_buses),
        }
    )
    return MappingSummary(rows=rows)
