from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import argparse

import pandas as pd
import pandapower as pp

from src.env.IEEE123bus import IEEE123_ZONE_BUS_MAP, build_ieee123_net, validate_ieee123_net
from src.environment.topology_manager import build_topology_snapshot


CONTROLLABLE_TYPES = {"pv", "wind", "bess", "storage"}


@dataclass(frozen=True)
class VppFormationConfig:
    p_min_der_kw: float = 10.0
    p_min_vpp_kw: float = 50.0
    n_min_buses: int = 1
    n_max_buses: int = 8
    max_unassigned_ratio: float = 0.5


@dataclass(frozen=True)
class VppFormationArtifacts:
    bus_to_vpp_csv: Path
    vpp_to_zone_csv: Path
    vpp_summary_csv: Path


def _controllable_type(raw_type: object) -> str:
    value = str(raw_type or "").strip().lower()
    if value == "battery":
        return "storage"
    return value


def _der_capacity_kw(row: pd.Series) -> float:
    sn_mva = row.get("sn_mva")
    if pd.notna(sn_mva) and float(sn_mva) > 0.0:
        return float(sn_mva) * 1000.0
    p_mw = row.get("p_mw", 0.0)
    if pd.notna(p_mw):
        return abs(float(p_mw)) * 1000.0
    return 0.0


def _bus_distance_lookup(net: pp.pandapowerNet) -> dict[int, dict[int, int]]:
    snapshot = build_topology_snapshot(net)
    buses = snapshot.bus_index
    adjacency = snapshot.adjacency
    neighbors = {
        bus: [buses[j] for j, linked in enumerate(adjacency[i]) if linked > 0.0]
        for i, bus in enumerate(buses)
    }

    out: dict[int, dict[int, int]] = {}
    for source in buses:
        dist: dict[int, int] = {source: 0}
        queue = [source]
        cursor = 0
        while cursor < len(queue):
            node = queue[cursor]
            cursor += 1
            for nxt in neighbors.get(node, []):
                if nxt in dist:
                    continue
                dist[nxt] = dist[node] + 1
                queue.append(nxt)
        out[source] = dist
    return out


def _bus_zone_map(net: pp.pandapowerNet, mapping_config: dict[str, Any]) -> dict[int, int]:
    bus_to_zone_csv = mapping_config.get("bus_to_zone_csv") if mapping_config else None
    if bus_to_zone_csv:
        df = pd.read_csv(bus_to_zone_csv)
        required = {"bus", "zone_id"}
        if not required.issubset(df.columns):
            raise ValueError(f"{bus_to_zone_csv} must contain columns {sorted(required)}")
        return {int(row.bus): int(row.zone_id) for row in df.itertuples(index=False)}

    bus_name_to_zone: dict[str, int] = {}
    for zone_id, names in IEEE123_ZONE_BUS_MAP.items():
        for name in names:
            bus_name_to_zone[str(name)] = int(zone_id)

    out: dict[int, int] = {}
    for bus_idx, row in net.bus.iterrows():
        out[int(bus_idx)] = int(bus_name_to_zone.get(str(row.get("name", "")), 1))
    return out


def _eligible_der_by_bus(
    net: pp.pandapowerNet,
    bus_to_zone: dict[int, int],
    cfg: VppFormationConfig,
) -> tuple[pd.DataFrame, dict[int, dict[str, float | int]]]:
    rows: list[dict[str, object]] = []
    stats: dict[int, dict[str, float | int]] = {}

    for bus in net.bus.index:
        stats[int(bus)] = {
            "der_count": 0,
            "p_cap_kw": 0.0,
            "controllable": False,
            "der_types": set(),
        }

    for _, row in net.sgen.iterrows():
        if row.get("in_service") is False:
            continue
        bus = int(row["bus"])
        der_type = _controllable_type(row.get("type"))
        p_cap_kw = _der_capacity_kw(row)
        controllable = der_type in CONTROLLABLE_TYPES
        if controllable and p_cap_kw >= cfg.p_min_der_kw:
            rows.append(
                {
                    "bus": bus,
                    "zone_id": int(bus_to_zone[bus]),
                    "der_type": der_type,
                    "p_cap_kw": p_cap_kw,
                    "controllable": True,
                }
            )
            stats[bus]["der_count"] = int(stats[bus]["der_count"]) + 1
            stats[bus]["p_cap_kw"] = float(stats[bus]["p_cap_kw"]) + p_cap_kw
            stats[bus]["controllable"] = True
            cast = stats[bus]["der_types"]
            if isinstance(cast, set):
                cast.add(der_type)

    eligible = pd.DataFrame(rows)
    return eligible, stats


def _build_zone_clusters(
    eligible: pd.DataFrame,
    bus_dist: dict[int, dict[int, int]],
    cfg: VppFormationConfig,
) -> dict[int, list[list[int]]]:
    clusters: dict[int, list[list[int]]] = {}
    if eligible.empty:
        return clusters

    per_bus = (
        eligible.groupby(["zone_id", "bus"], as_index=False)["p_cap_kw"]
        .sum()
        .sort_values(["zone_id", "p_cap_kw"], ascending=[True, False])
    )

    for zone_id, zone_df in per_bus.groupby("zone_id"):
        buses = zone_df["bus"].astype(int).tolist()
        cap_by_bus = {int(r.bus): float(r.p_cap_kw) for r in zone_df.itertuples(index=False)}
        remaining = set(buses)
        zone_clusters: list[list[int]] = []

        while remaining:
            seed = max(remaining, key=lambda b: cap_by_bus.get(b, 0.0))
            remaining.remove(seed)
            cluster = [seed]

            while remaining and len(cluster) < cfg.n_max_buses:
                nearest = min(
                    remaining,
                    key=lambda b: min(bus_dist.get(b, {}).get(c, 10**9) for c in cluster),
                )
                nearest_dist = min(bus_dist.get(nearest, {}).get(c, 10**9) for c in cluster)
                if nearest_dist == 10**9:
                    break
                cluster.append(nearest)
                remaining.remove(nearest)

            zone_clusters.append(cluster)

        clusters[int(zone_id)] = zone_clusters

    return clusters


def _assign_vpps(
    net: pp.pandapowerNet,
    bus_to_zone: dict[int, int],
    eligible_stats: dict[int, dict[str, float | int]],
    zone_clusters: dict[int, list[list[int]]],
    cfg: VppFormationConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    bus_rows: list[dict[str, object]] = []
    vpp_zone_rows: list[dict[str, object]] = []
    vpp_summary_rows: list[dict[str, object]] = []

    bus_to_vpp: dict[int, str | None] = {int(bus): None for bus in net.bus.index}

    for zone_id, clusters in sorted(zone_clusters.items()):
        vpp_index = 0
        for cluster in clusters:
            cluster_cap = float(sum(float(eligible_stats[b]["p_cap_kw"]) for b in cluster))
            if cluster_cap < cfg.p_min_vpp_kw or len(cluster) < cfg.n_min_buses:
                continue
            vpp_id = f"vpp_{zone_id}_{chr(ord('a') + vpp_index)}"
            vpp_index += 1
            vpp_zone_rows.append({"vpp_id": vpp_id, "zone_id": int(zone_id)})

            der_types = sorted(
                {
                    t
                    for b in cluster
                    for t in (eligible_stats[b].get("der_types") or set())
                    if isinstance(t, str)
                }
            )

            buses_text = ";".join(str(int(b)) for b in sorted(cluster))
            vpp_summary_rows.append(
                {
                    "vpp_id": vpp_id,
                    "zone_id": int(zone_id),
                    "bus_count": int(len(cluster)),
                    "der_count": int(sum(int(eligible_stats[b]["der_count"]) for b in cluster)),
                    "load_mw": float(
                        net.load.loc[net.load["bus"].isin(cluster), "p_mw"].sum() if not net.load.empty else 0.0
                    ),
                    "der_mw": float(
                        net.sgen.loc[net.sgen["bus"].isin(cluster), "p_mw"].sum() if not net.sgen.empty else 0.0
                    ),
                    "p_cap_total_kw": cluster_cap,
                    "der_types": ",".join(der_types),
                    "buses": buses_text,
                }
            )

            for bus in cluster:
                bus_to_vpp[int(bus)] = vpp_id

    eligible_buses = [bus for bus, stats in eligible_stats.items() if bool(stats.get("controllable", False))]
    assigned = sum(1 for bus in eligible_buses if bus_to_vpp.get(bus) is not None)
    if eligible_buses:
        unassigned_ratio = 1.0 - (assigned / float(len(eligible_buses)))
        if unassigned_ratio > cfg.max_unassigned_ratio:
            raise ValueError(
                f"VPP formation produced too many unassigned eligible buses ({unassigned_ratio:.2%}). "
                "Adjust thresholds in VppFormationConfig."
            )

    for bus in net.bus.index:
        bus_i = int(bus)
        stats = eligible_stats.get(bus_i, {"der_count": 0, "p_cap_kw": 0.0, "controllable": False, "der_types": set()})
        der_types = stats.get("der_types") or set()
        first_type = sorted(der_types)[0] if isinstance(der_types, set) and der_types else None
        bus_rows.append(
            {
                "bus": bus_i,
                "zone_id": int(bus_to_zone[bus_i]),
                "vpp_id": bus_to_vpp[bus_i],
                "der_type": first_type,
                "p_cap_kw": float(stats.get("p_cap_kw", 0.0)),
                "controllable": bool(stats.get("controllable", False)),
                "vpp_role": "member" if bus_to_vpp[bus_i] is not None else "non_vpp",
            }
        )

    bus_df = pd.DataFrame(bus_rows).sort_values(["zone_id", "bus"]).reset_index(drop=True)

    vpp_zone_df = pd.DataFrame(vpp_zone_rows)
    if vpp_zone_df.empty:
        vpp_zone_df = pd.DataFrame(columns=["vpp_id", "zone_id"])
    else:
        vpp_zone_df = vpp_zone_df.sort_values(["zone_id", "vpp_id"]).reset_index(drop=True)

    vpp_summary_df = pd.DataFrame(vpp_summary_rows)
    if vpp_summary_df.empty:
        vpp_summary_df = pd.DataFrame(
            columns=["vpp_id", "zone_id", "bus_count", "der_count", "load_mw", "der_mw", "p_cap_total_kw", "der_types", "buses"]
        )
    else:
        vpp_summary_df = vpp_summary_df.sort_values(["zone_id", "vpp_id"]).reset_index(drop=True)

    return bus_df, vpp_zone_df, vpp_summary_df


def run_vpp_formation(
    output_dir: Path,
    mapping_config: dict[str, Any],
    formation_config: VppFormationConfig | None = None,
) -> VppFormationArtifacts:
    cfg = formation_config or VppFormationConfig()
    output_dir.mkdir(parents=True, exist_ok=True)

    net = build_ieee123_net(mode="feeder123", balanced=True, convert_switches=True, slack_zones=None)
    validate_ieee123_net(net)

    bus_to_zone = _bus_zone_map(net, mapping_config)
    bus_dist = _bus_distance_lookup(net)
    eligible_df, eligible_stats = _eligible_der_by_bus(net, bus_to_zone, cfg)
    zone_clusters = _build_zone_clusters(eligible_df, bus_dist, cfg)
    bus_to_vpp_df, vpp_to_zone_df, vpp_summary_df = _assign_vpps(
        net,
        bus_to_zone,
        eligible_stats,
        zone_clusters,
        cfg,
    )

    bus_to_vpp_csv = output_dir / "bus_to_vpp.csv"
    vpp_to_zone_csv = output_dir / "vpp_to_zone.csv"
    vpp_summary_csv = output_dir / "vpp_summary.csv"

    bus_to_vpp_df.to_csv(bus_to_vpp_csv, index=False)
    vpp_to_zone_df.to_csv(vpp_to_zone_csv, index=False)
    vpp_summary_df.to_csv(vpp_summary_csv, index=False)

    return VppFormationArtifacts(
        bus_to_vpp_csv=bus_to_vpp_csv,
        vpp_to_zone_csv=vpp_to_zone_csv,
        vpp_summary_csv=vpp_summary_csv,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate static VPP formation artifacts.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/oedisi-ieee123-main/profiles/vpp_assignments"),
        help="Output directory for VPP mapping artifacts.",
    )
    parser.add_argument(
        "--bus-to-zone-csv",
        type=str,
        default="",
        help="Path to canonical bus_to_zone CSV.",
    )
    parser.add_argument("--p-min-der-kw", type=float, default=10.0)
    parser.add_argument("--p-min-vpp-kw", type=float, default=50.0)
    parser.add_argument("--n-min-buses", type=int, default=1)
    parser.add_argument("--n-max-buses", type=int, default=8)
    parser.add_argument("--max-unassigned-ratio", type=float, default=0.5)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    mapping_config: dict[str, Any] = {}
    if args.bus_to_zone_csv:
        mapping_config["bus_to_zone_csv"] = args.bus_to_zone_csv

    artifacts = run_vpp_formation(
        output_dir=args.output_dir,
        mapping_config=mapping_config,
        formation_config=VppFormationConfig(
            p_min_der_kw=float(args.p_min_der_kw),
            p_min_vpp_kw=float(args.p_min_vpp_kw),
            n_min_buses=int(args.n_min_buses),
            n_max_buses=int(args.n_max_buses),
            max_unassigned_ratio=float(args.max_unassigned_ratio),
        ),
    )
    print(f"bus_to_vpp: {artifacts.bus_to_vpp_csv}")
    print(f"vpp_to_zone: {artifacts.vpp_to_zone_csv}")
    print(f"vpp_summary: {artifacts.vpp_summary_csv}")


if __name__ == "__main__":
    main()
