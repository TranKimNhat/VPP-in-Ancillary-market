from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandapower as pp


@dataclass(frozen=True)
class TopologySnapshot:
    adjacency: np.ndarray
    bus_index: list[int]


def build_topology_snapshot(net: pp.pandapowerNet) -> TopologySnapshot:
    buses = [int(idx) for idx in net.bus.index]
    size = len(buses)
    adjacency = np.zeros((size, size), dtype=float)
    lookup = {bus: i for i, bus in enumerate(buses)}

    if not net.line.empty:
        for _, row in net.line.iterrows():
            i = lookup.get(int(row["from_bus"]))
            j = lookup.get(int(row["to_bus"]))
            if i is None or j is None:
                continue
            adjacency[i, j] = 1.0
            adjacency[j, i] = 1.0

    if not net.switch.empty:
        for _, row in net.switch.iterrows():
            if str(row.get("et", "")) != "b":
                continue
            if not bool(row.get("closed", True)):
                continue
            i = lookup.get(int(row["bus"]))
            j = lookup.get(int(row["element"]))
            if i is None or j is None:
                continue
            adjacency[i, j] = 1.0
            adjacency[j, i] = 1.0

    return TopologySnapshot(adjacency=adjacency, bus_index=buses)


def build_edge_index(net: pp.pandapowerNet) -> np.ndarray:
    """Build COO sparse edge_index of shape (2, E) from pandapower net.

    Only in-service lines and closed bus-bus switches are included.
    Both directions are added so the graph is treated as undirected.
    Returns shape (2, 0) when no edges exist.
    """
    buses = [int(idx) for idx in net.bus.index]
    lookup = {bus: i for i, bus in enumerate(buses)}
    src_nodes: list[int] = []
    dst_nodes: list[int] = []

    if not net.line.empty:
        for _, row in net.line.iterrows():
            if not bool(row.get("in_service", True)):
                continue
            i = lookup.get(int(row["from_bus"]))
            j = lookup.get(int(row["to_bus"]))
            if i is None or j is None:
                continue
            src_nodes.extend([i, j])
            dst_nodes.extend([j, i])

    if not net.switch.empty:
        for _, row in net.switch.iterrows():
            if str(row.get("et", "")) != "b":
                continue
            if not bool(row.get("closed", True)):
                continue
            i = lookup.get(int(row["bus"]))
            j = lookup.get(int(row["element"]))
            if i is None or j is None:
                continue
            src_nodes.extend([i, j])
            dst_nodes.extend([j, i])

    if not src_nodes:
        return np.zeros((2, 0), dtype=np.int64)
    return np.array([src_nodes, dst_nodes], dtype=np.int64)
