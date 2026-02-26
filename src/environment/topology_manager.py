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
