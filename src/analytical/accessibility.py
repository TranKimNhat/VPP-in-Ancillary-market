"""Canonical GFM accessibility index (concept v3.1 §8).

    w_k    = P_L,k / sum_j P_L,j                       (load weights, sum = 1)
    pi_g   = P_head,g / sum_h P_head,h                 (headroom weights, sum = 1)
    Abar   = sum_k w_k sum_g pi_g Z_base / (|Z_gk| + eps_g)     [dimensionless]

`Z_gk` is the electrical distance between GFM bus `g` and load bus `k` under
topology `G`; on a radial feeder that is the series impedance of the unique
path, so it is computed as a shortest-path sum over the branch graph.

`eps_g` is the regularizer of §8. It is NOT a numerical epsilon: several GFM
sit on load buses, so `|Z_gk| = 0` for those pairs and the value chosen for
eps sets their contribution. See `interface_impedance_ohm`.
"""

from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
import numpy as np


@dataclass(frozen=True)
class AccessibilityInputs:
    """Everything the index needs, already resolved to bus indices and ohms."""

    load_buses: list[int]
    p_load_mw: np.ndarray  # (n_load,)
    gfm_buses: list[int]
    p_head_mw: np.ndarray  # (n_gfm,)
    z_ohm: np.ndarray  # (n_load, n_gfm) electrical distance
    z_base_ohm: float
    eps_ohm: np.ndarray  # (n_gfm,) per-GFM regularizer


def build_branch_graph(net) -> nx.Graph:
    """Undirected graph of the in-service network, edge weight = |z| in ohm.

    Closed bus-bus switches and transformers are zero-impedance edges; open
    switches are omitted, which is what makes the graph topology-dependent.
    """
    graph = nx.Graph()
    graph.add_nodes_from(int(b) for b in net.bus.index)

    for _, line in net.line.iterrows():
        if not bool(line.get("in_service", True)):
            continue
        z = abs(complex(line["r_ohm_per_km"], line["x_ohm_per_km"])) * float(line["length_km"])
        f, t = int(line["from_bus"]), int(line["to_bus"])
        if graph.has_edge(f, t):
            # parallel branches: keep the lower impedance path
            graph[f][t]["z"] = min(graph[f][t]["z"], z)
        else:
            graph.add_edge(f, t, z=z)

    for _, switch in net.switch.iterrows():
        if switch["et"] != "b" or not bool(switch["closed"]):
            continue
        f, t = int(switch["bus"]), int(switch["element"])
        if not graph.has_edge(f, t):
            graph.add_edge(f, t, z=0.0)

    for _, trafo in net.trafo.iterrows():
        if not bool(trafo.get("in_service", True)):
            continue
        f, t = int(trafo["hv_bus"]), int(trafo["lv_bus"])
        if not graph.has_edge(f, t):
            graph.add_edge(f, t, z=0.0)

    return graph


def distance_matrix(graph: nx.Graph, load_buses: list[int], gfm_buses: list[int]) -> np.ndarray:
    """(n_load, n_gfm) electrical distance in ohm; inf where unreachable."""
    z = np.full((len(load_buses), len(gfm_buses)), np.inf)
    for j, g in enumerate(gfm_buses):
        lengths = nx.single_source_dijkstra_path_length(graph, g, weight="z")
        for i, k in enumerate(load_buses):
            if k in lengths:
                z[i, j] = lengths[k]
    return z


def interface_impedance_ohm(s_rated_mva: np.ndarray, v_base_kv: float, x_pu: float) -> np.ndarray:
    """Per-GFM regularizer: the converter's own coupling impedance, in ohm.

    A GFM never reaches a bus through zero impedance, not even its own: the
    LCL filter plus the step-up transformer always sit in between. Referring
    x_pu to each unit's own MVA base makes a larger unit electrically closer,
    which is the physically correct ordering.
    """
    return x_pu * v_base_kv**2 / np.asarray(s_rated_mva, dtype=float)


def a_gfm_bar(inp: AccessibilityInputs) -> float:
    """Normalized (dimensionless) accessibility index of §8."""
    w = inp.p_load_mw / inp.p_load_mw.sum()
    pi = inp.p_head_mw / inp.p_head_mw.sum()
    kernel = inp.z_base_ohm / (inp.z_ohm + inp.eps_ohm[None, :])
    return float(w @ kernel @ pi)


def a_gfm_raw(inp: AccessibilityInputs) -> float:
    """Unnormalized diagnostic of §8; mixes capacity with placement."""
    kernel = 1.0 / (inp.z_ohm + inp.eps_ohm[None, :])
    return float(inp.p_load_mw @ kernel @ inp.p_head_mw)


def distance_stats(inp: AccessibilityInputs) -> dict[str, float]:
    """Headroom-agnostic distance diagnostics: distance to the nearest GFM.

    Reported alongside Abar because they answer a different question (how far
    is the network stretched) and are what Z_avg / Z_max / Z_95 meant in the
    v3.1 concept tables.
    """
    z_near = inp.z_ohm.min(axis=1)
    w = inp.p_load_mw / inp.p_load_mw.sum()
    return {
        "Z_avg_ohm": float(z_near.mean()),
        "Z_avg_wload_ohm": float(w @ z_near),
        "Z_max_ohm": float(z_near.max()),
        "Z_95_ohm": float(np.percentile(z_near, 95)),
        "Z_max_bus": int(np.asarray(inp.load_buses)[int(z_near.argmax())]),
    }
