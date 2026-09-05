"""T46: place two grid-forming units for weak-grid SCR *and* reconfiguration richness.

T45 established that only n = 2 reaches Dozein's weak-grid threshold on this
feeder: in an all-GFM island each unit's Thevenin source is the other units, so
SCR ~ (n-1)/x_conv almost regardless of the feeder, and six comparable units
make each other stiff (SCR 18.7). Two units at 2.18 MVA give SCR 2.21.

Maximum electrical separation alone is the wrong placement rule for a
reconfiguration study, because what makes a configuration *count* is that it
changes which unit feeds which load. So this scores bus pairs on two things:

    SCR < 3          Dozein's weak-grid threshold. For two equal units,
                     SCR = 1 / ((z_ab + x_conv/S) * S), so it is a lower bound
                     on the a-b path impedance.
    distinguishable  how many of the network's radial configurations produce a
                     different partition of load buses by nearest unit.

The configuration space is enumerated exactly rather than sampled. As built,
the live network is a tree of 125 nodes and 124 branches; three of the five
bus-bus ties land on buses that `build_case` drops as isolated, so two ties are
usable, giving two independent loops of 13 and 26 branches. The matrix-tree
theorem puts the number of radial configurations at 322, and all of them are
enumerated here by branch exchange.

Run:
    uv run python experiments/t46_two_gfm_placement.py
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import networkx as nx
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.analytical.accessibility import build_branch_graph  # noqa: E402
from src.phasor.build_case import V_BASE_KV, _base_net  # noqa: E402

OUT = ROOT / "artifacts" / "T46_two_gfm_placement"
FLEET_MVA = 4.3624
X_CONV = 0.21
WEAK = 3.0


def enumerate_radial(g, ties):
    """Every radial configuration reachable by closing ties and opening branches.

    Each closed tie creates one loop, and radiality is restored by opening one
    branch of that loop. Loops can share branches, so the combinations are
    checked rather than assumed, and configurations with a tie left open are
    included.
    """
    n_nodes = g.number_of_nodes()
    loops = []
    for a, b in ties:
        path = nx.shortest_path(g, a, b)
        loops.append([(path[i], path[i + 1]) for i in range(len(path) - 1)])

    out = []
    # choice[k] is the branch opened for tie k, or None meaning "leave tie k open"
    choices = [[None] + loop for loop in loops]
    for combo in itertools.product(*choices):
        h = g.copy()
        for k, opened in enumerate(combo):
            if opened is None:
                continue
            h.add_edge(*ties[k], z=0.0)
            if not h.has_edge(*opened):
                break
            h.remove_edge(*opened)
        else:
            if (h.number_of_edges() == n_nodes - 1
                    and nx.is_connected(h)):
                out.append(h)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--top", type=int, default=15)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    net = _base_net()
    g0 = build_branch_graph(net)
    g = g0.subgraph(max(nx.connected_components(g0), key=len)).copy()
    zb = V_BASE_KV ** 2 / 1.0

    bb = net.switch[net.switch.et == "b"]
    ties = [(int(r.bus), int(r.element)) for _, r in bb.iterrows()
            if not bool(r.closed) and g.has_node(int(r.bus)) and g.has_node(int(r.element))]
    print(f"live network: {g.number_of_nodes()} nodes, {g.number_of_edges()} branches, "
          f"tree = {g.number_of_edges() == g.number_of_nodes() - 1}")
    print(f"usable ties: {ties}")

    configs = enumerate_radial(g, ties)
    print(f"radial configurations enumerated: {len(configs)}\n")

    nodes = sorted(g.nodes)
    idx = {b: i for i, b in enumerate(nodes)}
    load_cols = [idx[int(b)] for b in sorted({int(x) for x in net.load.bus.unique()})
                 if int(b) in idx]

    # all-pairs distances per configuration, once
    D = np.empty((len(configs), len(nodes), len(nodes)), dtype=np.float32)
    for c, h in enumerate(configs):
        sp = dict(nx.all_pairs_dijkstra_path_length(h, weight="z"))
        for b, i in idx.items():
            row = sp[b]
            D[c, i] = [row.get(q, np.inf) for q in nodes]
    D /= zb
    print("distance tensor built")

    s_i = FLEET_MVA / 2
    x_sys = X_CONV / s_i
    z_need = 1.0 / (WEAK * s_i) - x_sys
    print(f"two units of {s_i:.4f} MVA; SCR < {WEAK} needs z_ab > {z_need:.5f} pu\n")

    pp2name = {i: str(r["name"]) for i, r in net.bus.iterrows()}
    rows = []
    base = 0
    for ia, ib in itertools.combinations(range(len(nodes)), 2):
        if D[base, ia, ib] <= z_need:
            continue
        zab = D[:, ia, ib]
        sig = (D[:, ia, load_cols] <= D[:, ib, load_cols])
        distinct = len(np.unique(sig, axis=0))
        rows.append({
            "bus_a": pp2name[nodes[ia]], "bus_b": pp2name[nodes[ib]],
            "n_configs": len(configs), "n_distinct_partitions": int(distinct),
            "z_ab_base": float(zab[base]), "z_ab_min": float(zab.min()),
            "z_ab_max": float(zab.max()),
            "scr_base": float(1.0 / ((zab[base] + x_sys) * s_i)),
            "scr_worst": float(1.0 / ((zab.min() + x_sys) * s_i)),
        })

    print(f"{len(rows)} bus pairs meet SCR < {WEAK} at the base configuration\n")
    rows.sort(key=lambda r: (-r["n_distinct_partitions"], r["scr_worst"]))
    print(f"{'bus A':>7}{'bus B':>7}{'distinct':>10}{'/':>2}{'configs':<9}"
          f"{'z_ab base':>11}{'SCR base':>10}{'SCR worst':>11}")
    for r in rows[:args.top]:
        print(f"{r['bus_a']:>7}{r['bus_b']:>7}{r['n_distinct_partitions']:>10}"
              f"{'/':>2}{r['n_configs']:<9}{r['z_ab_base']:>11.5f}"
              f"{r['scr_base']:>10.2f}{r['scr_worst']:>11.2f}")

    (args.out / "placement_search.json").write_text(
        json.dumps(rows[:80], indent=1), encoding="utf-8")
    print(f"\nwrote {args.out / 'placement_search.json'}")


if __name__ == "__main__":
    main()
