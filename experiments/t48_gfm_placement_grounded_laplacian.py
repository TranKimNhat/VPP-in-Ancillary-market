"""T48: place grid-forming units by the criterion the literature actually supports.

T45-T47 used SCR, and T-checks against the literature showed that was wrong on
two counts. SCR < 3 is a *grid-following* threshold -- Dozein, Pal & Mancarella
never mention grid-forming and use PLL 43 times -- and for grid-forming the
relationship runs the other way: GFM is stable in weak grids and has
stiff-grid instability instead (Han 2024; Gao 2024; Li 2021; Zhou 2026).

The criterion that does exist for this problem is Yang, Xu, Zhang & Sun,
"Placing Grid-Forming Converters to Enhance Small Signal Stability of
PLL-Integrated Power Systems", IEEE TPWRS 2020 (220 citations). Its result:
placing a grid-forming converter is equivalent to increasing power grid
strength for the PLL-based converters around it, and the placement is optimised
by maximising the smallest eigenvalue of the weighted, Kron-reduced Laplacian
of the network.

That matters here because the fleet is mixed. The placement file carries 41
agents, of which six are grid-forming and the rest -- wind, DPV, EVCS -- are
grid-following. So grid-forming placement is not about how the GFM units share
power between themselves, which T44 measured directly and found to be
position-independent to 0.28 pp. It is about the grid strength the GFL fleet
sees.

Implementation follows the paper's construction:

    L        weighted Laplacian, edge weight = |1/z| (admittance magnitude);
             zero-impedance branches (closed switches, transformers) are
             contracted rather than given an infinite weight
    Kron     eliminate every bus that is neither GFM nor GFL
    ground   delete the GFM rows and columns -- a grid-forming unit is a voltage
             source, i.e. a node held at the reference
    lambda   the smallest eigenvalue of what remains is the grid strength the
             GFL buses see; larger is stronger

Run:
    uv run python experiments/t48_gfm_placement_grounded_laplacian.py
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
from src.phasor.build_case import (CaseSpec, V_BASE_KV, _base_net,  # noqa: E402
                                   gfm_table)

OUT = ROOT / "artifacts" / "T48_grounded_laplacian"
ZERO = 1e-9          # ohm; below this a branch is treated as a short


def contracted(g):
    """Merge zero-impedance branches; they are shorts, not huge admittances."""
    h = g.copy()
    mapping = {n: n for n in h.nodes}
    changed = True
    while changed:
        changed = False
        for a, b, d in list(h.edges(data=True)):
            if d.get("z", 0.0) <= ZERO:
                h = nx.contracted_nodes(h, a, b, self_loops=False)
                for k, v in mapping.items():
                    if v == b:
                        mapping[k] = a
                changed = True
                break
    return h, mapping


def laplacian(h, nodes):
    idx = {b: i for i, b in enumerate(nodes)}
    n = len(nodes)
    L = np.zeros((n, n))
    for a, b, d in h.edges(data=True):
        y = 1.0 / max(d.get("z", 0.0), ZERO)
        i, j = idx[a], idx[b]
        L[i, i] += y; L[j, j] += y
        L[i, j] -= y; L[j, i] -= y
    return L


def kron(L, keep):
    """Eliminate every index not in `keep`."""
    keep = sorted(keep)
    elim = [i for i in range(L.shape[0]) if i not in set(keep)]
    if not elim:
        return L[np.ix_(keep, keep)]
    A = L[np.ix_(keep, keep)]
    B = L[np.ix_(keep, elim)]
    D = L[np.ix_(elim, elim)]
    return A - B @ np.linalg.solve(D, B.T)


def strength(L, keep_idx, gfm_idx, gfl_idx):
    """Smallest eigenvalue of the Kron-reduced Laplacian grounded at the GFM buses."""
    Lr = kron(L, keep_idx)
    pos = {b: i for i, b in enumerate(sorted(keep_idx))}
    gfl_local = [pos[b] for b in gfl_idx]
    Lg = Lr[np.ix_(gfl_local, gfl_local)]      # grounding = drop the GFM block
    w = np.linalg.eigvalsh(0.5 * (Lg + Lg.T))
    return float(w.min())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--top", type=int, default=10)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    net = _base_net()
    g0 = build_branch_graph(net)
    live = g0.subgraph(max(nx.connected_components(g0), key=len)).copy()
    h, m = contracted(live)
    nodes = sorted(h.nodes)
    idx = {b: i for i, b in enumerate(nodes)}
    L = laplacian(h, nodes) * (V_BASE_KV ** 2)   # to per-unit admittance on 1 MVA

    p2n = {i: str(r["name"]) for i, r in net.bus.iterrows()}
    n2p = {v: k for k, v in p2n.items()}

    gfl_buses = sorted({idx[m[int(b)]] for b in net.sgen.bus.unique()
                        if int(b) in m and m[int(b)] in idx})
    print(f"live buses {live.number_of_nodes()}, after contracting shorts {len(nodes)}")
    print(f"GFL (sgen) buses in the reduced graph: {len(gfl_buses)}\n")

    cur = [idx[m[n2p[r['bus_name']]]] for r in gfm_table(CaseSpec())
           if n2p[r['bus_name']] in m]

    def score(sel):
        sel = [s for s in sel if s not in gfl_buses]
        if not sel:
            return None
        keep = sorted(set(sel) | set(gfl_buses))
        return strength(L, keep, sel, gfl_buses)

    base = score(cur)
    print(f"shipped 6 GFM at {','.join(p2n[nodes[i]] for i in cur)}: "
          f"lambda_min = {base:.6f}\n")

    pool = [i for i in range(len(nodes)) if i not in gfl_buses]
    print(f"{'n':>3}  {'greedy-max lambda_min placement':<38}{'lambda_min':>12}"
          f"{'vs shipped':>12}")
    rows = []
    for n in (2, 3, 4, 5, 6):
        sel: list[int] = []
        for _ in range(n):
            cand = max((b for b in pool if b not in sel),
                       key=lambda b: score(sel + [b]) or -np.inf)
            sel.append(cand)
        s = score(sel)
        names = ",".join(p2n[nodes[i]] for i in sel)
        rows.append({"n": n, "buses": names, "lambda_min": s,
                     "ratio_vs_shipped": s / base})
        print(f"{n:>3}  {names:<38}{s:>12.6f}{s / base:>11.2f}x")

    # what maximum electrical separation -- the rule T43-T47 used -- scores
    sp = dict(nx.all_pairs_dijkstra_path_length(h, weight="z"))
    print(f"\n{'n':>3}  {'max-separation placement (old rule)':<38}{'lambda_min':>12}"
          f"{'vs shipped':>12}")
    for n in (2, 6):
        best = None
        for seed in pool:
            s2 = [seed]
            for _ in range(n - 1):
                s2.append(max((b for b in pool if b not in s2),
                              key=lambda b: sum(sp[nodes[b]].get(nodes[x], 0)
                                                for x in s2)))
            sc = float(np.mean([sp[nodes[a]].get(nodes[b], 0)
                                for a, b in itertools.combinations(s2, 2)]))
            if best is None or sc > best[0]:
                best = (sc, s2)
        s = score(best[1])
        names = ",".join(p2n[nodes[i]] for i in best[1])
        rows.append({"n": n, "buses": names, "lambda_min": s,
                     "ratio_vs_shipped": s / base, "rule": "max-separation"})
        print(f"{n:>3}  {names:<38}{s:>12.6f}{s / base:>11.2f}x")

    (args.out / "grounded_laplacian.json").write_text(
        json.dumps(rows, indent=1), encoding="utf-8")
    print(f"\nwrote {args.out / 'grounded_laplacian.json'}")


if __name__ == "__main__":
    main()
