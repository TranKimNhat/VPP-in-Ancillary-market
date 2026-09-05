"""T45: pick a test feeder on the criterion the weak-grid literature actually uses.

T44 killed Lambda = X_conv/X_feeder as a design criterion -- the two levers that
lower it moved the outcome in opposite directions. Dozein, Pal & Mancarella
(IEEE TPWRS 37(5), 2022) use the standard one instead: the short-circuit ratio
at the inverter terminal, with SCR < 3 defining a weak grid. The real 22 kV
Australian PV-rich feeder they study sits at SCR ~ 2.

SCR is measured, not invented, and it is the metric AEMO and the IBR-stability
literature report, so it can be computed for any candidate feeder before
committing to one.

In an all-GFM island there is no external grid, so each unit's Thevenin source
is the *other units*, reached through the feeder:

    Z_th,i = parallel over j != i of ( z_feeder(i,j) + x_conv,j )
    SCR_i  = 1 / ( Z_th,i * S_i / V^2 )      [i.e. Z_th on unit i's own base]

That structure is the point. With several comparable units the parallel
combination is small whatever the feeder does, so the units make each other
stiff and SCR stays high. The feeder only dominates when z_feeder(i,j) exceeds
x_conv,j, which on a short distribution feeder it does not.

Two questions:

  1. Which candidate feeder, carrying this GFM fleet, has SCR < 3?
  2. If IEEE 123 is kept, what has to change to get there?

Run:
    uv run python experiments/t45_scr_feeder_survey.py
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import warnings
from pathlib import Path

import networkx as nx
import numpy as np
import pandapower.networks as pn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.analytical.accessibility import build_branch_graph  # noqa: E402
from src.phasor.build_case import (CaseSpec, V_BASE_KV, _base_net,  # noqa: E402
                                   gfm_table)

OUT = ROOT / "artifacts" / "T45_scr_survey"
X_CONV = 0.21          # x_f + x_tr, per unit on the unit's own base
S_BASE = 1.0           # MVA, as in build_case
WEAK = 3.0             # Dozein's threshold


def graph_of(net):
    g0 = build_branch_graph(net)
    return g0.subgraph(max(nx.connected_components(g0), key=len)).copy()


def scr(g, buses, s_mva, v_kv, x_conv=X_CONV):
    """SCR at each GFM terminal of an all-GFM island. See module docstring."""
    zb = v_kv ** 2 / S_BASE
    sp = dict(nx.all_pairs_dijkstra_path_length(g, weight="z"))
    xc = [x_conv / s for s in s_mva]                      # system pu
    out = []
    for i, bi in enumerate(buses):
        y = sum(1.0 / (sp[bi][bj] / zb + xc[j])
                for j, bj in enumerate(buses) if j != i and bj in sp[bi])
        z_th = 1.0 / y if y > 0 else float("inf")         # system pu
        out.append(1.0 / (z_th * s_mva[i]))               # own base -> SCR
    return np.array(out)


def spread_buses(g, n):
    """n buses at maximum mean pairwise electrical separation, greedy over seeds."""
    sp = dict(nx.all_pairs_dijkstra_path_length(g, weight="z"))
    nodes = sorted(sp)
    best = None
    for seed in nodes:
        sel = [seed]
        for _ in range(n - 1):
            sel.append(max((b for b in nodes if b not in sel),
                           key=lambda b: sum(sp[b][x] for x in sel)))
        sc = float(np.mean([sp[a][b] for a, b in itertools.combinations(sel, 2)]))
        if best is None or sc > best[0]:
            best = (sc, sel)
    return best[1]


def candidates():
    """Feeders the comparable literature uses, plus ours."""
    out = {}
    net = _base_net()
    rows = gfm_table(CaseSpec())
    n2p = {str(r["name"]): i for i, r in net.bus.iterrows()}
    out["IEEE 123 (ours, 4.16 kV)"] = (
        graph_of(net), V_BASE_KV, [n2p[r["bus_name"]] for r in rows],
        [r["s_mva"] for r in rows], float(net.load.p_mw.sum()))

    for label, fn, kv in (("IEEE 33 (12.66 kV)", pn.case33bw, 12.66),
                          ("CIGRE MV (20 kV)", pn.create_cigre_network_mv, 20.0),
                          ("CIGRE LV (0.4 kV)", pn.create_cigre_network_lv, 0.4),
                          ("MV Oberrhein (20 kV)", pn.mv_oberrhein, 20.0)):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                n = fn()
            g = graph_of(n)
            load = float(n.load.p_mw.sum())
            out[label] = (g, kv, None, None, load)
        except Exception as exc:
            print(f"  skip {label}: {type(exc).__name__}: {exc}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rows = []

    print("=== Q1: candidate feeders, carrying a fleet sized at 1.25x their own load ===")
    print("    (six equal units at maximum mean separation, so every feeder is given")
    print("     its best case; ours is also shown at its actual ratings and buses)\n")
    print(f"{'feeder':<26}{'load MW':>9}{'fleet MVA':>11}{'z_ij mean pu':>14}"
          f"{'SCR min':>9}{'SCR max':>9}  weak?")
    for label, (g, kv, buses, s_mva, load) in candidates().items():
        if buses is None:
            buses = spread_buses(g, 6)
            s_mva = [1.25 * load / 6] * 6
        zb = kv ** 2 / S_BASE
        sp = dict(nx.all_pairs_dijkstra_path_length(g, weight="z"))
        zij = float(np.mean([sp[a][b] for a, b in itertools.combinations(buses, 2)])) / zb
        v = scr(g, buses, s_mva, kv)
        rows.append({"feeder": label, "load_mw": load, "fleet_mva": float(sum(s_mva)),
                     "z_ij_mean_pu": zij, "scr_min": float(v.min()),
                     "scr_max": float(v.max())})
        print(f"{label:<26}{load:>9.3f}{sum(s_mva):>11.3f}{zij:>14.5f}"
              f"{v.min():>9.2f}{v.max():>9.2f}  {'YES' if v.max() < WEAK else 'no'}")

    print("\n=== Q2: keeping IEEE 123, what changes reach SCR < 3? ===\n")
    net = _base_net()
    g = graph_of(net)
    fleet = 4.3624
    print(f"{'n GFM':>6}{'S_i MVA':>10}{'z_ij mean pu':>14}{'SCR':>9}  weak?")
    for n in (2, 3, 4, 5, 6):
        buses = spread_buses(g, n)
        s_mva = [fleet / n] * n
        zb = V_BASE_KV ** 2 / S_BASE
        sp = dict(nx.all_pairs_dijkstra_path_length(g, weight="z"))
        zij = float(np.mean([sp[a][b] for a, b in itertools.combinations(buses, 2)])) / zb
        v = scr(g, buses, s_mva, V_BASE_KV)
        rows.append({"feeder": f"IEEE 123, n={n}", "n_gfm": n,
                     "fleet_mva": fleet, "z_ij_mean_pu": zij,
                     "scr_min": float(v.min()), "scr_max": float(v.max())})
        print(f"{n:>6}{fleet/n:>10.4f}{zij:>14.5f}{v.max():>9.2f}"
              f"  {'YES' if v.max() < WEAK else 'no'}")

    print(f"\n{'x_conv needed for SCR < 3 at n = 6 (all else fixed)':<52}", end="")
    buses6 = spread_buses(g, 6)
    lo, hi = 0.21, 20.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if scr(g, buses6, [fleet / 6] * 6, V_BASE_KV, x_conv=mid).max() < WEAK:
            hi = mid
        else:
            lo = mid
    print(f"x_conv >= {hi:.3f} pu   (REGFM_A1 caps X_L at 0.25)")

    (args.out / "scr_survey.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
    print(f"\nwrote {args.out / 'scr_survey.json'}")


if __name__ == "__main__":
    main()
