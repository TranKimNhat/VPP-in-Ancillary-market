"""T47: how many grid-forming units? Three axes that pull against each other.

Three separate measurements have to be satisfied at once, and T45/T46 showed
they do not agree:

  SCR < 3          Dozein's weak-grid threshold. In an all-GFM island each unit's
                   Thevenin source is the other units, so SCR ~ (n-1)/x_conv and
                   FEWER units is weaker (better).
  valid configs    a reconfiguration is valid only if every energised island
                   contains a grid-forming unit, so MORE units means more of the
                   switch space is reachable.
  damping          n = 2 measured zeta_min 0.020 against 0.156 at n = 6.

The configuration space is counted over switch operations, not arbitrary branch
cuts. Stripped of its switches this feeder is five sections (37, 36, 19, 16 and
16 buses) joined by five sectionalising switches, plus two usable ties; three
further ties land on buses `build_case` drops as isolated. That is seven
controllable switches, so 2^7 = 128 switch states, of which the valid ones are
those where every island carrying load is radial and has a grid-forming anchor.

Run:
    uv run python experiments/t47_gfm_count_sweep.py
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.t32_eig_map import modes  # noqa: E402
from src.analytical.accessibility import build_branch_graph  # noqa: E402
from src.phasor.build_case import (CaseSpec, Disturbance, V_BASE_KV,  # noqa: E402
                                   _base_net, build_system, gfm_table)

OUT = ROOT / "artifacts" / "T47_gfm_count"
FLEET_MVA = 4.3624
BESS_MW, BESS_MWH = 3.4140, 6.8281
X_CONV = 0.21
WEAK = 3.0
SRC = ROOT / "artifacts" / "placement" / "official_placement_v4_rescaled.json"


def line_graph(net):
    """Lines and transformers only: the sections a switch operation joins."""
    g = nx.Graph()
    g.add_nodes_from(int(b) for b in net.bus.index)
    for _, ln in net.line.iterrows():
        if bool(ln.get("in_service", True)):
            z = abs(complex(ln["r_ohm_per_km"], ln["x_ohm_per_km"])) * float(ln["length_km"])
            if g.has_edge(int(ln.from_bus), int(ln.to_bus)):
                g[int(ln.from_bus)][int(ln.to_bus)]["z"] = min(
                    g[int(ln.from_bus)][int(ln.to_bus)]["z"], z)
            else:
                g.add_edge(int(ln.from_bus), int(ln.to_bus), z=z)
    for _, tr in net.trafo.iterrows():
        if bool(tr.get("in_service", True)):
            g.add_edge(int(tr.hv_bus), int(tr.lv_bus), z=0.0)
    return g


def controllable(net, live):
    """Bus-bus switches whose both ends survive `build_case`'s isolated-bus drop."""
    bb = net.switch[net.switch.et == "b"]
    return [(int(r.bus), int(r.element)) for _, r in bb.iterrows()
            if live.has_node(int(r.bus)) and live.has_node(int(r.element))]


def count_valid(gl, sw, load_buses, gfm_buses):
    """Switch states whose every load-carrying island is radial and GFM-anchored."""
    n_ok = 0
    for state in itertools.product((0, 1), repeat=len(sw)):
        h = gl.copy()
        for on, (a, b) in zip(state, sw):
            if on:
                h.add_edge(a, b, z=0.0)
        ok = True
        for comp in nx.connected_components(h):
            sub = h.subgraph(comp)
            has_load = any(b in comp for b in load_buses)
            if not has_load:
                continue                                   # de-energised, ignore
            if sub.number_of_edges() != sub.number_of_nodes() - 1:
                ok = False; break                          # not radial
            if not any(b in comp for b in gfm_buses):
                ok = False; break                          # island with no anchor
        n_ok += ok
    return n_ok


def scr_of(live, buses, s_mva, zb):
    sp = dict(nx.all_pairs_dijkstra_path_length(live, weight="z"))
    xc = [X_CONV / s for s in s_mva]
    out = []
    for i, bi in enumerate(buses):
        y = sum(1.0 / (sp[bi][bj] / zb + xc[j])
                for j, bj in enumerate(buses) if j != i and bj in sp[bi])
        out.append(1.0 / ((1.0 / y) * s_mva[i]) if y > 0 else float("inf"))
    return float(max(out))


def best_placement(live, sp, zb, n, pool):
    """n buses maximising mean pairwise separation, i.e. the lowest-SCR layout."""
    best = None
    for seed in pool:
        sel = [seed]
        for _ in range(n - 1):
            sel.append(max((b for b in pool if b not in sel),
                           key=lambda b: sum(sp[b][x] for x in sel)))
        sc = float(np.mean([sp[a][b] for a, b in itertools.combinations(sel, 2)]))
        if best is None or sc > best[0]:
            best = (sc, sel)
    return best[1]


def eig(placement, keys):
    spec = CaseSpec(placement=placement, gfm_keys=keys,
                    disturbance=Disturbance(kind="gen_loss", t_event=1.0,
                                            step_mw=0.5, step_bus_name="76"))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ss, _ = build_system(spec)
        ss.PFlow.run()
        if not ss.PFlow.converged:
            return None
        ss.TDS.init(); ss.EIG.calc_As()
        mu, _ = ss.EIG.calc_eig()
    return modes(np.asarray(mu))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    net = _base_net()
    g0 = build_branch_graph(net)
    live = g0.subgraph(max(nx.connected_components(g0), key=len)).copy()
    gl = line_graph(net).subgraph(live.nodes).copy()
    sw = controllable(net, live)
    load_buses = {int(b) for b in net.load.bus.unique() if b in live}
    zb = V_BASE_KV ** 2 / 1.0
    sp = dict(nx.all_pairs_dijkstra_path_length(live, weight="z"))
    pool = sorted(sp)
    p2n = {i: str(r["name"]) for i, r in net.bus.iterrows()}
    n2p = {v: k for k, v in p2n.items()}

    print(f"live: {live.number_of_nodes()} buses; sections without switches: "
          f"{nx.number_connected_components(gl)}; controllable switches: {len(sw)} "
          f"-> {2 ** len(sw)} switch states\n")

    d0 = json.loads(SRC.read_text(encoding="utf-8"))
    proto = d0["gfm"]["G1"]
    rows = []
    print(f"{'n':>3}{'buses':<34}{'SCR':>7}{'valid cfg':>11}{'/128':<6}"
          f"{'maxRe':>10}{'zeta_min':>10}  verdict")

    for n in (2, 3, 4, 5, 6):
        sel = best_placement(live, sp, zb, n, pool)
        s_mva = [FLEET_MVA / n] * n
        scr = scr_of(live, sel, s_mva, zb)
        nv = count_valid(gl, sw, load_buses, set(sel))
        d = json.loads(json.dumps(d0))
        keys = tuple(f"G{i+1}" for i in range(n))
        d["gfm"] = {k: dict(proto, bus=int(p2n[b]), inverter_mva=FLEET_MVA / n,
                            bess_mw=BESS_MW / n, bess_mwh=BESS_MWH / n)
                    for k, b in zip(keys, sel)}
        p = args.out / f"placement_{n}gfm.json"
        p.write_text(json.dumps(d, indent=1), encoding="utf-8")
        m = eig(p, keys)
        names = ",".join(p2n[b] for b in sel)
        rows.append({"n": n, "buses": names, "scr": scr, "valid_cfg": nv,
                     "max_re": None if m is None else m["max_re"],
                     "zeta_min": None if m is None else m.get("zeta_min"),
                     "stable": None if m is None else m["stable"]})
        if m is None:
            print(f"{n:>3}{names:<34}{scr:>7.2f}{nv:>11}{'/128':<6}   PFLOW FAILED")
        else:
            print(f"{n:>3}{names:<34}{scr:>7.2f}{nv:>11}{'/128':<6}"
                  f"{m['max_re']:>10.4f}{m.get('zeta_min', float('nan')):>10.4f}"
                  f"  {'STABLE' if m['stable'] else 'UNSTABLE'}"
                  f"{'  [SCR ok]' if scr < WEAK else ''}")

    cur = [n2p[r["bus_name"]] for r in gfm_table(CaseSpec())]
    print(f"\nshipped 6 GFM at {','.join(p2n[b] for b in cur)}: "
          f"SCR {scr_of(live, cur, [r['s_mva'] for r in gfm_table(CaseSpec())], zb):.2f}, "
          f"valid cfg {count_valid(gl, sw, load_buses, set(cur))}/128")

    (args.out / "sweep.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
    print(f"\nwrote {args.out / 'sweep.json'}")


if __name__ == "__main__":
    main()
