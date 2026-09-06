"""T51: does island partitioning give us a topology-dependent variable at last?

Six structural quantities have now been measured invariant or uninformative on
this feeder: tie configuration (T22, 0.0000 MW), disturbance location (T23, 0.07%
over 84 runs), A_GFM (T1, unreproducible), SCR (T45-T47, a grid-following
threshold misapplied), the grounded Laplacian's lambda_min (T48-T49, corr +0.14)
and the PLL-dynamic arm (T50, spread 0.000). Every one of them varied the
*routing* of a single island while holding its generator set fixed -- and the
closed form

    dP_max = (f0 - f_min) * sum(S_g) / (kappa_os * f0 * R)

has no routing term in it. It has sum(S_g).

Partitioning does move sum(S_g). T47 counted the switch space: seven controllable
switches, 128 states, of which the valid ones are those where every load-carrying
island is radial and contains a grid-forming anchor. In a multi-island state each
island gets its own sum(S_g), its own load, its own GFL fleet -- and therefore its
own boundary. This run measures that.

Two targets, and they are not of equal standing:

  A. dP_max per island. Varies *by construction*: dP_max is linear in sum(S_g)
     and the islands hold different GFM subsets. Reported for completeness, but
     a spread here proves nothing that arithmetic did not already guarantee.

  B. security margin = dP_max(island) - largest credible loss in that island.
     This is the real dependent variable. It is not arithmetic: it depends on how
     load, GFL blocks and GFM units co-locate under a given switch state, which
     is a property of the topology and nothing else. It can be negative, and if
     it is, that partition is insecure against its own largest DER.

Pre-registered before running, in the T50 style:

    H1 (a topology-dependent variable exists):  relative spread of B >= 20%
    H0 (drop the direction):                    relative spread of B  < 5%

Scope limit, stated up front: this is a screening sweep on the closed form, not a
time-domain campaign. The closed form's nadir branch is used; the RoCoF branch
(which binds above f_band = 1.039 Hz at the *fleet* sum(S_g)) is not re-derived
per island, so an island reported secure here may still be RoCoF-limited. Any
partition this run flags has to be confirmed in ANDES before it is claimed.

Run:
    uv run python experiments/t51_partition_capability.py
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

from experiments.t47_gfm_count_sweep import controllable, line_graph  # noqa: E402
from src.analytical.accessibility import build_branch_graph  # noqa: E402
from src.phasor.build_case import CaseSpec, _base_net, gfm_table  # noqa: E402
from src.phasor.metrics import SecurityBand  # noqa: E402

OUT = ROOT / "artifacts" / "T51_partition_capability"

F0_HZ = 60.0
# kappa_os at the ship configuration, post-REGFM_A1 conformance. Recorded in the
# SecurityBand docstring of src/phasor/metrics.py, not in a field, so it is
# restated here rather than imported.
KAPPA_OS = 1.005


def dp_max_mw(s_g_mva: float, f_band_hz: float, droop_r: float) -> float:
    """Closed-form nadir-limited boundary for an island holding `s_g_mva`."""
    return f_band_hz * s_g_mva / (KAPPA_OS * F0_HZ * droop_r)


def islands_of(h: nx.Graph, load_buses: set[int]) -> list[set[int]]:
    """Connected components that carry load. De-energised pockets are ignored."""
    return [set(c) for c in nx.connected_components(h)
            if any(b in c for b in load_buses)]


def state_report(h, load_buses, gfm_at, load_at, sgen_at, band, droop_r):
    """Per-island rows for one switch state, plus the state's validity."""
    rows, valid = [], True
    for comp in islands_of(h, load_buses):
        sub = h.subgraph(comp)
        radial = sub.number_of_edges() == sub.number_of_nodes() - 1
        s_g = sum(s for b, s in gfm_at.items() if b in comp)
        n_gfm = sum(1 for b in gfm_at if b in comp)
        if not radial or n_gfm == 0:
            valid = False
        p_load = sum(p for b, p in load_at.items() if b in comp)
        gfl = [p for b, p in sgen_at.items() if b in comp]
        p_gfl = sum(gfl)
        # Largest credible loss inside the island: the biggest single GFL block.
        # A unit's own reserve may not cover its own outage, so this loss is
        # measured against the boundary the *remaining* fleet sets -- which here
        # is the GFM fleet, since the GFL blocks carry no frequency support.
        loss = max(gfl) if gfl else 0.0
        dpm = dp_max_mw(s_g, band.f_max_hz - F0_HZ, droop_r) if n_gfm else 0.0
        # The 0.20 MW block above is an artefact of how the 16 aggregated DER
        # were built, and a finer aggregation would shrink it. `trip_frac` is the
        # same question asked without that artefact: what fraction of the
        # island's own GFL output has to be lost, together, to reach the
        # boundary. A value <= 1 means a common-mode trip of the island's DER --
        # the GB 2019 failure mode -- is inside the credible set.
        rows.append({
            "n_bus": len(comp), "n_gfm": n_gfm, "radial": radial,
            "s_g_mva": s_g, "p_load_mw": p_load, "p_gfl_mw": p_gfl,
            "p_gfm_dispatch_mw": p_load - p_gfl,
            "headroom_mw": s_g - (p_load - p_gfl),
            "largest_gfl_block_mw": loss,
            "dp_max_mw": dpm, "margin_mw": dpm - loss,
            "trip_frac": (dpm / p_gfl) if p_gfl > 0 else float("inf"),
        })
    return rows, valid


def spread(xs: list[float]) -> tuple[float, float]:
    """(absolute spread, relative spread vs mean magnitude) -- 0 if degenerate."""
    if not xs:
        return 0.0, 0.0
    a = float(np.max(xs) - np.min(xs))
    m = float(np.mean(np.abs(xs)))
    return a, (a / m if m > 0 else 0.0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    spec = CaseSpec()
    band = SecurityBand()
    net = _base_net()
    g0 = build_branch_graph(net)
    live = g0.subgraph(max(nx.connected_components(g0), key=len)).copy()
    gl = line_graph(net).subgraph(live.nodes).copy()
    sw = controllable(net, live)

    p2n = {i: str(r["name"]) for i, r in net.bus.iterrows()}
    n2p = {v: k for k, v in p2n.items()}
    gfm_at = {n2p[r["bus_name"]]: r["s_mva"] for r in gfm_table(spec)}
    load_at, sgen_at = {}, {}
    for _, r in net.load.iterrows():
        b = int(r.bus)
        if b in live:
            load_at[b] = load_at.get(b, 0.0) + float(r.p_mw) * spec.load_scale
    for _, r in net.sgen.iterrows():
        b = int(r.bus)
        if b in live:
            sgen_at[b] = sgen_at.get(b, 0.0) + float(r.p_mw)
    load_buses = set(load_at)

    s_fleet = sum(gfm_at.values())
    f_band = band.f_max_hz - F0_HZ
    print(f"live buses {live.number_of_nodes()} | switches {len(sw)} -> "
          f"{2 ** len(sw)} states | GFM {len(gfm_at)} units, "
          f"sum(S_g) = {s_fleet:.4f} MVA")
    print(f"band f_min {band.f_min_hz} Hz (f_band {f_band} Hz), R {spec.droop_r}, "
          f"kappa_os {KAPPA_OS}")
    print(f"fleet dP_max = {dp_max_mw(s_fleet, f_band, spec.droop_r):.4f} MW  "
          f"(load {sum(load_at.values()):.4f} MW, GFL {sum(sgen_at.values()):.4f} MW, "
          f"largest GFL block {max(sgen_at.values()):.4f} MW)\n")

    states, all_rows = [], []
    for st in itertools.product((0, 1), repeat=len(sw)):
        h = gl.copy()
        for on, (a, b) in zip(st, sw):
            if on:
                h.add_edge(a, b, z=0.0)
        rows, valid = state_report(h, load_buses, gfm_at, load_at, sgen_at,
                                   band, spec.droop_r)
        rec = {"state": list(st), "valid": valid, "n_island": len(rows),
               "islands": rows}
        states.append(rec)
        if valid:
            all_rows.extend(rows)

    valid_states = [s for s in states if s["valid"]]
    single = [s for s in valid_states if s["n_island"] == 1]
    multi = [s for s in valid_states if s["n_island"] > 1]
    print(f"valid states {len(valid_states)}/{len(states)}  "
          f"({len(single)} single-island, {len(multi)} multi-island)")

    # Distinct islands, keyed on their bus set, so a partition appearing in many
    # switch states is counted once. Islands, not states, are the sample here.
    seen, uniq = set(), []
    for s in valid_states:
        h_rows = s["islands"]
        for r in h_rows:
            k = (r["n_bus"], r["n_gfm"], round(r["s_g_mva"], 6),
                 round(r["p_load_mw"], 6), round(r["p_gfl_mw"], 6))
            if k not in seen:
                seen.add(k)
                uniq.append(r)
    print(f"distinct load-carrying islands across valid states: {len(uniq)}\n")

    dpm = [r["dp_max_mw"] for r in uniq]
    mar = [r["margin_mw"] for r in uniq]
    hed = [r["headroom_mw"] for r in uniq]
    trf = [r["trip_frac"] for r in uniq if np.isfinite(r["trip_frac"])]
    a_dpm, r_dpm = spread(dpm)
    a_mar, r_mar = spread(mar)
    a_hed, r_hed = spread(hed)
    a_trf, r_trf = spread(trf)

    print(f"{'target':<34}{'min':>10}{'max':>10}{'spread':>10}{'rel':>9}  verdict")
    for name, xs, (a, rel), note in (
        ("A  dP_max per island [MW]", dpm, (a_dpm, r_dpm), "arithmetic"),
        ("B  security margin [MW]", mar, (a_mar, r_mar), "PRE-REGISTERED"),
        ("C  critical trip fraction [-]", trf, (a_trf, r_trf), "no block artefact"),
        ("   headroom [MW]", hed, (a_hed, r_hed), ""),
    ):
        v = "H1" if rel >= 0.20 else ("H0" if rel < 0.05 else "inconclusive")
        print(f"{name:<34}{min(xs):>10.4f}{max(xs):>10.4f}{a:>10.4f}"
              f"{rel:>8.1%}  {v}{'  (' + note + ')' if note else ''}")

    insecure = [r for r in uniq if r["margin_mw"] < 0]
    no_head = [r for r in uniq if r["headroom_mw"] < 0]
    cm_exposed = [r for r in uniq if r["trip_frac"] <= 1.0]
    print(f"\nislands insecure against their own largest GFL block: "
          f"{len(insecure)}/{len(uniq)}")
    print(f"islands where losing ALL their own GFL exceeds dP_max:  "
          f"{len(cm_exposed)}/{len(uniq)}")
    print(f"islands with negative GFM headroom:                    "
          f"{len(no_head)}/{len(uniq)}")
    if insecure:
        print(f"\n{'buses':>7}{'nGFM':>6}{'S_g':>9}{'load':>9}{'GFL':>8}"
              f"{'head':>9}{'dP_max':>9}{'loss':>8}{'margin':>9}")
        for r in sorted(insecure, key=lambda x: x["margin_mw"])[:15]:
            print(f"{r['n_bus']:>7}{r['n_gfm']:>6}{r['s_g_mva']:>9.4f}"
                  f"{r['p_load_mw']:>9.4f}{r['p_gfl_mw']:>8.4f}"
                  f"{r['headroom_mw']:>9.4f}{r['dp_max_mw']:>9.4f}"
                  f"{r['largest_gfl_block_mw']:>8.4f}{r['margin_mw']:>9.4f}")

    out = {
        "question": "does island partitioning produce a topology-dependent "
                    "security variable, where routing did not?",
        "pre_registered": {"H1": "relative spread of margin >= 0.20",
                           "H0": "relative spread of margin < 0.05"},
        "inputs": {"f0_hz": F0_HZ, "f_min_hz": band.f_min_hz,
                   "f_band_hz": f_band, "droop_r": spec.droop_r,
                   "kappa_os": KAPPA_OS, "s_fleet_mva": s_fleet,
                   "load_mw": sum(load_at.values()),
                   "gfl_mw": sum(sgen_at.values()),
                   "n_switch": len(sw), "n_state": len(states)},
        "counts": {"valid": len(valid_states), "single_island": len(single),
                   "multi_island": len(multi), "distinct_islands": len(uniq),
                   "insecure_islands": len(insecure),
                   "negative_headroom_islands": len(no_head)},
        "spread": {"dp_max": {"abs": a_dpm, "rel": r_dpm},
                   "margin": {"abs": a_mar, "rel": r_mar},
                   "headroom": {"abs": a_hed, "rel": r_hed}},
        "islands": uniq,
        "scope_limit": "closed-form nadir branch only; RoCoF branch not "
                       "re-derived per island; no time-domain confirmation",
    }
    (args.out / "results.json").write_text(json.dumps(out, indent=1),
                                           encoding="utf-8")
    print(f"\nwrote {args.out / 'results.json'}")


if __name__ == "__main__":
    main()
