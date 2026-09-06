"""T52: does the closed form survive on a sub-island? ANDES confirmation of T51.

T51 screened all 92 valid switch states with the closed form

    dP_max = (f0 - f_min) * sum(S_g) / (kappa_os * f0 * R)

and found the security margin varies by 282% across the 32 distinct load-carrying
islands, with 8 of them unable to survive the loss of their own largest DER block
while every one of them still has positive headroom. That is the first
topology-dependent variable this campaign has found, after six invariant ones.

It is also, so far, arithmetic. Three things in the screen are unverified outside
the six-GFM fleet they were measured on:

  1. kappa_os = 1.005 is a *fleet* measurement (T21, post-REGFM_A1 conformance).
     A one-GFM island is a different plant and may have a different overshoot.
  2. The screen evaluates only the nadir branch. On a small island the RoCoF
     criterion may bind first, in which case dP_max is lower than screened and the
     8 insecure islands are an undercount, not an artefact.
  3. Power flow may simply not converge on a small island whose GFM sits far from
     its load.

This run bisects the true boundary in ANDES on four islands spanning the range and
compares. The full feeder is included as the control arm: its boundary is already
known (T21), so if it does not reproduce here the island builder is wrong, not the
physics.

Pre-registered before running:

    H1 (closed form transfers):      |error| <= 5% on every island
    H0 (it does not):                |error| >  20% on any island
    anything between:                inconclusive, report as such

Island construction: every line with an endpoint outside the island is opened, and
every switch not internal-and-closed is opened, so `_isolated_buses` drops the rest
of the feeder and ANDES sees the island alone. The first GFM in the island becomes
the Slack, as `build_system` always does.

Run:
    uv run python experiments/t52_island_boundary_andes.py --smoke
    uv run python experiments/t52_island_boundary_andes.py
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
import warnings
from dataclasses import replace
from pathlib import Path

import networkx as nx
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.t47_gfm_count_sweep import controllable, line_graph  # noqa: E402
from experiments.t51_partition_capability import KAPPA_OS, F0_HZ, dp_max_mw  # noqa: E402
from src.analytical.accessibility import build_branch_graph  # noqa: E402
from src.campaign.boundary import bisect  # noqa: E402
from src.phasor.build_case import (CaseSpec, Disturbance, _base_net,  # noqa: E402
                                   build_system, gfm_table, solve)
from src.phasor.metrics import SecurityBand, extract  # noqa: E402

OUT = ROOT / "artifacts" / "T52_island_boundary"


def enumerate_islands(net, spec):
    """Every distinct load-carrying island over the 128 switch states.

    Same construction as T51, but the bus set and the switch state that produced
    it are kept, because this run has to rebuild the island as a case.
    """
    g0 = build_branch_graph(net)
    live = g0.subgraph(max(nx.connected_components(g0), key=len)).copy()
    gl = line_graph(net).subgraph(live.nodes).copy()
    sw = controllable(net, live)
    p2n = {i: str(r["name"]) for i, r in net.bus.iterrows()}
    n2p = {v: k for k, v in p2n.items()}

    gfm_bus = {n2p[r["bus_name"]]: (r["key"], r["s_mva"]) for r in gfm_table(spec)}
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

    seen, out = set(), []
    for st in itertools.product((0, 1), repeat=len(sw)):
        h = gl.copy()
        for on, (a, b) in zip(st, sw):
            if on:
                h.add_edge(a, b, z=0.0)
        comps = [set(c) for c in nx.connected_components(h)
                 if any(b in c for b in load_buses)]
        ok = True
        for c in comps:
            sub = h.subgraph(c)
            if sub.number_of_edges() != sub.number_of_nodes() - 1:
                ok = False
            if not any(b in c for b in gfm_bus):
                ok = False
        if not ok:
            continue
        for c in comps:
            keys = [gfm_bus[b][0] for b in sorted(c) if b in gfm_bus]
            s_g = sum(gfm_bus[b][1] for b in c if b in gfm_bus)
            p_load = sum(p for b, p in load_at.items() if b in c)
            gfl = {b: p for b, p in sgen_at.items() if b in c}
            k = (len(c), tuple(sorted(keys)), round(p_load, 6),
                 round(sum(gfl.values()), 6))
            if k in seen:
                continue
            seen.add(k)
            dpm = dp_max_mw(s_g, 0.5, spec.droop_r)
            out.append({
                "buses": sorted(c), "state": list(st), "gfm_keys": tuple(keys),
                "s_g_mva": s_g, "p_load_mw": p_load,
                "p_gfl_mw": sum(gfl.values()),
                "largest_gfl_block_mw": max(gfl.values()) if gfl else 0.0,
                "dp_max_pred_mw": dpm,
                "margin_pred_mw": dpm - (max(gfl.values()) if gfl else 0.0),
                # the biggest load bus in the island: where the event is placed
                "event_bus_name": p2n[max((b for b in c if b in load_at),
                                          key=lambda b: load_at[b])],
            })
    return out, sw, net


def island_open_elements(net, buses, state, sw) -> tuple[str, ...]:
    """Open everything that is not internal to `buses` and closed in `state`."""
    inside = set(buses)
    closed = {frozenset(p) for on, p in zip(state, sw) if on}
    op = [f"l{int(i)}" for i, ln in net.line.iterrows()
          if not (int(ln.from_bus) in inside and int(ln.to_bus) in inside)]
    for i, s in net.switch.iterrows():
        a, b = int(s.bus), int(s.element)
        keep = (s.et == "b" and a in inside and b in inside
                and frozenset((a, b)) in closed)
        if not keep:
            op.append(f"s{int(i)}")
    return tuple(op)


def island_spec(base: CaseSpec, net, isl, sw, dp_mw: float) -> CaseSpec:
    return replace(
        base,
        topology=f"isl{len(isl['buses'])}b_{len(isl['gfm_keys'])}gfm",
        open_elements=island_open_elements(net, isl["buses"], isl["state"], sw),
        gfm_keys=tuple(isl["gfm_keys"]),
        disturbance=Disturbance(kind="gen_loss", t_event=1.0, step_mw=float(dp_mw),
                                step_bus_name=isl["event_bus_name"]),
    )


def probe(base, net, isl, sw, band, log):
    """A bisection predicate over dP, on one island."""
    def f(dp_mw: float):
        spec = island_spec(base, net, isl, sw, dp_mw)
        t0 = time.time()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ss, index, status = solve(spec)
            try:
                m, verdict = extract(ss, index, status, spec, band)
            except ValueError as exc:
                # `metrics.extract` reduces over the settling window without
                # checking that it is non-empty, so a TDS that dies at the event
                # itself ("Simulation terminated at t=1.0000 s") raises instead of
                # returning a verdict. Pre-existing and not touched here; on some
                # partitions it is the *normal* outcome, so it is recorded as the
                # insecure result it plainly is rather than allowed to stop a
                # 32-island sweep. Worth fixing in `metrics.py` separately.
                rec = {"dp_mw": float(dp_mw), "secure": False,
                       "f_nadir": float("nan"), "f_ss": float("nan"),
                       "rocof": float("nan"), "v_min": float("nan"),
                       "mu_i": float("nan"), "mu_p": float("nan"), "ok": False,
                       "why": f"tds died at the event ({type(exc).__name__})",
                       "wall_s": round(time.time() - t0, 1)}
                log.append(rec)
                print(f"    dP {dp_mw:7.4f}  INSEC  {rec['why']}")
                return False, rec
        rec = {"dp_mw": float(dp_mw), "secure": bool(verdict.secure),
               "f_nadir": m.f_nadir_hz, "f_ss": m.f_ss_hz,
               "rocof": m.rocof_window_hz_s, "v_min": m.v_min_pu,
               "mu_i": m.mu_i, "mu_p": m.mu_p, "ok": m.ok,
               "why": "; ".join(verdict.reasons), "wall_s": round(time.time() - t0, 1)}
        log.append(rec)
        print(f"    dP {dp_mw:7.4f}  {'SEC ' if rec['secure'] else 'INSEC'}"
              f"  nadir {m.f_nadir_hz:7.4f}  RoCoF {m.rocof_window_hz_s:7.4f}"
              f"  vmin {m.v_min_pu:6.4f}  {rec['why'][:44]}")
        return bool(verdict.secure), rec
    return f


def feasible(base, net, isl, sw) -> dict:
    """Is the island operable *before* any event? Power flow only, no TDS.

    T51 screened the frequency boundary and nothing else, on the tacit assumption
    that an island with positive headroom is an operating point. The smoke run
    showed it is not: the 37-bus island anchored on G6 alone solves to v_min
    0.4495 pu with the converter already at mu_I = 1.21, because the fleet was
    sized and split for the *intact* feeder and one 0.3793 MVA unit cannot hold up
    37 buses. That is a reactive-support limit, and it binds before frequency
    does. Screening it here costs one power flow per island.
    """
    spec = island_spec(base, net, isl, sw, 0.0)
    t0 = time.time()
    q_ceil = sum(spec.q_max_pu * r["s_mva"] for r in gfm_table(spec))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            ss, _ = build_system(spec)
            ss.PFlow.run()
            conv = bool(ss.PFlow.converged)
            v = np.asarray(ss.Bus.v.v, dtype=float) if conv else np.array([np.nan])
            # The Slack GFM is deliberately unlimited in the power flow, so a
            # reactive shortfall does not show up as non-convergence -- it shows
            # up here, and then again as a TDS initialisation failure when
            # REGF1's own Qmax clamps it. Measure it rather than wait for that.
            q = 0.0
            for mdl in (ss.Slack, ss.PV):
                for idx, qi in zip(mdl.idx.v, np.asarray(mdl.q.v, dtype=float)):
                    if str(idx).startswith("SG_G") and "GFL" not in str(idx):
                        q += float(qi)
        except Exception as exc:                       # a singular island raises
            return {"pf_converged": False, "v_min_pu": float("nan"),
                    "q_gfm_mvar": float("nan"), "q_ceiling_mvar": q_ceil,
                    "mu_q": float("nan"),
                    "err": f"{type(exc).__name__}: {exc}"[:80],
                    "wall_s": round(time.time() - t0, 1)}
    return {"pf_converged": conv, "v_min_pu": float(np.nanmin(v)),
            "v_max_pu": float(np.nanmax(v)),
            "q_gfm_mvar": q, "q_ceiling_mvar": q_ceil,
            "mu_q": (q / q_ceil) if q_ceil > 0 else float("nan"),
            "err": "", "wall_s": round(time.time() - t0, 1)}


def kappa_from(rec_secure: dict) -> float:
    """kappa_os = (f0 - f_nadir) / (f0 - f_ss), read off a converged run."""
    dn, ds = F0_HZ - rec_secure["f_nadir"], F0_HZ - rec_secure["f_ss"]
    return float(dn / ds) if ds > 1e-9 else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--smoke", action="store_true",
                    help="one run on the worst island, to prove the case builds")
    ap.add_argument("--feasibility", action="store_true",
                    help="pre-event power flow on every island, no TDS")
    ap.add_argument("--tds0", action="store_true",
                    help="with --feasibility: also run a zero-disturbance TDS")
    ap.add_argument("--all", action="store_true",
                    help="bisect every physically distinct Q-feasible island")
    ap.add_argument("--tol", type=float, default=0.005)
    ap.add_argument("--verify", type=int, default=3)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    base = CaseSpec()
    band = SecurityBand()
    net = _base_net()
    isls, sw, net = enumerate_islands(net, base)
    isls.sort(key=lambda r: r["margin_pred_mw"])
    print(f"{len(isls)} distinct islands; margin "
          f"{isls[0]['margin_pred_mw']:.4f} .. {isls[-1]['margin_pred_mw']:.4f} MW\n")

    if args.feasibility:
        print(f"{'island':>10}{'nGFM':>6}{'S_g':>9}{'load':>9}{'GFL':>8}"
              f"{'v_min':>8}{'Q_gfm':>8}{'Q_ceil':>8}{'mu_Q':>7}  pf")
        rows = []
        for isl in sorted(isls, key=lambda r: len(r["buses"])):
            fz = feasible(base, net, isl, sw)
            rows.append({**{k: v for k, v in isl.items() if k != "buses"},
                         "n_bus": len(isl["buses"]), **fz})
            print(f"{len(isl['buses']):>10}{len(isl['gfm_keys']):>6}"
                  f"{isl['s_g_mva']:>9.4f}{isl['p_load_mw']:>9.4f}"
                  f"{isl['p_gfl_mw']:>8.4f}{fz['v_min_pu']:>8.4f}"
                  f"{fz['q_gfm_mvar']:>8.4f}{fz['q_ceiling_mvar']:>8.4f}"
                  f"{fz['mu_q']:>7.3f}  "
                  f"{'ok' if fz['pf_converged'] else 'FAIL'}"
                  f"{'  Q-INFEASIBLE' if fz['mu_q'] > 1 else ''} {fz['err'][:30]}")
        if args.tds0:
            # A power flow that converges is not an operating point. The 89-bus
            # island solves to v ~ 1.0 with mu_Q = 0.607 and still fails at
            # dP = 0, because G6 sits at mu_I = 1.25 before anything happens.
            # Only a zero-disturbance time-domain run finds that.
            print("\nzero-disturbance TDS on the Q-feasible islands:")
            for r in rows:
                if not (r["pf_converged"] and r["mu_q"] <= 1.0):
                    r["op_at_zero"] = None
                    continue
                isl = next(x for x in isls if len(x["buses"]) == r["n_bus"]
                           and tuple(x["gfm_keys"]) == tuple(r["gfm_keys"])
                           and abs(x["s_g_mva"] - r["s_g_mva"]) < 1e-9)
                lg = []
                probe(base, net, isl, sw, band, lg)(0.0)
                r["op_at_zero"] = lg[-1]["secure"]
                r["mu_i_at_zero"] = lg[-1]["mu_i"]
                r["why_at_zero"] = lg[-1]["why"]
            live = [r for r in rows if r.get("op_at_zero")]
            print(f"\noperable at dP = 0: {len(live)}/"
                  f"{sum(1 for r in rows if r['pf_converged'] and r['mu_q'] <= 1.0)}"
                  f" Q-feasible islands ({len(live)}/{len(rows)} of all)")

        okv = [r for r in rows if r["pf_converged"] and r["v_min_pu"] >= band.v_min_pu]
        print(f"\npower flow converged:            "
              f"{sum(r['pf_converged'] for r in rows)}/{len(rows)}")
        print(f"and v_min >= {band.v_min_pu}:                "
              f"{len(okv)}/{len(rows)}")
        (args.out / "feasibility.json").write_text(
            json.dumps({"band_v_min_pu": band.v_min_pu, "islands": rows}, indent=1),
            encoding="utf-8")
        print(f"\nwrote {args.out / 'feasibility.json'}")
        return

    # Only Q-feasible islands can be bisected: a converter asked for more reactive
    # power than its declared ceiling fails TDS initialisation, and the frequency
    # boundary of a case that never initialised is not a measurement. Screened
    # first, and the count is reported because it is a result in its own right.
    fz = {id(r): feasible(base, net, r, sw) for r in isls}
    qok = [r for r in isls if fz[id(r)]["pf_converged"] and fz[id(r)]["mu_q"] <= 1.0]
    print(f"Q-feasible at q_max_pu = {base.q_max_pu}: {len(qok)}/{len(isls)} islands")
    if not qok:
        raise SystemExit("no Q-feasible island to bisect")
    isls = qok

    if args.all:
        # Every physically distinct Q-feasible island. Two islands differing only
        # in a de-energised bus have the same generators, the same load and the
        # same GFL, so they would return the same boundary twice and flatter the
        # error distribution; they are collapsed. What is deliberately *not*
        # collapsed is a repeated sum(S_g) at a different unit count or load --
        # 3.6037 MVA occurs at 4 GFM with 1.975 MW of load and at 5 GFM with
        # 2.385 and 2.940 MW. The closed form says only sum(S_g) sets the
        # boundary, so those three must agree, and that is a direct test of it.
        picks, seen_phys = [], set()
        for r in sorted(isls, key=lambda x: x["s_g_mva"]):
            k = (round(r["s_g_mva"], 6), round(r["p_load_mw"], 6),
                 round(r["p_gfl_mw"], 6), len(r["gfm_keys"]))
            if k not in seen_phys:
                seen_phys.add(k)
                picks.append(r)
        print(f"bisecting {len(picks)} physically distinct islands "
              f"(of {len(isls)} Q-feasible)\n")
    else:
        # worst, the two straddling zero, and the full feeder as the control arm.
        full = max(isls, key=lambda r: len(r["buses"]))
        near = min(isls, key=lambda r: abs(r["margin_pred_mw"]))
        picks, seen_id = [], set()
        for r in (isls[0], near, isls[len(isls) // 2], full):
            k = (len(r["buses"]), r["gfm_keys"])
            if k not in seen_id:
                seen_id.add(k)
                picks.append(r)

    if args.smoke:
        picks = picks[:1]

    results = []
    for isl in picks:
        pred = isl["dp_max_pred_mw"]
        print(f"island {len(isl['buses'])} bus | GFM {','.join(isl['gfm_keys'])} "
              f"| S_g {isl['s_g_mva']:.4f} | load {isl['p_load_mw']:.4f} "
              f"| GFL {isl['p_gfl_mw']:.4f} | event at bus {isl['event_bus_name']}")
        print(f"  closed form predicts dP_max = {pred:.4f} MW")
        log: list[dict] = []
        f = probe(base, net, isl, sw, band, log)

        if args.smoke:
            f(pred * 0.5)
            f(pred)
            f(pred * 1.5)
            results.append({**{k: v for k, v in isl.items() if k != "buses"},
                            "n_bus": len(isl["buses"]), "probes": log})
            break

        lo, hi = max(1e-3, 0.25 * pred), 2.5 * pred
        res = bisect(f, lo, hi, direction="secure_below", tol=args.tol,
                     verify_points=args.verify)
        meas = res.x_boundary
        err = (meas - pred) / pred if (res.found and pred > 0) else float("nan")
        sec = [r for r in log if r["secure"] and r["ok"]]
        kap = kappa_from(max(sec, key=lambda r: r["dp_mw"])) if sec else float("nan")
        binding = next((r["why"] for r in log
                        if not r["secure"] and r["why"]), "")
        print(f"  -> measured dP_max = {meas if res.found else float('nan'):.4f} MW"
              f" | error {100 * err:+.2f}% | kappa_os {kap:.4f}"
              f" | first binding: {binding[:60]}\n")
        results.append({**{k: v for k, v in isl.items() if k != "buses"},
                        "n_bus": len(isl["buses"]),
                        "dp_max_meas_mw": meas, "found": res.found,
                        "rel_err": err, "kappa_os_meas": kap,
                        "binding": binding, "n_eval": res.n_eval,
                        "monotone": res.monotone, "note": res.note,
                        "probes": log})

    if not args.smoke:
        errs = [abs(r["rel_err"]) for r in results
                if r.get("found") and np.isfinite(r["rel_err"])]
        v = ("H1" if errs and max(errs) <= 0.05 else
             "H0" if errs and max(errs) > 0.20 else "inconclusive")
        print(f"{'island':<28}{'pred':>9}{'meas':>9}{'err':>9}{'kappa':>8}  binding")
        for r in results:
            print(f"{str(r['n_bus']) + 'b/' + str(len(r['gfm_keys'])) + 'gfm':<28}"
                  f"{r['dp_max_pred_mw']:>9.4f}"
                  f"{r.get('dp_max_meas_mw', float('nan')):>9.4f}"
                  f"{100 * r.get('rel_err', float('nan')):>8.2f}%"
                  f"{r.get('kappa_os_meas', float('nan')):>8.4f}  "
                  f"{r.get('binding', '')[:40]}")
        print(f"\nworst |error| = {100 * max(errs):.2f}%  -> {v}" if errs
              else "\nno boundary found")
        payload = {"pre_registered": {"H1": "max |rel_err| <= 0.05",
                                      "H0": "max |rel_err| > 0.20"},
                   "verdict": v, "kappa_os_screen": KAPPA_OS,
                   "band": {"f_min_hz": band.f_min_hz,
                            "rocof_max_hz_s": band.rocof_max_hz_s,
                            "v_min_pu": band.v_min_pu},
                   "islands": results}
    else:
        payload = {"smoke": True, "islands": results}

    (args.out / "results.json").write_text(json.dumps(payload, indent=1),
                                           encoding="utf-8")
    print(f"\nwrote {args.out / 'results.json'}")


if __name__ == "__main__":
    main()
