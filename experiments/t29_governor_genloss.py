"""T29: does the governor family move the `gen_loss` boundary with the diesel online?

The outstanding item flagged by claims-register entry N7. T27 swapped `TGOV1` for
`GAST` and got agreement to 1e-10 -- but it did so on the **diesel-off** boundary,
where the governor is disconnected together with the machine at t_event and
therefore cannot influence the outcome by construction. That agreement is a
consequence of the test's structure, not evidence about governor modelling.

T27 has no power over any scenario in which the diesel stays online, and
kappa_os^gen_loss -- the leading constant of Contribution I -- is exactly such a
scenario. Until this runs, that constant sits inside the un-cross-checked region of
decision D2b.

So: the same boundary, the same bisection, the diesel **online through the event**,
one flag changed. Two pre-trip loadings, because a governor that is nearly unloaded
has little to contribute and a null there would prove nothing.

Pre-registered before running:

    H1 (the family matters):  |dP_max(GAST) - dP_max(TGOV1)| > 2% at either loading
    H0 (it does not):         <= 0.5% at both
    between:                  inconclusive

Note what this run is *not*. `TGOV1`/`GAST` are ANDES phasor stand-ins for the
`GGOV1`/`DEGOV1` that decision D2b selected for the EMT model. Agreement between
these two says nothing about that substitution, which stays unverified in either
direction.

Run:
    uv run python experiments/t29_governor_genloss.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.campaign.boundary import bisect  # noqa: E402
from src.phasor.build_case import (CaseSpec, DieselSpec, Disturbance,  # noqa: E402
                                   solve)
from src.phasor.metrics import SecurityBand, extract  # noqa: E402

OUT = ROOT / "artifacts" / "T29_governor_genloss"


def probe(base: CaseSpec, log: list[dict]):
    def f(dp_mw: float):
        spec = replace(base, disturbance=replace(base.disturbance,
                                                 step_mw=float(dp_mw)))
        t0 = time.time()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ss, index, status = solve(spec)
            m, verdict = extract(ss, index, status, spec, SecurityBand())
        rec = {"dp_mw": float(dp_mw), "secure": bool(verdict.secure),
               "f_nadir": m.f_nadir_hz, "f_ss": m.f_ss_hz,
               "rocof": m.rocof_window_hz_s, "v_min": m.v_min_pu,
               "mu_i": m.mu_i, "why": "; ".join(verdict.reasons),
               "wall_s": round(time.time() - t0, 1)}
        log.append(rec)
        print(f"    dP {dp_mw:7.4f}  {'SEC ' if rec['secure'] else 'INSEC'}"
              f"  nadir {m.f_nadir_hz:8.5f}  f_ss {m.f_ss_hz:8.5f}"
              f"  RoCoF {m.rocof_window_hz_s:7.4f}  {rec['why'][:38]}")
        return bool(verdict.secure), rec
    return f


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--p-dg", type=float, nargs="+", default=[0.3, 0.5],
                    help="pre-event diesel output [MW], one bisection per value")
    ap.add_argument("--dp-lo", type=float, default=0.3)
    ap.add_argument("--dp-hi", type=float, default=2.5)
    ap.add_argument("--tol", type=float, default=0.004)
    ap.add_argument("--verify", type=int, default=3)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    rows = []
    for p_dg in args.p_dg:
        for gov in ("TGOV1", "GAST"):
            base = CaseSpec(
                diesel=DieselSpec(bus_name="76", s_mva=1.0, p_mw=float(p_dg),
                                  h_sec=1.0, droop_r=0.05, governor=gov),
                disturbance=Disturbance(kind="gen_loss", t_event=1.0,
                                        step_mw=0.5, step_bus_name="76"),
            )
            print(f"diesel {p_dg:.2f} MW online, governor {gov}")
            log: list[dict] = []
            res = bisect(probe(base, log), args.dp_lo, args.dp_hi,
                         direction="secure_below", tol=args.tol,
                         verify_points=args.verify)
            print(f"  -> dP_max = {res.x_boundary:.5f} MW  "
                  f"({res.n_eval} evals, monotone={res.monotone})\n")
            rows.append({"p_dg_mw": float(p_dg), "governor": gov,
                         "found": res.found, "dp_max_mw": res.x_boundary,
                         "n_eval": res.n_eval, "monotone": res.monotone,
                         "note": res.note, "probes": log})

    print(f"{'P_dg':>6}{'TGOV1':>12}{'GAST':>12}{'diff [kW]':>12}{'rel':>9}  verdict")
    verdicts = []
    for p_dg in args.p_dg:
        a = next(r for r in rows if r["p_dg_mw"] == p_dg and r["governor"] == "TGOV1")
        b = next(r for r in rows if r["p_dg_mw"] == p_dg and r["governor"] == "GAST")
        if not (a["found"] and b["found"]):
            print(f"{p_dg:>6.2f}{'':>12}{'':>12}   boundary not bracketed")
            verdicts.append("inconclusive")
            continue
        d = b["dp_max_mw"] - a["dp_max_mw"]
        rel = abs(d) / a["dp_max_mw"]
        v = "H1" if rel > 0.02 else ("H0" if rel <= 0.005 else "inconclusive")
        verdicts.append(v)
        print(f"{p_dg:>6.2f}{a['dp_max_mw']:>12.5f}{b['dp_max_mw']:>12.5f}"
              f"{1000 * d:>12.3f}{rel:>8.3%}  {v}")

    overall = ("H1" if "H1" in verdicts else
               "H0" if all(v == "H0" for v in verdicts) else "inconclusive")
    print(f"\noverall: {overall}")
    (args.out / "results.json").write_text(json.dumps(
        {"question": "does the governor family move the gen_loss boundary with "
                     "the diesel online?",
         "pre_registered": {"H1": "|rel diff| > 0.02 at either loading",
                            "H0": "<= 0.005 at both"},
         "verdict": overall, "per_loading": verdicts,
         "scope": "ANDES phasor; TGOV1/GAST are stand-ins for the GGOV1/DEGOV1 "
                  "of D2b and say nothing about that substitution",
         "runs": rows}, indent=1), encoding="utf-8")
    print(f"wrote {args.out / 'results.json'}")


if __name__ == "__main__":
    main()
