"""T53: the capability envelope -- how much FFR can a partition actually sell?

T51/T52 fixed the droop at the shipped R = 0.05 and asked what each partition
could withstand. But R is a *control* variable, and the closed form

    dP_max = (f0 - f_min) * sum(S_g) / (kappa_os * f0 * R)

is inversely proportional to it. So a VPP has two substitutable levers on the same
quantity: change which converters are in the island (planning, sum(S_g)) or
stiffen the droop (control, R). The exchange rate between them is what a VPP
actually needs, and neither lever is free -- R is bounded below by small-signal
stability.

This run measures that bound. For every reactive-feasible partition, R is swept
downward and the spectrum computed until the island loses stability; R_min is the
last stable value, and

    C_max(y) = (f0 - f_min) * sum(S_g) / (kappa_os * f0 * R_min(y))

is the largest capability that partition can be certified for. No time-domain run
is needed -- this is eigenvalues only.

Pre-registered before running:

    H1 (R_min is configuration-dependent):  relative spread of R_min >= 20%
    H0 (it is not):                          < 5%

H0 is the more consequential outcome and is not the boring one: it would mean the
capability ceiling is set by sum(S_g) alone, the two levers do not interact, and
a learned surrogate for R_min has no target. H1 means the envelope is genuinely
two-dimensional and the exchange rate has to be reported per partition.

Scope: ANDES 2.0 positive-sequence small-signal. The stability verdict excludes
the free reference angle (see `t32_eig_map.modes`). The inner-loop gains are the
shipped pair, which T33 measured stable in 77 of 81 cells but *not* at the corner
xf = 0.05, R = 0.02 -- so an instability found at small R may be the inner loop
rather than the droop, and the mode is reported so the two can be told apart.

Run:
    uv run python experiments/t53_capability_envelope.py
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

from experiments.t32_eig_map import modes  # noqa: E402
from experiments.t51_partition_capability import F0_HZ, KAPPA_OS  # noqa: E402
from experiments.t52_island_boundary_andes import (enumerate_islands,  # noqa: E402
                                                   feasible, island_spec)
from src.phasor.build_case import CaseSpec, build_system  # noqa: E402
from src.phasor.build_case import _base_net  # noqa: E402
from src.phasor.metrics import SecurityBand  # noqa: E402

OUT = ROOT / "artifacts" / "T53_capability_envelope"

# Descending: the first value that fails brackets R_min from below.
R_GRID = [0.050, 0.040, 0.030, 0.025, 0.020, 0.015, 0.012, 0.010, 0.008, 0.006]
ZETA_MIN = 0.02          # a mode below this is not a usable operating point


def spectrum(base, net, isl, sw, r) -> dict | None:
    """Eigenvalues of one island at one droop value. No time domain."""
    spec = replace(island_spec(base, net, isl, sw, 0.0), droop_r=float(r))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            ss, _ = build_system(spec)
            ss.PFlow.run()
            if not ss.PFlow.converged:
                return None
            ss.TDS.init()
            ss.EIG.calc_As()
            mu, _ = ss.EIG.calc_eig()
        except Exception:
            return None
    return modes(np.asarray(mu))


def r_min_of(base, net, isl, sw, log: list[dict]) -> dict:
    """Smallest droop on the grid that is still stable, then one bisection refine."""
    last_ok, first_bad = None, None
    for r in R_GRID:
        m = spectrum(base, net, isl, sw, r)
        rec = {"r": r, "ok": m is not None}
        if m is not None:
            z = m.get("zeta_min", float("nan"))
            rec.update({"max_re": m["max_re"], "zeta_min": z,
                        "n_unstable": m["n_unstable"],
                        "f_crit_hz": m.get("f_crit_hz", float("nan"))})
            stable = m["max_re"] < 0.0 and (not np.isfinite(z) or z >= ZETA_MIN)
        else:
            stable = False
        rec["stable"] = bool(stable)
        log.append(rec)
        if stable:
            last_ok = r
        else:
            first_bad = r
            break
    if last_ok is None:
        return {"r_min": float("nan"), "note": "unstable at the shipped R already"}
    if first_bad is None:
        return {"r_min": R_GRID[-1], "note": "stable to the bottom of the grid"}
    lo, hi = first_bad, last_ok               # lo unstable, hi stable
    for _ in range(4):                        # 4 halvings ~ 6% resolution
        mid = 0.5 * (lo + hi)
        m = spectrum(base, net, isl, sw, mid)
        z = m.get("zeta_min", float("nan")) if m else float("nan")
        ok = bool(m and m["max_re"] < 0.0 and (not np.isfinite(z) or z >= ZETA_MIN))
        log.append({"r": mid, "ok": m is not None, "stable": ok,
                    "max_re": m["max_re"] if m else float("nan"),
                    "zeta_min": z, "refine": True})
        if ok:
            hi = mid
        else:
            lo = mid
    return {"r_min": hi, "note": f"bracketed on [{lo:.5f}, {hi:.5f}]"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    base, band, net = CaseSpec(), SecurityBand(), _base_net()
    f_band = band.f_max_hz - F0_HZ
    isls, sw, net = enumerate_islands(net, base)

    fz = {id(r): feasible(base, net, r, sw) for r in isls}
    qok = [r for r in isls if fz[id(r)]["pf_converged"] and fz[id(r)]["mu_q"] <= 1.0]
    # collapse islands that are physically the same case
    picks, seen = [], set()
    for r in sorted(qok, key=lambda x: x["s_g_mva"]):
        k = (round(r["s_g_mva"], 6), round(r["p_load_mw"], 6),
             round(r["p_gfl_mw"], 6), len(r["gfm_keys"]))
        if k not in seen:
            seen.add(k)
            picks.append(r)
    print(f"{len(picks)} physically distinct Q-feasible islands "
          f"(of {len(qok)} Q-feasible, {len(isls)} total)\n")

    rows = []
    print(f"{'island':<13}{'S_g':>8}{'R_min':>9}{'C@0.05':>9}{'C_max':>9}"
          f"{'gain':>7}  note")
    for isl in picks:
        log: list[dict] = []
        t0 = time.time()
        res = r_min_of(base, net, isl, sw, log)
        rmin = res["r_min"]
        c_ship = f_band * isl["s_g_mva"] / (KAPPA_OS * F0_HZ * base.droop_r)
        c_max = (f_band * isl["s_g_mva"] / (KAPPA_OS * F0_HZ * rmin)
                 if np.isfinite(rmin) else float("nan"))
        rows.append({**{k: v for k, v in isl.items() if k != "buses"},
                     "n_bus": len(isl["buses"]), "r_min": rmin,
                     "c_at_ship_mw": c_ship, "c_max_mw": c_max,
                     "gain": c_max / c_ship if np.isfinite(c_max) else float("nan"),
                     "note": res["note"], "wall_s": round(time.time() - t0, 1),
                     "sweep": log})
        print(f"{str(len(isl['buses']))+'b/'+str(len(isl['gfm_keys']))+'g':<13}"
              f"{isl['s_g_mva']:>8.4f}{rmin:>9.5f}{c_ship:>9.4f}{c_max:>9.4f}"
              f"{rows[-1]['gain']:>7.2f}  {res['note'][:34]}")

    rm = np.array([r["r_min"] for r in rows if np.isfinite(r["r_min"])])
    spread = float(rm.max() - rm.min()) if rm.size else float("nan")
    rel = float(spread / rm.mean()) if rm.size and rm.mean() > 0 else float("nan")
    v = "H1" if rel >= 0.20 else ("H0" if rel < 0.05 else "inconclusive")
    print(f"\nR_min over {rm.size} islands: {rm.min():.5f} .. {rm.max():.5f}  "
          f"spread {spread:.5f} ({rel:.1%})  -> {v}")
    if v == "H0":
        print("  H0: the ceiling is set by sum(S_g) alone. The envelope is "
              "one-dimensional and R_min needs no surrogate.")
    print(f"capability at shipped R: {min(r['c_at_ship_mw'] for r in rows):.4f} .. "
          f"{max(r['c_at_ship_mw'] for r in rows):.4f} MW")
    cm = [r["c_max_mw"] for r in rows if np.isfinite(r["c_max_mw"])]
    if cm:
        print(f"capability at R_min:     {min(cm):.4f} .. {max(cm):.4f} MW  "
              f"(mean gain {np.mean([r['gain'] for r in rows if np.isfinite(r['gain'])]):.2f}x)")

    (args.out / "results.json").write_text(json.dumps(
        {"question": "is the capability ceiling set by sum(S_g) alone, or does the "
                     "droop stability floor move with the configuration?",
         "pre_registered": {"H1": "relative spread of R_min >= 0.20",
                            "H0": "< 0.05"},
         "verdict": v, "zeta_min_threshold": ZETA_MIN, "r_grid": R_GRID,
         "kappa_os": KAPPA_OS, "f_band_hz": f_band, "r_shipped": base.droop_r,
         "spread": {"abs": spread, "rel": rel},
         "islands": rows}, indent=1), encoding="utf-8")
    print(f"\nwrote {args.out / 'results.json'}")


if __name__ == "__main__":
    main()
