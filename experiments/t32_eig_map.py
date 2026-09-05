"""T32: is there a REGFM_A1-conformant `KPplim` that is also stable?

T31 settled the mechanism -- `KPplim` sets the frequency overshoot, the inner
loops do not -- and left one question that decides how much the result is worth:

    Every `KPplim` inside REGFM_A1's normal range (0.005-0.05), the specification
    example 0.01 included, failed to produce a usable time-domain result at the
    current inner-loop tuning. `solve()` reports "time step reduced to zero" for
    a physically collapsing trajectory and for a numerically intractable one
    alike, so the time domain cannot tell those apart.

If the conformant range is *physically* unstable, the KPplim sensitivity curve
is a result about standardised GFM models. If it merely fails numerically, a
reviewer answers "you swept outside the specification, of course it broke" and
the curve is worth nothing. Eigenvalues separate the two.

Why 2-D rather than a `KPplim` line. This study has changed one thing at a time
and never asked whether its two patches trade off against each other. REGF1's
*shipped* inner-loop gains (KPi=0.5, KIi=20) make this fleet linearly unstable
-- ten modes with Re > 0 at 92-260 Hz -- which is why `build_case` slowed them to
(0.20, 5.0). The conformant `KPplim` range then fails *at that slowed tuning*.
Nobody has asked whether some other point in (KPplim, KPi, KIi) is both
conformant and stable. If one exists, the whole "we ship a tool default outside
specification" problem disappears.

On what "conformant" means here. `KPplim` is dimensionless; REGFM_A1's `kppmax`
is a frequency-per-power gain, because the specification sums the overload
branch straight into omega alongside `mp` while REGF1 sums it into power and
multiplies by `w0*wdrp` afterwards. The comparable quantity is
`kppmax_eff = m_p * KPplim = 0.05 * KPplim`, so the conformant window is
KPplim in [0.1, 1.0] with the specification example at KPplim = 0.2. The grid
below was laid out before that was worked out, so its low end (0.005-0.05) sits
2-20x *below* the range rather than inside it; the cells covering the real
window were added afterwards and are in the README's table 1b.

This is eigenvalue analysis only: power flow, TDS initialisation, then the
reduced state matrix at the pre-event equilibrium. No time-domain integration,
so the disturbance is irrelevant and the grid has no dP axis.

Built-in check on the tool: the shipped inner-loop gains (0.5, 20) must come
back unstable. If that row is stable, the linearisation is wrong, not the
finding.

Decision rule, fixed before the run:

    a conformant cell (KPplim <= 0.05) with max Re(lambda) <= 0 exists
        -> ship the specification value in that cell; rerun T21-T26 there
    no conformant cell is stable
        -> ship the smallest stable KPplim; the specification deviation becomes
           a documented finding about the standardised model, with eigenvalue
           evidence
    either way -> do not ship KPplim = 5, which rests on nothing but the ANDES
                  default, and publish kappa_os and RoCoF as functions of
                  KPplim rather than as constants

Run:
    uv run python experiments/t32_eig_map.py
"""

from __future__ import annotations

import argparse
import sys
import time
import warnings
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.phasor.build_case import CaseSpec, Disturbance, build_system  # noqa: E402

OUT = ROOT / "artifacts" / "T32_eig_map"

# Spans REGFM_A1's normal range (0.005-0.05) and the range T31 could actually
# run (0.5-10). 5.0 is the ANDES default the campaign has been using.
KP_PLIM_GRID = (0.005, 0.01, 0.05, 0.5, 1.0, 2.0, 5.0, 10.0)
# (KPi, KIi). The first is REGF1's shipped pair, documented as unstable here and
# kept as the tool check. The third is what `build_case` currently ships.
INNER_GRID = ((0.50, 20.0), (0.35, 12.0), (0.20, 5.0), (0.10, 3.0))

RE_TOL = 1e-6        # a mode this close to the axis is the reference-angle mode
OSC_TOL = 1e-3       # |Im| below this is a real mode, not an oscillation
LF_HZ = 5.0          # the band kappa_os lives in (tau_eq ~ 0.26 s -> ~0.6 Hz)


def modes(mu: np.ndarray) -> dict:
    """Damping summary of one spectrum.

    The near-zero real mode is the free reference angle of an islanded system
    with no slack frequency -- every GFM's `delta` is defined up to a common
    rotation -- so it is excluded from the stability verdict rather than counted
    as marginal instability.
    """
    finite = mu[np.isfinite(mu)]
    free = np.abs(finite.real) < RE_TOL
    judged = finite[~free]
    osc = judged[np.abs(judged.imag) > OSC_TOL]
    zeta = -osc.real / np.abs(osc) if osc.size else np.array([])
    f_hz = np.abs(osc.imag) / (2 * np.pi) if osc.size else np.array([])

    out = {"n_modes": int(finite.size), "n_free_angle": int(free.sum()),
           "max_re_all": float(finite.real.max()) if finite.size else float("nan"),
           "max_re": float(judged.real.max()) if judged.size else float("nan"),
           "n_unstable": int((judged.real > RE_TOL).sum())}
    out["stable"] = bool(out["n_unstable"] == 0)

    if zeta.size:
        k = int(np.argmin(zeta))
        out["zeta_min"] = float(zeta[k])
        out["f_at_zeta_min_hz"] = float(f_hz[k])
        lf = f_hz < LF_HZ
        if lf.any():
            j = int(np.argmin(np.where(lf, zeta, np.inf)))
            out["zeta_min_lf"] = float(zeta[j])
            out["f_at_zeta_min_lf_hz"] = float(f_hz[j])
    # The unstable mode that matters for the verdict, if any.
    unst = judged[judged.real > RE_TOL]
    if unst.size:
        k = int(np.argmax(unst.real))
        out["f_worst_unstable_hz"] = float(abs(unst[k].imag) / (2 * np.pi))
    return out


def one(base: CaseSpec, kp_plim: float, kp_i: float, ki_i: float) -> dict:
    spec = replace(base, kp_plim=kp_plim, kp_i=kp_i, ki_i=ki_i)
    row = {"kp_plim": kp_plim, "kp_i": kp_i, "ki_i": ki_i,
           "pflow_converged": False, "eig_ok": False, "error": ""}
    t0 = time.time()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ss, index = build_system(spec)
            if not index["ceiling_representable"]:
                row["error"] = "P ceiling below zero"
                return row
            ss.PFlow.run()
            row["pflow_converged"] = bool(ss.PFlow.converged)
            if not row["pflow_converged"]:
                return row
            ss.TDS.init()
            ss.EIG.calc_As()
            mu, _ = ss.EIG.calc_eig()
        row.update(modes(np.asarray(mu)))
        row["eig_ok"] = True
    except Exception as exc:
        row["error"] = f"{type(exc).__name__}: {exc}"
    row["wallclock_s"] = round(time.time() - t0, 2)
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    base = CaseSpec(disturbance=Disturbance(kind="load_step", t_event=1.0,
                                            step_mw=1.0, step_bus_name="76"))
    rows = []
    for kp_i, ki_i in INNER_GRID:
        for kp_plim in KP_PLIM_GRID:
            r = one(base, kp_plim, kp_i, ki_i)
            rows.append(r)
            tag = ("STABLE" if r.get("stable") else
                   f"UNSTABLE n={r.get('n_unstable')}" if r.get("eig_ok") else
                   f"FAIL {r['error'][:40]}")
            print(f"KPi={kp_i:<5} KIi={ki_i:<5} KPplim={kp_plim:<6} "
                  f"maxRe={r.get('max_re', float('nan')):>10.4f} "
                  f"zeta_min={r.get('zeta_min', float('nan')):>7.4f} "
                  f"{tag}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(args.out / "eig_map.csv", index=False)
    print(f"\nwrote {args.out / 'eig_map.csv'}")

    ok = df[df.eig_ok]
    if not ok.empty:
        print("\nmax Re(lambda)   rows = (KPi, KIi), cols = KPplim:")
        print(ok.pivot_table(index=["kp_i", "ki_i"], columns="kp_plim",
                             values="max_re").round(4).to_string())
        print("\nstable?  (conformant KPplim <= 0.05 is the question):")
        print(ok.pivot_table(index=["kp_i", "ki_i"], columns="kp_plim",
                             values="stable").to_string())
        conf = ok[(ok.kp_plim <= 0.05) & (ok.stable)]
        print(f"\nconformant AND stable cells: {len(conf)}")
        if not conf.empty:
            print(conf[["kp_plim", "kp_i", "ki_i", "max_re", "zeta_min"]]
                  .round(5).to_string(index=False))


if __name__ == "__main__":
    main()
