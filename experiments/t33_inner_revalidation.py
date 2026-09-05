"""T33: re-earn the inner-loop stability claim at the ship configuration B.

`build_case.py:152-166` asserts that slowing the inner loops to (KPi, KIi) =
(0.20, 5.0) "restores Re(lambda) <= 0 for every deployment (2/5/6 GFM), every
headroom, xf in [0.05, 0.20], R in [0.02, 0.05] and both load models". That
sentence is the paper's entire small-signal stability claim, and T32 invalidates
it, because shipping the REGFM_A1-conformant P-limiter gain requires moving the
inner loops to (0.10, 3.0).

Configuration B, chosen in T32 against the pre-registered rule:

    KPplim = 0.2   (kppmax_eff = m_p * KPplim = 0.010, REGFM_A1's example value)
    KPi, KIi = 0.10, 3.0
    KPv, KIv = 3.0, 10.0   (unchanged; T32 section 3 shows this is the good region)

This sweeps the same axes the original claim covers and re-earns it, or does not.
The current shipped configuration is swept alongside as a control: it must
reproduce the documented result, otherwise the harness is wrong rather than the
finding.

Eigenvalues only -- power flow, TDS initialisation, reduced state matrix at the
pre-event equilibrium. No time-domain integration.

This is a gate. `build_case.py` defaults are not touched until every cell of
configuration B comes back stable, and any cell that does not is reported rather
than dropped.

Run:
    uv run python experiments/t33_inner_revalidation.py
"""

from __future__ import annotations

import argparse
import sys
import time
import warnings
from dataclasses import replace
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.t32_eig_map import modes  # noqa: E402
from src.phasor.build_case import CaseSpec, Disturbance, build_system  # noqa: E402

OUT = ROOT / "artifacts" / "T33_inner_revalidation"

# (label, kp_plim, kp_i, ki_i)
CONFIGS = (
    ("current", 5.0, 0.20, 5.0),   # control: must reproduce build_case.py:152-166
    ("A",       1.0, 0.20, 5.0),   # conformant at the top of the range, inner loops kept
    ("B",       0.2, 0.10, 3.0),   # the specification example value
    ("C",       1.0, 0.10, 3.0),   # top of the range with the slowed inner loops
)

DEPLOYMENTS = {                    # as used by t01_agfm.py
    "2gfm": ("G1", "G2"),
    "5gfm": ("G1", "G2", "G3", "G4", "G5"),
    "6gfm": ("G1", "G2", "G3", "G4", "G5", "G6"),
}
XF = (0.05, 0.10, 0.15, 0.20)      # the interval the claim covers; 0.15 ships
R = (0.02, 0.03, 0.05)             # the interval the claim covers; 0.05 ships
LOAD_P2Z = (0.0, 1.0)              # constant power / constant impedance
# `p_head_mw = None` is the full BESS rating (3.414 MW). The rest span the range
# the campaign bisects over, down to the smallest value T26 could represent.
HEADROOM = (None, 1.7320, 0.8910, 0.0500)


def eig_one(spec: CaseSpec) -> dict:
    row = {"pflow_converged": False, "eig_ok": False, "error": ""}
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


def cells():
    """The claim's axes. Full cross product over deployment/xf/R/load at nominal
    headroom, plus a headroom axis at the nominal point -- headroom enters the
    linearisation only through the P-limiter's anti-windup bounds, so it does not
    need to be crossed with everything else."""
    for dep, xf, r, p2z in product(DEPLOYMENTS, XF, R, LOAD_P2Z):
        yield {"deployment": dep, "x_f_pu": xf, "droop_r": r,
               "load_p2z": p2z, "p_head_mw": None}
    for dep, head in product(DEPLOYMENTS, HEADROOM[1:]):
        yield {"deployment": dep, "x_f_pu": 0.15, "droop_r": 0.05,
               "load_p2z": 0.0, "p_head_mw": head}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    # `gen_loss` rather than `load_step`: `build_case` rejects `load_step` once
    # `load_p2z > 0` (an `Alter` on `Ppf` is inert under constant impedance), and
    # a zero-MW `gen_loss` adds a PQ device carrying no power, so the pre-event
    # equilibrium this linearises around is the undisturbed one in every cell.
    base = CaseSpec(disturbance=Disturbance(kind="gen_loss", t_event=1.0,
                                            step_mw=0.0, step_bus_name="76"))
    rows = []
    grid = list(cells())
    print(f"{len(CONFIGS)} configurations x {len(grid)} cells\n")

    for label, kp_plim, kp_i, ki_i in CONFIGS:
        n_bad = 0
        for c in grid:
            spec = replace(base, kp_plim=kp_plim, kp_i=kp_i, ki_i=ki_i,
                           gfm_keys=DEPLOYMENTS[c["deployment"]],
                           x_f_pu=c["x_f_pu"], droop_r=c["droop_r"],
                           load_p2z=c["load_p2z"], p_head_mw=c["p_head_mw"])
            r = eig_one(spec)
            r.update(c, config=label, kp_plim=kp_plim, kp_i=kp_i, ki_i=ki_i)
            rows.append(r)
            if not r.get("stable"):
                n_bad += 1
                print(f"  [{label}] NOT STABLE  {c['deployment']} xf={c['x_f_pu']} "
                      f"R={c['droop_r']} p2z={c['load_p2z']} head={c['p_head_mw']}  "
                      f"maxRe={r.get('max_re', float('nan')):.4f} "
                      f"{r['error'][:40]}", flush=True)
        ok = sum(1 for x in rows if x["config"] == label and x.get("stable"))
        print(f"[{label}] {ok}/{len(grid)} cells stable, {n_bad} not\n", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(args.out / "revalidation.csv", index=False)
    print(f"wrote {args.out / 'revalidation.csv'}")

    for label, *_ in CONFIGS:
        d = df[df.config == label]
        good = d[d.stable == True]  # noqa: E712
        print(f"\n[{label}] stable {len(good)}/{len(d)}"
              f"   worst max_re = {d.max_re.max():.4f}"
              f"   min zeta = {d.zeta_min.min():.4f}"
              f"   pflow failures = {int((~d.pflow_converged).sum())}")


if __name__ == "__main__":
    main()
