"""T36: kappa_os as a function of KPplim, at the ship inner loops.

T31 measured this curve at the inner loops that shipped then, (KPi, KIi) =
(0.20, 5.0). T32/T33 moved them to (0.10, 3.0), and T33 showed the two are not
separable -- stability, and therefore the shape of this curve, is a property of
the pair. So the published curve does not describe the configuration that now
ships and has to be re-measured rather than re-labelled.

Why the curve rather than a constant. kappa_os = (f0 - f_nadir)/(f0 - f_ss) is a
quantity every reduced GFM model predicts to be exactly 1.000 by construction:
Ducoin's (33) is first order for an all-IBR system with fast PLLs, which is this
fleet, and a first-order step response is monotone. Measuring it as a *function*
of a converter control parameter is the result -- a single number invites the
reading that it is a property of the plant.

The grid spans the REGFM_A1-conformant window and both sides of it. Conformance
needs the unit conversion (`src/analytical/regf1_droop.py`): REGFM_A1 sums its
overload branch into omega alongside `mp`, REGF1 sums into power and multiplies
by `w0*wdrp`, so the comparable quantity is `kppmax_eff = m_p * KPplim =
droop_r * KPplim`. At R = 0.05 the specification's 0.005-0.05 range is
KPplim in [0.1, 1.0], with its example value at KPplim = 0.2.

Two disturbance sizes, both reused from T31 so the two curves are directly
comparable.

Run:
    uv run python experiments/t36_kappa_curve.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.t31_kpplim_mechanism import one  # noqa: E402
from src.phasor.build_case import CaseSpec, Disturbance  # noqa: E402
from src.phasor.metrics import SecurityBand  # noqa: E402

OUT = ROOT / "artifacts" / "T36_kappa_curve_shipC"

# Below the conformant window, across it, and above it. 1.0 is what ships.
KP_GRID = (0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0)
DP_GRID = (0.64, 1.1793)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    band = SecurityBand()
    base = CaseSpec(disturbance=Disturbance(kind="load_step", t_event=1.0,
                                            step_mw=1.0, step_bus_name="76"))
    print(f"inner loops from CaseSpec defaults: KPi={base.kp_i}, KIi={base.ki_i}; "
          f"R={base.droop_r}\n")

    rows = []
    for kp in KP_GRID:
        for dp in DP_GRID:
            r = one(base, kp, dp, band)
            r["kppmax_eff"] = base.droop_r * kp
            r["conformant"] = bool(0.005 <= r["kppmax_eff"] <= 0.05)
            rows.append(r)
            print(f"KPplim={kp:<6} kppmax_eff={r['kppmax_eff']:.4f} "
                  f"{'CONF' if r['conformant'] else '    '} dP={dp:<7} "
                  f"kappa={r.get('kappa_os', float('nan')):.4f} "
                  f"rocof={r['rocof_window_hz_s']:.4f} "
                  f"conv={r['tds_converged']} set={r['settled']}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(args.out / "kappa_curve.csv", index=False)
    print(f"\nwrote {args.out / 'kappa_curve.csv'}")
    ok = df[(df.tds_converged) & (df.settled)]
    print("\nkappa_os (converged & settled only):")
    print(ok.pivot_table(index=["kp_plim", "kppmax_eff"], columns="dP_mw",
                         values="kappa_os").round(4).to_string())
    print("\nrocof_window [Hz/s]:")
    print(ok.pivot_table(index=["kp_plim", "kppmax_eff"], columns="dP_mw",
                         values="rocof_window_hz_s").round(4).to_string())


if __name__ == "__main__":
    main()
