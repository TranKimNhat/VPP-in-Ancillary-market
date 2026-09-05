"""T31: is the 22.7% frequency undershoot caused by `KPplim`, or by the inner loops?

T30's Task-0 gate killed the Ducoin parameter mapping and left one open question
that matters more than the cross-check did. The measured overshoot ratio

    kappa_os = (f0 - f_nadir) / (f0 - f_ss) = 1.227

is a quantity every reduced GFM model predicts to be exactly 1.000: Ducoin's
(33) is *first order* for an all-IBR system with fast PLLs, which is this fleet
exactly, and a first-order step response is monotone. So the undershoot comes
from something the reduced models do not carry. Two candidates:

  (a) `KPplim`. The droop input is `PIplim_y - Psen_y`, not `Pref - Pe`, so the
      P-limiter PI sits in the droop's forward path even when nothing is
      saturated. Deriving it from `regf1.py` (Psen/Psig/PIplim/dw):

          dw/dPe = -w0*wdrp * (1 + KIplim*Tpm + s*Tpm*(1+KPplim))
                            / ((1 + s*Tr)(1 + s*Tpm))

      With the shipped KIplim = 0 the numerator zero sits at
      -1/(Tpm*(1+KPplim)) = -6.67 rad/s, *below* both poles (-40, -200): a
      lead-lag with 4.2x mid-band gain, which overshoots.
  (b) The inner loops. Ducoin states the assumption his reduction needs directly
      above (2): "It is assumed that the inner control loops of inverters track
      their references perfectly." REGF1 has real inner loops, and this study
      deliberately slowed them (KPi 0.5 -> 0.20, KIi 20 -> 5.0) to hold
      Re(lambda) <= 0.

The two are separable in one sweep, because the transfer function makes a sharp
prediction: at KPplim = 0 the zero lands exactly on -1/Tpm and *cancels the
pole*, leaving `-w0*wdrp/(1 + s*Tr)` -- a pure first-order lag, i.e. Ducoin's
model with w_Pf = 1/Tr. So if (a) is the mechanism, kappa_os -> 1.000 there.
Whatever undershoot survives KPplim = 0 belongs to (b).

Why this outranks the cross-check: kappa_os enters the security boundary in the
denominator,

    dP_max = (f0 - f_min) * sum(S_g) / (kappa_os * f0 * R)

so if kappa_os is a function of KPplim then dP_max = 1.1851 MW is a function of
KPplim too, and so are T22-T26. And KPplim = 5 is not a specification value:
PNNL-35110 gives `kppmax` = 0.01 pu, normal range 0.005-0.05.

CORRECTION (T32). This docstring originally called KPplim in {0.005, 0.01, 0.05}
"the specification range" and KPplim = 5 a 100x deviation. Both are wrong:
`KPplim` is dimensionless while `kppmax` is a frequency-per-power gain, because
REGFM_A1 sums the overload branch straight into omega (alongside `mp`, which is
why the specification gives both the same 0.005-0.05 range) whereas REGF1 sums
it into power and multiplies by `w0*wdrp` afterwards. The comparable quantity is

    kppmax_eff = m_p * KPplim = R * KPplim = 0.05 * KPplim

so the conformant window is KPplim in [0.1, 1.0], the specification example is
KPplim = 0.2, and the shipped 5.0 is 5x above the range -- not 100x. The values
<= 0.05 swept here sit 2-20x *below* the range. They are still a real result --
T32 confirms they are linearly unstable, not merely intractable -- but they are
not the specification range. See `artifacts/T32_eig_map/README.md`.

Registered before the run (T30 protocol):

    kappa_os -> 1.00 +/- 0.03 at KPplim = 0   ->  (a) is the mechanism
    kappa_os stays ~1.22                      ->  (b) is the mechanism
    lands between, e.g. 1.10                  ->  both; step 2 needed to split

Run:
    uv run python experiments/t31_kpplim_mechanism.py --smoke
    uv run python experiments/t31_kpplim_mechanism.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.phasor.build_case import CaseSpec, Disturbance, solve  # noqa: E402
from src.phasor.metrics import SecurityBand, extract, sliding_rocof  # noqa: E402

OUT = ROOT / "artifacts" / "T31_kpplim_mechanism"
F0 = 60.0
ROCOF_WINDOW_S = 0.5

# The specification example, the top of its normal range, and the ANDES default,
# plus the analytic cancellation point and two values above the default.
KP_GRID = (0.0, 0.01, 0.05, 1.0, 2.0, 5.0, 10.0)
# Three of the unsaturated T23 points, spanning ~3x. Reusing their dP keeps the
# new kappa_os directly comparable with the 13 points already measured.
DP_GRID = (0.64, 1.1793, 1.82)


def coi_metrics(ss, index, spec) -> dict:
    """kappa_os and the identified first-order time constant, on the COI.

    `f_nadir` in `metrics.extract` is the min over *all* monitored buses, which
    is the right thing for a security test but mixes in the inter-bus term that
    Ducoin's (32)/(33) explicitly neglects. Both are reported here; the bus
    spread is what says whether they can be read as the same number.
    """
    n = ss.dae.n
    t = np.asarray(ss.dae.ts.t, dtype=float)
    f_hz = F0 * ss.dae.ts.xy[:, n + ss.BusROCOF.f.a]
    coi = f_hz.mean(axis=1)

    tail = t >= t[-1] - 2.0
    dss = F0 - float(coi[tail].mean())
    dip = F0 - float(coi.min())
    spread = float((f_hz.max(axis=1) - f_hz.min(axis=1)).max())

    rocof_win = sliding_rocof(t, coi, ROCOF_WINDOW_S)
    # Ducoin (33): df(t) = df_ss (1 - exp(-t/tau)). Monotone, so the sliding
    # window's max sits at the smallest admissible span, W/2.
    dt_min = 0.5 * ROCOF_WINDOW_S
    g = rocof_win * dt_min / dss if dss > 1e-6 else np.nan
    tau = -dt_min / np.log(1 - g) if 0 < g < 1 else np.nan

    return {
        "df_ss_coi_hz": dss,
        "dip_coi_hz": dip,
        "kappa_os": dip / dss if dss > 1e-6 else np.nan,
        "bus_spread_hz": spread,
        "bus_spread_pct_of_dss": 100 * spread / dss if dss > 1e-6 else np.nan,
        "rocof_coi_window_hz_s": rocof_win,
        "tau_eq_s": tau,
    }


def one(base: CaseSpec, kp: float, dp: float, band: SecurityBand) -> dict:
    spec = replace(base, kp_plim=kp,
                   disturbance=replace(base.disturbance, step_mw=float(dp)))
    t0 = time.time()
    ss, index, status = solve(spec)
    m, verdict = extract(ss, index, status, spec, band, rocof_window_s=ROCOF_WINDOW_S)
    row = {"kp_plim": kp, "dP_mw": dp, "wallclock_s": round(time.time() - t0, 2),
           "ok": m.ok, "settled": m.settled, "tds_converged": status["tds_converged"],
           "n_units_saturated": m.n_units_saturated,
           "f_nadir_minbus_hz": m.f_nadir_hz, "df_ss_minbus_hz": m.df_ss_hz,
           "rocof_window_hz_s": m.rocof_window_hz_s, "mu_i": m.mu_i, "mu_p": m.mu_p,
           "secure": verdict.secure}
    if m.ok:
        row.update(coi_metrics(ss, index, spec))
    # The analytic droop-path prediction, for the same KPplim.
    zero = -1.0 / (base.t_pm * (1 + kp))
    row["droop_zero_rad_s"] = zero
    row["pole_zero_cancels"] = bool(abs(zero + 1.0 / base.t_pm) < 0.02 / base.t_pm)
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true",
                    help="one run at the shipped KPplim=5, to check the harness "
                         "reproduces the T23 kappa_os = 1.2276 before spending 5 min")
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    band = SecurityBand()
    base = CaseSpec(disturbance=Disturbance(kind="load_step", t_event=1.0,
                                            step_mw=1.0, step_bus_name="76"))

    if args.smoke:
        r = one(base, 5.0, 1.1793, band)
        print(json.dumps(r, indent=2, default=str))
        print("\nT23 reference at this dP: kappa_os = 1.2276, df_ss = -0.8110 Hz")
        return

    rows = []
    for kp in KP_GRID:
        for dp in DP_GRID:
            r = one(base, kp, dp, band)
            rows.append(r)
            print(f"KPplim={kp:<6} dP={dp:<7} "
                  f"kappa_os={r.get('kappa_os', float('nan')):.4f} "
                  f"df_ss={r.get('df_ss_coi_hz', float('nan')):.4f} "
                  f"tau={r.get('tau_eq_s', float('nan')):.4f} "
                  f"nsat={r['n_units_saturated']} {r['wallclock_s']}s", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(args.out / "metrics.csv", index=False)
    print(f"\nwrote {args.out / 'metrics.csv'}")
    print("\nkappa_os:\n",
          df.pivot_table(index="kp_plim", columns="dP_mw",
                         values="kappa_os").round(4).to_string())


if __name__ == "__main__":
    main()
