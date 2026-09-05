"""T38 axis 2: does the robust/fragile split hold when only the CONTROL axis moves?

The `x_f` half of T38 did not discriminate: over a 2x range of coupling
reactance `dP_max` moved 0.80%, which is one bisection bracket at tol = 0.02 and
sits inside the threshold-tangency regime T35 characterised. A test where
nothing moves cannot separate a robust quantity from a fragile one.

This axis moves only the inner-loop gains, holding KPplim at the ship value 1.0
and everything else at nominal. That is the direct test of the claim, which is
about quantities set by *converter control* rather than by energy balance.

Read the result against this, from `t20_andes_bisect.run_dp_max`:

    "`P_head^min` at a small step turns out to be the feasibility bound
     `P_head >= dP` and nothing more -- the fleet either can supply the step or
     it cannot, and the droop dynamics never get a say."

So `P_head_min` holding still is expected *by construction*, not evidence. It is
carried here as the negative control -- the quantity that must not move if the
harness is behaving -- and the informative half is whether `dP_max` moves.

Inner-loop pairs are checked for stability before use; anything with
Re(lambda) > 0 at the nominal operating point is skipped rather than bisected,
since T33 already established the stable region and this is not the place to
re-open it.

Run:
    uv run python experiments/t38_inner_axis.py
"""

from __future__ import annotations

import argparse
import sys
import warnings
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.t20_andes_bisect import Runner, run_dp_max, run_p_head  # noqa: E402
from experiments.t32_eig_map import modes  # noqa: E402
from src.phasor.build_case import (CaseSpec, Disturbance,  # noqa: E402
                                   build_system)
from src.phasor.metrics import SecurityBand  # noqa: E402

OUT = ROOT / "artifacts" / "T38_partition"

# Held at the ship value; only the inner loops move.
KP_PLIM = 1.0
# Nominal is (0.10, 3.0). These sit inside the region T33 verified.
INNER = ((0.10, 3.0), (0.12, 3.5), (0.14, 3.0))

BASE_DP_MAX = 1.438574      # ship C, T34
BASE_P_HEAD = 1.094680      # ship C, T34


def eig_ok(spec: CaseSpec) -> tuple[bool, float, float]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ss, index = build_system(spec)
        ss.PFlow.run()
        if not ss.PFlow.converged:
            return False, float("nan"), float("nan")
        ss.TDS.init()
        ss.EIG.calc_As()
        mu, _ = ss.EIG.calc_eig()
    m = modes(np.asarray(mu))
    return bool(m["stable"]), m["max_re"], m.get("zeta_min", float("nan"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

    band = SecurityBand()
    base = CaseSpec(t_end=8.0, t_step=0.002, q_max_pu=0.6, x_f_pu=0.15,
                    droop_r=0.05, i_max_f_pu=2.0, kp_plim=KP_PLIM,
                    disturbance=Disturbance(kind="gen_loss", t_event=1.0,
                                            step_mw=1.1, step_bus_name="76"))
    # what run_dp_max / run_p_head read off `args`
    ba = SimpleNamespace(dp_lo=0.05, dp_hi=3.0, head_lo=0.05, head_hi=3.414,
                         tol=0.02, verify=6)

    rows = []
    for kp_i, ki_i in INNER:
        spec = replace(base, kp_i=kp_i, ki_i=ki_i)
        stable, max_re, zeta = eig_ok(spec)
        tag = f"inner_{kp_i}_{ki_i}".replace(".", "p")
        print(f"\n=== KPi={kp_i} KIi={ki_i}  maxRe={max_re:+.4f} zeta={zeta:.4f} "
              f"{'STABLE' if stable else 'UNSTABLE -- skipped'}", flush=True)
        if not stable:
            rows.append({"kp_i": kp_i, "ki_i": ki_i, "stable": False,
                         "max_re": max_re, "zeta_min": zeta})
            continue

        def campaign(driver, sub):
            """`Runner` collects rows; only `t20`'s own `main` writes them out."""
            out = args.out / f"{tag}_{sub}"
            out.mkdir(parents=True, exist_ok=True)
            r = Runner(band, out, save_raw=False)
            b, _ = driver(r, spec, ba)
            pd.DataFrame(r.rows).to_csv(out / "metrics.csv", index=False)
            pd.DataFrame([b]).to_csv(out / "boundaries.csv", index=False)
            return b

        b1 = campaign(run_dp_max, "dpmax")
        b2 = campaign(run_p_head, "phead")

        rows.append({"kp_i": kp_i, "ki_i": ki_i, "stable": True,
                     "max_re": max_re, "zeta_min": zeta,
                     "dP_max_mw": b1["value"], "P_head_min_mw": b2["value"],
                     "kappa": b2.get("kappa", float("nan"))})
        print(f"  dP_max     = {b1['value']:.6f}  ({100*(b1['value']/BASE_DP_MAX-1):+.2f}% vs ship)")
        print(f"  P_head_min = {b2['value']:.6f}  ({100*(b2['value']/BASE_P_HEAD-1):+.2f}% vs ship)")

    df = pd.DataFrame(rows)
    args.out.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out / "inner_axis.csv", index=False)
    print(f"\nwrote {args.out / 'inner_axis.csv'}")
    print(df.round(6).to_string(index=False))


if __name__ == "__main__":
    main()
