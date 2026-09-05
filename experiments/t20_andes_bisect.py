"""T20: bisect the security boundary on ANDES, at one configuration.

The live-or-die test of plan Part 3-P5. If the bisection brackets a finite
boundary at G0 with six GFM and one disturbance, the campaign is feasible and the
grid sweep can be replaced by ~20 runs per boundary instead of a few thousand.

Two boundaries, one engine (`src.campaign.boundary.bisect`):

    P_head^min     smallest fleet headroom that survives a fixed disturbance
    P_DG,off^max   largest pre-trip diesel output whose loss is survived

Both write the plan Part 5 schema. Every mandated column is present so the file
concatenates with the EMT campaign later; the `*_emt` columns are left empty here
rather than filled with ANDES numbers, because ANDES is a positive-sequence
workhorse and not the ground truth. ANDES results live in `*_andes` columns and
`platform` says so on every row.

Run:
    uv run python experiments/t20_andes_bisect.py --what both
    uv run python experiments/t20_andes_bisect.py --what p_head --dp 1.0
"""

from __future__ import annotations

import argparse
import json
import platform as _platform
import subprocess
import sys
import time
from dataclasses import asdict, replace
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import andes  # noqa: E402

from src.analytical.headroom import df_ss_hz, dp_critical_mw, droop_shares  # noqa: E402
from src.campaign.boundary import bisect  # noqa: E402
from src.phasor.build_case import (  # noqa: E402
    CaseSpec,
    DieselSpec,
    Disturbance,
    S_BASE_MVA,
    gfm_table,
    solve,
)
from src.phasor.metrics import SecurityBand, extract  # noqa: E402

OUT = ROOT / "artifacts" / "T20_andes_bisect"

# Plan Part 5 mandates these columns, in this order.
SCHEMA = [
    "run_id", "config_id", "gfm_deployment", "event_type", "dP_mw", "P_dg_pre",
    "P_head_mw", "soc", "mu_I_ana", "mu_I_emt", "mu_P_ana",
    "f_nadir_ana", "f_nadir_emt", "rocof_ana", "rocof_emt", "v_min_emt", "i_peak_pu",
    "eps_f", "eps_V", "secure_flag", "wallclock_s", "solver", "dt",
]


def analytic_screen(spec: CaseSpec, dp_mw: float) -> dict:
    """The Level-2 screening estimates of plan Part 3, for the same operating point.

    `mu_I_ana` is the T4 estimator Ihat = |S| / |V| evaluated at V = 1 pu with the
    unit's post-event share of dP added to its pre-event dispatch. `f_nadir_ana` is
    left empty: the analytical hybrid model is T8 and does not exist yet, and the
    droop-only steady state is not a nadir.
    """
    rows = gfm_table(spec)
    s = np.array([r["s_mva"] for r in rows])
    p_head = np.array([r["p_head_mw"] for r in rows])
    p0 = np.array([r["p0_mw"] for r in rows])
    share = droop_shares(s, np.full(len(rows), spec.droop_r))

    p_post = p0 + dp_mw * share
    # Q is not predicted by the analytical layer, so the estimator sees active
    # current only. On this feeder that is not a small omission -- reactive current
    # dominates -- and the size of the resulting error is a T4 result in itself.
    mu_i = float(np.max(np.abs(p_post) / (spec.i_max_f_pu * s))) if len(s) else float("nan")
    # Defined as Pe/Pmax so it is comparable with the measured mu_P, which counts
    # the pre-event dispatch inside both numerator and ceiling.
    ceiling = p0 + p_head
    mu_p = float(np.max(p_post / np.where(ceiling > 0, ceiling, np.nan)))
    # `df_ss_hz` returns a magnitude; the measured deviation is signed.
    return {
        "mu_I_ana": mu_i,
        "mu_P_ana": mu_p,
        "mu_P_ana_headroom_only": float(
            np.max(dp_mw * share / np.where(p_head > 0, p_head, np.nan))),
        "df_ss_ana_hz": -float(df_ss_hz(dp_mw, s, spec.droop_r, 60.0)),
        "dP_critical_ana_mw": float(dp_critical_mw(p_head, share)),
    }


class Runner:
    """Evaluates one CaseSpec, records a schema row, and answers secure/insecure."""

    def __init__(self, band: SecurityBand, out: Path, save_raw: bool = True):
        self.band = band
        self.out = out
        self.save_raw = save_raw
        self.rows: list[dict] = []
        self._n = 0

    def __call__(self, spec: CaseSpec, tag: str) -> tuple[bool, dict]:
        self._n += 1
        run_id = f"{tag}_{self._n:03d}"
        t0 = time.time()
        ss, index, status = solve(spec)
        m, verdict = extract(ss, index, status, spec, self.band)
        wall = time.time() - t0

        d = spec.disturbance
        dp_mw = d.step_mw if d.kind in ("gen_loss", "load_step") else (
            spec.diesel.p_mw if spec.diesel else float("nan"))
        ana = analytic_screen(spec, dp_mw if np.isfinite(dp_mw) else 0.0)

        row = {
            "run_id": run_id,
            "config_id": spec.topology,
            "gfm_deployment": f"{len(spec.gfm_keys)}gfm",
            "event_type": d.kind,
            "dP_mw": dp_mw,
            "P_dg_pre": spec.diesel.p_mw if spec.diesel else 0.0,
            "P_head_mw": index["p_head_total_mw"],
            "soc": float("nan"),          # SoC enters only through P_head; see T02
            "mu_I_ana": ana["mu_I_ana"],
            "mu_I_emt": float("nan"),
            "mu_P_ana": ana["mu_P_ana"],
            "f_nadir_ana": float("nan"),  # needs the T8 hybrid model
            "f_nadir_emt": float("nan"),
            "rocof_ana": float("nan"),
            "rocof_emt": float("nan"),
            "v_min_emt": float("nan"),
            "i_peak_pu": m.i_peak_pu,
            "eps_f": float("nan"),
            "eps_V": float("nan"),
            "secure_flag": bool(verdict.secure),
            "wallclock_s": round(wall, 3),
            "solver": f"andes-{andes.__version__}-trapezoid",
            "dt": spec.t_step,
            # --- beyond the mandated schema ---
            "platform": "andes",
            "ok": m.ok,
            "beyond_platform": verdict.beyond_platform,
            "insecure_because": "; ".join(verdict.reasons),
            "f_nadir_andes": m.f_nadir_hz,
            "f_ss_andes": m.f_ss_hz,
            "df_ss_andes": m.df_ss_hz,
            "df_ss_ana": ana["df_ss_ana_hz"],
            "rocof_andes_window": m.rocof_window_hz_s,
            "rocof_andes_device": m.rocof_device_hz_s,
            "v_min_andes": m.v_min_pu,
            "v_max_andes": m.v_max_pu,
            "mu_I_andes": m.mu_i,
            "mu_I_andes_unit": m.mu_i_unit,
            "mu_I_cont_andes": m.mu_i_cont,
            "mu_I_cont_andes_unit": m.mu_i_cont_unit,
            "mu_P_andes": m.mu_p,
            "mu_P_andes_unit": m.mu_p_unit,
            "n_units_saturated": m.n_units_saturated,
            "dP_delivered_mw": m.dp_delivered_mw,
            "p_gfm_pre_mw": m.p_gfm_pre_mw,
            "settled": m.settled,
            "t_reached_s": m.t_reached_s,
            "pflow_converged": status["pflow_converged"],
            "tds_converged": status["tds_converged"],
            "mu_P_ana_headroom_only": ana["mu_P_ana_headroom_only"],
            "no_equilibrium": m.no_equilibrium,
            "dP_critical_ana_mw": ana["dP_critical_ana_mw"],
            "error": status["error"],
        }
        self.rows.append(row)

        if self.save_raw and m.ok:
            self._dump(run_id, ss, index)
        return verdict.secure, row

    def _dump(self, run_id: str, ss, index) -> None:
        """Per-run traces.

        Every state of every bus would be about 1 MB per run compressed, which a
        few-thousand-run campaign turns into gigabytes for data nobody reads. Kept
        in full: the measured buses (frequency, RoCoF) and the six converters
        (P, Q, |I|) -- everything a figure or a re-derived metric needs. Reduced:
        feeder voltage, stored as the per-timestep envelope plus the measured
        buses, which is what the security test actually reads.
        """
        raw = self.out / "raw"
        raw.mkdir(parents=True, exist_ok=True)
        n = ss.dae.n
        xy = ss.dae.ts.xy
        conv = {r["conv_bus_andes"] for r in index["gfm"]}
        feeder = [i for i, b in enumerate(ss.Bus.idx.v) if b not in conv]
        v_all = xy[:, n + ss.Bus.v.a]
        v_feeder = v_all[:, feeder]
        watched = [i for i, nm in enumerate(ss.Bus.name.v)
                   if str(nm) in set(index["watched_buses"])]
        np.savez_compressed(
            raw / f"{run_id}.npz",
            t=np.asarray(ss.dae.ts.t, dtype=np.float32),
            f_hz=(60.0 * xy[:, n + ss.BusROCOF.f.a]).astype(np.float32),
            rocof=xy[:, n + ss.BusROCOF.Wf_y.a].astype(np.float32),
            v_min=v_feeder.min(axis=1).astype(np.float32),
            v_max=v_feeder.max(axis=1).astype(np.float32),
            v_watched=v_all[:, watched].astype(np.float32),
            v_watched_bus=np.array([str(ss.Bus.name.v[i]) for i in watched]),
            pe=xy[:, n + ss.REGF1.Pe.a].astype(np.float32),
            qe=xy[:, n + ss.REGF1.Qe.a].astype(np.float32),
            idq=np.hypot(xy[:, n + ss.REGF1.Id.a],
                         xy[:, n + ss.REGF1.Iq.a]).astype(np.float32),
            gfm_keys=np.array([r["key"] for r in index["gfm"]]),
            gfm_sn=np.array([r["s_mva"] for r in index["gfm"]], dtype=np.float32),
            rocof_buses=np.array([str(x) for x in ss.BusROCOF.name.v]),
        )


def _echo(p) -> None:
    r = p.payload
    print("  x=%8.4f  %-9s nadir=%8.4f  rocof=%6.3f  Vmin=%.4f  muI=%6.3f  "
          "muIc=%6.3f  muP=%6.3f  sat=%d  %5.1fs  %s"
          % (p.x, "SECURE" if p.secure else "insecure",
             r["f_nadir_andes"], r["rocof_andes_window"], r["v_min_andes"],
             r["mu_I_andes"], r["mu_I_cont_andes"], r["mu_P_andes"],
             r["n_units_saturated"], p.wallclock_s,
             r["insecure_because"][:44]), flush=True)


def run_p_head(runner: Runner, base: CaseSpec, args) -> tuple[dict, object]:
    """Smallest fleet headroom that survives the fixed disturbance."""
    def predicate(head_mw: float):
        spec = replace(base, p_head_mw=float(head_mw))
        return runner(spec, "phead")

    res = bisect(predicate, args.head_lo, args.head_hi, direction="secure_above",
                 tol=args.tol, verify_points=args.verify, on_probe=_echo)
    return {
        "quantity": "P_head_min_mw",
        "event": base.disturbance.kind,
        "dP_mw": base.disturbance.step_mw,
        "P_dg_pre_mw": base.diesel.p_mw if base.diesel else 0.0,
        "bracket_lo": args.head_lo,
        "bracket_hi": args.head_hi,
        "found": res.found,
        "value": res.x_boundary,
        "x_secure": res.x_secure,
        "x_insecure": res.x_insecure,
        "tol": res.tol,
        "n_eval": res.n_eval,
        "monotone": res.monotone,
        "note": res.note,
        "kappa": (base.disturbance.step_mw / res.x_boundary
                  if res.found and res.x_boundary else float("nan")),
    }, res


def run_dp_max(runner: Runner, base: CaseSpec, args) -> tuple[dict, object]:
    """Largest disturbance the fleet survives at fixed headroom.

    This is the boundary that carries dynamics. `P_head^min` at a small step turns
    out to be the feasibility bound `P_head >= dP` and nothing more -- the fleet
    either can supply the step or it cannot, and the droop dynamics never get a
    say. Sweeping the disturbance instead lets the nadir, the RoCoF or the current
    margin bind first, which is the regime the contribution is about.
    """
    def predicate(dp_mw: float):
        spec = replace(base, disturbance=replace(base.disturbance, step_mw=float(dp_mw)))
        return runner(spec, "dpmax")

    res = bisect(predicate, args.dp_lo, args.dp_hi, direction="secure_below",
                 tol=args.tol, verify_points=args.verify, on_probe=_echo)
    return {
        "quantity": "dP_max_mw",
        "event": base.disturbance.kind,
        "dP_mw": float("nan"),
        "P_dg_pre_mw": 0.0,
        "bracket_lo": args.dp_lo,
        "bracket_hi": args.dp_hi,
        "found": res.found,
        "value": res.x_boundary,
        "x_secure": res.x_secure,
        "x_insecure": res.x_insecure,
        "tol": res.tol,
        "n_eval": res.n_eval,
        "monotone": res.monotone,
        "note": res.note,
        "kappa": float("nan"),
    }, res


def run_p_dg_off(runner: Runner, base: CaseSpec, args) -> tuple[dict, object]:
    """Largest pre-trip diesel output whose loss is survived."""
    if base.diesel is None:
        raise ValueError("P_DG,off needs a diesel; pass --diesel-bus / --diesel-mva")

    def predicate(p_dg: float):
        spec = replace(base, diesel=replace(base.diesel, p_mw=float(p_dg)),
                       disturbance=replace(base.disturbance, kind="gen_trip"))
        return runner(spec, "pdgoff")

    res = bisect(predicate, args.dg_lo, args.dg_hi, direction="secure_below",
                 tol=args.tol, verify_points=args.verify, on_probe=_echo)
    return {
        "quantity": "P_DG_off_max_mw",
        "event": "gen_trip",
        "dP_mw": float("nan"),
        "P_dg_pre_mw": float("nan"),
        "bracket_lo": args.dg_lo,
        "bracket_hi": args.dg_hi,
        "found": res.found,
        "value": res.x_boundary,
        "x_secure": res.x_secure,
        "x_insecure": res.x_insecure,
        "tol": res.tol,
        "n_eval": res.n_eval,
        "monotone": res.monotone,
        "note": res.note,
        "kappa": float("nan"),
    }, res


def write_artifacts(out: Path, runner: Runner, boundaries: list[dict],
                    base: CaseSpec, band: SecurityBand, args) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "figures").mkdir(exist_ok=True)

    df = pd.DataFrame(runner.rows)
    extra = [c for c in df.columns if c not in SCHEMA]
    df[SCHEMA + extra].to_csv(out / "metrics.csv", index=False)
    pd.DataFrame(boundaries).to_csv(out / "boundaries.csv", index=False)

    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        commit = "unknown"

    (out / "manifest.json").write_text(json.dumps({
        "task": "T20_andes_bisect",
        "date": date.today().isoformat(),
        "git_commit": commit,
        "python": _platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "platform": {
            "name": "andes",
            "version": andes.__version__,
            "domain": "positive-sequence phasor",
            "integrator": "trapezoidal, fixed step",
            "gfm_model": "REGF1 (droop grid-forming)",
            "sg_model": "GENROU + " + (base.diesel.governor if base.diesel else "-")
                        + " + SEXS",
            "not_the_ground_truth": (
                "EMT per plan Part 0-D1 is. The *_emt columns of metrics.csv are "
                "deliberately empty; ANDES results are in the *_andes columns."
            ),
        },
        "case": {k: (str(v) if isinstance(v, Path) else v)
                 for k, v in asdict(base).items() if k != "diesel"},
        "diesel": asdict(base.diesel) if base.diesel else None,
        "security_band": asdict(band),
        "search": {"tol": args.tol, "verify_points": args.verify,
                   "head_bracket": [args.head_lo, args.head_hi],
                   "dp_bracket": [args.dp_lo, args.dp_hi],
                   "dg_bracket": [args.dg_lo, args.dg_hi]},
        "n_runs": len(runner.rows),
        "wallclock_total_s": round(float(df.wallclock_s.sum()), 1),
    }, indent=2, default=str), encoding="utf-8")

    cfg = {k: (str(v) if isinstance(v, (Path, tuple)) else v) for k, v in asdict(base).items()}
    cfg["disturbance"] = asdict(base.disturbance)
    cfg["diesel"] = asdict(base.diesel) if base.diesel else None
    cfg["security_band"] = asdict(band)
    (out / "config.yaml").write_text(
        "\n".join(f"{k}: {v}" for k, v in cfg.items()) + "\n", encoding="utf-8")

    _plot(out, df, boundaries)


def _plot(out: Path, df: pd.DataFrame, boundaries: list[dict]) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    for b in boundaries:
        tag = {"P_head_min_mw": "phead", "dP_max_mw": "dpmax"}.get(
            b["quantity"], "pdgoff")
        sub = df[df.run_id.str.startswith(tag)].copy()
        if sub.empty:
            continue
        x = {"phead": sub.P_head_mw, "dpmax": sub.dP_mw}.get(tag, sub.P_dg_pre)
        fig, ax = plt.subplots(1, 2, figsize=(10, 3.6), constrained_layout=True)
        ax[0].scatter(x[sub.secure_flag], sub.f_nadir_andes[sub.secure_flag],
                      marker="o", label="secure")
        ax[0].scatter(x[~sub.secure_flag], sub.f_nadir_andes[~sub.secure_flag],
                      marker="x", label="insecure")
        if b["found"]:
            for a in ax:
                a.axvline(b["value"], ls="--", lw=1, color="k")
        ax[0].set_xlabel(b["quantity"]); ax[0].set_ylabel("f_nadir [Hz]")
        ax[0].legend(fontsize=8)
        ax[1].scatter(x, sub.mu_I_andes, marker="o", label=r"$\mu_I$ andes")
        ax[1].scatter(x, sub.mu_P_andes, marker="s", label=r"$\mu_P$ andes")
        ax[1].axhline(1.0, ls=":", lw=1, color="k")
        ax[1].set_xlabel(b["quantity"]); ax[1].set_ylabel("margin")
        ax[1].legend(fontsize=8)
        fig.savefig(out / "figures" / f"bisect_{tag}.png", dpi=140)
        plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--what", choices=["p_head", "dp_max", "p_dg_off", "all"],
                    default="p_head")
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--placement", type=Path,
                    default=ROOT / "artifacts" / "placement" / "official_placement_v4_rescaled.json")
    ap.add_argument("--gfm", default="G1,G2,G3,G4,G5,G6")
    ap.add_argument("--event", choices=["gen_loss", "load_step", "der_trip"],
                    default="gen_loss")
    ap.add_argument("--dp", type=float, default=0.5, help="disturbance size [MW]")
    ap.add_argument("--event-bus", default="76")
    ap.add_argument("--t-event", type=float, default=1.0)
    ap.add_argument("--t-end", type=float, default=8.0)
    ap.add_argument("--dt", type=float, default=0.002)
    ap.add_argument("--droop-r", type=float, default=0.05)
    ap.add_argument("--x-f", type=float, default=0.15,
                    help="coupling reactance X_L (REGFM_A1: 0.05-0.25)")
    ap.add_argument("--q-max", type=float, default=0.44,
                    help="Qmax/Qmin on the unit's own base (REGFM_A1: 0.44-1.0)")
    ap.add_argument("--i-max", type=float, default=2.00,
                    help="ImaxF, the transient current ceiling the security band "
                         "is read against (REGFM_A1: 1.5-3.0)")
    ap.add_argument("--i-cont", type=float, default=1.20,
                    help="continuous thermal rating; reported, not a security criterion")
    ap.add_argument("--load-p2z", type=float, default=0.0)
    ap.add_argument("--head-lo", type=float, default=0.05)
    ap.add_argument("--head-hi", type=float, default=3.414)
    ap.add_argument("--dp-lo", type=float, default=0.05)
    ap.add_argument("--dp-hi", type=float, default=3.0)
    ap.add_argument("--dg-lo", type=float, default=0.0)
    ap.add_argument("--dg-hi", type=float, default=1.0)
    ap.add_argument("--diesel-bus", default="76")
    ap.add_argument("--diesel-mva", type=float, default=1.0)
    ap.add_argument("--diesel-h", type=float, default=1.0)
    ap.add_argument("--governor", choices=["TGOV1", "GAST"], default="TGOV1",
                    help="diesel governor family; GAST has a load-limiter path "
                         "closer to a real prime mover")
    ap.add_argument("--tol", type=float, default=0.02)
    ap.add_argument("--verify", type=int, default=6)
    ap.add_argument("--f-band", type=float, default=0.5,
                    help="+/- Hz around 60; 0.5 is the band islanded-microgrid "
                         "scheduling uses, anchored above the first UFLS stage. "
                         "See reference/security_band_provenance.md")
    ap.add_argument("--rocof-max", type=float, default=2.0)
    ap.add_argument("--v-min", type=float, default=0.88,
                    help="IEEE 1547-2018 Cat III Continuous Operation floor; "
                         "see reference/security_band_provenance.md")
    ap.add_argument("--no-raw", action="store_true")
    args = ap.parse_args()
    args.out = args.out.resolve()
    args.placement = args.placement.resolve()

    band = SecurityBand(f_min_hz=60.0 - args.f_band, f_max_hz=60.0 + args.f_band,
                        rocof_max_hz_s=args.rocof_max, v_min_pu=args.v_min,
                        v_max_pu=2.0 - args.v_min, mu_i_max=1.0)

    diesel = DieselSpec(bus_name=args.diesel_bus, s_mva=args.diesel_mva,
                        p_mw=0.0, h_sec=args.diesel_h, governor=args.governor
                        ) if args.what in ("p_dg_off", "all") else None

    base = CaseSpec(
        placement=args.placement,
        gfm_keys=tuple(args.gfm.split(",")),
        droop_r=args.droop_r,
        x_f_pu=args.x_f,
        q_max_pu=args.q_max,
        i_max_f_pu=args.i_max,
        i_cont_pu=args.i_cont,
        load_p2z=args.load_p2z,
        t_end=args.t_end,
        t_step=args.dt,
        diesel=diesel,
        disturbance=Disturbance(kind=args.event, t_event=args.t_event,
                                step_mw=args.dp, step_bus_name=args.event_bus,
                                trip_target=args.event_bus),
    )

    runner = Runner(band, args.out, save_raw=not args.no_raw)
    boundaries: list[dict] = []

    def announce(name):
        print(f"\n=== {name} ===", flush=True)

    if args.what in ("p_head", "all"):
        announce(f"P_head^min   |  {args.event} dP={args.dp} MW")
        b, res = run_p_head(runner, replace(base, diesel=None), args)
        boundaries.append(b)
        _report(b, res)

    if args.what in ("dp_max", "all"):
        announce(f"dP_max        |  {args.event}, headroom fixed")
        b, res = run_dp_max(runner, replace(base, diesel=None), args)
        boundaries.append(b)
        _report(b, res)

    if args.what in ("p_dg_off", "all"):
        announce("P_DG,off^max |  diesel trip")
        b, res = run_p_dg_off(runner, base, args)
        boundaries.append(b)
        _report(b, res)

    write_artifacts(args.out, runner, boundaries, base, band, args)
    print(f"\nwrote {len(runner.rows)} runs -> {args.out.relative_to(ROOT)}")


def _report(b: dict, res) -> None:
    if b["found"]:
        print("  -> %s = %.4f  (secure %.4f / insecure %.4f, %d runs, monotone=%s)"
              % (b["quantity"], b["value"], b["x_secure"], b["x_insecure"],
                 b["n_eval"], b["monotone"]), flush=True)
    else:
        print("  -> NO BOUNDARY: " + b["note"], flush=True)


if __name__ == "__main__":
    main()
