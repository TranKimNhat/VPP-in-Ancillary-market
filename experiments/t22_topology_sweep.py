"""T22: does the security boundary move when the feeder is reconfigured?

T21 left the boundary in a specific place. At `dP_max = 1.1851 MW` the nadir
touches 59.00 Hz and the RoCoF window touches 2.00 Hz/s within +/-0.006 MW of
each other, while `mu_I = 0.68` and `mu_P = 0.32` are nowhere near their limits.
The droop steady state that sets the nadir depends on `sum(S_g)`, `R` and `f0`;
none of those is topological. RoCoF depends on where the event sits, and `mu_I`
on the impedance between the event and the fleet -- both of which are.

Prediction under test: **`dP_max` stands still while the nadir binds, and moves
once RoCoF or `mu_I` binds first.**

The boundary alone cannot answer that. At a corner two criteria trade places
while the boundary barely moves, so "which criterion binds" flips on noise and a
dominance map built from it is unreadable. Every row therefore carries the
**margin to each criterion** at the last secure probe, normalised so that 1 is
the undisturbed system and 0 is exactly at the limit:

    nadir   (f_nadir - f_min) / (f0 - f_min)
    rocof   (rocof_max - rocof) / rocof_max
    v       (v_min - v_min_lim) / (1 - v_min_lim)
    mu_I    1 - mu_I          mu_P   1 - mu_P

A criterion is "binding" when its margin is the smallest; how much smaller it is
than the runner-up is what says whether the map is trustworthy at that point.

Two axes, one engine. `--n` sweeps *tie configuration* at a fixed event bus;
`--event-buses` sweeps *event location* at a fixed topology. The first asks
whether the boundary cares how the feeder is wired, the second whether it cares
where the event lands -- and the T22 tie sweep answered the first with "no, and
neither does RoCoF", which is only informative once the second is run: with the
event pinned to one bus, reconfiguring ties changes the impedance *between*
branches and the fleet's aggregate droop response swamps it.

Topologies come from `TieSwitchReconfiguration`, the same generator the RL work
used, so a configuration here is a configuration there. G0 is always run first as
the reference. Note that the generator rewrites
`artifacts/topology_generation_diagnostics.json` as a side effect.

Event buses may be named explicitly or as `auto:<k>`, which picks `k` buses
spread evenly over the quantiles of electrical distance to the nearest GFM
(reactance-weighted shortest path, closed ties as zero impedance). The reference
bus is always included so the sweep ties back to T21.

Run:
    uv run python experiments/t22_topology_sweep.py --n 3                # ties
    uv run python experiments/t22_topology_sweep.py --event-buses auto:5 # location
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

from src.campaign.boundary import bisect  # noqa: E402
from src.phasor.build_case import (  # noqa: E402
    CaseSpec,
    Disturbance,
    _base_net,
    base_open_elements,
)
from src.phasor.metrics import SecurityBand  # noqa: E402

from experiments.t20_andes_bisect import Runner  # noqa: E402

OUT = ROOT / "artifacts" / "T22_topology_sweep"

CRITERIA = ("nadir", "rocof", "v", "mu_I", "mu_P")


def margins(row: dict, band: SecurityBand) -> dict:
    """Normalised distance to each security criterion. 0 = on the limit."""
    f0 = 60.0
    return {
        "nadir": (row["f_nadir_andes"] - band.f_min_hz) / (f0 - band.f_min_hz),
        "rocof": (band.rocof_max_hz_s - row["rocof_andes_window"]) / band.rocof_max_hz_s,
        "v": (row["v_min_andes"] - band.v_min_pu) / (1.0 - band.v_min_pu),
        "mu_I": 1.0 - row["mu_I_andes"],
        "mu_P": 1.0 - row["mu_P_andes"],
    }


def distance_to_fleet(placement: Path, gfm_keys: tuple[str, ...]) -> dict[str, float]:
    """Reactance-weighted shortest path from each island bus to the nearest GFM.

    Closed ties are zero impedance, matching the `Jumper` the case builder uses,
    so the number is the same electrical distance the network model sees.
    """
    import networkx as nx

    from src.phasor.build_case import Z_BASE_OHM, _placement

    net = _base_net()
    g = nx.Graph()
    for _, ln in net.line.iterrows():
        if not bool(ln.in_service):
            continue
        x = float(ln.x_ohm_per_km * ln.length_km) / Z_BASE_OHM
        g.add_edge(int(ln.from_bus), int(ln.to_bus), w=max(x, 1e-9))
    for _, sw in net.switch.iterrows():
        if sw.et == "b" and bool(sw.closed):
            g.add_edge(int(sw.bus), int(sw.element), w=1e-9)

    name = {int(i): str(nm).strip() for i, nm in zip(net.bus.index, net.bus.name)}
    to_bus = {v: k for k, v in name.items()}
    gfm = _placement(str(placement))["gfm"]
    roots = [to_bus[str(gfm[k]["bus"])] for k in gfm_keys]
    hv = set(int(b) for b in net.bus.index[net.bus.vn_kv > 50.0])

    out: dict[str, float] = {}
    for b in g.nodes:
        if int(b) in hv:
            continue
        try:
            out[name[int(b)]] = min(
                nx.shortest_path_length(g, b, r, weight="w") for r in roots)
        except nx.NetworkXNoPath:
            continue
    return out


def enumerate_event_buses(spec_arg: str, placement: Path, gfm_keys: tuple[str, ...],
                          reference: str) -> list[dict]:
    """Explicit bus names, or `auto:<k>` spread over distance quantiles."""
    d = distance_to_fleet(placement, gfm_keys)
    if spec_arg.startswith("auto:"):
        k = int(spec_arg.split(":", 1)[1])
        ordered = sorted(d.items(), key=lambda kv: kv[1])
        picks = []
        for q in np.linspace(0.0, 1.0, k):
            target = float(np.quantile([v for _, v in ordered], q))
            picks.append(min(ordered, key=lambda kv: abs(kv[1] - target))[0])
        names = list(dict.fromkeys([reference] + picks))
    else:
        names = list(dict.fromkeys(
            [b.strip() for b in spec_arg.split(",") if b.strip()]))

    missing = [b for b in names if b not in d]
    if missing:
        raise SystemExit(f"event bus(es) not in the island: {missing}")
    return [{"label": f"E{b}", "open_elements": list(base_open_elements()),
             "event_bus": b, "d_elec_pu": d[b], "source": "event-location sweep"}
            for b in sorted(names, key=lambda b: d[b])]


def enumerate_topologies(n: int, seed: int) -> list[dict]:
    """G0 plus `n` generated configurations, each as an absolute open set."""
    from src.opt.tie_switch_reconfig import TieSwitchReconfiguration

    topos = [{"label": "G0", "open_elements": list(base_open_elements()),
              "source": "as-built"}]
    if n <= 0:
        return topos

    scenarios = TieSwitchReconfiguration(_base_net(), seed=seed).generate_scenarios(n)
    for k, item in enumerate(scenarios, start=1):
        net_k = item[0]
        opened = (
            [f"s{int(i)}" for i in net_k.switch.index
             if not bool(net_k.switch.at[i, "closed"])]
            + [f"l{int(i)}" for i in net_k.line.index
               if not bool(net_k.line.at[i, "in_service"])]
        )
        topos.append({"label": f"G{k}", "open_elements": opened,
                      "source": f"TieSwitchReconfiguration(seed={seed})"})

    # Two generated configurations can land on the same open set; the boundary
    # would be identical and the run wasted.
    seen, unique = set(), []
    for t in topos:
        key = tuple(sorted(t["open_elements"]))
        if key in seen:
            continue
        seen.add(key)
        unique.append(t)
    return unique


def _echo(p) -> None:
    r = p.payload
    print("    x=%8.4f  %-9s nadir=%8.4f  rocof=%6.3f  Vmin=%.4f  muI=%6.3f  "
          "muP=%6.3f  %5.1fs  %s"
          % (p.x, "SECURE" if p.secure else "insecure",
             r["f_nadir_andes"], r["rocof_andes_window"], r["v_min_andes"],
             r["mu_I_andes"], r["mu_P_andes"], p.wallclock_s,
             r["insecure_because"][:40]), flush=True)


def sweep(args) -> tuple[list[dict], list[dict], list[dict], SecurityBand, CaseSpec]:
    band = SecurityBand(f_min_hz=60.0 - args.f_band, f_max_hz=60.0 + args.f_band,
                        rocof_max_hz_s=args.rocof_max, v_min_pu=args.v_min,
                        v_max_pu=2.0 - args.v_min, mu_i_max=1.0)
    if args.event_buses:
        cases = enumerate_event_buses(args.event_buses, args.placement,
                                      tuple(args.gfm.split(",")), args.event_bus)
        axis = "event location"
    else:
        cases = enumerate_topologies(args.n, args.seed)
        axis = "tie configuration"
    print(f"{len(cases)} cases to run, sweeping {axis}\n", flush=True)

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
        disturbance=Disturbance(kind=args.event, t_event=args.t_event,
                                step_mw=args.dp, step_bus_name=args.event_bus,
                                trip_target=args.event_bus),
    )

    runner = Runner(band, args.out, save_raw=False)
    boundaries: list[dict] = []

    for t in cases:
        bus = t.get("event_bus", args.event_bus)
        spec_t = replace(
            base, topology=t["label"], open_elements=tuple(t["open_elements"]),
            disturbance=replace(base.disturbance, step_bus_name=bus,
                                trip_target=bus))
        print(f"=== {t['label']}  bus={bus}"
              + (f"  d_elec={t['d_elec_pu']:.4f} pu" if "d_elec_pu" in t else "")
              + f"  open={t['open_elements']} ===", flush=True)

        def predicate(dp_mw: float, s=spec_t):
            return runner(replace(s, disturbance=replace(s.disturbance,
                                                         step_mw=float(dp_mw))),
                          f"dpmax_{s.topology}")

        t0 = time.time()
        res = bisect(predicate, args.dp_lo, args.dp_hi, direction="secure_below",
                     tol=args.tol, verify_points=args.verify, on_probe=_echo)
        wall = time.time() - t0

        # Margins are read at the last secure probe: the boundary itself is never
        # evaluated, and the insecure side has already left the band so its
        # margins are negative and carry no information about what bound first.
        secure = [p for p in res.probes if p.secure]
        anchor = max(secure, key=lambda p: p.x) if secure else None
        mg = margins(anchor.payload, band) if anchor else {c: float("nan") for c in CRITERIA}
        binding = min(CRITERIA, key=lambda c: mg[c]) if anchor else ""
        runner_up = sorted(CRITERIA, key=lambda c: mg[c])[1] if anchor else ""

        row = {
            "topology": t["label"],
            "event_bus": bus,
            "d_elec_pu": t.get("d_elec_pu", float("nan")),
            "open_elements": " ".join(t["open_elements"]),
            "n_open": len(t["open_elements"]),
            "dP_max_mw": res.x_boundary,
            "found": res.found,
            "x_secure": res.x_secure,
            "x_insecure": res.x_insecure,
            "n_eval": len(res.probes),
            "monotone": res.monotone,
            "note": res.note,
            "anchor_dP_mw": anchor.x if anchor else float("nan"),
            **{f"margin_{c}": mg[c] for c in CRITERIA},
            "binding": binding,
            "runner_up": runner_up,
            "margin_gap": (mg[runner_up] - mg[binding]) if anchor else float("nan"),
            "wallclock_s": round(wall, 1),
        }
        boundaries.append(row)
        print(f"  -> dP_max = {res.x_boundary:.4f} MW | binding={binding} "
              f"(margin {mg[binding]:+.4f}), runner-up={runner_up} "
              f"(margin {mg[runner_up]:+.4f}), gap={row['margin_gap']:.4f}\n",
              flush=True)

    return cases, boundaries, runner.rows, band, base


def write_artifacts(out: Path, topos, boundaries, rows, band, base, args) -> None:
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "metrics.csv", index=False)
    pd.DataFrame(boundaries).to_csv(out / "boundaries.csv", index=False)
    (out / "topologies.json").write_text(json.dumps(topos, indent=2), encoding="utf-8")

    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        commit = "unknown"

    (out / "manifest.json").write_text(json.dumps({
        "task": "T22_topology_sweep",
        "question": "is the frequency-limited dP_max invariant to feeder topology?",
        "date": date.today().isoformat(),
        "git_commit": commit,
        "python": _platform.python_version(),
        "andes": andes.__version__,
        "n_cases": len(topos),
        "sweep_axis": "event location" if args.event_buses else "tie configuration",
        "seed": args.seed,
        "case": {k: (str(v) if isinstance(v, (Path, tuple)) else v)
                 for k, v in asdict(base).items() if k != "diesel"},
        "security_band": asdict(band),
        "search": {"tol": args.tol, "verify_points": args.verify,
                   "dp_bracket": [args.dp_lo, args.dp_hi]},
        "n_runs": len(rows),
    }, indent=2, default=str), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=3, help="alternative topologies besides G0")
    ap.add_argument("--event-buses", default="",
                    help="sweep event location instead of ties: comma-separated bus "
                         "names, or 'auto:<k>' for k buses spread over the quantiles "
                         "of electrical distance to the nearest GFM")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--placement", type=Path,
                    default=ROOT / "artifacts" / "placement" / "official_placement_v4_rescaled.json")
    ap.add_argument("--gfm", default="G1,G2,G3,G4,G5,G6")
    ap.add_argument("--event", choices=["gen_loss", "load_step", "der_trip"],
                    default="gen_loss")
    ap.add_argument("--dp", type=float, default=0.5)
    ap.add_argument("--event-bus", default="76")
    ap.add_argument("--t-event", type=float, default=1.0)
    ap.add_argument("--t-end", type=float, default=8.0)
    ap.add_argument("--dt", type=float, default=0.002)
    ap.add_argument("--droop-r", type=float, default=0.05)
    ap.add_argument("--x-f", type=float, default=0.15)
    ap.add_argument("--q-max", type=float, default=0.60)
    ap.add_argument("--i-max", type=float, default=2.00)
    ap.add_argument("--i-cont", type=float, default=1.20)
    ap.add_argument("--load-p2z", type=float, default=0.0)
    ap.add_argument("--dp-lo", type=float, default=0.05)
    ap.add_argument("--dp-hi", type=float, default=3.0)
    ap.add_argument("--tol", type=float, default=0.02)
    ap.add_argument("--verify", type=int, default=6)
    ap.add_argument("--f-band", type=float, default=1.0)
    ap.add_argument("--rocof-max", type=float, default=2.0)
    ap.add_argument("--v-min", type=float, default=0.90)
    args = ap.parse_args()
    args.out = args.out.resolve()
    args.placement = args.placement.resolve()

    topos, boundaries, rows, band, base = sweep(args)
    write_artifacts(args.out, topos, boundaries, rows, band, base, args)

    df = pd.DataFrame(boundaries)
    cols = ["topology", "event_bus", "d_elec_pu", "n_open", "dP_max_mw",
            "binding", "runner_up", "margin_gap"] \
        + [f"margin_{c}" for c in CRITERIA]
    print("\n" + df[cols].to_string(index=False,
                                    float_format=lambda v: f"{v:8.4f}"))
    if df.found.all():
        spread = df.dP_max_mw.max() - df.dP_max_mw.min()
        print(f"\ndP_max spread across {len(df)} topologies: {spread:.4f} MW "
              f"({100 * spread / df.dP_max_mw.mean():.2f}% of mean)")
    print(f"\nwrote {len(rows)} runs -> {args.out}")


if __name__ == "__main__":
    main()
