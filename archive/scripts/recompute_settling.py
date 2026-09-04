"""Recompute settling time UNCAPPED over the full episode.

The Table-1 metric searches for the ±50 mHz re-entry only inside a 50-step
post-event window and defaults to the window length (50 s) when the system has
not settled yet (compute_ffr_metrics, eval_ffr_topology.py). Under the current
AGC-concurrent dynamics the COI takes >50 s to restore into the ENTSO-E band, so
EVERY method saturates at ~50 s and the metric is censored — it cannot support the
"9-14 s vs 17-43 s" claim. Here we measure the true settling time on the full
300-s COI trace: the first time after the event at which |Δf| stays within the
band for the remainder of the trace (NaN = never settles within the episode).
"""
from __future__ import annotations
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from scripts.dso_cost_per_event import build_evaluator

BAND_HZ = 0.05          # ENTSO-E ±50 mHz standard frequency range (paper settling band)
ORDER = ["GraphSAGE-MAPPO", "GCNN-PPO", "MATD3", "MLP-MAPPO", "Fixed Droop", "No FFR"]
LEARNING_BASELINES = ["GCNN-PPO", "MATD3", "MLP-MAPPO"]


def settling_after_event(f_trace: np.ndarray, dt: float, t_event_s: float,
                         band: float = BAND_HZ) -> float:
    """First time AFTER the event at which |Δf| stays <= band for the rest of the
    trace. Returns NaN if it never settles within the episode."""
    df = np.abs(np.asarray(f_trace, dtype=float) - 50.0)
    t = np.arange(len(df)) * dt
    ev_i = int(round(t_event_s / dt))
    ev_i = min(max(ev_i, 0), len(df) - 1)
    post = df[ev_i:]
    within = post <= band
    # last index that is OUT of band; settling is the sample right after it
    out_idx = np.where(~within)[0]
    if out_idx.size == 0:
        settle_i = 0                       # already in band at the event
    elif out_idx[-1] + 1 >= len(post):
        return float("nan")                # still out of band at trace end
    else:
        settle_i = out_idx[-1] + 1
    return float(settle_i * dt)            # seconds AFTER the event


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-runs", type=int, default=10)
    ap.add_argument("--topology", default="base", choices=["base"],
                    help="base reference feeder (matches the per-contingency Table 1 framing)")
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "results_260620")
    args = ap.parse_args()

    ev = build_evaluator()
    ev.env.hires_substeps = max(int(getattr(ev.env, "hires_substeps", 0)), 10)
    topo_idx = -1  # base reference

    rows = []
    for scen, event in ev.scenarios.items():
        for name in ORDER:
            if name not in ev.policies:
                continue
            pol = ev.policies[name]
            vals = []
            for _ in range(args.n_runs):
                m = ev.run_episode(pol, event=event, topology_idx=topo_idx)
                tr = np.asarray(m.f_trace_hires, dtype=float)
                dt = float(m.dt_hires)
                vals.append(settling_after_event(tr, dt, float(event.t_inject)))
            v = np.asarray(vals, dtype=float)
            settled = v[~np.isnan(v)]
            rows.append({
                "scenario": scen, "method": name, "n_runs": args.n_runs,
                "settling_s_mean": float(np.mean(settled)) if settled.size else float("nan"),
                "settling_s_std": float(np.std(settled)) if settled.size else float("nan"),
                "n_settled": int(settled.size),
                "n_censored": int(np.isnan(v).sum()),
            })
            print(f"  {scen:18s} {name:16s} "
                  f"settle={rows[-1]['settling_s_mean']!s:>7.7} s  "
                  f"settled={rows[-1]['n_settled']}/{args.n_runs}", flush=True)

    df = pd.DataFrame(rows)
    args.out.mkdir(parents=True, exist_ok=True)
    csv_path = args.out / "settling_recomputed_fulltrace.csv"
    df.to_csv(csv_path, index=False)

    pd.set_option("display.width", 200)
    piv = df.pivot_table(index="method", columns="scenario", values="settling_s_mean")
    piv = piv.reindex([m for m in ORDER if m in df.method.values])
    print("\n=== Settling time (s after event), full-trace, +-50 mHz ===")
    print(piv.round(1).to_string())

    prop = piv.loc["GraphSAGE-MAPPO"]
    base = piv.reindex([m for m in LEARNING_BASELINES if m in piv.index])
    print("\nProposed (GraphSAGE-MAPPO) range across scenarios: "
          f"{np.nanmin(prop):.1f}-{np.nanmax(prop):.1f} s")
    print("Learning-baselines range (GCNN/MATD3/MLP): "
          f"{np.nanmin(base.values):.1f}-{np.nanmax(base.values):.1f} s "
          f"(NaN = never settled within 300 s)")
    print(f"\nSaved -> {csv_path}")


if __name__ == "__main__":
    main()
