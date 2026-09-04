"""Paired Wilcoxon on per-event DSO cost over the 24 UNSEEN reconfigurations.

Matched design identical to the frequency-metric test (build_topology_paired_stats):
the held-out topology is the experimental unit. For each policy we cost the SAME
S2 generator-trip event on each unseen topology, average over N rollouts within a
topology -> one cost value per topology, then run a paired Wilcoxon signed-rank test
of the proposed controller vs the next-best STANDARDS-COMPLIANT baseline over the
matched per-topology vector (n=24, the same topologies enter both arms).

Reuses the exact cost model (deliverability weights, staged UFLS shedding, price
deck, VOLL sweep) from dso_cost_per_event.run_one so the numbers are consistent
with Fig. pareto / Table cost_mwh; only the experimental unit changes (topology,
not contingency).
"""
from __future__ import annotations
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from scipy import stats as _stats

from scripts.dso_cost_per_event import (
    build_evaluator, run_one, load_deliverability_weights, DLMP_CSV,
    LAMBDA_CAP, LAMBDA_ACT, VOLL_CENTRAL, VOLL_SWEEP,
)

PROPOSED = "GraphSAGE-MAPPO"
# Standards-compliant comparator pool: the learning controllers only. No-FFR /
# Fixed-Droop are out of the RoCoF standard and settle near zero on benign events
# by declining reserve, so they are not a fair least-cost comparator for the claim.
COMPLIANT_BASELINES = ("GCNN-PPO", "MATD3", "MLP-MAPPO")
COST_COLS = ["c_total"] + [f"c_total@{int(v)}" for v in VOLL_SWEEP]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-runs", type=int, default=5,
                    help="rollouts averaged WITHIN each topology (the topology is "
                         "the experimental unit; n_pairs = #unseen topologies)")
    ap.add_argument("--out", type=Path,
                    default=ROOT / "results/ffr_topology_baseref_final")
    args = ap.parse_args()

    ev = build_evaluator()
    env = ev.env
    dt_h = env.dt_fast_s / 3600.0
    P_load_total = float(env.net.load["p_mw"].sum())
    DER_W = load_deliverability_weights(env, DLMP_CSV)

    event = ev.scenarios["S2_gen_trip"]            # same event as the freq paired test
    topo_list = list(ev.test_topologies)           # the 24 unseen reconfigurations
    if not topo_list:
        raise SystemExit("No unseen topologies in evaluator; cannot run paired test.")
    print(f"[setup] {len(topo_list)} unseen topologies, {args.n_runs} rollouts each, "
          f"event=S2_gen_trip, policies={list(ev.policies)}", flush=True)

    # per-policy: one cost row per topology (mean over its rollouts)
    rows = []
    per_topo: dict[str, dict[str, list[float]]] = {
        p: {c: [] for c in COST_COLS} for p in ev.policies
    }
    for ti, topo_idx in enumerate(topo_list):
        for name, pol in ev.policies.items():
            accs = [run_one(pol, event, env, dt_h, P_load_total, DER_W,
                            topology_idx=topo_idx) for _ in range(args.n_runs)]
            agg = {k: float(np.mean([a[k] for a in accs])) for k in accs[0]}
            for c in COST_COLS:
                per_topo[name][c].append(agg[c])
            agg.update(topology_id=int(topo_idx), method=name)
            rows.append(agg)
        print(f"  [{ti+1}/{len(topo_list)}] topo {topo_idx} done", flush=True)

    df = pd.DataFrame(rows)
    args.out.mkdir(parents=True, exist_ok=True)
    csv_path = args.out / "dso_cost_per_topology.csv"
    df.to_csv(csv_path, index=False)

    # ---- Paired Wilcoxon: proposed vs each compliant baseline, per VOLL level ----
    a_means = {c: float(np.mean(per_topo[PROPOSED][c])) for c in COST_COLS}
    wil_rows = []
    for c in COST_COLS:
        a = np.asarray(per_topo[PROPOSED][c], dtype=float)
        # next-best = lowest mean cost among the compliant baselines present
        present = [b for b in COMPLIANT_BASELINES if b in per_topo]
        nb = min(present, key=lambda b: float(np.mean(per_topo[b][c])))
        for b in present:
            bvec = np.asarray(per_topo[b][c], dtype=float)
            try:
                stat, p_two = _stats.wilcoxon(a, bvec)
                # one-sided: proposed strictly cheaper
                _, p_less = _stats.wilcoxon(a, bvec, alternative="less")
            except ValueError:
                stat = p_two = p_less = float("nan")
            wil_rows.append({
                "cost_metric": c, "baseline": b, "is_next_best": (b == nb),
                "n_pairs": int(a.size),
                "mean_proposed": a_means[c], "mean_baseline": float(np.mean(bvec)),
                "delta_mean": a_means[c] - float(np.mean(bvec)),
                "wins_proposed": int(np.sum(a < bvec)),
                "wilcoxon_stat": float(stat),
                "p_two_sided": float(p_two), "p_one_sided_less": float(p_less),
            })
    wil = pd.DataFrame(wil_rows)
    wil_path = args.out / "dso_cost_wilcoxon.csv"
    wil.to_csv(wil_path, index=False)

    pd.set_option("display.width", 240); pd.set_option("display.max_columns", 30)
    print("\n=== Per-topology mean cost (EUR/event) ===")
    print(df.groupby("method")[COST_COLS].mean().round(1).to_string())
    print("\n=== Paired Wilcoxon: proposed vs compliant baselines (n_pairs = #topos) ===")
    print(wil.round(4).to_string(index=False))
    head = wil[(wil.cost_metric == "c_total") & (wil.is_next_best)].iloc[0]
    print(f"\n[HEADLINE] {PROPOSED} vs next-best compliant ({head.baseline}) on c_total "
          f"@VOLL={int(VOLL_CENTRAL)}: proposed wins {int(head.wins_proposed)}/{int(head.n_pairs)} "
          f"topologies, two-sided p={head.p_two_sided:.2e}, one-sided p={head.p_one_sided_less:.2e}")
    print(f"\nSaved -> {csv_path}\nSaved -> {wil_path}")


if __name__ == "__main__":
    main()
