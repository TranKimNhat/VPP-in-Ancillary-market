"""T44: can x_f = 0.05 plus max-separation placement make allocation visible?

T43 showed Lambda > 1 everywhere at the current fleet size and that no placement
fixes it -- the feeder diameter is smaller than every unit's converter reactance.
T44 tests the one lever that does not require resizing the fleet.

    Lambda = (x_f + x_tr) * n / (alpha * zeta),   zeta = Z_feeder * P_load / V^2

`x_f` is REGFM_A1's coupling reactance X_L, normal range 0.05-0.25. This study
ships 0.15; it used 0.05 before, and `build_case` records that move as "a
conformance change and nothing more", noting the initial reactive split was
identical to four decimals at both values. Going back to 0.05 divides X_conv by
1.91, and max-separation placement divides Lambda by a further 2.06 (T43), for
0.96-3.37 against the current 3.78-13.24 -- Lambda <= 1 for the largest unit,
with no change to fleet size, feeder, or specification compliance.

Three steps, each gating the next:

  1. Lambda at the new configuration. Graph only, no ANDES.
  2. Eigenvalues. `x_f` = 0.05 is inside the interval T33 swept, but T33 ran at
     the old KPplim and inner loops, so this has to be re-earned.
  3. The controlled test. G2/G3/G5 have identical ratings (0.7587 MVA) and sit at
     different electrical distances from the disturbance. At the shipped
     configuration their transient power shares agree to 0.28 pp while rating
     alone drives 11.87 pp -- position explains 43x less than rating. If the
     lever works, that 0.28 pp grows. If it does not, this feeder at this scale
     is electrically a single node and that is the result.

Run:
    uv run python experiments/t44_lambda_lever_test.py
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import warnings
from dataclasses import replace
from pathlib import Path

import networkx as nx
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.t32_eig_map import modes  # noqa: E402
from src.analytical.accessibility import build_branch_graph  # noqa: E402
from src.phasor.build_case import (CaseSpec, Disturbance, V_BASE_KV,  # noqa: E402
                                   _base_net, build_system, gfm_table, solve)
from src.phasor.metrics import SecurityBand, extract  # noqa: E402

OUT = ROOT / "artifacts" / "T44_lambda_lever"
SRC = ROOT / "artifacts" / "placement" / "official_placement_v4_rescaled.json"

# T43's max-mean-separation set. The three 0.7587 MVA units are placed at the
# three most different distances from the disturbance bus so step 3 has the
# widest possible position contrast at identical rating.
NEW_BUSES = {"G1": 114, "G2": 85, "G3": 151, "G4": 33, "G5": 96, "G6": 66}
X_F_NEW = 0.05
DP_TEST = 0.6                      # inside dP_max at every configuration here


def write_placement(out: Path) -> Path:
    d = json.loads(SRC.read_text(encoding="utf-8"))
    for k, bus in NEW_BUSES.items():
        d["gfm"][k]["bus"] = bus
    d["rescale"] = dict(d.get("rescale", {}), t44_note=(
        "GFM buses moved to the max-mean electrical separation set of T43; "
        "ratings, E/P and pi_g unchanged"))
    p = out / "placement_maxsep.json"
    p.write_text(json.dumps(d, indent=1), encoding="utf-8")
    return p


def lambdas(spec: CaseSpec) -> dict:
    net = _base_net()
    g0 = build_branch_graph(net)
    g = g0.subgraph(max(nx.connected_components(g0), key=len))
    zb = V_BASE_KV ** 2 / 1.0
    n2p = {str(r["name"]): i for i, r in net.bus.iterrows()}
    rows = gfm_table(spec)
    sp = dict(nx.all_pairs_dijkstra_path_length(g, weight="z"))
    buses = [n2p[r["bus_name"]] for r in rows]
    d = [sp[a][b] for a, b in itertools.combinations(buses, 2)]
    mean = float(np.mean(d)) / zb
    xc = [(spec.x_f_pu + spec.x_tr_pu) / r["s_mva"] for r in rows]
    return {"x_feeder_mean_pu": mean, "x_conv_min": min(xc), "x_conv_max": max(xc),
            "lambda_min": min(xc) / mean, "lambda_max": max(xc) / mean}


def share_test(spec: CaseSpec, band: SecurityBand) -> dict:
    """Transient power share of the three identical-rating units.

    Position is the only thing that differs between G2, G3 and G5, so the spread
    of their share of the delivered dP is a direct measurement of how much
    electrical position matters.
    """
    ss, index, status = solve(replace(
        spec, disturbance=replace(spec.disturbance, step_mw=DP_TEST)))
    m, _ = extract(ss, index, status, spec, band)
    if not (m.ok and status["tds_converged"]):
        return {"ok": False, "reason": m.reason or status["error"]}
    n = ss.dae.n
    t = np.asarray(ss.dae.ts.t, dtype=float)
    pe = ss.dae.ts.xy[:, n + ss.REGF1.Pe.a]
    keys = [r["key"] for r in index["gfm"]]
    sn = np.asarray([r["s_mva"] for r in index["gfm"]], dtype=float)

    i0 = np.searchsorted(t, spec.disturbance.t_event - 1e-3)
    dp = pe - pe[i0]
    tot = dp.sum(axis=1)
    live = np.abs(tot) > 0.05 * DP_TEST
    sh = dp[live] / tot[live][:, None]
    droop = sn / sn.sum()

    trio = [keys.index(k) for k in ("G2", "G3", "G5")]
    spread = (sh[:, trio].max(axis=1) - sh[:, trio].min(axis=1)).max()
    big = keys.index("G1")
    rating_dev = float(np.abs(sh[:, big] - droop[big]).max())
    return {"ok": True, "position_spread_pp": 100 * float(spread),
            "rating_dev_pp": 100 * rating_dev,
            "ratio_rating_over_position": rating_dev / spread if spread else float("inf"),
            "steady_state_dev": float(np.abs(sh[-1] - droop).max())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    pl = write_placement(args.out)
    band = SecurityBand()
    base = CaseSpec(disturbance=Disturbance(kind="gen_loss", t_event=1.0,
                                            step_mw=DP_TEST, step_bus_name="76"))

    configs = {
        "shipped        (x_f 0.15, current buses)": base,
        "x_f 0.05 only": replace(base, x_f_pu=X_F_NEW),
        "max-sep only": replace(base, placement=pl),
        "x_f 0.05 + max-sep": replace(base, x_f_pu=X_F_NEW, placement=pl),
    }

    print("--- step 1: Lambda (graph only) ---")
    for lbl, spec in configs.items():
        L = lambdas(spec)
        print(f"  {lbl:<42} X_feeder_mean={L['x_feeder_mean_pu']:.5f}  "
              f"Lambda = {L['lambda_min']:5.2f} .. {L['lambda_max']:5.2f}")

    print("\n--- step 2: eigenvalues at the ship controller ---")
    ok = {}
    for lbl, spec in configs.items():
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                ss, index = build_system(spec)
                ss.PFlow.run()
                if not ss.PFlow.converged:
                    print(f"  {lbl:<42} PFLOW FAILED"); ok[lbl] = False; continue
                ss.TDS.init(); ss.EIG.calc_As()
                mu, _ = ss.EIG.calc_eig()
            r = modes(np.asarray(mu))
            ok[lbl] = bool(r["stable"])
            print(f"  {lbl:<42} maxRe={r['max_re']:+8.4f}  zeta_min="
                  f"{r.get('zeta_min', float('nan')):7.4f}  "
                  f"{'STABLE' if r['stable'] else 'UNSTABLE'}")
        except Exception as exc:
            ok[lbl] = False
            print(f"  {lbl:<42} FAIL {type(exc).__name__}: {exc}")

    print(f"\n--- step 3: controlled position test, dP = {DP_TEST} MW ---")
    print("    G2/G3/G5 share the same 0.7587 MVA rating; only position differs.")
    rows = []
    for lbl, spec in configs.items():
        if not ok.get(lbl):
            print(f"  {lbl:<42} skipped (not stable)"); continue
        r = share_test(spec, band)
        r["config"] = lbl
        rows.append(r)
        if not r["ok"]:
            print(f"  {lbl:<42} run failed: {r['reason'][:40]}"); continue
        print(f"  {lbl:<42} position {r['position_spread_pp']:6.3f} pp   "
              f"rating {r['rating_dev_pp']:6.2f} pp   "
              f"rating/position = {r['ratio_rating_over_position']:6.1f}x")

    (args.out / "results.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
    print(f"\nwrote {args.out / 'results.json'}")


if __name__ == "__main__":
    main()
