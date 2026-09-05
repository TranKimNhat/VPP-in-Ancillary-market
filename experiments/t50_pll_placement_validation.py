"""T50: does grid-forming placement matter once the grid-following fleet has a PLL?

Four framings for "network structure affects security" have died on this
platform, and T49 killed the fifth. Every one of them died the same way: an
index borrowed from the literature, never validated against a direct
measurement on this system.

    Lambda = X_conv/X_feeder   mine       T44: both levers moved Lambda the same
                                          way and the outcome opposite ways
    SCR < 3                    mine       a *grid-following* threshold; for GFM
                                          the relation runs the other way
    n = 2 to reach SCR < 3     mine       built on the above
    robust/fragile partition   the user   T38: neither axis moved dP_max beyond
                                          the campaign's resolution
    lambda_min (Yang 2020)     mine       T49: corr +0.14 against measured
                                          eigenvalues over 8 placements

T49 is the one worth reopening, because its failure had an identifiable cause
rather than a refutation. Yang, Xu, Zhang & Sun (IEEE TPWRS 2020) prove that
placing a grid-forming converter is equivalent to raising the grid strength
seen by the *PLL-based* converters around it, and that the placement optimum
maximises the smallest eigenvalue of the grounded, Kron-reduced Laplacian. On
this feeder the sixteen sgen carry 2.88 MW against 3.49 MW of load -- 82.5% of
supply -- and until now they were negative constant-power load with no dynamics
of any kind. There was no PLL anywhere in the model. The mechanism Yang
describes was absent by construction, so T49 measured the absence of a
mechanism, not the absence of an effect.

`CaseSpec.gfl_dynamic` now builds those sixteen as REGCP1 + PLL1 on a pinned-Q
PV static gen. The power flow is unchanged to every digit ANDES prints; the
state count goes 108 -> 220.

This script runs both arms over the same placements, so the control is the
model that produced T49 rather than a remembered number.

Pre-registered before the first run, and the thresholds are on the dynamic arm:

    H1  validated      corr(lambda_min, zeta_min) >= +0.7 and the zeta_min
                       spread is >= 20% of its mean. Yang's mechanism holds
                       here, placement has directional content, and the
                       allocation problem this project is built on exists.
    H2  wrong sign     corr <= -0.7 with the same spread. The mechanism is
                       present and runs opposite to Yang on this feeder. A
                       result, and one that needs explaining before use.
    H0  no content     |corr| < 0.4, or the spread is < 5% of the mean.
                       lambda_min joins the other four. The placement question
                       closes on this platform.
        unresolved     anything else -- reported as not resolvable at this
                       resolution, which is not the same as no effect.

The PLL bandwidth is the second axis and it is not a free parameter. PLL1
integrates 2*pi*fn*(Kp*e + Ki*int e), so the loop crossover is about 60*Kp Hz:
Kp = 0.1 is a 6 Hz PLL, Kp = 0.5 is 30 Hz, Kp = 1.0 is 60 Hz. Distribution
inverters sit at 10-50 Hz, so 0.5 is the realistic case and 1.0 and 5.0 are
stress points. Ki is held equal to Kp, the ratio PLL1 ships with.

Run:
    uv run python experiments/t50_pll_placement_validation.py
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import networkx as nx
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.t48_gfm_placement_grounded_laplacian import (  # noqa: E402
    contracted, kron, laplacian, strength)
from src.analytical.accessibility import build_branch_graph  # noqa: E402
from src.phasor.build_case import (CaseSpec, Disturbance, V_BASE_KV,  # noqa: E402
                                   _base_net, build_system, gfm_table)

OUT = ROOT / "artifacts" / "T50_pll_placement"
SRC = ROOT / "artifacts" / "placement" / "official_placement_v4_rescaled.json"
KP_GRID = (0.5, 1.0, 5.0)          # 30 / 60 / 300 Hz PLL
N_RANDOM = 400                     # pool to draw a lambda_min-spread sample from
N_SAMPLE = 16                      # placements actually simulated
SEED = 20260905


DIST = Disturbance(kind="gen_loss", t_event=1.0, step_mw=0.5, step_bus_name="76")


def state_matrix(placement: Path | str, kp: float | None, z_scale: float = 1.0, **kw):
    """Build, solve, linearise. Returns (A, state names) or None if pflow fails."""
    spec = CaseSpec(placement=placement, disturbance=DIST,
                    gfl_dynamic=kp is not None,
                    pll_kp=kp or 0.1, pll_ki=kp or 0.1, **kw)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ss, _ = build_system(spec)
        if z_scale != 1.0:
            ss.Line.r.v[:] *= z_scale
            ss.Line.x.v[:] *= z_scale
        ss.PFlow.run()
        if not ss.PFlow.converged:
            return None
        ss.TDS.init()
        ss.EIG.calc_As()
    return np.asarray(ss.EIG.As), list(ss.dae.x_name)


def critical(A, names):
    """zeta_min, its frequency, and where that mode actually lives.

    ANDES leaves 13 exact zeros in this case -- the free reference angle and the
    integrator nulls -- and they pin max(Re) at 0.0000 whatever the parameters
    do, which is why T49's `maxRe` column was uninformative on its own. They are
    dropped here, and the mode is reported with its participation factors,
    because "the damping did not move" and "the mode is not a network mode" are
    different claims and only the second one explains anything.
    """
    w, V = np.linalg.eig(A)
    W = np.linalg.inv(V)
    osc = [i for i in range(len(w)) if abs(w[i]) >= 1e-6 and abs(w[i].imag) > 1e-6]
    z = np.array([-w[i].real / abs(w[i]) for i in osc])
    k = osc[int(np.argmin(z))]
    P = np.abs(V[:, k] * W[k, :])
    P = P / P.sum()
    top = [(names[i], float(P[i])) for i in np.argsort(P)[::-1][:3]]
    return {"zeta_min": float(z.min()), "f_hz": float(abs(w[k].imag) / (2 * np.pi)),
            "max_re": float(max(w[i].real for i in range(len(w)) if abs(w[i]) >= 1e-6)),
            "participation": top, "top_share": float(sum(p for _, p in top))}


def pll_coupling(A, names):
    """How hard the network pulls on the PLL states, against the PLL's own gain.

    PLL1 integrates 2*pi*fn*(Kp*e + Ki*int e), so its self-gain sits on the
    diagonal at 2*pi*fn*Kp. If the largest network->PLL and PLL->network entries
    are far below that, the PLL modes are set by their own tuning and no
    placement of anything can move them -- which is a statement with a number
    attached, unlike "no effect".
    """
    pll = [i for i, n in enumerate(names) if "PLL" in n]
    oth = [i for i in range(len(names)) if i not in pll]
    if not pll:
        return None
    return {"n_pll_states": len(pll),
            "coupling_max": float(max(np.abs(A[np.ix_(pll, oth)]).max(),
                                      np.abs(A[np.ix_(oth, pll)]).max())),
            "self_gain_max": float(np.abs(A[np.ix_(pll, pll)]).max())}


def eigen(placement: Path, kp: float | None):
    r = state_matrix(placement, kp)
    if r is None:
        return None
    A, names = r
    out = critical(A, names)
    c = pll_coupling(A, names)
    if c:
        out.update(c)
        out["gain_ratio"] = c["self_gain_max"] / c["coupling_max"]
    return out


def corr(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3:
        return float("nan"), float("nan")
    x, y = x[ok], y[ok]
    pear = float(np.corrcoef(x, y)[0, 1])
    rx = np.argsort(np.argsort(x)); ry = np.argsort(np.argsort(y))
    return pear, float(np.corrcoef(rx, ry)[0, 1])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--n", type=int, default=N_SAMPLE)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    # --- lambda_min machinery, identical to T48 -----------------------------
    net = _base_net()
    g0 = build_branch_graph(net)
    live = g0.subgraph(max(nx.connected_components(g0), key=len)).copy()
    h, m = contracted(live)
    nodes = sorted(h.nodes)
    idx = {b: i for i, b in enumerate(nodes)}
    L = laplacian(h, nodes) * (V_BASE_KV ** 2)
    p2n = {i: str(r["name"]) for i, r in net.bus.iterrows()}
    n2p = {v: k for k, v in p2n.items()}
    gfl_buses = sorted({idx[m[int(b)]] for b in net.sgen.bus.unique()
                        if int(b) in m and m[int(b)] in idx})
    pool = [i for i in range(len(nodes)) if i not in gfl_buses]

    def lam(sel):
        sel = [s for s in sel if s not in gfl_buses]
        return strength(L, sorted(set(sel) | set(gfl_buses)), sel, gfl_buses)

    shipped = [idx[m[n2p[r["bus_name"]]]] for r in gfm_table(CaseSpec())
               if n2p[r["bus_name"]] in m]

    # A sample spread over lambda_min, not a uniform random draw: the test is a
    # correlation, so leverage along the x axis is what buys resolution.
    cand = [(lam(shipped), shipped, "shipped")]
    for k in range(N_RANDOM):
        sel = list(rng.choice(pool, size=6, replace=False))
        cand.append((lam(sel), sel, f"r{k}"))
    greedy: list[int] = []
    for _ in range(6):
        greedy.append(max((b for b in pool if b not in greedy),
                          key=lambda b: lam(greedy + [b])))
    cand.append((lam(greedy), greedy, "greedy"))
    cand.sort(key=lambda t: t[0])
    picks = [cand[0], cand[-1], next(c for c in cand if c[2] == "shipped"),
             next(c for c in cand if c[2] == "greedy")]
    q = np.linspace(0, len(cand) - 1, args.n - len(picks) + 2)[1:-1]
    picks += [cand[int(round(i))] for i in q]
    seen, sample = set(), []
    for lm, sel, tag in sorted(picks, key=lambda t: t[0]):
        key = tuple(sorted(sel))
        if key not in seen:
            seen.add(key); sample.append((lm, sel, tag))

    print(f"live buses {live.number_of_nodes()} -> {len(nodes)} after contracting shorts")
    print(f"GFL (sgen) buses: {len(gfl_buses)};  simulating {len(sample)} placements "
          f"over lambda_min {sample[0][0]:.2f}-{sample[-1][0]:.2f} "
          f"({sample[-1][0] / sample[0][0]:.1f}x)\n")

    # --- write placement files ---------------------------------------------
    d0 = json.loads(SRC.read_text(encoding="utf-8"))
    keys = tuple(f"G{i + 1}" for i in range(6))
    files = []
    for j, (lm, sel, tag) in enumerate(sample):
        d = json.loads(json.dumps(d0))
        for k, b in zip(keys, sel):
            d["gfm"][k]["bus"] = int(p2n[nodes[b]])
        p = args.out / f"pl_{j:02d}_{tag}.json"
        p.write_text(json.dumps(d, indent=1), encoding="utf-8")
        files.append(p)

    # --- both arms ----------------------------------------------------------
    arms = [("static  (T49 control)", None)] + [(f"PLL Kp={k}", k) for k in KP_GRID]
    rows = []
    for j, ((lm, sel, tag), p) in enumerate(zip(sample, files)):
        row = {"tag": tag, "lambda_min": lm,
               "buses": ",".join(p2n[nodes[b]] for b in sel)}
        for label, kp in arms:
            r = eigen(p, kp)
            row[label] = r
        rows.append(row)
        print(f"[{j + 1:>2}/{len(sample)}] {tag:<8} lambda={lm:>7.3f}  " +
              "  ".join(f"{lab.split()[-1]}:z={row[lab]['zeta_min']:.4f}"
                        f"@{row[lab]['f_hz']:.0f}Hz" if row[lab] else f"{lab}:FAIL"
                        for lab, _ in arms))

    # --- verdict ------------------------------------------------------------
    lams = [r["lambda_min"] for r in rows]
    print(f"\n{'arm':<22}{'zeta_min range':>22}{'spread/mean':>13}"
          f"{'pearson':>10}{'spearman':>10}  pre-registered verdict")
    verdict = {}
    for label, _ in arms:
        z = [r[label]["zeta_min"] if r[label] else np.nan for r in rows]
        za = np.asarray(z, float)
        spread = float(np.nanmax(za) - np.nanmin(za)) / float(np.nanmean(za))
        pe, sp = corr(lams, z)
        if spread < 0.05 or abs(pe) < 0.4:
            v = "H0  no content"
        elif pe >= 0.7 and spread >= 0.20:
            v = "H1  Yang validated"
        elif pe <= -0.7 and spread >= 0.20:
            v = "H2  wrong sign"
        else:
            v = "--  unresolved at this resolution"
        verdict[label] = {"spread_rel": spread, "pearson": pe, "spearman": sp,
                          "verdict": v}
        print(f"{label:<22}{np.nanmin(za):>10.4f} -{np.nanmax(za):>10.4f}"
              f"{spread:>13.3f}{pe:>10.3f}{sp:>10.3f}  {v}")

    # --- why: where the critical mode lives, and what can move it -----------
    ref = rows[0][f"PLL Kp={KP_GRID[1]}"]
    print(f"\ncritical mode of the dynamic arm at Kp={KP_GRID[1]}: "
          f"{ref['f_hz']:.1f} Hz, zeta {ref['zeta_min']:.4f}")
    for n, p in ref["participation"]:
        print(f"    participation {p:>6.2f}  {n}")
    print(f"    top three sum to {ref['top_share']:.2f}")
    print(f"    PLL self-gain {ref['self_gain_max']:.1f} /s  vs  max network "
          f"coupling {ref['coupling_max']:.2f} /s   ratio {ref['gain_ratio']:.1f}x")

    print("\nwhich impedance sets that coupling (shipped placement, Kp=1.0):")
    print(f"{'lever':<30}{'coupling /s':>13}{'ratio':>8}{'zeta_min':>10}"
          f"{'f Hz':>7}  critical mode lives in")
    levers = [("baseline", dict()),
              ("feeder Z x100", dict(z_scale=100.0)),
              ("x_tr 0.06 -> 2.00 (33x)", dict(x_tr_pu=2.00)),
              ("x_f 0.15 -> 1.00 (out of spec)", dict(x_f_pu=1.00))]
    lever_rows = []
    for lab, kw in levers:
        r = state_matrix(files[0], KP_GRID[1], **kw)
        if r is None:
            print(f"{lab:<30}   PFLOW FAIL"); continue
        A, names = r
        c, cr = pll_coupling(A, names), critical(A, names)
        lever_rows.append({"lever": lab, **c, **cr})
        print(f"{lab:<30}{c['coupling_max']:>13.2f}"
              f"{c['self_gain_max'] / c['coupling_max']:>8.1f}{cr['zeta_min']:>10.5f}"
              f"{cr['f_hz']:>7.1f}  {cr['participation'][0][0]}")

    # The x_f row destabilises, so it needs its control: if the static arm goes
    # unstable at the same x_f with the same mode, that instability is the GFM
    # voltage loop and has nothing to do with the PLL.
    ctl = state_matrix(files[0], None, x_f_pu=1.00)
    if ctl is not None:
        c = critical(*ctl)
        print(f"{'  ^ same, static arm (control)':<30}{'--':>13}{'--':>8}"
              f"{c['zeta_min']:>10.5f}{c['f_hz']:>7.1f}  {c['participation'][0][0]}")
        lever_rows.append({"lever": "x_f 1.00, static control", **c})

    (args.out / "results.json").write_text(
        json.dumps({"sample": rows, "verdict": verdict, "levers": lever_rows},
                   indent=1), encoding="utf-8")
    print(f"\nwrote {args.out / 'results.json'}")


if __name__ == "__main__":
    main()
