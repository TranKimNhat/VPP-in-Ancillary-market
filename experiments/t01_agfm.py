"""T1 (plan v3.1 §3-P0): recompute Abar_GFM against the canonical §8 definition.

Writes artifacts/T01_agfm/ per the artifact schema of plan §5.

Run:
    uv run python experiments/t01_agfm.py
"""

from __future__ import annotations

import argparse
import itertools
import json
import platform
import subprocess
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.analytical.accessibility import (  # noqa: E402
    AccessibilityInputs,
    a_gfm_bar,
    a_gfm_raw,
    build_branch_graph,
    distance_matrix,
    distance_stats,
    interface_impedance_ohm,
)
from src.env.IEEE123bus import build_ieee123_net  # noqa: E402

OUT = ROOT / "artifacts" / "T01_agfm"
PLACEMENT = ROOT / "artifacts" / "placement" / "official_placement_v3.json"
LEGACY = ROOT / "artifacts" / "electrical_distance_analysis.json"

V_BASE_KV = 4.16
S_BASE_MVA = 1.0
Z_BASE_OHM = V_BASE_KV**2 / S_BASE_MVA

# Axis A of concept §18. The 2-GFM set is the slack anchor plus the mid-feeder
# unit; it is an assumption -- the artifact that produced the v3.1 numbers is
# not in the repo and not in git history, so the original set is unrecorded.
DEPLOYMENTS = {
    "2gfm": ["G1", "G2"],
    "5gfm": ["G1", "G2", "G3", "G4", "G5"],
    "6gfm": ["G1", "G2", "G3", "G4", "G5", "G6"],
}

# Regularizer sweep. x_pu is the GFM coupling impedance (LCL + step-up) on the
# unit's own MVA base; 0.0 falls back to a bare numerical floor, which is the
# literal reading of "+ eps" in §8.
EPS_CASES = {"eps_numerical": 0.0, "eps_x005": 0.05, "eps_x010": 0.10, "eps_x015": 0.15}
EPS_FLOOR_OHM = 1e-6
EPS_PRIMARY = "eps_x010"


# P_head,g of §8 is "upward active-power headroom". At the idle reference point
# used here that is the full BESS rating; the converter rating is carried as an
# alternative because it changes pi_g and therefore Abar.
P_HEAD_BASES = {"bess_mw": "bess_mw", "inverter_mva": "inverter_mva"}
P_HEAD_PRIMARY = "bess_mw"


def load_gfm_spec(p_head_basis: str) -> dict[str, dict[str, float]]:
    payload = json.loads(PLACEMENT.read_text(encoding="utf-8"))
    spec = {}
    for name, entry in payload["gfm"].items():
        spec[name] = {
            "bus_name": str(entry["bus"]),
            "p_head_mw": float(entry[P_HEAD_BASES[p_head_basis]]),
            "s_rated_mva": float(entry["inverter_mva"]),
        }
    return spec


def make_inputs(net, graph, gfm_spec, keys, name_to_idx, x_pu) -> AccessibilityInputs:
    load_p = net.load.groupby("bus").p_mw.sum()
    load_p = load_p[load_p > 0.0]
    load_buses = [int(b) for b in load_p.index]

    gfm_buses = [name_to_idx[gfm_spec[k]["bus_name"]] for k in keys]
    p_head = np.array([gfm_spec[k]["p_head_mw"] for k in keys])
    s_rated = np.array([gfm_spec[k]["s_rated_mva"] for k in keys])

    eps = (
        interface_impedance_ohm(s_rated, V_BASE_KV, x_pu)
        if x_pu > 0.0
        else np.full(len(keys), EPS_FLOOR_OHM)
    )

    return AccessibilityInputs(
        load_buses=load_buses,
        p_load_mw=load_p.to_numpy(dtype=float),
        gfm_buses=gfm_buses,
        p_head_mw=p_head,
        z_ohm=distance_matrix(graph, load_buses, gfm_buses),
        z_base_ohm=Z_BASE_OHM,
        eps_ohm=eps,
    )


def main() -> None:
    global PLACEMENT, OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--placement", type=Path, default=PLACEMENT)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()
    PLACEMENT, OUT = args.placement.resolve(), args.out.resolve()

    OUT.mkdir(parents=True, exist_ok=True)

    net = build_ieee123_net(
        mode="feeder123",
        balanced=True,
        convert_switches=True,
        source_mode="publish",
        islanded_override_slack_to_g1=True,
    )
    name_to_idx = {str(n).strip(): int(i) for i, n in zip(net.bus.index, net.bus.name)}
    graph = build_branch_graph(net)

    rows = []
    for p_head_basis in P_HEAD_BASES:
        gfm_spec = load_gfm_spec(p_head_basis)
        for eps_case, x_pu in EPS_CASES.items():
            for deployment, keys in DEPLOYMENTS.items():
                inp = make_inputs(net, graph, gfm_spec, keys, name_to_idx, x_pu)
                n_unreachable = int(np.isinf(inp.z_ohm).any(axis=1).sum())
                row = {
                    "deployment": deployment,
                    "p_head_basis": p_head_basis,
                    "eps_case": eps_case,
                    "x_int_pu": x_pu,
                    "n_gfm": len(keys),
                    "gfm_buses": "|".join(gfm_spec[k]["bus_name"] for k in keys),
                    "P_head_total_mw": float(inp.p_head_mw.sum()),
                    "P_load_total_mw": float(inp.p_load_mw.sum()),
                    "n_load_bus": len(inp.load_buses),
                    "n_load_bus_unreachable": n_unreachable,
                    "eps_min_ohm": float(inp.eps_ohm.min()),
                    "eps_max_ohm": float(inp.eps_ohm.max()),
                    "A_bar": a_gfm_bar(inp),
                    "A_raw": a_gfm_raw(inp),
                }
                row.update(distance_stats(inp))
                rows.append(row)

    metrics = pd.DataFrame(rows)

    # percentage changes along axis A, per (P_head basis, eps) case
    deltas = []
    for p_head_basis in P_HEAD_BASES:
        for eps_case in EPS_CASES:
            sub = metrics[
                (metrics.eps_case == eps_case) & (metrics.p_head_basis == p_head_basis)
            ].set_index("deployment")
            deltas.append(
                {
                    "p_head_basis": p_head_basis,
                    "eps_case": eps_case,
                    "dAbar_2to6_pct": 100.0 * (sub.A_bar["6gfm"] / sub.A_bar["2gfm"] - 1.0),
                    "dAbar_5to6_pct": 100.0 * (sub.A_bar["6gfm"] / sub.A_bar["5gfm"] - 1.0),
                    "dAraw_2to6_pct": 100.0 * (sub.A_raw["6gfm"] / sub.A_raw["2gfm"] - 1.0),
                    "dAraw_5to6_pct": 100.0 * (sub.A_raw["6gfm"] / sub.A_raw["5gfm"] - 1.0),
                    "dZavg_2to6_pct": 100.0 * (sub.Z_avg_ohm["6gfm"] / sub.Z_avg_ohm["2gfm"] - 1.0),
                    "dZmax_2to6_pct": 100.0 * (sub.Z_max_ohm["6gfm"] / sub.Z_max_ohm["2gfm"] - 1.0),
                }
            )
    delta_df = pd.DataFrame(deltas)

    # The 2-GFM set is unrecorded; sweep every pair so the 2->6 claim carries a range.
    spec_primary = load_gfm_spec(P_HEAD_PRIMARY)
    inp6 = make_inputs(net, graph, spec_primary, list(spec_primary), name_to_idx,
                       EPS_CASES[EPS_PRIMARY])
    a6 = a_gfm_bar(inp6)
    pair_rows = []
    for pair in itertools.combinations(spec_primary, 2):
        inp2 = make_inputs(net, graph, spec_primary, list(pair), name_to_idx,
                           EPS_CASES[EPS_PRIMARY])
        st = distance_stats(inp2)
        pair_rows.append(
            {
                "pair": "+".join(pair),
                "gfm_buses": "|".join(spec_primary[k]["bus_name"] for k in pair),
                "A_bar_2gfm": a_gfm_bar(inp2),
                "dAbar_2to6_pct": 100.0 * (a6 / a_gfm_bar(inp2) - 1.0),
                "Z_avg_ohm": st["Z_avg_ohm"],
                "dZavg_2to6_pct": 100.0 * (inp6.z_ohm.min(axis=1).mean() / st["Z_avg_ohm"] - 1.0),
            }
        )
    pair_df = pd.DataFrame(pair_rows).sort_values("dAbar_2to6_pct", ascending=False)

    metrics.to_csv(OUT / "metrics.csv", index=False)
    metrics[
        (metrics.eps_case == EPS_PRIMARY) & (metrics.p_head_basis == P_HEAD_PRIMARY)
    ].to_csv(OUT / "agfm_recomputed.csv", index=False)
    delta_df.to_csv(OUT / "axis_a_deltas.csv", index=False)
    pair_df.to_csv(OUT / "two_gfm_pair_sweep.csv", index=False)

    legacy = json.loads(LEGACY.read_text(encoding="utf-8"))["gfm_axis_same_topology"]
    pd.DataFrame(
        [{"deployment": k, **{f"legacy_{m}": v for m, v in d.items()}} for k, d in legacy.items()]
    ).to_csv(OUT / "legacy_v3.1_values.csv", index=False)

    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()
    except Exception:
        commit = "unknown"

    (OUT / "manifest.json").write_text(
        json.dumps(
            {
                "task": "T01_agfm",
                "date": date.today().isoformat(),
                "git_commit": commit,
                "python": platform.python_version(),
                "numpy": np.__version__,
                "pandas": pd.__version__,
                "network": {
                    "source": "data/feeder123 (IEEE 123 node test feeder)",
                    "mode": "feeder123",
                    "balanced": True,
                    "convert_switches": True,
                    "islanded_override_slack_to_g1": True,
                    "n_bus": int(len(net.bus)),
                    "n_line": int(len(net.line)),
                    "n_switch_closed": int(net.switch.closed.sum()),
                    "topology": "G0 (nominal tie-switch state)",
                },
                "bases": {"V_base_kV": V_BASE_KV, "S_base_MVA": S_BASE_MVA, "Z_base_ohm": Z_BASE_OHM},
                "eps_primary": EPS_PRIMARY,
                "p_head_primary": P_HEAD_PRIMARY,
                "deployments": DEPLOYMENTS,
                "placement": str(PLACEMENT.relative_to(ROOT)).replace("\\", "/"),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    (OUT / "config.yaml").write_text(
        "\n".join(
            [
                "# T1 inputs -- mirrors the constants in experiments/t01_agfm.py",
                f"v_base_kv: {V_BASE_KV}",
                f"s_base_mva: {S_BASE_MVA}",
                f"z_base_ohm: {Z_BASE_OHM}",
                f"eps_floor_ohm: {EPS_FLOOR_OHM}",
                f"eps_primary: {EPS_PRIMARY}",
                f"p_head_primary: {P_HEAD_PRIMARY}",
                "eps_cases:",
                *[f"  {k}: {v}" for k, v in EPS_CASES.items()],
                "deployments:",
                *[f"  {k}: [{', '.join(v)}]" for k, v in DEPLOYMENTS.items()],
                "network:",
                "  mode: feeder123",
                "  balanced: true",
                "  convert_switches: true",
                "  islanded_override_slack_to_g1: true",
                "  topology: G0",
                "",
            ]
        ),
        encoding="utf-8",
    )

    pd.set_option("display.width", 200)
    print(metrics.to_string(index=False))
    print()
    print(delta_df.to_string(index=False))
    print()
    print(pair_df.to_string(index=False))


if __name__ == "__main__":
    main()
