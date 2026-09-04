"""T2 (plan v3.1 §3-P0): does SoC belong in the slow set Omega_E?

The question is settled without EMT by following the only channel SoC has in
the analytical layer: SoC -> P_max derating -> P_head -> (pi_g, and the
saturation boundary). Topology is fixed at G0 and the GFM placement is fixed
at the 6-unit set, per the task statement.

Run:
    uv run python experiments/t02_soc.py
"""

from __future__ import annotations

import argparse
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
    build_branch_graph,
    distance_matrix,
    interface_impedance_ohm,
)
from src.analytical.headroom import (  # noqa: E402
    DeratingModel,
    df_ss_hz,
    dp_critical_mw,
    droop_shares,
    p_head_mw,
)
from src.env.IEEE123bus import build_ieee123_net  # noqa: E402

OUT = ROOT / "artifacts" / "T02_soc"
PLACEMENT = ROOT / "artifacts" / "placement" / "official_placement_v3.json"

V_BASE_KV = 4.16
S_BASE_MVA = 1.0
Z_BASE_OHM = V_BASE_KV**2 / S_BASE_MVA
X_INT_PU = 0.10  # T1 primary regularizer
DROOP_R = 0.05  # plan §0-D2, sourced to M1/M2
F_NOM_HZ = 60.0

GFM_KEYS = ["G1", "G2", "G3", "G4", "G5", "G6"]
SOC_GRID = np.round(np.arange(0.10, 1.001, 0.025), 4)
HOLD_TIMES_H = [5 / 60, 15 / 60, 30 / 60]
DP_GRID_MW = [0.5, 1.0, 2.0, 3.0, 5.0, 8.0]


def main() -> None:
    global PLACEMENT, OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--placement", type=Path, default=PLACEMENT)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()
    PLACEMENT, OUT = args.placement.resolve(), args.out.resolve()

    OUT.mkdir(parents=True, exist_ok=True)

    payload = json.loads(PLACEMENT.read_text(encoding="utf-8"))["gfm"]
    bus_names = [str(payload[k]["bus"]) for k in GFM_KEYS]
    p_rated = np.array([float(payload[k]["bess_mw"]) for k in GFM_KEYS])
    e_rated = np.array([float(payload[k]["bess_mwh"]) for k in GFM_KEYS])
    s_rated = np.array([float(payload[k]["inverter_mva"]) for k in GFM_KEYS])

    net = build_ieee123_net(
        mode="feeder123",
        balanced=True,
        convert_switches=True,
        source_mode="publish",
        islanded_override_slack_to_g1=True,
    )
    name_to_idx = {str(n).strip(): int(i) for i, n in zip(net.bus.index, net.bus.name)}
    graph = build_branch_graph(net)

    load_p = net.load.groupby("bus").p_mw.sum()
    load_p = load_p[load_p > 0.0]
    load_buses = [int(b) for b in load_p.index]
    gfm_buses = [name_to_idx[b] for b in bus_names]
    z_ohm = distance_matrix(graph, load_buses, gfm_buses)
    eps = interface_impedance_ohm(s_rated, V_BASE_KV, X_INT_PU)

    shares = droop_shares(s_rated, DROOP_R)

    sweep_rows = []
    kappa_rows = []
    for hold_h in HOLD_TIMES_H:
        model = DeratingModel(hold_time_h=hold_h)
        for soc in SOC_GRID:
            ph = np.atleast_1d(p_head_mw(soc, p_rated, e_rated, model))
            ph_total = float(ph.sum())

            # pi_g is headroom-weighted, so a non-uniform derating moves Abar
            # even when the network is untouched.
            a_bar = (
                a_gfm_bar(
                    AccessibilityInputs(
                        load_buses=load_buses,
                        p_load_mw=load_p.to_numpy(dtype=float),
                        gfm_buses=gfm_buses,
                        p_head_mw=ph,
                        z_ohm=z_ohm,
                        z_base_ohm=Z_BASE_OHM,
                        eps_ohm=eps,
                    )
                )
                if ph_total > 0.0
                else float("nan")
            )

            dp_crit = dp_critical_mw(ph, shares)
            sweep_rows.append(
                {
                    "soc": float(soc),
                    "hold_time_h": hold_h,
                    "P_head_total_mw": ph_total,
                    "P_head_frac_of_rated": ph_total / p_rated.sum(),
                    "dP_critical_mw": dp_crit,
                    "A_bar": a_bar,
                    "binding_gfm": GFM_KEYS[int(np.argmin(np.where(shares > 0, ph / shares, np.inf)))],
                    **{f"P_head_{k}_mw": float(v) for k, v in zip(GFM_KEYS, ph)},
                }
            )

            for dp in DP_GRID_MW:
                mu_p_agg = dp / ph_total if ph_total > 0 else float("inf")
                mu_p_bind = float(np.max(dp * shares / np.where(ph > 0, ph, np.nan))) if ph_total > 0 else float("inf")
                kappa_rows.append(
                    {
                        "soc": float(soc),
                        "hold_time_h": hold_h,
                        "dP_mw": dp,
                        "P_head_total_mw": ph_total,
                        "mu_P_aggregate": mu_p_agg,
                        "mu_P_binding_unit": mu_p_bind,
                        "kappa": ph_total / dp if dp > 0 else float("inf"),
                        "df_ss_hz_unconstrained": df_ss_hz(dp, s_rated, DROOP_R, F_NOM_HZ),
                        "secure_flag": bool(mu_p_bind <= 1.0),
                    }
                )

    sweep = pd.DataFrame(sweep_rows)
    kappa = pd.DataFrame(kappa_rows)
    sweep.to_csv(OUT / "soc_sweep.csv", index=False)
    kappa.to_csv(OUT / "soc_kappa.csv", index=False)

    # Under a uniform SoC every unit derates by the same factor, so pi_g -- and
    # therefore Abar -- cannot move. The only way SoC reaches Abar is a spread
    # across units, so drive one unit down while the rest sit at reference SoC.
    model = DeratingModel(hold_time_h=0.25)
    soc_ref = 0.80
    spread_rows = []
    for target_idx, target in enumerate(GFM_KEYS):
        for soc in SOC_GRID:
            socs = np.full(len(GFM_KEYS), soc_ref)
            socs[target_idx] = soc
            ph = np.array(
                [
                    float(p_head_mw(s, p_rated[i : i + 1], e_rated[i : i + 1], model))
                    for i, s in enumerate(socs)
                ]
            )
            if ph.sum() <= 0.0:
                continue
            spread_rows.append(
                {
                    "target_gfm": target,
                    "target_soc": float(soc),
                    "soc_others": soc_ref,
                    "P_head_total_mw": float(ph.sum()),
                    "dP_critical_mw": dp_critical_mw(ph, shares),
                    "A_bar": a_gfm_bar(
                        AccessibilityInputs(
                            load_buses=load_buses,
                            p_load_mw=load_p.to_numpy(dtype=float),
                            gfm_buses=gfm_buses,
                            p_head_mw=ph,
                            z_ohm=z_ohm,
                            z_base_ohm=Z_BASE_OHM,
                            eps_ohm=eps,
                        )
                    ),
                }
            )
    spread = pd.DataFrame(spread_rows)
    spread.to_csv(OUT / "soc_spread.csv", index=False)

    # The headline: the lowest SoC at which nothing saturates, per (hold, dP).
    thresholds = (
        kappa[kappa.secure_flag]
        .groupby(["hold_time_h", "dP_mw"], as_index=False)
        .soc.min()
        .rename(columns={"soc": "soc_min_secure"})
    )
    thresholds.to_csv(OUT / "soc_thresholds.csv", index=False)

    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()
    except Exception:
        commit = "unknown"

    (OUT / "manifest.json").write_text(
        json.dumps(
            {
                "task": "T02_soc",
                "date": date.today().isoformat(),
                "git_commit": commit,
                "python": platform.python_version(),
                "numpy": np.__version__,
                "pandas": pd.__version__,
                "placement": str(PLACEMENT.relative_to(ROOT)).replace("\\", "/"),
                "fixed": {
                    "topology": "G0",
                    "gfm_deployment": "6gfm",
                    "gfm_buses": bus_names,
                    "P_rated_mw": p_rated.tolist(),
                    "E_rated_mwh": e_rated.tolist(),
                    "S_rated_mva": s_rated.tolist(),
                    "droop_R_pu": DROOP_R,
                    "f_nom_hz": F_NOM_HZ,
                    "x_int_pu": X_INT_PU,
                },
                "derating_model": {
                    "soc_min": DeratingModel().soc_min,
                    "soc_taper": DeratingModel().soc_taper,
                    "hold_times_h": HOLD_TIMES_H,
                    "form": "P_head = min(taper(SoC) * P_rated, (SoC - soc_min) * E_rated / hold)",
                },
                "grids": {"soc": SOC_GRID.tolist(), "dP_mw": DP_GRID_MW},
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    (OUT / "config.yaml").write_text(
        "\n".join(
            [
                "# T2 inputs -- mirrors the constants in experiments/t02_soc.py",
                f"droop_r_pu: {DROOP_R}",
                f"f_nom_hz: {F_NOM_HZ}",
                f"x_int_pu: {X_INT_PU}",
                f"soc_min: {DeratingModel().soc_min}",
                f"soc_taper: {DeratingModel().soc_taper}",
                f"hold_times_h: [{', '.join(f'{h:.4f}' for h in HOLD_TIMES_H)}]",
                f"dp_grid_mw: [{', '.join(str(d) for d in DP_GRID_MW)}]",
                f"soc_grid: [{SOC_GRID.min()}, ..., {SOC_GRID.max()}] step 0.025",
                "gfm_deployment: 6gfm",
                "topology: G0",
                "",
            ]
        ),
        encoding="utf-8",
    )

    pd.set_option("display.width", 220)
    print("--- headroom vs SoC (hold = 15 min) ---")
    print(
        sweep[np.isclose(sweep.hold_time_h, 0.25)][
            ["soc", "P_head_total_mw", "P_head_frac_of_rated", "dP_critical_mw", "A_bar", "binding_gfm"]
        ]
        .iloc[::4]
        .to_string(index=False)
    )
    print("\n--- lowest secure SoC ---")
    print(thresholds.to_string(index=False))
    print("\n--- Abar under a single-unit SoC spread (others at 0.80) ---")
    print(
        spread.groupby("target_gfm").A_bar.agg(["min", "max"]).assign(
            span_pct=lambda d: 100.0 * (d["max"] / d["min"] - 1.0)
        ).to_string()
    )
    print("\n--- df_ss is SoC-invariant (unconstrained droop) ---")
    print(
        kappa[np.isclose(kappa.hold_time_h, 0.25)]
        .groupby("dP_mw")
        .df_ss_hz_unconstrained.agg(["min", "max"])
        .to_string()
    )


if __name__ == "__main__":
    main()
