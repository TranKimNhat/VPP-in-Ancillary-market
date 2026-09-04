"""Rescale the GFM fleet to a physically reachable size for the frequency study.

The v3 placement sizes 23 MVA of grid-forming converters onto a feeder whose peak
load is 3.49 MW. Every unit then idles near 10% of rating, which makes the whole
interesting region unreachable: headroom can never be scarce, mu_I never
approaches 1, and a diesel trip is absorbed at any pre-trip loading. The campaign
would return "secure everywhere" -- a non-result.

Scaling is *uniform* across the six units on purpose: it preserves pi_g, each
unit's E/P ratio, and each unit's inverter/BESS ratio.

It does NOT leave Abar_GFM invariant, which is worth stating because it is easy
to assume otherwise. pi_g and w_k are untouched, but the T1 regularizer
eps_g = x_pu * V^2 / S_g scales as 1/S_g, so shrinking the converters by 5.27x
inflates eps_g by the same factor and pushes every unit electrically further from
every load. Measured: Abar(6 GFM) falls 15.83 -> 6.01 at x_pu = 0.10, while at
eps = 0 it is identical to all printed digits. See artifacts/T01_agfm_v4/.

Run:
    uv run python experiments/rescale_gfm_fleet.py [--ratio 1.25]
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from datetime import date
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.env.IEEE123bus import build_ieee123_net  # noqa: E402

SRC = ROOT / "artifacts" / "placement" / "official_placement_v3.json"
DST = ROOT / "artifacts" / "placement" / "official_placement_v4_rescaled.json"
OUT = ROOT / "artifacts" / "T00_rescale"

# Scaled fields, per GFM entry. pv_mw rides along so a GFM's co-located PV keeps
# its proportion to the converter it shares.
SCALED = ("bess_mw", "bess_mwh", "inverter_mva", "pv_mw")

# The ratio is defined on converter apparent rating, matching how the Bach Long Vy
# reference plant is quoted (630 kVA BESS against ~0.6-1 MW peak).
DEFAULT_RATIO = 1.25


def peak_load_mw() -> float:
    net = build_ieee123_net(
        mode="feeder123",
        balanced=True,
        convert_switches=True,
        source_mode="publish",
        islanded_override_slack_to_g1=True,
    )
    # every load shape in data/profiles peaks at 1.000, so nominal spot load is peak
    return float(net.load.p_mw.sum())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--ratio",
        type=float,
        default=DEFAULT_RATIO,
        help="target total GFM inverter MVA as a multiple of peak load MW",
    )
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    payload = json.loads(SRC.read_text(encoding="utf-8"))
    gfm = payload["gfm"]

    peak = peak_load_mw()
    s_now = sum(float(g.get("inverter_mva", 0.0)) for g in gfm.values())
    s_target = args.ratio * peak
    factor = s_target / s_now

    rows = []
    for name, entry in gfm.items():
        row = {"gfm": name, "bus": entry["bus"]}
        for field in SCALED:
            if field in entry:
                before = float(entry[field])
                after = round(before * factor, 4)
                row[f"{field}_before"] = before
                row[f"{field}_after"] = after
                entry[field] = after
        rows.append(row)

    payload["version"] = "4.0-rescaled"
    payload["rescale"] = {
        "date": date.today().isoformat(),
        "source": SRC.name,
        "reason": (
            "v3 sized the GFM fleet for the VPP/market study, not for frequency "
            "security: 23 MVA of converters on a 3.49 MW peak-load feeder leaves "
            "every unit at ~10% loading, so the headroom-scarce region is "
            "physically unreachable and every boundary metric saturates."
        ),
        "rule": "uniform scaling of all six units; pi_g, E/P and MVA/MW ratios preserved",
        "peak_load_mw": peak,
        "target_ratio_inverter_mva_over_peak_load": args.ratio,
        "scale_factor": factor,
        "inverter_mva_total_before": s_now,
        "inverter_mva_total_after": round(s_now * factor, 4),
        "unscaled_and_still_oversized": {
            "note": (
                "Only the GFM fleet is rescaled here. Wind, DPV and EVCS keep v3 "
                "sizing and remain far above feeder load; they set the disturbance "
                "magnitudes for the campaign and need a separate decision."
            ),
            "wind_mw": sum(float(w["mw"]) for w in payload.get("wind", [])),
            "dpv_mw": sum(float(p["mw"]) for p in payload.get("dpv", [])),
            "evcs_bess_mw": sum(float(e.get("bess_mw", 0.0)) for e in payload.get("evcs", [])),
        },
    }

    DST.write_text(json.dumps(payload, indent=1, ensure_ascii=False), encoding="utf-8")

    table = pd.DataFrame(rows)
    table.to_csv(OUT / "gfm_rescale.csv", index=False)

    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()
    except Exception:
        commit = "unknown"

    (OUT / "manifest.json").write_text(
        json.dumps(
            {
                "task": "T00_rescale",
                "date": date.today().isoformat(),
                "git_commit": commit,
                "python": platform.python_version(),
                "source_placement": SRC.name,
                "output_placement": DST.name,
                "peak_load_mw": peak,
                "ratio": args.ratio,
                "scale_factor": factor,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    pd.set_option("display.width", 200)
    print(f"peak load            {peak:.4f} MW")
    print(f"GFM inverter MVA     {s_now:.3f}  ->  {s_now * factor:.3f}   (x{factor:.5f})")
    print(f"GFM BESS MW          {sum(r['bess_mw_before'] for r in rows):.3f}  ->  "
          f"{sum(r['bess_mw_after'] for r in rows):.3f}")
    print()
    print(table.to_string(index=False))
    print(f"\nwritten: {DST.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
