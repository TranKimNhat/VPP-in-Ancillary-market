"""Section 3 DSO-cost figure: 1×2 horizontal bar comparison.

(a) DSO gross payment per FFR event  (sorted ascending)
(b) Avg committed capacity per event  (same method order as panel a)

Each bar tagged with its FFR success rate to keep reliability visible.
The same method order in both panels lets the reader compare
"how much DSO pays" against "how much capacity is procured" at a glance.

Input:  results/section3_economic/tab_cost_effectiveness.csv
Output: results/section3_economic/fig_pareto.{png,pdf}
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.figures_style import (
    apply_style,
    FIGSIZE_DOUBLE_COL,
    color_for_method,
    edge_color_for_method,
    style_grid,
    tighten_spines,
)

LAMBDA_CAP_EUR_PER_MW_PER_H = 50.0
EVENT_DURATION_H = 0.0833


def main() -> int:
    apply_style()
    src_path = ROOT / "results/section3_economic/tab_cost_effectiveness.csv"
    if not src_path.exists():
        raise SystemExit(f"missing: {src_path}")
    df = pd.read_csv(src_path)
    df = df.sort_values("dso_gross_ffr_payment_per_event_eur").reset_index(drop=True)
    df["committed_mw"] = df["dso_capacity_pay_per_event_eur"] / (
        LAMBDA_CAP_EUR_PER_MW_PER_H * EVENT_DURATION_H
    )

    methods = df["method"].tolist()
    y_pos = np.arange(len(methods))[::-1]    # top = best (lowest bill)
    colors = [color_for_method(m) for m in methods]
    edges = [edge_color_for_method(m) for m in methods]

    fig, (axL, axR) = plt.subplots(
        1, 2, figsize=(FIGSIZE_DOUBLE_COL[0], FIGSIZE_DOUBLE_COL[1] * 0.95),
        sharey=True,
    )

    # ---------------- (a) DSO gross payment per event ----------------
    bill = df["dso_gross_ffr_payment_per_event_eur"].to_numpy(float)
    axL.barh(y_pos, bill, color=colors, edgecolor=edges, linewidth=1.2,
             height=0.65)
    bill_max = float(bill.max())
    for yp, val, sr in zip(y_pos, bill, df["ffr_success_rate"]):
        axL.text(val + bill_max * 0.02, yp, f"€{val:.2f}",
                 va="center", ha="left",
                 fontsize=11, fontweight="bold", color="#222222")
        # FFR_SR badge left of the bar.
        axL.text(-bill_max * 0.02, yp, f"SR={sr:.2f}",
                 va="center", ha="right",
                 fontsize=10, color="#444444", fontfamily="monospace")
    axL.set_xlim(-bill_max * 0.18, bill_max * 1.20)
    axL.set_xlabel("DSO gross payment (€ / FFR event)")
    axL.set_title("(a) Cost the DSO pays")
    axL.set_yticks(y_pos)
    axL.set_yticklabels(methods)
    style_grid(axL, minor=False)
    tighten_spines(axL)

    # ---------------- (b) Committed capacity per event ----------------
    commit = df["committed_mw"].to_numpy(float)
    axR.barh(y_pos, commit, color=colors, edgecolor=edges, linewidth=1.2,
             height=0.65, alpha=0.85)
    commit_max = float(commit.max())
    for yp, val in zip(y_pos, commit):
        axR.text(val + commit_max * 0.02, yp, f"{val:.3f} MW",
                 va="center", ha="left",
                 fontsize=11, fontweight="bold", color="#222222")
    axR.set_xlim(0, commit_max * 1.25)
    axR.set_xlabel("Avg committed capacity (MW / FFR event)")
    axR.set_title("(b) Capacity the DSO procures")
    style_grid(axR, minor=False)
    tighten_spines(axR)

    fig.suptitle(
        "Section 3 — DSO procurement cost for FFR.  Lower bar = cheaper DSO bill\n"
        "and lower committed MW = more energy-efficient procurement by the controller.",
        fontsize=13, y=1.02,
    )
    fig.tight_layout()
    out = ROOT / "results/section3_economic/fig_pareto"
    fig.savefig(out.with_suffix(".png"))
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)
    print(f"Saved: {out}.png + .pdf")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
