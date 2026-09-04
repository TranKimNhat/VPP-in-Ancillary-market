"""Three-layer DSO-VPP-DER coordination schematic (Arial, paper style).

Renders results/paper_figures/fig_three_layer_coordination.{png,pdf}.
Layer roles/signals follow src/layer0_dso, src/layer1_vpp, src/layer2_control.
"""
from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.eval.figures_style import apply_paper_style  # noqa: E402

OUT = ROOT / "results" / "paper_figures"

# Per-layer palette: fill (band), edge (band + sub-box stroke).
LAYERS = {
    "L0": dict(fill="#dce6f2", edge="#2f5d8a"),   # DSO  — steel blue
    "L1": dict(fill="#d8ece4", edge="#2f8a6b"),   # VPP  — teal
    "L2": dict(fill="#fbe3d0", edge="#c4631a"),   # DER  — orange
    "PH": dict(fill="#ececec", edge="#555555"),   # phys — gray
}


def _band(ax, y0, y1, key, title, timescale):
    c = LAYERS[key]
    ax.add_patch(FancyBboxPatch(
        (0.55, y0), 9.45 - 0.55, y1 - y0,
        boxstyle="round,pad=0.02,rounding_size=0.12",
        linewidth=1.4, edgecolor=c["edge"], facecolor=c["fill"], zorder=2))
    ax.text(0.78, y1 - 0.27, title, fontsize=12.5, fontweight="bold",
            color=c["edge"], va="center", ha="left", zorder=5)
    ax.text(9.22, y1 - 0.27, timescale, fontsize=9.5, fontstyle="italic",
            color="#444444", va="center", ha="right", zorder=5)


def _box(ax, cx, y0, y1, key, title, sub):
    c = LAYERS[key]
    w = 2.55
    ax.add_patch(FancyBboxPatch(
        (cx - w / 2, y0), w, y1 - y0,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        linewidth=1.1, edgecolor=c["edge"], facecolor="white", zorder=3))
    ax.text(cx, (y0 + y1) / 2 + 0.16, title, fontsize=11, fontweight="bold",
            color="#1a1a1a", va="center", ha="center", zorder=4)
    ax.text(cx, (y0 + y1) / 2 - 0.27, sub, fontsize=9, color="#555555",
            fontstyle="italic", va="center", ha="center", zorder=4)


def _arrow(ax, x, y_from, y_to, color, dashed=False):
    ax.add_patch(FancyArrowPatch(
        (x, y_from), (x, y_to),
        arrowstyle="-|>", mutation_scale=16, linewidth=1.6,
        color=color, linestyle="--" if dashed else "-",
        shrinkA=0, shrinkB=0, zorder=4))


def _label(ax, x, y, text, ha="center", color="#222222"):
    ax.text(x, y, text, fontsize=9.3, color=color, ha=ha, va="center",
            zorder=6, bbox=dict(boxstyle="round,pad=0.18", fc="white",
                                ec="none", alpha=0.85))


def main() -> None:
    apply_paper_style()
    fig, ax = plt.subplots(figsize=(7.6, 9.6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 13)
    ax.axis("off")

    # ---- Layer bands (top -> bottom) ----
    _band(ax, 10.35, 12.45, "L0", "Layer 0  —  DSO  (Distribution System Operator)",
          "minutes–hours")
    _band(ax, 7.45, 9.55, "L1", "Layer 1  —  VPP Aggregator", "15-min market")
    _band(ax, 4.55, 6.65, "L2", "Layer 2  —  DER Control  (MAPPO)", "1 s  (fast)")
    _band(ax, 1.55, 3.45, "PH", "Physical Layer  —  IEEE 123-bus islanded microgrid",
          "100% inverter-based")

    # ---- Sub-boxes per layer ----
    bx = [2.45, 5.0, 7.55]
    _box(ax, bx[0], 10.55, 11.95, "L0", "Network\nReconfiguration", "MISOCP tie-switch")
    _box(ax, bx[1], 10.55, 11.95, "L0", "Zonal / DLMP\nPricing", "ancillary $\\lambda^{\\mathrm{ffr}}$")
    _box(ax, bx[2], 10.55, 11.95, "L0", "VPP Formation", "zone partition")

    _box(ax, bx[0], 7.65, 9.05, "L1", "DRO Bidding", "Wasserstein")
    _box(ax, bx[1], 7.65, 9.05, "L1", "Virtual Battery", "reserve limits")
    _box(ax, bx[2], 7.65, 9.05, "L1", "Scenario Gen.", "price / PV / load")

    _box(ax, bx[0], 4.75, 6.15, "L2", "Graph Encoder", "GraphSAGE / GAT")
    _box(ax, bx[1], 4.75, 6.15, "L2", "Actor–Critic", "shared MAPPO policy")
    _box(ax, bx[2], 4.75, 6.15, "L2", "Safety Layer", "nadir projection")

    _box(ax, bx[0], 1.75, 3.05, "PH", "BESS", "SoC integrator")
    _box(ax, bx[1], 1.75, 3.05, "PH", "V2G", "availability lag")
    _box(ax, bx[2], 1.75, 3.05, "PH", "DPV", "curtail-only")

    # ---- Inter-layer signals: solid (down=command), dashed (up=feedback) ----
    xd, xu = 3.3, 6.7
    cmd, fb = "#1a1a1a", "#7a4a12"

    _arrow(ax, xd, 10.35, 9.55, cmd)            # L0 -> L1
    _arrow(ax, xu, 9.55, 10.35, fb, dashed=True)
    _label(ax, xd, 9.95, "$G_t,\\ \\lambda^{\\mathrm{lmp}},\\ \\lambda^{\\mathrm{ffr}}$, zones", color=cmd)
    _label(ax, xu, 9.95, "reserve bids,\ncapacity", color=fb)

    _arrow(ax, xd, 7.45, 6.65, cmd)             # L1 -> L2
    _arrow(ax, xu, 6.65, 7.45, fb, dashed=True)
    _label(ax, xd, 7.05, "$\\Delta P_{\\mathrm{ref}}$, price, $K_{\\max}$", color=cmd)
    _label(ax, xu, 7.05, "SoC, headroom,\ndelivered FFR", color=fb)

    _arrow(ax, xd, 4.55, 3.45, cmd)             # L2 -> grid
    _arrow(ax, xu, 3.45, 4.55, fb, dashed=True)
    _label(ax, xd, 4.0, "$\\Delta P^{\\mathrm{ffr}}=\\Delta P-K\\,\\Delta f$", color=cmd)
    _label(ax, xu, 4.0, "$\\Delta f,\\ V$, graph obs", color=fb)

    # ---- Legend for arrow semantics ----
    ax.add_patch(FancyArrowPatch((0.75, 0.7), (1.65, 0.7), arrowstyle="-|>",
                 mutation_scale=14, linewidth=1.6, color=cmd, zorder=6))
    ax.text(1.8, 0.7, "command / setpoint (downward)", fontsize=9, va="center", ha="left")
    ax.add_patch(FancyArrowPatch((5.35, 0.7), (6.25, 0.7), arrowstyle="-|>",
                 mutation_scale=14, linewidth=1.6, color=fb, linestyle="--", zorder=6))
    ax.text(6.4, 0.7, "measurement / feedback (upward)", fontsize=9, va="center", ha="left")

    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"fig_three_layer_coordination.{ext}",
                    dpi=300, bbox_inches="tight")
    print(f"Saved -> {OUT / 'fig_three_layer_coordination.png'}")


if __name__ == "__main__":
    main()
