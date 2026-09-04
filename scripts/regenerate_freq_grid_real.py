"""Stitch real S1-S4 frequency traces into a 2x2 grid for the paper.

Reads:
    paper/figures_real/trace_S{1,2,3,4}_*.csv  (columns: t_s, No FFR, Fixed Droop, GraphSAGE-MAPPO)

Writes:
    paper/figures/fig_freq_grid_S1_S4.{png,pdf}
"""
from __future__ import annotations
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from beautiful_style import apply_style, beautify_all, METHOD_PALETTE_BF

apply_style()

ROOT = Path(__file__).resolve().parent.parent
REAL = ROOT / "paper" / "figures_real"
OUT = ROOT / "paper" / "figures" / "fig_freq_grid_S1_S4"

SCENARIOS = [
    ("S1: load step",       "trace_S1_load_step.csv"),
    ("S2: generator trip",  "trace_S2_gen_trip.csv"),
    ("S3: line trip",       "trace_S3_line_trip.csv"),
    ("S4: severe gen trip", "trace_S4_gen_trip_severe.csv"),
]
METHOD_LW = {"GraphSAGE-MAPPO": 2.2, "Fixed Droop": 1.6, "No FFR": 1.4}
METHOD_LS = {"GraphSAGE-MAPPO": "-", "Fixed Droop": "--", "No FFR": ":"}

T_WINDOW = 30.0  # seconds shown after t=0


def main() -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 6.5), sharex=True)
    axes = axes.flatten()

    for ax, (title, fname) in zip(axes, SCENARIOS):
        df = pd.read_csv(REAL / fname)
        df = df[df.t_s <= T_WINDOW]
        for m in ["No FFR", "Fixed Droop", "GraphSAGE-MAPPO"]:
            ax.plot(df.t_s, df[m],
                    color=METHOD_PALETTE_BF[m],
                    lw=METHOD_LW[m], ls=METHOD_LS[m],
                    label=m)
        ax.axhline(49.5, color=METHOD_PALETTE_BF["GraphSAGE-MAPPO"],
                   ls=":", lw=0.9, alpha=0.6)
        ax.axhline(50.0, color="black", lw=0.4, alpha=0.4)
        ax.set_title(title)
        ax.set_xlim(0, T_WINDOW)
        if ax is axes[0]:
            ax.legend(loc="lower right")
        if ax in (axes[0], axes[2]):
            ax.set_ylabel("Frequency (Hz)")
        if ax in (axes[2], axes[3]):
            ax.set_xlabel("Time (s)")

    beautify_all(axes)
    fig.suptitle("Measured frequency response across S1-S4 (real env rollouts)", y=1.00)
    plt.tight_layout()
    fig.savefig(OUT.with_suffix(".png"), dpi=200, bbox_inches="tight")
    fig.savefig(OUT.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"[ok] {OUT.with_suffix('.png').name} (+ .pdf)")


if __name__ == "__main__":
    main()
