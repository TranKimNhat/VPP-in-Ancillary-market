"""Matplotlib styling inspired by AndreyChurkin/BeautifulFigures.

Usage:
    from beautiful_style import apply_style, beautify_axes, save_fig
    apply_style()
    fig, ax = plt.subplots(...)
    # ... plot ...
    beautify_axes(ax)
    save_fig(fig, out_path)   # writes .png + .pdf + .svg

Design choices kept from BeautifulFigures:
  - Monospace font (Courier New / DejaVu Sans Mono fallback) at a single size.
  - Dual major/minor light grids, axis-below.
  - Frameless legend, placed cleanly.
  - Vector output (PDF + SVG) alongside PNG.

We DO NOT override per-figure semantic colors (the paper's method palette is
intentional and meaningful), only typography / grid / spine / legend / output.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib as mpl
import matplotlib.pyplot as plt


_RC = {
    "font.family": ["Courier New", "DejaVu Sans Mono", "monospace"],
    "font.size": 10,
    "axes.titlesize": 10,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8,
    "figure.titlesize": 11,
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.axisbelow": True,
    "axes.grid": True,
    "grid.linestyle": "-",
    "grid.linewidth": 0.6,
    "grid.alpha": 0.25,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.size": 3.5,
    "ytick.major.size": 3.5,
    "xtick.minor.size": 2.0,
    "ytick.minor.size": 2.0,
    "legend.frameon": False,
    "legend.handlelength": 2.0,
    "legend.borderaxespad": 0.4,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}


# BeautifulFigures palette (color-hex.com/color-palette/106106).
# Fills + matching darker edge colors + neutral greys.
BF_PALETTE = {
    "purple":      "#9671bd",
    "purple_dark": "#6a408d",
    "teal":        "#77b5b6",
    "teal_dark":   "#378d94",
    "grey":        "#7e7e7e",
    "grey_dark":   "#4e4e4e",
    "grey_light":  "#bfbfbf",
    "neutral":     "#8a8a8a",
}

# Mapping from method labels to BF palette colors. Proposed method gets the
# strongest fill (purple); strongest baseline gets teal; weak baselines greyscale.
METHOD_PALETTE_BF = {
    "GraphSAGE-MAPPO": BF_PALETTE["purple"],
    "MLP-MAPPO":       BF_PALETTE["teal"],
    "GCNN-PPO":        BF_PALETTE["teal_dark"],
    "MATD3":           BF_PALETTE["grey_dark"],
    "Fixed Droop":     BF_PALETTE["grey"],
    "No FFR":          BF_PALETTE["grey_light"],
}

METHOD_EDGE_BF = {
    "GraphSAGE-MAPPO": BF_PALETTE["purple_dark"],
    "MLP-MAPPO":       BF_PALETTE["teal_dark"],
    "GCNN-PPO":        BF_PALETTE["teal_dark"],
    "MATD3":           "black",
    "Fixed Droop":     BF_PALETTE["grey_dark"],
    "No FFR":          BF_PALETTE["grey_dark"],
}


def apply_style() -> None:
    """Apply BeautifulFigures-inspired rcParams globally."""
    mpl.rcParams.update(_RC)


def beautify_axes(ax, minor: bool = True) -> None:
    """Add the dual-grid look + clean spines to a single Axes."""
    ax.grid(True, which="major", linestyle="-", linewidth=0.6, alpha=0.25)
    if minor:
        ax.minorticks_on()
        ax.grid(True, which="minor", linestyle="-", linewidth=0.3, alpha=0.12)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def beautify_all(axes: Iterable, minor: bool = True) -> None:
    for ax in axes:
        beautify_axes(ax, minor=minor)


def save_fig(fig, out_path: Path, formats: Iterable[str] = ("png", "pdf")) -> None:
    """Save figure in multiple vector + raster formats."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    stem = out_path.with_suffix("")
    for ext in formats:
        fig.savefig(stem.with_suffix(f".{ext}"))
    plt.close(fig)
