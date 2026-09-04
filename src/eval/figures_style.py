"""Beautiful-figures style module for matplotlib.

Encodes the principles from Andrey Churkin's "How to Create Beautiful
Research Figures" tutorial (BeautifulFigures repo, MIT 2024):

  1. Vector output (PDF + SVG; PNG only as preview)
  2. Fixed aspect ratio (no data-dependent figure size)
  3. Custom plot limits centred on data median
  4. Professional fonts (Times New Roman, 8pt minimum at IEEE 2-col width)
  5. Curated palette (purple/teal + neutral grays; no default blue/orange,
     no green/red unless meaningful)
  6. Z-order: main data on top, trend/background lower
  7. Subtle grids (low alpha, thin lines)
  8. Legend outside axes, no frame
  9. Tuned marker sizes and line widths

Usage:
    from src.eval.figures_style import apply_style, METHOD_COLORS, save_figure

    apply_style()  # global rcParams
    fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE_COL)
    for m in methods:
        ax.plot(x, y[m], color=METHOD_COLORS[m], label=m, zorder=ZORDER_MAIN)
    style_legend_outside(ax)
    save_figure(fig, "results/section1_stability/fig_freq_grid")
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

# =============================================================================
# Palette — purple/teal + neutral grays (color-hex.com/color-palette/106106)
# This matches the Iris-example palette in Andrey Churkin's BeautifulFigures
# (https://github.com/AndreyChurkin/BeautifulFigures), extended from 3 to 6
# colours so we can colour-code 6 controllers while preserving harmony.
# =============================================================================

METHOD_COLORS: dict[str, str] = {
    "GraphSAGE-MAPPO":       "#9671bd",  # purple (proposed)
    "GNN-MAPPO (Proposed)":  "#9671bd",
    "MLP-MAPPO":             "#77b5b6",  # teal (encoder ablation)
    "GCNN-PPO":              "#5e3a87",  # deep purple
    "Graph-PPO":             "#5e3a87",
    "AGCN-Decentralized":    "#c39ed6",
    "MATD3":                 "#378d94",  # deep teal
    "Fixed Droop":           "#7e7e7e",  # gray
    "No FFR":                "#2a2a2a",  # near-black
}

# Darker edge colour per method (matches repo idiom: explicit darker outline
# for scatter markers).
METHOD_EDGE_COLORS: dict[str, str] = {
    "GraphSAGE-MAPPO":       "#6a408d",
    "GNN-MAPPO (Proposed)":  "#6a408d",
    "MLP-MAPPO":             "#378d94",
    "GCNN-PPO":              "#3d2658",
    "Graph-PPO":             "#3d2658",
    "AGCN-Decentralized":    "#9671bd",
    "MATD3":                 "#246569",
    "Fixed Droop":           "#4e4e4e",
    "No FFR":                "#000000",
}

# Neutral grey for regression / trend lines (repo idiom: trend lines do not
# carry a method colour; they recede to grey behind the markers).
TREND_COLOR = "#8a8a8a"

# Sequential colormap for heatmaps (THD per-bus etc).
SEQUENTIAL_CMAP = "viridis"

# Hard-coded fallback ordering when methods are unnamed/derived.
DEFAULT_PALETTE: tuple[str, ...] = (
    "#9671bd", "#77b5b6", "#5e3a87", "#378d94", "#7e7e7e", "#2a2a2a",
)
DEFAULT_EDGE_PALETTE: tuple[str, ...] = (
    "#6a408d", "#378d94", "#3d2658", "#246569", "#4e4e4e", "#000000",
)

# =============================================================================
# Geometry — repo idiom: save BIG (10" square), scale 0.9*\columnwidth in LaTeX
# (font ends up ~8 pt after the scale-down, matching the IEEE minimum).
# =============================================================================

FIGSIZE_SQUARE = (10.0, 10.0)               # primary publication shape
FIGSIZE_SINGLE_COL = (10.0, 7.5)            # taller-than-wide variants when needed
FIGSIZE_DOUBLE_COL = (16.0, 7.5)            # 2-col grid (paired figures)
FIGSIZE_GRID_2x2 = (16.0, 13.0)             # 2x2 panel (e.g. fig_freq_grid)

# Z-order layers.
ZORDER_BACKGROUND = 1
ZORDER_GRID = 2
ZORDER_TREND = 3
ZORDER_BAND = 4
ZORDER_MAIN = 5
ZORDER_MARKER = 6
ZORDER_ANNOT = 7

# =============================================================================
# Global style
# =============================================================================

def apply_style() -> None:
    """Apply the beautiful-figures rcParams globally.

    Matches Andrey Churkin's BeautifulFigures (Iris example) idiom:
    Courier New 20 pt rendered at 10" square, which displays as 8 pt
    when scaled to 0.9*\\columnwidth in an IEEE 2-column template.
    Idempotent — safe to call multiple times.
    """
    mpl.rcParams.update({
        # --- Fonts (Courier New mono, 20pt -> 8pt at 0.9*col scale-down) ---
        "font.family": "Courier New",
        "font.size": 20.0,
        "axes.titlesize": 20.0,
        "axes.labelsize": 20.0,
        "xtick.labelsize": 20.0,
        "ytick.labelsize": 20.0,
        "legend.fontsize": 20.0,
        "figure.titlesize": 20.0,
        "mathtext.fontset": "stix",

        # --- Lines / markers (repo idiom: scatter s=90, regression lw=2.6) ---
        "lines.linewidth": 2.6,
        "lines.markersize": 9.0,
        "lines.markeredgewidth": 1.5,

        # --- Axes (keep top/right spines as in repo example) ---
        "axes.linewidth": 1.0,
        "axes.edgecolor": "#333333",
        "axes.labelcolor": "#222222",
        "axes.axisbelow": True,           # data above grid

        # --- Ticks ---
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.color": "#333333",
        "ytick.color": "#333333",

        # --- Grids (repo: major lw=0.75 α=0.25; minor lw=0.25 α=0.15) ---
        "axes.grid": True,
        "grid.color": "#bbbbbb",
        "grid.linewidth": 0.75,
        "grid.alpha": 0.25,
        "grid.linestyle": "-",

        # --- Legend (no frame; default ncol filled per-call) ---
        "legend.frameon": False,
        "legend.handlelength": 1.8,
        "legend.columnspacing": 1.2,
        "legend.labelspacing": 0.3,
        "legend.handletextpad": 0.6,

        # --- Figure ---
        "figure.dpi": 110,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42,               # embed TrueType so vector editors keep text live
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })


# =============================================================================
# Paper style (true-size design) — figures are drawn AT their final printed
# size, so the font sizes below are the sizes the reader actually sees.
#   - IEEE single column = 3.5 in; full text width = 7.16 in.
#   - Arial throughout (incl. mathtext), 8 pt base / 7 pt ticks-legend:
#     at or above the IEEE 8 pt-after-scaling minimum WITHOUT any scaling.
#   - Tight bounding box, 0.01 in pad -> minimal whitespace.
#   - Vector PDF output; decimate dense traces before plotting to keep
#     file sizes small (see decimate_trace).
# The legacy apply_style() (render-big-then-scale idiom) is kept untouched
# for the in-eval diagnostic plots.
# =============================================================================

PAPER_FIGSIZE_COL = (3.5, 2.5)          # single-column default
PAPER_FIGSIZE_COL_TALL = (3.5, 3.6)     # single-column, stacked panels
PAPER_FIGSIZE_GRID_2x2 = (7.16, 5.0)    # full-width 2x2 grid
PAPER_FIGSIZE_WIDE = (7.16, 2.3)        # full-width single row


def apply_paper_style() -> None:
    """Arial, true-size IEEE paper style. Idempotent."""
    mpl.rcParams.update({
        # --- Fonts: Arial everywhere, including math ---
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
        "font.size": 8.0,
        "axes.titlesize": 8.0,
        "axes.labelsize": 8.0,
        "xtick.labelsize": 7.0,
        "ytick.labelsize": 7.0,
        "legend.fontsize": 7.0,
        "figure.titlesize": 8.5,
        "mathtext.fontset": "custom",
        "mathtext.rm": "Arial",
        "mathtext.it": "Arial:italic",
        "mathtext.bf": "Arial:bold",

        # --- Lines / markers scaled for true-size rendering ---
        "lines.linewidth": 1.1,
        "lines.markersize": 4.0,
        "lines.markeredgewidth": 0.7,
        "axes.linewidth": 0.6,
        "axes.edgecolor": "#333333",
        "axes.labelcolor": "#111111",
        "axes.axisbelow": True,

        # --- Ticks ---
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "xtick.major.size": 2.2,
        "ytick.major.size": 2.2,
        "xtick.major.pad": 1.8,
        "ytick.major.pad": 1.8,
        "xtick.color": "#333333",
        "ytick.color": "#333333",

        # --- Grid: subtle ---
        "axes.grid": True,
        "grid.color": "#c8c8c8",
        "grid.linewidth": 0.4,
        "grid.alpha": 0.45,
        "grid.linestyle": "-",

        # --- Legend: compact, no frame ---
        "legend.frameon": False,
        "legend.handlelength": 1.4,
        "legend.columnspacing": 0.9,
        "legend.labelspacing": 0.25,
        "legend.handletextpad": 0.4,
        "legend.borderpad": 0.2,
        "legend.borderaxespad": 0.3,

        # --- Margins / labels hug the axes ---
        "axes.labelpad": 2.0,
        "axes.titlepad": 3.0,
        "axes.xmargin": 0.02,
        "axes.ymargin": 0.04,

        # --- Output: tight vector, minimal pad, embedded TrueType ---
        "figure.dpi": 130,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.01,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })


def decimate_trace(t: np.ndarray, y: np.ndarray, max_points: int = 600) -> tuple[np.ndarray, np.ndarray]:
    """Uniformly thin a dense trace to <= max_points samples for plotting.

    Keeps vector-PDF size small (each retained point is a path segment) while
    staying visually indistinguishable at column width. Always keeps the
    first and last samples.
    """
    t = np.asarray(t)
    y = np.asarray(y)
    n = len(t)
    if n <= max_points:
        return t, y
    idx = np.unique(np.concatenate([
        np.linspace(0, n - 1, max_points).astype(int), [0, n - 1]
    ]))
    return t[idx], y[idx]


# =============================================================================
# Helpers
# =============================================================================

def color_for_method(name: str, fallback_idx: int = 0) -> str:
    """Return the palette colour for a method, with a stable fallback."""
    if name in METHOD_COLORS:
        return METHOD_COLORS[name]
    return DEFAULT_PALETTE[fallback_idx % len(DEFAULT_PALETTE)]


def edge_color_for_method(name: str, fallback_idx: int = 0) -> str:
    """Darker outline colour matching the repo idiom (scatter markers get an
    explicit, slightly-darker edge in the same hue family)."""
    if name in METHOD_EDGE_COLORS:
        return METHOD_EDGE_COLORS[name]
    return DEFAULT_EDGE_PALETTE[fallback_idx % len(DEFAULT_EDGE_PALETTE)]


def style_legend_outside(
    ax: plt.Axes,
    location: str = "top",
    ncol: int | None = None,
    bbox_y: float = 1.10,
) -> None:
    """Place legend outside the axes, no frame.

    Repo idiom (Iris example): ``loc='upper center', bbox_to_anchor=(0.5, 1.10),
    ncol=4, frameon=False`` for the top-of-axes legend.
    """
    handles, labels = ax.get_legend_handles_labels()
    if not handles:
        return
    if location == "top":
        ncol = ncol or len(handles)
        ax.legend(
            handles, labels,
            loc="upper center",
            bbox_to_anchor=(0.5, bbox_y),
            ncol=ncol,
            frameon=False,
        )
    elif location == "right":
        ax.legend(
            handles, labels,
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            ncol=1,
            frameon=False,
        )
    else:
        ax.legend(handles, labels, frameon=False)


def center_limits(
    ax: plt.Axes,
    x_data: Iterable[float],
    y_data: Iterable[float],
    zoom_out: float = 0.6,
    unified: bool = True,
) -> None:
    """Centre BOTH axes on their data median, repo's plotting_range idiom.

    ``unified=True`` (default) sets the same plotting half-range on x and y
    using ``max(x_range, y_range) + zoom_out``, which yields a square plot in
    data space (matches ``ax.set_aspect('equal')``).
    """
    xa = np.asarray(list(x_data), dtype=float)
    ya = np.asarray(list(y_data), dtype=float)
    xa = xa[np.isfinite(xa)]
    ya = ya[np.isfinite(ya)]
    if xa.size == 0 or ya.size == 0:
        return
    x_min, x_max = float(np.min(xa)), float(np.max(xa))
    y_min, y_max = float(np.min(ya)), float(np.max(ya))
    x_med = 0.5 * (x_min + x_max)
    y_med = 0.5 * (y_min + y_max)
    x_range = x_max - x_min
    y_range = y_max - y_min
    if unified:
        half = (max(x_range, y_range) + float(zoom_out)) / 2.0
        ax.set_xlim(x_med - half, x_med + half)
        ax.set_ylim(y_med - half, y_med + half)
    else:
        ax.set_xlim(x_med - (x_range + zoom_out) / 2.0,
                    x_med + (x_range + zoom_out) / 2.0)
        ax.set_ylim(y_med - (y_range + zoom_out) / 2.0,
                    y_med + (y_range + zoom_out) / 2.0)


def style_grid(ax: plt.Axes, minor: bool = True) -> None:
    """Apply the major+minor grid look from the repo Iris example.

    Major: linewidth 0.75, alpha 0.25. Minor (on by default): linewidth 0.25,
    alpha 0.15. Both gray. ``set_axisbelow(True)`` already set globally.
    """
    ax.grid(True, which="major", linestyle="-", linewidth=0.75, alpha=0.25,
            color="#bbbbbb", zorder=ZORDER_GRID)
    if minor:
        ax.minorticks_on()
        ax.grid(True, which="minor", linestyle="-", linewidth=0.25, alpha=0.15,
                color="#bbbbbb", zorder=ZORDER_GRID)


def add_band(ax: plt.Axes, x: np.ndarray, lo: np.ndarray, hi: np.ndarray,
             color: str, alpha: float = 0.18, label: str | None = None) -> None:
    """Add a translucent ±band (e.g. ±1σ) under the mean curve."""
    ax.fill_between(x, lo, hi, color=color, alpha=alpha, linewidth=0,
                    zorder=ZORDER_BAND, label=label)


def save_figure(fig: plt.Figure, path_no_ext: str | Path, formats: tuple[str, ...] = ("pdf", "png")) -> list[Path]:
    """Save figure to vector PDF (preferred) + raster PNG (preview) at high DPI.

    The PDF is the file to embed in LaTeX; the PNG is for quick previews
    or for Word/Markdown that cannot render PDFs natively.
    """
    base = Path(path_no_ext)
    base.parent.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    for ext in formats:
        out = base.with_suffix(f".{ext}")
        fig.savefig(out)
        saved.append(out)
    return saved


def equal_aspect(ax: plt.Axes) -> None:
    """Force square aspect (useful for scatter plots and Pareto fronts)."""
    ax.set_aspect("equal", adjustable="box")


def tighten_spines(ax: plt.Axes, hide_top_right: bool = False) -> None:
    """Set spine colour/width. Repo default: keep ALL four spines (full box).

    Pass ``hide_top_right=True`` for a sparser style (top/right hidden).
    """
    if hide_top_right:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    for s in ("left", "bottom", "top", "right"):
        if ax.spines[s].get_visible():
            ax.spines[s].set_color("#333333")
            ax.spines[s].set_linewidth(1.0)
