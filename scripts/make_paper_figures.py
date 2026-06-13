"""Generate ALL result figures for the paper, true-size IEEE style (Arial).

Design contract (user spec):
  1. Arial everywhere (incl. mathtext)        -> apply_paper_style()
  2. Text legible at single-column width      -> figures drawn AT final size,
     8 pt base / 7 pt ticks (no shrink-on-include)
  3. Minimal whitespace                        -> tight bbox, 0.01" pad,
     compact legends, no in-panel tables
  4. Minimal file size                         -> vector PDF, decimated traces,
     subset-embedded TrueType
  5. Reader-first layouts                      -> proposed method visually
     dominant, classical baselines receded

Inputs (CSV, from src.eval.eval_ffr_topology run):
    <results-dir>/table1_ffr_comparison.csv
    <results-dir>/table2_topology_adaptation.csv
    <results-dir>/table3_severity_scaling.csv
    <results-dir>/fig6_iae_vs_distance.csv          (per-topology IAE)
    results/ffr_topology_baseref_final/dso_cost_per_event.csv  (pareto)

Outputs -> results/paper_figures/ and copies into IEEE_Journal_Paper_Template
with the exact names main.tex includes.

Usage:
    PYTHONPATH=. python scripts/make_paper_figures.py
    PYTHONPATH=. python scripts/make_paper_figures.py --with-freq-grid   # +episodes
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import sys
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.eval.figures_style import (  # noqa: E402
    PAPER_FIGSIZE_COL, PAPER_FIGSIZE_COL_TALL, PAPER_FIGSIZE_GRID_2x2,
    apply_paper_style, decimate_trace,
)

# High-contrast paper palette: proposed = vivid red (thick, on top);
# learning baselines = cool hues; classical refs = grays. Hue separation
# is deliberately large so the six curves stay distinguishable in print
# and for color-blind readers (red/blue/green/purple + 2 grays, distinct
# linestyles on top).
METHOD_COLORS = {
    "GraphSAGE-MAPPO": "#d62728",  # vivid red — proposed
    "MATD3":           "#1f77b4",  # blue
    "GCNN-PPO":        "#2ca02c",  # green
    "MLP-MAPPO":       "#9467bd",  # purple
    "Fixed Droop":     "#8c8c8c",  # mid gray
    "No FFR":          "#3a3a3a",  # near-black
}
METHOD_EDGE_COLORS = {
    "GraphSAGE-MAPPO": "#8f1414",
    "MATD3":           "#14507e",
    "GCNN-PPO":        "#1c6b1c",
    "MLP-MAPPO":       "#5e3a87",
    "Fixed Droop":     "#5a5a5a",
    "No FFR":          "#000000",
}

OUT = ROOT / "results" / "paper_figures"
TEMPLATE = ROOT / "IEEE_Journal_Paper_Template"

# Display order: proposed first, classical last.
ORDER = ["GraphSAGE-MAPPO", "MATD3", "GCNN-PPO", "MLP-MAPPO", "Fixed Droop", "No FFR"]
SHORT = {
    "GraphSAGE-MAPPO": "GraphSAGE (ours)",
    "MATD3": "MATD3",
    "GCNN-PPO": "GCNN-PPO",
    "MLP-MAPPO": "MLP-MAPPO",
    "Fixed Droop": "Fixed droop",
    "No FFR": "No FFR",
}
SCEN_TITLE = {
    "S1_load_step": r"S1 load step (+2.5 MW)",
    "S2_gen_trip": r"S2 gen trip ($-$3.9 MW)",
    "S3_line_trip": r"S3 line trip ($-$2.4 MW)",
    "S4_high_ren_surge": "S4 high-ren surge (+4.7 MW)",
}
LINESTYLE = {
    "GraphSAGE-MAPPO": "-", "MATD3": "-.", "GCNN-PPO": ":",
    "MLP-MAPPO": "--", "Fixed Droop": "-", "No FFR": "-",
}
PROPOSED = "GraphSAGE-MAPPO"


def lw_for(m: str) -> float:
    return 1.6 if m == PROPOSED else 0.9


def save(fig: plt.Figure, name: str, also: list[Path] | None = None) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    pdf = OUT / f"{name}.pdf"
    fig.savefig(pdf)
    fig.savefig(OUT / f"{name}.png", dpi=200)
    plt.close(fig)
    for dst in also or []:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(pdf, dst)
    kb = pdf.stat().st_size / 1024
    print(f"  {name}.pdf  {kb:7.1f} kB")


def _pivot_t1(rdir: Path) -> pd.DataFrame:
    df = pd.read_csv(rdir / "table1_ffr_comparison.csv")
    return df.pivot_table(index=["scenario", "method"], columns="metric", values="mean")


# ---------------------------------------------------------------- fig1_radar
def fig_radar(rdir: Path) -> None:
    t1 = _pivot_t1(rdir)
    f6 = pd.read_csv(rdir / "fig6_iae_vs_distance.csv")
    # System-level topology score: best unseen mean / unseen mean (1 = best).
    unseen = f6.groupby("method")["iae_test"].mean()
    topo_score = (unseen.min() / unseen).to_dict()

    # Short single-line labels: two-line labels collided with neighbouring
    # panels and the figure legend at print size.
    axes_labels = ["Low IAE", "Margin", "Settling", "Reserve", "Topology"]
    n_ax = len(axes_labels)
    angles = np.linspace(0, 2 * np.pi, n_ax, endpoint=False).tolist()
    angles += angles[:1]

    fig, axs = plt.subplots(2, 2, figsize=(7.16, 6.2), subplot_kw=dict(polar=True))
    for ax, scen in zip(axs.flat, SCEN_TITLE):
        over = scen == "S4_high_ren_surge"
        vals = {}
        for m in ORDER:
            row = t1.loc[(scen, m)]
            iae = 1.0 / max(row["iae_post"], 1e-9)
            margin = (51.0 - row["zenith_hz"]) if over else (row["nadir_hz"] - 48.0)
            settle = 1.0 / max(row["settling_time_s"], 1e-9)
            reserve = row["ffr_success"]
            vals[m] = [iae, max(margin, 0.0), settle, reserve, topo_score.get(m, 0.0)]
        arr = np.array([vals[m] for m in ORDER])
        best = arr.max(axis=0)
        best[best <= 0] = 1.0
        for m in ORDER:
            v = (np.array(vals[m]) / best).tolist()
            v += v[:1]
            c = METHOD_COLORS[m]
            ax.plot(angles, v, color=c, linewidth=lw_for(m), linestyle=LINESTYLE[m],
                    zorder=6 if m == PROPOSED else 4)
            if m == PROPOSED:
                ax.fill(angles, v, color=c, alpha=0.12, zorder=3)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(axes_labels, fontsize=8)
        # Radial rings without text: the "0.5"/"1" labels collided with the
        # spoke labels at this size; the rings alone convey scale.
        ax.set_yticks([0.5, 1.0])
        ax.set_yticklabels([])
        ax.set_ylim(0, 1.02)
        ax.tick_params(pad=3)
        ax.set_title(SCEN_TITLE[scen], fontsize=8.5, pad=14)
        ax.grid(linewidth=0.4, alpha=0.5)
        ax.spines["polar"].set_linewidth(0.5)
    handles = [plt.Line2D([], [], color=METHOD_COLORS[m], linestyle=LINESTYLE[m],
                          linewidth=lw_for(m), label=SHORT[m]) for m in ORDER]
    fig.legend(handles=handles, loc="upper center", ncol=6, frameon=False,
               bbox_to_anchor=(0.5, 1.005), fontsize=8, columnspacing=1.2)
    # Single-line spoke labels need little clearance: tight margins let each
    # radar fill its quadrant instead of floating small in whitespace.
    fig.subplots_adjust(top=0.875, bottom=0.045, left=0.045, right=0.955,
                        hspace=0.34, wspace=0.16)
    save(fig, "fig1_radar", [TEMPLATE / "img" / "fig1_radar.pdf"])


# ------------------------------------------------------------- fig_severity
def fig_severity(rdir: Path) -> None:
    t3 = pd.read_csv(rdir / "table3_severity_scaling.csv")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=PAPER_FIGSIZE_COL_TALL, sharex=True)
    for m in ORDER:
        d = t3[t3.method == m].sort_values("delta_P_mw")
        kw = dict(color=METHOD_COLORS[m], linestyle=LINESTYLE[m], linewidth=lw_for(m),
                  marker="o", markersize=3.2 if m == PROPOSED else 2.5,
                  markeredgecolor=METHOD_EDGE_COLORS[m], markeredgewidth=0.4,
                  zorder=6 if m == PROPOSED else 4, label=SHORT[m])
        # Panel (a): all four learning methods sit exactly on the 49.5 Hz
        # plateau; draw the dashed baselines ABOVE the thick proposed line so
        # their dash patterns stay visible (otherwise only ours is seen).
        kw_a = {**kw, "zorder": 7 if m != PROPOSED else 6}
        ax1.plot(d.delta_P_mw, d.nadir_hz_mean, **kw_a)
        ax2.plot(d.delta_P_mw, d.settling_time_mean, **kw)
    ax1.axhline(49.0, color="#b03030", linewidth=0.7, linestyle="--", zorder=2)
    ax1.text(6.0, 49.03, "UFLS-1 49.0 Hz", fontsize=6.5, color="#b03030", ha="right")
    ax1.set_ylabel("Nadir (Hz)")
    ax1.text(0.02, 0.06, "(a)", transform=ax1.transAxes, fontsize=8, fontweight="bold")
    ax2.set_ylabel("Settling time (s)")
    ax2.set_xlabel(r"Disturbance magnitude $|\Delta P|$ (MW)")
    ax2.text(0.02, 0.88, "(b)", transform=ax2.transAxes, fontsize=8, fontweight="bold")
    # Figure-level legend ABOVE the axes: in-panel placement collided with the
    # descending Fixed-droop / No-FFR curves.
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False,
               bbox_to_anchor=(0.56, 1.005), fontsize=6.5, columnspacing=0.8,
               handlelength=1.6)
    fig.subplots_adjust(left=0.13, right=0.99, top=0.915, bottom=0.10, hspace=0.08)
    save(fig, "fig_severity", [TEMPLATE / "img" / "fig_severity.pdf"])


# ------------------------------------------------------------- fig_topology
def fig_topology(rdir: Path) -> None:
    t2 = pd.read_csv(rdir / "table2_topology_adaptation.csv")
    base = t2[t2.topology_split == "base"].set_index("method")["iae_post_mean"]
    uns = t2[t2.topology_split == "unseen"].set_index("method")["iae_post_mean"]
    methods = [m for m in ORDER if m in base.index][::-1]  # proposed at top
    fig, ax = plt.subplots(figsize=(3.5, 2.1))
    for i, m in enumerate(methods):
        c = METHOD_COLORS[m]
        b, u = base[m], uns[m]
        ax.plot([b, u], [i, i], color=c, linewidth=1.0, zorder=3)
        ax.scatter([b], [i], s=22, facecolor="white", edgecolor=c, linewidth=1.0, zorder=5)
        ax.scatter([u], [i], s=24, facecolor=c, edgecolor=METHOD_EDGE_COLORS[m],
                   linewidth=0.5, zorder=5)
        gap = 100.0 * (u - b) / b
        ax.annotate(f"{gap:+.1f}%", (max(b, u), i), xytext=(4, 0),
                    textcoords="offset points", va="center", fontsize=6.5,
                    color=c, fontweight="bold" if m == PROPOSED else "normal")
    # Marker-meaning legend instead of a cryptic symbol-coded xlabel.
    ax.scatter([], [], s=22, facecolor="white", edgecolor="#444444", linewidth=1.0,
               label="training feeder")
    ax.scatter([], [], s=24, facecolor="#444444", edgecolor="#000000", linewidth=0.5,
               label="mean of 24 unseen reconfig.")
    ax.legend(loc="lower right", fontsize=6.5, handletextpad=0.3)
    ax.set_yticks(range(len(methods)))
    ax.set_yticklabels([SHORT[m] for m in methods], fontsize=7)
    ax.set_xlabel("Post-event frequency error IAE (Hz·s)")
    # Explicit xlim so the largest gap label never clips at the right edge.
    xmax = float(max(base.max(), uns.max()))
    xmin = float(min(base.min(), uns.min()))
    ax.set_xlim(xmin - 0.06 * (xmax - xmin), xmax + 0.24 * (xmax - xmin))
    fig.subplots_adjust(left=0.27, right=0.99, top=0.99, bottom=0.19)
    save(fig, "fig_topology", [TEMPLATE / "img" / "fig_topology.pdf"])


# ----------------------------------------------------------- fig_unseen_iae
def fig_unseen_iae(rdir: Path) -> None:
    f6 = pd.read_csv(rdir / "fig6_iae_vs_distance.csv")
    t2 = pd.read_csv(rdir / "table2_topology_adaptation.csv")
    base = t2[t2.topology_split == "base"].set_index("method")["iae_post_mean"]
    methods = [m for m in ORDER if m in set(f6.method)][::-1]
    data = [f6[f6.method == m]["iae_test"].values for m in methods]
    fig, ax = plt.subplots(figsize=(3.5, 2.1))
    bp = ax.boxplot(data, vert=False, positions=range(len(methods)), widths=0.55,
                    patch_artist=True, whis=(0, 100), showfliers=False,
                    medianprops=dict(color="#222222", linewidth=0.8),
                    boxprops=dict(linewidth=0.6), whiskerprops=dict(linewidth=0.6),
                    capprops=dict(linewidth=0.6))
    for patch, m in zip(bp["boxes"], methods):
        patch.set_facecolor(METHOD_COLORS[m])
        patch.set_alpha(0.55)
        patch.set_edgecolor(METHOD_EDGE_COLORS[m])
    for i, m in enumerate(methods):
        if m in base.index:
            ax.scatter([base[m]], [i], marker="D", s=14, facecolor="white",
                       edgecolor=METHOD_EDGE_COLORS[m], linewidth=0.8, zorder=6)
    ax.scatter([], [], marker="D", s=14, facecolor="white", edgecolor="#444444",
               linewidth=0.8, label="training topology $G_0$")
    ax.legend(loc="lower right", fontsize=6.5)
    ax.set_yticks(range(len(methods)))
    ax.set_yticklabels([SHORT[m] for m in methods], fontsize=7)
    ax.set_xlabel("Post-event frequency error IAE (Hz·s),\nspread over the 24 unseen reconfigurations")
    # Explicit xlim: the MATD3 max whisker otherwise touches the right spine.
    allv = np.concatenate(data + [base.values])
    lo, hi = float(np.min(allv)), float(np.max(allv))
    ax.set_xlim(lo - 0.05 * (hi - lo), hi + 0.06 * (hi - lo))
    fig.subplots_adjust(left=0.27, right=0.99, top=0.99, bottom=0.27)  # 2-line xlabel
    save(fig, "fig_unseen_iae", [TEMPLATE / "img" / "fig_unseen_iae.pdf"])


# --------------------------------------------------------------- fig_pareto
def fig_pareto(dso_csv: Path) -> None:
    if not dso_csv.exists():
        print(f"  [skip] fig_pareto: {dso_csv} not found")
        return
    df = pd.read_csv(dso_csv)
    g = df.groupby("method")[["c_cap", "c_act", "c_shed", "c_total"]].mean()
    methods = [m for m in ORDER if m in g.index][::-1]
    fig, ax = plt.subplots(figsize=(3.5, 1.9))
    y = np.arange(len(methods))
    serv = [g.loc[m, "c_cap"] + g.loc[m, "c_act"] for m in methods]
    shed = [g.loc[m, "c_shed"] for m in methods]
    ax.barh(y, serv, height=0.6, color=[METHOD_COLORS[m] for m in methods],
            edgecolor=[METHOD_EDGE_COLORS[m] for m in methods], linewidth=0.5,
            label="capacity + activation")
    ax.barh(y, shed, height=0.6, left=serv, color="#c0392b", alpha=0.85,
            edgecolor="#7c2418", linewidth=0.5, label="load shedding (VOLL)")
    for i, m in enumerate(methods):
        tot = g.loc[m, "c_total"]
        ax.annotate(f"{tot:.2f}", (serv[i] + shed[i], i), xytext=(3, 0),
                    textcoords="offset points", va="center", fontsize=6.5,
                    fontweight="bold" if m == PROPOSED else "normal")
    ax.set_yticks(y)
    ax.set_yticklabels([SHORT[m] for m in methods], fontsize=7)
    ax.set_xlabel("Average DSO cost per contingency event (€)")
    ax.margins(x=0.14)
    # One-row legend above the axes: never collides with bars or labels.
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2,
              fontsize=6.5, frameon=False)
    fig.subplots_adjust(left=0.27, right=0.99, top=0.93, bottom=0.21)
    save(fig, "fig_pareto", [TEMPLATE / "img" / "fig_pareto.pdf"])


# ------------------------------------------------------------ fig_freq_grid
def fig_freq_grid(ckpts: dict[str, Path], n_runs: int = 2) -> None:
    """2x2 frequency-response grid from fresh episodes (hi-res traces)."""
    from src.eval.eval_ffr_topology import FFRTopologyEvaluator, EventConfig

    env_config = {
        "placement_path": "artifacts/placement/official_placement_v3.json",
        "mpc_path": "data/grid_IEEE123_complete.m",
        "seed": 42, "ffr_mode": "mappo_dual", "day_split": "eval",
    }
    ev = FFRTopologyEvaluator(
        env_config=env_config,
        checkpoint_path=ckpts["proposed"], gcnn_checkpoint=ckpts["gcnn"],
        matd3_checkpoint=ckpts["matd3"], mlp_mappo_checkpoint=ckpts["mlp"],
        output_dir=OUT / "_freqgrid_tmp", base_reference=True,
    )
    ev.env.hires_substeps = 20  # 0.05 s -- ample at column width, small PDF
    apply_paper_style()         # evaluator import may have applied legacy style

    # Designed for SINGLE-COLUMN inclusion (main.tex: 0.95\linewidth):
    # 3.5 in wide so the 8/7 pt fonts print at true size with no shrink.
    fig, axs = plt.subplots(2, 2, figsize=(3.5, 4.2), sharex=True, sharey=True)
    t0, t1v = 28.0, 50.0
    for ax, (scen, event) in zip(axs.flat, ev.scenarios.items()):
        over = scen == "S4_high_ren_surge"
        for m in ORDER:
            pol = ev.policies.get(m)
            if pol is None:
                continue
            runs = []
            for _ in range(n_runs):
                met = ev.run_episode(pol, event=event, topology_idx=-1)
                tr = met.f_trace_hires if met.f_trace_hires.size else met.f_trace
                runs.append(np.asarray(tr, dtype=float))
            L = min(len(r) for r in runs)
            mat = np.stack([r[:L] for r in runs])
            dt = met.dt_hires if met.f_trace_hires.size else 1.0
            t = np.arange(L) * dt
            mean = mat.mean(axis=0)
            sel = (t >= t0) & (t <= t1v)
            td, yd = decimate_trace(t[sel], mean[sel], 450)
            # Thinner strokes at single-column size; proposed still dominant.
            lw = 1.4 if m == PROPOSED else 0.7
            ax.plot(td, yd, color=METHOD_COLORS[m], linewidth=lw,
                    linestyle=LINESTYLE[m], zorder=6 if m == PROPOSED else 4,
                    label=SHORT[m])
            if m == PROPOSED and n_runs > 1:
                sd = mat.std(axis=0)
                _, lo = decimate_trace(t[sel], (mean - sd)[sel], 450)
                _, hi = decimate_trace(t[sel], (mean + sd)[sel], 450)
                ax.fill_between(td, lo, hi, color=METHOD_COLORS[m], alpha=0.15,
                                linewidth=0, zorder=3)
        ax.axvline(30.0, color="#555555", linewidth=0.5, linestyle="--", zorder=2)
        ax.axhline(49.0, color="#a01818", linewidth=0.6, linestyle=(0, (4, 2)), zorder=2)
        ax.axhline(49.5 if not over else 50.5, color="#e08a18", linewidth=0.6,
                   linestyle=(0, (4, 2)), zorder=2)
        ax.axhspan(49.95, 50.05, color="#3a9d5d", alpha=0.10, zorder=1)
        ax.set_title(SCEN_TITLE[scen], fontsize=7, pad=2)
        ax.set_xlim(t0, t1v)
        ax.set_xticks([30, 40, 50])
        ax.tick_params(labelsize=6.5)
    for ax in axs[1, :]:
        ax.set_xlabel("Time (s)", fontsize=7)
    for ax in axs[:, 0]:
        ax.set_ylabel("Frequency (Hz)", fontsize=7)
    handles, labels = axs[0, 0].get_legend_handles_labels()
    extra = [
        plt.Line2D([], [], color="#a01818", linewidth=0.6, linestyle=(0, (4, 2)), label="UFLS-1 49.0 Hz"),
        plt.Line2D([], [], color="#e08a18", linewidth=0.6, linestyle=(0, (4, 2)), label="alarm 49.5/50.5 Hz"),
    ]
    fig.legend(handles=handles + extra, loc="upper center", ncol=3, frameon=False,
               bbox_to_anchor=(0.54, 1.005), fontsize=6, columnspacing=0.7,
               handlelength=1.5, labelspacing=0.3)
    fig.subplots_adjust(top=0.855, bottom=0.075, left=0.125, right=0.99,
                        hspace=0.16, wspace=0.06)
    save(fig, "fig_freq_grid",
         [TEMPLATE / "section1_stability" / "fig_freq_grid.pdf"])


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", type=Path,
                   default=ROOT / "results" / "eval_final_band005_n20")
    p.add_argument("--dso-csv", type=Path,
                   default=ROOT / "results" / "ffr_topology_baseref_final" / "dso_cost_per_event.csv")
    p.add_argument("--with-freq-grid", action="store_true")
    p.add_argument("--freq-grid-runs", type=int, default=2)
    args = p.parse_args()

    apply_paper_style()
    print(f"Reading {args.results_dir}")
    fig_radar(args.results_dir)
    fig_severity(args.results_dir)
    fig_topology(args.results_dir)
    fig_unseen_iae(args.results_dir)
    fig_pareto(args.dso_csv)
    if args.with_freq_grid:
        fig_freq_grid({
            "proposed": ROOT / "artifacts/ckpt_proposed_s42/am_mappo_final.pt",
            "gcnn": ROOT / "artifacts/ckpt_gcnn_ppo/final.pt",
            "matd3": ROOT / "artifacts/ckpt_matd3/matd3_ep5700.pt",
            "mlp": ROOT / "artifacts/ckpt_mlp_mappo/mlp_mappo_final.pt",
        }, n_runs=args.freq_grid_runs)
    print(f"Done -> {OUT}")


if __name__ == "__main__":
    main()
