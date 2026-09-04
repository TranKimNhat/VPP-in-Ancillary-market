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
    "Fixed Droop":     "#ff7f0e",  # orange — distinct from the No-FFR gray
    "No FFR":          "#3a3a3a",  # near-black
}
METHOD_EDGE_COLORS = {
    "GraphSAGE-MAPPO": "#8f1414",
    "MATD3":           "#14507e",
    "GCNN-PPO":        "#1c6b1c",
    "MLP-MAPPO":       "#5e3a87",
    "Fixed Droop":     "#b35900",
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
    # Per-scenario topology-robustness score: lowest unseen-mean IAE in the
    # panel / this method's unseen-mean IAE (1 = most robust). Falls back to the
    # system-level fig6 score (identical in every panel) if the per-scenario
    # sweep CSV is absent.
    topo_csv = rdir / "topology_score_per_scenario.csv"
    topo_by_scen: dict[object, dict[str, float]] = {}
    if topo_csv.exists():
        topo = pd.read_csv(topo_csv)
        for scen, g in topo.groupby("scenario"):
            s = g.set_index("method")["itae_test_mean"]
            topo_by_scen[scen] = (s.min() / s).to_dict()
    else:
        f6 = pd.read_csv(rdir / "fig6_iae_vs_distance.csv")
        unseen = f6.groupby("method")["itae_test"].mean()
        topo_by_scen[None] = (unseen.min() / unseen).to_dict()
        print("  [fig_radar] topology_score_per_scenario.csv missing; "
              "using system-level fig6 ITAE score (repeated across panels)")

    # Short single-line labels: two-line labels collided with neighbouring
    # panels and the figure legend at print size.
    axes_labels = ["ITAE", "Margin", "Settling", "Reserve", "Topology"]
    n_ax = len(axes_labels)
    angles = np.linspace(0, 2 * np.pi, n_ax, endpoint=False).tolist()
    angles += angles[:1]

    fig, axs = plt.subplots(2, 2, figsize=(7.16, 6.2), subplot_kw=dict(polar=True))
    for ax, scen in zip(axs.flat, SCEN_TITLE):
        over = scen == "S4_high_ren_surge"
        topo_score = topo_by_scen.get(scen, topo_by_scen.get(None, {}))
        vals = {}
        for m in ORDER:
            row = t1.loc[(scen, m)]
            itae = 1.0 / max(row["itae"], 1e-9)
            margin = (51.0 - row["zenith_hz"]) if over else (row["nadir_hz"] - 48.0)
            settle = 1.0 / max(row["settling_time_s"], 1e-9)
            reserve = row["ffr_success"]
            vals[m] = [itae, max(margin, 0.0), settle, reserve, topo_score.get(m, 0.0)]
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
        # Radial scale: values are fractions of the best in-panel method
        # (1.0 = best). Labels sit at 36 deg -- the gap between the first two
        # spokes -- in a small grey font so they clear the spoke labels.
        ax.set_yticks([0.5, 1.0])
        ax.set_yticklabels(["0.5", "1.0"], fontsize=5.5, color="#666666")
        ax.set_rlabel_position(36)
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
    fig, ax1 = plt.subplots(figsize=PAPER_FIGSIZE_COL)
    for m in ORDER:
        d = t3[t3.method == m].sort_values("delta_P_mw")
        kw = dict(color=METHOD_COLORS[m], linestyle=LINESTYLE[m], linewidth=lw_for(m),
                  marker="o", markersize=3.2 if m == PROPOSED else 2.5,
                  markeredgecolor=METHOD_EDGE_COLORS[m], markeredgewidth=0.4,
                  zorder=6 if m == PROPOSED else 4, label=SHORT[m])
        # All four learning methods sit exactly on the 49.5 Hz plateau; draw the
        # dashed baselines ABOVE the thick proposed line so their dash patterns
        # stay visible (otherwise only ours is seen).
        kw_a = {**kw, "zorder": 7 if m != PROPOSED else 6}
        ax1.plot(d.delta_P_mw, d.nadir_hz_mean, **kw_a)
    ax1.axhline(49.0, color="#b03030", linewidth=0.7, linestyle="--", zorder=2)
    ax1.text(6.0, 49.03, "UFLS-1 49.0 Hz", fontsize=6.5, color="#b03030", ha="right")
    ax1.set_ylabel("Nadir (Hz)")
    ax1.set_xlabel(r"Disturbance magnitude $|\Delta P|$ (MW)")
    # Figure-level legend ABOVE the axes: in-panel placement collided with the
    # descending Fixed-droop / No-FFR curves.
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False,
               bbox_to_anchor=(0.56, 1.005), fontsize=6.5, columnspacing=0.8,
               handlelength=1.6)
    fig.subplots_adjust(left=0.13, right=0.99, top=0.86, bottom=0.16)
    save(fig, "fig_severity", [TEMPLATE / "img" / "fig_severity.pdf"])


# ----------------------------------------------------------- fig_unseen_iae
def fig_unseen_iae(rdir: Path) -> None:
    f6 = pd.read_csv(rdir / "fig6_iae_vs_distance.csv")
    t2 = pd.read_csv(rdir / "table2_topology_adaptation.csv")
    base = t2[t2.topology_split == "base"].set_index("method")["iae_post_mean"]
    uns = t2[t2.topology_split == "unseen"].set_index("method")["iae_post_mean"]
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
    # Overlay the training-G0 diamond, the unseen-mean dot, and the
    # train-to-unseen gap label (merges the former fig_topology dumbbell).
    # base/unseen means come from Table 2 so the labels match it exactly.
    for i, m in enumerate(methods):
        if m in uns.index:
            mu = float(uns[m])
            ax.scatter([mu], [i], marker="o", s=16,
                       facecolor=METHOD_EDGE_COLORS[m], edgecolor="white",
                       linewidth=0.5, zorder=7)
        if m in base.index and m in uns.index:
            g0 = float(base[m])
            ax.scatter([g0], [i], marker="D", s=14, facecolor="white",
                       edgecolor=METHOD_EDGE_COLORS[m], linewidth=0.8, zorder=6)
            gap = 100.0 * (uns[m] - g0) / g0
            xr = max(float(np.max(data[i])), g0, uns[m])
            ax.annotate(f"{gap:+.0f}%", (xr, i), xytext=(4, 0),
                        textcoords="offset points", va="center", fontsize=6.0,
                        color=METHOD_EDGE_COLORS[m])
    ax.scatter([], [], marker="D", s=14, facecolor="white", edgecolor="#444444",
               linewidth=0.8, label="training $G_0$")
    ax.scatter([], [], marker="o", s=16, facecolor="#444444", edgecolor="white",
               linewidth=0.5, label="unseen mean")
    ax.legend(loc="upper right", fontsize=6.0, handletextpad=0.3, labelspacing=0.2)
    ax.set_yticks(range(len(methods)))
    ax.set_yticklabels([SHORT[m] for m in methods], fontsize=7)
    for lbl, m in zip(ax.get_yticklabels(), methods):
        if m == PROPOSED:
            lbl.set_fontweight("bold")
    n_unseen = int(f6["test_topology_id"].nunique())
    ax.set_xlabel(f"Post-event frequency error IAE (Hz·s),\nspread over the {n_unseen} unseen reconfigurations")
    # Explicit xlim: leave headroom on the right for the gap labels.
    allv = np.concatenate(data + [base.values])
    lo, hi = float(np.min(allv)), float(np.max(allv))
    ax.set_xlim(lo - 0.05 * (hi - lo), hi + 0.20 * (hi - lo))
    fig.subplots_adjust(left=0.27, right=0.99, top=0.99, bottom=0.27)  # 2-line xlabel
    save(fig, "fig_unseen_iae", [TEMPLATE / "img" / "fig_unseen_iae.pdf"])


# ---------------------------------------------------------- fig_unseen_itae
def fig_unseen_itae(rdir: Path) -> None:
    """Topology generalization on ITAE, told as a ranked bar (lower = better).

    Two messages, kept in separate visual channels so neither crowds the other:
      - bar length + whisker = unseen-mean ITAE and its spread over the unseen
        reconfigurations  (which method is best AND how consistent it is);
      - hollow diamond = the training-G0 ITAE  (how little the score moves from
        the trained feeder to unseen ones = generalization).
    Linear x: a log axis compressed the four learned controllers into one
    indistinguishable cluster; on a linear ranked bar the ~10x gap to No-FFR is
    the headline and the learned ordering is still legible via the value labels.
    """
    f6 = pd.read_csv(rdir / "fig6_iae_vs_distance.csv")
    t2 = pd.read_csv(rdir / "table2_topology_adaptation.csv")
    base = t2[t2.topology_split == "base"].set_index("method")["itae_mean"]
    # Per-method unseen mean/std over the VALID (event-perturbed) topologies.
    # A few heavily reconfigured topologies island the S2 event bus -> flat
    # trace -> ITAE=0; those are invalid points and are dropped.
    stat: dict[str, tuple[float, float]] = {}
    for m in ORDER:
        v = f6[(f6.method == m) & (f6["itae_test"] > 0)]["itae_test"].values
        if len(v):
            stat[m] = (float(np.mean(v)), float(np.std(v)))
    # Rank worst->best; barh fills bottom-up, so the best (lowest) ends up on top.
    methods = sorted((m for m in ORDER if m in stat), key=lambda m: stat[m][0],
                     reverse=True)
    y = np.arange(len(methods))
    means = np.array([stat[m][0] for m in methods])
    stds = np.array([stat[m][1] for m in methods])

    fig, ax = plt.subplots(figsize=(3.5, 2.2))
    for i, m in enumerate(methods):
        ax.barh(i, means[i], height=0.62, color=METHOD_COLORS[m],
                alpha=0.88 if m == PROPOSED else 0.7,
                edgecolor=METHOD_EDGE_COLORS[m],
                linewidth=1.1 if m == PROPOSED else 0.6,
                zorder=4 if m == PROPOSED else 3)
    # Channel 1: spread over unseen topologies.
    ax.errorbar(means, y, xerr=stds, fmt="none", ecolor="#333333",
                elinewidth=0.7, capsize=2.0, zorder=6)
    # Channel 2: training-G0 reference (hollow diamond) = generalization anchor.
    for i, m in enumerate(methods):
        if m in base.index:
            ax.scatter([float(base[m])], [i], marker="D", s=20, facecolor="white",
                       edgecolor=METHOD_EDGE_COLORS[m], linewidth=0.9, zorder=7)
    # Value labels (unseen mean, Hz*s^2) past the whisker.
    for i, m in enumerate(methods):
        ax.annotate(f"{means[i]:,.0f}", (means[i] + stds[i], i), xytext=(4, 0),
                    textcoords="offset points", va="center", fontsize=6.4,
                    color=METHOD_EDGE_COLORS[m],
                    fontweight="bold" if m == PROPOSED else "normal")
    ax.scatter([], [], marker="D", s=20, facecolor="white", edgecolor="#444444",
               linewidth=0.9, label=r"training $G_0$")
    # Upper-right is empty (the best controllers have short bars), so the
    # legend sits there instead of colliding with the long No-FFR bar/diamond.
    ax.legend(loc="upper right", fontsize=6.4, handletextpad=0.3, frameon=False)
    ax.set_yticks(y)
    ax.set_yticklabels([SHORT[m] for m in methods], fontsize=7)
    for lbl, m in zip(ax.get_yticklabels(), methods):
        if m == PROPOSED:
            lbl.set_fontweight("bold")
    ax.set_xlabel("Post-event ITAE (Hz·s)", fontsize=8)
    ax.set_xlim(0, float((means + stds).max()) * 1.16)
    ax.set_ylim(-0.6, len(methods) - 0.4)
    ax.grid(axis="x", linewidth=0.4, alpha=0.5)
    ax.set_axisbelow(True)
    fig.subplots_adjust(left=0.27, right=0.975, top=0.985, bottom=0.2)
    save(fig, "fig_unseen_itae", [TEMPLATE / "img" / "fig_unseen_itae.pdf"])


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
    # Encoding dimension here is the COST COMPONENT (not the method): a single
    # steel-blue service segment + a crimson shedding segment, so the legend is
    # unambiguous. The proposed method is emphasized by its bold axis label
    # (all bars share the same outline weight).
    SERV_C, SHED_C = "#3b6ea5", "#c0392b"
    ax.barh(y, serv, height=0.6, color=SERV_C,
            edgecolor="#26465f", linewidth=0.5,
            label="capacity + activation")
    ax.barh(y, shed, height=0.6, left=serv, color=SHED_C, alpha=0.9,
            edgecolor="#7c2418", linewidth=0.5, label="load shedding (VOLL)")
    for i, m in enumerate(methods):
        tot = g.loc[m, "c_total"]
        ax.annotate(f"{tot:.2f}", (serv[i] + shed[i], i), xytext=(3, 0),
                    textcoords="offset points", va="center", fontsize=6.5)
    ax.set_yticks(y)
    ax.set_yticklabels([SHORT[m] for m in methods], fontsize=7)
    for lbl, m in zip(ax.get_yticklabels(), methods):
        if m == PROPOSED:
            lbl.set_fontweight("bold")
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

    # Enlarged canvas (7.2 in) so the bumped ~11 pt fonts fit without title/
    # legend collisions; scale down in LaTeX (e.g. 0.95\linewidth) as needed.
    fig, axs = plt.subplots(2, 2, figsize=(7.2, 6.8), sharex=True, sharey=True)
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
            # Thicker baseline strokes for visibility; proposed still dominant.
            lw = 1.6 if m == PROPOSED else 1.2
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
        ax.set_title(SCEN_TITLE[scen], fontsize=11.5, pad=4)
        ax.set_xlim(t0, t1v)
        ax.set_xticks([30, 40, 50])
        ax.tick_params(labelsize=11)
    for ax in axs[1, :]:
        ax.set_xlabel("Time (s)", fontsize=11.5)
    for ax in axs[:, 0]:
        ax.set_ylabel("Frequency (Hz)", fontsize=11.5)
    handles, labels = axs[0, 0].get_legend_handles_labels()
    extra = [
        plt.Line2D([], [], color="#a01818", linewidth=0.6, linestyle=(0, (4, 2)), label="UFLS-1 49.0 Hz"),
        plt.Line2D([], [], color="#e08a18", linewidth=0.6, linestyle=(0, (4, 2)), label="alarm 49.5/50.5 Hz"),
    ]
    fig.legend(handles=handles + extra, loc="upper center", ncol=3, frameon=False,
               bbox_to_anchor=(0.54, 1.0), fontsize=10.5, columnspacing=1.0,
               handlelength=1.6, labelspacing=0.3)
    fig.subplots_adjust(top=0.86, bottom=0.085, left=0.10, right=0.99,
                        hspace=0.22, wspace=0.07)
    save(fig, "fig_freq_grid",
         [TEMPLATE / "img" / "fig_freq_grid.pdf"])


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", type=Path,
                   default=ROOT / "results" / "eval_final_band005_n20")
    p.add_argument("--dso-csv", type=Path,
                   default=ROOT / "results" / "ffr_topology_baseref_final" / "dso_cost_per_event.csv")
    p.add_argument("--itae-dir", type=Path, default=None,
                   help="Dir with ITAE-augmented table2/fig6 for fig_unseen_itae "
                        "(defaults to --results-dir).")
    p.add_argument("--with-freq-grid", action="store_true")
    p.add_argument("--freq-grid-runs", type=int, default=2)
    args = p.parse_args()

    apply_paper_style()
    print(f"Reading {args.results_dir}")
    fig_radar(args.results_dir)
    fig_severity(args.results_dir)
    fig_unseen_iae(args.results_dir)
    itae_dir = args.itae_dir if args.itae_dir is not None else args.results_dir
    try:
        fig_unseen_itae(itae_dir)
    except (KeyError, FileNotFoundError) as exc:
        print(f"  [skip] fig_unseen_itae ({itae_dir}): {exc}")
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
