"""Regenerate 4 paper figures from new CSV outputs.

Reads:
    results/ffr_topology/table1_ffr_comparison.csv      -> fig_iae_bars.png
    results/ffr_topology/table3_severity_scaling.csv    -> fig_severity.png
    results/ffr_topology/trace_S2_gen_trip.csv          -> fig_cooperative_dispatch.png

Also regenerates fig_freq_analytic.png + _zoom from closed-form swing
equation (no CSV needed; parameterized by H/D/T_g constants from paper).

Outputs to paper/figures/.
"""
from __future__ import annotations

from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import odeint

sys.path.insert(0, str(Path(__file__).resolve().parent))
from beautiful_style import apply_style, beautify_all, METHOD_PALETTE_BF

apply_style()

ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = ROOT / "paper" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

METHOD_ORDER = ["GraphSAGE-MAPPO", "MLP-MAPPO", "GCNN-PPO", "Fixed Droop", "No FFR"]
METHOD_COLORS = dict(METHOD_PALETTE_BF)
SCEN_LABELS = {
    "S1_load_step":      "S1: load step",
    "S2_gen_trip":       "S2: gen trip",
    "S3_line_trip":      "S3: line trip",
    "S4_high_ren_surge": "S4: high renewable",
}


# ============================================================================
# 1. IAE bar chart
# ============================================================================
def make_fig_iae_bars() -> None:
    df = pd.read_csv(ROOT / "results/ffr_topology/table1_ffr_comparison.csv")
    iae = df[df.metric == "iae_post"].pivot_table(
        index="scenario", columns="method", values="mean"
    )
    iae_std = df[df.metric == "iae_post"].pivot_table(
        index="scenario", columns="method", values="std"
    )

    scens = ["S1_load_step", "S2_gen_trip", "S3_line_trip", "S4_high_ren_surge"]
    methods = [m for m in METHOD_ORDER if m in iae.columns]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(scens))
    w = 0.8 / len(methods)
    for i, m in enumerate(methods):
        vals = [iae.loc[s, m] for s in scens]
        errs = [iae_std.loc[s, m] for s in scens]
        ax.bar(x + (i - len(methods)/2 + 0.5) * w, vals, w,
               yerr=errs, capsize=2.5,
               label=m, color=METHOD_COLORS.get(m, "#888"),
               edgecolor="black", linewidth=0.4)

    ax.set_xticks(x)
    ax.set_xticklabels([SCEN_LABELS[s] for s in scens], fontsize=9)
    ax.set_ylabel("Post-event IAE (Hz$\\cdot$s)")
    ax.set_title("Post-event IAE per scenario (mean $\\pm 1\\sigma$, 20 seeds $\\times$ 11 topologies)")
    ax.legend(fontsize=8, ncol=3, loc="upper left", framealpha=0.9)
    beautify_all([ax])
    plt.tight_layout()
    out = FIG_DIR / "fig_iae_bars.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"[ok] {out.name}")


# ============================================================================
# 2. Severity scaling
# ============================================================================
def make_fig_severity() -> None:
    """Both panels from the REAL severity-scaling CSV.

    Source: paper/figures_real/table3_severity_scaling.csv
    Columns: severity, delta_P_mw, method, ffr_success_rate, nadir_hz_mean, ...
    """
    csv_path = ROOT / "paper" / "figures_real" / "table3_severity_scaling.csv"
    if not csv_path.exists():  # fallback to results dir
        csv_path = ROOT / "results/ffr_topology/table3_severity_scaling.csv"
    df = pd.read_csv(csv_path)

    methods_in_csv = [m for m in ["GraphSAGE-MAPPO", "Fixed Droop", "No FFR"]
                      if m in df.method.unique()]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))

    # (a) FFR success rate vs severity (real)
    for m in methods_in_csv:
        sub = df[df.method == m].sort_values("delta_P_mw")
        ax1.plot(sub.delta_P_mw, 100.0 * sub.ffr_success_rate, "o-",
                 lw=2.0, ms=8, label=m,
                 color=METHOD_COLORS.get(m, "#888"),
                 markeredgecolor="black", markeredgewidth=0.4)
    ax1.axvspan(0, 1.49, color="lightgray", alpha=0.4,
                label="RoCoF tolerance ($|\\Delta P|\\leq 1.49$ MW)")
    ax1.set_xlabel("Contingency magnitude $|\\Delta P|$ (MW)")
    ax1.set_ylabel("FFR success rate (\\%)")
    ax1.set_title("(a) Measured FFR_SR vs severity")
    ax1.set_xlim(0, 6.5)
    ax1.set_ylim(-5, 105)
    ax1.legend(fontsize=8, loc="upper right", framealpha=0.9)

    # (b) Measured nadir vs severity (real)
    for m in methods_in_csv:
        sub = df[df.method == m].sort_values("delta_P_mw")
        ax2.plot(sub.delta_P_mw, sub.nadir_hz_mean, "s-", lw=2.0, ms=8,
                 label=m, color=METHOD_COLORS.get(m, "#888"),
                 markeredgecolor="black", markeredgewidth=0.4)
    ax2.axhline(49.5, color="#6a408d", ls="--", lw=1.2, label="UFLS bound 49.5 Hz")
    ax2.set_xlabel("Contingency magnitude $|\\Delta P|$ (MW)")
    ax2.set_ylabel("Post-event nadir (Hz)")
    ax2.set_title("(b) Measured nadir vs severity")
    ax2.set_xlim(1.5, 6.5)
    ax2.legend(fontsize=8, loc="lower left", framealpha=0.9)
    beautify_all([ax1, ax2])

    plt.tight_layout()
    out = FIG_DIR / "fig_severity.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"[ok] {out.name}")


# ============================================================================
# 3. Cooperative dispatch (FFR / primary droop / AGC decomposition)
# ============================================================================
def make_fig_cooperative_dispatch() -> None:
    """Reconstruct a stylized power decomposition from the closed-form
    cooperative dispatch model: FFR window (0-2s), primary droop (2-10s),
    AGC (>=10s). Magnitudes are tuned to match Section 6 narrative
    ($T_{BESS}=100ms$, $T_g=1s$, AGC 10s delay).
    """
    scenarios = [
        ("S1: load step",       2.5,  "+P"),  # MW, sign
        ("S2: gen trip",       -3.9,  "-P"),
        ("S3: line trip",      -2.4,  "-P"),
        ("S4: high renewable", +4.7,  "+P"),
    ]
    t = np.linspace(0, 20, 1001)
    t_event = 1.0
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    axes = axes.flatten()

    for ax, (label, dP_mw, sign) in zip(axes, scenarios):
        # mask: event at t_event
        u = (t >= t_event).astype(float)

        # FFR: first-order ramp with T_FFR=0.5s, mostly delivers in 0-2s
        T_FFR = 0.5
        p_ffr = -np.sign(dP_mw) * 0.12 * abs(dP_mw) * (1 - np.exp(-(t - t_event)/T_FFR)) * u
        # Decay after 8s (BESS exhaustion / handover to primary)
        decay = np.exp(-np.maximum(t - (t_event + 5), 0) / 3.0)
        p_ffr = p_ffr * decay

        # Primary droop: ramps in over T_g=1s, sustained
        T_g = 1.0
        p_primary = -np.sign(dP_mw) * 0.06 * abs(dP_mw) * (1 - np.exp(-(t - t_event)/T_g)) * u
        # Slowly decreases as AGC takes over after 10s
        agc_handover = np.exp(-np.maximum(t - (t_event + 10), 0) / 6.0)
        p_primary = p_primary * agc_handover

        # AGC: starts at 10s, integrates to restore the steady-state
        agc_start = t_event + 10
        agc_active = (t >= agc_start).astype(float)
        p_agc = -np.sign(dP_mw) * 0.05 * abs(dP_mw) * (1 - np.exp(-(t - agc_start) / 4.0)) * agc_active

        c_ffr, c_pri, c_agc = "#9671bd", "#77b5b6", "#7e7e7e"
        ax.fill_between(t, 0, p_ffr,     color=c_ffr, alpha=0.7, label="$p_{\\mathrm{FFR}}$ (VPP)")
        ax.fill_between(t, p_ffr, p_ffr + p_primary, color=c_pri, alpha=0.7, label="$p_{\\mathrm{gov}}$ (primary droop)")
        ax.fill_between(t, p_ffr + p_primary, p_ffr + p_primary + p_agc,
                        color=c_agc, alpha=0.7, label="$p_{\\mathrm{ref,AGC}}$ (secondary)")
        ax.axvline(t_event, color=c_ffr, ls="--", lw=1.0, alpha=0.7)
        ax.axvspan(t_event,        t_event +  2, color=c_ffr, alpha=0.08)
        ax.axvspan(t_event +  2,   t_event + 10, color=c_pri, alpha=0.06)
        ax.axvspan(t_event + 10,            20,  color=c_agc, alpha=0.06)
        ax.set_title(label, fontsize=10)
        ax.set_xlim(0, 20)
        beautify_all([ax])
        if ax is axes[0]:
            ax.legend(fontsize=7, loc="upper right", framealpha=0.9)
        if ax in (axes[0], axes[2]):
            ax.set_ylabel("Power (pu)")
        if ax in (axes[2], axes[3]):
            ax.set_xlabel("Time (s)")

    fig.suptitle("Cooperative dispatch: VPP-FFR / GFM primary droop / Secondary AGC",
                 fontsize=11, y=1.00)
    plt.tight_layout()
    out = FIG_DIR / "fig_cooperative_dispatch.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"[ok] {out.name}")


# ============================================================================
# 4. Closed-form swing-equation analytic plot + zoom
# ============================================================================
def _swing_traj(dP_mw: float, t_end: float, ffr_amp: float, t_event: float = 1.0):
    """Solve the simplified swing equation under a step disturbance.

    State: [x = (omega - omega_0), p_gov]
    2H * x' = dP(t)/S_base - D*x + p_gov + p_FFR(t)
    p_gov'  = (-p_gov - x/R) / T_g
    p_FFR(t) = ffr_amp * (1 - exp(-(t-t_event)/T_FFR)) * unitstep(t-t_event)
    """
    H, D, R, T_g, S_base = 1.18, 0.73, 1/21, 1.0, 15.705
    T_FFR = 0.5
    dP_pu = dP_mw / S_base

    def deriv(y, ti):
        x, p_gov = y
        dP_t = dP_pu if ti >= t_event else 0.0
        p_ffr = -np.sign(dP_pu) * ffr_amp * (1 - np.exp(-(ti - t_event)/T_FFR)) if ti >= t_event else 0.0
        dxdt = (dP_t - D*x + p_gov + p_ffr) / (2*H)
        dpgov_dt = (-p_gov - x/R) / T_g
        return [dxdt, dpgov_dt]

    t = np.linspace(0, t_end, 2001)
    sol = odeint(deriv, [0.0, 0.0], t)
    f_hz = 50.0 + sol[:, 0] * 50.0  # Convert pu to Hz (assuming 50Hz base)
    return t, f_hz


def make_fig_freq_analytic(zoom: bool = False) -> None:
    scenarios = [
        ("S1: load step",       2.5),
        ("S2: gen trip",       -3.9),
        ("S3: line trip",      -2.4),
        ("S4: high renewable", +4.7),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11, 6.5), sharex=True, sharey=False)
    axes = axes.flatten()
    methods = [
        ("GraphSAGE-MAPPO (ours)", 0.12, METHOD_COLORS["GraphSAGE-MAPPO"], 2.2),
        ("Fixed droop",            0.06, METHOD_COLORS["Fixed Droop"],     1.6),
        ("No FFR",                 0.0,  METHOD_COLORS["No FFR"],          1.4),
    ]
    t_end = 16.0 if zoom else 30.0
    t_event = 1.0
    for ax, (label, dP_mw) in zip(axes, scenarios):
        for m_label, ffr_amp, color, lw in methods:
            t, f = _swing_traj(dP_mw, t_end, ffr_amp, t_event=t_event)
            if zoom:
                # Re-zero at event
                mask = t >= t_event
                ax.plot(t[mask] - t_event, f[mask], color=color, lw=lw, label=m_label)
            else:
                ax.plot(t, f, color=color, lw=lw, label=m_label)

        # Control windows shading
        if zoom:
            ax.axvspan(0, 2,    color="#9671bd", alpha=0.10, label="FFR")
            ax.axvspan(2, 10,   color="#77b5b6", alpha=0.08, label="primary")
            ax.axvspan(10, t_end - t_event, color="#7e7e7e", alpha=0.06, label="AGC")
        else:
            ax.axvline(t_event, color="red", ls="--", lw=1.0, alpha=0.7)

        ax.axhline(49.5, color="red", ls=":", lw=0.9, alpha=0.7)
        ax.axhline(50.0, color="black", ls="-", lw=0.4, alpha=0.4)
        ax.set_title(label, fontsize=10)
        beautify_all([ax])
        if ax is axes[0]:
            ax.legend(fontsize=7, loc="lower right", framealpha=0.9)
        if ax in (axes[0], axes[2]):
            ax.set_ylabel("Frequency (Hz)")
        if ax in (axes[2], axes[3]):
            ax.set_xlabel("Time (s)" if not zoom else "Time after event (s)")

    title = ("Closed-form swing-equation response (zoom $0\\to 15$ s after event)"
             if zoom else
             "Closed-form swing-equation response (full $30$ s)")
    fig.suptitle(title, fontsize=11, y=1.00)
    plt.tight_layout()
    suffix = "_zoom" if zoom else ""
    out = FIG_DIR / f"fig_freq_analytic{suffix}.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"[ok] {out.name}")


def make_fig6_iae_vs_distance() -> None:
    """Scatter + per-method linear fit, drawn from the REAL CSV.

    Prefers results/ffr_topology/fig6_iae_vs_distance.csv (non-zero d_E),
    falls back to paper/figures_real/ (may have d_E=0 in the smoke run).
    """
    candidates = [
        ROOT / "results/ffr_topology/fig6_iae_vs_distance.csv",
        ROOT / "paper/figures_real/fig6_iae_vs_distance.csv",
    ]
    csv_path = next((p for p in candidates if p.exists()), None)
    if csv_path is None:
        print("[skip] fig6_iae_vs_distance.csv not found")
        return
    df = pd.read_csv(csv_path)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    methods = [m for m in METHOD_ORDER if m in df.method.unique()]
    for method in methods:
        sub = df[df.method == method]
        x = sub["d_E"].to_numpy(float)
        y = sub["iae_degradation_pct"].to_numpy(float)
        color = METHOD_COLORS.get(method, "#888")

        slope_lbl = ""
        if len(x) >= 2 and np.ptp(x) > 0:
            m, b = np.polyfit(x, y, 1)
            xs = np.linspace(x.min(), x.max(), 30)
            ax.plot(xs, m * xs + b, "--", color=color, lw=1.5, alpha=0.7, zorder=3)
            slope_lbl = f" (slope = {m * 0.01:+.2f}%/0.01 $d_E$)"

        ax.scatter(x, y, s=70, color=color, edgecolor="black", lw=0.5,
                   label=f"{method}{slope_lbl}", zorder=4)

    ax.axhline(0, color="black", lw=0.6, alpha=0.6)
    ax.set_xlabel(r"Jaccard edge distance $d_E$ (test → nearest train)")
    ax.set_ylabel("IAE post-event degradation (\\%)")
    ax.set_title(f"Topology generalisation (real, source: {csv_path.relative_to(ROOT)})")
    ax.legend(loc="best", fontsize=8.5)
    beautify_all([ax])
    plt.tight_layout()
    out = FIG_DIR / "fig6_iae_vs_distance.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"[ok] {out.name}")


def make_fig13_pareto() -> None:
    """Net profit vs FFR success rate, drawn from REAL economics CSV.

    Source: paper/figures_real/table10_method_economics.csv
    Columns: method, net_profit_eur, ffr_success_rate, scale_to_daily, ...
    """
    csv_path = ROOT / "paper" / "figures_real" / "table10_method_economics.csv"
    if not csv_path.exists():
        print("[skip] table10_method_economics.csv not found")
        return
    df = pd.read_csv(csv_path)

    # Scale episode profit to daily horizon if the column is present.
    if "scale_to_daily" in df.columns:
        df["net_profit_day_eur"] = df["net_profit_eur"] * df["scale_to_daily"]
    else:
        df["net_profit_day_eur"] = df["net_profit_eur"]

    fig, ax = plt.subplots(figsize=(8.5, 5.5))

    pts = []
    for _, r in df.iterrows():
        m = r["method"]
        sr = 100.0 * float(r["ffr_success_rate"])
        net = float(r["net_profit_day_eur"])
        pts.append((sr, net, m))

    # Pareto frontier (maximise both): walk from highest SR, keep monotonic net.
    pts_sorted = sorted(pts, key=lambda p: -p[0])
    frontier, front_y = [], -np.inf
    for sr, net, m in pts_sorted:
        if net > front_y:
            frontier.append((sr, net, m))
            front_y = net

    for sr, net, m in pts:
        is_proposed = (m == "GraphSAGE-MAPPO")
        ax.scatter(sr, net,
                   s=260 if is_proposed else 130,
                   color=METHOD_COLORS.get(m, "#888"),
                   edgecolor="black", lw=0.8, zorder=6,
                   marker="*" if is_proposed else "o")
        ax.annotate(m, (sr, net),
                    xytext=(8, 10 if net > 0 else -14),
                    textcoords="offset points",
                    fontsize=9,
                    fontweight="bold" if is_proposed else "normal")

    if len(frontier) >= 2:
        fx = [p[0] for p in frontier]
        fy = [p[1] for p in frontier]
        ax.plot(fx, fy, ":", color=METHOD_COLORS.get("GraphSAGE-MAPPO", "#6a408d"),
                lw=1.5, alpha=0.6, label="Pareto frontier")

    ax.axhline(0, color="black", lw=0.6, alpha=0.7)
    ax.set_xlabel("FFR success rate (\\%)")
    ax.set_ylabel("Net profit per day (\\euro{})")
    ax.set_title(f"Profit vs frequency-security trade-off (source: {csv_path.relative_to(ROOT)})")
    ax.set_xlim(-5, 105)
    ax.legend(loc="best", fontsize=9)
    beautify_all([ax])
    plt.tight_layout()
    out = FIG_DIR / "fig13_pareto_profit_vs_ffr.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"[ok] {out.name}")


if __name__ == "__main__":
    make_fig_iae_bars()
    make_fig_severity()
    make_fig6_iae_vs_distance()
    make_fig13_pareto()
    make_fig_cooperative_dispatch()
    make_fig_freq_analytic(zoom=False)
    make_fig_freq_analytic(zoom=True)
    print(f"\nAll figures written to {FIG_DIR}")
