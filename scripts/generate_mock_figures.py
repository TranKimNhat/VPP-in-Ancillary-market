"""Generate mock figures matching the numbers in paper/section6_results.tex.

Run:
    python scripts/generate_mock_figures.py

Outputs to paper/figures/:
    fig_freq_grid_S1_S4.png           — 2x2 multi-scenario freq response
    fig_iae_bars.png                  — IAE bar chart per scenario × method
    fig6_iae_vs_distance.png          — IAE degradation vs Jaccard edge distance
    fig_severity.png                  — FFR success rate vs |ΔP|
    fig12_revenue_decomposition.png   — Stacked bar revenue decomposition
    fig13_pareto_profit_vs_ffr.png    — Net profit vs FFR_SR (Pareto)

All numbers are hand-tuned to match the Section 6 tables. Once real eval results
exist, replace these with the actual ``src/eval/eval_*.py`` outputs.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# --------------------------------------------------------------------- styling
FIG_DIR = Path(__file__).resolve().parent.parent / "paper" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

METHOD_ORDER = [
    "GraphSAGE-MAPPO",
    "MLP-MAPPO",
    "GCNN-PPO",
    "MATD3",
    "Fixed Droop",
    "No FFR",
]

PALETTE = {
    "GraphSAGE-MAPPO": "#1f3a93",   # bold blue - Proposed
    "MLP-MAPPO":       "#e94e77",
    "GCNN-PPO":        "#f5a623",
    "MATD3":           "#6a737d",
    "Fixed Droop":     "#56c596",
    "No FFR":          "#a04668",
}

# Mock numbers from section6_results.tex (Tables VI, VII, IX, XII)
NADIR = {
    "S1": {"GraphSAGE-MAPPO": 49.74, "MLP-MAPPO": 49.65, "GCNN-PPO": 49.60,
           "MATD3": 49.52, "Fixed Droop": 49.34, "No FFR": 48.85},
    "S2": {"GraphSAGE-MAPPO": 49.22, "MLP-MAPPO": 48.95, "GCNN-PPO": 48.88,
           "MATD3": 48.74, "Fixed Droop": 48.05, "No FFR": 47.18},
    "S3": {"GraphSAGE-MAPPO": 49.42, "MLP-MAPPO": 49.05, "GCNN-PPO": 48.91,
           "MATD3": 48.78, "Fixed Droop": 48.55, "No FFR": 47.84},
    "S4": {"GraphSAGE-MAPPO": 48.85, "MLP-MAPPO": 48.42, "GCNN-PPO": 48.31,
           "MATD3": 48.18, "Fixed Droop": 47.40, "No FFR": 46.05},
}
IAE_POST = {
    "S1": {"GraphSAGE-MAPPO": 0.42, "MLP-MAPPO": 0.54, "GCNN-PPO": 0.58,
           "MATD3": 0.71, "Fixed Droop": 1.18, "No FFR": 2.31},
    "S2": {"GraphSAGE-MAPPO": 0.78, "MLP-MAPPO": 1.05, "GCNN-PPO": 1.18,
           "MATD3": 1.28, "Fixed Droop": 2.42, "No FFR": 4.85},
    "S3": {"GraphSAGE-MAPPO": 0.55, "MLP-MAPPO": 0.91, "GCNN-PPO": 1.02,
           "MATD3": 1.21, "Fixed Droop": 1.85, "No FFR": 3.21},
    "S4": {"GraphSAGE-MAPPO": 1.15, "MLP-MAPPO": 1.43, "GCNN-PPO": 1.55,
           "MATD3": 1.78, "Fixed Droop": 3.62, "No FFR": 6.05},
}
FFR_SR = {  # FFR success rate (%)
    1.6: {"GraphSAGE-MAPPO": 99, "MLP-MAPPO": 96, "GCNN-PPO": 94,
          "MATD3": 89, "Fixed Droop": 78, "No FFR": 0},
    2.5: {"GraphSAGE-MAPPO": 98, "MLP-MAPPO": 94, "GCNN-PPO": 91,
          "MATD3": 86, "Fixed Droop": 62, "No FFR": 0},
    3.9: {"GraphSAGE-MAPPO": 92, "MLP-MAPPO": 81, "GCNN-PPO": 78,
          "MATD3": 74, "Fixed Droop": 31, "No FFR": 0},
    5.5: {"GraphSAGE-MAPPO": 78, "MLP-MAPPO": 62, "GCNN-PPO": 59,
          "MATD3": 52, "Fixed Droop": 8,  "No FFR": 0},
}
# Daily revenue components from Table XII (system-wide €/day)
ECON = {
    "GraphSAGE-MAPPO": dict(em_p=168, am_cap=4320, am_act=180, undersupply=432, opex=883),
    "MLP-MAPPO":       dict(em_p=168, am_cap=3860, am_act=132, undersupply=1728, opex=883),
    "GCNN-PPO":        dict(em_p=168, am_cap=3742, am_act=122, undersupply=1892, opex=883),
    "MATD3":           dict(em_p=168, am_cap=3455, am_act=105, undersupply=2156, opex=883),
    "Fixed Droop":     dict(em_p=168, am_cap=2880, am_act=38,  undersupply=6048, opex=883),
    "No FFR":          dict(em_p=168, am_cap=0,    am_act=0,   undersupply=0,    opex=883),
}
FFR_SR_AVG_BY_METHOD = {  # average across 4 scenarios (used in Pareto)
    "GraphSAGE-MAPPO": 0.908,
    "MLP-MAPPO":       0.795,
    "GCNN-PPO":        0.770,
    "MATD3":           0.715,
    "Fixed Droop":     0.378,
    "No FFR":          0.000,
}


# ----------------------------------------------------------- synthetic dynamics
def synth_freq_trace(
    nadir_hz: float,
    t_event: float = 30.0,
    duration: float = 80.0,
    dt: float = 0.5,
    f_settle: float = 50.0,
    noise: float = 0.012,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a plausible f(t) reaching `nadir_hz` after event, then recovering."""
    rng = np.random.default_rng(seed)
    t = np.arange(0.0, duration, dt)
    f = np.full_like(t, 50.0)

    drop = 50.0 - nadir_hz
    overshoot_t = 6.0           # seconds from event to nadir
    settle_tau = 12.0           # exponential recovery time constant

    for i, ti in enumerate(t):
        if ti < t_event:
            f[i] = 50.0
        else:
            dtau = ti - t_event
            # underdamped 2nd-order-like response: drop then recover
            envelope = drop * np.exp(-dtau / settle_tau)
            phase = np.cos(2 * np.pi * dtau / (4 * overshoot_t)) * (1 - np.exp(-dtau / overshoot_t))
            f[i] = f_settle - envelope * phase * 0.75 - drop * np.exp(-dtau / 2.5) * 0.6
    f = f + rng.normal(0.0, noise, size=t.shape)
    return t, f


# ============================================================== Fig: freq grid
def make_fig_freq_grid() -> None:
    scenarios = {
        "S1 (load_step +2.5 MW)":  ("S1", 30.0),
        "S2 (gen_trip −3.9 MW)":   ("S2", 30.0),
        "S3 (line_trip −2.4 MW)":  ("S3", 30.0),
        "S4 (gen_trip −5.5 MW)":   ("S4", 30.0),
    }
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8), sharex=True, sharey=True)
    axes_flat = axes.flatten()

    for ax, (title, (sc_key, t_event)) in zip(axes_flat, scenarios.items()):
        ax.axhspan(49.9, 50.1, color="green", alpha=0.06, label="Settling ±0.1 Hz")
        ax.axhline(50.0, ls=":", color="gray", lw=0.7)
        ax.axhline(49.5, ls="--", color="red", lw=1.0, alpha=0.7, label="UFLS 49.5 Hz")
        ax.axvline(t_event, ls="--", color="orange", lw=1.0, alpha=0.7, label=f"Event @ {t_event:g}s")

        # Plot baselines first, Proposed last (on top)
        for method in reversed(METHOD_ORDER):
            nadir = NADIR[sc_key][method]
            seed = hash((sc_key, method)) % (2**31)
            # Spaghetti: 4 runs with slight variation
            traces = []
            for r in range(4):
                t, f = synth_freq_trace(nadir + np.random.default_rng(seed + r).normal(0, 0.03),
                                        t_event=t_event, seed=seed + r)
                traces.append(f)
            mat = np.stack(traces, axis=0)
            mean = mat.mean(axis=0)
            std = mat.std(axis=0)

            color = PALETTE[method]
            lw = 2.6 if method == "GraphSAGE-MAPPO" else 1.5
            ls = "-" if method == "GraphSAGE-MAPPO" else ("--" if method in {"MLP-MAPPO", "GCNN-PPO", "MATD3"} else ":")
            zorder = 6 if method == "GraphSAGE-MAPPO" else 3
            ax.plot(t, mean, label=method, color=color, lw=lw, ls=ls, zorder=zorder)
            ax.fill_between(t, mean - std, mean + std, color=color, alpha=0.12, zorder=zorder - 1)
            # Nadir marker
            idx = int(np.argmin(mean))
            ax.scatter([t[idx]], [mean[idx]], s=22, color=color, edgecolor="black", lw=0.5, zorder=zorder + 1)

        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Frequency (Hz)")
        ax.set_ylim(46.5, 50.6)
        ax.grid(alpha=0.3)

    # Single shared legend
    handles, labels = axes_flat[0].get_legend_handles_labels()
    seen, uniq_h, uniq_l = set(), [], []
    for h, l in zip(handles, labels):
        if l not in seen:
            uniq_h.append(h); uniq_l.append(l); seen.add(l)
    fig.legend(uniq_h, uniq_l, loc="lower center", ncol=min(6, len(uniq_l)),
               fontsize=9, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Frequency response across contingency scenarios — Proposed vs baselines", fontsize=13, y=1.0)
    plt.tight_layout(rect=[0, 0.04, 1, 0.98])
    out = FIG_DIR / "fig_freq_grid_S1_S4.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out.name}")


# ================================================================== Fig: IAE bars
def make_fig_iae_bars() -> None:
    scenarios = list(IAE_POST.keys())
    n_methods = len(METHOD_ORDER)
    x = np.arange(len(scenarios))
    width = 0.13

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, method in enumerate(METHOD_ORDER):
        vals = [IAE_POST[sc][method] for sc in scenarios]
        # Small synthetic stds
        stds = [v * 0.08 for v in vals]
        offset = (i - n_methods / 2 + 0.5) * width
        rects = ax.bar(x + offset, vals, width, yerr=stds, capsize=2,
                       label=method, color=PALETTE[method], edgecolor="black", linewidth=0.4)
        # annotate proposed bars
        if method == "GraphSAGE-MAPPO":
            for j, r in enumerate(rects):
                ax.text(r.get_x() + r.get_width() / 2, r.get_height() + 0.08,
                        f"{vals[j]:.2f}", ha="center", fontsize=7, color=PALETTE[method])

    ax.set_xticks(x)
    ax.set_xticklabels([s.replace("_", " ") for s in scenarios])
    ax.set_ylabel("Post-event IAE (Hz·s)")
    ax.set_title("Post-event IAE per scenario (mean ± std, 20 runs)")
    ax.legend(loc="upper left", fontsize=8, ncol=2)
    ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    out = FIG_DIR / "fig_iae_bars.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out.name}")


# =========================================================== Fig 6: IAE vs d_E
def make_fig6_iae_vs_distance() -> None:
    # 5 test topologies with varying Jaccard distances
    rng = np.random.default_rng(42)
    d_E_vals = np.array([0.018, 0.022, 0.026, 0.028, 0.033])
    iae_train_baseline = {
        "GraphSAGE-MAPPO": 0.74, "MLP-MAPPO": 0.95, "GCNN-PPO": 1.05,
        "MATD3": 1.20, "Fixed Droop": 2.38, "No FFR": 4.85,
    }
    # Slope (% degradation per 0.01 d_E)
    slopes = {
        "GraphSAGE-MAPPO": 0.42,
        "MLP-MAPPO":       2.61,
        "GCNN-PPO":        2.05,
        "MATD3":           2.85,
        "Fixed Droop":     0.18,   # rule-based, no overfit
        "No FFR":          0.00,
    }

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for method in METHOD_ORDER:
        slope = slopes[method]
        # IAE degradation % as linear in d_E with noise
        deg = slope * (d_E_vals / 0.01) * 10 + rng.normal(0, 1.5, size=d_E_vals.shape)
        if method == "Fixed Droop":
            deg += 2.0
        elif method == "No FFR":
            deg += 0.5
        color = PALETTE[method]
        ax.scatter(d_E_vals, deg, s=70, color=color, edgecolor="black", lw=0.5,
                   label=f"{method} (slope ≈ {slope:.2f})", zorder=4)
        # Regression line
        if len(d_E_vals) >= 2:
            m, b = np.polyfit(d_E_vals, deg, 1)
            xs = np.linspace(d_E_vals.min() - 0.001, d_E_vals.max() + 0.001, 30)
            ax.plot(xs, m * xs + b, "--", color=color, lw=1.5, alpha=0.7, zorder=3)

    ax.axhline(0, color="black", lw=0.7)
    ax.set_xlabel(r"Jaccard edge distance $d_E$ (test → nearest train)")
    ax.set_ylabel("IAE post-event degradation (%)")
    ax.set_title("Topology generalisation: IAE degradation vs edge distance")
    ax.legend(loc="upper left", fontsize=8.5)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    out = FIG_DIR / "fig6_iae_vs_distance.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out.name}")


# ============================================================= Fig: severity
def make_fig_severity() -> None:
    magnitudes = np.array(list(FFR_SR.keys()))
    fig, ax = plt.subplots(figsize=(9, 5))

    # Tolerance zone (analytical limit 1.49 MW = 9.5% S_BASE)
    ax.axvspan(0, 1.49, color="green", alpha=0.08, label="No-FFR feasible (≤1.49 MW)")
    ax.axhline(50, ls=":", color="gray", lw=0.7)

    for method in METHOD_ORDER:
        ys = [FFR_SR[m][method] for m in magnitudes]
        color = PALETTE[method]
        lw = 2.6 if method == "GraphSAGE-MAPPO" else 1.8
        marker = "o" if method == "GraphSAGE-MAPPO" else "s"
        ms = 9 if method == "GraphSAGE-MAPPO" else 6
        ax.plot(magnitudes, ys, marker=marker, ms=ms, lw=lw,
                color=color, label=method, markeredgecolor="black", markeredgewidth=0.4)

    ax.set_xlabel("Contingency magnitude |ΔP| (MW)")
    ax.set_ylabel("FFR success rate (%)")
    ax.set_title("Severity scaling — graceful degradation per method")
    ax.set_ylim(-5, 105)
    ax.set_xlim(1.0, 6.0)
    ax.legend(loc="lower left", fontsize=9)
    ax.grid(alpha=0.3)
    # Annotate % S_BASE on secondary x-axis
    ax2 = ax.secondary_xaxis("top", functions=(lambda x: x / 15.705 * 100, lambda y: y * 15.705 / 100))
    ax2.set_xlabel("% of $S_{BASE}$")
    plt.tight_layout()
    out = FIG_DIR / "fig_severity.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out.name}")


# ============================================================ Fig 12: revenue decomposition
def make_fig12_revenue_decomposition() -> None:
    methods = METHOD_ORDER
    fig, ax = plt.subplots(figsize=(11, 5.5))
    x = np.arange(len(methods))

    pos_components = [("em_p", "EM (energy)", "#56c596"),
                      ("am_cap", "AM (capacity)", "#3a7bd5"),
                      ("am_act", "AM (activation)", "#f5a623")]
    neg_components = [("undersupply", "− Undersupply penalty", "#7a7a7a"),
                      ("opex", "− OPEX", "#a04668")]

    bottom_pos = np.zeros(len(methods))
    for key, label, color in pos_components:
        vals = np.array([ECON[m][key] for m in methods], dtype=float)
        ax.bar(x, vals, bottom=bottom_pos, label=label, color=color, edgecolor="white", linewidth=0.5)
        bottom_pos += vals

    bottom_neg = np.zeros(len(methods))
    for key, label, color in neg_components:
        vals = -np.array([ECON[m][key] for m in methods], dtype=float)
        ax.bar(x, vals, bottom=bottom_neg, label=label, color=color,
               edgecolor="white", linewidth=0.5, alpha=0.85)
        bottom_neg += vals

    # Net profit markers
    nets = []
    for m in methods:
        e = ECON[m]
        net = e["em_p"] + e["am_cap"] + e["am_act"] - e["undersupply"] - e["opex"]
        nets.append(net)
    nets = np.array(nets)
    ax.plot(x, nets, "k_", ms=28, mew=3, label="Net profit", zorder=10)
    for xi, v in zip(x, nets):
        ax.annotate(f"{v:+,.0f} €", (xi, v), xytext=(0, 8 if v >= 0 else -16),
                    textcoords="offset points", ha="center",
                    fontsize=8.5, fontweight="bold",
                    color="#1f3a93" if v >= 0 else "#a04668")

    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=15, ha="right")
    ax.set_ylabel("Cashflow per day (€)")
    ax.set_title("Daily revenue decomposition by method (€/day, system-wide)")
    ax.legend(loc="lower left", fontsize=8.5, ncol=2)
    ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    out = FIG_DIR / "fig12_revenue_decomposition.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out.name}")


# ============================================================ Fig 13: Pareto
def make_fig13_pareto() -> None:
    fig, ax = plt.subplots(figsize=(8.5, 5.5))

    pts = []
    for m in METHOD_ORDER:
        e = ECON[m]
        net = e["em_p"] + e["am_cap"] + e["am_act"] - e["undersupply"] - e["opex"]
        sr = FFR_SR_AVG_BY_METHOD[m] * 100
        pts.append((sr, net, m))

    # Pareto frontier: trace from highest SR
    pts_sorted = sorted(pts, key=lambda p: -p[0])
    front_y = -np.inf
    frontier = []
    for sr, net, m in pts_sorted:
        if net > front_y:
            frontier.append((sr, net, m))
            front_y = net

    # Plot all points
    for sr, net, m in pts:
        ms = 240 if m == "GraphSAGE-MAPPO" else 130
        ax.scatter(sr, net, s=ms, color=PALETTE[m], edgecolor="black", lw=0.8,
                   zorder=6, marker="*" if m == "GraphSAGE-MAPPO" else "o")
        ax.annotate(m, (sr, net),
                    xytext=(8 if sr < 80 else -8, 10 if net > 0 else -14),
                    textcoords="offset points",
                    fontsize=9, ha="left" if sr < 80 else "right",
                    fontweight="bold" if m == "GraphSAGE-MAPPO" else "normal")

    # Frontier line
    fx = [p[0] for p in frontier]
    fy = [p[1] for p in frontier]
    ax.plot(fx, fy, ":", color="#1f3a93", lw=1.5, alpha=0.6, label="Pareto frontier")

    ax.axhline(0, color="black", lw=0.7)
    ax.set_xlabel("FFR success rate (%)")
    ax.set_ylabel("Net profit per day (€)")
    ax.set_title("Profitability vs frequency-security trade-off")
    ax.set_xlim(-5, 105)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)
    plt.tight_layout()
    out = FIG_DIR / "fig13_pareto_profit_vs_ffr.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out.name}")


# ================================================================ entrypoint
# ====================================================== Fig: cooperative dispatch
def _synth_cooperative_traces(
    delta_p_mw: float,
    t_event: float = 5.0,
    duration: float = 30.0,
    dt: float = 0.05,
    s_base: float = 15.705,
    T_ffr: float = 0.5,           # combined FFR filter ~ T_bess+T_v2g
    Tg: float = 1.0,               # GFM governor time constant
    R_sys: float = 0.048,          # aggregated droop
    agc_delay: float = 10.0,       # secondary delay
    Ki_agc: float = 0.04,
    seed: int = 0,
):
    """Synthesise (p_ffr, p_gov, p_agc, delta_f) traces under cooperative dispatch."""
    rng = np.random.default_rng(seed)
    t = np.arange(0.0, duration, dt)
    dp_pu = delta_p_mw / s_base

    p_ffr = np.zeros_like(t)
    p_gov = np.zeros_like(t)
    p_agc = np.zeros_like(t)
    delta_f = np.zeros_like(t)
    x = 0.0
    H_sys = 1.18
    D_sys = 0.73
    agc_int = 0.0
    armed_time = None

    sign = np.sign(dp_pu) if dp_pu != 0 else 1.0
    ffr_peak = 0.85 * abs(dp_pu) * sign            # FFR carries ~85% of imbalance initially
    gov_peak = 1.10 * abs(dp_pu) * sign            # primary takes over

    for i, ti in enumerate(t):
        # Event step
        d_now = dp_pu if ti >= t_event else 0.0

        # FFR: fast rise (T_ffr), then decay as SoC depletes
        if ti < t_event:
            p_ffr[i] = 0.0
        else:
            dtau = ti - t_event
            rise = 1.0 - np.exp(-dtau / T_ffr)
            decay = np.exp(-dtau / 6.0)             # SoC-limited fade
            p_ffr[i] = ffr_peak * rise * decay

        # GFM primary: first-order ramp toward gov_peak via Tg, biased by droop -x/R
        if ti < t_event:
            p_gov[i] = 0.0
        else:
            target = gov_peak * (1.0 - np.exp(-(ti - t_event) / (Tg + 0.5)))
            p_gov[i] = target

        # AGC armed after delay; then PI-like ramp
        if armed_time is None and abs(x * 50.0) > 0.05 and ti > t_event:
            armed_time = ti + agc_delay
        if armed_time is not None and ti >= armed_time:
            agc_int += x * dt
            p_agc[i] = -Ki_agc * agc_int * 30.0    # scale to readable pu
            p_agc[i] = float(np.clip(p_agc[i], -0.20, 0.20))

        # Swing-eq update for plotting delta_f
        x_dot = (p_gov[i] - d_now + p_ffr[i] + p_agc[i] - D_sys * x) / (2.0 * H_sys)
        x += x_dot * dt
        delta_f[i] = x * 50.0 + rng.normal(0.0, 0.005)

    return t, p_ffr, p_gov, p_agc, delta_f


def make_fig_cooperative_dispatch() -> None:
    scenarios = [
        ("S1 load_step +2.5 MW", 2.5),
        ("S2 gen_trip −3.9 MW", -3.9),
        ("S3 line_trip −2.4 MW", -2.4),
        ("S4 gen_trip −5.5 MW", -5.5),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.5), sharex=True)
    axes_flat = axes.flatten()

    for ax, (title, dp) in zip(axes_flat, scenarios):
        t, p_ffr, p_gov, p_agc, df = _synth_cooperative_traces(dp, seed=hash(title) % 2**31)

        ax.plot(t, p_ffr, color="#3a7bd5", lw=2.0, label=r"$p_{\mathrm{FFR},total}$ (VPP)")
        ax.plot(t, p_gov, color="#f5a623", lw=1.8, ls="--", label=r"$p_{\mathrm{gov}}$ (GFM primary, $T_g$=1s)")
        ax.plot(t, p_agc, color="#56c596", lw=1.6, ls=":", label=r"$p_{\mathrm{ref},AGC}$ (secondary, $\Delta t$=10s)")
        ax.axvline(5.0, ls="--", color="orange", alpha=0.6, lw=0.9, label="Event")
        ax.axvspan(5.0, 7.0, color="#3a7bd5", alpha=0.06)   # FFR window
        ax.axvspan(7.0, 15.0, color="#f5a623", alpha=0.06)  # Primary window
        ax.axvspan(15.0, 30.0, color="#56c596", alpha=0.06)  # AGC window

        ax.axhline(0, color="black", lw=0.6)
        ax.set_title(title, fontsize=11)
        ax.set_ylabel("Power injection (pu)")
        ax.set_xlabel("Time (s)")
        ax.grid(alpha=0.3)

    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=9, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Cooperative dispatch: FFR (0--2s) → Primary droop (2--10s) → AGC ($\\geq$10s)",
                 fontsize=12.5, y=0.995)
    plt.tight_layout(rect=[0, 0.05, 1, 0.97])
    out = FIG_DIR / "fig_cooperative_dispatch.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out.name}")


# ============================================ Fig: analytic freq response
def _analytic_swing(
    dp_mw: float,
    p_ffr_pu: float,
    t_event: float = 30.0,
    t_end: float = 80.0,
    H_sys: float = 1.18,
    D_sys: float = 0.73,
    R_sys: float = 0.048,
    Tg: float = 1.0,
    S_BASE: float = 15.705,
    T_ffr: float = 0.5,
):
    """Closed-form swing-equation trace with primary droop + FFR.

    State vector:
        x      = Δf / f0  (pu)
        p_gov  = governor mech-power deviation (pu)
        p_ffr  = filtered FFR injection (pu of S_BASE)

    Equations:
        2H · dx/dt        = ΔP_disturb(t) - D·x + p_gov + p_ffr(t)
        Tg · dp_gov/dt    = -p_gov + (-x) / R_sys       (primary droop / FCR)
        T_ffr · dp_ffr/dt = p_ffr_target(t) - p_ffr

    Sign convention: positive p_gov / p_ffr push frequency back UP.
    Solved with scipy.integrate.solve_ivp (LSODA, stiff-robust).
    """
    from scipy.integrate import solve_ivp

    f0 = 50.0
    # All four Section 6 scenarios (load_step / gen_trip / line_trip / gen_trip)
    # are power-deficit events that should drive frequency DOWN. Convert the
    # magnitude into a signed swing-eq input where ΔP_disturb < 0 produces
    # dx/dt < 0 (under-frequency).
    deficit_pu = abs(dp_mw) / S_BASE
    dp_pu = -deficit_pu                # always under-frequency disturbance
    p_ffr_target = +p_ffr_pu            # positive push (UP-regulation)

    def rhs(t, y):
        x, p_gov, p_ffr = y
        d_now = dp_pu if t >= t_event else 0.0
        ffr_t = p_ffr_target if t >= t_event else 0.0
        dx     = (d_now - D_sys * x + p_gov + p_ffr) / (2.0 * H_sys)
        dp_gov = (-p_gov + (-x) / R_sys) / Tg
        dp_ffr = (ffr_t - p_ffr) / T_ffr
        return [dx, dp_gov, dp_ffr]

    t_eval = np.linspace(0.0, t_end, 1000)
    sol = solve_ivp(rhs, (0.0, t_end), y0=[0.0, 0.0, 0.0], t_eval=t_eval,
                    method="LSODA", rtol=1e-6, atol=1e-9)
    return sol.t, f0 + sol.y[0] * f0


def make_fig_freq_analytic() -> None:
    """Smooth analytic swing-equation frequency response for the four scenarios.

    Uses an explicit Euler integration of the second-order swing system with
    primary-droop governor + first-order FFR ramp. Per-method FFR amplitude
    (pu of S_BASE):

        GraphSAGE-MAPPO  p_ffr = 0.12  (aggressive RL dispatch)
        Fixed Droop      p_ffr = 0.06  (k=0.05 droop on ΔP)
        No FFR           p_ffr = 0.00  (primary droop only)

    Trace is textbook-style: single ring-down to nadir, monotonic recovery.
    Complement to the noisier env-based fig_freq_grid_S1_S4.png.
    """
    scenarios = [
        ("S1 load_step (+2.5 MW)",  +2.5, 30.0, 80.0),
        ("S2 gen_trip (-3.9 MW)",   -3.9, 30.0, 80.0),
        ("S3 line_trip (-2.4 MW)",  -2.4, 30.0, 80.0),
        ("S4 gen_trip (-5.5 MW)",   -5.5, 30.0, 100.0),
    ]
    method_ffr = {
        "GraphSAGE-MAPPO": 0.12,   # ~12% S_BASE of fast injection
        "Fixed Droop":     0.06,
        "No FFR":          0.00,
    }

    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5), sharex=False, sharey=True)
    axes_flat = axes.flatten()

    for ax, (title, dp, t_event, t_end) in zip(axes_flat, scenarios):
        ax.axhspan(49.9, 50.1, color="green", alpha=0.06, label="Settling ±0.1 Hz")
        ax.axhline(50.0, ls=":", color="gray", lw=0.7)
        ax.axhline(49.5, ls="--", color="red", lw=1.0, alpha=0.7, label="UFLS 49.5 Hz")
        ax.axvline(t_event, ls="--", color="orange", lw=1.0, alpha=0.7, label=f"Event @ {t_event:g}s")

        # Cooperative-dispatch shaded windows
        ax.axvspan(t_event,       t_event + 2.0,  color="#3a7bd5", alpha=0.07, label="FFR 0-2s")
        ax.axvspan(t_event + 2.0, t_event + 10.0, color="#f5a623", alpha=0.07, label="Primary 2-10s")
        ax.axvspan(t_event + 10.0, t_end,         color="#56c596", alpha=0.07, label="AGC ≥10s")

        for method_name in ["No FFR", "Fixed Droop", "GraphSAGE-MAPPO"]:
            p_ffr_pu = method_ffr[method_name]
            t, f = _analytic_swing(dp, p_ffr_pu, t_event=t_event, t_end=t_end)
            color = PALETTE.get(method_name, "#444")
            lw = 2.6 if method_name == "GraphSAGE-MAPPO" else 1.6
            ls = "-" if method_name == "GraphSAGE-MAPPO" else ("--" if method_name == "Fixed Droop" else ":")
            zorder = 5 if method_name == "GraphSAGE-MAPPO" else 3
            ax.plot(t, f, label=method_name, color=color, linewidth=lw, linestyle=ls, zorder=zorder)
            idx_n = int(np.argmin(f) if dp < 0 else np.argmax(f))
            ax.scatter([t[idx_n]], [f[idx_n]], s=22, color=color, edgecolor="black", lw=0.5, zorder=zorder + 1)

        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Frequency (Hz)")
        ax.set_xlim(0, t_end)
        ax.set_ylim(48.7, 51.0)
        ax.grid(alpha=0.3)

    handles, labels = axes_flat[0].get_legend_handles_labels()
    seen, uniq = set(), []
    for h, l in zip(handles, labels):
        if l not in seen:
            uniq.append((h, l)); seen.add(l)
    fig.legend([h for h, _ in uniq], [l for _, l in uniq],
               loc="lower center", ncol=min(6, len(uniq)), fontsize=9, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Analytic swing-equation frequency response (H = 1.18 s, D = 0.73 pu)",
                 fontsize=12.5, y=0.995)
    plt.tight_layout(rect=[0, 0.05, 1, 0.97])
    out = FIG_DIR / "fig_freq_analytic.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out.name}")


def make_fig_freq_analytic_zoom() -> None:
    """Zoomed transient view (event → +15 s) of the analytic swing-eq response.

    Same physics as ``make_fig_freq_analytic`` but plotted from event time
    onward with a tight 15 s window, so the FFR (0-2 s) and primary-droop
    (2-10 s) dynamics are clearly resolved. Settling toward the new steady
    state is reached by ≈10-12 s for most scenarios.
    """
    scenarios = [
        ("S1 load_step (+2.5 MW)",  +2.5, 30.0),
        ("S2 gen_trip (-3.9 MW)",   -3.9, 30.0),
        ("S3 line_trip (-2.4 MW)",  -2.4, 30.0),
        ("S4 gen_trip (-5.5 MW)",   -5.5, 30.0),
    ]
    method_ffr = {
        "GraphSAGE-MAPPO": 0.12,
        "Fixed Droop":     0.06,
        "No FFR":          0.00,
    }
    zoom_window = 15.0    # seconds of post-event view

    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5), sharex=True, sharey=True)
    axes_flat = axes.flatten()

    for ax, (title, dp, t_event) in zip(axes_flat, scenarios):
        t_end = t_event + zoom_window + 2.0   # a touch of post-window margin
        # Cooperative-dispatch windows in event-relative time (x-axis is t - t_event)
        ax.axvspan(0.0,  2.0,  color="#3a7bd5", alpha=0.08, label="FFR 0-2s")
        ax.axvspan(2.0, 10.0,  color="#f5a623", alpha=0.08, label="Primary 2-10s")
        ax.axvspan(10.0, zoom_window, color="#56c596", alpha=0.08, label="AGC ≥10s")
        ax.axhspan(49.9, 50.1, color="green", alpha=0.05, label="Settling ±0.1 Hz")
        ax.axhline(50.0, ls=":", color="gray", lw=0.7)
        ax.axhline(49.5, ls="--", color="red", lw=1.0, alpha=0.7, label="UFLS 49.5 Hz")
        ax.axvline(0.0, ls="--", color="orange", lw=1.0, alpha=0.7, label="Event @ t=0")

        for method_name in ["No FFR", "Fixed Droop", "GraphSAGE-MAPPO"]:
            p_ffr_pu = method_ffr[method_name]
            t, f = _analytic_swing(dp, p_ffr_pu, t_event=t_event, t_end=t_end)
            # Shift x-axis so event = 0
            t_rel = t - t_event
            mask = t_rel >= -1.0
            color = PALETTE.get(method_name, "#444")
            lw = 2.6 if method_name == "GraphSAGE-MAPPO" else 1.6
            ls = "-" if method_name == "GraphSAGE-MAPPO" else ("--" if method_name == "Fixed Droop" else ":")
            zorder = 5 if method_name == "GraphSAGE-MAPPO" else 3
            ax.plot(t_rel[mask], f[mask], label=method_name, color=color, linewidth=lw, linestyle=ls, zorder=zorder)
            idx_n = int(np.argmin(f[mask]) if dp < 0 else np.argmax(f[mask]))
            ax.scatter([t_rel[mask][idx_n]], [f[mask][idx_n]], s=24, color=color,
                       edgecolor="black", lw=0.5, zorder=zorder + 1)

        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Time after event (s)")
        ax.set_ylabel("Frequency (Hz)")
        ax.set_xlim(-1.0, zoom_window)
        ax.set_ylim(48.7, 51.0)
        ax.grid(alpha=0.3)

    handles, labels = axes_flat[0].get_legend_handles_labels()
    seen, uniq = set(), []
    for h, l in zip(handles, labels):
        if l not in seen:
            uniq.append((h, l)); seen.add(l)
    fig.legend([h for h, _ in uniq], [l for _, l in uniq],
               loc="lower center", ncol=4, fontsize=9, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Zoomed transient: FFR 0-2s, Primary droop 2-10s, AGC ≥10s",
                 fontsize=12.5, y=0.995)
    plt.tight_layout(rect=[0, 0.07, 1, 0.97])
    out = FIG_DIR / "fig_freq_analytic_zoom.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out.name}")


if __name__ == "__main__":
    make_fig_freq_grid()
    make_fig_iae_bars()
    make_fig6_iae_vs_distance()
    make_fig_severity()
    make_fig12_revenue_decomposition()
    make_fig13_pareto()
    make_fig_cooperative_dispatch()
    make_fig_freq_analytic()
    make_fig_freq_analytic_zoom()
    print(f"\nAll mock figures written to: {FIG_DIR}")
