"""
scripts/plot_all.py
===================
Master plot script — generates all figures for the IEEE TSG paper.
Run AFTER evaluation CSVs are produced by comparison_runner.py.

Usage:
    python scripts/plot_all.py --results-dir results/ --out-dir figures/

Produces:
    fig1_frequency_metrics.{png,pdf}   — Table I grouped bar (4 metrics * 3 scenarios)
    fig2_iae_heatmap.{png,pdf}         — IAE improvement % heatmap
    fig3_economic.{png,pdf}            — Table II P2P + r_slow
    fig4_generalization.{png,pdf}      — Table III topology generalization
    fig5_thd_v_per_bus.{png,pdf}       — THD_V per bus (all 123)
    fig6_thd_i_per_branch.{png,pdf}    — THD_I per branch
    fig7_training_curves.{png,pdf}     — Reward curves A→F phases
"""

import argparse
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

# ── IEEE-style rcParams ────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family':      'serif',
    'font.serif':       ['Times New Roman', 'DejaVu Serif'],
    'font.size':         8,
    'axes.titlesize':    9,
    'axes.labelsize':    8,
    'xtick.labelsize':   7,
    'ytick.labelsize':   7,
    'legend.fontsize':   7,
    'figure.dpi':        300,
    'axes.linewidth':    0.8,
    'grid.linewidth':    0.4,
    'lines.linewidth':   1.2,
    'savefig.dpi':       300,
    'savefig.bbox':      'tight',
})

# ── Color palette (consistent across all figures) ─────────────────────────
COLORS = {
    'GNN-MAPPO (Proposed)': '#D62728',
    'GCNN-PPO':             '#1F77B4',
    'AGCN-Decentralized':   '#FF7F0E',
    'Graph-PPO':            '#2CA02C',
    'Fixed Droop':          '#9467BD',
    'No FFR':               '#7F7F7F',
}
METHOD_ORDER = [
    'GNN-MAPPO (Proposed)', 'GCNN-PPO', 'AGCN-Decentralized',
    'Graph-PPO', 'Fixed Droop', 'No FFR',
]
SCENARIO_LABELS = {
    'S1_load_step': 'S1: Load Step',
    'S2_gen_trip':  'S2: Gen. Trip',
    'S3_line_trip': 'S3: Line Trip',
}
SCENARIOS = ['S1_load_step', 'S2_gen_trip', 'S3_line_trip']
IEEE519_LIMIT = 5.0


def _save(fig, outdir, name):
    os.makedirs(outdir, exist_ok=True)
    for ext in ('png', 'pdf'):
        path = os.path.join(outdir, f'{name}.{ext}')
        fig.savefig(path, format=ext)
    print(f'  Saved {name}.{{png,pdf}}')
    plt.close(fig)


def _legend_handles(methods=None):
    methods = methods or METHOD_ORDER
    return [
        mpatches.Patch(facecolor=COLORS.get(m, '#333333'), label=m,
                       edgecolor='grey', linewidth=0.4)
        for m in methods
    ]


# ══════════════════════════════════════════════════════════════════════════════
#  FIG 1 — Frequency metrics grouped bar  (Table I)
# ══════════════════════════════════════════════════════════════════════════════
def plot_frequency_metrics(t1: pd.DataFrame, outdir: str):
    metrics = ['IAE', 'delta_f_max', 'rocof_max', 'time_violation']
    labels  = {
        'IAE':            'IAE (Hz·s)',
        'delta_f_max':    r'$\Delta f_{\max}$ (Hz)',
        'rocof_max':      r'RoCoF$_{\max}$ (Hz/s)',
        'time_violation': 'Time violation (s)',
    }

    fig, axes = plt.subplots(2, 2, figsize=(7.16, 5.5))
    axes = axes.flatten()

    n_m = len(METHOD_ORDER)
    n_s = len(SCENARIOS)
    w   = 0.12
    x   = np.arange(n_s)

    for ai, metric in enumerate(metrics):
        ax  = axes[ai]
        sub = t1[t1['metric'] == metric]

        for mi, method in enumerate(METHOD_ORDER):
            vals, errs = [], []
            for sc in SCENARIOS:
                row = sub[(sub['scenario'] == sc) & (sub['method'] == method)]
                vals.append(row['mean'].values[0] if len(row) else 0.0)
                errs.append(row['std'].values[0]  if len(row) else 0.0)

            offset = (mi - (n_m - 1) / 2) * w
            bars = ax.bar(x + offset, vals, w,
                          color=COLORS.get(method, '#999'),
                          alpha=0.85, edgecolor='white', linewidth=0.3,
                          yerr=errs, capsize=1.5,
                          error_kw={'linewidth': 0.5})
            if method == 'GNN-MAPPO (Proposed)':
                for b in bars:
                    b.set_edgecolor('#8B0000')
                    b.set_linewidth(1.1)

        ax.set_xticks(x)
        ax.set_xticklabels([SCENARIO_LABELS[s] for s in SCENARIOS])
        ax.set_ylabel(labels[metric])
        ax.set_title(f'({chr(97+ai)}) {labels[metric]}',
                     pad=3, fontweight='bold')
        ax.yaxis.grid(True, linestyle='--', alpha=0.45)
        ax.set_axisbelow(True)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    fig.legend(handles=_legend_handles(), loc='lower center', ncol=3,
               bbox_to_anchor=(0.5, -0.02), frameon=True,
               framealpha=0.9, edgecolor='lightgrey')
    fig.suptitle(
        'Table I: Frequency Response — All Methods × All Scenarios',
        fontsize=9, fontweight='bold', y=1.01)
    fig.tight_layout(rect=[0, 0.07, 1, 1])
    _save(fig, outdir, 'fig1_frequency_metrics')


# ══════════════════════════════════════════════════════════════════════════════
#  FIG 2 — IAE improvement heatmap
# ══════════════════════════════════════════════════════════════════════════════
def plot_iae_heatmap(t1: pd.DataFrame, outdir: str):
    iae_sub   = t1[t1['metric'] == 'IAE']
    baselines = [m for m in METHOD_ORDER if m != 'GNN-MAPPO (Proposed)']

    matrix = []
    for sc in SCENARIOS:
        prop = iae_sub[(iae_sub['scenario'] == sc) &
                       (iae_sub['method'] == 'GNN-MAPPO (Proposed)')]['mean']
        prop_val = prop.values[0] if len(prop) else np.nan
        row = []
        for bl in baselines:
            r = iae_sub[(iae_sub['scenario'] == sc) & (iae_sub['method'] == bl)]
            bl_val = r['mean'].values[0] if len(r) else np.nan
            row.append((bl_val - prop_val) / bl_val * 100
                       if (not np.isnan(bl_val) and bl_val > 0) else 0.0)
        matrix.append(row)

    matrix = np.array(matrix)
    fig, ax = plt.subplots(figsize=(5.5, 2.8))
    im = ax.imshow(matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=100)

    bl_labels = ['GCNN-PPO', 'AGCN', 'Graph-PPO', 'Fixed\nDroop', 'No FFR']
    ax.set_xticks(range(len(baselines)))
    ax.set_xticklabels(bl_labels, fontsize=7)
    ax.set_yticks(range(3))
    ax.set_yticklabels([SCENARIO_LABELS[s] for s in SCENARIOS])

    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label('IAE Improvement (%)', fontsize=7)
    cb.ax.tick_params(labelsize=6)

    for i in range(len(SCENARIOS)):
        for j in range(len(baselines)):
            v = matrix[i, j]
            ax.text(j, i, f'{v:.1f}%', ha='center', va='center',
                    fontsize=7, fontweight='bold',
                    color='white' if v > 65 else 'black')

    ax.set_title('IAE Improvement of GNN-MAPPO Over Baselines (%)',
                 pad=5, fontsize=9, fontweight='bold')
    fig.tight_layout()
    _save(fig, outdir, 'fig2_iae_heatmap')


# ══════════════════════════════════════════════════════════════════════════════
#  FIG 3 — Economic metrics  (Table II)
# ══════════════════════════════════════════════════════════════════════════════
def plot_economic(t2: pd.DataFrame, outdir: str):
    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.5))
    methods  = t2['method'].tolist()
    colors_t = [COLORS.get(m, '#999') for m in methods]
    short    = ['Proposed', 'GCNN-PPO', 'AGCN', 'Graph-PPO', 'No FFR', 'Fixed\nDroop']
    short    = short[:len(methods)]

    def _bar(ax, col_mean, col_std, title, ylabel):
        vals = t2[col_mean].values
        errs = t2[col_std].values if col_std in t2.columns else np.zeros(len(vals))
        bars = ax.bar(range(len(methods)), vals, color=colors_t,
                      yerr=errs, capsize=3, edgecolor='white', linewidth=0.4,
                      error_kw={'linewidth': 0.5})
        # Bold outline for proposed
        prop_idx = methods.index('GNN-MAPPO (Proposed)') if 'GNN-MAPPO (Proposed)' in methods else 0
        bars[prop_idx].set_edgecolor('#8B0000')
        bars[prop_idx].set_linewidth(1.4)
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels(short, fontsize=6.5, rotation=15, ha='right')
        ax.set_ylabel(ylabel)
        ax.set_title(title, pad=3, fontweight='bold')
        ax.yaxis.grid(True, linestyle='--', alpha=0.45)
        ax.set_axisbelow(True)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    _bar(axes[0], 'P2P_mean', 'P2P_std',
         '(a) P2P Revenue', 'P2P Revenue ($/interval)')
    _bar(axes[1], 'r_slow_mean', 'r_slow_std',
         '(b) Slow-loop Reward', 'Slow-loop reward')

    fig.suptitle('Table II: Voltage & Economic Performance',
                 fontsize=9, fontweight='bold')
    fig.tight_layout()
    _save(fig, outdir, 'fig3_economic')


# ══════════════════════════════════════════════════════════════════════════════
#  FIG 4 — Topology generalization  (Table III)
# ══════════════════════════════════════════════════════════════════════════════
def plot_generalization(t3: pd.DataFrame, outdir: str):
    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.6))
    methods = t3['method'].tolist()
    colors  = [COLORS.get(m, '#999') for m in methods]
    short   = ['Proposed', 'GCNN-PPO', 'Graph-PPO'][:len(methods)]
    x = np.arange(len(methods))
    w = 0.35

    # (a) Train vs Test IAE
    ax = axes[0]
    b1 = ax.bar(x - w/2, t3['IAE_train'].values, w,
                color=colors, edgecolor='white', linewidth=0.4, label='Train')
    b2 = ax.bar(x + w/2, t3['IAE_test'].values, w,
                color=colors, alpha=0.5, edgecolor='black', linewidth=0.6,
                hatch='//', label='Test (unseen)')
    ax.set_xticks(x)
    ax.set_xticklabels(short)
    ax.set_ylabel('IAE (Hz·s)')
    ax.set_title('(a) Train vs. Test IAE', pad=3, fontweight='bold')
    ax.legend(fontsize=6, framealpha=0.85)
    ax.yaxis.grid(True, linestyle='--', alpha=0.45)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # (b) Generalization gap %
    ax2 = axes[1]
    drops = t3['drop_pct'].values
    bar_colors = ['#2CA02C' if d <= 0 else '#D62728' for d in drops]
    bars = ax2.bar(x, drops, color=bar_colors, edgecolor='white', linewidth=0.4)
    ax2.axhline(0, color='black', linewidth=0.8)
    ax2.set_xticks(x)
    ax2.set_xticklabels(short)
    ax2.set_ylabel('Performance Change (%)')
    ax2.set_title('(b) Generalization Gap (%)', pad=3, fontweight='bold')
    for b, v in zip(bars, drops):
        ax2.text(b.get_x() + b.get_width()/2,
                 v + (0.06 if v >= 0 else -0.2),
                 f'{v:+.2f}%', ha='center', va='bottom', fontsize=6.5)
    green_p = mpatches.Patch(color='#2CA02C', label='Improves on unseen')
    red_p   = mpatches.Patch(color='#D62728', label='Degrades on unseen')
    ax2.legend(handles=[green_p, red_p], fontsize=6, framealpha=0.85)
    ax2.yaxis.grid(True, linestyle='--', alpha=0.45)
    ax2.set_axisbelow(True)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    fig.suptitle('Table III: Topology Generalization (GraphSAGE Inductive)',
                 fontsize=9, fontweight='bold')
    fig.tight_layout()
    _save(fig, outdir, 'fig4_generalization')


# ══════════════════════════════════════════════════════════════════════════════
#  FIG 5 — THD_V per bus
# ══════════════════════════════════════════════════════════════════════════════
def plot_thd_v_per_bus(thd_v_data: dict, outdir: str):
    """
    thd_v_data: {method: np.ndarray (n_bus,)}
    """
    methods  = [m for m in METHOD_ORDER if m in thd_v_data]
    n_m      = len(methods)
    n_bus    = len(next(iter(thd_v_data.values())))
    x        = np.arange(n_bus)
    w        = 0.8 / n_m

    fig, ax = plt.subplots(figsize=(7.16, 3.5))

    for mi, method in enumerate(methods):
        thd = thd_v_data[method]
        offset = (mi - (n_m - 1) / 2) * w
        ax.bar(x + offset, thd, w * 0.92,
               color=COLORS.get(method, '#999'),
               alpha=0.78, label=method)

    ax.axhline(IEEE519_LIMIT, color='red', linewidth=1.2,
               linestyle='--', label='IEEE 519 limit (5%)', zorder=5)

    ax.set_xlabel('Bus index')
    ax.set_ylabel('THD$_V$ (%)')
    ax.set_title('Voltage THD at All Buses — Method Comparison',
                 fontsize=9, fontweight='bold', pad=4)
    ax.set_xlim(-1, n_bus)
    ax.set_xticks(np.arange(0, n_bus, 10))
    ax.yaxis.grid(True, linestyle='--', alpha=0.4)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(fontsize=6, ncol=4, loc='upper right',
              framealpha=0.9, edgecolor='lightgrey')

    fig.tight_layout()
    _save(fig, outdir, 'fig5_thd_v_per_bus')


# ══════════════════════════════════════════════════════════════════════════════
#  FIG 6 — THD_I per branch
# ══════════════════════════════════════════════════════════════════════════════
def plot_thd_i_per_branch(thd_i_data: dict, outdir: str):
    """
    thd_i_data: {method: np.ndarray (n_branch,)}
    """
    methods  = [m for m in METHOD_ORDER if m in thd_i_data]
    n_m      = len(methods)
    n_branch = len(next(iter(thd_i_data.values())))
    x        = np.arange(n_branch)
    w        = 0.8 / n_m

    fig, ax = plt.subplots(figsize=(7.16, 3.5))

    for mi, method in enumerate(methods):
        thd = thd_i_data[method]
        offset = (mi - (n_m - 1) / 2) * w
        ax.bar(x + offset, thd, w * 0.92,
               color=COLORS.get(method, '#999'),
               alpha=0.78, label=method)

    ax.axhline(IEEE519_LIMIT, color='red', linewidth=1.2,
               linestyle='--', label='IEEE 519 limit (5%)', zorder=5)

    ax.set_xlabel('Branch index')
    ax.set_ylabel('THD$_I$ (%)')
    ax.set_title('Current THD at All Branches — Method Comparison',
                 fontsize=9, fontweight='bold', pad=4)
    ax.set_xlim(-1, n_branch)
    ax.set_xticks(np.arange(0, n_branch, 10))
    ax.yaxis.grid(True, linestyle='--', alpha=0.4)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(fontsize=6, ncol=4, loc='upper right',
              framealpha=0.9, edgecolor='lightgrey')

    fig.tight_layout()
    _save(fig, outdir, 'fig6_thd_i_per_branch')


# ══════════════════════════════════════════════════════════════════════════════
#  FIG 7 — Training curves A→F
# ══════════════════════════════════════════════════════════════════════════════
def plot_training_curves(log_path: str, outdir: str):
    """
    Reads training log file.
    Log format (one line per log_interval):
        phase=A ep=50 r_fast=-0.459 r_slow=0.577 entropy_f=1.053 loss_f=22.1 loss_s=9.4
    """
    import re

    if not os.path.exists(log_path):
        print(f'  [SKIP] Training log not found: {log_path}')
        return

    records = []
    pattern = re.compile(
        r'phase=(\S+)\s+ep=(\d+)\s+r_fast=([-\d.]+)\s+r_slow=([-\d.]+)'
        r'(?:\s+entropy_f=([\d.]+))?(?:\s+loss_f=([\d.]+))?(?:\s+loss_s=([\d.]+))?'
    )
    with open(log_path) as f:
        for line in f:
            m = pattern.search(line)
            if m:
                records.append({
                    'phase':     m.group(1),
                    'ep':        int(m.group(2)),
                    'r_fast':    float(m.group(3)),
                    'r_slow':    float(m.group(4)),
                    'entropy_f': float(m.group(5)) if m.group(5) else np.nan,
                    'loss_f':    float(m.group(6)) if m.group(6) else np.nan,
                })

    if not records:
        print('  [SKIP] No matching log entries found')
        return

    df = pd.DataFrame(records)

    # Build global episode counter across phases
    phase_order = ['A', 'B', 'C1', 'C2', 'D', 'E', 'F']
    df['phase_idx'] = df['phase'].apply(
        lambda p: phase_order.index(p) if p in phase_order else len(phase_order))
    df = df.sort_values(['phase_idx', 'ep']).reset_index(drop=True)
    df['global_ep'] = range(len(df))

    # Phase boundary x positions
    boundaries = {}
    for ph in df['phase'].unique():
        first_idx = df[df['phase'] == ph]['global_ep'].iloc[0]
        boundaries[ph] = first_idx

    # Phase colors
    phase_colors = {
        'A': '#AEC6E8', 'B': '#FFBB78', 'C1': '#98DF8A',
        'C2': '#FF9896', 'D': '#C5B0D5', 'E': '#C49C94', 'F': '#F7B6D2',
    }

    fig, axes = plt.subplots(2, 1, figsize=(7.16, 4.5), sharex=True)

    for ax, col, ylabel, title in [
        (axes[0], 'r_fast', 'Fast-loop reward', '(a) Fast-loop reward (frequency)'),
        (axes[1], 'r_slow', 'Slow-loop reward', '(b) Slow-loop reward (voltage + P2P)'),
    ]:
        # Shade phases
        for ph, x_start in sorted(boundaries.items(),
                                   key=lambda kv: phase_order.index(kv[0])
                                   if kv[0] in phase_order else 99):
            ph_data = df[df['phase'] == ph]
            if len(ph_data) == 0:
                continue
            x_end = ph_data['global_ep'].iloc[-1]
            ax.axvspan(x_start, x_end, alpha=0.12,
                       color=phase_colors.get(ph, '#DDDDDD'))
            ax.text((x_start + x_end) / 2,
                    ax.get_ylim()[1] if ax.get_ylim()[1] != ax.get_ylim()[0]
                    else df[col].max() * 0.98,
                    ph, ha='center', va='top', fontsize=7,
                    color='#444444', fontweight='bold')

        # Smooth with rolling mean (window=5)
        vals = df[col].values
        smoothed = pd.Series(vals).rolling(window=5, min_periods=1).mean().values

        ax.plot(df['global_ep'], vals, color='#BBBBBB',
                linewidth=0.5, alpha=0.6)
        ax.plot(df['global_ep'], smoothed, color=COLORS['GNN-MAPPO (Proposed)'],
                linewidth=1.4, label='GNN-MAPPO (smoothed)')
        ax.axhline(0, color='black', linewidth=0.6, linestyle='--', alpha=0.4)

        ax.set_ylabel(ylabel)
        ax.set_title(title, pad=3, fontweight='bold')
        ax.yaxis.grid(True, linestyle='--', alpha=0.4)
        ax.set_axisbelow(True)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    axes[1].set_xlabel('Training episode (cumulative)')
    fig.suptitle('Training Curves — GNN-MAPPO Curriculum (Phases A→F)',
                 fontsize=9, fontweight='bold')
    fig.tight_layout()
    _save(fig, outdir, 'fig7_training_curves')


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description='Generate all IEEE TSG paper figures')
    parser.add_argument('--results-dir', default='results/',
                        help='Directory with table1/2/3.csv and thd_results.npz')
    parser.add_argument('--log',         default='logs/train_dual_v4_s42.txt',
                        help='Training log file for fig7')
    parser.add_argument('--out-dir',     default='figures/',
                        help='Output directory for figures')
    parser.add_argument('--skip-thd',    action='store_true',
                        help='Skip THD figures (if thd_results.npz not ready)')
    parser.add_argument('--skip-train',  action='store_true',
                        help='Skip training curve (if log not ready)')
    args = parser.parse_args()

    rd  = args.results_dir
    out = args.out_dir
    os.makedirs(out, exist_ok=True)

    # ── Load CSVs ─────────────────────────────────────────────────────────
    def _load(name):
        path = os.path.join(rd, name)
        if not os.path.exists(path):
            print(f'  [SKIP] {name} not found')
            return None
        return pd.read_csv(path)

    t1 = _load('table1.csv')
    t2 = _load('table2.csv')
    t3 = _load('table3.csv')

    print('Generating figures...')

    if t1 is not None:
        print('fig1: frequency metrics')
        plot_frequency_metrics(t1, out)
        print('fig2: IAE heatmap')
        plot_iae_heatmap(t1, out)

    if t2 is not None:
        print('fig3: economic')
        plot_economic(t2, out)

    if t3 is not None:
        print('fig4: generalization')
        plot_generalization(t3, out)

    # ── THD figures ───────────────────────────────────────────────────────
    if not args.skip_thd:
        thd_path = os.path.join(rd, 'thd_results.npz')
        if os.path.exists(thd_path):
            print('fig5: THD_V per bus')
            data = np.load(thd_path, allow_pickle=True)
            thd_v_raw = data['thd_v_per_method']
            thd_i_raw = data['thd_i_per_method']
            thd_v = thd_v_raw.item() if not isinstance(thd_v_raw, dict) else thd_v_raw
            thd_i = thd_i_raw.item() if not isinstance(thd_i_raw, dict) else thd_i_raw
            plot_thd_v_per_bus(thd_v, out)
            print('fig6: THD_I per branch')
            plot_thd_i_per_branch(thd_i, out)
        else:
            print('  [SKIP] thd_results.npz not found — run comparison_runner first')

    # ── Training curves ───────────────────────────────────────────────────
    if not args.skip_train:
        print('fig7: training curves')
        plot_training_curves(args.log, out)

    print(f'\nDone. All figures saved to: {out}')


if __name__ == '__main__':
    main()