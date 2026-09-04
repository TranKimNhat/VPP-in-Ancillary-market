"""Aggregate eval outputs into paper-section-organized layout.

Source layout (legacy, from eval_ffr_topology.py / eval_economics.py / eval_thd.py):
  paper/figures_real/table1_ffr_comparison.csv
  paper/figures_real/table2_topology_adaptation.csv
  paper/figures_real/table3_severity_scaling.csv
  paper/figures_real/table10_method_economics.csv
  paper/figures_real/fig_freq_grid_S1_S4.png
  paper/figures_real/fig6_iae_vs_distance.png
  paper/figures_real/fig13_pareto_profit_vs_ffr.png
  results/thd_verify/thd_per_method.csv
  results/eval_full/thd_results.npz

Target layout (one folder per paper section):
  results/
    section1_stability/{tab_ffr_main.csv, tab_severity.csv, fig_freq_grid.png}
    section2_topology/{tab_topo_train_test.csv, tab_encoder_ablation.csv, fig_iae_vs_distance.png}
    section3_economic/{tab_economic_methods.csv, tab_cost_effectiveness.csv, fig_pareto.png}
    section4_harmonic/{tab_thd_compliance.csv, fig_thd_bus_heatmap.png}

Derived artefacts (built fresh, not copied):
  - tab_cost_effectiveness.csv  : net_profit / (1 - ffr_sr) per method (joins table1 + table10)
  - fig_thd_bus_heatmap.png     : per-bus THD_V heatmap (method x bus) from thd_results.npz

Obsolete artefacts removed by --clean:
  paper/figures/fig_freq_analytic.png
  paper/figures/fig_freq_analytic_zoom.png
  paper/figures/fig_iae_bars.png
  paper/figures_real/fig12_revenue_decomposition.{pdf,png}
  paper/figures_real/table9_revenue_breakdown.csv

Usage:
  python scripts/eval_summary.py                       # aggregate only
  python scripts/eval_summary.py --clean               # aggregate + delete obsolete
  python scripts/eval_summary.py --src paper/figures_real --dst results
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent


SECTION_MAP: dict[str, list[tuple[str, str]]] = {
    "section1_stability": [
        ("table1_ffr_comparison.csv", "tab_ffr_main.csv"),
        ("table3_severity_scaling.csv", "tab_severity.csv"),
        ("fig_freq_grid_S1_S4.png", "fig_freq_grid.png"),
    ],
    "section2_topology": [
        ("table2_topology_adaptation.csv", "tab_topo_train_test.csv"),
        ("fig6_iae_vs_distance.png", "fig_iae_vs_distance.png"),
    ],
    "section3_economic": [
        ("table10_method_economics.csv", "tab_economic_methods.csv"),
        ("fig13_pareto_profit_vs_ffr.png", "fig_pareto.png"),
    ],
    "section4_harmonic": [],  # tab_thd_compliance.csv is copied separately from --thd-csv
}

OBSOLETE: list[str] = [
    "paper/figures/fig_freq_analytic.png",
    "paper/figures/fig_freq_analytic_zoom.png",
    "paper/figures/fig_iae_bars.png",
    "paper/figures_real/fig12_revenue_decomposition.pdf",
    "paper/figures_real/fig12_revenue_decomposition.png",
    "paper/figures_real/table9_revenue_breakdown.csv",
]


def copy_section_files(src_dir: Path, dst_root: Path) -> dict[str, int]:
    """Copy/rename legacy outputs into section-organized layout."""
    counts: dict[str, int] = {}
    for section, mappings in SECTION_MAP.items():
        section_dir = dst_root / section
        section_dir.mkdir(parents=True, exist_ok=True)
        n = 0
        for src_name, dst_name in mappings:
            src = (src_dir / src_name).resolve()
            if not src.exists():
                print(f"  [SKIP] {section}/{dst_name}: source missing ({src})")
                continue
            shutil.copy2(src, section_dir / dst_name)
            print(f"  [OK]   {section}/{dst_name}  <-  {src.name}")
            n += 1
        counts[section] = n
    return counts


def derive_cost_effectiveness(src_dir: Path, dst_dir: Path) -> bool:
    """tab_cost_effectiveness.csv built from table10_method_economics.csv alone.

    Definition: cost_per_lost_success = -net_profit / max(1 - ffr_sr, 1e-3)
    Lower = more cost-efficient per unit of frequency-security insurance.
    table10 already contains both `net_profit_eur` and `ffr_success_rate`, so
    no join with table1 is required.
    """
    econ_path = src_dir / "table10_method_economics.csv"
    if not econ_path.exists():
        print(f"  [SKIP] tab_cost_effectiveness.csv: missing {econ_path.name}")
        return False

    econ = pd.read_csv(econ_path)
    profit_col = next((c for c in econ.columns if "net" in c.lower() and "profit" in c.lower()), None)
    sr_col = next((c for c in econ.columns if "ffr" in c.lower() and "success" in c.lower()), None)
    method_col = next((c for c in econ.columns if c.lower() in {"method", "controller", "policy"}), econ.columns[0])

    if profit_col is None or sr_col is None:
        print(f"  [SKIP] tab_cost_effectiveness.csv: net_profit or ffr_success_rate column not found in {econ_path.name}")
        print(f"         cols: {list(econ.columns)}")
        return False

    out = econ[[method_col, profit_col, sr_col]].copy()
    sr = out[sr_col].clip(upper=0.999).astype(float)
    out["cost_per_lost_success_eur"] = -out[profit_col].astype(float) / (1.0 - sr).clip(lower=1e-3)
    out = out.sort_values("cost_per_lost_success_eur").reset_index(drop=True)

    out_path = dst_dir / "tab_cost_effectiveness.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"  [OK]   section3_economic/tab_cost_effectiveness.csv  (derived from table10)")
    return True


def copy_thd_compliance(src_csv: Path, dst_dir: Path) -> bool:
    """Copy per-method THD compliance CSV (resolved separately from REPO_ROOT)."""
    if not src_csv.exists():
        print(f"  [SKIP] tab_thd_compliance.csv: source missing ({src_csv})")
        return False
    dst_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_csv, dst_dir / "tab_thd_compliance.csv")
    print(f"  [OK]   section4_harmonic/tab_thd_compliance.csv  <-  {src_csv.name}")
    return True


def build_thd_heatmap(npz_path: Path, dst_dir: Path) -> bool:
    """Per-bus THD_V line plot: one curve per controller, IEEE 519 limit shown.

    Output is `fig_thd_per_bus.png` + `.pdf`. The function name is kept for
    backward compatibility; the content is a line plot (outline revision).
    """
    if not npz_path.exists():
        print(f"  [SKIP] fig_thd_per_bus.png: source missing ({npz_path})")
        return False

    data = np.load(npz_path, allow_pickle=True)
    if "thd_v_per_method" not in data.files:
        print(f"  [SKIP] fig_thd_per_bus.png: 'thd_v_per_method' key missing")
        return False

    thd_v: dict = data["thd_v_per_method"].item()
    from src.eval.eval_ffr_topology import _plot_thd_per_bus_lines
    out_path = dst_dir / "fig_thd_per_bus.png"
    _plot_thd_per_bus_lines(thd_v, out_path)
    print(f"  [OK]   section4_harmonic/fig_thd_per_bus.png  (line plot built from thd_results.npz)")
    return True


def clean_obsolete(repo_root: Path, dry_run: bool) -> int:
    """Delete obsolete output files; return count deleted."""
    n = 0
    for rel in OBSOLETE:
        p = repo_root / rel
        if not p.exists():
            print(f"  [skip ] {rel}  (not present)")
            continue
        if dry_run:
            print(f"  [DRY  ] would delete {rel}")
        else:
            p.unlink()
            print(f"  [DEL  ] {rel}")
        n += 1
    return n


def write_encoder_ablation_placeholder(dst_dir: Path) -> None:
    """Encoder-ablation table is not produced by current evals; write a placeholder."""
    placeholder = dst_dir / "tab_encoder_ablation.csv"
    if placeholder.exists():
        return
    placeholder.parent.mkdir(parents=True, exist_ok=True)
    placeholder.write_text(
        "# Encoder ablation table is not produced by current eval suite.\n"
        "# Run train_baselines.py + eval_ffr_topology.py over the encoder variants\n"
        "# (GraphSAGE, GCN, GAT, MLP) to populate this file.\n"
        "encoder,iae_post_mean,ffr_sr_mean\n",
        encoding="utf-8",
    )
    print(f"  [TODO] section2_topology/tab_encoder_ablation.csv  (placeholder written)")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--src", default="paper/figures_real",
                        help="Source dir with legacy table*/fig* files (default: paper/figures_real)")
    parser.add_argument("--dst", default="results",
                        help="Destination root for section-organized layout (default: results)")
    parser.add_argument("--thd-npz", default="results/eval_full/thd_results.npz",
                        help="Path to per-bus THD npz for heatmap")
    parser.add_argument("--thd-csv", default="results/thd_verify/thd_per_method.csv",
                        help="Path to per-method THD compliance CSV (copied to section4_harmonic)")
    parser.add_argument("--clean", action="store_true",
                        help="Also delete obsolete output files (see OBSOLETE list)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print actions without modifying disk (only affects --clean)")
    args = parser.parse_args(argv)

    src_dir = (REPO_ROOT / args.src).resolve()
    dst_root = (REPO_ROOT / args.dst).resolve()
    thd_npz = (REPO_ROOT / args.thd_npz).resolve()
    thd_csv = (REPO_ROOT / args.thd_csv).resolve()

    if not src_dir.exists():
        print(f"ERROR: source dir does not exist: {src_dir}", file=sys.stderr)
        return 2

    print(f"Source : {src_dir}")
    print(f"Target : {dst_root}")
    print()
    print("=== Copying legacy outputs into section layout ===")
    counts = copy_section_files(src_dir, dst_root)

    print()
    print("=== Deriving cost-effectiveness table ===")
    derive_cost_effectiveness(src_dir, dst_root / "section3_economic")

    print()
    print("=== Copying THD compliance CSV ===")
    copy_thd_compliance(thd_csv, dst_root / "section4_harmonic")

    print()
    print("=== Building THD per-bus heatmap ===")
    build_thd_heatmap(thd_npz, dst_root / "section4_harmonic")

    print()
    print("=== Encoder-ablation placeholder ===")
    write_encoder_ablation_placeholder(dst_root / "section2_topology")

    if args.clean:
        print()
        print("=== Cleaning obsolete outputs ===")
        n_del = clean_obsolete(REPO_ROOT, dry_run=args.dry_run)
        print(f"({n_del} file(s) targeted)")

    print()
    print("Summary:", {k: v for k, v in counts.items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
