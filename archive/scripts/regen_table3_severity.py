"""Regenerate table3_severity_scaling.csv (the source for fig_severity) and redraw
the figure. Uses the current settling metric unchanged (the settling-definition
question is still open); panel (b) will therefore saturate near the 50-s cap.
"""
from __future__ import annotations
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")

from scripts.dso_cost_per_event import build_evaluator
from src.eval.figures_style import apply_paper_style
import scripts.make_paper_figures as F

OUT_DIR = ROOT / "results" / "results_260620"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ev = build_evaluator()
ev.output_dir = OUT_DIR            # write table3 where fig_severity reads it

print("[1/2] Table 3 (severity scaling, n_runs=10)...", flush=True)
df = ev.build_table3_severity_scaling(n_runs=10)
print(df.round(3).to_string(index=False), flush=True)

print("\n[2/2] Drawing fig_severity...", flush=True)
apply_paper_style()
F.OUT = ROOT / "results" / "paper_figures"
F.OUT.mkdir(parents=True, exist_ok=True)
F.fig_severity(OUT_DIR)

print("\nDone.", flush=True)
print(f"  -> {OUT_DIR / 'table3_severity_scaling.csv'}", flush=True)
print(f"  -> IEEE_Journal_Paper_Template/img/fig_severity.pdf", flush=True)
