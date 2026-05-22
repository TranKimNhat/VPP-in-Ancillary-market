"""Replace mock numbers in paper/section6_results.tex with measured values.

Reads CSVs produced by ``src/eval/eval_ffr_topology.py`` and
``src/eval/eval_economics.py``, then rewrites the affected LaTeX tables
in place. Use after a full re-train + eval:

    python -m src.eval.eval_ffr_topology --checkpoint ... --output-dir paper/figures_real
    python -m src.eval.eval_economics    --output-dir paper/figures_real
    python scripts/finalize_paper.py

The script is idempotent: it locates each table block by ``\\label{...}``
and replaces the row body between ``\\midrule`` and ``\\bottomrule``.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
TEX_PATH = ROOT / "paper" / "section6_results.tex"
REAL_DIR = ROOT / "paper" / "figures_real"


def _replace_table_rows(tex: str, label: str, new_body: str) -> str:
    """Replace the rows between \\midrule and \\bottomrule of the table
    whose ``\\label{label}`` precedes them."""
    pattern = re.compile(
        rf"(\\label\{{{re.escape(label)}\}}.*?\\midrule\s*\n)(.*?)(\\bottomrule)",
        re.DOTALL,
    )
    match = pattern.search(tex)
    if match is None:
        print(f"[skip] label '{label}' not found")
        return tex
    body = new_body.rstrip() + "\n"
    return tex[: match.start(2)] + body + tex[match.end(2) :]


def _fmt_row(values: Iterable, fmt: str | None = None) -> str:
    vals = list(values)
    if fmt is None:
        return " & ".join(str(v) for v in vals) + " \\\\"
    return " & ".join(fmt.format(v) if isinstance(v, (int, float)) else str(v) for v in vals) + " \\\\"


# ----------------------------------------------------------------- Table III
def _build_ffr_main_rows(df: pd.DataFrame) -> str:
    """Build the Table III body from table1_ffr_comparison.csv.

    Columns expected: scenario, method, metric, mean, std.
    Produces a multirow block per scenario × all methods present.
    """
    methods_order = [
        "GraphSAGE-MAPPO", "MLP-MAPPO", "GCNN-PPO", "MATD3",
        "Fixed Droop", "No FFR",
    ]
    pivot = df.pivot_table(
        index=["scenario", "method"], columns="metric", values="mean", aggfunc="first"
    )
    rows: list[str] = []
    scenarios = sorted(pivot.index.get_level_values("scenario").unique())
    for sc in scenarios:
        rows.append(f"\\multirow{{{6}}}{{*}}{{{sc.replace('_', ' ')}}}")
        first = True
        for m in methods_order:
            if (sc, m) not in pivot.index:
                continue
            row = pivot.loc[(sc, m)]
            tag = " (\\emph{ours})" if m == "GraphSAGE-MAPPO" else ""
            cells = [
                f"{row.get('nadir_hz', float('nan')):.2f}",
                f"{row.get('rocof_max_hz_s', float('nan')):.2f}",
                f"{row.get('iae_post', float('nan')):.2f}",
                f"{row.get('itae', float('nan')):.1f}",
                f"{row.get('settling_time_s', float('nan')):.1f}",
                f"{100.0 * row.get('ffr_success', float('nan')):.1f}",
            ]
            if not first:
                lead = " "
            else:
                lead = ""
                first = False
            rows.append(f" {lead}& {m}{tag} & " + " & ".join(cells) + " \\\\")
        rows.append("\\midrule")
    if rows and rows[-1] == "\\midrule":
        rows.pop()
    return "\n".join(rows)


# ---------------------------------------------------------------- Table topo
def _build_topo_rows(df: pd.DataFrame) -> str:
    methods_order = ["GraphSAGE-MAPPO", "MLP-MAPPO", "GCNN-PPO", "MATD3",
                     "Fixed Droop", "No FFR"]
    rows: list[str] = []
    for m in methods_order:
        sub_train = df[(df["method"] == m) & (df["topology_split"] == "train")]
        sub_test = df[(df["method"] == m) & (df["topology_split"] == "unseen")]
        if sub_train.empty or sub_test.empty:
            continue
        tr = sub_train.iloc[0]
        te = sub_test.iloc[0]
        sr_tr = 100 * float(tr["ffr_success_rate"])
        sr_te = 100 * float(te["ffr_success_rate"])
        retention = sr_te / sr_tr * 100 if sr_tr > 0 else 0.0
        iae_gap = (te["iae_post_mean"] - tr["iae_post_mean"]) / max(tr["iae_post_mean"], 1e-6) * 100
        nadir_gap = te["nadir_hz_mean"] - tr["nadir_hz_mean"]
        tag = " (\\emph{ours})" if m == "GraphSAGE-MAPPO" else ""
        cells = [
            f"{m}{tag}",
            f"{sr_tr:.1f}",
            f"{sr_te:.1f}",
            f"{retention:.1f}\\,\\%",
            f"${'+' if iae_gap >= 0 else ''}{iae_gap:.1f}$",
            f"${'+' if nadir_gap >= 0 else ''}{nadir_gap:.2f}$",
        ]
        rows.append(" & ".join(cells) + " \\\\")
    return "\n".join(rows)


# --------------------------------------------------------------- Table severity
def _build_severity_rows(df: pd.DataFrame) -> str:
    methods_order = ["GraphSAGE-MAPPO", "MLP-MAPPO", "GCNN-PPO", "MATD3",
                     "Fixed Droop", "No FFR"]
    severities = sorted(df["delta_P_mw"].unique())
    rows: list[str] = []
    # Header summary rows are inside the LaTeX already; we just emit data rows.
    for m in methods_order:
        if m not in df["method"].unique():
            continue
        tag = " (\\emph{ours})" if m == "GraphSAGE-MAPPO" else ""
        sr_cells: list[str] = []
        for sev in severities:
            sub = df[(df["method"] == m) & (df["delta_P_mw"] == sev)]
            if sub.empty:
                sr_cells.append("---")
            else:
                sr_cells.append(f"{100 * float(sub.iloc[0]['ffr_success_rate']):.0f}")
        rows.append(f"{m}{tag} & " + " & ".join(sr_cells) + " \\\\")
    return "\n".join(rows)


# ----------------------------------------------------------- Table X (economics)
def _build_economic_method_rows(df: pd.DataFrame) -> str:
    methods_order = ["GraphSAGE-MAPPO", "MLP-MAPPO", "GCNN-PPO", "MATD3",
                     "Fixed Droop", "No FFR"]
    rows: list[str] = []
    for m in methods_order:
        sub = df[df["method"] == m]
        if sub.empty:
            continue
        r = sub.iloc[0]
        tag = " (\\emph{ours})" if m == "GraphSAGE-MAPPO" else ""
        cells = [
            f"{m}{tag}",
            f"{r.get('em_p_eur', 0):,.0f}".replace(",", "\\,"),
            f"{r.get('am_cap_eur', 0):,.0f}".replace(",", "\\,"),
            f"{r.get('am_act_eur', 0):,.0f}".replace(",", "\\,"),
            f"$-${r.get('undersupply_eur', 0):,.0f}".replace(",", "\\,"),
            f"$-${r.get('opex_eur', 0):,.0f}".replace(",", "\\,"),
            f"\\textbf{{{r.get('net_profit_eur', 0):,.0f}}}".replace(",", "\\,"),
        ]
        rows.append(" & ".join(cells) + " \\\\")
    return "\n".join(rows)


# ============================================================================
def main() -> None:
    if not TEX_PATH.exists():
        raise SystemExit(f"section6 tex not found at {TEX_PATH}")

    tex = TEX_PATH.read_text(encoding="utf-8")

    csv_map = {
        "tab:ffr_main": (REAL_DIR / "table1_ffr_comparison.csv", _build_ffr_main_rows),
        "tab:topo":     (REAL_DIR / "table2_topology_adaptation.csv", _build_topo_rows),
        "tab:severity": (REAL_DIR / "table3_severity_scaling.csv", _build_severity_rows),
        "tab:economic_methods": (REAL_DIR / "table10_method_economics.csv", _build_economic_method_rows),
    }

    for label, (csv_path, builder) in csv_map.items():
        if not csv_path.exists():
            print(f"[skip] {csv_path.name} not found")
            continue
        try:
            df = pd.read_csv(csv_path)
        except pd.errors.EmptyDataError:
            print(f"[skip] {csv_path.name} is empty")
            continue
        if df.empty:
            print(f"[skip] {csv_path.name} has no rows")
            continue
        body = builder(df)
        if not body.strip():
            print(f"[skip] {label} builder returned empty body")
            continue
        tex = _replace_table_rows(tex, label, body)
        print(f"[ok] patched {label} from {csv_path.name}")

    TEX_PATH.write_text(tex, encoding="utf-8")
    print(f"Wrote: {TEX_PATH}")


if __name__ == "__main__":
    main()
