from __future__ import annotations

import argparse
import json
from math import sqrt
from pathlib import Path

import numpy as np
from scipy import stats


def _extract_values(path: Path) -> tuple[str, np.ndarray]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    values = []
    for item in payload.get("individual", []):
        if item.get("status") == "ok" and "final_reward" in item:
            values.append(float(item["final_reward"]))
    arr = np.asarray(values, dtype=np.float64)
    method = path.stem
    return method, arr


def _welch_t_test(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    if a.size < 2 or b.size < 2:
        return float("nan"), float("nan")

    mean_a, mean_b = float(np.mean(a)), float(np.mean(b))
    var_a, var_b = float(np.var(a, ddof=1)), float(np.var(b, ddof=1))
    n_a, n_b = int(a.size), int(b.size)

    denom = sqrt(var_a / n_a + var_b / n_b)
    if denom <= 0:
        return float("nan"), float("nan")

    t_stat = (mean_a - mean_b) / denom
    df_num = (var_a / n_a + var_b / n_b) ** 2
    df_den = ((var_a / n_a) ** 2) / max(n_a - 1, 1) + ((var_b / n_b) ** 2) / max(n_b - 1, 1)
    if df_den <= 0:
        return float("nan"), float("nan")

    df = df_num / df_den
    p_value = float(2.0 * stats.t.sf(abs(t_stat), df))
    return float(t_stat), p_value


def compare(baseline_path: Path, candidate_path: Path) -> str:
    base_name, base = _extract_values(baseline_path)
    cand_name, cand = _extract_values(candidate_path)

    t_stat, p_value = _welch_t_test(cand, base)

    lines = []
    lines.append("method | mean | std | N | p-value vs baseline")
    lines.append("---|---:|---:|---:|---:")
    lines.append(f"{base_name} | {np.mean(base):.4f} | {np.std(base):.4f} | {base.size} | -")
    lines.append(f"{cand_name} | {np.mean(cand):.4f} | {np.std(cand):.4f} | {cand.size} | {p_value:.6f}")
    lines.append("")
    lines.append(f"Welch t-statistic: {t_stat:.6f}")
    return "\n".join(lines)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare two multi-seed JSON outputs.")
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    print(compare(args.baseline, args.candidate))


if __name__ == "__main__":
    main()
