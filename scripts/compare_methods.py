from __future__ import annotations

import argparse
import json
from math import sqrt
from pathlib import Path

import numpy as np
from scipy import stats


_SUPPORTED_METRICS = ("final_reward", "best_eval_reward", "mean_eval_reward")


def _extract_values(path: Path, metric: str = "final_reward") -> tuple[str, np.ndarray]:
    if metric not in _SUPPORTED_METRICS:
        raise ValueError(f"metric must be one of {_SUPPORTED_METRICS}, got {metric!r}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    values = []
    for item in payload.get("individual", []):
        if item.get("status") == "ok" and metric in item:
            values.append(float(item[metric]))
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


def _cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 2 or b.size < 2:
        return float("nan")

    var_a = float(np.var(a, ddof=1))
    var_b = float(np.var(b, ddof=1))
    pooled_num = (a.size - 1) * var_a + (b.size - 1) * var_b
    pooled_den = float(a.size + b.size - 2)
    if pooled_den <= 0:
        return float("nan")

    pooled_std = sqrt(pooled_num / pooled_den)
    if pooled_std <= 0:
        return float("nan")

    return float((np.mean(a) - np.mean(b)) / pooled_std)


def _gate_decision(candidate: np.ndarray, baseline: np.ndarray, d_value: float, metric: str) -> str:
    if not np.isfinite(d_value):
        return "INSUFFICIENT_DATA"

    direction_positive = float(np.mean(candidate)) > float(np.mean(baseline))

    if metric == "final_reward":
        # Path A pre-registered rule: d >= 0.5 and positive direction
        if direction_positive and d_value >= 0.5:
            return "SCALE_TO_N5"
        return "STOP_N3_NEGATIVE"

    if metric == "best_eval_reward":
        # Path B D.5 pre-registered rule: d >= 0.3 and positive direction
        if direction_positive and d_value >= 0.3:
            return "SCALE_TO_N5"
        return "STOP_N3_NEGATIVE"

    # generic: report only, no pre-registered rule
    return "NO_PREREG_RULE"


def compare(baseline_path: Path, candidate_path: Path, metric: str = "final_reward") -> str:
    base_name, base = _extract_values(baseline_path, metric)
    cand_name, cand = _extract_values(candidate_path, metric)

    t_stat, p_value = _welch_t_test(cand, base)
    d_value = _cohens_d(cand, base)
    decision = _gate_decision(cand, base, d_value, metric)

    lines = []
    lines.append(f"metric: {metric}")
    lines.append("method | mean | std | N | p-value vs baseline")
    lines.append("---|---:|---:|---:|---:")
    lines.append(f"{base_name} | {np.mean(base):.4f} | {np.std(base):.4f} | {base.size} | -")
    lines.append(f"{cand_name} | {np.mean(cand):.4f} | {np.std(cand):.4f} | {cand.size} | {p_value:.6f}")
    lines.append("")
    lines.append(f"Welch t-statistic: {t_stat:.6f}")
    lines.append(f"Cohen's d (candidate-baseline): {d_value:.6f}")
    lines.append(f"DECISION={decision}")
    return "\n".join(lines)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare two multi-seed JSON outputs.")
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--metric", default="final_reward", choices=list(_SUPPORTED_METRICS))
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    print(compare(args.baseline, args.candidate, metric=args.metric))


if __name__ == "__main__":
    main()
