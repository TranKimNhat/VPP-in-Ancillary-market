from __future__ import annotations

from pathlib import Path
import argparse
import json

import pandas as pd


def _safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def analyze(results_dir: Path, output_json: Path) -> Path:
    train_df = _safe_read_csv(results_dir.parent / "logs" / "train_metrics.csv")
    eval_path = results_dir / "eval_metrics.json"
    eval_data = json.loads(eval_path.read_text(encoding="utf-8")) if eval_path.exists() else {}

    benchmark_df = _safe_read_csv(results_dir / "benchmark_results.csv")
    ablation_df = _safe_read_csv(results_dir / "ablation_results.csv")
    generalization_df = _safe_read_csv(results_dir / "generalization_metrics.csv")

    summary = {
        "training": {
            "updates": int(train_df["update"].max()) if not train_df.empty else 0,
            "reward_trend_last": float(train_df["reward_mean"].iloc[-1]) if not train_df.empty else 0.0,
            "voltage_violation_last": float(train_df["voltage_violation"].iloc[-1]) if not train_df.empty else 0.0,
            "tracking_error_last": float(train_df["tracking_error"].iloc[-1]) if not train_df.empty else 0.0,
        },
        "evaluation": eval_data,
        "benchmarks_best_reward": (
            float(benchmark_df["episode_reward_mean"].max()) if not benchmark_df.empty else 0.0
        ),
        "ablation_count": int(len(ablation_df.index)),
        "generalization_count": int(len(generalization_df.index)),
    }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    trend_csv = output_json.with_suffix(".training_trend.csv")
    if not train_df.empty:
        train_df[["update", "reward_mean", "voltage_violation", "tracking_error"]].to_csv(trend_csv, index=False)

    return output_json


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate experiment outputs into a summary report.")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("artifacts/results"),
        help="Directory containing result CSV/JSON files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/results/summary.json"),
        help="Output summary JSON file.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    summary_path = analyze(args.results_dir, args.output)
    print(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    main()
