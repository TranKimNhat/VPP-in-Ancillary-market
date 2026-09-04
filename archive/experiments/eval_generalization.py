from __future__ import annotations

from pathlib import Path
from datetime import datetime, timezone
import argparse
import json

import pandas as pd

from experiments.eval_policy import _build_env, _build_policy, _load_yaml, evaluate


def run_generalization(
    training_config: Path,
    env_config: Path,
    checkpoint: Path,
    levels: list[str],
    output: Path,
    run_id: str | None = None,
    generated_at: str | None = None,
    source_signal_paths: dict[str, str] | None = None,
) -> Path:
    repo_root = Path(__file__).resolve().parents[1]
    train_cfg = _load_yaml(training_config)
    env_cfg = _load_yaml(env_config)

    base_layer0 = env_cfg.get("signals", {}).get("market_signal_csv")
    if not base_layer0:
        raise ValueError("env_config signals.market_signal_csv is required")

    policy = _build_policy(train_cfg)
    policy.load_checkpoint(checkpoint)

    resolved_run_id = run_id or checkpoint.stem
    resolved_generated_at = generated_at or datetime.now(timezone.utc).isoformat()
    resolved_source_signal_paths = source_signal_paths or {"market_signal_csv": str(base_layer0)}

    rows: list[dict[str, float | str]] = []
    for level in levels:
        level_cfg = dict(env_cfg)
        level_cfg["signals"] = dict(env_cfg.get("signals", {}))
        level_cfg["signals"]["market_signal_csv"] = base_layer0

        env = _build_env(level_cfg, repo_root)
        metrics = evaluate(policy, env, episodes=3)
        rows.append(
            {
                "run_id": resolved_run_id,
                "generated_at": resolved_generated_at,
                "training_config": str(training_config),
                "env_config": str(env_config),
                "checkpoint": str(checkpoint),
                "source_signal_paths": json.dumps(resolved_source_signal_paths, separators=(",", ":")),
                "level": level,
                "episode_reward_mean": metrics["episode_reward_mean"],
                "voltage_violation_rate": metrics["voltage_violation_rate"],
                "tracking_error": metrics["tracking_error"],
                "generalization_gap_proxy": -metrics["episode_reward_mean"],
            }
        )

    df = pd.DataFrame(rows)
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)

    summary_path = output.with_suffix(".json")
    summary_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return output


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run topology generalization evaluation.")
    parser.add_argument("--training-config", type=Path, default=Path("configs/training_config.yaml"))
    parser.add_argument("--env-config", type=Path, default=Path("configs/env_config.yaml"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--levels",
        nargs="+",
        default=["interpolation", "extrapolation", "extreme_shift"],
        help="Generalization levels to report.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/results/generalization_metrics.csv"),
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    result_path = run_generalization(
        training_config=args.training_config,
        env_config=args.env_config,
        checkpoint=args.checkpoint,
        levels=args.levels,
        output=args.output,
    )
    print(f"Saved generalization results to {result_path}")


if __name__ == "__main__":
    main()
