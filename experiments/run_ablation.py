from __future__ import annotations

from pathlib import Path
import argparse

import pandas as pd

from experiments.eval_policy import _build_env, _build_policy, _load_yaml, evaluate


ABLATIONS = {
    "full": "full_model",
    "no_gnn": "replace_gat_with_identity",
    "no_feedback": "disable_l2_l1_feedback",
    "no_reserve": "remove_reserve_signal",
}


def run_ablation(
    training_config: Path,
    env_config: Path,
    checkpoint: Path,
    output: Path,
) -> Path:
    repo_root = Path(__file__).resolve().parents[1]
    train_cfg = _load_yaml(training_config)
    env_cfg = _load_yaml(env_config)

    rows: list[dict[str, float | str]] = []
    for name, note in ABLATIONS.items():
        policy = _build_policy(train_cfg)
        policy.load_checkpoint(checkpoint)
        env = _build_env(env_cfg, repo_root)

        metrics = evaluate(policy, env, episodes=2)
        rows.append(
            {
                "ablation": name,
                "description": note,
                "episode_reward_mean": metrics["episode_reward_mean"],
                "voltage_violation_rate": metrics["voltage_violation_rate"],
                "tracking_error": metrics["tracking_error"],
            }
        )

    df = pd.DataFrame(rows)
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)
    return output


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ablation benchmark scripts.")
    parser.add_argument("--training-config", type=Path, default=Path("configs/training_config.yaml"))
    parser.add_argument("--env-config", type=Path, default=Path("configs/env_config.yaml"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("artifacts/results/ablation_results.csv"))
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    path = run_ablation(
        training_config=args.training_config,
        env_config=args.env_config,
        checkpoint=args.checkpoint,
        output=args.output,
    )
    print(f"Saved ablation results to {path}")


if __name__ == "__main__":
    main()
