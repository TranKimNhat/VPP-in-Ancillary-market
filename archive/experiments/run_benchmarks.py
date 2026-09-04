from __future__ import annotations

from pathlib import Path
import argparse

import numpy as np
import pandas as pd

from experiments.eval_policy import _build_env, _build_policy, _load_yaml, evaluate


def _evaluate_random(env, episodes: int = 2) -> dict[str, float]:
    rewards = []
    violations = []
    tracking = []
    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        ep_reward = 0.0
        while not done:
            actions = {agent: np.random.uniform(-1.0, 1.0, size=2).astype(np.float32) for agent in obs}
            obs, r, terminated, truncated, _ = env.step(actions)
            ep_reward += float(np.mean(list(r.values()))) if r else 0.0
            done = bool(terminated.get("__all__", False) or truncated.get("__all__", False))
        m = env.metrics()
        rewards.append(ep_reward)
        violations.append(m["voltage_violation"])
        tracking.append(m["tracking_error"])
    return {
        "episode_reward_mean": float(np.mean(rewards)),
        "voltage_violation_rate": float(np.mean(violations)),
        "tracking_error": float(np.mean(tracking)),
    }


def _evaluate_rule_based(env, episodes: int = 2) -> dict[str, float]:
    rewards = []
    violations = []
    tracking = []
    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        ep_reward = 0.0
        while not done:
            actions = {}
            for agent, agent_obs in obs.items():
                local_state = np.asarray(agent_obs["local_state"], dtype=float)
                v = float(local_state[0])
                q_cmd = np.clip((1.0 - v) * 4.0, -1.0, 1.0)
                actions[agent] = np.array([0.0, q_cmd], dtype=np.float32)
            obs, r, terminated, truncated, _ = env.step(actions)
            ep_reward += float(np.mean(list(r.values()))) if r else 0.0
            done = bool(terminated.get("__all__", False) or truncated.get("__all__", False))
        m = env.metrics()
        rewards.append(ep_reward)
        violations.append(m["voltage_violation"])
        tracking.append(m["tracking_error"])
    return {
        "episode_reward_mean": float(np.mean(rewards)),
        "voltage_violation_rate": float(np.mean(violations)),
        "tracking_error": float(np.mean(tracking)),
    }


def run_benchmarks(
    training_config: Path,
    env_config: Path,
    checkpoint: Path,
    output: Path,
) -> Path:
    repo_root = Path(__file__).resolve().parents[1]
    train_cfg = _load_yaml(training_config)
    env_cfg = _load_yaml(env_config)

    rows: list[dict[str, float | str]] = []

    env_rl = _build_env(env_cfg, repo_root)
    policy = _build_policy(train_cfg)
    policy.load_checkpoint(checkpoint)
    rl_metrics = evaluate(policy, env_rl, episodes=2)
    rows.append({"baseline": "mappo_gat", **rl_metrics})

    env_rule = _build_env(env_cfg, repo_root)
    rows.append({"baseline": "rule_based_droop", **_evaluate_rule_based(env_rule, episodes=2)})

    env_random = _build_env(env_cfg, repo_root)
    rows.append({"baseline": "random", **_evaluate_random(env_random, episodes=2)})

    df = pd.DataFrame(rows)
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)
    return output


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run baseline benchmarks.")
    parser.add_argument("--training-config", type=Path, default=Path("configs/training_config.yaml"))
    parser.add_argument("--env-config", type=Path, default=Path("configs/env_config.yaml"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("artifacts/results/benchmark_results.csv"))
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    out = run_benchmarks(
        training_config=args.training_config,
        env_config=args.env_config,
        checkpoint=args.checkpoint,
        output=args.output,
    )
    print(f"Saved benchmark results to {out}")


if __name__ == "__main__":
    main()
