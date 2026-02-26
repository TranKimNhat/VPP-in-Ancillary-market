from __future__ import annotations

from pathlib import Path
import argparse
import json

import numpy as np
import yaml

from src.env.IEEE123bus import build_ieee123_net, validate_ieee123_net
from src.environment.grid_env import EnvConfig, GridEnvironment
from src.layer2_control.actor_critic import ActorCritic, ActorCriticConfig
from src.layer2_control.gat_encoder import GATEncoder, GATEncoderConfig
from src.layer2_control.mappo_policy import MappoPolicy, MappoPolicyConfig


def _load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _build_env(env_cfg: dict, repo_root: Path) -> GridEnvironment:
    reward_weights = env_cfg.get("reward_weights", {})
    cfg = EnvConfig(
        max_steps=int(env_cfg.get("max_steps", 96)),
        voltage_tolerance=float(env_cfg.get("voltage_tolerance", 0.05)),
        action_scale_p=float(env_cfg.get("action_scale_p", 0.2)),
        action_scale_q=float(env_cfg.get("action_scale_q", 0.2)),
    )
    cfg = EnvConfig(
        max_steps=cfg.max_steps,
        voltage_tolerance=cfg.voltage_tolerance,
        action_scale_p=cfg.action_scale_p,
        action_scale_q=cfg.action_scale_q,
        reward_weights=cfg.reward_weights.__class__(
            tracking=float(reward_weights.get("tracking", cfg.reward_weights.tracking)),
            voltage=float(reward_weights.get("voltage", cfg.reward_weights.voltage)),
            curtailment=float(reward_weights.get("curtailment", cfg.reward_weights.curtailment)),
        ),
    )

    signals = env_cfg.get("signals", {})
    layer1_csv = repo_root / signals.get("layer1_pref_csv", "") if signals.get("layer1_pref_csv") else None
    market_csv = repo_root / signals.get("market_signal_csv", "") if signals.get("market_signal_csv") else None

    net = build_ieee123_net(mode="feeder123", balanced=True, convert_switches=True, slack_zones=None)
    validate_ieee123_net(net)
    return GridEnvironment(
        net=net,
        config=cfg,
        layer1_pref_csv=layer1_csv,
        market_signal_csv=market_csv,
    )


def _build_policy(train_cfg: dict) -> MappoPolicy:
    model_cfg = train_cfg.get("model", {})
    policy_cfg = train_cfg.get("policy", {})

    encoder = GATEncoder(
        GATEncoderConfig(
            in_dim=int(model_cfg.get("in_dim", 6)),
            hidden_dim=int(model_cfg.get("hidden_dim", 32)),
            output_dim=int(model_cfg.get("output_dim", 64)),
            heads_l1=int(model_cfg.get("heads_l1", 4)),
            dropout=float(model_cfg.get("dropout", 0.1)),
        )
    )
    actor_critic = ActorCritic(
        ActorCriticConfig(
            local_state_dim=int(model_cfg.get("local_state_dim", 6)),
            graph_emb_dim=int(model_cfg.get("output_dim", 64)),
            global_state_dim=int(model_cfg.get("global_state_dim", 2)),
            action_dim=int(model_cfg.get("action_dim", 2)),
            actor_hidden=tuple(model_cfg.get("actor_hidden", [128, 64])),
            critic_hidden=tuple(model_cfg.get("critic_hidden", [256, 128])),
        )
    )
    policy = MappoPolicy(
        encoder=encoder,
        actor_critic=actor_critic,
        config=MappoPolicyConfig(
            action_low=float(policy_cfg.get("action_low", -1.0)),
            action_high=float(policy_cfg.get("action_high", 1.0)),
            gamma=float(policy_cfg.get("gamma", 0.99)),
            gae_lambda=float(policy_cfg.get("gae_lambda", 0.95)),
            clip_ratio=float(policy_cfg.get("clip_ratio", 0.2)),
            value_coef=float(policy_cfg.get("value_coef", 0.5)),
            entropy_coef=float(policy_cfg.get("entropy_coef", 0.01)),
            learning_rate=float(policy_cfg.get("learning_rate", 3e-4)),
            max_grad_norm=float(policy_cfg.get("max_grad_norm", 0.5)),
            update_epochs=int(policy_cfg.get("update_epochs", 4)),
            minibatch_size=int(policy_cfg.get("minibatch_size", 256)),
        ),
    )
    return policy


def evaluate(policy: MappoPolicy, env: GridEnvironment, episodes: int) -> dict[str, float]:
    rewards: list[float] = []
    violations: list[float] = []
    tracking_errors: list[float] = []
    curtailments: list[float] = []

    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        ep_reward = 0.0

        while not done:
            actions = {agent: policy.act_deterministic(agent_obs) for agent, agent_obs in obs.items()}
            obs, r, terminated, truncated, _ = env.step(actions)
            if r:
                ep_reward += float(np.mean(list(r.values())))
            done = bool(terminated.get("__all__", False) or truncated.get("__all__", False))

        metrics = env.metrics()
        rewards.append(ep_reward)
        violations.append(metrics["voltage_violation"])
        tracking_errors.append(metrics["tracking_error"])
        curtailments.append(metrics["curtailment_ratio"])

    return {
        "episode_reward_mean": float(np.mean(rewards)) if rewards else 0.0,
        "episode_reward_std": float(np.std(rewards)) if rewards else 0.0,
        "voltage_violation_rate": float(np.mean(violations)) if violations else 0.0,
        "tracking_error": float(np.mean(tracking_errors)) if tracking_errors else 0.0,
        "curtailment_ratio": float(np.mean(curtailments)) if curtailments else 0.0,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate trained MAPPO policy checkpoint.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Checkpoint path from train_mappo.py")
    parser.add_argument(
        "--training-config",
        type=Path,
        default=Path("configs/training_config.yaml"),
        help="Training config YAML.",
    )
    parser.add_argument(
        "--env-config",
        type=Path,
        default=Path("configs/env_config.yaml"),
        help="Environment config YAML.",
    )
    parser.add_argument("--episodes", type=int, default=3, help="Evaluation episodes.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/results/eval_metrics.json"),
        help="Output json path.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    repo_root = Path(__file__).resolve().parents[1]

    train_cfg = _load_yaml(args.training_config)
    env_cfg = _load_yaml(args.env_config)

    env = _build_env(env_cfg, repo_root)
    policy = _build_policy(train_cfg)
    policy.load_checkpoint(args.checkpoint)

    metrics = evaluate(policy, env, args.episodes)

    output_path = repo_root / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(json.dumps(metrics, indent=2))
    print(f"Saved evaluation metrics to {output_path}")


if __name__ == "__main__":
    main()
