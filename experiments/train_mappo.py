from __future__ import annotations

from pathlib import Path
import argparse
import csv

import numpy as np
import yaml

from src.env.IEEE123bus import build_ieee123_net, validate_ieee123_net
from src.environment.grid_env import EnvConfig, GridEnvironment
from src.layer0_dso.layer0_dso import run_layer0_pipeline
from src.layer1_vpp.layer1_vpp import Layer1Config, run_layer1
from src.layer2_control.actor_critic import ActorCritic, ActorCriticConfig
from src.layer2_control.gat_encoder import GATEncoder, GATEncoderConfig
from src.layer2_control.mappo_policy import MappoPolicy, MappoPolicyConfig, RolloutBuffer


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


def _eval_episode(env: GridEnvironment, policy: MappoPolicy) -> dict[str, float]:
    obs, _ = env.reset()
    done = False
    total_reward = 0.0

    while not done:
        actions = {agent: policy.act_deterministic(agent_obs) for agent, agent_obs in obs.items()}
        obs, rewards, terminated, truncated, _ = env.step(actions)
        total_reward += float(np.mean(list(rewards.values()))) if rewards else 0.0
        done = bool(terminated.get("__all__", False) or truncated.get("__all__", False))

    metrics = env.metrics()
    metrics["episode_reward"] = total_reward
    return metrics


def _run_layer0_layer1(repo_root: Path, env_cfg: dict) -> tuple[Path, Path]:
    signals = env_cfg.get("signals", {})
    layer0_csv = repo_root / signals.get("market_signal_csv", "data/oedisi-ieee123-main/profiles/layer0_hourly/layer0_zone_prices.csv")
    layer1_csv = repo_root / signals.get("layer1_pref_csv", "data/oedisi-ieee123-main/profiles/layer1_vpp/layer1_pref.csv")

    layer0_bundle = run_layer0_pipeline(
        output_dir=layer0_csv.parent,
        pricing_method="load_weighted",
        ac_tolerance=0.001,
    )

    if not layer0_bundle.valid_for_layer1:
        raise RuntimeError(
            "Layer0 quality gate failed. See diagnostics at "
            f"{layer0_bundle.diagnostics_csv} before running Layer1."
        )

    layer1_cfg = Layer1Config(
        input_csv=layer0_bundle.zone_prices_csv,
        output_csv=layer1_csv,
        weights={"offpeak": 0.5, "median": 0.3, "peak": 0.2},
        sign="inject",
        wasserstein_radius=0.02,
        degradation_cost=1.0,
    )
    run_layer1(layer1_cfg)
    return layer0_bundle.zone_prices_csv, layer1_csv


def run_training(train_cfg_path: Path, env_cfg_path: Path, bootstrap_tri_layer: bool = False) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    train_cfg = _load_yaml(train_cfg_path)
    env_cfg = _load_yaml(env_cfg_path)

    if bootstrap_tri_layer:
        layer0_out, layer1_out = _run_layer0_layer1(repo_root, env_cfg)
        print(f"Layer0 output: {layer0_out}")
        print(f"Layer1 output: {layer1_out}")

    seed = int(train_cfg.get("seed", 42))
    np.random.seed(seed)

    env = _build_env(env_cfg, repo_root)
    policy = _build_policy(train_cfg)

    updates = int(train_cfg.get("updates", 20))
    rollout_steps = int(train_cfg.get("rollout_steps", 128))
    eval_interval = int(train_cfg.get("eval_interval", 5))
    save_interval = int(train_cfg.get("save_interval", 5))

    checkpoint_dir = repo_root / str(train_cfg.get("checkpoint_dir", "artifacts/checkpoints"))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    log_path = repo_root / str(train_cfg.get("log_path", "artifacts/logs/train_metrics.csv"))
    log_path.parent.mkdir(parents=True, exist_ok=True)

    with log_path.open("w", newline="", encoding="utf-8") as log_file:
        writer = csv.DictWriter(
            log_file,
            fieldnames=[
                "update",
                "policy_loss",
                "value_loss",
                "entropy",
                "total_loss",
                "reward_mean",
                "tracking_error",
                "voltage_violation",
                "curtailment_ratio",
            ],
        )
        writer.writeheader()

        obs, _ = env.reset(seed=seed)
        for update in range(1, updates + 1):
            buffer = RolloutBuffer()
            reward_trace: list[float] = []

            for _ in range(rollout_steps):
                actions: dict[str, np.ndarray] = {}
                action_meta: dict[str, tuple[float, float]] = {}
                for agent, agent_obs in obs.items():
                    action, log_prob, value = policy.act(agent_obs)
                    actions[agent] = action
                    action_meta[agent] = (log_prob, value)

                next_obs, rewards, terminated, truncated, _ = env.step(actions)
                done_all = bool(terminated.get("__all__", False) or truncated.get("__all__", False))

                for agent, agent_obs in obs.items():
                    log_prob, value = action_meta[agent]
                    reward = float(rewards.get(agent, 0.0))
                    reward_trace.append(reward)
                    buffer.add(
                        node_features=np.asarray(agent_obs["node_features"], dtype=np.float32),
                        adjacency=np.asarray(agent_obs["adjacency"], dtype=np.float32),
                        agent_index=int(agent_obs["agent_index"]),
                        local_state=np.asarray(agent_obs["local_state"], dtype=np.float32),
                        global_state=np.asarray(agent_obs["global_state"], dtype=np.float32),
                        action=np.asarray(actions[agent], dtype=np.float32),
                        log_prob=log_prob,
                        reward=reward,
                        value=value,
                        done=done_all,
                    )

                obs = next_obs
                if done_all:
                    obs, _ = env.reset()

            last_obs = next(iter(obs.values()))
            last_value = policy.evaluate_value(last_obs)
            losses = policy.update(buffer, last_value=last_value)

            if not np.isfinite(losses["total_loss"]):
                raise RuntimeError("Training diverged: total_loss is NaN/Inf.")

            metrics = env.metrics()
            row = {
                "update": update,
                "policy_loss": losses["policy_loss"],
                "value_loss": losses["value_loss"],
                "entropy": losses["entropy"],
                "total_loss": losses["total_loss"],
                "reward_mean": float(np.mean(reward_trace)) if reward_trace else 0.0,
                "tracking_error": metrics["tracking_error"],
                "voltage_violation": metrics["voltage_violation"],
                "curtailment_ratio": metrics["curtailment_ratio"],
            }
            writer.writerow(row)
            log_file.flush()

            if update % save_interval == 0:
                policy.save_checkpoint(checkpoint_dir / f"mappo_update_{update}.pt")

            if update % eval_interval == 0:
                eval_metrics = _eval_episode(env, policy)
                print(
                    f"[eval] update={update} reward={eval_metrics['episode_reward']:.4f} "
                    f"v_viol={eval_metrics['voltage_violation']:.6f} "
                    f"track_err={eval_metrics['tracking_error']:.6f}"
                )

                if eval_metrics.get("curtailment_ratio", 0.0) > 0.05:
                    signals = env_cfg.get("signals", {})
                    layer1_csv = repo_root / signals.get("layer1_pref_csv", "data/oedisi-ieee123-main/profiles/layer1_vpp/layer1_pref.csv")
                    layer0_csv = repo_root / signals.get("market_signal_csv", "data/oedisi-ieee123-main/profiles/layer0_hourly/layer0_zone_prices.csv")
                    feedback_cfg = Layer1Config(
                        input_csv=layer0_csv,
                        output_csv=layer1_csv,
                        weights={"offpeak": 0.5, "median": 0.3, "peak": 0.2},
                        sign="inject",
                        wasserstein_radius=0.02,
                        degradation_cost=1.0,
                        curtailment_ratio=eval_metrics["curtailment_ratio"],
                        feedback_threshold=0.05,
                    )
                    run_layer1(feedback_cfg)
                    print("Triggered Layer1 re-optimization from Layer2 curtailment feedback.")

    policy.save_checkpoint(checkpoint_dir / "mappo_final.pt")
    print(f"Training complete. Logs: {log_path}")
    print(f"Checkpoint: {checkpoint_dir / 'mappo_final.pt'}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train GAT-MAPPO for Layer 2 control.")
    parser.add_argument(
        "--training-config",
        type=Path,
        default=Path("configs/training_config.yaml"),
        help="Training configuration YAML.",
    )
    parser.add_argument(
        "--env-config",
        type=Path,
        default=Path("configs/env_config.yaml"),
        help="Environment configuration YAML.",
    )
    parser.add_argument(
        "--bootstrap-tri-layer",
        action="store_true",
        help="Run Layer0->Layer1 pipeline before training.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    run_training(args.training_config, args.env_config, bootstrap_tri_layer=args.bootstrap_tri_layer)


if __name__ == "__main__":
    main()
