from __future__ import annotations

from pathlib import Path

import pandas as pd

from experiments.train_mappo import run_training


def test_training_smoke(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]

    train_cfg = tmp_path / "training_config.yaml"
    env_cfg = tmp_path / "env_config.yaml"

    train_cfg.write_text(
        "\n".join(
            [
                "seed: 7",
                "updates: 2",
                "rollout_steps: 16",
                "eval_interval: 1",
                "save_interval: 1",
                "checkpoint_dir: artifacts/checkpoints",
                "log_path: artifacts/logs/train_metrics.csv",
                "policy:",
                "  learning_rate: 0.0003",
                "  minibatch_size: 16",
                "model:",
                "  in_dim: 6",
                "  hidden_dim: 8",
                "  output_dim: 16",
                "  heads_l1: 2",
                "  local_state_dim: 6",
                "  global_state_dim: 2",
                "  action_dim: 2",
                "  actor_hidden: [32, 16]",
                "  critic_hidden: [64, 32]",
            ]
        ),
        encoding="utf-8",
    )

    env_cfg.write_text(
        "\n".join(
            [
                "max_steps: 16",
                "voltage_tolerance: 0.05",
                "action_scale_p: 0.1",
                "action_scale_q: 0.1",
                "signals:",
                "  layer1_pref_csv: data/oedisi-ieee123-main/profiles/layer1_vpp/layer1_pref.csv",
                "  market_signal_csv: data/oedisi-ieee123-main/profiles/layer0_hourly/layer0_zone_prices.csv",
            ]
        ),
        encoding="utf-8",
    )

    run_training(train_cfg, env_cfg, bootstrap_tri_layer=False)

    log_path = repo_root / "artifacts/logs/train_metrics.csv"
    ckpt_path = repo_root / "artifacts/checkpoints/mappo_final.pt"

    assert log_path.exists()
    assert ckpt_path.exists()

    df = pd.read_csv(log_path)
    assert len(df) >= 2
    assert df["total_loss"].notna().all()
