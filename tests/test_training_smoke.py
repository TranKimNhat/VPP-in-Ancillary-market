from __future__ import annotations

from pathlib import Path

import pandas as pd

from experiments.train_mappo import run_training
from src.env.IEEE123bus import build_ieee123_net


def test_training_smoke(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]

    train_cfg = tmp_path / "training_config.yaml"
    env_cfg = tmp_path / "env_config.yaml"
    bus_to_zone_csv = tmp_path / "bus_to_zone.csv"
    vpp_to_zone_csv = tmp_path / "vpp_to_zone.csv"

    net = build_ieee123_net(mode="feeder123", balanced=True, convert_switches=True, slack_zones=None)
    bus_ids = [int(b) for b in net.bus.index]
    pd.DataFrame({"bus": bus_ids, "zone_id": [1] * len(bus_ids)}).to_csv(bus_to_zone_csv, index=False)
    pd.DataFrame(columns=["vpp_id", "zone_id"]).to_csv(vpp_to_zone_csv, index=False)

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
                "layer1:",
                "  vpp_mode:",
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
                "zoning_mode: static",
                "signals:",
                "  layer1_pref_csv: data/oedisi-ieee123-main/profiles/layer1_vpp/layer1_pref.csv",
                "  market_signal_csv: data/oedisi-ieee123-main/profiles/layer0_hourly/layer0_zone_prices.csv",
                "mappings:",
                f"  bus_to_zone_csv: {bus_to_zone_csv.as_posix()}",
                f"  vpp_to_zone_csv: {vpp_to_zone_csv.as_posix()}",
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


def test_training_smoke_vpp_mode(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]

    train_cfg = tmp_path / "training_config_vpp.yaml"
    env_cfg = tmp_path / "env_config_vpp.yaml"

    bus_to_zone_csv = tmp_path / "bus_to_zone.csv"
    vpp_to_zone_csv = tmp_path / "vpp_to_zone.csv"
    bus_to_vpp_csv = tmp_path / "bus_to_vpp.csv"
    layer1_vpp_csv = tmp_path / "layer1_vpp.csv"

    net = build_ieee123_net(mode="feeder123", balanced=True, convert_switches=True, slack_zones=None)
    bus_ids = [int(b) for b in net.bus.index]
    pd.DataFrame({"bus": bus_ids, "zone_id": [1 if i < len(bus_ids) // 2 else 2 for i, _ in enumerate(bus_ids)]}).to_csv(bus_to_zone_csv, index=False)
    pd.DataFrame({"vpp_id": ["vpp_a", "vpp_b", "vpp_c"], "zone_id": [1, 1, 2]}).to_csv(vpp_to_zone_csv, index=False)
    bus_to_vpp_rows = []
    for i, bus in enumerate(bus_ids):
        zone_id = 1 if i < len(bus_ids) // 2 else 2
        if zone_id == 1:
            owner = "vpp_a" if i % 3 == 0 else "vpp_b" if i % 3 == 1 else None
        else:
            owner = "vpp_c" if i % 3 == 1 else None
        bus_to_vpp_rows.append({"bus": bus, "vpp_id": owner})
    pd.DataFrame(bus_to_vpp_rows).to_csv(bus_to_vpp_csv, index=False)

    pd.DataFrame(
        {
            "day": ["expected"] * 6,
            "hour": [0, 0, 0, 1, 1, 1],
            "zone_id": ["1", "1", "2", "1", "1", "2"],
            "vpp_id": ["vpp_a", "vpp_b", "vpp_c", "vpp_a", "vpp_b", "vpp_c"],
            "P_ref": [0.3, 0.4, 0.5, 0.35, 0.45, 0.55],
            "R_commit": [0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
        }
    ).to_csv(layer1_vpp_csv, index=False)

    train_cfg.write_text(
        "\n".join(
            [
                "seed: 9",
                "updates: 1",
                "rollout_steps: 8",
                "eval_interval: 1",
                "save_interval: 1",
                "checkpoint_dir: artifacts/checkpoints",
                "log_path: artifacts/logs/train_metrics_vpp.csv",
                "policy:",
                "  learning_rate: 0.0003",
                "  minibatch_size: 8",
                "layer1:",
                "  vpp_mode:",
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
                "max_steps: 8",
                "voltage_tolerance: 0.05",
                "action_scale_p: 0.1",
                "action_scale_q: 0.1",
                "zoning_mode: static",
                "vpp_mode: true",
                "signals:",
                f"  layer1_pref_csv: {layer1_vpp_csv.as_posix()}",
                "  market_signal_csv: data/oedisi-ieee123-main/profiles/layer0_hourly/layer0_zone_prices.csv",
                "mappings:",
                f"  bus_to_zone_csv: {bus_to_zone_csv.as_posix()}",
                f"  vpp_to_zone_csv: {vpp_to_zone_csv.as_posix()}",
                f"  bus_to_vpp_csv: {bus_to_vpp_csv.as_posix()}",
            ]
        ),
        encoding="utf-8",
    )

    run_training(train_cfg, env_cfg, bootstrap_tri_layer=False)

    log_path = repo_root / "artifacts/logs/train_metrics_vpp.csv"
    assert log_path.exists()
    df = pd.read_csv(log_path)
    assert len(df) >= 1
    assert df["total_loss"].notna().all()


def test_training_rejects_non_static_zoning(tmp_path: Path) -> None:
    train_cfg = tmp_path / "training_config.yaml"
    env_cfg = tmp_path / "env_config.yaml"

    train_cfg.write_text(
        "\n".join(
            [
                "seed: 1",
                "updates: 1",
                "rollout_steps: 2",
                "layer1:",
                "  vpp_mode:",
            ]
        ),
        encoding="utf-8",
    )

    env_cfg.write_text(
        "\n".join(
            [
                "max_steps: 2",
                "zoning_mode: dynamic",
            ]
        ),
        encoding="utf-8",
    )

    try:
        run_training(train_cfg, env_cfg, bootstrap_tri_layer=False)
    except NotImplementedError:
        return
    raise AssertionError("Expected NotImplementedError when zoning_mode is not static")
