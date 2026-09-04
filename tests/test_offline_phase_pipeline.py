from __future__ import annotations

from pathlib import Path
import json

from experiments.offline_phase import OfflineContext, run_offline_phase


def _write_cfgs(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    train_cfg = tmp_path / "training.yaml"
    env_cfg = tmp_path / "env.yaml"
    layer1_csv = tmp_path / "layer1.csv"
    market_csv = tmp_path / "market.csv"

    train_cfg.write_text(
        "\n".join(
            [
                "seed: 1",
                "updates: 1",
                "rollout_steps: 2",
                "eval_interval: 1",
                "save_interval: 1",
                "checkpoint_dir: artifacts/checkpoints",
                "log_path: artifacts/logs/train_metrics.csv",
                "policy:",
                "  minibatch_size: 2",
                "model:",
                "  in_dim: 6",
                "  hidden_dim: 8",
                "  output_dim: 16",
                "  heads_l1: 2",
                "  local_state_dim: 6",
                "  global_state_dim: 7",
                "  action_dim: 2",
            ]
        ),
        encoding="utf-8",
    )

    env_cfg.write_text(
        "\n".join(
            [
                "max_steps: 2",
                "zoning_mode: static",
                "signals:",
                f"  layer1_pref_csv: {layer1_csv.as_posix()}",
                f"  market_signal_csv: {market_csv.as_posix()}",
            ]
        ),
        encoding="utf-8",
    )

    layer1_csv.write_text("hour,P_ref,Q_ref,R_commit\n0,0.1,0.0,0.1\n", encoding="utf-8")
    market_csv.write_text("hour,energy_price,reserve_price\n0,10.0,1.0\n", encoding="utf-8")

    return train_cfg, env_cfg, layer1_csv, market_csv


def test_offline_phase_dry_run_stage_order(tmp_path: Path) -> None:
    train_cfg, env_cfg, _, _ = _write_cfgs(tmp_path)

    manifest = run_offline_phase(
        OfflineContext(
            training_config=train_cfg,
            env_config=env_cfg,
            stage="all",
            output_root=Path("artifacts/results/offline-test"),
            dry_run=True,
        )
    )

    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert payload["dry_run"] is True
    assert payload["stage_order"] == ["prepare", "bootstrap", "train", "eval"]
    assert all(item["status"] == "ok" for item in payload["stage_status"])


def test_offline_phase_fail_closed_missing_artifact(tmp_path: Path) -> None:
    train_cfg, env_cfg, layer1_csv, _ = _write_cfgs(tmp_path)
    layer1_csv.unlink()

    try:
        run_offline_phase(
            OfflineContext(
                training_config=train_cfg,
                env_config=env_cfg,
                stage="prepare",
                output_root=Path("artifacts/results/offline-test"),
                dry_run=False,
            )
        )
    except FileNotFoundError:
        return
    raise AssertionError("Expected fail-closed FileNotFoundError for missing required artifact")


def test_offline_phase_manifest_contains_required_fields(tmp_path: Path) -> None:
    train_cfg, env_cfg, _, _ = _write_cfgs(tmp_path)

    manifest = run_offline_phase(
        OfflineContext(
            training_config=train_cfg,
            env_config=env_cfg,
            stage="prepare",
            output_root=Path("artifacts/results/offline-test"),
            dry_run=True,
        )
    )

    payload = json.loads(manifest.read_text(encoding="utf-8"))
    for key in [
        "run_id",
        "generated_at",
        "stage_order",
        "training_config",
        "env_config",
        "produced_artifacts",
        "stage_status",
    ]:
        assert key in payload
