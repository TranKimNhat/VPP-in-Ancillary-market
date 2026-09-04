from __future__ import annotations

from pathlib import Path
import json

import pandas as pd

from experiments import eval_generalization as module


class _DummyPolicy:
    def load_checkpoint(self, checkpoint: Path) -> None:
        return


def test_generalization_protocol_fields(tmp_path: Path, monkeypatch) -> None:
    train_cfg = tmp_path / "train.yaml"
    env_cfg = tmp_path / "env.yaml"
    ckpt = tmp_path / "ckpt.pt"
    output = tmp_path / "generalization.csv"

    train_cfg.write_text("seed: 1\n", encoding="utf-8")
    env_cfg.write_text(
        "\n".join(
            [
                "signals:",
                "  market_signal_csv: data/oedisi-ieee123-main/profiles/layer0_hourly/layer0_zone_prices.csv",
            ]
        ),
        encoding="utf-8",
    )
    ckpt.write_bytes(b"ok")

    monkeypatch.setattr(module, "_load_yaml", lambda _: {"signals": {"market_signal_csv": "dummy.csv"}})
    monkeypatch.setattr(module, "_build_policy", lambda _: _DummyPolicy())
    monkeypatch.setattr(module, "_build_env", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(
        module,
        "evaluate",
        lambda *_args, **_kwargs: {
            "episode_reward_mean": 1.0,
            "voltage_violation_rate": 0.0,
            "tracking_error": 0.1,
        },
    )

    module.run_generalization(
        training_config=train_cfg,
        env_config=env_cfg,
        checkpoint=ckpt,
        levels=["interpolation"],
        output=output,
        run_id="run-123",
        generated_at="2026-03-11T00:00:00Z",
        source_signal_paths={"market_signal_csv": "dummy.csv"},
    )

    rows = json.loads(output.with_suffix(".json").read_text(encoding="utf-8"))
    assert rows
    row = rows[0]
    assert row["run_id"] == "run-123"
    assert row["checkpoint"] == str(ckpt)
    assert row["env_config"] == str(env_cfg)
    assert row["training_config"] == str(train_cfg)
    assert row["generated_at"] == "2026-03-11T00:00:00Z"
    assert json.loads(row["source_signal_paths"]) == {"market_signal_csv": "dummy.csv"}

    csv_df = pd.read_csv(output)
    assert "run_id" in csv_df.columns
