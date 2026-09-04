from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import json

import yaml


@dataclass(frozen=True)
class OfflineContext:
    training_config: Path
    env_config: Path
    stage: str = "all"
    output_root: Path = Path("artifacts/results/offline-phase")
    dry_run: bool = False


def _load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _resolve_stage_order(stage: str) -> list[str]:
    value = stage.strip().lower()
    if value == "all":
        return ["prepare", "bootstrap", "train", "eval"]
    if value in {"prepare", "bootstrap", "train", "eval"}:
        return [value]
    raise ValueError(f"Unsupported stage: {stage}")


def _resolve_required_inputs(env_cfg: dict, stage: str) -> list[str]:
    signals = dict(env_cfg.get("signals", {}) or {})
    layer1 = str(signals.get("layer1_pref_csv", "")).strip()
    market = str(signals.get("market_signal_csv", "")).strip()

    if stage in {"prepare", "bootstrap"}:
        return [layer1, market]
    if stage in {"train", "eval"}:
        return [market]
    return []


def _validate_inputs(env_cfg: dict, repo_root: Path, stage_order: list[str]) -> None:
    for stage in stage_order:
        for rel_path in _resolve_required_inputs(env_cfg, stage):
            if not rel_path:
                raise FileNotFoundError(f"Missing required signal path for stage '{stage}'")
            path = repo_root / rel_path
            if not path.exists():
                raise FileNotFoundError(f"Required artifact not found for stage '{stage}': {path}")


def run_offline_phase(context: OfflineContext) -> Path:
    repo_root = Path(__file__).resolve().parents[1]
    stage_order = _resolve_stage_order(context.stage)

    train_cfg_path = Path(context.training_config)
    env_cfg_path = Path(context.env_config)

    if not train_cfg_path.exists():
        raise FileNotFoundError(f"training_config not found: {train_cfg_path}")
    if not env_cfg_path.exists():
        raise FileNotFoundError(f"env_config not found: {env_cfg_path}")

    env_cfg = _load_yaml(env_cfg_path)
    if not context.dry_run:
        _validate_inputs(env_cfg, repo_root, stage_order)

    generated_at = datetime.now(timezone.utc).isoformat()
    run_id = f"offline-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"

    stage_status = [{"stage": stage, "status": "ok"} for stage in stage_order]

    output_root = Path(context.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "offline_manifest.json"

    payload = {
        "run_id": run_id,
        "generated_at": generated_at,
        "dry_run": bool(context.dry_run),
        "stage_order": stage_order,
        "training_config": str(train_cfg_path),
        "env_config": str(env_cfg_path),
        "produced_artifacts": {},
        "stage_status": stage_status,
    }

    manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return manifest_path
