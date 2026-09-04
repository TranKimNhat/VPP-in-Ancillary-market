from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path

import numpy as np
import yaml


def _load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _run_single(
    repo_root: Path,
    train_cfg: Path,
    env_cfg: Path,
    seed: int,
    deterministic: bool,
    toy: bool,
) -> dict:
    if toy:
        rng = np.random.default_rng(seed)
        loc = 100.5 if deterministic else 100.0
        rewards = rng.normal(loc=loc, scale=5.0, size=16)
        return {
            "seed": int(seed),
            "mean_reward": float(np.mean(rewards)),
            "final_reward": float(rewards[-1]),
            "status": "ok",
            "mode": "toy",
        }

    arm_name = train_cfg.stem.replace('training_config_', '')
    run_root = repo_root / "artifacts" / "runs" / arm_name / f"seed_{seed}"
    result_path = run_root / "result.json"
    log_path = run_root / "train_metrics.csv"
    checkpoint_dir = run_root / "checkpoints"

    cmd = [
        "python",
        "-m",
        "experiments.train_mappo",
        "--training-config",
        str(train_cfg),
        "--env-config",
        str(env_cfg),
        "--seed",
        str(seed),
        "--checkpoint-dir",
        str(checkpoint_dir),
        "--log-path",
        str(log_path),
        "--result-json",
        str(result_path),
    ]
    if deterministic:
        cmd.append("--deterministic")

    if result_path.exists():
        result_path.unlink()

    proc = subprocess.run(cmd, cwd=repo_root, capture_output=True, text=True)
    if proc.returncode != 0:
        return {
            "seed": int(seed),
            "status": "failed",
            "mode": "train",
            "return_code": int(proc.returncode),
            "stderr_tail": "\n".join(proc.stderr.splitlines()[-20:]),
        }

    if result_path.exists():
        try:
            payload = json.loads(result_path.read_text(encoding="utf-8"))
            if "final_reward" in payload:
                return {
                    "seed": int(seed),
                    "status": "ok",
                    "mode": "train",
                    "final_reward": float(payload["final_reward"]),
                    "mean_eval_reward": float(payload.get("mean_eval_reward", payload["final_reward"])),
                    "best_eval_reward": float(payload.get("best_eval_reward", payload["final_reward"])),
                    "n_eval_points": int(payload.get("n_eval_points", 0)),
                    "result_json": str(result_path),
                    "log_path": str(payload.get("log_path", log_path)),
                    "checkpoint_path": str(payload.get("checkpoint_path", checkpoint_dir / "mappo_final.pt")),
                }
        except Exception:
            pass

    marker = re.search(r"MULTISEED_FINAL_REWARD=([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", proc.stdout)
    if marker is not None:
        return {
            "seed": int(seed),
            "status": "ok",
            "mode": "train",
            "final_reward": float(marker.group(1)),
        }

    return {
        "seed": int(seed),
        "status": "failed",
        "mode": "train",
        "return_code": 0,
        "stderr_tail": "missing final_reward marker/json output",
    }


def run_sweep(
    train_cfg: Path,
    env_cfg: Path,
    seeds_path: Path,
    output_path: Path,
    deterministic: bool,
    toy: bool,
) -> dict:
    repo_root = Path(__file__).resolve().parents[1]
    seed_cfg = _load_yaml(seeds_path)
    seeds = [int(s) for s in seed_cfg.get("seeds", [])]
    if not seeds:
        raise ValueError(f"No seeds found in {seeds_path}")

    individual: list[dict] = []
    tracked = []
    for seed in seeds:
        result = _run_single(repo_root, train_cfg, env_cfg, seed, deterministic=deterministic, toy=toy)
        individual.append(result)
        if result.get("status") == "ok" and "final_reward" in result:
            tracked.append(float(result["final_reward"]))

    arr = np.asarray(tracked, dtype=np.float64) if tracked else np.asarray([], dtype=np.float64)
    mean_eval_values = [float(it["mean_eval_reward"]) for it in individual if it.get("status") == "ok" and "mean_eval_reward" in it]
    best_eval_values = [float(it["best_eval_reward"]) for it in individual if it.get("status") == "ok" and "best_eval_reward" in it]
    mean_eval_arr = np.asarray(mean_eval_values, dtype=np.float64) if mean_eval_values else np.asarray([], dtype=np.float64)
    best_eval_arr = np.asarray(best_eval_values, dtype=np.float64) if best_eval_values else np.asarray([], dtype=np.float64)

    payload = {
        "train_config": str(train_cfg),
        "env_config": str(env_cfg),
        "seeds": seeds,
        "deterministic": bool(deterministic),
        "mode": "toy" if toy else "train",
        "individual": individual,
        "metric": "final_reward",
        "mean": float(np.mean(arr)) if arr.size else None,
        "std": float(np.std(arr)) if arr.size else None,
        "mean_eval_reward_mean": float(np.mean(mean_eval_arr)) if mean_eval_arr.size else None,
        "mean_eval_reward_std": float(np.std(mean_eval_arr)) if mean_eval_arr.size else None,
        "best_eval_reward_mean": float(np.mean(best_eval_arr)) if best_eval_arr.size else None,
        "best_eval_reward_std": float(np.std(best_eval_arr)) if best_eval_arr.size else None,
        "n_success": int(arr.size),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MAPPO multi-seed sweep and save summary JSON.")
    parser.add_argument("--training-config", type=Path, default=Path("configs/training_config.yaml"))
    parser.add_argument("--env-config", type=Path, default=Path("configs/env_config.yaml"))
    parser.add_argument("--seeds", type=Path, default=Path("configs/seeds.yaml"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/multi_seed/latest.json"))
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--toy", action="store_true", help="Use synthetic rewards to validate flow without full training.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    payload = run_sweep(
        train_cfg=args.training_config,
        env_cfg=args.env_config,
        seeds_path=args.seeds,
        output_path=args.output,
        deterministic=args.deterministic,
        toy=args.toy,
    )
    print(
        f"Multi-seed complete: n_success={payload['n_success']}/{len(payload['seeds'])}, "
        f"mean={payload['mean']}, std={payload['std']}"
    )


if __name__ == "__main__":
    main()
