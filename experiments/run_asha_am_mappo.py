from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import subprocess
import sys
from typing import Any

import numpy as np


def _sample_config(rng: random.Random) -> dict[str, Any]:
    """Sample HP config for dual-action MAPPO ASHA search.

    Search space widened post-softplus fix (former 0.5 clamp killed log_std gradient):
    - LR upper bound raised from 5e-4 to 1e-3 (slow learning observed at 3e-5)
    - log_std_init: controls initial exploration; softplus(x)+0.05 gives initial std
      e.g. x=-2.0 -> std≈0.17, x=-1.0 -> 0.36, x=0.0 -> 0.74, x=+0.5 -> 1.03
    """
    return {
        "lr": 10 ** rng.uniform(-4.5, -3.0),
        "entropy_coef": rng.choice([0.001, 0.003, 0.01, 0.03]),
        "embed_dim": rng.choice([64, 128]),
        "hidden_dim": rng.choice([64, 128]),
        "update_epochs": rng.choice([2, 4, 8]),
        "mini_batch_size": rng.choice([16, 32, 64]),
        "log_std_init": rng.choice([-2.0, -1.5, -1.0, -0.5, 0.0]),
    }


# Eval-IAE objective: the metric that actually discriminates methods in the paper
# eval (post-event integral abs frequency error), measured on forced contingencies
# across a couple of topologies — far more informative than the training-time
# aggregates, and consistent with Approach-C (no RoCoF term; RoCoF is backbone-
# governed and not VPP-actionable). Lower is better.
_EVAL_SCENARIOS = [
    ("load_step", 2.5, 45),
    ("gen_trip", -3.9, 67),
    ("high_ren", 4.7, 105),
]
_EVAL_TOPOS = [0, 8]   # one from each former-bimodal group (post AGC-fix both fine)


def _eval_iae(checkpoint_path: Path, args: argparse.Namespace) -> float:
    """Mean post-event IAE over forced scenarios × topologies for a checkpoint."""
    from copy import deepcopy
    if not Path(checkpoint_path).exists():
        return float("inf")
    from src.env.microgrid_env_dual import MicrogridEnvDual
    from src.env.events import EventConfig
    from src.eval.eval_ffr_topology import GraphSAGEMAPPOPolicy
    from src.rl.train_am_mappo import build_am_full_feeder_obs, ensure_edge_index

    env = MicrogridEnvDual(placement_path=args.placement, mpc_path=args.mpc_path)
    try:
        pol = GraphSAGEMAPPOPolicy(Path(checkpoint_path), env)
    except Exception as exc:  # malformed/partial checkpoint -> worst score
        print(f"    [eval] load failed: {exc}")
        return float("inf")
    env.ffr_mode = "mappo_dual"
    iaes: list[float] = []
    for et, emw, eloc in _EVAL_SCENARIOS:
        for topo in _EVAL_TOPOS:
            ev = EventConfig(type=et, delta_P_mw=emw, location=eloc, t_inject=30.0)
            obs_fast, _, _ = env.reset(seed=42, options={"force_event": deepcopy(ev),
                                                          "force_topology": int(topo)})
            n_bus = int(len(env.net.bus.index))
            df = []
            for _ in range(120):
                of = build_am_full_feeder_obs(env, obs_fast)
                ei = ensure_edge_index(env.edge_index, n_nodes=n_bus)
                fa = pol.act(of, ei, env, obs_fast)
                obs_fast, _, _, _, info = env.step_fast(fa)
                df.append(info["delta_f"])
            df = np.asarray(df)
            iaes.append(float(np.sum(np.abs(df[30:]))))
    return float(np.mean(iaes)) if iaes else float("inf")


def _run_trial_single_seed(
    trial_id: int,
    stage: int,
    resource_episodes: int,
    seed: int,
    params: dict[str, Any],
    args: argparse.Namespace,
    output_dir: Path,
) -> dict[str, Any]:
    trial_dir = output_dir / f"trial_{trial_id}" / f"stage_{stage}" / f"seed_{seed}"
    ckpt_dir = trial_dir / "checkpoints"
    result_json = trial_dir / "result.json"
    trial_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        "-m",
        "src.rl.train_am_mappo",
        "--n-episodes",
        str(resource_episodes),
        "--steps-per-episode",
        str(args.steps_per_episode),
        "--seed",
        str(seed),
        "--placement",
        args.placement,
        "--mpc-path",
        args.mpc_path,
        "--checkpoint-dir",
        str(ckpt_dir),
        "--log-interval",
        str(args.log_interval),
        "--lr",
        str(params["lr"]),
        "--entropy-coef",
        str(params["entropy_coef"]),
        "--embed-dim",
        str(params["embed_dim"]),
        "--hidden-dim",
        str(params["hidden_dim"]),
        "--update-epochs",
        str(params["update_epochs"]),
        "--mini-batch-size",
        str(params["mini_batch_size"]),
        "--log-std-init",
        str(params["log_std_init"]),
        "--ffr-mode",
        args.ffr_mode,
        "--result-json",
        str(result_json),
    ]

    subprocess.run(cmd, check=True)
    result = json.loads(result_json.read_text(encoding="utf-8"))
    # Eval-IAE objective: load the trained checkpoint and score on forced
    # contingencies (the paper-relevant metric), not the training aggregates.
    final_ckpt = ckpt_dir / "am_mappo_final.pt"
    score = _eval_iae(final_ckpt, args)
    return {
        "seed": seed,
        "metrics": result,
        "eval_iae": float(score),
        "score": float(score),
        "result_json": str(result_json),
    }


def _run_trial_multi_seed(
    trial_id: int,
    stage: int,
    resource_episodes: int,
    seeds: list[int],
    params: dict[str, Any],
    args: argparse.Namespace,
    output_dir: Path,
) -> dict[str, Any]:
    per_seed: list[dict[str, Any]] = []
    for seed in seeds:
        per_seed.append(
            _run_trial_single_seed(
                trial_id=trial_id,
                stage=stage,
                resource_episodes=resource_episodes,
                seed=seed,
                params=params,
                args=args,
                output_dir=output_dir,
            )
        )

    scores = [x["score"] for x in per_seed]
    return {
        "trial_id": trial_id,
        "stage": stage,
        "resource_episodes": resource_episodes,
        "seeds": seeds,
        "params": params,
        "score": float(np.mean(scores)),
        "score_std": float(np.std(scores)),
        "per_seed": per_seed,
    }


def run_asha(args: argparse.Namespace) -> dict[str, Any]:
    rng = random.Random(args.random_seed)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    current: list[dict[str, Any]] = [
        {
            "trial_id": i,
            "params": _sample_config(rng),
        }
        for i in range(args.num_trials)
    ]

    all_results: list[dict[str, Any]] = []
    stage = 0
    resource = args.min_episodes

    while current and resource <= args.max_episodes:
        stage_results: list[dict[str, Any]] = []
        seeds_this_stage = [args.seed + i for i in range(min(stage + 1, args.max_seeds_per_stage))]
        for item in current:
            r = _run_trial_multi_seed(
                trial_id=item["trial_id"],
                stage=stage,
                resource_episodes=resource,
                seeds=seeds_this_stage,
                params=item["params"],
                args=args,
                output_dir=output_dir,
            )
            stage_results.append(r)
            all_results.append(r)

        stage_results.sort(key=lambda x: x["score"])  # lower is better

        keep = max(1, len(stage_results) // args.eta)
        promoted = stage_results[:keep]
        current = [{"trial_id": p["trial_id"], "params": p["params"]} for p in promoted]

        stage_summary = {
            "stage": stage,
            "resource_episodes": resource,
            "seeds": seeds_this_stage,
            "num_candidates": len(stage_results),
            "num_promoted": len(current),
            "best_trial_id": stage_results[0]["trial_id"],
            "best_score": stage_results[0]["score"],
            "best_score_std": stage_results[0]["score_std"],
        }
        print(json.dumps(stage_summary, ensure_ascii=False))

        stage += 1
        resource *= args.eta

    best = min(all_results, key=lambda x: x["score"]) if all_results else None
    payload = {
        "config": vars(args),
        "best": best,
        "all_results": all_results,
    }
    summary_path = output_dir / "asha_summary.json"
    summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ASHA runner for src.rl.train_am_mappo")
    parser.add_argument("--num-trials", type=int, default=12)
    parser.add_argument("--eta", type=int, default=3)
    parser.add_argument("--min-episodes", type=int, default=25)
    parser.add_argument("--max-episodes", type=int, default=500)
    parser.add_argument("--steps-per-episode", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--random-seed", type=int, default=123)
    parser.add_argument("--max-seeds-per-stage", type=int, default=3)
    parser.add_argument("--placement", type=str, default="artifacts/placement/official_placement_v3.json")
    parser.add_argument("--mpc-path", type=str, default="data/grid_IEEE123_complete.m")
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--output", type=str, default="artifacts/asha_am_mappo_dual")
    parser.add_argument(
        "--ffr-mode",
        type=str,
        default="mappo_dual",
        choices=["droop", "mappo", "mappo_dual"],
        help="FFR control mode passed to train_am_mappo (default: mappo_dual).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_asha(args)
    best = result.get("best")
    if best is None:
        print("No trials executed.")
        return
    print(
        f"Best trial={best['trial_id']} stage={best['stage']} episodes={best['resource_episodes']} "
        f"score={best['score']:.6f}"
    )


if __name__ == "__main__":
    main()
