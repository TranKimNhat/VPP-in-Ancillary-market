"""Canonical ASHA for MARL hyperparameter search.

Implements Li et al. 2018 "A System for Massively Parallel Hyperparameter Tuning",
Algorithm 1 (ASHA) + get_job() subroutine verbatim.

Key parameters (Li et al. notation):
    r   -- minimum resource per config (episodes at rung 0)
    R   -- maximum resource (episodes at top rung)
    eta -- halving/promotion factor (≥2, typically 3 or 4)
    s   -- minimum early-stopping rate / bracket index offset (default 0)
    K   -- number of rungs = floor(log_eta(R/r)) - s
           rung k has resource r_k = r * eta^(s+k),  k ∈ {0, …, K}

Rung structure:
    rungs[k] = list of {"trial_id", "params", "score", "checkpoint", "promoted"}
    • "promoted" flag marks configs already moved to rung k+1 so they are
      never promoted twice (canonical invariant from get_job).

Promotion rule (canonical):
    top_k(rung, |rung| / eta) — lower score is better (validation loss).
    A config is promotable iff it is in top-1/eta AND not yet promoted.

Workers:
    ThreadPoolExecutor(max_workers=n_workers).
    Each future calls _run_job() → subprocess train_am_mappo → result dict.
    Completed jobs update the rung; the scheduler immediately calls get_job()
    for that freed worker (asynchronous, not stage-gated).

Multi-bracket (optional):
    Run s_max+1 independent brackets (s=0…s_max), each with its own rungs.
    Configs are allocated round-robin across brackets (Li et al. §3.2).
    Best result is taken across all brackets.

Checkpoint continuity (promotion = continue, not restart):
    At rung k a checkpoint is saved as:
        <output>/<bracket_s>/trial_<id>/rung_<k>/am_mappo_final.pt
    When promoted to rung k+1 the trainer receives:
        --resume-from <rung_k_checkpoint>  --n-episodes <delta_episodes>
    so it extends from r_k to r_{k+1} without retraining from scratch.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
import concurrent.futures
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import random


# ---------------------------------------------------------------------------
# Resolve project-local Python interpreter (.venv preferred over sys.executable)
# ---------------------------------------------------------------------------

def _resolve_python() -> str:
    """Return .venv python if it exists next to this file's project root, else sys.executable."""
    project_root = Path(__file__).resolve().parent.parent
    for candidate in [
        project_root / ".venv" / "Scripts" / "python.exe",   # Windows
        project_root / ".venv" / "bin" / "python",           # Unix
    ]:
        if candidate.is_file():
            return str(candidate)
    return sys.executable


_PYTHON = _resolve_python()


# ---------------------------------------------------------------------------
# Search space
# ---------------------------------------------------------------------------

def _sample_config(rng: random.Random) -> dict[str, Any]:
    """Sample a hyperparameter configuration for GA-MAPPO (AM dual-action)."""
    return {
        "lr": 10 ** rng.uniform(-4.5, -3.0),
        "entropy_coef": rng.choice([0.001, 0.003, 0.01, 0.03]),
        "embed_dim": rng.choice([64, 128]),
        "hidden_dim": rng.choice([64, 128]),
        "update_epochs": rng.choice([2, 4, 8]),
        "mini_batch_size": rng.choice([16, 32, 64]),
        "log_std_init": rng.choice([-2.0, -1.5, -1.0, -0.5, 0.0]),
    }


# ---------------------------------------------------------------------------
# Objective (lower = better; treated as validation loss by ASHA)
# ---------------------------------------------------------------------------

def _objective(result: dict[str, Any]) -> float:
    max_abs_delta_f   = float(result.get("max_abs_delta_f",    np.inf))
    violation_fraction = float(result.get("violation_fraction", np.inf))
    mean_rocof        = float(result.get("mean_rocof",          np.inf))
    mean_nadir        = float(result.get("mean_nadir",          0.0))
    return max_abs_delta_f + 2.0 * violation_fraction + 0.2 * mean_rocof - 0.02 * mean_nadir


# ---------------------------------------------------------------------------
# Single job runner (called by ThreadPoolExecutor workers)
# ---------------------------------------------------------------------------

def _run_job(
    *,
    trial_id: int,
    bracket_s: int,
    rung_k: int,
    resource_episodes: int,       # cumulative budget at this rung
    delta_episodes: int,          # incremental budget = resource_k - resource_{k-1}
    seeds: list[int],
    params: dict[str, Any],
    resume_from: str | None,      # path to prior rung checkpoint (or None at rung 0)
    output_dir: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Run train_am_mappo for `delta_episodes`, optionally resuming, for every seed.

    Returns a result dict compatible with ASHA rung entries:
        {trial_id, bracket_s, rung_k, resource_episodes, params, score, score_std,
         checkpoint, per_seed}
    """
    per_seed: list[dict[str, Any]] = []

    for seed in seeds:
        rung_dir   = output_dir / f"bracket_{bracket_s}" / f"trial_{trial_id}" / f"rung_{rung_k}" / f"seed_{seed}"
        ckpt_dir   = rung_dir / "checkpoints"
        result_json = rung_dir / "result.json"
        rung_dir.mkdir(parents=True, exist_ok=True)

        cmd: list[str] = [
            _PYTHON, "-m", "src.rl.train_am_mappo",
            "--n-episodes",       str(delta_episodes),
            "--steps-per-episode", str(args.steps_per_episode),
            "--seed",             str(seed),
            "--placement",        args.placement,
            "--mpc-path",         args.mpc_path,
            "--checkpoint-dir",   str(ckpt_dir),
            "--log-interval",     str(args.log_interval),
            "--lr",               str(params["lr"]),
            "--entropy-coef",     str(params["entropy_coef"]),
            "--embed-dim",        str(params["embed_dim"]),
            "--hidden-dim",       str(params["hidden_dim"]),
            "--update-epochs",    str(params["update_epochs"]),
            "--mini-batch-size",  str(params["mini_batch_size"]),
            "--log-std-init",     str(params["log_std_init"]),
            "--ffr-mode",         args.ffr_mode,
            "--result-json",      str(result_json),
        ]
        if resume_from is not None:
            # Per-seed resume: look for seed-matched checkpoint; fall back to seed_42
            seed_ckpt = Path(resume_from).parent.parent / f"seed_{seed}" / "checkpoints" / "am_mappo_final.pt"
            fallback   = Path(resume_from)
            cmd += ["--resume-from", str(seed_ckpt if seed_ckpt.is_file() else fallback)]

        subprocess.run(cmd, check=True)
        result = json.loads(result_json.read_text(encoding="utf-8"))
        score  = _objective(result)
        per_seed.append({"seed": seed, "metrics": result, "score": float(score)})

    scores     = [x["score"] for x in per_seed]
    mean_score = float(np.mean(scores))

    # Canonical checkpoint path for this trial at this rung (use seed_42 as canonical)
    canonical_seed = seeds[0]
    checkpoint = str(
        output_dir
        / f"bracket_{bracket_s}" / f"trial_{trial_id}" / f"rung_{rung_k}"
        / f"seed_{canonical_seed}" / "checkpoints" / "am_mappo_final.pt"
    )

    return {
        "trial_id":          trial_id,
        "bracket_s":         bracket_s,
        "rung_k":            rung_k,
        "resource_episodes": resource_episodes,
        "params":            params,
        "score":             mean_score,
        "score_std":         float(np.std(scores)),
        "checkpoint":        checkpoint,
        "per_seed":          per_seed,
    }


# ---------------------------------------------------------------------------
# ASHA rung / state
# ---------------------------------------------------------------------------

class _Bracket:
    """One ASHA bracket (one value of s).

    Rungs indexed 0 … K.  Rung k holds configs trained for r * eta^(s+k) episodes.
    """

    def __init__(self, s: int, r: int, R: int, eta: int) -> None:
        self.s   = s
        self.r   = r
        self.R   = R
        self.eta = eta
        # K = floor(log_eta(R/r)) - s.  Must be >= 0.
        self.K   = max(0, math.floor(math.log(R / r) / math.log(eta)) - s)
        self.resource: dict[int, int] = {
            k: round(r * (eta ** (s + k))) for k in range(self.K + 1)
        }
        # rungs[k] = list of result dicts (from _run_job)
        self.rungs: dict[int, list[dict[str, Any]]] = {k: [] for k in range(self.K + 1)}
        # promoted_ids[k] = set of trial_ids already moved from rung k to k+1
        self.promoted_ids: dict[int, set[int]] = {k: set() for k in range(self.K + 1)}

    # -----------------------------------------------------------------------
    # Core ASHA subroutine: get_job()  (Li et al. 2018, Algorithm 2)
    # -----------------------------------------------------------------------
    def get_job(self) -> tuple[int | None, dict[str, Any] | None, int, str | None]:
        """Return (trial_id, params, target_rung_k, resume_from_checkpoint).

        Scans rungs from K-1 down to 0 for a promotable config.
        If found, returns the config + rung k+1 (promotion).
        Otherwise returns (None, None, 0, None) meaning "sample a new config".
        """
        for k in range(self.K - 1, -1, -1):          # K-1 downto 0
            rung     = self.rungs[k]
            n_rung   = len(rung)
            n_top    = max(1, n_rung // self.eta)     # top-1/eta
            if n_rung < self.eta:                      # not enough data yet
                continue

            # Sort by score ascending (lower = better); take top n_top
            sorted_rung = sorted(rung, key=lambda x: x["score"])
            top_set     = {r["trial_id"] for r in sorted_rung[:n_top]}
            already     = self.promoted_ids[k]
            promotable  = [r for r in sorted_rung[:n_top] if r["trial_id"] not in already]

            if promotable:
                entry = promotable[0]          # oldest (first inserted) among top
                self.promoted_ids[k].add(entry["trial_id"])
                return (
                    entry["trial_id"],
                    entry["params"],
                    k + 1,
                    entry["checkpoint"],       # resume from rung-k checkpoint
                )

        # No promotion available → assign new config to rung 0
        return (None, None, 0, None)

    def has_promotable(self) -> bool:
        """Pure check — no mutation. True if any rung has a config in top-1/eta not yet promoted."""
        for k in range(self.K - 1, -1, -1):
            n_rung = len(self.rungs[k])
            if n_rung < self.eta:
                continue
            n_top   = max(1, n_rung // self.eta)
            top_ids = {
                r["trial_id"]
                for r in sorted(self.rungs[k], key=lambda x: x["score"])[:n_top]
            }
            if top_ids - self.promoted_ids[k]:
                return True
        return False

    def add_result(self, result: dict[str, Any]) -> None:
        k = result["rung_k"]
        self.rungs[k].append(result)

    def to_dict(self) -> dict[str, Any]:
        return {
            "s":       self.s,
            "K":       self.K,
            "resource": self.resource,
            "rungs":   {
                str(k): v for k, v in self.rungs.items()
            },
            "promoted_ids": {
                str(k): list(v) for k, v in self.promoted_ids.items()
            },
        }


# ---------------------------------------------------------------------------
# Main ASHA loop
# ---------------------------------------------------------------------------

def run_asha(args: argparse.Namespace) -> dict[str, Any]:
    rng        = random.Random(args.random_seed)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build brackets: s = 0 … s_max
    brackets = [_Bracket(s=s, r=args.r_min, R=args.R_max, eta=args.eta)
                for s in range(args.s_max + 1)]

    for b in brackets:
        print(
            f"Bracket s={b.s}: K={b.K} rungs, "
            f"resources={list(b.resource.values())} episodes"
        )

    seeds_per_trial = [args.seed + i for i in range(args.seeds_per_trial)]
    trial_counter   = 0          # global NEW-config counter
    all_results: list[dict[str, Any]] = []
    # pending: live futures → metadata dict
    pending: dict[Future[dict[str, Any]], dict[str, Any]] = {}
    bracket_cycle = 0

    summary_path = output_dir / "asha_summary.json"

    def _save_summary() -> None:
        payload = {
            "config":      vars(args),
            "brackets":    [b.to_dict() for b in brackets],
            "all_results": all_results,
        }
        summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _try_dispatch(executor: ThreadPoolExecutor) -> bool:
        """Pick next job via get_job(), submit it, add to `pending`. Returns False if nothing to do."""
        nonlocal trial_counter, bracket_cycle

        # Round-robin across brackets to pick the one that has work
        for _ in range(len(brackets)):
            b = brackets[bracket_cycle % len(brackets)]
            bracket_cycle += 1

            trial_id, params, target_k, resume_from = b.get_job()
            is_new = trial_id is None

            if is_new:
                if trial_counter >= args.max_configs:
                    continue                      # this bracket has no work; try next
                trial_id = trial_counter
                params   = _sample_config(rng)
                trial_counter += 1
            # else: it's a promotion — always valid

            r_k   = b.resource[target_k]
            r_prev = b.resource[target_k - 1] if target_k > 0 else 0
            delta  = r_k - r_prev

            print(
                f"[{time.strftime('%H:%M:%S')}] "
                f"Dispatch trial={trial_id} bracket_s={b.s} "
                f"rung={target_k} r={r_k} (d={delta})"
                + (" [PROMOTE]" if not is_new else " [NEW]")
            )

            fut = executor.submit(
                _run_job,
                trial_id=trial_id,
                bracket_s=b.s,
                rung_k=target_k,
                resource_episodes=r_k,
                delta_episodes=delta,
                seeds=seeds_per_trial,
                params=params,
                resume_from=resume_from,
                output_dir=output_dir,
                args=args,
            )
            pending[fut] = {"bracket": b, "trial_id": trial_id, "rung_k": target_k}
            return True

        return False   # no bracket had any work

    def _has_work() -> bool:
        if trial_counter < args.max_configs:
            return True
        return any(b.has_promotable() for b in brackets)

    # -------------------------------------------------------------------
    # Asynchronous main loop — Li et al. 2018 Algorithm 1
    # Uses wait(FIRST_COMPLETED) so new futures added after startup are
    # included, unlike as_completed() which freezes the initial set.
    # -------------------------------------------------------------------
    with ThreadPoolExecutor(max_workers=args.n_workers) as executor:

        # Fill initial worker slots
        while len(pending) < args.n_workers and _has_work():
            if not _try_dispatch(executor):
                break

        # Drain: wait for one completion, re-dispatch, repeat
        while pending:
            done, _ = concurrent.futures.wait(pending, return_when=concurrent.futures.FIRST_COMPLETED)

            for fut in done:
                meta   = pending.pop(fut)
                result = fut.result()             # re-raises subprocess exception

                meta["bracket"].add_result(result)
                all_results.append(result)

                score_str = f"{result['score']:.6f}" + (
                    f" ±{result['score_std']:.4f}" if args.seeds_per_trial > 1 else ""
                )
                print(
                    f"[{time.strftime('%H:%M:%S')}] "
                    f"Done  trial={result['trial_id']} "
                    f"bracket_s={result['bracket_s']} rung={result['rung_k']} "
                    f"r={result['resource_episodes']} score={score_str}"
                )
                _save_summary()

                # Re-fill the freed worker slot
                if _has_work() and len(pending) < args.n_workers:
                    _try_dispatch(executor)

    _save_summary()

    best = min(all_results, key=lambda x: x["score"]) if all_results else None
    if best:
        print(
            f"\nBest: trial={best['trial_id']} bracket_s={best['bracket_s']} "
            f"rung={best['rung_k']} r={best['resource_episodes']} "
            f"score={best['score']:.6f}"
        )
        print(f"Best params: {json.dumps(best['params'], indent=2)}")
        print(f"Best checkpoint: {best['checkpoint']}")

    return {"config": vars(args), "best": best, "all_results": all_results}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Canonical ASHA (Li et al. 2018 Alg.1+2) for GA-MAPPO hyperparameter search.\n"
            "rung k resource = r_min * eta^(s+k);  K = floor(log_eta(R/r)) - s."
        )
    )
    # --- ASHA algorithm parameters ---
    parser.add_argument("--r-min",    type=int,   default=20,
                        help="Minimum resource (episodes at rung 0).")
    parser.add_argument("--R-max",    type=int,   default=540,
                        help="Maximum resource (episodes at top rung).")
    parser.add_argument("--eta",      type=int,   default=3,
                        help="Halving/promotion factor (default 3).")
    parser.add_argument("--s-max",    type=int,   default=0,
                        help="Maximum bracket index (0 = single bracket, 1 = two brackets, …).")
    parser.add_argument("--max-configs", type=int, default=27,
                        help="Total NEW configs to sample (budget; does not count promotions).")
    parser.add_argument("--n-workers",   type=int, default=3,
                        help="Number of parallel workers (ThreadPoolExecutor).")
    parser.add_argument("--seeds-per-trial", type=int, default=1,
                        help="Seeds per trial per rung (1 = canonical ASHA; >1 for noise reduction).")
    parser.add_argument("--seed",        type=int, default=42,
                        help="Base seed for trial seeds (seed+0, seed+1, …).")
    parser.add_argument("--random-seed", type=int, default=123,
                        help="RNG seed for config sampling.")
    # --- Trainer pass-through ---
    parser.add_argument("--steps-per-episode", type=int, default=300)
    parser.add_argument("--placement", type=str,
                        default="artifacts/placement/official_placement_v3.json")
    parser.add_argument("--mpc-path",  type=str,
                        default="data/grid_IEEE123_complete.m")
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument(
        "--ffr-mode", type=str, default="mappo_dual",
        choices=["droop", "mappo", "mappo_dual"],
    )
    # --- Output ---
    parser.add_argument("--output", type=str, default="artifacts/asha_canonical",
                        help="Directory for results and checkpoints.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    # Validate: at least one complete rung must exist
    K_bracket0 = max(0, math.floor(math.log(args.R_max / args.r_min) / math.log(args.eta)))
    if K_bracket0 < 1:
        raise ValueError(
            f"R_max/r_min={args.R_max}/{args.r_min} with eta={args.eta} yields K={K_bracket0}. "
            "Need K≥1 (at least 2 rungs) for ASHA promotion to work."
        )

    print(
        f"ASHA: r={args.r_min}, R={args.R_max}, eta={args.eta}, "
        f"s_max={args.s_max}, max_configs={args.max_configs}, "
        f"n_workers={args.n_workers}, seeds_per_trial={args.seeds_per_trial}"
    )
    result = run_asha(args)


if __name__ == "__main__":
    main()
