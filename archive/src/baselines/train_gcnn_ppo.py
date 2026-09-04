"""train_gcnn_ppo.py — train GCNN-PPO baseline with optional curriculum.

Reuses AM_PHASES from src/rl/train_am_mappo.py for a fair comparison.

Usage:
  python src/baselines/train_gcnn_ppo.py --curriculum --n-episodes 6000 \
      --checkpoint-dir checkpoints/baseline_gcnn_ppo_v2 --seed 42
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from src.rl.train_am_mappo import AM_PHASES
from src.env.microgrid_env_dual import MicrogridEnvDual
from src.baselines.gcnn_ppo import GCNNPPOAgent, ensure_edge_index


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _apply_phase(envs: list[MicrogridEnvDual], phase_cfg: dict[str, Any]) -> None:
    probs = dict(phase_cfg["event_probs"])
    n_topo = phase_cfg.get("n_topologies")
    for env in envs:
        env.event_injector.set_probs(probs)
        if n_topo is not None:
            env.reconfig.set_active_topologies(int(n_topo))


def _rollout_and_update(
    agent: GCNNPPOAgent,
    envs: list[MicrogridEnvDual],
) -> tuple[float, float, dict[str, float]]:
    agent.buffer.clear()
    r_fasts, r_slows = [], []

    for env in envs:
        agent.env = env
        try:
            r_f, r_s = agent.rollout_episode()
        except Exception as exc:
            print(f"    [rollout error] {exc}")
            continue
        r_fasts.append(r_f)
        r_slows.append(r_s)

    if not r_fasts:
        agent.buffer.clear()
        return 0.0, 0.0, {}

    try:
        metrics = agent._update_model()
    except Exception as exc:
        print(f"    [update error] {exc}")
        metrics = {}
    agent.buffer.clear()

    return float(np.mean(r_fasts)), float(np.mean(r_slows)), metrics


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(args: argparse.Namespace) -> None:
    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    env_kwargs: dict[str, Any] = dict(
        placement_path=str(args.placement),
        mpc_path=str(args.mpc_path),
        precomputed_dir=str(args.precomputed_dir),
        ffr_mode="mappo_dual",  # same MDP as proposed; exercises the (a_P, a_K) dual action
    )
    envs = [MicrogridEnvDual(seed=args.seed + i, **env_kwargs) for i in range(args.n_envs)]
    for _e in envs:
        _e.fixed_base_topology = args.fixed_base_topology
    print(f"fixed_base_topology={args.fixed_base_topology}")
    agent = GCNNPPOAgent(envs[0], lr=args.lr, device=args.device)
    print(f"[gcnn_ppo] device={agent.device}")

    if args.curriculum:
        phase_items = list(AM_PHASES.items())
    else:
        phase_items = [
            ("FULL", {
                "n_episodes": args.n_episodes,
                "event_probs": {"load_step": 0.3, "gen_trip": 0.3, "line_trip": 0.2, "high_ren": 0.2},
                "lr_factor": 1.0,
            })
        ]

    best_r_fast = -float("inf")
    total_ep = 0

    for phase_name, cfg in phase_items:
        n_phase_eps = int(cfg["n_episodes"])
        lr_factor = float(cfg.get("lr_factor", 1.0))

        # Apply lr factor
        for pg in agent.optimizer.param_groups:
            pg["lr"] = args.lr * lr_factor

        _apply_phase(envs, cfg)
        print(f"\n=== Phase {phase_name} ({n_phase_eps} ep, lr_factor={lr_factor}) ===")

        n_updates = max(1, n_phase_eps // args.n_envs)
        for update in range(1, n_updates + 1):
            ep = update * args.n_envs
            total_ep += args.n_envs

            r_fast, r_slow, metrics = _rollout_and_update(agent, envs)

            if r_fast > best_r_fast:
                best_r_fast = r_fast
                agent.save(ckpt_dir / "best.pt")

            if ep % args.log_every == 0 or update == 1:
                loss = metrics.get("loss", float("nan"))
                ent = metrics.get("entropy", float("nan"))
                print(f"  phase={phase_name}  ep={ep:4d}/{n_phase_eps}"
                      f"  total={total_ep}  r_fast={r_fast:.4f}  r_slow={r_slow:.4f}"
                      f"  loss={loss:.2f}  entropy={ent:.2f}")

            if total_ep % args.save_every == 0:
                agent.save(ckpt_dir / "latest.pt")
                agent.save(ckpt_dir / f"ep_{total_ep:05d}.pt")

        agent.save(ckpt_dir / f"phase_{phase_name}_final.pt")
        print(f"  Saved phase checkpoint -> {ckpt_dir / f'phase_{phase_name}_final.pt'}")

    agent.save(ckpt_dir / "final.pt")
    print(f"\n[gcnn_ppo] Done. Final -> {ckpt_dir / 'final.pt'}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train GCNN-PPO baseline")
    parser.add_argument("--n-episodes", type=int, default=6000)
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--curriculum", action="store_true")
    parser.add_argument("--fixed-base-topology", action="store_true",
                        help="Train only on the nominal base feeder (no reconfiguration). "
                             "Used for the train-on-base, eval-on-all-reconfig generalization protocol.")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints/baseline_gcnn_ppo_v2"))
    parser.add_argument("--placement", type=Path, default=Path("artifacts/placement/official_placement_v3.json"))
    parser.add_argument("--mpc-path", type=Path, default=Path("data/grid_IEEE123_complete.m"))
    parser.add_argument("--precomputed-dir", type=Path, default=Path("data/precomputed_365d_97to67"))
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--device", type=str, default=None,
                        help="cuda | cpu | None(auto). Note: env sim is CPU-bound, so GPU util stays low.")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
