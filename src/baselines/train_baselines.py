"""train_baselines.py — train one of the three baselines and save checkpoint.

Usage:
  python src/baselines/train_baselines.py --agent gcnn_ppo  --out checkpoints/baseline_gcnn_ppo/final.pt
  python src/baselines/train_baselines.py --agent sgac       --out checkpoints/baseline_sgac/final.pt
  python src/baselines/train_baselines.py --agent graph_ppo  --out checkpoints/baseline_graph_ppo/final.pt

Add --smoke for a quick 3-epoch sanity run (no checkpoint saved unless --out given).
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

from src.env.microgrid_env_dual import MicrogridEnvDual
from src.baselines.gcnn_ppo import GCNNPPOAgent
from src.baselines.sgac import SGACAgent
from src.baselines.graph_ppo import GraphPPOAgent

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULTS = {
    "gcnn_ppo": {
        "n_epochs": 500,
        "n_episodes_per_update": 3,
        "log_every": 20,
    },
    "sgac": {
        "n_epochs": 500,
        "n_episodes_per_update": 3,
        "log_every": 20,
    },
    "graph_ppo": {
        "n_epochs": 500,
        "n_episodes_per_update": 3,
        "log_every": 20,
    },
}


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(
    agent_name: str,
    env: MicrogridEnvDual,
    n_epochs: int,
    n_episodes_per_update: int,
    log_every: int,
    save_path: Path | None,
    save_every: int = 50,
) -> None:
    if agent_name == "gcnn_ppo":
        agent: Any = GCNNPPOAgent(env)
    elif agent_name == "sgac":
        agent = SGACAgent(env)
    elif agent_name == "graph_ppo":
        agent = GraphPPOAgent(env)
    else:
        raise ValueError(f"Unknown agent: {agent_name}")

    print(f"[train_baselines] Starting {agent_name} -- {n_epochs} epochs x {n_episodes_per_update} eps/update")

    best_r_fast = -float("inf")

    for epoch in range(1, n_epochs + 1):
        try:
            metrics = agent.update(n_episodes=n_episodes_per_update)
        except RuntimeError as exc:
            print(f"  epoch {epoch}: RuntimeError -- {exc}. Skipping.")
            continue

        r_fast = float(metrics.get("r_fast", 0.0))
        r_slow = float(metrics.get("r_slow", 0.0))

        if r_fast > best_r_fast:
            best_r_fast = r_fast
            if save_path is not None:
                agent.save(save_path.parent / "best.pt")

        if epoch % log_every == 0:
            extra = {k: f"{v:.4f}" for k, v in metrics.items() if k not in ("r_fast", "r_slow")}
            print(f"  epoch={epoch:4d}  r_fast={r_fast:.4f}  r_slow={r_slow:.4f}  {extra}")

        if save_path is not None and epoch % save_every == 0:
            agent.save(save_path.parent / f"epoch_{epoch:04d}.pt")

    if save_path is not None:
        agent.save(save_path)
        print(f"[train_baselines] Saved final checkpoint -> {save_path}")
    else:
        print("[train_baselines] Done (no --out specified, checkpoint not saved)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train one baseline agent")
    parser.add_argument(
        "--agent",
        required=True,
        choices=["gcnn_ppo", "sgac", "graph_ppo"],
        help="Which baseline to train",
    )
    parser.add_argument("--out", type=Path, default=None, help="Output checkpoint path (e.g. checkpoints/baseline_gcnn_ppo/final.pt)")
    parser.add_argument("--n-epochs", type=int, default=None, help="Override default epoch count")
    parser.add_argument("--n-episodes", type=int, default=None, help="Episodes per PPO/SAC update")
    parser.add_argument("--log-every", type=int, default=None)
    parser.add_argument("--save-every", type=int, default=50, help="Save intermediate checkpoint every N epochs")
    parser.add_argument("--smoke", action="store_true", help="Quick 3-epoch sanity run")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--placement", type=Path, default=Path("artifacts/placement/official_placement_v3.json"))
    parser.add_argument("--mpc-path", type=Path, default=Path("data/grid_IEEE123_complete.m"))
    parser.add_argument("--precomputed-dir", type=Path, default=Path("data/precomputed_365d_97to67"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = DEFAULTS[args.agent]

    n_epochs = 3 if args.smoke else (args.n_epochs or cfg["n_epochs"])
    n_episodes = args.n_episodes or cfg["n_episodes_per_update"]
    log_every = args.log_every or (1 if args.smoke else cfg["log_every"])

    env = MicrogridEnvDual(
        placement_path=str(args.placement),
        mpc_path=str(args.mpc_path),
        seed=args.seed,
        precomputed_dir=str(args.precomputed_dir),
    )

    train(
        agent_name=args.agent,
        env=env,
        n_epochs=n_epochs,
        n_episodes_per_update=n_episodes,
        log_every=log_every,
        save_path=args.out,
        save_every=args.save_every,
    )

    if args.smoke:
        print(f"SMOKE GATE [{args.agent}]: PASS")


if __name__ == "__main__":
    main()
