"""Directional IAE check for the frequency skip-connection fix.

Evaluates a short-trained mappo_dual checkpoint against the env's built-in droop
baseline on identical forced contingencies, in identical units
(sum|delta_f[30:]| over a 120-step rollout), mirroring run_asha_am_mappo._eval_iae.

Purpose: confirm the trained policy's IAE drops to droop-competitive range rather
than the broken ~35 seen before the fix. NOT a production benchmark.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.env.microgrid_env_dual import MicrogridEnvDual
from src.env.events import EventConfig
from src.rl.train_am_mappo import build_am_full_feeder_obs, ensure_edge_index

# Same scenarios/topos as run_asha_am_mappo._eval_iae
_SCENARIOS = [
    ("load_step", 2.5, 45),
    ("gen_trip", -3.9, 67),
    ("high_ren", 4.7, 105),
]
_TOPOS = [0, 8]


def _iae_rollout(env, act_fn, n_steps: int = 120) -> float:
    iaes = []
    for et, emw, eloc in _SCENARIOS:
        for topo in _TOPOS:
            ev = EventConfig(type=et, delta_P_mw=emw, location=eloc, t_inject=30.0)
            obs_fast, _, _ = env.reset(
                seed=42, options={"force_event": deepcopy(ev), "force_topology": int(topo)}
            )
            n_bus = int(len(env.net.bus.index))
            df = []
            for _ in range(n_steps):
                fa = act_fn(env, obs_fast, n_bus)
                obs_fast, _, _, _, info = env.step_fast(fa)
                df.append(info["delta_f"])
            df = np.asarray(df)
            iaes.append(float(np.sum(np.abs(df[30:]))))
    return float(np.mean(iaes)) if iaes else float("inf")


def eval_droop(args) -> float:
    """Env built-in droop, zero RL action (pure droop reference)."""
    env = MicrogridEnvDual(placement_path=args.placement, mpc_path=args.mpc_path)
    env.ffr_mode = "droop"
    n_vpps = len(env._vpp_droop_agents)
    action_dim = env.n_agents + n_vpps  # droop mode: a_P + VPP-K

    def act_fn(env, obs_fast, n_bus):
        return np.zeros(action_dim, dtype=np.float32)

    return _iae_rollout(env, act_fn)


def eval_checkpoint(args) -> float:
    from src.eval.eval_ffr_topology import GraphSAGEMAPPOPolicy

    env = MicrogridEnvDual(placement_path=args.placement, mpc_path=args.mpc_path)
    pol = GraphSAGEMAPPOPolicy(Path(args.checkpoint), env)
    env.ffr_mode = "mappo_dual"

    def act_fn(env, obs_fast, n_bus):
        of = build_am_full_feeder_obs(env, obs_fast)
        ei = ensure_edge_index(env.edge_index, n_nodes=n_bus)
        return pol.act(of, ei, env, obs_fast)

    return _iae_rollout(env, act_fn)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, default="artifacts/checkpoints_shorttest/am_mappo_final.pt")
    p.add_argument("--placement", type=str, default="artifacts/placement/official_placement_v3.json")
    p.add_argument("--mpc-path", type=str, default="data/grid_IEEE123_complete.m")
    args = p.parse_args()

    print("Evaluating droop baseline (reference)...")
    droop_iae = eval_droop(args)
    print(f"  Droop IAE          = {droop_iae:.3f}")

    print("Evaluating trained checkpoint (freq skip-connection)...")
    ckpt_iae = eval_checkpoint(args)
    print(f"  Trained IAE        = {ckpt_iae:.3f}")

    print("\n=== Result ===")
    print(f"  Droop (reference)  : {droop_iae:.3f}")
    print(f"  Trained (short)    : {ckpt_iae:.3f}")
    ratio = ckpt_iae / droop_iae if droop_iae > 1e-9 else float("inf")
    print(f"  Trained / Droop    : {ratio:.2f}x")
    if ratio < 1.5:
        print("  VERDICT: PASS — trained policy is droop-competitive (fix working).")
    elif ratio < 4.0:
        print("  VERDICT: PARTIAL — improved vs broken (~7x) but not yet competitive; more training needed.")
    else:
        print("  VERDICT: FAIL — IAE still far above droop; fix did not take effect.")


if __name__ == "__main__":
    main()
