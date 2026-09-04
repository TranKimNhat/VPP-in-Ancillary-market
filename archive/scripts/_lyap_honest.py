"""Honest Lyapunov readout: report worst Re(eig) (lambda_max), test whether a
TRUE common-P certificate holds (verified eta>0), and run the dwell-time
fallback for a defensible certificate given the near-marginal COI mode."""
from __future__ import annotations
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import numpy as np
from experiments.lyapunov_certificate import (
    MPC, PLACEMENT, collect_A_matrices, check_hurwitz, find_common_P,
    decay_rate, dwell_time_fallback,
)
from src.env.microgrid_env_dual import MicrogridEnvDual

env = MicrogridEnvDual(placement_path=PLACEMENT, mpc_path=MPC)
n_topo = len(getattr(env.reconfig, "_cache", []))
topo_ids = list(range(max(n_topo, 1)))
A_list, meta = collect_A_matrices(env, topo_ids)

worst, n_bad = check_hurwitz(A_list)
print(f"lambda_max (worst Re eig over {len(A_list)} vertex systems) = {worst:.6e}")
print(f"non-Hurwitz matrices = {n_bad}/{len(A_list)}")

P, status = find_common_P(A_list)
if P is None:
    print(f"CLF: INFEASIBLE (status={status})")
else:
    eta = decay_rate(P, A_list)
    print(f"CLF solver status = {status}; VERIFIED decay rate eta = {eta:.5f}")
    print(f"  -> CLF valid? {'YES' if eta > 0 else 'NO (eta<=0: certificate INVALID)'}")

print("\n--- Dwell-time / strictly-damped-subspace certificate ---")
dwell_time_fallback(env, topo_ids)
