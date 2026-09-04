"""Try stronger conic solvers (MOSEK / CLARABEL) + multiple eps for the CLF LMI.
For each result we VERIFY the true decay rate eta from the returned P; only
eta>0 is a valid exponential-stability certificate."""
from __future__ import annotations
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import numpy as np
import cvxpy as cp
from experiments.lyapunov_certificate import (
    MPC, PLACEMENT, collect_A_matrices, decay_rate,
)
from src.env.microgrid_env_dual import MicrogridEnvDual

print("Installed CVXPY solvers:", cp.installed_solvers())

env = MicrogridEnvDual(placement_path=PLACEMENT, mpc_path=MPC)
n_topo = len(getattr(env.reconfig, "_cache", []))
A_list, _ = collect_A_matrices(env, list(range(max(n_topo, 1))))
n = A_list[0].shape[0]
print(f"{len(A_list)} A-matrices, state dim {n}")


def solve_clf(solver, eps):
    P = cp.Variable((n, n), symmetric=True)
    cons = [P >> np.eye(n)]
    for A in A_list:
        cons.append(A.T @ P + P @ A << -eps * np.eye(n))
    prob = cp.Problem(cp.Minimize(cp.trace(P)), cons)
    try:
        kw = {}
        if solver == "MOSEK":
            kw = {"verbose": False}
        prob.solve(solver=getattr(cp, solver), **kw)
    except Exception as e:
        return None, f"ERR:{type(e).__name__}", None, None
    if P.value is None:
        return None, prob.status, None, None
    eta = decay_rate(P.value, A_list)
    condP = float(np.linalg.cond(P.value))
    return P.value, prob.status, eta, condP


candidates = [s for s in ["MOSEK", "CLARABEL", "SCS"] if s in cp.installed_solvers()]
print("\nTrying:", candidates)
for solver in candidates:
    for eps in [1e-3, 1e-5, 1e-7]:
        Pv, status, eta, condP = solve_clf(solver, eps)
        if Pv is None:
            print(f"  {solver:9s} eps={eps:.0e}: status={status}")
        else:
            ok = "VALID (eta>0)" if (eta is not None and eta > 0) else "INVALID (eta<=0)"
            print(f"  {solver:9s} eps={eps:.0e}: status={status:20s} "
                  f"verified eta={eta:+.3e}  cond(P)={condP:.2e}  -> {ok}")
