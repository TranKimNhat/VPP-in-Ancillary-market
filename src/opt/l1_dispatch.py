from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict
import time

import cvxpy as cp
import numpy as np
import yaml


@dataclass
class L1Result:
    p_ref: Dict[int, float]
    r_as: Dict[int, float]
    p_p2p: Dict[int, float]
    lambda_p2p: float
    revenue: Dict[int, float]
    solve_time: float
    status: str


class L1Dispatcher:
    """
    Per-step LP dispatch for 3 VPPs with LP-relaxed charge/discharge decisions.
    """

    def __init__(self) -> None:
        self.VPP_PARAMS = self._load_vpp_params()
        self.soc_state = {1: 0.5, 2: 0.5, 3: 0.5}

    @staticmethod
    def _load_vpp_params() -> dict[int, dict[str, float]]:
        cfg_path = Path(__file__).resolve().parents[2] / "configs" / "vpp_params.yaml"
        payload = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
        return {
            1: {k: float(v) for k, v in payload.get("vpp_1", {}).items()},
            2: {k: float(v) for k, v in payload.get("vpp_2", {}).items()},
            3: {k: float(v) for k, v in payload.get("vpp_3", {}).items()},
        }

    def _solve_vpp_lp(
        self,
        vpp_id: int,
        step: int,
        profiles: dict,
        l0_result,
        lambda_p2p: float,
        price_as: float,
    ) -> tuple[float, float, float, float, float, float]:
        params = self.VPP_PARAMS[vpp_id]
        soc = float(self.soc_state[vpp_id])
        pv_avail = float(params["pv_mw"]) * float(profiles.get("pv_pu", 1.0))

        P_ch = cp.Variable(nonneg=True)
        P_dis = cp.Variable(nonneg=True)
        P_v2g = cp.Variable(nonneg=True)
        R_commit = cp.Variable(nonneg=True)
        u = cp.Variable()

        constraints = [
            P_ch <= params["bess_mw"] * u,
            P_dis <= params["bess_mw"] * (1.0 - u),
            P_v2g <= params["v2g_mw"],
            u >= 0.0,
            u <= 1.0,
        ]

        P_net = pv_avail + P_dis + P_v2g - P_ch
        soc_next = soc + 0.25 * (
            params["bess_eta"] * P_ch / params["bess_mwh"]
            - P_dis / (params["bess_eta"] * params["bess_mwh"])
        )
        constraints += [soc_next >= 0.1, soc_next <= 0.9]

        S_agg = params["S_agg"]
        P_net_abs = cp.Variable(nonneg=True)
        constraints += [
            P_net_abs >= P_net,
            P_net_abs >= -P_net,
        ]
        # AM-only S_agg envelope: ||(P_net, R_commit)||_2 <= S_agg (Q dropped)
        constraints.append(cp.norm(cp.vstack([P_net_abs, R_commit]), 2) <= S_agg)
        P_flex_up = 0.5 * S_agg
        constraints.append(R_commit <= P_flex_up)

        deg_cost = params["deg_cost"]

        pv_pu = float(profiles.get("pv_pu", 0.0))
        pv_rated_vpp = float(params["pv_mw"])
        p_pv_signal = pv_pu * pv_rated_vpp

        load_total = float(profiles.get("load_total", 18.0))
        load_ratio = min(load_total / 26.0, 1.0)

        bess_rated_vpp = float(params["bess_mw"])
        p_ref_target = p_pv_signal + bess_rated_vpp * 0.3 * load_ratio

        w_track = 1.0
        objective = cp.Maximize(
            lambda_p2p * P_net
            + price_as * R_commit
            - deg_cost * (P_ch + P_dis)
            - w_track * cp.sum_squares(P_net - p_ref_target)
        )
        prob = cp.Problem(objective, constraints)
        prob.solve(solver=cp.MOSEK, verbose=False)

        if prob.status not in {"optimal", "optimal_inaccurate"}:
            return pv_avail, 0.0, 0.0, 0.0, soc

        p_val = float(P_net.value)
        r_val = float(R_commit.value)

        if p_val < 0:
            p_pv_avail = pv_avail
            p_curt = 0.0
            p_dis = float(P_dis.value)
            p_ch = float(P_ch.value)
            p_v2g = float(P_v2g.value)
            P_commit = 0.0
            if hasattr(l0_result, "vpp_capacities") and vpp_id in l0_result.vpp_capacities:
                P_commit = float(l0_result.vpp_capacities[vpp_id].get("P_commit", 0.0))
                S_agg = float(l0_result.vpp_capacities[vpp_id].get("S_agg", S_agg))

            print(f"\n=== p_ref NEGATIVE DEBUG (VPP{vpp_id}, step={step}) ===")
            print(f"  p_pv_avail    = {p_pv_avail:.4f} MW  (pv_pu={profiles.get('pv_pu', 0.0):.3f} x rated)")
            print(f"  p_curt        = {p_curt:.4f} MW  (curtailment)")
            print(f"  p_dis         = {p_dis:.4f} MW  (BESS discharge)")
            print(f"  p_ch          = {p_ch:.4f} MW  (BESS charge)")
            print(f"  p_v2g         = {p_v2g:.4f} MW  (V2G discharge)")
            print(f"  p_dispatch    = {p_val:.4f} MW")
            print("  Formula: p_dispatch = p_pv_avail - p_curt + p_dis - p_ch + p_v2g")
            print(
                f"  Verified: {p_pv_avail:.4f} - {p_curt:.4f} + {p_dis:.4f} - {p_ch:.4f} + {p_v2g:.4f} = "
                f"{p_pv_avail - p_curt + p_dis - p_ch + p_v2g:.4f}"
            )
            print(f"  lambda_p2p    = {profiles.get('lambda_p2p', 0.0):.3f} $/MWh")
            print(f"  lambda_as_ffr = {profiles.get('lambda_as_ffr', 0.0):.3f} $/MWh")
            print(f"  P_commit      = {P_commit:.4f} MW  (L0 reference)")
            print(f"  S_agg         = {S_agg:.4f} MVA")
            print(f"  r_as          = {r_val:.4f} MW")
            print("=========================================\n")

        soc_next_val = float(np.clip(soc_next.value, 0.1, 0.9))
        revenue_val = lambda_p2p * p_val + price_as * r_val
        return p_val, r_val, p_val, revenue_val, soc_next_val

    def solve_step(
        self,
        step: int,
        profiles: dict,
        l0_result,
        vpp_id: int = 1,
        lambda_p2p_init: float = 5.0,
        admm_rho: float = 1.0,
        admm_max_iter: int = 30,
    ) -> L1Result:
        t0 = time.time()

        p_ref: Dict[int, float] = {}
        r_as: Dict[int, float] = {}
        p_p2p: Dict[int, float] = {}
        revenue: Dict[int, float] = {}

        lambda_p2p = float(lambda_p2p_init)
        vpp_zone = {1: 1, 2: 2, 3: 4}
        vpp_ids = [vpp_id]

        for _iter in range(admm_max_iter):
            p_balance = 0.0
            for vpp_id in vpp_ids:
                params = self.VPP_PARAMS[vpp_id]
                soc = float(self.soc_state[vpp_id])
                pv_avail = float(params["pv_mw"]) * float(profiles.get("pv_pu", 1.0))

                P_ch = cp.Variable(nonneg=True)
                P_dis = cp.Variable(nonneg=True)
                P_v2g = cp.Variable(nonneg=True)
                R_commit = cp.Variable(nonneg=True)
                u = cp.Variable()  # relaxed [0,1]

                constraints = [
                    P_ch <= params["bess_mw"] * u,
                    P_dis <= params["bess_mw"] * (1.0 - u),
                    P_v2g <= params["v2g_mw"],
                    u >= 0.0,
                    u <= 1.0,
                ]

                P_net = pv_avail + P_dis + P_v2g - P_ch
                soc_next = soc + 0.25 * (
                    params["bess_eta"] * P_ch / params["bess_mwh"]
                    - P_dis / (params["bess_eta"] * params["bess_mwh"])
                )
                constraints += [soc_next >= 0.1, soc_next <= 0.9]

                S_agg = params["S_agg"]
                P_net_abs = cp.Variable(nonneg=True)
                constraints += [
                    P_net_abs >= P_net,
                    P_net_abs >= -P_net,
                ]
                # AM-only S_agg envelope: ||(P_net, R_commit)||_2 <= S_agg (Q dropped)
                constraints.append(cp.norm(cp.vstack([P_net_abs, R_commit]), 2) <= S_agg)
                P_flex_up = 0.5 * S_agg
                constraints.append(R_commit <= P_flex_up)

                deg_cost = params["deg_cost"]
                zone = vpp_zone.get(vpp_id)
                price_as = getattr(
                    l0_result,
                    f"lambda_as_z{zone}",
                    float(profiles.get("lambda_as_ffr", l0_result.lambda_as.get("ffr", 0.0))),
                )

                pv_pu = float(profiles.get("pv_pu", 0.0))
                pv_rated_vpp = float(params["pv_mw"])
                p_pv_signal = pv_pu * pv_rated_vpp

                load_total = float(profiles.get("load_total", 18.0))
                load_ratio = min(load_total / 26.0, 1.0)

                bess_rated_vpp = float(params["bess_mw"])
                p_ref_target = p_pv_signal + bess_rated_vpp * 0.3 * load_ratio

                zone_lambda_p2p = getattr(
                    l0_result,
                    f"lambda_p2p_z{zone}",
                    float(profiles.get("lambda_p2p", l0_result.lambda_p2p)),
                )

                w_track = 1.0
                objective = cp.Maximize(
                    zone_lambda_p2p * P_net
                    + price_as * R_commit
                    - deg_cost * (P_ch + P_dis)
                    - w_track * cp.sum_squares(P_net - p_ref_target)
                )
                prob = cp.Problem(objective, constraints)
                prob.solve(solver=cp.MOSEK, verbose=False)

                if prob.status not in {"optimal", "optimal_inaccurate"}:
                    p_ref[vpp_id] = pv_avail
                    r_as[vpp_id] = 0.0
                    p_p2p[vpp_id] = 0.0
                    revenue[vpp_id] = 0.0
                    continue

                p_val = float(P_net.value)
                r_val = float(R_commit.value)

                p_ref[vpp_id] = p_val
                r_as[vpp_id] = r_val
                p_p2p[vpp_id] = p_val
                revenue[vpp_id] = zone_lambda_p2p * p_val + price_as * r_val

                self.soc_state[vpp_id] = float(np.clip(soc_next.value, 0.1, 0.9))
                p_balance += p_val

            if abs(p_balance) < 1e-3:
                break
            lambda_p2p -= admm_rho * p_balance
            lambda_p2p = float(np.clip(lambda_p2p, 3.0, 8.0))

        solve_time = time.time() - t0
        status = "optimal" if p_ref else "failed"

        return L1Result(
            p_ref=p_ref,
            r_as=r_as,
            p_p2p=p_p2p,
            lambda_p2p=lambda_p2p,
            revenue=revenue,
            solve_time=solve_time,
            status=status,
        )
