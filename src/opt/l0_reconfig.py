from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List
import time

import cvxpy as cp
import numpy as np
import pandas as pd


@dataclass
class L0Result:
    alpha_sw: Dict[str, bool]
    lambda_p2p: float
    lambda_p2p_z1: float
    lambda_p2p_z2: float
    lambda_p2p_z4: float
    lambda_as: Dict[str, float]
    lambda_as_z1: float
    lambda_as_z2: float
    lambda_as_z4: float
    lambda_q: float
    r_as_req: Dict[int, float]
    q_target: Dict[int, float]
    v_profile: np.ndarray
    solve_time: float
    status: str
    zone_net_mw: Dict[int, float]


class L0Optimizer:
    """
    Layer 0: Hourly reconfiguration + market clearing via BFM-SOCP.
    """

    def __init__(self, net_data: dict):
        self.net_data = net_data
        if "der_by_bus" not in self.net_data:
            raise ValueError("net_data missing der_by_bus")
        self._build_topology()

    def _build_topology(self) -> None:
        buses = self.net_data["buses"]
        branches = self.net_data["branches"]
        self.n_bus = len(buses)
        self.n_branch = len(branches)
        self.bus_idx = {b[0]: i for i, b in enumerate(buses)}
        self.children = {i: [] for i in range(self.n_bus)}
        self.parent = {}
        self.bus_zone_map = {}
        for bus in buses:
            if len(bus) == 4:
                bus_id, _p_load, _q_load, zone = bus
            else:
                bus_id, _p_load, _q_load = bus
                zone = None
            self.bus_zone_map[bus_id] = zone
        for k, (fb, tb, *_rest) in enumerate(branches):
            fi, ti = self.bus_idx[fb], self.bus_idx[tb]
            self.children[fi].append((ti, k))
            self.parent[ti] = (fi, k)

    def compute_zone_lmp(self, profiles: dict, placement: dict, lambda_ref: float = 5.0) -> tuple[dict[int, float], dict[int, float]]:
        """
        Zone LMP từ net injection deficit — không cần MOSEK solve.
        """
        zone_load = {
            1: float(profiles.get("load_z1", 4.0)),
            2: float(profiles.get("load_z2", 5.5)),
            3: float(profiles.get("load_z3", 3.0)),
            4: float(profiles.get("load_z4", 5.0)),
        }

        pv_pu = float(profiles.get("pv_pu", 0.5))
        wind_mw = float(profiles.get("wind_mw", 8.0))

        zone_gen = {z: 0.0 for z in [1, 2, 3, 4]}
        zone_gen[3] += wind_mw

        for ev in placement.get("evcs", []):
            z = int(ev.get("zone", 1))
            zone_gen[z] += float(ev.get("pv_mw", 0.0)) * pv_pu
        for pv in placement.get("dpv", []):
            z = int(pv.get("zone", 1))
            zone_gen[z] += float(pv.get("mw", 0.0)) * pv_pu

        gfm = placement.get("gfm", {})
        if "G2" in gfm:
            zone_gen[4] += float(gfm["G2"].get("pv_mw", 0.0)) * pv_pu

        zone_net = {z: zone_gen[z] - zone_load[z] for z in [1, 2, 3, 4]}

        alpha = 0.15
        lmps: dict[int, float] = {}
        for z in [1, 2, 4]:
            load_z = max(zone_load[z], 0.5)
            signed_net = zone_net[z]
            lmps[z] = float(lambda_ref) * (1.0 - alpha * signed_net / load_z)
            lmps[z] = max(float(lambda_ref) * 0.70, min(lmps[z], float(lambda_ref) * 1.30))

        return lmps, zone_net

    def solve(self, hour_block: int, profiles: dict, vpp_capacities: dict, placement: dict | None = None) -> L0Result:
        t0 = time.time()

        n_bus = self.n_bus
        n_br = self.n_branch
        n_sw = 3
        n_vpp = 3
        n_as = 3

        buses = self.net_data["buses"]
        branches = self.net_data["branches"]
        cap_banks = self.net_data["cap_banks"]
        der_by_bus = self.net_data.get("der_by_bus", {})
        zone_totals = self.net_data.get("zone_totals", {})

        print(f"Buses: {len(buses)}")
        print(f"Branches: {len(branches)}")
        print(f"DERs: {len(der_by_bus)}")

        P_br = cp.Variable(n_br, name="P_branch")
        Q_br = cp.Variable(n_br, name="Q_branch")
        v = cp.Variable(n_bus, name="v_sq", nonneg=True)
        P_der = cp.Variable(n_bus, nonneg=True)
        Q_der = cp.Variable(n_bus)
        P_slack = cp.Variable(name="P_slack")
        Q_slack = cp.Variable(name="Q_slack")
        alpha_cb = cp.Variable(n_sw, boolean=True, name="alpha_cb")
        q_as = cp.Variable((n_vpp, n_as), nonneg=True, name="q_as")
        q_commit = cp.Variable(n_vpp, nonneg=True, name="q_commit")

        sw_caps = [(b["bus"], b["Q_Mvar"]) for b in cap_banks if b["switchable"]]
        constraints: list[cp.Constraint] = []

        slack_bus = self.net_data.get("slack_bus")
        slack_idx = self.bus_idx.get(slack_bus, 0)
        constraints.append(v[slack_idx] == 1.0)

        load_scales = {}
        for zone, total in zone_totals.items():
            if total <= 0.0:
                load_scales[zone] = 1.0
            else:
                load_scales[zone] = float(profiles.get(f"load_z{zone}", total)) / total

        bus_zone: dict[int, int | None] = {}
        bus_p_scaled: dict[int, float] = {}
        for j, bus in enumerate(buses):
            if len(bus) == 4:
                bus_id, P_load, Q_load, zone = bus
            else:
                bus_id, P_load, Q_load = bus
                zone = None
            bus_zone[bus_id] = zone
            scale = load_scales.get(zone, 1.0)
            P_scaled = float(P_load) * scale
            Q_scaled = float(Q_load) * scale
            bus_p_scaled[bus_id] = P_scaled

            Q_cap_fixed = sum(
                b["Q_Mvar"] for b in cap_banks if b["bus"] == bus_id and not b["switchable"]
            )
            Q_cap_sw_expr = (
                sum(
                    alpha_cb[i] * sw_caps[i][1]
                    for i, (sb, _) in enumerate(sw_caps)
                    if sb == bus_id
                )
                if sw_caps
                else 0
            )

            in_P = sum(P_br[k] for k, (fb, tb, *_rest) in enumerate(branches) if self.bus_idx[tb] == j)
            in_Q = sum(Q_br[k] for k, (fb, tb, *_rest) in enumerate(branches) if self.bus_idx[tb] == j)
            out_P = sum(P_br[k] for k, (fb, tb, *_rest) in enumerate(branches) if self.bus_idx[fb] == j)
            out_Q = sum(Q_br[k] for k, (fb, tb, *_rest) in enumerate(branches) if self.bus_idx[fb] == j)

            if j == slack_idx:
                constraints.append(in_P - out_P + P_der[j] + P_slack == P_scaled)
                constraints.append(
                    in_Q - out_Q + Q_der[j] + Q_cap_fixed + Q_cap_sw_expr + Q_slack == Q_scaled
                )
            else:
                constraints.append(in_P - out_P + P_der[j] == P_scaled)
                constraints.append(in_Q - out_Q + Q_der[j] + Q_cap_fixed + Q_cap_sw_expr == Q_scaled)

        for k, (fb, tb, r, x, _imax) in enumerate(branches):
            fi, ti = self.bus_idx[fb], self.bus_idx[tb]
            constraints.append(v[ti] == v[fi] - 2.0 * (r * P_br[k] + x * Q_br[k]))

        v_viol_lo = cp.Variable(n_bus, nonneg=True, name="v_viol_lo")
        v_viol_hi = cp.Variable(n_bus, nonneg=True, name="v_viol_hi")
        constraints.append(v >= 0.95**2 - v_viol_lo)
        constraints.append(v <= 1.05**2 + v_viol_hi)

        for k, (_fb, _tb, _r, _x, i_max) in enumerate(branches):
            constraints.append(cp.norm(cp.vstack([P_br[k], Q_br[k]]), 2) <= i_max)

        for j, bus in enumerate(buses):
            if len(bus) == 4:
                bus_id = bus[0]
            else:
                bus_id = bus[0]

            if bus_id in der_by_bus:
                P_max, Q_max = der_by_bus[bus_id]
                p_avail = float(P_max) * float(profiles.get("pv_pu", 1.0))
                constraints.append(P_der[j] <= p_avail)
                constraints.append(Q_der[j] >= -float(Q_max))
                constraints.append(Q_der[j] <= float(Q_max))
            else:
                constraints.append(P_der[j] == 0.0)
                constraints.append(Q_der[j] == 0.0)

        R_req = np.array([2.0, 1.5, 1.0], dtype=np.float64)
        as_slack = cp.Variable(n_as, nonneg=True, name="as_slack")
        for s in range(n_as):
            constraints.append(cp.sum(q_as[:, s]) >= R_req[s] - as_slack[s])

        for vi, vpp_id in enumerate([1, 2, 3]):
            caps = vpp_capacities[vpp_id]
            S_agg = float(caps["S_agg"])
            P_commit = min(float(caps["P_max"]) * 0.7, 0.90 * S_agg)
            constraints.append(
                cp.norm(
                    cp.vstack([P_commit + cp.sum(q_as[vi, :]), q_commit[vi]]),
                    2,
                )
                <= S_agg
            )
            constraints.append(q_commit[vi] <= float(caps["Q_max"]))

        Q_load_total = sum(b[2] for b in buses)
        Q_loss_est = 0.05 * Q_load_total
        Q_margin = 0.5
        q_slack = cp.Variable(nonneg=True, name="q_slack")
        constraints.append(
            cp.sum(Q_der)
            + sum(b["Q_Mvar"] for b in cap_banks if not b["switchable"])
            + cp.sum(alpha_cb) * (np.mean([s[1] for s in sw_caps]) if sw_caps else 0.0)
            + q_slack
            >= Q_load_total + Q_loss_est + Q_margin
        )

        r_vals = np.array([br[2] for br in branches], dtype=np.float64) if branches else np.zeros(1)
        C_loss = r_vals @ cp.abs(P_br) if branches else 0.0
        C_switch = 0.1 * cp.sum(alpha_cb)
        P_total_avail = sum(P_max for P_max, _Q in der_by_bus.values()) * float(
            profiles.get("pv_pu", 1.0)
        )
        P_dispatched = cp.sum(P_der) if n_bus else 0.0
        C_curt = 5.0 * (P_total_avail - P_dispatched)

        lam_as = np.array([15.0, 10.0, 6.0], dtype=np.float64)
        lam_q = 2.5
        lam_p2p = 5.0
        zone_net_mw: dict[int, float] = {1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0}
        if placement is not None:
            lmps, zone_net_mw = self.compute_zone_lmp(profiles, placement, lambda_ref=lam_p2p)
            lambda_p2p_z1 = float(lmps.get(1, lam_p2p))
            lambda_p2p_z2 = float(lmps.get(2, lam_p2p))
            lambda_p2p_z4 = float(lmps.get(4, lam_p2p))
            lambda_as_base = float(lam_as[0])
            lam_p2p_safe = float(lam_p2p) if abs(float(lam_p2p)) > 1e-6 else 1.0
            lambda_as_z1 = lambda_as_base * lambda_p2p_z1 / lam_p2p_safe
            lambda_as_z2 = lambda_as_base * lambda_p2p_z2 / lam_p2p_safe
            lambda_as_z4 = lambda_as_base * lambda_p2p_z4 / lam_p2p_safe
        Rev_as = lam_as @ cp.sum(q_as, axis=0)
        Rev_q = lam_q * cp.sum(q_commit)

        slack_penalty = 1e6 * (
            cp.sum(v_viol_lo)
            + cp.sum(v_viol_hi)
            + q_slack
            + cp.sum(as_slack)
            + cp.abs(P_slack)
            + cp.abs(Q_slack)
        )
        objective = cp.Minimize(C_loss + C_switch + C_curt + slack_penalty - Rev_as - Rev_q)
        prob = cp.Problem(objective, constraints)

        try:
            prob.solve(
                solver=cp.MOSEK,
                verbose=True,
                mosek_params={
                    "MSK_IPAR_INTPNT_MAX_ITERATIONS": 200,
                    "MSK_IPAR_INFEAS_REPORT_AUTO": 1,
                    "MSK_IPAR_INFEAS_REPORT_LEVEL": 10,
                },
            )
        except Exception:
            prob.solve(
                solver=cp.MOSEK,
                verbose=True,
                mosek_params={
                    "MSK_IPAR_INFEAS_REPORT_AUTO": 1,
                    "MSK_IPAR_INFEAS_REPORT_LEVEL": 10,
                },
            )

        solve_time = time.time() - t0

        if prob.status not in {"optimal", "optimal_inaccurate"}:
            print(f"v_viol_lo max: {float(np.max(v_viol_lo.value)) if v_viol_lo.value is not None else None}")
            print(f"v_viol_hi max: {float(np.max(v_viol_hi.value)) if v_viol_hi.value is not None else None}")
            print(f"q_slack: {float(q_slack.value) if q_slack.value is not None else None}")
            print(f"as_slack: {as_slack.value if as_slack.value is not None else None}")
            print(f"P_slack: {float(P_slack.value) if P_slack.value is not None else None}")
            print(f"Q_slack: {float(Q_slack.value) if Q_slack.value is not None else None}")

            print("Dual variables (Farkas certificate):")
            for i, c in enumerate(constraints):
                if hasattr(c, "dual_value") and c.dual_value is not None:
                    if np.any(np.abs(c.dual_value) > 1e-6):
                        print(f"  Constraint {i}: dual={c.dual_value}")

            if placement is not None:
                lambda_p2p_z1 = float(lmps.get(1, lam_p2p))
                lambda_p2p_z2 = float(lmps.get(2, lam_p2p))
                lambda_p2p_z4 = float(lmps.get(4, lam_p2p))
                lambda_as_base = float(lam_as[0])
                lam_p2p_safe = float(lam_p2p) if abs(float(lam_p2p)) > 1e-6 else 1.0
                lambda_as_z1 = lambda_as_base * lambda_p2p_z1 / lam_p2p_safe
                lambda_as_z2 = lambda_as_base * lambda_p2p_z2 / lam_p2p_safe
                lambda_as_z4 = lambda_as_base * lambda_p2p_z4 / lam_p2p_safe
            else:
                lambda_p2p_z1 = lam_p2p
                lambda_p2p_z2 = lam_p2p
                lambda_p2p_z4 = lam_p2p
                lambda_as_z1 = lam_as[0]
                lambda_as_z2 = lam_as[0]
                lambda_as_z4 = lam_as[0]

            return L0Result(
                alpha_sw={"CB-A": True, "CB-B": True, "CB-C": True},
                lambda_p2p=lam_p2p,
                lambda_p2p_z1=lambda_p2p_z1,
                lambda_p2p_z2=lambda_p2p_z2,
                lambda_p2p_z4=lambda_p2p_z4,
                lambda_as={"ffr": lam_as[0], "pfr": lam_as[1], "sfr": lam_as[2]},
                lambda_as_z1=lambda_as_z1,
                lambda_as_z2=lambda_as_z2,
                lambda_as_z4=lambda_as_z4,
                lambda_q=lam_q,
                r_as_req={1: R_req[0] / 3, 2: R_req[1] / 3, 3: R_req[2] / 3},
                q_target={1: 0.5, 2: 0.5, 3: 0.3},
                v_profile=np.ones(123),
                solve_time=solve_time,
                status=prob.status or "failed",
                zone_net_mw=zone_net_mw,
            )

        cb_names = ["CB-A", "CB-B", "CB-C"]
        alpha_val = alpha_cb.value
        alpha_sw = {name: bool(alpha_val[i] > 0.5) for i, name in enumerate(cb_names)}
        v_profile = np.sqrt(np.clip(v.value, 0.9**2, 1.1**2))

        if placement is not None:
            lambda_p2p_z1 = float(lmps.get(1, lam_p2p))
            lambda_p2p_z2 = float(lmps.get(2, lam_p2p))
            lambda_p2p_z4 = float(lmps.get(4, lam_p2p))
            lambda_as_base = float(lam_as[0])
            lam_p2p_safe = float(lam_p2p) if abs(float(lam_p2p)) > 1e-6 else 1.0
            lambda_as_z1 = lambda_as_base * lambda_p2p_z1 / lam_p2p_safe
            lambda_as_z2 = lambda_as_base * lambda_p2p_z2 / lam_p2p_safe
            lambda_as_z4 = lambda_as_base * lambda_p2p_z4 / lam_p2p_safe
        else:
            lambda_p2p_z1 = float(lam_p2p)
            lambda_p2p_z2 = float(lam_p2p)
            lambda_p2p_z4 = float(lam_p2p)
            lambda_as_z1 = float(lam_as[0])
            lambda_as_z2 = float(lam_as[0])
            lambda_as_z4 = float(lam_as[0])

        return L0Result(
            alpha_sw=alpha_sw,
            lambda_p2p=float(lam_p2p),
            lambda_p2p_z1=lambda_p2p_z1,
            lambda_p2p_z2=lambda_p2p_z2,
            lambda_p2p_z4=lambda_p2p_z4,
            lambda_as={"ffr": float(lam_as[0]), "pfr": float(lam_as[1]), "sfr": float(lam_as[2])},
            lambda_as_z1=lambda_as_z1,
            lambda_as_z2=lambda_as_z2,
            lambda_as_z4=lambda_as_z4,
            lambda_q=float(lam_q),
            r_as_req={vi + 1: float(np.sum(q_as.value[vi, :])) for vi in range(n_vpp)},
            q_target={vi + 1: float(q_commit.value[vi]) for vi in range(n_vpp)},
            v_profile=v_profile,
            solve_time=solve_time,
            status=prob.status,
            zone_net_mw=zone_net_mw,
        )




def build_net_data_from_pandapower(net) -> dict:
    """
    Convert pandapower net → net_data dict cho L0Optimizer.
    net: pandapower network đã inject DER (env._net_base)
    """
    import pandapower as pp

    _ = pp
    Z_base = 22.0**2 / 1.0

    buses = []
    for idx, row in net.bus.iterrows():
        bus_id = int(row["bus_id"]) if "bus_id" in row else int(idx)
        loads_at_bus = net.load[net.load.bus == idx]
        P_load = float(loads_at_bus.p_mw.sum()) if len(loads_at_bus) > 0 else 0.0
        Q_load = float(loads_at_bus.q_mvar.sum()) if len(loads_at_bus) > 0 else 0.0
        if pd.isna(P_load):
            P_load = 0.0
        if pd.isna(Q_load):
            Q_load = 0.0
        buses.append((bus_id, P_load, Q_load))

    print("sgen count:", len(net.sgen))
    print("storage count:", len(net.storage))
    if not net.sgen.empty:
        cols = [c for c in ["bus", "name", "p_mw", "sn_mva", "type", "controllable"] if c in net.sgen.columns]
        print(net.sgen[cols].to_string())

    branches = []
    for _, line in net.line.iterrows():
        from_pp_idx = int(line.from_bus)
        to_pp_idx = int(line.to_bus)
        from_mpc = int(net.bus.at[from_pp_idx, "bus_id"]) if "bus_id" in net.bus.columns else int(from_pp_idx)
        to_mpc = int(net.bus.at[to_pp_idx, "bus_id"]) if "bus_id" in net.bus.columns else int(to_pp_idx)

        r_ohm = float(line.r_ohm_per_km) * float(line.length_km)
        x_ohm = float(line.x_ohm_per_km) * float(line.length_km)
        if pd.isna(r_ohm):
            r_ohm = 0.0
        if pd.isna(x_ohm):
            x_ohm = 0.0
        r_pu = r_ohm / Z_base
        x_pu = x_ohm / Z_base

        i_max_ka = (
            float(line.max_i_ka)
            if hasattr(line, "max_i_ka") and not pd.isna(line.max_i_ka)
            else 0.2
        )
        i_max_mva = i_max_ka * 22.0 * np.sqrt(3)

        branches.append((from_mpc, to_mpc, r_pu, x_pu, i_max_mva))

    ders = []
    stub_mask = (
        (net.sgen["p_mw"] == 0.0)
        & (net.sgen["sn_mva"] == 1.0)
        & (net.sgen["name"].isna() | (net.sgen["name"] == "None"))
    )
    real_sgens = net.sgen[~stub_mask].copy()

    real_sgens = real_sgens[
        (real_sgens["p_mw"] > 0)
        | (real_sgens["type"].isin(["WP", "wye", "wp"]))
        | (real_sgens["name"].notna() & (real_sgens["name"] != ""))
    ].copy()

    print(f"Filtered: {len(net.sgen)} sgen -> {len(real_sgens)} real DERs")
    if not (20 <= len(real_sgens) <= 30):
        print(f"Unexpected DER count: {len(real_sgens)}")
    if "sn_mva" in real_sgens.columns:
        if not ((real_sgens["sn_mva"].notna()) | (real_sgens["p_mw"] > 0)).all():
            print("DER sn_mva/p_mw check failed")

    for _, sgen in real_sgens.iterrows():
        pp_idx = int(sgen.bus)
        bus_id = int(net.bus.at[pp_idx, "bus_id"]) if "bus_id" in net.bus.columns else int(pp_idx)
        p_mw = float(sgen.p_mw) if pd.notna(sgen.p_mw) else 0.0
        p_max = p_mw if p_mw > 0 else 0.0
        if pd.notna(sgen.get("sn_mva")):
            sn_mva = float(sgen.sn_mva)
        else:
            sn_mva = float(p_mw) / 0.9 if p_mw > 0 else 0.0
        if pd.isna(sn_mva):
            sn_mva = 0.0
        q_max = float(sn_mva * 0.4359)
        dtype = "wind" if sgen.get("type") == "WP" else "pv"
        ders.append((bus_id, p_max, q_max, dtype))

    for _, stor in net.storage.iterrows():
        pp_idx = int(stor.bus)
        bus_id = int(net.bus.at[pp_idx, "bus_id"]) if "bus_id" in net.bus.columns else int(pp_idx)
        p_max = float(stor.max_p_mw) if pd.notna(stor.max_p_mw) else 0.0
        sn_raw = stor.get("sn_mva")
        if pd.notna(sn_raw):
            sn_mva = float(sn_raw)
        else:
            sn_mva = float(p_max / 0.9) if p_max > 0 else 0.0
        q_max = float(sn_mva * 0.4359)
        ders.append((bus_id, p_max, q_max, "bess"))

    der_by_bus: dict[int, tuple[float, float]] = {}
    for bus_id, p_max, q_max, _dtype in ders:
        if bus_id in der_by_bus:
            prev_p, prev_q = der_by_bus[bus_id]
            der_by_bus[bus_id] = (prev_p + p_max, prev_q + q_max)
        else:
            der_by_bus[bus_id] = (p_max, q_max)

    cap_banks = []
    switchable_names = ["CB-A", "CB-B", "CB-C"]
    for _, shunt in net.shunt.iterrows():
        pp_idx = int(shunt.bus)
        bus_id = int(net.bus.at[pp_idx, "bus_id"]) if "bus_id" in net.bus.columns else int(pp_idx)
        name = str(shunt.get("name", ""))
        is_sw = name in switchable_names
        q_mvar = float(shunt.q_mvar) if pd.notna(shunt.q_mvar) else 0.0
        cap_banks.append(
            {
                "bus": bus_id,
                "Q_Mvar": q_mvar,
                "switchable": is_sw,
                "name": name,
            }
        )

    return {
        "buses": buses,
        "branches": branches,
        "ders": ders,
        "der_by_bus": der_by_bus,
        "cap_banks": cap_banks,
    }
