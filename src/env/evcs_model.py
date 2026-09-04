from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np


class EVCSModel:
    """
    State machine cho 1 EVCS station. Quản lý BESS SoC, EV fleet, V2G eligibility.
    Được gọi từ MicrogridEnv.step() mỗi 15 phút (dT=0.25h).
    """

    N_POLES_MAX = {
        "E1": 30,
        "E2": 15,
        "E3": 25,
        "E4": 20,
        "E5": 15,
        "E6": 10,
    }

    def __init__(self, station_id: str, config: Dict[str, Any]) -> None:
        self.station_id = station_id
        self.config = config
        self.bess_mw = float(config["bess_mw"])
        self.bess_mwh = float(config["bess_mwh"])
        self.v2g_mw = float(config["v2g_mw"])
        self.pv_mw = float(config["pv_mw"])
        self.inverter_mva = float(config.get("inverter_mva", self.pv_mw))
        self.zone = int(config.get("zone", 0))

        self.bess_soc = 0.5
        self.ev_fleet: List[Dict[str, Any]] = []
        self.step_count = 0

        self.eta_ch = 0.95
        self.eta_dis = 0.95

    def arrive(self, ev_params: Dict[str, Any]) -> None:
        self.ev_fleet.append(dict(ev_params))

    def step(
        self,
        P_bess_cmd: float,
        P_v2g_cmd: float,
        P_pv_actual: float,
        dT: float = 0.25,
    ) -> Dict[str, Any]:
        P_bess_cmd = float(P_bess_cmd)
        P_v2g_cmd = max(float(P_v2g_cmd), 0.0)
        P_pv_actual = float(P_pv_actual)

        if P_bess_cmd > 0:  # discharge
            energy_out = P_bess_cmd * dT / self.eta_dis
            self.bess_soc -= energy_out / self.bess_mwh
        else:  # charge
            energy_in = abs(P_bess_cmd) * dT * self.eta_ch
            self.bess_soc += energy_in / self.bess_mwh
        self.bess_soc = float(np.clip(self.bess_soc, 0.10, 0.90))

        p_ch_min = 0.0
        if self.ev_fleet:
            p_ch_min = (
                sum(
                    ev["energy_req_kwh"]
                    / max(ev["departure_step"] - self.step_count, 1)
                    / 1000
                    for ev in self.ev_fleet
                )
                / dT
            )

        total_ev_charging_power = max(0.0, -P_bess_cmd)
        obligation_violated = total_ev_charging_power < p_ch_min * 0.99

        eligible_evs = [
            ev
            for ev in self.ev_fleet
            if ev["soc"] >= 0.25 and ev["departure_step"] > self.step_count + 1
        ]
        eligible_caps = [
            ev["battery_kwh"] * ev["soc"] * 0.5 / 1000 / dT
            for ev in eligible_evs
        ]
        p_v2g_limit = sum(eligible_caps)
        p_v2g_actual = min(P_v2g_cmd, p_v2g_limit)

        if p_v2g_actual > 0 and eligible_evs:
            total_cap = sum(eligible_caps)
            for ev, cap in zip(eligible_evs, eligible_caps):
                share = p_v2g_actual * (cap / total_cap) if total_cap > 0 else 0.0
                energy_out_kwh = share * dT / ev.get("eta_dis", 0.95)
                ev["soc"] = max(ev["soc"] - energy_out_kwh / ev["battery_kwh"], 0.0)

        remaining_evs = []
        ev_count_departed = 0
        for ev in self.ev_fleet:
            if ev["departure_step"] <= self.step_count:
                ev_count_departed += 1
            else:
                remaining_evs.append(ev)
        self.ev_fleet = remaining_evs

        self.step_count += 1

        soc_v2g_agg = (
            float(np.mean([ev["soc"] for ev in self.ev_fleet])) if self.ev_fleet else 0.0
        )

        return {
            "bess_soc": self.bess_soc,
            "soc_v2g_agg": soc_v2g_agg,
            "p_ch_min": p_ch_min,
            "p_v2g_actual": p_v2g_actual,
            "n_ev_connected": len(self.ev_fleet),
            "obligation_violated": bool(obligation_violated),
            "ev_count_departed": ev_count_departed,
        }

    def get_obs_features(self) -> np.ndarray:
        n_poles_max = self.N_POLES_MAX.get(self.station_id, 1)
        n_ev_norm = len(self.ev_fleet) / n_poles_max
        p_ch_min = 0.0
        if self.ev_fleet:
            p_ch_min = (
                sum(
                    ev["energy_req_kwh"]
                    / max(ev["departure_step"] - self.step_count, 1)
                    / 1000
                    for ev in self.ev_fleet
                )
                / 0.25
            )
        p_ch_min_norm = p_ch_min / self.bess_mw if self.bess_mw > 0 else 0.0
        soc_v2g_agg = (
            float(np.mean([ev["soc"] for ev in self.ev_fleet])) if self.ev_fleet else 0.0
        )
        return np.array(
            [self.bess_soc, soc_v2g_agg, n_ev_norm, p_ch_min_norm],
            dtype=np.float32,
        )


class EVSessionGenerator:
    """
    Sinh EV arrivals theo distribution calibrated từ ACN-Data patterns.
    Được gọi từ env.reset() để generate session list cho 1 ngày.
    """

    STATION_PROFILES = {
        "E1": {
            "type": "residential_evening",
            "n_poles": 30,
            "lambda_per_hour": [
                0,
                0,
                0,
                0,
                0,
                0.5,
                1,
                2,
                1,
                1,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                1,
                2,
                3,
                4,
                4,
                3,
                2,
                1,
                0.5,
            ],
        },
        "E2": {
            "type": "apartment_overnight",
            "n_poles": 15,
            "lambda_per_hour": [
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                1,
                1,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                1,
                2,
                3,
                3,
                2,
                1,
                1,
                0.5,
            ],
        },
        "E3": {
            "type": "office_daytime",
            "n_poles": 25,
            "lambda_per_hour": [
                0,
                0,
                0,
                0,
                0,
                0,
                1,
                3,
                4,
                3,
                2,
                1,
                1,
                1,
                1,
                1,
                2,
                2,
                1,
                0,
                0,
                0,
                0,
                0,
            ],
        },
        "E4": {
            "type": "shopping_spread",
            "n_poles": 20,
            "lambda_per_hour": [
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0.5,
                1,
                2,
                2,
                2,
                2,
                2,
                2,
                2,
                2,
                2,
                1,
                1,
                0.5,
                0,
                0,
                0,
            ],
        },
        "E5": {
            "type": "community_mixed",
            "n_poles": 15,
            "lambda_per_hour": [
                0,
                0,
                0,
                0,
                0,
                0.5,
                1,
                2,
                2,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                2,
                2,
                2,
                1,
                1,
                0.5,
                0,
                0,
            ],
        },
        "E6": {
            "type": "fleet_batch",
            "n_poles": 10,
            "lambda_per_hour": [
                0,
                0,
                0,
                0,
                0,
                2,
                2,
                2,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                2,
                2,
                2,
                0,
                0,
                0,
                0,
                0,
            ],
        },
        "E7": {
            "type": "community_mixed",
            "n_poles": 12,
            "lambda_per_hour": [
                0,
                0,
                0,
                0,
                0,
                0.5,
                1,
                2,
                2,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                2,
                2,
                2,
                1,
                1,
                0.5,
                0,
                0,
            ],
        },
        "E8": {
            "type": "shopping_spread",
            "n_poles": 12,
            "lambda_per_hour": [
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0.5,
                1,
                2,
                2,
                2,
                2,
                2,
                2,
                2,
                2,
                2,
                1,
                1,
                0.5,
                0,
                0,
                0,
            ],
        },
        "E9": {
            "type": "residential_evening",
            "n_poles": 12,
            "lambda_per_hour": [
                0,
                0,
                0,
                0,
                0,
                0.5,
                1,
                2,
                1,
                1,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                1,
                2,
                3,
                4,
                4,
                3,
                2,
                1,
                0.5,
            ],
        },
    }

    def generate(
        self, station_id: str, n_steps: int = 96, seed: int | None = None
    ) -> List[Tuple[int, Dict[str, Any]]]:
        rng = np.random.default_rng(seed)
        profile = self.STATION_PROFILES[station_id]
        dT = 0.25
        arrivals: List[Tuple[int, Dict[str, Any]]] = []
        ev_id = 0

        for step in range(n_steps):
            hour = step // 4
            lam = profile["lambda_per_hour"][hour] * dT
            n_arrivals = rng.poisson(lam)
            for _ in range(n_arrivals):
                ev_id += 1
                stay_hours = float(rng.lognormal(mean=2.5, sigma=0.6))
                stay_hours = float(np.clip(stay_hours, 0.5, 12.0))
                stay_steps = max(int(np.ceil(stay_hours / dT)), 1)
                departure_step = min(step + stay_steps, n_steps)

                energy_req_kwh = float(rng.lognormal(mean=25, sigma=0.5))
                energy_req_kwh = float(np.clip(energy_req_kwh, 5.0, 80.0))
                battery_kwh = float(rng.choice([40, 60, 75, 100], p=[0.1, 0.3, 0.4, 0.2]))
                soc_arrival = float(rng.uniform(0.15, 0.65))

                ev_dict = {
                    "id": ev_id,
                    "soc": soc_arrival,
                    "battery_kwh": battery_kwh,
                    "energy_req_kwh": energy_req_kwh,
                    "departure_step": departure_step,
                    "eta_ch": 0.95,
                    "eta_dis": 0.95,
                }
                arrivals.append((step, ev_dict))

        return arrivals


@dataclass
class EVBatteryConfig:
    E_cap_kwh: float = 50.0
    E0: float = 400.0
    R0: float = 0.05
    R: float = 0.02
    C: float = 5000.0
    Rx: float = 0.05
    Cx: float = 1000.0
    SoC_min: float = 0.20
    SoC_max: float = 0.90
    SoC_init: float = 0.50
    P_rated_kw: float = 50.0
    J_vsg: float = 0.5
    Kd_vsg: float = 20.0
    omega0: float = 2 * np.pi * 50.0
    t_dep_h: float = 8.0
    eta_charge: float = 0.95
    eta_discharge: float = 0.95


class EVBatteryModel:
    def __init__(self, config: EVBatteryConfig | None = None) -> None:
        self.cfg = config or EVBatteryConfig()
        self.reset()

    def reset(self, SoC_init: float | None = None, t_dep_h: float | None = None) -> None:
        cfg = self.cfg
        self.SoC = float(SoC_init if SoC_init is not None else cfg.SoC_init)
        self.Uc = 0.0
        self.Uc1 = 0.0
        self.omega = cfg.omega0
        self.theta = 0.0

        dep_h = t_dep_h if t_dep_h is not None else cfg.t_dep_h
        self.t_dep_remain = float(dep_h) * 3600.0

        self.Ib = 0.0
        self.Vbat = cfg.E0
        self.P_out_kw = 0.0

    def step(self, dt: float, P_cmd_kw: float) -> Tuple[float, float, float, float, float]:
        cfg = self.cfg

        P_cmd_kw = self._clip_power(P_cmd_kw)

        Vbat_prev = cfg.E0 - self.Uc - self.Uc1
        Ib = P_cmd_kw * 1000.0 / max(Vbat_prev, 200.0)

        dUc = (Ib / cfg.C - self.Uc / (cfg.R * cfg.C)) * dt
        dUc1 = (Ib / cfg.Cx - self.Uc1 / (cfg.Rx * cfg.Cx)) * dt
        self.Uc = np.clip(self.Uc + dUc, -200.0, 200.0)
        self.Uc1 = np.clip(self.Uc1 + dUc1, -200.0, 200.0)

        self.Vbat = cfg.E0 - Ib * cfg.R0 - self.Uc - self.Uc1

        E_cap_j = cfg.E_cap_kwh * 3.6e6
        if P_cmd_kw >= 0:
            eta = cfg.eta_discharge
            dSoC = -(eta * P_cmd_kw * 1000.0 * dt) / E_cap_j
        else:
            eta = cfg.eta_charge
            dSoC = -(P_cmd_kw * 1000.0 * dt) / (eta * E_cap_j)
        self.SoC = np.clip(self.SoC + dSoC, cfg.SoC_min, cfg.SoC_max)

        P_ref_w = P_cmd_kw * 1000.0
        P_cs_w = Ib * self.Vbat
        domega = (P_ref_w - P_cs_w + cfg.Kd_vsg * (cfg.omega0 - self.omega)) / (
            cfg.J_vsg * cfg.omega0
        ) * dt
        self.omega = self.omega + domega
        self.theta = self.theta + self.omega * dt

        self.t_dep_remain = max(0.0, self.t_dep_remain - dt)

        self.Ib = Ib
        self.P_out_kw = P_cmd_kw
        return self.Uc, self.Uc1, self.SoC, self.Vbat, self.Ib

    def dcob(self) -> float:
        cfg = self.cfg
        if self.t_dep_remain < 1.0:
            return 0.0
        available_kwh = cfg.E_cap_kwh * (self.SoC - cfg.SoC_min)
        return available_kwh * 3600.0 / self.t_dep_remain

    def ccob(self) -> float:
        cfg = self.cfg
        if self.t_dep_remain < 1.0:
            return 0.0
        headroom_kwh = cfg.E_cap_kwh * (cfg.SoC_max - self.SoC)
        return headroom_kwh * 3600.0 / self.t_dep_remain

    def power_bounds(self) -> Tuple[float, float]:
        cfg = self.cfg
        if self.t_dep_remain < 1.0:
            return (0.0, 0.0)

        E_cap_j = cfg.E_cap_kwh * 3.6e6
        dt_rem = self.t_dep_remain

        E_need_min = E_cap_j * (cfg.SoC_min - self.SoC)
        P_min_kw = E_need_min / (dt_rem * 1000.0)

        E_avail_max = E_cap_j * (self.SoC - cfg.SoC_min)
        P_max_kw = E_avail_max / (dt_rem * 1000.0)

        P_min_kw = max(P_min_kw, -cfg.P_rated_kw)
        P_max_kw = min(P_max_kw, cfg.P_rated_kw)
        return (float(P_min_kw), float(P_max_kw))

    def _clip_power(self, P_cmd_kw: float) -> float:
        return float(np.clip(P_cmd_kw, -self.cfg.P_rated_kw, self.cfg.P_rated_kw))

    def get_state(self) -> dict:
        return {
            "SoC": self.SoC,
            "Uc": self.Uc,
            "Uc1": self.Uc1,
            "Vbat": self.Vbat,
            "Ib": self.Ib,
            "omega": self.omega,
            "DCoB": self.dcob(),
            "CCoB": self.ccob(),
            "t_dep_remain_h": self.t_dep_remain / 3600.0,
        }


def mpc_correction(
    evcs_list: list[EVBatteryModel],
    delta_f: float,
    rocof: float,
    H_sys: float = 1.432,
    S_base_mw: float = 15.7,
    p: int = 3,
    dt: float = 1.0,
    alpha: float = 1.0,
    beta: float = 0.1,
    deadband_hz: float = 0.1,
) -> np.ndarray:
    from scipy.optimize import minimize

    n = len(evcs_list)
    if abs(delta_f) <= deadband_hz:
        return np.zeros(n)

    bounds = [ev.power_bounds() for ev in evcs_list]
    P_current = np.array([ev.P_out_kw for ev in evcs_list])

    if np.all([b[1] <= 0.0 for b in bounds]):
        return np.zeros(n)

    def objective(dP: np.ndarray) -> float:
        delta_f_k = float(delta_f)
        total_cost = 0.0
        for _k in range(p):
            total_P_correction_mw = float(dP.sum()) / 1000.0
            delta_f_k += (total_P_correction_mw * dt) / (2 * H_sys * S_base_mw)

            total_cost += alpha * delta_f_k ** 2

            for i in range(n):
                total_cost += beta * (dP[i] ** 2) / (1000.0 ** 2)

        return float(total_cost)

    scipy_bounds = [
        (b[0] - P_current[i], b[1] - P_current[i]) for i, b in enumerate(bounds)
    ]

    result = minimize(
        objective,
        x0=np.zeros(n),
        method="SLSQP",
        bounds=scipy_bounds,
        options={"maxiter": 50, "ftol": 1e-4},
    )

    dP = result.x if result.success else np.zeros(n)

    if dP.sum() <= 1e-6 and abs(delta_f) > deadband_hz:
        if delta_f < 0:
            headroom = np.array([max(0.0, b[1] - P_current[i]) for i, b in enumerate(bounds)])
            total = float(headroom.sum())
            if total > 0:
                dP = headroom
        elif delta_f > 0:
            headroom = np.array([min(0.0, b[0] - P_current[i]) for i, b in enumerate(bounds)])
            total = float(headroom.sum())
            if total < 0:
                dP = headroom

    return dP
