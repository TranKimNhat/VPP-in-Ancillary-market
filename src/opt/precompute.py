import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

import sys

sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.env.evcs_model import EVSessionGenerator
from src.env.microgrid_env import MicrogridEnv
from src.env.freq_model import compute_nadir
from src.opt.l0_reconfig import L0Optimizer, L0Result, build_net_data_from_pandapower
from src.opt.l1_dispatch import L1Dispatcher


SEASONS = ["summer", "winter", "spring", "fall"]
DAY_TYPES = ["weekday", "weekend"]

BASE_LOADS = {
    "z1": 0.9307,
    "z2": 1.0470,
    "z3": 0.8143,
    "z4": 0.6980,
}

PEAK_HOUR_BY_SEASON = {
    "summer": 14,
    "winter": 18,
    "spring": 16,
    "fall": 16,
}


def _compute_vpp_totals(placement: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
    totals: Dict[str, Dict[str, float]] = {"__placement__": placement}

    def ensure(vpp: str) -> Dict[str, float]:
        if vpp not in totals:
            totals[vpp] = {
                "evcs_pv_mw": 0.0,
                "evcs_bess_mw": 0.0,
                "evcs_v2g_mw": 0.0,
                "dpv_mw": 0.0,
                "inverter_mva": 0.0,
            }
        return totals[vpp]

    for ev in placement.get("evcs", []):
        vpp = ev.get("vpp", "")
        if not vpp:
            continue
        bucket = ensure(vpp)
        bucket["evcs_pv_mw"] += float(ev.get("pv_mw", 0.0))
        bucket["evcs_bess_mw"] += float(ev.get("bess_mw", 0.0))
        bucket["evcs_v2g_mw"] += float(ev.get("v2g_mw", 0.0))
        bucket["inverter_mva"] += float(ev.get("inverter_mva", ev.get("pv_mw", 0.0)))

    for pv in placement.get("dpv", []):
        vpp = pv.get("vpp", "")
        if not vpp:
            continue
        bucket = ensure(vpp)
        bucket["dpv_mw"] += float(pv.get("mw", 0.0))
        bucket["inverter_mva"] += float(
            pv.get("inverter_mva", pv.get("sn_mva", pv.get("mw", 0.0)))
        )

    return totals


def _daily_load_profile(
    base_mw: float, peak_hour: int, weekday_factor: float, rng: np.random.Generator
) -> np.ndarray:
    steps = 96
    hours = np.arange(steps) * 0.25
    shape = 0.7 + 0.3 * np.cos(2 * math.pi * (hours - peak_hour) / 24.0)
    noise = rng.lognormal(mean=0.0, sigma=0.05, size=steps)
    base = base_mw * weekday_factor
    load = base * shape * noise
    return np.clip(load, 0.4 * base, 1.0 * base)


def _pv_profile(rng: np.random.Generator) -> np.ndarray:
    steps = 96
    hours = np.arange(steps) * 0.25
    irradiance = np.where(
        (hours >= 6.0) & (hours <= 18.0), np.sin(math.pi * (hours - 6.0) / 12.0), 0.0
    )
    irradiance = np.clip(irradiance, 0.0, 1.0)

    cloud_factor = rng.beta(5, 2)
    pv_pu = irradiance * cloud_factor

    n_events = rng.poisson(2)
    for _ in range(n_events):
        start = int(rng.integers(0, steps))
        duration = int(rng.integers(1, 5))
        drop = float(rng.uniform(0.3, 0.7))
        end = min(start + duration, steps)
        pv_pu[start:end] *= 1.0 - drop

    return np.clip(pv_pu, 0.0, 1.0)


def _wind_profile(rng: np.random.Generator) -> np.ndarray:
    steps = 96
    k = 2.0
    c = 8.0
    alpha = 0.85

    v = np.zeros(steps, dtype=float)
    for t in range(steps):
        sample = float(rng.weibull(k) * c)
        if t == 0:
            v[t] = sample
        else:
            v[t] = alpha * v[t - 1] + (1.0 - alpha) * sample

    p_single = np.where(
        v < 3.0,
        0.0,
        np.where(v < 11.0, 3.0 * ((v - 3.0) / 8.0) ** 3, np.where(v < 25.0, 3.0, 0.0)),
    )
    wind_mw = 4.0 * p_single
    return np.clip(wind_mw, 0.0, 12.0)


def _sample_trunc_normal(
    rng: np.random.Generator,
    mean: float,
    sigma: float,
    low: float,
    high: float,
    size: int,
) -> np.ndarray:
    samples = np.zeros(size, dtype=float)
    for i in range(size):
        while True:
            value = float(rng.normal(mean, sigma))
            if low <= value <= high:
                samples[i] = value
                break
    return samples


def _ev_occupancy(
    generator: EVSessionGenerator, station_id: str, n_steps: int, seed: int
) -> np.ndarray:
    arrivals = generator.generate(station_id, n_steps=n_steps, seed=seed)
    occupancy = np.zeros(n_steps, dtype=float)
    n_poles = int(generator.STATION_PROFILES[station_id]["n_poles"])

    for step, ev in arrivals:
        departure = int(ev["departure_step"])
        occupancy[step:departure] += 1.0

    return np.minimum(occupancy, n_poles)


def _build_vpp_caps(placement_totals: Dict[str, Dict[str, float]]):
    caps = {}
    for idx, vpp_name in enumerate(["VPP_1", "VPP_2", "VPP_3"], start=1):
        totals = placement_totals.get(vpp_name, {})
        p_max = float(totals.get("evcs_bess_mw", 0.0) + totals.get("evcs_v2g_mw", 0.0) + totals.get("dpv_mw", 0.0))
        q_max = float(totals.get("inverter_mva", p_max))
        caps[idx] = {"P_max": p_max, "Q_max": q_max, "S_agg": float(totals.get("inverter_mva", q_max))}
    return caps


def _generate_day(
    day_idx: int,
    placement_totals: Dict[str, Dict[str, float]],
    seed: int,
    l0_optimizer: L0Optimizer | None = None,
    l1_dispatcher: L1Dispatcher | None = None,
) -> tuple[pd.DataFrame, str]:
    rng = np.random.default_rng(seed + day_idx)
    steps = 96
    hours = np.arange(steps) * 0.25

    season = str(rng.choice(SEASONS))
    day_type = str(rng.choice(DAY_TYPES))
    weekday_factor = 1.0 if day_type == "weekday" else 0.85
    peak_hour = PEAK_HOUR_BY_SEASON[season]

    load_z1 = _daily_load_profile(BASE_LOADS["z1"], peak_hour, weekday_factor, rng)
    load_z2 = _daily_load_profile(BASE_LOADS["z2"], peak_hour, weekday_factor, rng)
    load_z3 = _daily_load_profile(BASE_LOADS["z3"], peak_hour, weekday_factor, rng)
    load_z4 = _daily_load_profile(BASE_LOADS["z4"], peak_hour, weekday_factor, rng)

    pv_pu = _pv_profile(rng)
    wind_mw = _wind_profile(rng)

    ev_gen = EVSessionGenerator()
    n_ev_e1 = _ev_occupancy(ev_gen, "E1", steps, seed + day_idx * 11 + 1)
    n_ev_e2 = _ev_occupancy(ev_gen, "E2", steps, seed + day_idx * 11 + 2)
    n_ev_e3 = _ev_occupancy(ev_gen, "E3", steps, seed + day_idx * 11 + 3)
    n_ev_e4 = _ev_occupancy(ev_gen, "E4", steps, seed + day_idx * 11 + 4)
    n_ev_e5 = _ev_occupancy(ev_gen, "E5", steps, seed + day_idx * 11 + 5)
    n_ev_e6 = _ev_occupancy(ev_gen, "E6", steps, seed + day_idx * 11 + 6)
    n_ev_e7 = _ev_occupancy(ev_gen, "E7", steps, seed + day_idx * 11 + 7)
    n_ev_e8 = _ev_occupancy(ev_gen, "E8", steps, seed + day_idx * 11 + 8)
    n_ev_e9 = _ev_occupancy(ev_gen, "E9", steps, seed + day_idx * 11 + 9)

    total_load = load_z1 + load_z2 + load_z3 + load_z4
    base_total = sum(BASE_LOADS.values()) * weekday_factor
    load_ratio = np.clip(total_load / base_total, 0.4, 1.0)

    lambda_p2p_base = rng.uniform(3.0, 8.0, size=steps)
    lambda_p2p = lambda_p2p_base * (0.6 + 0.6 * load_ratio)
    lambda_as_ffr = rng.uniform(8.0, 20.0, size=steps)
    lambda_as_pfr = rng.uniform(5.0, 15.0, size=steps)
    lambda_as_sfr = rng.uniform(3.0, 10.0, size=steps)
    lambda_q = rng.uniform(0.5, 5.0, size=steps)

    lambda_p2p_z1 = lambda_p2p.copy()
    lambda_p2p_z2 = lambda_p2p.copy()
    lambda_p2p_z4 = lambda_p2p.copy()
    lambda_as_z1 = lambda_as_ffr.copy()
    lambda_as_z2 = lambda_as_ffr.copy()
    lambda_as_z4 = lambda_as_ffr.copy()

    p_ref_vpp = [np.zeros(steps, dtype=float) for _ in range(3)]
    r_as_vpp = [np.zeros(steps, dtype=float) for _ in range(3)]
    q_commit_vpp = [np.zeros(steps, dtype=float) for _ in range(3)]

    load_z_peak = {
        "VPP_1": float(BASE_LOADS["z1"] * weekday_factor),
        "VPP_2": float(BASE_LOADS["z2"] * weekday_factor),
        "VPP_3": float(BASE_LOADS["z3"] * weekday_factor),
    }
    load_z = {
        "VPP_1": load_z1,
        "VPP_2": load_z2,
        "VPP_3": load_z3,
    }

    l0_result = None
    l0_status = "skipped"
    if l0_optimizer is not None:
        vpp_caps = _build_vpp_caps(placement_totals)
        l0_results: dict[int, L0Result] = {}
        l0_block = 4
        for step in range(steps):
            if step % l0_block == 0:
                l0_result = l0_optimizer.solve(
                    int(step // 24),
                    {
                        "load_z1": float(load_z1[step]),
                        "load_z2": float(load_z2[step]),
                        "load_z3": float(load_z3[step]),
                        "load_z4": float(load_z4[step]),
                        "pv_pu": float(pv_pu[step]),
                        "wind_mw": float(wind_mw[step]),
                    },
                    vpp_caps,
                    placement=placement_totals.get("__placement__"),
                )
                l0_results[step] = l0_result
            if l0_result is None:
                continue
            l0_status = "feasible" if l0_result.status in {"optimal", "optimal_inaccurate"} else "infeasible"
            lambda_p2p[step] = float(l0_result.lambda_p2p)
            lambda_as_ffr[step] = float(l0_result.lambda_as.get("ffr", lambda_as_ffr[step]))
            lambda_as_pfr[step] = float(l0_result.lambda_as.get("pfr", lambda_as_pfr[step]))
            lambda_as_sfr[step] = float(l0_result.lambda_as.get("sfr", lambda_as_sfr[step]))
            lambda_q[step] = float(l0_result.lambda_q)

            lambda_p2p_z1[step] = float(getattr(l0_result, "lambda_p2p_z1", lambda_p2p[step]))
            lambda_p2p_z2[step] = float(getattr(l0_result, "lambda_p2p_z2", lambda_p2p[step]))
            lambda_p2p_z4[step] = float(getattr(l0_result, "lambda_p2p_z4", lambda_p2p[step]))
            lambda_as_z1[step] = float(getattr(l0_result, "lambda_as_z1", lambda_as_ffr[step]))
            lambda_as_z2[step] = float(getattr(l0_result, "lambda_as_z2", lambda_as_ffr[step]))
            lambda_as_z4[step] = float(getattr(l0_result, "lambda_as_z4", lambda_as_ffr[step]))

    for vpp_name in ["VPP_1", "VPP_2", "VPP_3"]:
        totals = placement_totals.get(vpp_name, {})
        evcs_pv_mw = totals.get("evcs_pv_mw", 0.0)
        evcs_bess_mw = totals.get("evcs_bess_mw", 0.0)
        evcs_v2g_mw = totals.get("evcs_v2g_mw", 0.0)
        dpv_mw = totals.get("dpv_mw", 0.0)
        inverter_mva = totals.get("inverter_mva", 0.0)

        pv_actual = (evcs_pv_mw + dpv_mw) * pv_pu
        load_ratio_z = np.clip(load_z[vpp_name] / max(load_z_peak[vpp_name], 1e-6), 0.4, 1.0)
        bess_headroom = evcs_bess_mw * 0.3
        p_ref = pv_actual + bess_headroom * load_ratio_z

        dispatchable = evcs_bess_mw + evcs_v2g_mw + dpv_mw
        r_as_base = max(0.15 * dispatchable, 0.1)
        q_commit = 0.1 * inverter_mva

        idx = int(vpp_name.split("_")[1]) - 1
        p_ref_vpp[idx] = p_ref
        r_as_vpp[idx] = np.full(steps, r_as_base, dtype=float)
        q_commit_vpp[idx] = np.full(steps, q_commit, dtype=float)

    if l1_dispatcher is not None:
        for step in range(steps):
            l0_input = l0_result
            if l0_input is None and l0_optimizer is not None:
                l0_input = l0_results.get(step)
                if l0_input is None:
                    vpp_caps = _build_vpp_caps(placement_totals)
                    l0_input = l0_optimizer.solve(
                        int(step // 24),
                        {
                            "load_z1": float(load_z1[step]),
                            "load_z2": float(load_z2[step]),
                            "load_z3": float(load_z3[step]),
                            "load_z4": float(load_z4[step]),
                            "pv_pu": float(pv_pu[step]),
                            "wind_mw": float(wind_mw[step]),
                        },
                        vpp_caps,
                        placement=placement_totals.get("__placement__"),
                    )
            if l0_input is None:
                continue
            l0_status = "feasible" if l0_input.status in {"optimal", "optimal_inaccurate"} else "infeasible"
            l1_result = l1_dispatcher.solve_step(
                step,
                {"pv_pu": float(pv_pu[step])},
                l0_input,
                vpp_id=1,
                lambda_p2p_init=float(lambda_p2p[step]),
            )
            if l1_result.status not in {"optimal", "optimal_inaccurate"}:
                continue
            for vpp_id in [1, 2, 3]:
                p_ref_vpp[vpp_id - 1][step] = float(l1_result.p_ref.get(vpp_id, p_ref_vpp[vpp_id - 1][step]))
                r_as_vpp[vpp_id - 1][step] = float(l1_result.r_as.get(vpp_id, r_as_vpp[vpp_id - 1][step]))
                q_commit_vpp[vpp_id - 1][step] = float(l1_result.q_commit.get(vpp_id, q_commit_vpp[vpp_id - 1][step]))
            lambda_p2p[step] = float(l1_result.lambda_p2p)

    freq_event_flag = (rng.random(steps) < 0.10).astype(int)
    delta_p_choices = np.array([-1.5, -1.0, -0.5, -0.3, 0.3, 0.5], dtype=np.float32)
    delta_p_cont = rng.choice(delta_p_choices, size=steps)
    delta_p_cont = delta_p_cont * freq_event_flag
    f_nadir = np.full(steps, 50.0, dtype=np.float32)
    rocof = np.zeros(steps, dtype=np.float32)
    for step in range(steps):
        if freq_event_flag[step]:
            freq = compute_nadir(float(delta_p_cont[step]))
            f_nadir[step] = float(freq["f_nadir"])
            rocof[step] = float(freq["rocof"])

    df = pd.DataFrame(
        {
            "step": np.arange(steps, dtype=int),
            "hour": hours,
            "load_z1": load_z1,
            "load_z2": load_z2,
            "load_z3": load_z3,
            "load_z4": load_z4,
            "pv_pu": pv_pu,
            "wind_mw": wind_mw,
            "n_ev_e1": n_ev_e1,
            "n_ev_e2": n_ev_e2,
            "n_ev_e3": n_ev_e3,
            "n_ev_e4": n_ev_e4,
            "n_ev_e5": n_ev_e5,
            "n_ev_e6": n_ev_e6,
            "n_ev_e7": n_ev_e7,
            "n_ev_e8": n_ev_e8,
            "n_ev_e9": n_ev_e9,
            "lambda_p2p": lambda_p2p,
            "lambda_p2p_z1": lambda_p2p_z1,
            "lambda_p2p_z2": lambda_p2p_z2,
            "lambda_p2p_z4": lambda_p2p_z4,
            "lambda_as_ffr": lambda_as_ffr,
            "lambda_as_pfr": lambda_as_pfr,
            "lambda_as_sfr": lambda_as_sfr,
            "lambda_as_z1": lambda_as_z1,
            "lambda_as_z2": lambda_as_z2,
            "lambda_as_z4": lambda_as_z4,
            "lambda_q": lambda_q,
            "p_ref_vpp1": p_ref_vpp[0],
            "p_ref_vpp2": p_ref_vpp[1],
            "p_ref_vpp3": p_ref_vpp[2],
            "r_as_vpp1": r_as_vpp[0],
            "r_as_vpp2": r_as_vpp[1],
            "r_as_vpp3": r_as_vpp[2],
            "q_commit_vpp1": q_commit_vpp[0],
            "q_commit_vpp2": q_commit_vpp[1],
            "q_commit_vpp3": q_commit_vpp[2],
            "freq_event_flag": freq_event_flag,
            "delta_p_cont": delta_p_cont,
            "f_nadir": f_nadir,
            "rocof": rocof,
            "season": [season] * steps,
            "day_type": [day_type] * steps,
        }
    )

    return df, l0_status


def generate_all_days(
    n_days: int = 200,
    output_dir: str = "data/precomputed",
    seed: int = 42,
    use_optimizers: bool = False,
    placement_path: str | Path = "artifacts/placement/official_placement_v3.json",
) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    placement_path = Path(placement_path)
    with placement_path.open(encoding="utf-8") as f:
        placement = json.load(f)
    placement_totals = _compute_vpp_totals(placement)

    l0_optimizer = None
    l1_dispatcher = None
    if use_optimizers:
        l0_optimizer = L0Optimizer(build_net_data_from_pandapower(MicrogridEnv(placement_path, "data/grid_IEEE123_complete.m").base_net))
        l1_dispatcher = L1Dispatcher()

    last_df: pd.DataFrame | None = None
    for day_idx in range(n_days):
        df, l0_status = _generate_day(day_idx, placement_totals, seed, l0_optimizer, l1_dispatcher)
        df.to_parquet(output_path / f"day_{day_idx:03d}.parquet", index=False)
        last_df = df
        if use_optimizers:
            print(f"Generated {day_idx + 1}/{n_days} days | L0={l0_status}")
        else:
            print(f"Generated {day_idx + 1}/{n_days} days")

    eval_days = [f"day_{day_idx:03d}.parquet" for day_idx in range(max(n_days - 20, 0), n_days)]
    (output_path / "eval_days.txt").write_text("\n".join(eval_days))

    if last_df is None:
        return

    schema = [
        {"name": col, "dtype": str(dtype)}
        for col, dtype in zip(last_df.columns, last_df.dtypes, strict=False)
    ]

    metadata = {
        "seed": seed,
        "n_days": n_days,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "schema": schema,
        "rules": {
            "loads": "Seasonal sinusoid with lognormal noise, clamped to [0.4, 1.0] * base",
            "pv": "Clear-sky sinusoid with Beta cloud factor and random cloud drops",
            "wind": "AR(1) Weibull wind speed with cubic power curve",
            "frequency": "Bernoulli(0.83) events with TruncNormal(2,3) contingency",
        },
    }
    (output_path / "metadata.json").write_text(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-days", type=int, default=200)
    parser.add_argument("--output-dir", default="data/precomputed")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use-optimizers", action="store_true")
    parser.add_argument(
        "--placement",
        default="artifacts/placement/official_placement_v3.json",
        help="Path to placement JSON",
    )
    args = parser.parse_args()

    generate_all_days(
        args.n_days,
        args.output_dir,
        args.seed,
        use_optimizers=args.use_optimizers,
        placement_path=args.placement,
    )
