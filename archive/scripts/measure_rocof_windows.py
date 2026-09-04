#!/usr/bin/env python3
"""Measure RoCoF at different observation windows for S1-S4.

Reproduces EXACTLY the env's event-step forcing (microgrid_env_dual fast loop):
delta_P_ref = 0 (controller has not yet reacted at the disturbance instant),
K_droop = K_backbone = 1.0, located disturbance via J_L / event_location_pp.
Uses freq_dyn_lti.simulate_hires (the same call the env makes for plotting) at a
fine micro-step to resolve the true sub-second inertial transient, then computes:

  - instantaneous peak RoCoF (steepest micro-step slope, ~ dP/2H_sys),
  - 100 ms / 500 ms windowed RoCoF (grid-code style sliding windows),
  - 1 s windowed RoCoF  (== the metric get_state() currently reports).
"""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np
import pandapower as pp

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.env.events import EventConfig  # noqa: E402
from src.env.microgrid_env_dual import MicrogridEnvDual  # noqa: E402

PLACEMENT = "artifacts/placement/official_placement_v3.json"
MPC = "data/grid_IEEE123_complete.m"
N_SUB = 2000          # micro_dt = 1.0 / 2000 = 0.5 ms
DT_FAST = 1.0
K_BACKBONE = 1.0

SCENARIOS = {
    "S1": EventConfig(type="load_step", delta_P_mw=2.4, location=45, t_inject=30.0),
    "S2": EventConfig(type="gen_trip", delta_P_mw=-3.9, location=97, t_inject=30.0),
    "S3": EventConfig(type="line_trip", delta_P_mw=-2.4, location=108, t_inject=30.0),
    "S4": EventConfig(type="high_ren", delta_P_mw=4.7, location=97, t_inject=30.0),
}


def windowed_rocof(trace: np.ndarray, micro_dt: float, window_s: float) -> float:
    """Max |Δf(t+w) - Δf(t)| / w over the trace (sliding window)."""
    # Prepend the pre-event sample Δf=0 at t=0.
    f = np.concatenate([[0.0], trace])
    k = max(1, int(round(window_s / micro_dt)))
    if k >= len(f):
        return abs(f[-1] - f[0]) / (micro_dt * (len(f) - 1))
    diffs = np.abs(f[k:] - f[:-k]) / (k * micro_dt)
    return float(np.max(diffs))


def main() -> None:
    env = MicrogridEnvDual(placement_path=PLACEMENT, mpc_path=MPC)
    # BASE feeder (h_sys=1.34, all 6 GFM) — the value the paper's formula uses.
    env.fixed_base_topology = True
    env.reset(seed=42, options={})
    env.fixed_base_topology = False
    lti = env.freq_dyn_lti
    n_gfm = lti.n_gfm
    micro_dt = DT_FAST / N_SUB
    print(f"H_sys = {lti.h_sys:.4f} s | n_gfm = {n_gfm} | micro_dt = {micro_dt*1e3:.2f} ms")
    print(f"{'Sce':>4} {'dP[MW]':>7} {'dP[pu]':>7} {'loc_pp':>6} "
          f"{'analytic dP/2H':>14} {'inst.peak':>10} {'100ms':>8} {'500ms':>8} {'1s(rep.)':>9}")

    f0 = lti.f0
    for name, ev in SCENARIOS.items():
        net = copy.deepcopy(env.net)
        ev2 = copy.copy(ev)
        ev2.injected = False
        ev2.location_pp = None
        _, _, dP_pu = env.event_injector.inject(net, ev2, t_current=ev2.t_inject)
        loc_pp = ev2.location_pp

        topo_id = 0
        if ev.type == "line_trip":
            # Topology changed: rebind J_r on the post-trip network.
            pp.runpp(net, algorithm="nr", init="auto", calculate_voltage_angles=True)
            lti.bind_operating_point(net, topology_id=99)
            topo_id = 99

        lti.reset(f0=50.0)
        trace = lti.simulate_hires(
            dt=DT_FAST,
            delta_P_ref=np.zeros(n_gfm, dtype=float),
            delta_P_L=float(dP_pu),
            K_droop=np.full(n_gfm, K_BACKBONE, dtype=float),
            topology_id=topo_id,
            n_sub=N_SUB,
            event_location_pp=loc_pp,
        )
        trace = np.asarray(trace, dtype=float)

        analytic = abs(dP_pu) / (2.0 * lti.h_sys) * f0
        inst_peak = windowed_rocof(trace, micro_dt, micro_dt)   # steepest micro-step
        r100 = windowed_rocof(trace, micro_dt, 0.1)
        r500 = windowed_rocof(trace, micro_dt, 0.5)
        r1s = abs(trace[-1]) / 1.0                               # what get_state reports

        print(f"{name:>4} {ev.delta_P_mw:>7.1f} {dP_pu:>7.4f} {str(loc_pp):>6} "
              f"{analytic:>14.3f} {inst_peak:>10.3f} {r100:>8.3f} {r500:>8.3f} {r1s:>9.3f}")


if __name__ == "__main__":
    main()
