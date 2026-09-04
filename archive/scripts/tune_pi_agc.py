"""Tune the VSG secondary-control integral gain (agc_ki) by grid search.

The legacy SG model exposed PI gains (Kp, Ki) on a turbine-governor AGC. The VSG
model has a single secondary-control integral gain `agc_ki`; this script sweeps
it on the real environment and reports the value minimising a frequency-recovery
cost (IAE + settling + oscillation).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts._vsg_freq_sim import build_env, simulate_event, trace_metrics


def cost_of(env, agc_ki: float, n_steps: int = 60) -> tuple[float, dict]:
    tr = simulate_event(env, n_steps=n_steps, control_gain=0.0, agc_ki=agc_ki, seed=42)
    m = trace_metrics(tr)
    freq = tr["freq"]
    oscillation = float(np.ptp(freq[len(freq) // 4:]))  # range after early transient
    settle = 40.0 if m["settling_time"] == float("inf") else m["settling_time"]
    cost = m["iae"] + 0.5 * settle + 2.0 * oscillation
    return cost, m


def main():
    print("Tuning VSG secondary-control gain agc_ki ...")
    print("=" * 50)
    env = build_env(seed=42)
    print(f"H_sys = {env.freq_dyn_lti.h_sys:.3f} s")

    best_ki, best_cost, best_m = None, float("inf"), None
    print(f"\n{'agc_ki':<10}{'cost':<12}{'nadir':<10}{'steady':<10}")
    print("-" * 42)
    for ki in np.round(np.arange(0.01, 0.21, 0.02), 4):
        cost, m = cost_of(env, float(ki))
        print(f"{ki:<10.4f}{cost:<12.4f}{m['nadir']:<10.3f}{m['steady_state']:<10.3f}")
        if cost < best_cost:
            best_cost, best_ki, best_m = cost, float(ki), m

    print("\n" + "=" * 50)
    print(f"Best agc_ki = {best_ki:.4f}  (cost {best_cost:.4f})")
    print(f"  Nadir: {best_m['nadir']:.3f} Hz | Steady: {best_m['steady_state']:.3f} Hz "
          f"| Max|RoCoF|: {best_m['max_abs_rocof']:.3f} Hz/s")


if __name__ == "__main__":
    main()
