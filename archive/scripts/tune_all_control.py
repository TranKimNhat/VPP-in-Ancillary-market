"""Jointly tune the VSG control knobs (secondary gain agc_ki + FFR gain).

The legacy SG model tuned primary droop R, FFR gain, and AGC PI gains. In the
VSG model the primary response is the swing itself (virtual inertia H + droop
damping K, set per GFM in the placement), so the runtime-tunable knobs are the
secondary-control gain `agc_ki` and the proportional FFR control gain. This
grid-searches both on the real environment and compares against the no-FFR
baseline.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts._vsg_freq_sim import build_env, simulate_event, trace_metrics


def evaluate(env, agc_ki: float, control_gain: float, n_steps: int = 60) -> tuple[float, dict]:
    tr = simulate_event(env, n_steps=n_steps, control_gain=control_gain, agc_ki=agc_ki, seed=42)
    m = trace_metrics(tr)
    freq = tr["freq"]
    oscillation = float(np.ptp(freq[len(freq) // 4:]))
    settle = 40.0 if m["settling_time"] == float("inf") else m["settling_time"]
    nadir_penalty = max(0.0, 49.5 - m["nadir"]) * 100.0
    rocof_penalty = max(0.0, m["max_abs_rocof"] - 2.0) * 5.0
    cost = m["iae"] + 0.3 * settle + 3.0 * oscillation + nadir_penalty + rocof_penalty + (50.0 - m["nadir"]) * 2.0
    return cost, m


def main():
    print("=" * 60)
    print("TUNING VSG CONTROL KNOBS (agc_ki, FFR gain)")
    print("=" * 60)
    env = build_env(seed=42)
    print(f"H_sys = {env.freq_dyn_lti.h_sys:.3f} s (primary inertia is fixed by placement)")

    best, best_cost, best_m = None, float("inf"), None
    for agc_ki in np.round(np.arange(0.02, 0.13, 0.02), 4):
        for gain in np.round(np.arange(0.0, 1.01, 0.25), 4):
            cost, m = evaluate(env, float(agc_ki), float(gain))
            if cost < best_cost:
                best_cost, best, best_m = cost, (float(agc_ki), float(gain)), m

    agc_ki_opt, gain_opt = best
    print(f"\nBest: agc_ki = {agc_ki_opt:.4f}, FFR gain = {gain_opt:.4f}  (cost {best_cost:.4f})")
    print(f"  Nadir: {best_m['nadir']:.3f} Hz | Steady: {best_m['steady_state']:.3f} Hz "
          f"| Max|RoCoF|: {best_m['max_abs_rocof']:.3f} Hz/s")

    # Baseline: best agc_ki, no FFR
    base_cost, base_m = evaluate(env, agc_ki_opt, 0.0)
    print("\n" + "=" * 60)
    print(f"{'Metric':<18}{'Baseline (no FFR)':<20}{'Optimized':<15}")
    print("-" * 55)
    print(f"{'Nadir (Hz)':<18}{base_m['nadir']:<20.3f}{best_m['nadir']:<15.3f}")
    print(f"{'Max|RoCoF|':<18}{base_m['max_abs_rocof']:<20.3f}{best_m['max_abs_rocof']:<15.3f}")
    print(f"{'IAE':<18}{base_m['iae']:<20.3f}{best_m['iae']:<15.3f}")


if __name__ == "__main__":
    main()
