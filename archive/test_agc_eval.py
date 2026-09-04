#!/usr/bin/env python3
"""Evaluate AGC impact on frequency recovery."""
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import numpy as np
import pandas as pd
from pathlib import Path
from src.env.microgrid_env_dual import MicrogridEnvDual

def run_scenario(env, scenario_name, event_type, n_steps=800):
    """Run a single scenario and return frequency trace."""
    obs_fast, obs_slow, info = env.reset(seed=42)

    action_fast = np.zeros((44,), dtype=np.float32)
    f_trace = []
    rocof_trace = []
    time_trace = []

    for step in range(n_steps):
        obs_fast, reward, done, truncated, info = env.step_fast(action_fast)
        f_current = info.get('delta_f', 0.0) + 50.0
        rocof_current = info.get('rocof', 0.0)

        f_trace.append(f_current)
        rocof_trace.append(rocof_current)
        time_trace.append(step * env.dt_fast_s)

    f_trace = np.array(f_trace)
    rocof_trace = np.array(rocof_trace)
    time_trace = np.array(time_trace)

    # Compute metrics
    nadir = float(np.min(f_trace[30:100]))  # Find min between 30-100 steps (after event settles)
    rocof_max = float(np.max(np.abs(rocof_trace[20:60])))  # Peak RoCoF

    # Settling time: when frequency stays within ±0.2 Hz of nadir for >50 steps
    nadir_band = nadir + 0.2
    idx_settling = None
    for i in range(100, len(f_trace)-50):
        if np.all(f_trace[i:i+50] < nadir_band) and np.all(f_trace[i:i+50] > nadir - 0.5):
            idx_settling = i
            break
    settling_time = (idx_settling * env.dt_fast_s) if idx_settling else 800.0

    # Return to nominal: check if f approaches 50 Hz
    f_final_mean = np.mean(f_trace[-50:])  # Last 50 steps (50 seconds)
    return {
        "scenario": scenario_name,
        "nadir_hz": nadir,
        "rocof_max_hz_s": rocof_max,
        "settling_time_s": settling_time,
        "f_final_mean_hz": f_final_mean,
        "agc_active": True,
    }

print("=" * 120)
print("AGC IMPACT EVALUATION - WITH AGC ENABLED")
print("=" * 120)
print()

scenarios = [
    ("S1_load_step", "load_step"),
    ("S2_gen_trip", "gen_trip"),
    ("S3_line_trip", "line_trip"),
]

results = []
for scenario_name, event_type in scenarios:
    print(f"Running {scenario_name}...")
    env = MicrogridEnvDual(
        placement_path="artifacts/placement/official_placement_v3.json",
        mpc_path="data/grid_IEEE123_complete.m",
    )

    try:
        result = run_scenario(env, scenario_name, event_type)
        results.append(result)
        print(f"  ✓ Nadir: {result['nadir_hz']:.4f} Hz")
        print(f"  ✓ RoCoF max: {result['rocof_max_hz_s']:.3f} Hz/s")
        print(f"  ✓ Settling time: {result['settling_time_s']:.1f} s")
        print(f"  ✓ Final frequency: {result['f_final_mean_hz']:.4f} Hz")
    except Exception as e:
        print(f"  ✗ Error: {str(e)}")

print()
print("=" * 120)
print("SUMMARY TABLE")
print("=" * 120)

df = pd.DataFrame(results)
print(df.to_string(index=False))

print()
print("=" * 120)
print("IEEE 1547-III COMPLIANCE (Nadir ≥ 49.5 Hz)")
print("=" * 120)
for _, row in df.iterrows():
    status = "✓ PASS" if row['nadir_hz'] >= 49.5 else "✗ FAIL"
    print(f"{row['scenario']:20} | Nadir: {row['nadir_hz']:6.3f} Hz {status}")

print()
print("=" * 120)
print("INTERPRETATION")
print("=" * 120)
print()
print("With AGC enabled:")
print("  - Frequency should return toward 50 Hz (not stay stuck at nadir)")
print("  - Final frequency (f_final_mean) should be > 49.8 Hz if AGC working")
print("  - Settling time should be < 100 seconds")
print()
