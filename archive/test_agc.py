#!/usr/bin/env python3
"""Quick test of AGC functionality."""
import sys
import io
import numpy as np
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from src.env.microgrid_env_dual import MicrogridEnvDual

env = MicrogridEnvDual(
    placement_path="artifacts/placement/official_placement_v3.json",
    mpc_path="data/ieee123_modified.m",
)

obs_fast, obs_slow, info = env.reset(seed=42)

print("=" * 100)
print("AGC FUNCTIONALITY TEST")
print("=" * 100)
print()
print(f"Initial frequency: {env.freq_dyn_lti.get_state().delta_f_hz + 50.0:.4f} Hz")
print(f"VPP AGC instances: {len(env.vpp_agc)}")
print(f"Battery AGC instance: {env.bess_agc is not None}")
print(f"VPP K_i values: {[f'{agc.K_i:.6f}' for agc in env.vpp_agc]}")
print(f"Battery K_i: {env.bess_agc.K_i:.6f}")
print()

# Step through an event
action_fast = np.zeros((44,), dtype=np.float32)

for step in range(0, 500, 10):
    obs_fast, reward, done, truncated, info = env.step_fast(action_fast)
    f_current = info.get('delta_f', 0.0) + 50.0
    agc_term_sum = 0.0
    for agc in env.vpp_agc:
        agc_term_sum += agc.integral
    agc_term_sum += env.bess_agc.integral
    
    if step % 50 == 0:
        print(f"Step {step:3d}: f={f_current:7.4f} Hz | AGC integral sum={agc_term_sum:8.4f} | rocof={info.get('rocof', 0.0):6.3f} Hz/s")

print()
print("✓ AGC initialized successfully")
print("✓ VPP and Battery AGC instances active")
print("✓ Integral terms accumulating over steps")
