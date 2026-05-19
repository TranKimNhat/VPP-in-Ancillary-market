"""Optimize all control parameters: Primary droop, FFR, and AGC."""
import sys
sys.path.insert(0, r"C:\Users\admin\Desktop\VPP in Ancillary market in 100% renewable islanded microgrid - Copy")

import numpy as np
from scipy.optimize import minimize, differential_evolution
from src.env.freq_dynamics import FrequencyDynamics


def simulate_full(R, k_ffr, Kp, Ki, delta_P_pu=0.15, duration=40.0, dt=0.01, event_time=1.0):
    """Simulate frequency response with all control parameters."""
    fd = FrequencyDynamics()

    # Set primary droop (R) for all GFM units
    for gfm_id in fd._gfm_params:
        fd._gfm_params[gfm_id]["R"] = R
    fd._update_system_params()

    # Set AGC PI gains
    fd.Kp = Kp
    fd.Ki = Ki
    fd.reset()

    # FFR parameters
    ffr_delay = 0.1
    ffr_duration = 3.0
    ffr_ramp_up = 0.2
    ffr_ramp_down = 2.0
    ffr_threshold = 0.15
    ffr_active = False
    ffr_start_time = None

    freqs, rocofs, nadirs = [], [], []
    t = 0.0

    while t < duration:
        current_delta_P = delta_P_pu if t >= event_time else 0.0
        state = fd.get_state()

        # FFR with droop-based control: P_ffr = -k_ffr * delta_f
        p_bess, p_v2g = 0.0, 0.0

        if k_ffr > 0:
            if not ffr_active and abs(state.delta_f_hz) > ffr_threshold and t >= event_time + ffr_delay:
                ffr_active = True
                ffr_start_time = t

            if ffr_active:
                elapsed = t - ffr_start_time
                if elapsed < ffr_ramp_up:
                    ramp_factor = elapsed / ffr_ramp_up
                elif elapsed < ffr_duration - ffr_ramp_down:
                    ramp_factor = 1.0
                elif elapsed < ffr_duration:
                    ramp_factor = (ffr_duration - elapsed) / ffr_ramp_down
                else:
                    ramp_factor = 0.0
                    ffr_active = False

                # Droop-based FFR: P = -k_ffr * delta_f
                delta_f_clamped = np.clip(state.delta_f_hz, -1.0, 1.0)
                total_ffr = -k_ffr * delta_f_clamped * ramp_factor
                p_bess = total_ffr * 0.70  # BESS 70%
                p_v2g = total_ffr * 0.30   # V2G 30%

        state = fd.step(dt, current_delta_P, P_bess_pu=p_bess, P_v2g_pu=p_v2g,
                        P_pv_pu=0.0, ffr_active=ffr_active)
        freqs.append(state.f_hz)
        rocofs.append(abs(state.rocof_hz_s))
        t += dt

    return np.array(freqs), np.array(rocofs)


def objective(params, delta_P_pu=0.15):
    """Multi-objective: minimize frequency deviation + oscillation + RoCoF."""
    R, k_ffr, Kp, Ki = params

    # Constraints
    if R < 0.02 or R > 0.10:
        return 1e6
    if k_ffr < 0 or k_ffr > 1.0:
        return 1e6
    if Kp < 0.01 or Kp > 1.0:
        return 1e6
    if Ki < 0.01 or Ki > 0.5:
        return 1e6

    try:
        freqs, rocofs = simulate_full(R, k_ffr, Kp, Ki, delta_P_pu)
    except Exception as e:
        return 1e6

    # Metrics
    delta_f = freqs - 50.0

    # 1. Nadir (minimum frequency)
    nadir = np.min(freqs)
    nadir_penalty = max(0, 49.5 - nadir) * 100  # Heavy penalty if below 49.5 Hz

    # 2. IAE after event
    iae = np.sum(np.abs(delta_f[100:])) * 0.01

    # 3. Settling time (reach 49.95 Hz)
    settled_idx = np.where(freqs > 49.95)[0]
    if len(settled_idx) > 0:
        settling_time = settled_idx[0] * 0.01
    else:
        settling_time = 40.0

    # 4. Oscillation (range after t=5s)
    freqs_after_5s = freqs[500:]
    oscillation = np.max(freqs_after_5s) - np.min(freqs_after_5s)

    # 5. Overshoot penalty
    overshoot = max(0, np.max(freqs) - 50.05) * 10

    # 6. Max RoCoF penalty
    max_rocof = np.max(rocofs)
    rocof_penalty = max(0, max_rocof - 2.0) * 5  # Penalty if RoCoF > 2 Hz/s

    # Combined objective
    cost = (
        iae * 1.0 +
        settling_time * 0.3 +
        oscillation * 3.0 +
        overshoot * 5.0 +
        nadir_penalty +
        rocof_penalty +
        (50.0 - nadir) * 2.0  # Reward better nadir
    )

    return cost


def main():
    print("=" * 60)
    print("OPTIMIZING ALL CONTROL PARAMETERS")
    print("=" * 60)
    print("\nParameters to optimize:")
    print("  - R (Primary droop): 0.02 - 0.10")
    print("  - k_ffr (FFR gain): 0.0 - 1.0")
    print("  - Kp (AGC proportional): 0.01 - 1.0")
    print("  - Ki (AGC integral): 0.01 - 0.5")

    # Bounds: [R, k_ffr, Kp, Ki]
    bounds = [(0.02, 0.10), (0.1, 0.8), (0.1, 0.8), (0.05, 0.4)]

    print("\n1. Running Differential Evolution (global optimization)...")
    result_de = differential_evolution(
        objective,
        bounds,
        maxiter=50,
        seed=42,
        disp=True,
        workers=1,
        polish=True
    )

    R_opt, k_ffr_opt, Kp_opt, Ki_opt = result_de.x
    print(f"\nDE Optimal parameters:")
    print(f"  R     = {R_opt:.4f} (Primary droop)")
    print(f"  k_ffr = {k_ffr_opt:.4f} (FFR gain)")
    print(f"  Kp    = {Kp_opt:.4f} (AGC P)")
    print(f"  Ki    = {Ki_opt:.4f} (AGC I)")
    print(f"  Cost  = {result_de.fun:.4f}")

    # Test with optimal parameters
    print("\n" + "=" * 60)
    print("TESTING OPTIMAL PARAMETERS")
    print("=" * 60)

    freqs, rocofs = simulate_full(R_opt, k_ffr_opt, Kp_opt, Ki_opt)

    print(f"\nResults:")
    print(f"  Nadir: {np.min(freqs):.3f} Hz")
    print(f"  SS: {np.mean(freqs[-100:]):.3f} Hz")
    print(f"  Max freq: {np.max(freqs):.3f} Hz")
    print(f"  Range (t>5s): {np.max(freqs[500:]) - np.min(freqs[500:]):.4f} Hz")
    print(f"  Max RoCoF: {np.max(rocofs):.3f} Hz/s")
    print(f"  Settling time: {np.where(freqs > 49.95)[0][0] * 0.01:.1f}s" if np.any(freqs > 49.95) else "  Settling time: >40s")

    # Compare with baseline (no FFR)
    print("\n" + "=" * 60)
    print("COMPARISON: Baseline vs Optimized")
    print("=" * 60)

    # Baseline: current values, no FFR
    freqs_base, rocofs_base = simulate_full(0.05, 0.0, 0.50, 0.30)

    print(f"\n{'Metric':<20} {'Baseline':<15} {'Optimized':<15} {'Improvement':<15}")
    print("-" * 65)
    print(f"{'Nadir':<20} {np.min(freqs_base):.3f} Hz{'':<6} {np.min(freqs):.3f} Hz{'':<6} {(np.min(freqs)-np.min(freqs_base))*1000:+.1f} mHz")
    print(f"{'SS':<20} {np.mean(freqs_base[-100:]):.3f} Hz{'':<6} {np.mean(freqs[-100:]):.3f} Hz")
    print(f"{'Range (t>5s)':<20} {np.max(freqs_base[500:])-np.min(freqs_base[500:]):.4f} Hz{'':<4} {np.max(freqs[500:])-np.min(freqs[500:]):.4f} Hz")
    print(f"{'Max RoCoF':<20} {np.max(rocofs_base):.3f} Hz/s{'':<4} {np.max(rocofs):.3f} Hz/s")

    # Save optimal parameters
    print("\n" + "=" * 60)
    print("RECOMMENDED CONFIG FOR freq_dynamics.py:")
    print("=" * 60)
    print(f"""
    R: float = {R_opt:.4f},      # Primary droop (optimized)
    Kp: float = {Kp_opt:.4f},    # AGC P gain (optimized)
    Ki: float = {Ki_opt:.4f},    # AGC I gain (optimized)

    # FFR in plot_freq_comparison.py:
    ffr_power = {k_ffr_opt:.4f}  # FFR gain (optimized)
    """)


if __name__ == "__main__":
    main()
