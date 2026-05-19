"""Optimize PI gains for AGC using scipy.optimize."""
import sys
sys.path.insert(0, r"C:\Users\admin\Desktop\VPP in Ancillary market in 100% renewable islanded microgrid - Copy")

import numpy as np
from scipy.optimize import minimize
from src.env.freq_dynamics import FrequencyDynamics


def simulate_with_params(Kp, Ki, delta_P_pu=0.1, duration=40.0, dt=0.01, event_time=1.0):
    """Simulate frequency response with given PI parameters."""
    fd = FrequencyDynamics()
    fd.Kp = Kp
    fd.Ki = Ki
    fd.bias_rate = 0.0  # Disable bias, use pure PI
    fd.reset()

    freqs = []
    t = 0.0
    while t < duration:
        current_delta_P = delta_P_pu if t >= event_time else 0.0
        state = fd.step(dt, current_delta_P, P_bess_pu=0.0, P_v2g_pu=0.0, P_pv_pu=0.0)
        freqs.append(state.f_hz)
        t += dt

    return np.array(freqs)


def objective(params, delta_P_pu=0.1):
    """Objective function: minimize IAE + oscillation penalty."""
    Kp, Ki = params

    # Constraints
    if Kp < 0 or Ki < 0 or Kp > 1.0 or Ki > 0.5:
        return 1e6

    try:
        freqs = simulate_with_params(Kp, Ki, delta_P_pu)
    except Exception:
        return 1e6

    # Metrics
    delta_f = freqs - 50.0

    # 1. IAE (Integral Absolute Error) after event
    iae = np.sum(np.abs(delta_f[100:])) * 0.01  # after t=1s

    # 2. Settling time penalty (want to reach 49.95 Hz quickly)
    settled_idx = np.where(freqs > 49.95)[0]
    if len(settled_idx) > 0:
        settling_time = settled_idx[0] * 0.01
    else:
        settling_time = 40.0

    # 3. Oscillation penalty (range after t=5s)
    freqs_after_5s = freqs[500:]
    oscillation = np.max(freqs_after_5s) - np.min(freqs_after_5s)

    # 4. Overshoot penalty
    overshoot = max(0, np.max(freqs) - 50.05)

    # Combined objective
    cost = iae + 0.5 * settling_time + 2.0 * oscillation + 10.0 * overshoot

    return cost


def main():
    print("Optimizing PI gains for AGC...")
    print("=" * 50)

    # Initial guess
    x0 = [0.05, 0.03]

    # Bounds
    bounds = [(0.01, 0.5), (0.01, 0.3)]

    # Optimize
    result = minimize(
        objective,
        x0,
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': 100, 'disp': True}
    )

    Kp_opt, Ki_opt = result.x
    print(f"\nOptimal PI gains:")
    print(f"  Kp = {Kp_opt:.4f}")
    print(f"  Ki = {Ki_opt:.4f}")
    print(f"  Cost = {result.fun:.4f}")

    # Test with optimal gains
    print("\n" + "=" * 50)
    print("Testing optimal gains...")
    freqs = simulate_with_params(Kp_opt, Ki_opt)

    print(f"  Nadir: {np.min(freqs):.3f} Hz")
    print(f"  SS: {np.mean(freqs[-100:]):.3f} Hz")
    print(f"  Range (t>5s): {np.max(freqs[500:]) - np.min(freqs[500:]):.3f} Hz")

    # Also test with different initial guesses
    print("\n" + "=" * 50)
    print("Grid search for comparison...")

    best_cost = float('inf')
    best_params = None

    for Kp in np.arange(0.02, 0.20, 0.02):
        for Ki in np.arange(0.02, 0.15, 0.02):
            cost = objective([Kp, Ki])
            if cost < best_cost:
                best_cost = cost
                best_params = (Kp, Ki)

    print(f"\nGrid search best:")
    print(f"  Kp = {best_params[0]:.4f}")
    print(f"  Ki = {best_params[1]:.4f}")
    print(f"  Cost = {best_cost:.4f}")

    freqs = simulate_with_params(best_params[0], best_params[1])
    print(f"  Nadir: {np.min(freqs):.3f} Hz")
    print(f"  SS: {np.mean(freqs[-100:]):.3f} Hz")
    print(f"  Range (t>5s): {np.max(freqs[500:]) - np.min(freqs[500:]):.3f} Hz")


if __name__ == "__main__":
    main()
