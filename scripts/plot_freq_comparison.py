"""Plot frequency comparison: No FFR (Droop only) vs Proposed (Droop+AGC+FFR)."""
import sys
sys.path.insert(0, r"C:\Users\admin\Desktop\VPP in Ancillary market in 100% renewable islanded microgrid - Copy")

import matplotlib.pyplot as plt
import numpy as np
from src.env.freq_dynamics import FrequencyDynamics
from src.env.events import EventConfig, WIND_BUSES, S_BASE_MW

# Per-DER configuration from official_placement_v3.json
# 9 EVCS units with BESS and V2G
BESS_CAPACITIES_MW = [
    0.325, 0.325, 0.325,  # E1-E3 (Zone 1)
    0.275, 0.275, 0.275,  # E4-E6 (Zone 2)
    0.225, 0.225, 0.225,  # E7-E9 (Zone 4)
]
V2G_CAPACITIES_MW = [
    0.10, 0.10, 0.10,  # E1-E3 (Zone 1)
    0.10, 0.10, 0.10,  # E4-E6 (Zone 2)
    0.075, 0.075, 0.075,  # E7-E9 (Zone 4)
]

# Per-DER droop coefficients (scaled for effective FFR)
# Formula: P_ffr = -k_droop * delta_f / S_BASE (pu)
# Target: ~0.1 pu aggregate FFR at delta_f = -0.15 Hz
# k_droop units: MW/Hz
K_DROOP_BESS = [3.5 * p for p in BESS_CAPACITIES_MW]  # Higher droop for BESS (70% FFR share)
K_DROOP_V2G = [2.0 * p for p in V2G_CAPACITIES_MW]    # Lower droop for V2G (30% FFR share)

# Time constants for DER response
T_BESS = 0.1   # 100ms (fastest)
T_V2G = 0.3    # 300ms (medium)


def simulate_frequency(fd, event: EventConfig, duration=40.0, dt=0.01,
                       use_per_der_droop=False, ffr_delay=0.1, ffr_duration=3.0):
    """Simulate frequency with per-DER droop FFR support.

    Args:
        event: EventConfig with type, delta_P_mw, location, t_inject
        use_per_der_droop: If True, use per-DER droop; else use aggregate
    """
    fd.reset()
    times, freqs, rocofs, p_refs, agc_ints = [], [], [], [], []
    p_bess_list, p_v2g_list, p_pv_list = [], [], []

    ffr_active = False
    ffr_start_time = None
    ffr_ramp_up = 0.2
    ffr_ramp_down = 2.0
    ffr_threshold = 0.15

    # Convert event to delta_P_pu
    delta_P_pu = abs(event.delta_P_mw) / S_BASE_MW
    event_time = event.t_inject

    # Per-DER state for first-order lag filter
    p_bess_der = np.zeros(len(BESS_CAPACITIES_MW))
    p_v2g_der = np.zeros(len(V2G_CAPACITIES_MW))

    t = 0.0
    while t < duration:
        current_delta_P = delta_P_pu if t >= event_time else 0.0
        state = fd.get_state()
        p_bess, p_v2g, p_pv = 0.0, 0.0, 0.0

        if use_per_der_droop:
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

                delta_f = np.clip(state.delta_f_hz, -1.0, 1.0)

                # Per-DER droop control with first-order lag
                for i in range(len(BESS_CAPACITIES_MW)):
                    # Target power: P = -k_droop * delta_f (inject when freq low)
                    p_target = -K_DROOP_BESS[i] * delta_f * ramp_factor / S_BASE_MW
                    # First-order lag: tau * dp/dt + p = p_target
                    alpha = dt / (T_BESS + dt)
                    p_bess_der[i] = (1 - alpha) * p_bess_der[i] + alpha * p_target

                for i in range(len(V2G_CAPACITIES_MW)):
                    p_target = -K_DROOP_V2G[i] * delta_f * ramp_factor / S_BASE_MW
                    alpha = dt / (T_V2G + dt)
                    p_v2g_der[i] = (1 - alpha) * p_v2g_der[i] + alpha * p_target

                # Aggregate per-DER contributions
                p_bess = float(np.sum(p_bess_der))
                p_v2g = float(np.sum(p_v2g_der))
            else:
                # FFR not active - decay to zero
                for i in range(len(BESS_CAPACITIES_MW)):
                    alpha = dt / (T_BESS + dt)
                    p_bess_der[i] = (1 - alpha) * p_bess_der[i]
                for i in range(len(V2G_CAPACITIES_MW)):
                    alpha = dt / (T_V2G + dt)
                    p_v2g_der[i] = (1 - alpha) * p_v2g_der[i]
                p_bess = float(np.sum(p_bess_der))
                p_v2g = float(np.sum(p_v2g_der))

        state = fd.step(dt, current_delta_P, P_bess_pu=p_bess, P_v2g_pu=p_v2g,
                        P_pv_pu=p_pv, ffr_active=ffr_active)
        times.append(state.t)
        freqs.append(state.f_hz)
        rocofs.append(state.rocof_hz_s)
        p_refs.append(state.p_ref_pu)
        agc_ints.append(state.agc_integral_hz_s)
        p_bess_list.append(state.p_bess_pu)
        p_v2g_list.append(state.p_v2g_pu)
        p_pv_list.append(state.p_pv_pu)
        t += dt

    return (np.array(times), np.array(freqs), np.array(rocofs), np.array(p_refs),
            np.array(agc_ints), np.array(p_bess_list), np.array(p_v2g_list), np.array(p_pv_list))


def main():
    # Create gen_trip event: 15% of S_BASE at wind bus 97
    delta_p_pct = 0.15  # 15% generation loss
    delta_p_mw = delta_p_pct * S_BASE_MW  # ~3.14 MW
    event = EventConfig(
        type="gen_trip",
        delta_P_mw=-delta_p_mw,  # Negative = generation loss
        location=WIND_BUSES[0],  # Bus 97 (wind farm)
        t_inject=1.0
    )
    duration = 40.0

    print("=" * 60)
    print("EVENT INJECTION TEST")
    print("=" * 60)
    print(f"Event type: {event.type}")
    print(f"Location: Bus {event.location} (Wind farm)")
    print(f"Magnitude: {abs(event.delta_P_mw):.2f} MW ({delta_p_pct*100:.0f}% of S_BASE={S_BASE_MW:.2f} MW)")
    print(f"Inject time: t={event.t_inject:.1f}s")

    # Case 1: Droop only (No AGC, No FFR) - Baseline
    fd_droop = FrequencyDynamics()
    fd_droop.Kp = 0.0  # Disable AGC
    fd_droop.Ki = 0.0
    fd_droop.Kd = 0.0
    t1, f1, r1, pref1, agc1, _, _, _ = simulate_frequency(fd_droop, event, duration, use_per_der_droop=False)

    # Case 2: Proposed (Droop + AGC + per-DER FFR)
    # Per-DER droop control:
    #   - 9 BESS units with k_droop_i = 0.15 * P_rated_i (T=100ms)
    #   - 9 V2G units with k_droop_i = 0.10 * P_rated_i (T=300ms)
    # FFR duration: 3s (handoff to AGC at t=4s)
    fd_proposed = FrequencyDynamics()
    t2, f2, r2, pref2, agc2, p_bess2, p_v2g2, p_pv2 = simulate_frequency(
        fd_proposed, event, duration,
        use_per_der_droop=True, ffr_delay=0.1, ffr_duration=3.0)

    # Case 3: AGC only (no FFR) - to isolate FFR effect
    fd_agc_only = FrequencyDynamics()
    t3, f3, r3, pref3, agc3, _, _, _ = simulate_frequency(fd_agc_only, event, duration, use_per_der_droop=False)
    print(f"\nAGC only (no FFR): SS = {f3[-100:].mean():.3f} Hz, Min = {f3[t3>5].min():.3f}, Max = {f3[t3>5].max():.3f}")

    # Print per-DER FFR contribution
    print(f"\nPer-DER Droop FFR (Proposed):")
    print(f"  BESS: {len(BESS_CAPACITIES_MW)} units, total {sum(BESS_CAPACITIES_MW):.3f} MW")
    print(f"    k_droop: {[f'{k:.4f}' for k in K_DROOP_BESS]}")
    print(f"    Peak aggregate: {np.max(np.abs(p_bess2)):.4f} pu (T={T_BESS*1000:.0f}ms)")
    print(f"  V2G: {len(V2G_CAPACITIES_MW)} units, total {sum(V2G_CAPACITIES_MW):.3f} MW")
    print(f"    k_droop: {[f'{k:.4f}' for k in K_DROOP_V2G]}")
    print(f"    Peak aggregate: {np.max(np.abs(p_v2g2)):.4f} pu (T={T_V2G*1000:.0f}ms)")

    # Debug: print key values during transient (t=2-15s)
    print("\nDebug values during transient:")
    for ti in [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 15, 20, 30]:
        idx = int(ti / 0.01)
        if idx < len(t2):
            print(f"  t={ti:.0f}s: f={f2[idx]:.3f} Hz, p_ref={pref2[idx]:.4f} pu, agc_int={agc2[idx]:.3f}")

    # Check fast variations after t=5s
    print("\nFrequency variance check (t>5s):")
    mask = t2 > 5.0
    print(f"  Min freq: {f2[mask].min():.3f}, Max freq: {f2[mask].max():.3f}")
    print(f"  Range: {f2[mask].max() - f2[mask].min():.3f} Hz")
    # Print samples around t=11-12s with high resolution
    print("\nHigh-res values t=10.9 to 11.5s (every 0.05s):")
    for ti in np.arange(10.9, 11.5, 0.05):
        idx = int(ti / 0.01)
        if idx < len(t2):
            print(f"  t={ti:.2f}s: f={f2[idx]:.4f} Hz")

    # Metrics
    nadir_droop = f1.min()
    nadir_proposed = f2.min()
    ss_droop = f1[-100:].mean()
    ss_proposed = f2[-100:].mean()

    print("=" * 50)
    print("FREQUENCY RESPONSE COMPARISON")
    print("=" * 50)
    print(f"Disturbance: {delta_p_pct*100:.0f}% generation loss ({abs(event.delta_P_mw):.2f} MW)")
    print(f"H_sys: {fd_proposed.H_SYS:.3f} s (low inertia)")
    print()
    print(f"{'Metric':<20} {'Droop Only':<15} {'Proposed':<15} {'Improvement':<15}")
    print("-" * 65)
    print(f"{'Nadir':<20} {nadir_droop:.3f} Hz{'':<6} {nadir_proposed:.3f} Hz{'':<6} +{(nadir_proposed-nadir_droop)*1000:.1f} mHz")
    print(f"{'Steady-state':<20} {ss_droop:.3f} Hz{'':<6} {ss_proposed:.3f} Hz{'':<6} +{(ss_proposed-ss_droop)*1000:.1f} mHz")
    print(f"{'Max RoCoF':<20} {np.abs(r1).max():.3f} Hz/s{'':<4} {np.abs(r2).max():.3f} Hz/s")

    # Plot
    plt.rcParams.update({
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 14,
        'legend.fontsize': 11,
        'font.family': 'serif',
    })

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # Frequency plot
    ax1.plot(t1, f1, 'r--', linewidth=2, label='Droop Only (No AGC/FFR)')
    ax1.plot(t2, f2, 'b-', linewidth=2.5, label='Proposed (Droop + AGC + FFR)')

    ax1.axhline(50.0, color='gray', linestyle=':', alpha=0.7, linewidth=1)
    ax1.axhline(49.8, color='orange', linestyle=':', alpha=0.7, linewidth=1)
    ax1.axvline(event.t_inject, color='red', linestyle='--', alpha=0.5, linewidth=1.5)

    # Annotations
    ax1.annotate(f'Nadir: {nadir_droop:.2f} Hz', xy=(t1[f1.argmin()], nadir_droop),
                 xytext=(t1[f1.argmin()]+2, nadir_droop-0.03), fontsize=10, color='red',
                 arrowprops=dict(arrowstyle='->', color='red', lw=1))
    ax1.annotate(f'Nadir: {nadir_proposed:.2f} Hz', xy=(t2[f2.argmin()], nadir_proposed),
                 xytext=(t2[f2.argmin()]+2, nadir_proposed+0.02), fontsize=10, color='blue',
                 arrowprops=dict(arrowstyle='->', color='blue', lw=1))
    ax1.annotate(f'Gen Trip\nBus {event.location}\n({delta_p_pct*100:.0f}% loss)',
                 xy=(event.t_inject, 50.0),
                 xytext=(event.t_inject+0.5, 50.03), fontsize=9, color='red')

    # FFR active window
    ax1.fill_between([1.1, 4.6], 49.5, 50.1, alpha=0.1, color='green', label='FFR Active (0-3.5s)')
    # AGC activation (after 2s delay)
    ax1.axvline(event.t_inject + fd_proposed.agc_activation_delay_s, color='purple', linestyle='-.',
                alpha=0.6, linewidth=1.5, label=f'AGC Activation (t={event.t_inject + fd_proposed.agc_activation_delay_s:.0f}s)')

    ax1.set_ylabel('Frequency (Hz)')
    ax1.set_ylim([49.55, 50.05])
    ax1.legend(loc='lower right')
    ax1.set_title(f'Frequency Response to {delta_p_pct*100:.0f}% Gen Trip at Bus {event.location} (H_sys = {fd_proposed.H_SYS:.2f}s)')
    ax1.grid(True, alpha=0.3)

    # RoCoF plot
    ax2.plot(t1, r1, 'r--', linewidth=2, label='Droop Only')
    ax2.plot(t2, r2, 'b-', linewidth=2.5, label='Proposed')

    ax2.axhline(0, color='gray', linestyle='-', alpha=0.5, linewidth=0.5)
    ax2.axhline(-0.5, color='orange', linestyle=':', alpha=0.7, linewidth=1.5)
    ax2.axhline(-1.0, color='red', linestyle=':', alpha=0.7, linewidth=1.5)
    ax2.axvline(event.t_inject, color='red', linestyle='--', alpha=0.5, linewidth=1.5)

    ax2.set_ylabel('RoCoF (Hz/s)')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylim([-2.5, 0.5])
    ax2.legend(loc='lower right')
    ax2.set_title('Rate of Change of Frequency')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(r"C:\Users\admin\Desktop\VPP in Ancillary market in 100% renewable islanded microgrid - Copy\artifacts\freq_droop_vs_proposed.png", dpi=150, bbox_inches='tight')
    print(f"\nPlot saved: artifacts/freq_droop_vs_proposed.png")
    plt.show()


if __name__ == "__main__":
    main()
