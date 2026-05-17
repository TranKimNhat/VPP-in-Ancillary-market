"""Test frequency dynamics: Compare droop vs no-droop to evaluate AGC and FFR opportunity."""

import sys
sys.path.insert(0, r"C:\Users\admin\Desktop\VPP in Ancillary market in 100% renewable islanded microgrid - Copy")

import matplotlib.pyplot as plt
import numpy as np
from dataclasses import dataclass
from copy import deepcopy

from src.env.freq_dynamics import FrequencyDynamics, FrequencyState


def simulate_frequency(
    fd: FrequencyDynamics,
    delta_P_pu: float,
    duration: float = 30.0,
    dt: float = 0.01,
    event_time: float = 1.0,
    ffr_power: float = 0.0,
    ffr_delay: float = 0.1,
    ffr_duration: float = 2.0,
) -> dict:
    """Simulate frequency response to a power disturbance.

    Args:
        fd: FrequencyDynamics instance
        delta_P_pu: Power disturbance in pu (positive = generation loss)
        duration: Total simulation time (s)
        dt: Time step (s)
        event_time: Time when disturbance occurs (s)
        ffr_power: FFR power injection in pu (0 = no FFR)
        ffr_delay: FFR activation delay after frequency threshold (s)
        ffr_duration: FFR active duration (s)

    Returns:
        Dictionary with time series data
    """
    fd.reset()

    times = []
    freqs = []
    delta_fs = []
    rocofs = []
    p_govs = []
    p_refs = []
    agc_integrals = []

    ffr_active = False
    ffr_start_time = None

    t = 0.0
    while t < duration:
        # Determine current disturbance
        if t >= event_time:
            current_delta_P = delta_P_pu
        else:
            current_delta_P = 0.0

        # FFR logic: proportional droop-based response with smooth ramp
        state = fd.get_state()
        p_bess = 0.0

        if ffr_power != 0.0:
            # FFR activation threshold
            ffr_threshold = 0.15  # Hz
            ffr_ramp_up = 0.2     # Fast ramp-up for quick response
            ffr_ramp_down = 1.0   # Slow ramp-down for smooth AGC handoff

            if not ffr_active and abs(state.delta_f_hz) > ffr_threshold and t >= event_time + ffr_delay:
                ffr_active = True
                ffr_start_time = t

            if ffr_active:
                elapsed = t - ffr_start_time

                # Ramp-up phase (fast)
                if elapsed < ffr_ramp_up:
                    ramp_factor = elapsed / ffr_ramp_up
                # Sustain phase
                elif elapsed < ffr_duration - ffr_ramp_down:
                    ramp_factor = 1.0
                # Ramp-down phase (slow for smooth AGC handoff)
                elif elapsed < ffr_duration:
                    ramp_factor = (ffr_duration - elapsed) / ffr_ramp_down
                else:
                    ramp_factor = 0.0
                    ffr_active = False

                # Proportional FFR: power proportional to frequency deviation (droop-like)
                # Clamp delta_f to avoid excessive response
                delta_f_clamped = np.clip(state.delta_f_hz, -1.0, 1.0)
                p_bess = -delta_f_clamped * ffr_power * ramp_factor * 2.0  # Scale factor

        # Step simulation
        state = fd.step(dt, current_delta_P, P_bess_pu=p_bess)

        times.append(state.t)
        freqs.append(state.f_hz)
        delta_fs.append(state.delta_f_hz)
        rocofs.append(state.rocof_hz_s)
        p_govs.append(state.p_gov_pu)
        p_refs.append(state.p_ref_pu)
        agc_integrals.append(state.agc_integral_hz_s)

        t += dt

    return {
        "time": np.array(times),
        "freq": np.array(freqs),
        "delta_f": np.array(delta_fs),
        "rocof": np.array(rocofs),
        "p_gov": np.array(p_govs),
        "p_ref": np.array(p_refs),
        "agc_integral": np.array(agc_integrals),
    }


def main():
    # Disturbance: 10% generation loss (0.1 pu)
    delta_P_pu = 0.1
    duration = 60.0
    event_time = 1.0

    print("=" * 60)
    print("FREQUENCY DYNAMICS TEST: AGC & FFR EVALUATION")
    print("=" * 60)
    print(f"Disturbance: {delta_P_pu * 100:.1f}% generation loss at t={event_time}s")
    print(f"Simulation duration: {duration}s")
    print()

    # Case 1: With Droop + AGC (baseline)
    fd_droop = FrequencyDynamics()
    print(f"System parameters (with droop):")
    print(f"  H_SYS = {fd_droop.H_SYS:.3f} s")
    print(f"  D_SYS = {fd_droop.D_SYS:.3f}")
    print(f"  R_SYS = {fd_droop.R_SYS:.4f}")
    print(f"  AGC: Kp={fd_droop.Kp}, Ki={fd_droop.Ki}")
    print()

    result_droop = simulate_frequency(fd_droop, delta_P_pu, duration, event_time=event_time)

    # Case 2: No Droop (R -> infinity, only load damping)
    fd_no_droop = FrequencyDynamics()
    fd_no_droop.R = 1000.0  # Effectively infinite R = no droop
    fd_no_droop.Kp = 0.0    # Disable AGC
    fd_no_droop.Ki = 0.0

    result_no_droop = simulate_frequency(fd_no_droop, delta_P_pu, duration, event_time=event_time)

    # Case 3: Droop only (no AGC)
    fd_droop_only = FrequencyDynamics()
    fd_droop_only.Kp = 0.0
    fd_droop_only.Ki = 0.0

    result_droop_only = simulate_frequency(fd_droop_only, delta_P_pu, duration, event_time=event_time)

    # Case 4: Droop + AGC + FFR
    fd_with_ffr = FrequencyDynamics()
    result_with_ffr = simulate_frequency(
        fd_with_ffr, delta_P_pu, duration, event_time=event_time,
        ffr_power=0.20, ffr_delay=0.1, ffr_duration=3.5  # 0.20 pu = 3.14 MW = 35% of 9.02 MW BESS, longer for smooth AGC handoff
    )

    # Print metrics
    print("=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)

    cases = [
        ("No Droop (D only)", result_no_droop),
        ("Droop only (no AGC)", result_droop_only),
        ("Droop + AGC", result_droop),
        ("Droop + AGC + FFR", result_with_ffr),
    ]

    for name, r in cases:
        nadir = r["freq"].min()
        nadir_time = r["time"][r["freq"].argmin()]
        max_rocof = np.abs(r["rocof"]).max()
        steady_state = r["freq"][-100:].mean()
        settling_idx = np.where(np.abs(r["delta_f"]) < 0.1)[0]
        settling_time = settling_idx[0] * 0.01 if len(settling_idx) > 0 and settling_idx[0] > 100 else float('inf')

        print(f"\n{name}:")
        print(f"  Nadir: {nadir:.3f} Hz at t={nadir_time:.2f}s (delta_f = {nadir - 50:.3f} Hz)")
        print(f"  Max RoCoF: {max_rocof:.3f} Hz/s")
        print(f"  Steady-state: {steady_state:.3f} Hz (delta_f = {steady_state - 50:.3f} Hz)")

    # FFR opportunity analysis
    print("\n" + "=" * 60)
    print("FFR OPPORTUNITY ANALYSIS")
    print("=" * 60)

    # Time window where |delta_f| > 0.2 Hz (FFR trigger threshold)
    ffr_needed_mask = np.abs(result_droop["delta_f"]) > 0.2
    if ffr_needed_mask.any():
        ffr_start = result_droop["time"][ffr_needed_mask][0]
        ffr_end_idx = np.where(~ffr_needed_mask & (result_droop["time"] > ffr_start))[0]
        ffr_end = result_droop["time"][ffr_end_idx[0]] if len(ffr_end_idx) > 0 else result_droop["time"][-1]
        print(f"FFR activation window: t={ffr_start:.2f}s to t={ffr_end:.2f}s")
        print(f"FFR duration needed: {ffr_end - ffr_start:.2f}s")

    # RoCoF > 0.5 Hz/s window (critical for FFR)
    rocof_critical_mask = np.abs(result_droop["rocof"]) > 0.5
    if rocof_critical_mask.any():
        rocof_critical_times = result_droop["time"][rocof_critical_mask]
        print(f"Critical RoCoF (>0.5 Hz/s) window: t={rocof_critical_times[0]:.2f}s to t={rocof_critical_times[-1]:.2f}s")

    # ==================== PAPER-QUALITY PLOT ====================
    plt.rcParams.update({
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 13,
        'legend.fontsize': 10,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'font.family': 'serif',
    })

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))

    # Define styles for paper
    styles = {
        'droop_only': {'color': '#E69F00', 'linestyle': '--', 'linewidth': 2, 'label': 'Droop only'},
        'droop_agc': {'color': '#0072B2', 'linestyle': '-', 'linewidth': 2, 'label': 'Droop + AGC'},
        'droop_agc_ffr': {'color': '#009E73', 'linestyle': '-', 'linewidth': 2.5, 'label': 'Droop + AGC + FFR'},
    }

    # ===== Plot 1: Frequency (zoom on transient, 0-15s) =====
    ax1 = axes[0, 0]
    t_max = 15.0
    mask_droop = result_droop_only["time"] <= t_max
    mask_agc = result_droop["time"] <= t_max
    mask_ffr = result_with_ffr["time"] <= t_max

    ax1.plot(result_droop_only["time"][mask_droop], result_droop_only["freq"][mask_droop], **styles['droop_only'])
    ax1.plot(result_droop["time"][mask_agc], result_droop["freq"][mask_agc], **styles['droop_agc'])
    ax1.plot(result_with_ffr["time"][mask_ffr], result_with_ffr["freq"][mask_ffr], **styles['droop_agc_ffr'])

    ax1.axhline(50.0, color='gray', linestyle=':', alpha=0.7, linewidth=1)
    ax1.axhline(49.8, color='#CC79A7', linestyle=':', alpha=0.7, linewidth=1, label='Warning (49.8 Hz)')
    ax1.axvline(event_time, color='red', linestyle='--', alpha=0.6, linewidth=1.5)

    # Annotations
    nadir_agc = result_droop["freq"].min()
    nadir_ffr = result_with_ffr["freq"].min()
    nadir_time_agc = result_droop["time"][result_droop["freq"].argmin()]
    nadir_time_ffr = result_with_ffr["time"][result_with_ffr["freq"].argmin()]

    ax1.annotate(f'Nadir: {nadir_agc:.2f} Hz', xy=(nadir_time_agc, nadir_agc),
                 xytext=(nadir_time_agc + 1.5, nadir_agc - 0.08),
                 fontsize=9, color='#0072B2',
                 arrowprops=dict(arrowstyle='->', color='#0072B2', lw=1))
    ax1.annotate(f'Nadir: {nadir_ffr:.2f} Hz', xy=(nadir_time_ffr, nadir_ffr),
                 xytext=(nadir_time_ffr + 2, nadir_ffr + 0.05),
                 fontsize=9, color='#009E73',
                 arrowprops=dict(arrowstyle='->', color='#009E73', lw=1))
    ax1.annotate('Disturbance', xy=(event_time, 50.0), xytext=(event_time + 0.5, 50.05),
                 fontsize=9, color='red')

    # FFR active region shading
    ax1.axvspan(1.1, 1.6, alpha=0.15, color='green', label='FFR active')

    ax1.set_ylabel('Frequency (Hz)')
    ax1.set_xlabel('Time (s)')
    ax1.set_xlim([0, t_max])
    ax1.set_ylim([49.5, 50.15])
    ax1.legend(loc='lower right', framealpha=0.9)
    ax1.set_title(f'(a) Frequency Response (H_sys = {fd_droop.H_SYS:.2f}s)')
    ax1.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)

    # ===== Plot 2: RoCoF (zoom on transient, 0-5s) =====
    ax2 = axes[0, 1]
    t_max_rocof = 5.0
    mask_droop_r = result_droop_only["time"] <= t_max_rocof
    mask_agc_r = result_droop["time"] <= t_max_rocof
    mask_ffr_r = result_with_ffr["time"] <= t_max_rocof

    ax2.plot(result_droop_only["time"][mask_droop_r], result_droop_only["rocof"][mask_droop_r], **styles['droop_only'])
    ax2.plot(result_droop["time"][mask_agc_r], result_droop["rocof"][mask_agc_r], **styles['droop_agc'])
    ax2.plot(result_with_ffr["time"][mask_ffr_r], result_with_ffr["rocof"][mask_ffr_r], **styles['droop_agc_ffr'])

    ax2.axhline(0, color='gray', linestyle='-', alpha=0.5, linewidth=0.5)
    ax2.axhline(-0.5, color='#CC79A7', linestyle=':', alpha=0.7, linewidth=1.5, label='RoCoF limit (-0.5 Hz/s)')
    ax2.axhline(-1.0, color='red', linestyle=':', alpha=0.7, linewidth=1.5, label='Critical (-1.0 Hz/s)')
    ax2.axvline(event_time, color='red', linestyle='--', alpha=0.6, linewidth=1.5)

    # Annotation for max RoCoF
    max_rocof_idx = np.argmin(result_droop["rocof"])
    max_rocof_val = result_droop["rocof"][max_rocof_idx]
    max_rocof_time = result_droop["time"][max_rocof_idx]
    ax2.annotate(f'Max: {max_rocof_val:.2f} Hz/s', xy=(max_rocof_time, max_rocof_val),
                 xytext=(max_rocof_time + 0.5, max_rocof_val + 0.3),
                 fontsize=9, color='#0072B2',
                 arrowprops=dict(arrowstyle='->', color='#0072B2', lw=1))

    ax2.axvspan(1.0, 1.65, alpha=0.15, color='red', label='Critical RoCoF window')

    ax2.set_ylabel('RoCoF (Hz/s)')
    ax2.set_xlabel('Time (s)')
    ax2.set_xlim([0, t_max_rocof])
    ax2.set_ylim([-2.5, 0.5])
    ax2.legend(loc='lower right', framealpha=0.9)
    ax2.set_title('(b) Rate of Change of Frequency')
    ax2.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)

    # ===== Plot 3: Governor Power Response =====
    ax3 = axes[1, 0]
    t_max_gov = 30.0
    mask_gov = result_droop["time"] <= t_max_gov

    ax3.plot(result_droop_only["time"][mask_gov], result_droop_only["p_gov"][mask_gov], **styles['droop_only'])
    ax3.plot(result_droop["time"][mask_gov], result_droop["p_gov"][mask_gov], **styles['droop_agc'])
    ax3.plot(result_with_ffr["time"][mask_gov], result_with_ffr["p_gov"][mask_gov], **styles['droop_agc_ffr'])
    ax3.plot(result_droop["time"][mask_gov], result_droop["p_ref"][mask_gov],
             color='#0072B2', linestyle=':', linewidth=1.5, alpha=0.7, label='AGC setpoint (P_ref)')

    ax3.axvline(event_time, color='red', linestyle='--', alpha=0.6, linewidth=1.5)
    ax3.axhline(delta_P_pu, color='gray', linestyle=':', alpha=0.5, label=f'Disturbance ({delta_P_pu} pu)')

    ax3.set_ylabel('Governor Power (pu)')
    ax3.set_xlabel('Time (s)')
    ax3.set_xlim([0, t_max_gov])
    ax3.legend(loc='lower right', framealpha=0.9)
    ax3.set_title('(c) Governor Response')
    ax3.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)

    # ===== Plot 4: Full timeline comparison =====
    ax4 = axes[1, 1]
    ax4.plot(result_droop_only["time"], result_droop_only["freq"], **styles['droop_only'])
    ax4.plot(result_droop["time"], result_droop["freq"], **styles['droop_agc'])
    ax4.plot(result_with_ffr["time"], result_with_ffr["freq"], **styles['droop_agc_ffr'])

    ax4.axhline(50.0, color='gray', linestyle=':', alpha=0.7, linewidth=1, label='Nominal (50 Hz)')
    ax4.axvline(event_time, color='red', linestyle='--', alpha=0.6, linewidth=1.5)

    # Steady-state annotations
    ss_droop = result_droop_only["freq"][-100:].mean()
    ss_agc = result_droop["freq"][-100:].mean()
    ax4.annotate(f'SS: {ss_droop:.2f} Hz', xy=(55, ss_droop), fontsize=9, color='#E69F00', va='center')
    ax4.annotate(f'SS: {ss_agc:.2f} Hz', xy=(55, ss_agc), fontsize=9, color='#0072B2', va='center')

    ax4.set_ylabel('Frequency (Hz)')
    ax4.set_xlabel('Time (s)')
    ax4.set_xlim([0, 60])
    ax4.set_ylim([49.5, 50.15])
    ax4.legend(loc='lower right', framealpha=0.9)
    ax4.set_title('(d) Full Timeline - AGC Restores Frequency')
    ax4.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)

    plt.tight_layout()
    plt.savefig(r"C:\Users\admin\Desktop\VPP in Ancillary market in 100% renewable islanded microgrid - Copy\artifacts\freq_agc_comparison.png", dpi=120, bbox_inches='tight')
    print(f"\nPlot saved to: artifacts/freq_agc_comparison.png")
    plt.show()


if __name__ == "__main__":
    main()
