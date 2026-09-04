"""Plot VSG frequency comparison: No-FFR (backbone VSG only) vs Proposed (VSG + FFR).

Drives the single VSG frequency model through the real environment (legacy
synchronous-generator model removed). "No FFR" lets the grid-forming VSG
backbone respond alone (inertia + droop damping); "Proposed" adds the
proportional FFR control on top.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts._vsg_freq_sim import build_env, simulate_event, trace_metrics


def main():
    n_steps = 60
    env = build_env(seed=42)

    print("=" * 60)
    print("VSG FREQUENCY RESPONSE: No-FFR vs Proposed (FFR)")
    print("=" * 60)
    print(f"Aggregate virtual inertia H_sys = {env.freq_dyn_lti.h_sys:.3f} s")

    no_ffr = simulate_event(env, n_steps=n_steps, control_gain=0.0, seed=42)
    proposed = simulate_event(env, n_steps=n_steps, control_gain=0.5, seed=42)

    m0, m1 = trace_metrics(no_ffr), trace_metrics(proposed)
    print(f"\n{'Metric':<18}{'No FFR':<14}{'Proposed':<14}{'Improvement'}")
    print("-" * 60)
    print(f"{'Nadir (Hz)':<18}{m0['nadir']:<14.3f}{m1['nadir']:<14.3f}{(m1['nadir']-m0['nadir'])*1000:+.1f} mHz")
    print(f"{'Steady (Hz)':<18}{m0['steady_state']:<14.3f}{m1['steady_state']:<14.3f}{(m1['steady_state']-m0['steady_state'])*1000:+.1f} mHz")
    print(f"{'Max|RoCoF|':<18}{m0['max_abs_rocof']:<14.3f}{m1['max_abs_rocof']:<14.3f}{m0['max_abs_rocof']-m1['max_abs_rocof']:+.3f} Hz/s")

    plt.rcParams.update({"font.size": 12, "axes.labelsize": 14, "font.family": "serif"})
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    ax1.plot(no_ffr["time"], no_ffr["freq"], "r--", lw=2, label="No FFR (VSG backbone)")
    ax1.plot(proposed["time"], proposed["freq"], "b-", lw=2.5, label="Proposed (VSG + FFR)")
    ax1.axhline(50.0, color="gray", ls=":", alpha=0.7)
    ax1.axhline(49.8, color="orange", ls=":", alpha=0.7)
    ax1.set_ylabel("Frequency (Hz)")
    ax1.legend(loc="lower right")
    ax1.set_title(f"VSG Frequency Response (H_sys = {env.freq_dyn_lti.h_sys:.2f} s)")
    ax1.grid(True, alpha=0.3)

    ax2.plot(no_ffr["time"], no_ffr["rocof"], "r--", lw=2, label="No FFR")
    ax2.plot(proposed["time"], proposed["rocof"], "b-", lw=2.5, label="Proposed")
    ax2.axhline(0, color="gray", ls="-", alpha=0.5, lw=0.5)
    ax2.axhline(-0.5, color="orange", ls=":", alpha=0.7)
    ax2.axhline(-1.0, color="red", ls=":", alpha=0.7)
    ax2.set_ylabel("RoCoF (Hz/s)")
    ax2.set_xlabel("Time (s)")
    ax2.legend(loc="lower right")
    ax2.set_title("Rate of Change of Frequency")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    out = ROOT / "artifacts" / "freq_droop_vs_proposed.png"
    out.parent.mkdir(exist_ok=True)
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved: {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
