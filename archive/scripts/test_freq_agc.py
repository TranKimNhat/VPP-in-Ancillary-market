"""Evaluate AGC & FFR on the VSG frequency model across control settings.

Replaces the legacy synchronous-generator test. Cases now exercise the single
VSG model (no governor/turbine state): backbone-only, AGC-aided recovery, and
AGC + proportional FFR. The old governor-power subplot is dropped (the VSG has
no turbine governor); we plot frequency, RoCoF, and the AGC integral instead.
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
    print("VSG FREQUENCY DYNAMICS: AGC & FFR EVALUATION")
    print("=" * 60)
    print(f"Aggregate virtual inertia H_sys = {env.freq_dyn_lti.h_sys:.3f} s")

    cases = {
        "No FFR (low AGC)": dict(control_gain=0.0, agc_ki=0.01),
        "AGC only": dict(control_gain=0.0, agc_ki=0.05),
        "AGC + FFR": dict(control_gain=0.5, agc_ki=0.05),
    }
    results = {name: simulate_event(env, n_steps=n_steps, **kw, seed=42) for name, kw in cases.items()}

    print(f"\n{'Case':<20}{'Nadir':<10}{'Max|RoCoF|':<12}{'Steady':<10}{'Settle(s)'}")
    print("-" * 62)
    for name, tr in results.items():
        m = trace_metrics(tr)
        st = "inf" if m["settling_time"] == float("inf") else f"{m['settling_time']:.1f}"
        print(f"{name:<20}{m['nadir']:<10.3f}{m['max_abs_rocof']:<12.3f}{m['steady_state']:<10.3f}{st}")

    plt.rcParams.update({"font.size": 11, "axes.labelsize": 12, "font.family": "serif"})
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    styles = ["--", "-", "-"]
    colors = ["#E69F00", "#0072B2", "#009E73"]

    for (name, tr), ls, c in zip(results.items(), styles, colors):
        axes[0].plot(tr["time"], tr["freq"], ls, color=c, lw=2, label=name)
        axes[1].plot(tr["time"], tr["rocof"], ls, color=c, lw=2, label=name)
        axes[2].plot(tr["time"], tr["agc_integral"], ls, color=c, lw=2, label=name)

    axes[0].axhline(50.0, color="gray", ls=":", alpha=0.7)
    axes[0].set(ylabel="Frequency (Hz)", xlabel="Time (s)", title="(a) Frequency")
    axes[1].axhline(-0.5, color="#CC79A7", ls=":", alpha=0.7)
    axes[1].axhline(-1.0, color="red", ls=":", alpha=0.7)
    axes[1].set(ylabel="RoCoF (Hz/s)", xlabel="Time (s)", title="(b) RoCoF")
    axes[2].set(ylabel="AGC integral (Hz·s)", xlabel="Time (s)", title="(c) Secondary control")
    for ax in axes:
        ax.legend(loc="best", framealpha=0.9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = ROOT / "artifacts" / "freq_agc_comparison.png"
    out.parent.mkdir(exist_ok=True)
    plt.savefig(out, dpi=120, bbox_inches="tight")
    print(f"\nPlot saved to: {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
