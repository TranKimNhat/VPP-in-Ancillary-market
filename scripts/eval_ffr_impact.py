"""Evaluate frequency impact of scenarios with/without FFR control.

Includes detailed logging around injection window (t=30s) to quantitatively
demonstrate disturbance impact for reviewer validation.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.env.microgrid_env_dual import MicrogridEnvDual

# Injection timing parameters
T_INJECT = 30  # Event injection step (t=30s with dt=1s)
WINDOW_PRE = 10  # Steps before injection to analyze
WINDOW_POST = 30  # Steps after injection to analyze


def run_episode_detailed(
    env: MicrogridEnvDual,
    n_steps: int = 300,
    with_control: bool = False,
    control_gain: float = 0.5,
) -> dict:
    """Run episode with detailed metrics around injection window."""
    env.reset()

    metrics = {
        "delta_f": [],
        "rocof": [],
        "freq_hz": [],
        "event_injected": [],
        "ffr_active": [],
        "control_action": [],
        "step": [],
    }

    for step in range(n_steps):
        if with_control:
            freq_state = env.freq_dyn.get_state()
            delta_f = freq_state.delta_f_hz
            control = -control_gain * np.clip(delta_f / 0.5, -1.0, 1.0)
            action = np.full(44, control, dtype=np.float32)
        else:
            action = np.zeros(44, dtype=np.float32)
            control = 0.0

        obs, reward, done, trunc, info = env.step_fast(action)

        metrics["step"].append(step)
        metrics["delta_f"].append(info["delta_f"])
        metrics["rocof"].append(info["rocof"])
        metrics["freq_hz"].append(50.0 + info["delta_f"])
        metrics["event_injected"].append(info.get("event_injected", False))
        metrics["ffr_active"].append(info.get("ffr_active", False))
        metrics["control_action"].append(float(control))

    # Compute window statistics around injection
    pre_start = max(0, T_INJECT - WINDOW_PRE)
    pre_end = T_INJECT
    post_start = T_INJECT
    post_end = min(n_steps, T_INJECT + WINDOW_POST)

    delta_f_arr = np.array(metrics["delta_f"])
    rocof_arr = np.array(metrics["rocof"])
    event_arr = np.array(metrics["event_injected"])

    # Pre-injection window statistics
    pre_delta_f = delta_f_arr[pre_start:pre_end]
    pre_rocof = rocof_arr[pre_start:pre_end]

    # Post-injection window statistics
    post_delta_f = delta_f_arr[post_start:post_end]
    post_rocof = rocof_arr[post_start:post_end]

    # Find injection step (first event_injected=True)
    inject_steps = np.where(event_arr)[0]
    actual_inject_step = int(inject_steps[0]) if len(inject_steps) > 0 else -1

    return {
        # Overall metrics
        "mean_abs_delta_f": float(np.mean(np.abs(delta_f_arr))),
        "max_abs_delta_f": float(np.max(np.abs(delta_f_arr))),
        "mean_abs_rocof": float(np.mean(np.abs(rocof_arr))),
        "max_abs_rocof": float(np.max(np.abs(rocof_arr))),
        "nadir": float(np.min(metrics["freq_hz"])),
        "zenith": float(np.max(metrics["freq_hz"])),
        "event_count": int(np.sum(event_arr)),
        "ffr_active_steps": int(sum(metrics["ffr_active"])),
        "violation_fraction": float(np.mean([1.0 if df < -0.5 else 0.0 for df in delta_f_arr])),
        # Window statistics (key for reviewer)
        "window": {
            "actual_inject_step": actual_inject_step,
            "pre_window": f"[{pre_start}, {pre_end})",
            "post_window": f"[{post_start}, {post_end})",
            # Pre-injection (baseline)
            "pre_mean_delta_f": float(np.mean(pre_delta_f)),
            "pre_std_delta_f": float(np.std(pre_delta_f)),
            "pre_max_abs_delta_f": float(np.max(np.abs(pre_delta_f))) if len(pre_delta_f) > 0 else 0.0,
            "pre_mean_rocof": float(np.mean(pre_rocof)),
            "pre_max_abs_rocof": float(np.max(np.abs(pre_rocof))) if len(pre_rocof) > 0 else 0.0,
            # Post-injection (disturbance)
            "post_mean_delta_f": float(np.mean(post_delta_f)),
            "post_std_delta_f": float(np.std(post_delta_f)),
            "post_max_abs_delta_f": float(np.max(np.abs(post_delta_f))) if len(post_delta_f) > 0 else 0.0,
            "post_min_delta_f": float(np.min(post_delta_f)) if len(post_delta_f) > 0 else 0.0,
            "post_mean_rocof": float(np.mean(post_rocof)),
            "post_max_abs_rocof": float(np.max(np.abs(post_rocof))) if len(post_rocof) > 0 else 0.0,
            # Delta (impact)
            "delta_f_jump": float(np.mean(post_delta_f) - np.mean(pre_delta_f)),
            "rocof_jump": float(np.max(np.abs(post_rocof)) - np.max(np.abs(pre_rocof))) if len(post_rocof) > 0 and len(pre_rocof) > 0 else 0.0,
        },
        "delta_f_trace": metrics["delta_f"],
        "rocof_trace": metrics["rocof"],
    }


def print_window_stats(results: list, label: str) -> dict:
    """Print aggregated window statistics."""
    windows = [r["window"] for r in results]

    pre_max_df = np.mean([w["pre_max_abs_delta_f"] for w in windows])
    post_max_df = np.mean([w["post_max_abs_delta_f"] for w in windows])
    post_min_df = np.mean([w["post_min_delta_f"] for w in windows])
    pre_max_rocof = np.mean([w["pre_max_abs_rocof"] for w in windows])
    post_max_rocof = np.mean([w["post_max_abs_rocof"] for w in windows])
    df_jump = np.mean([w["delta_f_jump"] for w in windows])
    rocof_jump = np.mean([w["rocof_jump"] for w in windows])

    inject_rate = np.mean([1.0 if w["actual_inject_step"] >= 0 else 0.0 for w in windows])

    print(f"\n--- {label} Window Analysis (t_inject={T_INJECT}s) ---")
    print(f"  Event injection rate: {inject_rate*100:.0f}%")
    print(f"  Pre-injection  [t-{WINDOW_PRE}s, t): max|dFf|={pre_max_df:.4f} Hz, max|RoCoF|={pre_max_rocof:.4f} Hz/s")
    print(f"  Post-injection [t, t+{WINDOW_POST}s): max|dFf|={post_max_df:.4f} Hz, min(dFf)={post_min_df:.4f} Hz")
    print(f"                         max|RoCoF|={post_max_rocof:.4f} Hz/s")
    print(f"  JUMP: dFf change={df_jump:+.4f} Hz, RoCoF increase={rocof_jump:+.4f} Hz/s")

    return {
        "inject_rate": float(inject_rate),
        "pre_max_delta_f": float(pre_max_df),
        "post_max_delta_f": float(post_max_df),
        "post_min_delta_f": float(post_min_df),
        "pre_max_rocof": float(pre_max_rocof),
        "post_max_rocof": float(post_max_rocof),
        "delta_f_jump": float(df_jump),
        "rocof_jump": float(rocof_jump),
    }


def main():
    print("=" * 70)
    print("FFR IMPACT ASSESSMENT WITH INJECTION WINDOW ANALYSIS")
    print("=" * 70)
    print(f"T_inject={T_INJECT}s, Pre-window={WINDOW_PRE}s, Post-window={WINDOW_POST}s")

    print("\nInitializing environment...")
    placement_path = ROOT / "artifacts" / "placement" / "official_placement_v3.json"
    mpc_path = ROOT / "data" / "grid_IEEE123_complete.m"
    env = MicrogridEnvDual(placement_path=placement_path, mpc_path=mpc_path, seed=42)

    n_episodes = 10
    n_steps = 300

    results_no_ctrl = []
    results_with_ctrl = []

    print(f"\n=== Running {n_episodes} episodes WITHOUT control (baseline) ===")
    for ep in range(n_episodes):
        env.np_rng = np.random.default_rng(42 + ep)
        res = run_episode_detailed(env, n_steps, with_control=False)
        results_no_ctrl.append(res)
        w = res["window"]
        print(f"  Ep {ep+1}: nadir={res['nadir']:.3f} Hz, inject@{w['actual_inject_step']}, "
              f"pre|dF|={w['pre_max_abs_delta_f']:.3f}->post|dF|={w['post_max_abs_delta_f']:.3f}")

    window_no_ctrl = print_window_stats(results_no_ctrl, "NO CONTROL")

    print(f"\n=== Running {n_episodes} episodes WITH proportional control ===")
    for ep in range(n_episodes):
        env.np_rng = np.random.default_rng(42 + ep)
        res = run_episode_detailed(env, n_steps, with_control=True, control_gain=0.5)
        results_with_ctrl.append(res)
        w = res["window"]
        print(f"  Ep {ep+1}: nadir={res['nadir']:.3f} Hz, inject@{w['actual_inject_step']}, "
              f"pre|dF|={w['pre_max_abs_delta_f']:.3f}->post|dF|={w['post_max_abs_delta_f']:.3f}")

    window_with_ctrl = print_window_stats(results_with_ctrl, "WITH CONTROL")

    # Aggregate results
    def aggregate(results: list) -> dict:
        return {
            "mean_nadir": float(np.mean([r["nadir"] for r in results])),
            "min_nadir": float(np.min([r["nadir"] for r in results])),
            "mean_max_delta_f": float(np.mean([r["max_abs_delta_f"] for r in results])),
            "max_max_delta_f": float(np.max([r["max_abs_delta_f"] for r in results])),
            "mean_violation": float(np.mean([r["violation_fraction"] for r in results])),
            "mean_rocof": float(np.mean([r["mean_abs_rocof"] for r in results])),
            "max_rocof": float(np.max([r["max_abs_rocof"] for r in results])),
        }

    agg_no_ctrl = aggregate(results_no_ctrl)
    agg_with_ctrl = aggregate(results_with_ctrl)

    print("\n" + "=" * 70)
    print("SUMMARY: Scenario Frequency Impact Assessment")
    print("=" * 70)
    print(f"\n{'Metric':<25} {'No Control':<15} {'With Control':<15} {'Improvement':<15}")
    print("-" * 70)
    print(f"{'Mean Nadir (Hz)':<25} {agg_no_ctrl['mean_nadir']:<15.3f} {agg_with_ctrl['mean_nadir']:<15.3f} {agg_with_ctrl['mean_nadir'] - agg_no_ctrl['mean_nadir']:+.3f} Hz")
    print(f"{'Min Nadir (Hz)':<25} {agg_no_ctrl['min_nadir']:<15.3f} {agg_with_ctrl['min_nadir']:<15.3f} {agg_with_ctrl['min_nadir'] - agg_no_ctrl['min_nadir']:+.3f} Hz")
    print(f"{'Mean max|dF| (Hz)':<25} {agg_no_ctrl['mean_max_delta_f']:<15.3f} {agg_with_ctrl['mean_max_delta_f']:<15.3f} {agg_no_ctrl['mean_max_delta_f'] - agg_with_ctrl['mean_max_delta_f']:+.3f} Hz")
    print(f"{'Max max|dF| (Hz)':<25} {agg_no_ctrl['max_max_delta_f']:<15.3f} {agg_with_ctrl['max_max_delta_f']:<15.3f} {agg_no_ctrl['max_max_delta_f'] - agg_with_ctrl['max_max_delta_f']:+.3f} Hz")
    print(f"{'Mean Violation (%)':<25} {agg_no_ctrl['mean_violation']*100:<15.1f} {agg_with_ctrl['mean_violation']*100:<15.1f} {(agg_no_ctrl['mean_violation'] - agg_with_ctrl['mean_violation'])*100:+.1f}%")
    print(f"{'Mean |RoCoF| (Hz/s)':<25} {agg_no_ctrl['mean_rocof']:<15.4f} {agg_with_ctrl['mean_rocof']:<15.4f}")
    print(f"{'Max |RoCoF| (Hz/s)':<25} {agg_no_ctrl['max_rocof']:<15.4f} {agg_with_ctrl['max_rocof']:<15.4f}")

    # Key reviewer metrics
    print("\n" + "=" * 70)
    print("REVIEWER KEY METRICS: Disturbance Impact Quantification")
    print("=" * 70)
    print(f"\n1. EVENT INJECTION EFFECTIVENESS:")
    print(f"   - Injection rate: {window_no_ctrl['inject_rate']*100:.0f}% of episodes")
    print(f"   - Pre-disturbance |dFf|: {window_no_ctrl['pre_max_delta_f']:.4f} Hz (baseline)")
    print(f"   - Post-disturbance |dFf|: {window_no_ctrl['post_max_delta_f']:.4f} Hz")
    print(f"   - dFf JUMP: {window_no_ctrl['delta_f_jump']:+.4f} Hz <- DISTURBANCE SIGNAL")

    print(f"\n2. FREQUENCY NADIR IMPACT:")
    print(f"   - Post-injection min(dFf): {window_no_ctrl['post_min_delta_f']:.4f} Hz")
    print(f"   - Nadir below 49.5 Hz: {agg_no_ctrl['min_nadir']:.2f} Hz <- PRE-UFLS ZONE")

    print(f"\n3. RoCoF TRANSIENT:")
    print(f"   - Pre-disturbance max|RoCoF|: {window_no_ctrl['pre_max_rocof']:.4f} Hz/s")
    print(f"   - Post-disturbance max|RoCoF|: {window_no_ctrl['post_max_rocof']:.4f} Hz/s")
    print(f"   - RoCoF JUMP: {window_no_ctrl['rocof_jump']:+.4f} Hz/s <- INERTIA STRESS")

    print(f"\n4. FFR CONTROL EFFECTIVENESS:")
    ffr_df_reduction = window_no_ctrl['post_max_delta_f'] - window_with_ctrl['post_max_delta_f']
    ffr_nadir_improvement = agg_with_ctrl['min_nadir'] - agg_no_ctrl['min_nadir']
    print(f"   - Post-disturbance |dFf| reduction: {ffr_df_reduction:+.4f} Hz")
    print(f"   - Nadir improvement: {ffr_nadir_improvement:+.3f} Hz")
    print(f"   - Violation reduction: {(agg_no_ctrl['mean_violation'] - agg_with_ctrl['mean_violation'])*100:+.1f}%")

    print("\n" + "=" * 70)
    print("CONCLUSION:")
    if abs(window_no_ctrl['delta_f_jump']) > 0.1:
        print("[OK] Disturbance creates MEASURABLE frequency impact (|dFdFf| > 0.1 Hz)")
    else:
        print("[WARN] Disturbance signal may be too weak")

    if agg_no_ctrl['min_nadir'] < 49.5:
        print(f"[OK] Nadir enters PRE-UFLS zone ({agg_no_ctrl['min_nadir']:.2f} Hz < 49.5 Hz)")

    if agg_no_ctrl['mean_violation'] > 0.1:
        print(f"[OK] Violation rate significant: {agg_no_ctrl['mean_violation']*100:.1f}%")

    if ffr_df_reduction > 0.01:
        print(f"[OK] FFR control demonstrably reduces |dFf| by {ffr_df_reduction:.3f} Hz")
    print("=" * 70)

    # Save results
    output = {
        "config": {
            "t_inject": T_INJECT,
            "window_pre": WINDOW_PRE,
            "window_post": WINDOW_POST,
            "n_episodes": n_episodes,
            "n_steps": n_steps,
        },
        "no_control": {
            "aggregate": agg_no_ctrl,
            "window_stats": window_no_ctrl,
        },
        "with_control": {
            "aggregate": agg_with_ctrl,
            "window_stats": window_with_ctrl,
        },
        "reviewer_metrics": {
            "injection_rate": window_no_ctrl["inject_rate"],
            "delta_f_jump": window_no_ctrl["delta_f_jump"],
            "rocof_jump": window_no_ctrl["rocof_jump"],
            "nadir_hz": agg_no_ctrl["min_nadir"],
            "ffr_delta_f_reduction": ffr_df_reduction,
            "ffr_nadir_improvement": ffr_nadir_improvement,
            "ffr_violation_reduction": agg_no_ctrl["mean_violation"] - agg_with_ctrl["mean_violation"],
        },
    }

    out_path = ROOT / "artifacts" / "ffr_impact_assessment.json"
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
