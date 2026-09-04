"""Shared VSG frequency-simulation helper for the analysis/tuning scripts.

All four standalone scripts (plot_freq_comparison, test_freq_agc, tune_pi_agc,
tune_all_control) now drive the SINGLE VSG frequency model
(LTITopologyFreqDynamics) through the real environment, replacing the deleted
legacy synchronous-generator model. The env binds the VSG swing dynamics to the
true IEEE-123 operating point, so traces reflect the actual per-GFM inertia
(2H·dΔω/dt) and learned-droop damping rather than a stand-alone SG swing.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

DEFAULT_PLACEMENT = ROOT / "artifacts" / "placement" / "official_placement_v3.json"
DEFAULT_MPC = ROOT / "data" / "grid_IEEE123_complete.m"


def build_env(seed: int = 42):
    """Construct a MicrogridEnvDual bound to the VSG frequency model."""
    from src.env.microgrid_env_dual import MicrogridEnvDual

    return MicrogridEnvDual(
        placement_path=DEFAULT_PLACEMENT, mpc_path=DEFAULT_MPC, seed=seed
    )


def simulate_event(
    env,
    n_steps: int = 60,
    control_gain: float = 0.0,
    agc_ki: float | None = None,
    seed: int = 42,
    event_mw: float = -2.5,
    event_step: int = 5,
) -> dict:
    """Roll the env forward and record the VSG frequency trace.

    A fixed gen-trip event is forced at `event_step` so that runs with different
    control settings see the IDENTICAL disturbance (fair comparison). Without
    this, the env's stochastic event injector would fire different events on
    successive rolls of the same env.

    Args:
        control_gain: proportional FFR gain on COI Δf (0 = no FFR / backbone only).
        agc_ki: if given, override the VSG secondary-control integral gain.
        event_mw: forced generation-loss magnitude (negative = deficit).
        event_step: fast-step index at which the event injects.
    """
    if agc_ki is not None:
        env.freq_dyn_lti.agc_ki = float(agc_ki)
    env.np_rng = np.random.default_rng(seed)
    force_event = {
        "type": "gen_trip",
        "delta_P_mw": float(event_mw),
        "location": 97,
        "t_inject": float(event_step),
    }
    env.reset(options={"force_event": force_event})

    n_act = int(env.action_space_fast.shape[0])
    t, freq, delta_f, rocof, agc_int = [], [], [], [], []

    for _ in range(n_steps):
        st = env.freq_dyn_lti.get_state()
        if control_gain != 0.0:
            ctrl = -control_gain * float(np.clip(st.delta_f_hz / 0.5, -1.0, 1.0))
            action = np.full(n_act, ctrl, dtype=np.float32)
        else:
            action = np.zeros(n_act, dtype=np.float32)

        env.step_fast(action)
        s = env.freq_dyn_lti.get_state()
        t.append(s.t)
        freq.append(s.f_hz)
        delta_f.append(s.delta_f_hz)
        rocof.append(s.rocof_hz_s)
        agc_int.append(s.agc_integral_hz_s)

    return {
        "time": np.asarray(t),
        "freq": np.asarray(freq),
        "delta_f": np.asarray(delta_f),
        "rocof": np.asarray(rocof),
        "agc_integral": np.asarray(agc_int),
    }


def trace_metrics(trace: dict, settle_band_hz: float = 0.1) -> dict:
    """Standard scalar metrics from a frequency trace."""
    freq = trace["freq"]
    delta_f = trace["delta_f"]
    rocof = trace["rocof"]
    dt = float(trace["time"][1] - trace["time"][0]) if len(trace["time"]) > 1 else 1.0

    settled = np.where(np.abs(delta_f) < settle_band_hz)[0]
    settling_time = float(settled[0] * dt) if len(settled) else float("inf")

    return {
        "nadir": float(freq.min()),
        "zenith": float(freq.max()),
        "max_abs_rocof": float(np.abs(rocof).max()),
        "steady_state": float(freq[-max(1, len(freq) // 10):].mean()),
        "iae": float(np.sum(np.abs(delta_f)) * dt),  # ∫|Δf| dt
        "settling_time": settling_time,
    }
