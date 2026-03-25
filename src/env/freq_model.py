from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class FreqState:
    delta_f: float
    rocof: float
    delta_f_max_this_step: float
    freq_violated: bool
    rocof_violated: bool


class AnalyticalFrequencyModel:
    """
    Analytical closed-form frequency model (không dùng substep loop).
    O(1) computation per RL step.
    """

    def __init__(self) -> None:
        self.H_sys = 10.0
        self.S_base = 11.0
        self.D_sys = 2.0
        self.f0 = 50.0
        self.RoCoF_max = 1.5
        self.delta_f_max = 0.5

        self.event_prob = 0.83
        self.event_mean = 2.0
        self.event_sigma = 3.0
        self.event_max = 5.0

        self.delta_f = 0.0
        self.rng = np.random.default_rng()

    def step(self, delta_P_net: float, dT: float = 900) -> FreqState:
        tau = 2 * self.H_sys * self.S_base / (self.D_sys * self.f0)
        delta_f_ss = delta_P_net / self.D_sys
        decay = np.exp(-dT / tau)
        delta_f_end = delta_f_ss * (1 - decay) + self.delta_f * decay
        rocof = (
            self.f0 * (delta_P_net - self.D_sys * self.delta_f)
            / (2 * self.H_sys * self.S_base)
        )
        delta_f_max_this = max(abs(delta_f_end), abs(self.delta_f), abs(delta_f_ss))
        self.delta_f = float(delta_f_end)
        return FreqState(
            delta_f=float(delta_f_end),
            rocof=float(rocof),
            delta_f_max_this_step=float(delta_f_max_this),
            freq_violated=bool(delta_f_max_this > self.delta_f_max),
            rocof_violated=bool(abs(rocof) > self.RoCoF_max),
        )

    def sample_contingency(self) -> float:
        if self.rng.random() < self.event_prob:
            raw = self.rng.normal(self.event_mean, self.event_sigma)
            return float(np.clip(raw, 0.1, self.event_max))
        return 0.0

    def reset(self) -> None:
        self.delta_f = 0.0


def compute_nadir(
    delta_P_mw: float,
    H_sys: float = 2.5,
    D_sys: float = 1.5,
    f0: float = 50.0,
    S_base: float = 15.705,
) -> dict:
    """
    delta_P_mw: power imbalance (MW), negative = generation loss
    Returns: dict với rocof, f_nadir, f_ss, t_nadir
    """
    dP = delta_P_mw / S_base

    rocof = (dP * f0) / (2 * H_sys)

    dP_abs = abs(dP) + 1e-9
    exponent = -(D_sys**2) / (2 * H_sys * dP_abs)
    df_nadir = (dP * f0 / D_sys) * (1 - np.exp(exponent))
    f_nadir = f0 + df_nadir

    f_ss = f0 + (dP * f0) / D_sys

    t_nadir = (2 * H_sys / D_sys) * np.log(
        2 * H_sys * abs(rocof) / (D_sys * abs(f_ss - f0) + 1e-9) + 1
    )

    return {
        "rocof": float(rocof),
        "f_nadir": float(np.clip(f_nadir, 47.0, 51.0)),
        "f_ss": float(np.clip(f_ss, 47.0, 51.0)),
        "t_nadir": float(max(t_nadir, 0.1)),
        "delta_P_pu": float(dP),
    }


# Gate test:
# Gen loss 1.0 MW với H=2.5, D=1.5, S=15.705MW
# result = compute_nadir(-1.0)
# print(result)
# assert result["rocof"] < 0, "Gen loss → negative RoCoF"
# assert result["f_nadir"] < 50.0, "Gen loss → frequency drops"
# assert result["f_nadir"] > 47.0, "Not catastrophic"
# assert result["f_ss"] < 50.0, "Steady-state also below nominal"
#
# # Load surge 0.5 MW
# result2 = compute_nadir(+0.5)
# assert result2["rocof"] > 0, "Load surge → positive RoCoF (freq rises)"
# assert result2["f_nadir"] > 50.0, "Load surge → frequency rises"
# print("freq_model GATE: PASS")
