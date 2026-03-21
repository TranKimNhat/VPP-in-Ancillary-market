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
