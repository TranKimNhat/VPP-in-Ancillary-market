"""The REGF1 droop path, in closed form, and its reduction to Ducoin's GFM swing.

Derived from `andes/models/renewable/regf1.py` (ANDES 2.0), blocks `Psen`,
`Psig`, `PIplim`, `dw`:

    Psen   = Lag(Pe, T=Tr)                                   # the measurement filter
    Psig   = LagAntiWindup(Psen + Paux, T=Tpm, Pmin, Pmax)   # inside the P limiter
    PIplim = PIController(Psig - Psen, kp=KPplim, ki=KIplim, x0=Psen)
    dw     = w0 * wdrp * (PIplim_y - Psen_y)

so the droop input is `PIplim_y - Psen_y`, *not* `Pref - Pe`: the P-limiter PI
sits in the droop's forward path whether or not anything is saturated. Taking
`x0` for what `block.PIController` makes it -- the integrator's initial value,
with no feedforward term in `e_str` -- the small-signal path is `tf()` below.

Two consequences the campaign depends on:

  * The DC gain is `w0*wdrp*(1 + KIplim*Tpm)`, so the realised droop is
    `R*(1 + KIplim*Tpm)`. This reproduces the five-point KIplim measurement in
    `build_case._wdrp` to four digits and is why `ki_plim = 0` ships.
  * With KIplim = 0 the numerator zero sits at `-1/(Tpm*(1 + KPplim))`. At the
    ANDES default KPplim = 5 that is -6.67 rad/s, *below* both poles (-40,
    -200): a lead-lag with ~4.2x mid-band gain, which overshoots. At KPplim = 0
    the zero lands on -1/Tpm and cancels that pole, collapsing the path to
    `-w0*wdrp/(1 + s*Tr)` -- exactly Ducoin's first-order GFM with
    `w_Pf = 1/Tr`. PNNL-35110 specifies `kppmax` = 0.01 (range 0.005-0.05),
    where the cancellation still holds to ~1%.

Reference for the reduced model this is compared against: E. A. S. Ducoin,
Y. Gu, B. Chaudhuri, T. C. Green, "Analytical Design of Contributions of
Grid-Forming and Grid-Following Inverters to Frequency Stability", IEEE Trans.
Power Systems 39(5), Sept 2024, eqs. (2)-(5), (21), (32), (33)-(34).
"""

from __future__ import annotations

import numpy as np


def tf(wdrp: float, Tr: float, Tpm: float, KPplim: float, KIplim: float,
       f0: float = 60.0) -> tuple[np.ndarray, np.ndarray]:
    """(num, den) of dOmega/dPe [rad/s per system p.u.], in descending powers of s.

        dw/dPe = -w0*wdrp * (1 + KIplim*Tpm + s*Tpm*(1 + KPplim))
                          / ((1 + s*Tr)(1 + s*Tpm))

    `wdrp` is REGF1's own parameter, i.e. already referred to the unit's rating
    by `build_case._wdrp`; the sign is negative because a power increase lowers
    frequency.
    """
    w0 = 2 * np.pi * f0
    num = -w0 * wdrp * np.array([Tpm * (1 + KPplim), 1 + KIplim * Tpm])
    den = np.polymul([Tr, 1.0], [Tpm, 1.0])
    return num, den


def droop_zero_rad_s(Tpm: float, KPplim: float, KIplim: float = 0.0) -> float:
    """Where the numerator zero sits. Only meaningful while KIplim = 0."""
    return -(1 + KIplim * Tpm) / (Tpm * (1 + KPplim))


def reduces_to_first_order(Tpm: float, KPplim: float, KIplim: float = 0.0,
                           tol: float = 0.02) -> bool:
    """True when the zero cancels the -1/Tpm pole to within `tol` (relative).

    In that regime REGF1's droop path *is* Ducoin (3)-(5) with `w_Pf = 1/Tr`,
    and `H_gfm` below is the inertia his (32)/(34) call for. Outside it, no
    `w_Pf` exists and neither equation has an argument to evaluate.
    """
    return abs(droop_zero_rad_s(Tpm, KPplim, KIplim) + 1 / Tpm) < tol / Tpm


def ducoin_H_D(m_p: float, w_Pf: float) -> tuple[float, float]:
    """Ducoin's GFM inertia and damping, `H = 1/(2 m_P w_Pf)` and `D = 1/m_P`.

    Valid only where `reduces_to_first_order` holds; see the module docstring.
    """
    return 1.0 / (2 * m_p * w_Pf), 1.0 / m_p


def tau_from_windowed_rocof(df_ss_hz: float, rocof_hz_s: float,
                            window_s: float = 0.5) -> float:
    """Identify `tau = 2H/D` of Ducoin (33) from a sliding-window RoCoF.

    (33) is `df(t) = df_ss (1 - exp(-t/tau))` -- first order, hence monotone --
    so `metrics.sliding_rocof`'s maximum sits at the smallest span it accepts,
    `window_s/2`. Inverting that endpoint difference gives `tau`.

    This is an *identification*, not a mapping: it is the dominant-pole time
    constant of whatever the system actually is. Where the measured overshoot
    exceeds 1 the system is at least second order and this first-order fit is
    misspecified, so report the result as a dominant time constant and not as
    an inertia.
    """
    dt = 0.5 * window_s
    g = rocof_hz_s * dt / df_ss_hz
    if not 0.0 < g < 1.0:
        return float("nan")
    return -dt / np.log(1.0 - g)
