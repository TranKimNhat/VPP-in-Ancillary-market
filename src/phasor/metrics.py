"""Turn one ANDES run into the fields plan Part 5 requires, plus a security verdict.

Two things are worth stating up front because they bound what this layer can claim.

`mu_I` is computed **post hoc**. REGF1 limits active and reactive *power* (`Pmax`,
`Qmax`); it has no current limiter and never switches into current-source mode.
So `mu_I` here answers "did the current magnitude exceed the converter's rating?"
and not "did the converter's limiter engage, and what did it do to the dynamics?".
The first question is the screening question; the second needs EMT (plan T6). A
run with `mu_I > 1` is therefore reported as insecure *and* as outside this
platform's validity -- see `SecurityVerdict.beyond_platform`.

There are two current ceilings and only one of them is that criterion. `mu_i` is
measured against `spec.i_max_f_pu`, REGFM_A1's transient limit `ImaxF`: that is
the ceiling a limiter would actually enforce over the few hundred ms these peaks
last, so it is the ceiling that bounds where a limiter-free model is still
faithful. `mu_i_cont` is measured against `spec.i_cont_pu`, the continuous
thermal rating; it is reported and never enters the verdict, because exceeding a
continuous rating transiently is a duty-cycle question for the converter's
thermal design, not a statement about the operating point. Reading a security
boundary off the continuous rating -- which is what earlier runs of this layer
did with a single 1.20 pu ceiling -- reports a thermal derating as a stability
limit.

`rocof` is measured two ways. ANDES's own `BusROCOF` is a washout-filtered
derivative with `Tr = 0.1 s`, which is a device model, not a grid-code
measurement. A sliding-window RoCoF over the frequency trace is added alongside
it because grid codes specify the window (concept section 13); the window length
is a parameter here, not a constant.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

F_NOM_HZ = 60.0


@dataclass(frozen=True)
class SecurityBand:
    """Omega_dyn. Voltage and reactive coordinates are in from the start, not bolted on."""

    # `v_min_pu` is the lower edge of the Continuous Operation region for a DER
    # under IEEE 1547-2018 Category III (0.88-1.00 pu, held for 5 s and 120 s),
    # the region IEEE 1547.1-2020 exists to test compliance against. That is the
    # question this band is asking -- the voltage a DER must ride through -- and
    # not the question ANSI C84.1 answers, which is the voltage delivered to a
    # load. The 0.90 that shipped here before was ANSI utilization Range A
    # (108 V on 120 V) and had no recorded provenance at all.
    #
    # The choice is not cosmetic: it selects which coordinate binds. At 0.90 the
    # voltage criterion crossed 0.56% before the nadir one, so the campaign read
    # as voltage-limited on a margin thinner than anything else it measured; at
    # 0.88 the nadir criterion binds and the voltage one sits 16% out. See
    # `reference/security_band_provenance.md` for the measurement and the
    # citations.
    #
    # `f_min_hz`, `f_max_hz` and `rocof_max_hz_s` still have no such provenance.
    # With v_min at 0.88 the nadir criterion is what sets dP_max, so `f_min_hz`
    # now needs the citation `v_min_pu` just got, and does not have it.
    f_min_hz: float = 59.0
    f_max_hz: float = 61.0
    rocof_max_hz_s: float = 2.0
    v_min_pu: float = 0.88
    v_max_pu: float = 1.10
    mu_i_max: float = 1.0        # relative to ImaxF, not to the continuous rating
    # A settled endpoint is part of security: a run that is still moving at t_end
    # has not shown that it recovers.
    settle_tol_hz: float = 0.01
    settle_window_s: float = 2.0


@dataclass
class SecurityVerdict:
    secure: bool
    beyond_platform: bool
    reasons: list[str] = field(default_factory=list)


@dataclass
class RunMetrics:
    ok: bool
    reason: str
    f_nadir_hz: float = float("nan")
    f_zenith_hz: float = float("nan")
    f_ss_hz: float = float("nan")
    df_ss_hz: float = float("nan")
    rocof_device_hz_s: float = float("nan")
    rocof_window_hz_s: float = float("nan")
    v_min_pu: float = float("nan")
    v_max_pu: float = float("nan")
    v_min_post_pu: float = float("nan")
    i_peak_pu: float = float("nan")
    mu_i: float = float("nan")           # vs ImaxF -- the security criterion
    mu_i_unit: str = ""
    mu_i_cont: float = float("nan")      # vs the continuous rating -- reported only
    mu_i_cont_unit: str = ""
    mu_p: float = float("nan")
    mu_p_unit: str = ""
    p_gfm_pre_mw: float = float("nan")
    p_gfm_post_mw: float = float("nan")
    dp_delivered_mw: float = float("nan")
    n_units_saturated: int = 0
    settled: bool = False
    no_equilibrium: bool = False
    t_reached_s: float = float("nan")


def sliding_rocof(t: np.ndarray, f_hz: np.ndarray, window_s: float) -> float:
    """Largest |df/dt| over a sliding window, grid-code style.

    `f_hz` is one frequency trace. Uses the endpoint difference over the window
    rather than a fit, which is what grid codes specify.
    """
    if len(t) < 3 or window_s <= 0:
        return float("nan")
    worst = 0.0
    j = 0
    for i in range(len(t)):
        while t[i] - t[j] > window_s:
            j += 1
        dt = t[i] - t[j]
        if dt >= 0.5 * window_s:
            worst = max(worst, abs(f_hz[i] - f_hz[j]) / dt)
    return float(worst)


def extract(ss, index: dict, status: dict, spec, band: SecurityBand,
            rocof_window_s: float = 0.5) -> tuple[RunMetrics, SecurityVerdict]:
    """Read one solved system. Never raises: a failed run comes back as ok=False."""
    if status.get("infeasible_ceiling"):
        # Settled before the solver: every unit's P ceiling sits below zero, so
        # the fleet cannot discharge at all and no post-event equilibrium exists.
        # Reported as the physical verdict it is, not as a platform limitation --
        # see `build_case.build_system` for why ANDES cannot represent it.
        return (RunMetrics(ok=False, no_equilibrium=True,
                           reason="P ceiling below zero: fleet confined to charging"),
                SecurityVerdict(False, False,
                                ["no post-event equilibrium: P ceiling below zero"]))
    if not status["pflow_converged"]:
        return (RunMetrics(ok=False, reason="power flow did not converge"),
                SecurityVerdict(False, False, ["pflow"]))

    t = np.asarray(ss.dae.ts.t, dtype=float)
    if len(t) < 2:
        return (RunMetrics(ok=False, reason=status["error"] or "no time-domain samples"),
                SecurityVerdict(False, False, ["tds"]))

    n = ss.dae.n                      # dae.ts.xy is [states | algebraics]
    f_pu = ss.dae.ts.xy[:, n + ss.BusROCOF.f.a]
    f_hz = F_NOM_HZ * f_pu
    rocof_dev = ss.dae.ts.xy[:, n + ss.BusROCOF.Wf_y.a]
    v = ss.dae.ts.xy[:, n + ss.Bus.v.a]
    pe = ss.dae.ts.xy[:, n + ss.REGF1.Pe.a]            # system p.u.
    idq = np.hypot(ss.dae.ts.xy[:, n + ss.REGF1.Id.a],
                   ss.dae.ts.xy[:, n + ss.REGF1.Iq.a])  # system p.u.

    keys = [r["key"] for r in index["gfm"]]
    sn = np.asarray([r["s_mva"] for r in index["gfm"]], dtype=float)
    pmax = np.asarray(ss.REGF1.Pmax.v, dtype=float)     # already system p.u.

    # Only feeder buses count for the voltage test; the six converter-terminal
    # buses sit behind the step-up branch and are not a point of supply.
    conv = {r["conv_bus_andes"] for r in index["gfm"]}
    feeder_cols = [i for i, b in enumerate(ss.Bus.idx.v) if b not in conv]
    v_feeder = v[:, feeder_cols]

    t_evt = spec.disturbance.t_event
    post = t >= t_evt
    tail = t >= max(t[-1] - band.settle_window_s, t_evt)

    m = RunMetrics(ok=True, reason="")
    m.t_reached_s = float(t[-1])
    m.f_nadir_hz = float(f_hz.min())
    m.f_zenith_hz = float(f_hz.max())
    m.f_ss_hz = float(f_hz[tail].mean())
    m.df_ss_hz = m.f_ss_hz - F_NOM_HZ
    m.rocof_device_hz_s = float(np.abs(rocof_dev[post]).max()) if post.any() else float("nan")
    m.rocof_window_hz_s = max(
        sliding_rocof(t, f_hz[:, k], rocof_window_s) for k in range(f_hz.shape[1]))
    m.v_min_pu = float(v_feeder.min())
    m.v_max_pu = float(v_feeder.max())
    m.v_min_post_pu = float(v_feeder[post].min()) if post.any() else float("nan")

    mu_i_t = idq / (spec.i_max_f_pu * sn)              # both sides in system p.u.
    k = int(np.unravel_index(np.argmax(mu_i_t), mu_i_t.shape)[1])
    m.i_peak_pu = float((idq / sn).max())               # device p.u., for reporting
    m.mu_i = float(mu_i_t.max())
    m.mu_i_unit = keys[k]

    # Continuous-rating utilisation: same trace, thermal ceiling. Reported so the
    # duty-cycle question stays visible, deliberately absent from `why` below.
    mu_ic_t = idq / (spec.i_cont_pu * sn)
    kc = int(np.unravel_index(np.argmax(mu_ic_t), mu_ic_t.shape)[1])
    m.mu_i_cont = float(mu_ic_t.max())
    m.mu_i_cont_unit = keys[kc]

    mu_p_t = pe / np.where(pmax > 0, pmax, np.nan)
    k = int(np.unravel_index(np.nanargmax(mu_p_t), mu_p_t.shape)[1])
    m.mu_p = float(np.nanmax(mu_p_t))
    m.mu_p_unit = keys[k]
    m.n_units_saturated = int((mu_p_t[tail].mean(axis=0) > 0.999).sum())

    m.p_gfm_pre_mw = float(pe[0].sum())                 # system base is 1 MVA
    m.p_gfm_post_mw = float(pe[tail].mean(axis=0).sum())
    m.dp_delivered_mw = m.p_gfm_post_mw - m.p_gfm_pre_mw
    m.settled = bool(np.ptp(f_hz[tail]) <= band.settle_tol_hz)

    # --- verdict ------------------------------------------------------------
    why: list[str] = []
    if not status["tds_converged"]:
        # Distinguish the two ways a run can stop early, because they mean opposite
        # things. A saturated fleet whose ceiling is below the post-event demand has
        # *no equilibrium to converge to*: frequency falls without bound and the
        # integrator's step collapses. That is a physical verdict, and it is the one
        # that sets P_head^min. A run that stalls with every coordinate inside the
        # band and nothing saturated is a platform failure, and must not be counted
        # as evidence of insecurity.
        m.no_equilibrium = bool(m.n_units_saturated == len(keys) and m.mu_p > 1.0)
        if m.no_equilibrium:
            why.append("no post-event equilibrium: whole fleet at its P ceiling")
        else:
            why.append(f"tds stopped at t={m.t_reached_s:.3f}s")
    if m.f_nadir_hz < band.f_min_hz:
        why.append(f"f_nadir {m.f_nadir_hz:.4f} < {band.f_min_hz}")
    if m.f_zenith_hz > band.f_max_hz:
        why.append(f"f_zenith {m.f_zenith_hz:.4f} > {band.f_max_hz}")
    if m.rocof_window_hz_s > band.rocof_max_hz_s:
        why.append(f"rocof {m.rocof_window_hz_s:.4f} > {band.rocof_max_hz_s}")
    if m.v_min_pu < band.v_min_pu:
        why.append(f"v_min {m.v_min_pu:.4f} < {band.v_min_pu}")
    if m.v_max_pu > band.v_max_pu:
        why.append(f"v_max {m.v_max_pu:.4f} > {band.v_max_pu}")
    if m.mu_i > band.mu_i_max:
        why.append(f"mu_I(ImaxF) {m.mu_i:.4f} > {band.mu_i_max} ({m.mu_i_unit})")
    if status["tds_converged"] and not m.settled:
        why.append("not settled at t_end")

    # mu_I > 1 means the run left the region where a limiter-free converter model
    # is a faithful description, whatever the other coordinates say. A stall with
    # no saturation is the other way out of validity; a no-equilibrium stop is not
    # -- that one is a result.
    beyond = bool(m.mu_i > band.mu_i_max) or (
        not status["tds_converged"] and not m.no_equilibrium)
    return m, SecurityVerdict(secure=not why, beyond_platform=beyond, reasons=why)
