"""Assemble an ANDES System for the islanded IEEE 123 feeder.

The network comes from the same `build_ieee123_net` used by T1/T2, so topology,
load and GFM buses are traceable to one source. The 115 kV substation bus and its
transformer are dropped: the microgrid is islanded, so that branch has no source
behind it and would only add an isolated island for the power flow to trip over.

Modelling choices a reviewer will ask about, all switchable in `CaseSpec`:

- Loads are **constant power** in the time domain. ANDES converts PQ to constant
  impedance by default (`pq2z=1`), which self-relieves under a voltage dip and
  quietly flatters every frequency result. Constant power is the conservative
  reading; `load_p2z` moves along the ZIP line if a reviewer wants it.
- The 16 aggregated DER already in the network build (2.88 MW of rooftop PV, one
  small wind and one storage unit) are entered as **negative constant-power
  loads**: grid-following, no frequency support, no ride-through logic. That is
  the pessimistic reading and keeps the frequency response attributable to the
  GFM fleet alone.
- Closed tie switches become `Jumper` (true zero impedance) rather than a small
  fictitious line, so no artificial impedance enters the electrical distances.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
import time

import andes

ROOT = Path(__file__).resolve().parents[2]

V_BASE_KV = 4.16
S_BASE_MVA = 1.0
Z_BASE_OHM = V_BASE_KV**2 / S_BASE_MVA
F_NOM_HZ = 60.0

GFM_KEYS = ("G1", "G2", "G3", "G4", "G5", "G6")


@dataclass(frozen=True)
class DieselSpec:
    """A synchronous diesel genset: GENROU + a governor + a simple AVR.

    ANDES ships no GGOV1/DEGOV1 -- plan Part 0-D2b governs the *EMT* model, not
    this one. TGOV1 is the phasor-domain stand-in; `governor="GAST"` selects a
    model whose load-limiter path is closer to a real prime mover. The mismatch
    against D2b is deliberate and has to be closed at T7/T9 by an EMT
    cross-check, not by assuming the two families agree.
    """

    bus_name: str = "76"
    s_mva: float = 1.0
    p_mw: float = 0.0
    h_sec: float = 1.0          # 0.5-2 s is typical for a diesel genset + alternator
    d_pu: float = 0.0
    droop_r: float = 0.05
    governor: str = "TGOV1"     # or "GAST"
    vset_pu: float = 1.0


@dataclass(frozen=True)
class Disturbance:
    """One event. `kind` picks which of the fields below are read."""

    kind: str                    # "gen_loss" | "load_step" | "gen_trip" | "der_trip"
    t_event: float = 1.0
    step_mw: float = 0.0         # load_step: MW added (positive = load increase)
    step_mvar: float = 0.0
    step_bus_name: str = "76"
    trip_target: str = ""        # der_trip: the `name` of the sgen row to drop


@dataclass(frozen=True)
class CaseSpec:
    placement: Path = ROOT / "artifacts" / "placement" / "official_placement_v4_rescaled.json"
    gfm_keys: tuple[str, ...] = GFM_KEYS
    topology: str = "G0"          # label carried into `config_id`
    # The open branches that *make* that topology, as tagged ids: "s<i>" opens
    # `net.switch[i]`, "l<i>" takes `net.line[i]` out of service. The set is
    # absolute, not a delta: every other switch is closed and every other line in
    # service, so a tie that is normally open closes when it is left out. `None`
    # means "leave the feeder exactly as built", which is G0 and is what every
    # run before T22 used. G0 written out is ("s5","s6","s7","s8","s9").
    # `_isolated_buses` is already recomputed per topology, so a bus that a
    # closing tie brings back reappears on its own.
    open_elements: tuple[str, ...] | None = None

    # GFM converter model. The coupling to the feeder is split in two on purpose:
    # `x_f_pu` is the LCL filter *inside* the current loop (REGF1.xf), `x_tr_pu` is
    # an explicit step-up branch between the converter terminal and the feeder bus.
    # The split matters. With every GFM sitting directly on the feeder, six voltage
    # setpoints face a network whose inter-GFM reactance is ~0.02-0.04 pu on a 1 MVA
    # base: measured dQ/dV is ~3.7e4 pu/pu, so a 1e-3 pu setpoint difference swings a
    # unit's reactive output by 37 pu of its own rating. That is not a solver
    # artefact -- it is the same stiff-network reactive coupling that sank the 100%
    # GFM case in the literature -- but it makes any setpoint-based reactive dispatch
    # meaningless. Behind an explicit branch the sensitivity falls to ~1/x_tr.
    # `x_f_pu` is REGFM_A1's coupling reactance X_L (PNNL-35110 Table 1: example
    # 0.15, normal range 0.05-0.25). Moved off the floor of that range onto the
    # specification's own example; it stays inside the [0.05, 0.20] band over
    # which the inner-loop retune below was verified to hold Re(lambda) <= 0.
    # This is a conformance change and nothing more. It does *not* touch the
    # reactive split: at 0.05 and at 0.15 the initial Q is identical to four
    # decimals ([0.253, 0.494, 0.389, 0.822, 0.464, 0.663] pu of own rating),
    # because the converter terminals are PV/Slack buses pinned at v = 1.0 and
    # the reactive sharing is set by the network they face -- `x_tr_pu` and the
    # feeder -- not by a reactance behind the terminal.
    x_f_pu: float = 0.15         # LCL filter, on the unit's own base
    x_tr_pu: float = 0.06        # step-up transformer, on the unit's own base
    r_f_pu: float = 0.0
    droop_r: float = 0.05        # P-f droop, same R as the analytical layer
    q_droop: float = 0.045       # REGF1 default Q-V droop
    # Reactive ceiling, on the unit's own base. REGFM_A1 gives Qmax/Qmin =
    # +/-0.44 as its example (normal range +/-0.44 to +/-1). At the previous
    # +/-1.0 the Q limiter (REGF1's KPqlim/KIqlim, which ship enabled) never
    # engaged, so nothing pushed back on exactly the circulating reactive power
    # the Q-V droop block exists to prevent -- one unit carried 0.82 pu of its
    # own rating at t=0 while another carried 0.25.
    # Read the number against the feeder before trusting it: the fleet's
    # steady-state reactive demand at nominal dispatch is 1.921 MVAr on 4.362 MVA
    # of converter, i.e. 0.4403 pu -- the REGFM_A1 example ceiling almost exactly.
    # Enforcing 0.44 therefore equalises the split perfectly (spread 0.569 ->
    # 0.001 pu) by putting every unit *on* its ceiling with no reactive margin
    # left, and the Slack GFM finishes 0.0011 pu above it. 0.60 is the smallest
    # value inside the specification's range that still forces the redistribution
    # (G4: 0.82 -> 0.60) while leaving margin; see the sweep in the T21 artifacts.
    # 0.60 is what ships. The ceiling does not move the security boundary --
    # dP_max is 1.18506 MW at both 0.44 and 0.60, because that boundary is set by
    # frequency -- so this is chosen on operating grounds, not on the boundary:
    # a fleet with zero reactive reserve cannot reproduce the failure mode that
    # sinks 100%-GFM cases in the literature (unexpected reactive flows during
    # synchronisation), which is exactly what C2 has to look at.
    q_max_pu: float = 0.60
    # Declaring `q_max_pu` is not the same as enforcing it, and by default ANDES
    # enforces neither. Two behaviours have to be turned off, or the ceiling is
    # decorative:
    #   1. `PV.pv2pq` ships off, so the power flow ignores `qmax` entirely and
    #      pins all six converter terminals at v = 1.0 whatever Q that costs.
    #   2. REGF1's `adjust_upper` ships on, so at TDS initialisation ANDES
    #      *widens* Qmax to whatever Q the power flow handed it -- silently, and
    #      per unit. Measured here: a declared 0.44 became [0.44, 0.49, 0.44,
    #      0.82, 0.46, 0.66], i.e. exactly the initial Q wherever it bound.
    # With both off the declared ceiling holds and an initialisation that cannot
    # respect it fails loudly instead of being erased. The Slack GFM is left
    # unlimited on purpose: it is the reactive balance node of an islanded feeder
    # and clamping it removes the only degree of freedom that closes Q.
    enforce_q_limits: bool = True

    # Inner-loop gains. REGF1's defaults (KPi=0.5, KIi=20) make this fleet
    # *linearly unstable*: six units on a 4.16 kV feeder give ten eigenvalues with
    # Re > 0 at 92-260 Hz and zeta about -0.03 to -0.05, and every time-domain run
    # then collapses ~0.25 s after any event. The instability is in the current
    # loop, not the voltage loop -- holding KPv/KIv at their defaults and slowing
    # only KPi/KIi restores Re(lambda) <= 0 for every deployment (2/5/6 GFM), every
    # headroom, xf in [0.05, 0.20], R in [0.02, 0.05] and both load models.
    # Modes above ~100 Hz are outside the validity of a positive-sequence model
    # anyway, so the retune moves the converter dynamics into the band this
    # platform can actually represent. It must be re-derived, not reused, when the
    # EMT model of T3 sets the real gains.
    kp_i: float = 0.20
    ki_i: float = 5.0
    kp_v: float = 3.0
    ki_v: float = 10.0

    # P limiter loop. REGF1 ships KIplim = 30, which is *not* used here: see
    # `_wdrp` for why it silently stiffens the droop by 75%, and for the measurement
    # showing the ceiling still binds without it.
    kp_plim: float = 5.0
    ki_plim: float = 0.0
    t_pm: float = 0.025
    p_head_mw: float | None = None   # total upward headroom; None = full BESS rating
    # Two current ceilings, both on the unit's own base, because they answer
    # different questions and only one of them is a security criterion.
    # `i_max_f_pu` is REGFM_A1's ImaxF, the *transient* ceiling a fault-current
    # limiter enforces (PNNL-35110: example 2.0, normal range 1.5-3.0). The
    # current peaks in this study last a few hundred ms, which is the regime that
    # parameter describes, so this is the ceiling the security band reads against.
    # `i_cont_pu` is the *continuous* thermal rating. Crossing it for a fraction
    # of a second is a duty-cycle question for the converter's thermal design,
    # reported alongside the verdict but not part of it. Quoting 1.20 pu as "the
    # converter current limit" while citing REGFM_A1 conflates the two.
    i_max_f_pu: float = 2.00
    i_cont_pu: float = 1.20

    # dispatch
    load_scale: float = 1.0
    diesel: DieselSpec | None = None

    # load model
    load_p2z: float = 0.0        # 0 = constant power, 1 = constant impedance
    der_as_negative_load: bool = True

    # solver
    t_end: float = 20.0
    t_step: float = 0.005

    disturbance: Disturbance = field(default_factory=lambda: Disturbance(kind="load_step"))


@lru_cache(maxsize=1)
def _base_net():
    """The pandapower feeder. Cached: a bisection rebuilds the case ~20 times."""
    import sys

    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from src.env.IEEE123bus import build_ieee123_net

    return build_ieee123_net(
        mode="feeder123",
        balanced=True,
        convert_switches=True,
        source_mode="publish",
        islanded_override_slack_to_g1=True,
    )


@lru_cache(maxsize=32)
def _net_for(open_elements: tuple[str, ...] | None):
    """The feeder with `open_elements` open and everything else closed.

    Cached per topology because a bisection rebuilds the case ~20 times and the
    deep copy is the expensive part. The base net is never mutated: callers that
    reconfigure always get their own copy, so `_base_net`'s cache stays clean.
    """
    base = _base_net()
    if open_elements is None:
        return base

    open_set = set(open_elements)
    unknown = {t for t in open_set if not (t[:1] in "sl" and t[1:].isdigit())}
    if unknown:
        raise ValueError(f"open_elements must be 's<i>'/'l<i>', got {sorted(unknown)}")

    net = copy.deepcopy(base)
    for i in net.switch.index:
        net.switch.at[i, "closed"] = f"s{int(i)}" not in open_set
    for i in net.line.index:
        net.line.at[i, "in_service"] = f"l{int(i)}" not in open_set
    return net


def base_open_elements() -> tuple[str, ...]:
    """G0 written in the `open_elements` vocabulary, for the sweep to start from."""
    net = _base_net()
    return tuple(
        [f"s{int(i)}" for i in net.switch.index if not bool(net.switch.at[i, "closed"])]
        + [f"l{int(i)}" for i in net.line.index if not bool(net.line.at[i, "in_service"])]
    )


@lru_cache(maxsize=4)
def _placement(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def gfm_table(spec: CaseSpec) -> list[dict]:
    """One row per in-service GFM, with the headroom split resolved."""
    gfm = _placement(str(spec.placement))["gfm"]
    rows = []
    for key in spec.gfm_keys:
        e = gfm[key]
        rows.append(
            {
                "key": key,
                "bus_name": str(e["bus"]),
                "s_mva": float(e["inverter_mva"]),
                "p_rated_mw": float(e["bess_mw"]),
                "e_rated_mwh": float(e["bess_mwh"]),
            }
        )
    s_tot = sum(r["s_mva"] for r in rows)
    p_tot = sum(r["p_rated_mw"] for r in rows)
    head_tot = p_tot if spec.p_head_mw is None else spec.p_head_mw
    for r in rows:
        # Headroom is split on inverter rating, the same key the droop shares on,
        # so every unit reaches its ceiling at the same instant and P_head^min is
        # a property of the fleet rather than of one unlucky unit.
        r["share"] = r["s_mva"] / s_tot
        r["p_head_mw"] = head_tot * r["share"]
        r["p0_mw"] = gfm_dispatch_mw(spec) * r["share"]
    return rows


def gfm_dispatch_mw(spec: CaseSpec) -> float:
    """Total MW the GFM fleet carries before the event.

    Split out of `build_system` so the analytical screening layer sees the same
    pre-event operating point the time-domain run does, rather than assuming zero.
    Network losses are not included -- they land on the slack GFM at solve time.
    """
    net = _base_net()
    der = float(net.sgen.p_mw.sum()) if spec.der_as_negative_load else 0.0
    p_diesel = spec.diesel.p_mw if spec.diesel else 0.0
    return spec.load_scale * float(net.load.p_mw.sum()) - der - p_diesel


def _wdrp(spec: CaseSpec, s_mva: float) -> float:
    """REGF1's `wdrp` such that the fleet realises the intended droop R.

    Two corrections, both measured rather than assumed.

    *Base.* REGF1's law is `dw = w0 * wdrp * (PIplim_y - Psen_y)` with the power
    signals on the **system** base, and `wdrp` is not per-unit converted. Passing R
    straight through gives every unit the same MW/Hz gain whatever its rating -- a
    0.38 MVA unit then picks up as many MW as a 1.33 MVA one. Referring R to the
    unit's own rating restores proportional sharing.

    *Limiter bias.* The droop input is not `Pref - Pe` but `PIplim_y - Psen_y`, and
    `PIplim` integrates `Psig - Psen`, where `Psig` is `Psen` through a lag of
    `Tpm`. Because integral(Psig - Psen) dt = -Tpm * delta(Psen) exactly, that
    integrator lands `KIplim * Tpm * delta(Pe)` below where it started and never
    comes back: the realised droop is `R * (1 + KIplim * Tpm)` -- path-independent,
    and with REGF1's shipped KIplim = 30 a silent 75% stiffening. Measured across
    KIplim = 30 / 10 / 4 / 1 / 0 at dP = 0.5 MW, the realised R came out
    0.0875 / 0.0625 / 0.0550 / 0.0513 / 0.0500 against R = 0.05: the formula holds
    to four digits.

    Dividing the factor out of `wdrp` is the obvious fix and it is the wrong one --
    a smaller `wdrp` is a stiffer droop, which costs damping, and the fleet goes
    numerically unstable. The fix used instead is `ki_plim = 0`. The concern that
    the ceiling then stops binding does not survive measurement: with KIplim = 0 a
    saturated fleet still settles at max(Pe/Pmax) = 1.0001, because the clamp lives
    in the anti-windup lag on `Psig` and the proportional term (KPplim = 5) is
    enough to carry it into `dw`.
    """
    return spec.droop_r * S_BASE_MVA / s_mva


def _isolated_buses(net, already_dropped: set[int]) -> set[int]:
    """Buses with no in-service branch once `already_dropped` is gone.

    In G0 these are the open ends of the IEEE 123 tie switches (251, 350, 451):
    zero load, no line. ANDES leaves them in the Jacobian as singleton islands
    and the time domain then reports a bus voltage near zero, which pollutes the
    V_min security test. Recomputed per topology, so a tie that closes in some
    G_k brings its bus back automatically.
    """
    live = set()
    for _, ln in net.line.iterrows():
        f, t = int(ln.from_bus), int(ln.to_bus)
        if bool(ln.in_service) and f not in already_dropped and t not in already_dropped:
            live.update((f, t))
    for _, sw in net.switch.iterrows():
        if sw.et == "b" and bool(sw.closed):
            live.update((int(sw.bus), int(sw.element)))
    for _, tr in net.trafo.iterrows():
        hv, lv = int(tr.hv_bus), int(tr.lv_bus)
        if bool(tr.in_service) and hv not in already_dropped and lv not in already_dropped:
            live.update((hv, lv))
    return {int(b) for b in net.bus.index if int(b) not in already_dropped and int(b) not in live}


def build_system(spec: CaseSpec) -> tuple[andes.System, dict]:
    """Return an ANDES System (setup done, power flow not yet run) and its index maps."""
    net = _net_for(spec.open_elements)
    name_to_pp = {str(n).strip(): int(i) for i, n in zip(net.bus.index, net.bus.name)}

    drop = set(int(b) for b in net.bus.index[net.bus.vn_kv > 50.0])  # islanded: drop 115 kV
    drop |= _isolated_buses(net, drop)
    keep = [int(b) for b in net.bus.index if int(b) not in drop]
    pp_to_andes = {b: i + 1 for i, b in enumerate(keep)}
    hv = drop  # every dropped bus is skipped by the same guards below

    ss = andes.System(setup=False, no_output=True, default_config=True)
    ss.config.mva = S_BASE_MVA
    ss.config.freq = F_NOM_HZ

    for b in keep:
        ss.add("Bus", dict(idx=pp_to_andes[b], name=str(net.bus.at[b, "name"]).strip(),
                           Vn=V_BASE_KV, v0=1.0, area=1, zone=1))

    for i, ln in net.line.iterrows():
        f, t = int(ln.from_bus), int(ln.to_bus)
        if f in hv or t in hv or not bool(ln.in_service):
            continue
        ss.add("Line", dict(
            idx=f"L_{i}", name=str(ln["name"]),
            bus1=pp_to_andes[f], bus2=pp_to_andes[t], Vn1=V_BASE_KV, Vn2=V_BASE_KV,
            r=float(ln.r_ohm_per_km * ln.length_km) / Z_BASE_OHM,
            x=float(ln.x_ohm_per_km * ln.length_km) / Z_BASE_OHM,
            b=0.0, trans=0, tap=1.0, phi=0.0,
        ))

    for i, sw in net.switch.iterrows():
        if sw.et != "b" or not bool(sw.closed):
            continue
        f, t = int(sw.bus), int(sw.element)
        if f in hv or t in hv:
            continue
        ss.add("Jumper", dict(idx=f"J_{i}", name=f"tie_{f}_{t}",
                              bus1=pp_to_andes[f], bus2=pp_to_andes[t]))

    # --- loads --------------------------------------------------------------
    load = net.load.groupby("bus")[["p_mw", "q_mvar"]].sum()
    for b, row in load.iterrows():
        b = int(b)
        if b in hv:
            continue
        ss.add("PQ", dict(idx=f"PQ_{b}", name=f"load_{net.bus.at[b, 'name']}",
                          bus=pp_to_andes[b], Vn=V_BASE_KV,
                          p0=spec.load_scale * float(row.p_mw) / S_BASE_MVA,
                          q0=spec.load_scale * float(row.q_mvar) / S_BASE_MVA))

    der_mw = 0.0
    if spec.der_as_negative_load and len(net.sgen):
        sg = net.sgen.groupby("bus")[["p_mw", "q_mvar"]].sum()
        for b, row in sg.iterrows():
            b = int(b)
            if b in hv:
                continue
            der_mw += float(row.p_mw)
            ss.add("PQ", dict(idx=f"DER_{b}", name=f"der_{net.bus.at[b, 'name']}",
                              bus=pp_to_andes[b], Vn=V_BASE_KV,
                              p0=-float(row.p_mw) / S_BASE_MVA,
                              q0=-float(row.q_mvar) / S_BASE_MVA))

    # --- dispatch -----------------------------------------------------------
    gfm = gfm_table(spec)
    p_net_mw = spec.load_scale * float(net.load.p_mw.sum()) - der_mw
    p_diesel_mw = spec.diesel.p_mw if spec.diesel else 0.0
    p_lost_mw = spec.disturbance.step_mw if spec.disturbance.kind == "gen_loss" else 0.0
    p_gfm_mw = p_net_mw - p_diesel_mw - p_lost_mw   # losses land on the slack GFM

    for r in gfm:
        r["p0_mw"] = p_gfm_mw * r["share"]

    # --- GFM: one Slack + the rest PV, each carrying a REGF1 ----------------
    next_bus = max(pp_to_andes.values()) + 1
    for n, r in enumerate(gfm):
        poi = pp_to_andes[name_to_pp[r["bus_name"]]]      # point of interconnection
        bus = next_bus
        next_bus += 1
        r["poi_bus_andes"] = poi
        r["conv_bus_andes"] = bus
        ss.add("Bus", dict(idx=bus, name=f"{r['key']}_conv", Vn=V_BASE_KV, v0=1.0,
                           area=1, zone=1))
        ss.add("Line", dict(
            idx=f"TR_{r['key']}", name=f"stepup_{r['key']}", bus1=bus, bus2=poi,
            Vn1=V_BASE_KV, Vn2=V_BASE_KV,
            r=0.0, x=spec.x_tr_pu * S_BASE_MVA / r["s_mva"], b=0.0,
            trans=1, tap=1.0, phi=0.0,
        ))

        gen_idx = f"SG_{r['key']}"
        common = dict(idx=gen_idx, name=r["key"], bus=bus, Sn=r["s_mva"], Vn=V_BASE_KV,
                      p0=r["p0_mw"] / S_BASE_MVA, q0=0.0, v0=1.0,
                      pmax=(r["p0_mw"] + r["p_head_mw"]) / S_BASE_MVA,
                      pmin=-r["s_mva"] / S_BASE_MVA,
                      qmax=spec.q_max_pu * r["s_mva"] / S_BASE_MVA,
                      qmin=-spec.q_max_pu * r["s_mva"] / S_BASE_MVA,
                      ra=0.0, xs=spec.x_f_pu)
        if n == 0:
            ss.add("Slack", dict(common, a0=0.0))
        else:
            ss.add("PV", common)

        ss.add("REGF1", dict(
            idx=f"REGF1_{r['key']}", name=r["key"], bus=bus, gen=gen_idx,
            Sn=r["s_mva"], fn=F_NOM_HZ,
            rf=spec.r_f_pu, xf=spec.x_f_pu,
            wdrp=_wdrp(spec, r["s_mva"]),
            Qdrp=spec.q_droop * S_BASE_MVA / r["s_mva"],
            KPi=spec.kp_i, KIi=spec.ki_i, KPv=spec.kp_v, KIv=spec.ki_v,
            KPplim=spec.kp_plim, KIplim=spec.ki_plim, Tpm=spec.t_pm,
            # Pmax/Pmin/Qmax/Qmin are on the unit's own MVA base. Pmin < 0 is
            # what makes this a storage unit in REGFM_A1's terms.
            Pmax=(r["p0_mw"] + r["p_head_mw"]) / r["s_mva"],
            Pmin=-1.0, Qmax=spec.q_max_pu, Qmin=-spec.q_max_pu,
        ))

    # --- diesel -------------------------------------------------------------
    if spec.diesel is not None:
        d = spec.diesel
        bus = pp_to_andes[name_to_pp[d.bus_name]]
        ss.add("PV", dict(idx="SG_DIESEL", name="diesel", bus=bus, Sn=d.s_mva, Vn=V_BASE_KV,
                          p0=d.p_mw / S_BASE_MVA, q0=0.0, v0=d.vset_pu,
                          pmax=d.s_mva / S_BASE_MVA, pmin=0.0,
                          qmax=0.6 * d.s_mva / S_BASE_MVA, qmin=-0.6 * d.s_mva / S_BASE_MVA,
                          ra=0.0, xs=0.20))
        ss.add("GENROU", dict(
            idx="GENROU_DIESEL", name="diesel", bus=bus, gen="SG_DIESEL",
            Sn=d.s_mva, Vn=V_BASE_KV, fn=F_NOM_HZ,
            D=d.d_pu, M=2.0 * d.h_sec, ra=0.0, xl=0.10,
            xd=1.8, xq=1.75, xd1=0.30, xq1=0.55, xd2=0.22, xq2=0.22,
            Td10=5.0, Td20=0.05, Tq10=1.0, Tq20=0.10, S10=0.0, S12=0.0,
        ))
        if d.governor == "GAST":
            ss.add("GAST", dict(idx="GOV_DIESEL", name="gov", syn="GENROU_DIESEL",
                                R=d.droop_r, VMAX=1.0, VMIN=0.0,
                                T1=0.4, T2=0.1, T3=3.0, AT=1.0, KT=2.0, Dt=0.0))
        else:
            ss.add("TGOV1", dict(idx="GOV_DIESEL", name="gov", syn="GENROU_DIESEL",
                                 R=d.droop_r, VMAX=1.0, VMIN=0.0,
                                 T1=0.4, T2=1.0, T3=3.0, Dt=0.0))
        ss.add("SEXS", dict(idx="AVR_DIESEL", name="avr", syn="GENROU_DIESEL",
                            TATB=0.4, TB=5.0, K=50.0, TE=0.1, EMIN=-5.0, EMAX=5.0))

    # --- measurement --------------------------------------------------------
    watched = {r["bus_name"] for r in gfm}
    big = load.p_mw.sort_values(ascending=False).head(5).index
    watched |= {str(net.bus.at[int(b), "name"]).strip() for b in big if int(b) not in hv}
    for nm in sorted(watched):
        bus = pp_to_andes[name_to_pp[nm]]
        ss.add("BusROCOF", dict(idx=f"ROCOF_{nm}", name=f"rocof_{nm}", bus=bus,
                                Tf=0.02, Tw=0.1, Tr=0.1, fn=F_NOM_HZ))

    _add_disturbance(ss, spec, net, name_to_pp, pp_to_andes)

    ss.setup()

    # ANDES converts PQ to constant impedance in TDS by default; see module docstring.
    z = float(spec.load_p2z)
    ss.PQ.config.p2p, ss.PQ.config.p2i, ss.PQ.config.p2z = 1.0 - z, 0.0, z
    ss.PQ.config.q2q, ss.PQ.config.q2i, ss.PQ.config.q2z = 1.0 - z, 0.0, z
    ss.PQ.config.pq2z = 1 if z > 0 else 0

    if spec.enforce_q_limits:
        ss.PV.qlim.enable = True          # power flow respects qmax/qmin
        ss.REGF1.config.adjust_upper = 0  # ANDES must not widen the declared limits
        ss.REGF1.config.adjust_lower = 0

    ss.TDS.config.tf = spec.t_end
    ss.TDS.config.tstep = spec.t_step
    ss.TDS.config.no_tqdm = 1
    # Left on. A run that trips ANDES's own divergence criteria is insecure by
    # any reading, and the alternative is watching the step size crawl toward
    # zero -- one such run cost 798 s against a 15 s median. `tds_converged`
    # carries the outcome into the verdict either way.
    ss.TDS.config.criteria = 1

    index = {
        "pp_to_andes": pp_to_andes,
        "name_to_pp": name_to_pp,
        "gfm": gfm,
        "watched_buses": sorted(watched),
        "p_load_mw": spec.load_scale * float(net.load.p_mw.sum()),
        "p_der_mw": der_mw,
        "p_gfm_mw": p_gfm_mw,
        "p_diesel_mw": p_diesel_mw,
        "p_head_total_mw": sum(r["p_head_mw"] for r in gfm),
        "pmax_dev": [(r["p0_mw"] + r["p_head_mw"]) / r["s_mva"] for r in gfm],
        # REGF1.Pmax is declared `non_negative`, and ANDES replaces any value
        # *below* zero with the parameter default of 1.0 -- silently, per unit,
        # at TDS init. A charging fleet with little headroom has a genuinely
        # negative ceiling (measured: p0 = -0.49 MW total at dP = 1.1 MW
        # gen_loss, because the feeder's net load is only 0.61 MW), and that
        # ceiling is not representable in this model. Erasing it turns an
        # infeasible operating point into a comfortable one: the whole P ceiling
        # went from 0.15 pu to 1.0 pu per unit and the bisection reported
        # "secure at both ends". Detected here and settled analytically in
        # `solve`, because a ceiling below zero means the fleet cannot leave the
        # charging quadrant at all -- there is no post-event equilibrium to
        # simulate towards, and that is a result, not a platform failure.
        "ceiling_representable": all(
            (r["p0_mw"] + r["p_head_mw"]) / r["s_mva"] >= 0.0 for r in gfm),
        "s_gfm_total_mva": sum(r["s_mva"] for r in gfm),
    }
    return ss, index


def _add_disturbance(ss, spec: CaseSpec, net, name_to_pp, pp_to_andes) -> None:
    """Attach the event.

    `gen_loss` is the default and the one to use. The two obvious alternatives
    both have traps that cost a wrong answer rather than an error:

    - *A PQ offline at t=0, toggled on.* ANDES fixes a PQ's time-domain injection
      from the power-flow solution, so an offline device carries `Ppf = 0` into
      the time domain; switching it on perturbs the Jacobian without adding load.
      Measured: a step-to-zero divergence about 0.34 s after the event, at every
      step size and every magnitude down to 10 kW.
    - *`Alter` on `Ppf`,* which is what the ANDES docs prescribe for a load change.
      Correct, but only while the constant-power weight is nonzero. Under
      constant-impedance loads the time domain never reads `Ppf`, and a 3 MW
      "step" moved the frequency by 0.0000 Hz -- a silent no-op that the bisection
      reported as "secure at both ends". Guarded below.

    Toggling a device that was online for the power flow gates its whole
    contribution and so behaves identically under any ZIP mix. It is also the
    event a 100% IBR microgrid actually faces.
    """
    d = spec.disturbance
    if d.kind == "gen_loss":
        # A grid-following generator of `step_mw`, online for the power flow and
        # dropped at `t_event`. Preferred over `load_step` because `Toggle` gates a
        # device's whole contribution and so works under any ZIP mix, whereas an
        # `Alter` on `Ppf` is silently inert once the constant-power weight is zero
        # -- measured: under constant-impedance loads a 3 MW "step" moved the
        # frequency by 0.0000 Hz. It is also the event a 100% IBR microgrid actually
        # faces.
        bus = pp_to_andes[name_to_pp[d.step_bus_name]]
        ss.add("PQ", dict(idx="PQ_GENLOSS", name="gen_loss", bus=bus, Vn=V_BASE_KV,
                          p0=-d.step_mw / S_BASE_MVA, q0=-d.step_mvar / S_BASE_MVA))
        ss.add("Toggle", dict(idx="TG_GENLOSS", name="gen_loss", model="PQ",
                              dev="PQ_GENLOSS", t=d.t_event))
    elif d.kind == "load_step":
        if spec.load_p2z > 0.0:
            raise ValueError(
                "load_step alters PQ.Ppf, which the time domain ignores once "
                "load_p2z > 0; use kind='gen_loss' for a load-model-independent event")
        pp = name_to_pp[d.step_bus_name]
        if pp not in pp_to_andes:
            raise ValueError(f"step bus {d.step_bus_name!r} is not in the island")
        dev = f"PQ_{pp}"
        if dev not in set(ss.PQ.idx.v):
            raise ValueError(f"bus {d.step_bus_name!r} carries no load to step")
        ss.add("Alter", dict(idx="ALT_P", name="step_p", t=d.t_event, model="PQ",
                             dev=dev, src="Ppf", attr="v", method="+",
                             amount=d.step_mw / S_BASE_MVA))
        if d.step_mvar:
            ss.add("Alter", dict(idx="ALT_Q", name="step_q", t=d.t_event, model="PQ",
                                 dev=dev, src="Qpf", attr="v", method="+",
                                 amount=d.step_mvar / S_BASE_MVA))
    elif d.kind == "gen_trip":
        if spec.diesel is None:
            raise ValueError("gen_trip needs a diesel in the CaseSpec")
        ss.add("Toggle", dict(idx="TG_TRIP", name="diesel_trip", model="GENROU",
                              dev="GENROU_DIESEL", t=d.t_event))
    elif d.kind == "der_trip":
        hit = net.sgen[net.sgen.name == d.trip_target]
        if not len(hit):
            raise ValueError(f"no sgen named {d.trip_target!r}")
        b = int(hit.bus.iloc[0])
        ss.add("Toggle", dict(idx="TG_DER", name="der_trip", model="PQ",
                              dev=f"DER_{b}", t=d.t_event))
    else:
        raise ValueError(f"unknown disturbance kind: {d.kind!r}")


def solve(spec: CaseSpec, budget_s: float = 60.0, chunk_s: float = 0.1):
    """Build, run power flow, run TDS. Returns (system, index, status dict).

    The time domain is advanced in `chunk_s` pieces so a wall-clock budget can be
    enforced. The chunk has to be short -- the budget is only checked between
    chunks, and a collapsing case can spend minutes inside a single one. Without it a badly saturated case does not fail -- it crawls: the
    adaptive step shrinks toward zero while every tiny step still "converges", and
    ANDES's own divergence criteria never fire. One measured case at 0.25 MW of
    headroom took 798 s against a 15 s median for the same horizon. Inside a
    bisection that is not a slow run, it is a hang.

    A run that exhausts the budget is reported as not converged, which the
    security verdict already treats as insecure -- the correct reading, since the
    integrator only crawls when the trajectory is falling apart.
    """
    ss, index = build_system(spec)
    status = {"pflow_converged": False, "tds_converged": False, "t_reached": 0.0,
              "budget_exhausted": False, "error": "", "infeasible_ceiling": False}
    if not index["ceiling_representable"]:
        status["infeasible_ceiling"] = True
        return ss, index, status
    try:
        ss.PFlow.run()
        status["pflow_converged"] = bool(ss.PFlow.converged)
        if not status["pflow_converged"]:
            return ss, index, status
        ss.TDS.init()
        t0 = time.time()
        while ss.dae.t < spec.t_end - 1e-9:
            ss.TDS.config.tf = min(ss.dae.t + chunk_s, spec.t_end)
            ss.TDS.run()
            if not ss.TDS.converged:
                break
            if time.time() - t0 > budget_s:
                status["budget_exhausted"] = True
                break
        status["t_reached"] = float(ss.dae.t)
        status["tds_converged"] = (bool(ss.TDS.converged)
                                   and not status["budget_exhausted"]
                                   and status["t_reached"] >= spec.t_end - 1e-6)
    except Exception as exc:                      # a diverged TDS raises, not returns
        status["error"] = f"{type(exc).__name__}: {exc}"
    return ss, index, status
