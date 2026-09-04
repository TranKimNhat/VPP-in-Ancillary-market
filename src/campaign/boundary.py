"""Monotone boundary search over a scalar knob.

Both quantities the campaign needs are one-dimensional searches on the same
secure/insecure predicate, differing only in which way security runs:

    P_head^min    smallest headroom that is still secure   -> secure above
    P_DG,off^max  largest pre-trip diesel output still secure -> secure below

So one routine covers both, with a `direction` flag.

Two guards make the result honest rather than merely convergent:

1. **Bracket check.** Both endpoints are evaluated first. If they agree, no
   boundary exists inside the interval and the search says so instead of
   returning its midpoint. This is the live-or-die test: a bisection that
   converges on a bracket it never straddled is reporting an artefact.
2. **Monotonicity check.** After converging, a coarse uniform grid is evaluated
   across the original interval and the secure/insecure pattern is required to
   switch exactly once. Power-system security is not monotone in general; when it
   is not, the returned scalar is one boundary among several and the result says
   so rather than implying uniqueness.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable


@dataclass
class Probe:
    """One evaluation of the predicate."""

    x: float
    secure: bool
    payload: dict = field(default_factory=dict)
    wallclock_s: float = 0.0


@dataclass
class BoundaryResult:
    found: bool
    x_boundary: float
    x_secure: float
    x_insecure: float
    direction: str
    tol: float
    n_eval: int
    monotone: bool | None
    note: str
    probes: list[Probe] = field(default_factory=list)


def bisect(
    predicate: Callable[[float], tuple[bool, dict]],
    lo: float,
    hi: float,
    direction: str = "secure_above",
    tol: float = 1e-3,
    max_iter: int = 40,
    verify_points: int = 6,
    on_probe: Callable[[Probe], None] | None = None,
) -> BoundaryResult:
    """Find the switch point of `predicate` on [lo, hi].

    `predicate(x)` returns `(secure, payload)`; the payload is carried into the
    artifact so no run is wasted. `direction` is "secure_above" (the knob helps,
    e.g. headroom) or "secure_below" (the knob hurts, e.g. pre-trip output).
    """
    if direction not in ("secure_above", "secure_below"):
        raise ValueError(direction)
    if not hi > lo:
        raise ValueError(f"empty bracket [{lo}, {hi}]")

    probes: list[Probe] = []

    def ev(x: float) -> Probe:
        t0 = time.time()
        secure, payload = predicate(x)
        p = Probe(x=float(x), secure=bool(secure), payload=payload,
                  wallclock_s=time.time() - t0)
        probes.append(p)
        if on_probe:
            on_probe(p)
        return p

    p_lo, p_hi = ev(lo), ev(hi)
    want_lo = (direction == "secure_below")     # the end expected to be secure

    secure_end = p_lo if want_lo else p_hi
    insecure_end = p_hi if want_lo else p_lo
    if not secure_end.secure or insecure_end.secure:
        if p_lo.secure and p_hi.secure:
            note = "secure at both ends: no boundary inside the bracket"
        elif not p_lo.secure and not p_hi.secure:
            note = "insecure at both ends: no boundary inside the bracket"
        else:
            note = (f"predicate runs the other way: lo secure={p_lo.secure}, "
                    f"hi secure={p_hi.secure}, but direction={direction}")
        return BoundaryResult(False, float("nan"), float("nan"), float("nan"),
                              direction, tol, len(probes), None, note, probes)

    a, b = lo, hi                                # a is the secure_below end
    x_sec, x_ins = secure_end.x, insecure_end.x
    for _ in range(max_iter):
        if abs(b - a) <= tol:
            break
        mid = 0.5 * (a + b)
        p = ev(mid)
        if p.secure:
            x_sec = mid
            if direction == "secure_below":
                a = mid
            else:
                b = mid
        else:
            x_ins = mid
            if direction == "secure_below":
                b = mid
            else:
                a = mid

    monotone, mono_note = None, ""
    if verify_points >= 3:
        step = (hi - lo) / (verify_points - 1)
        pattern = []
        for k in range(verify_points):
            x = lo + k * step
            hit = next((q for q in probes if abs(q.x - x) < 1e-12), None)
            pattern.append((hit or ev(x)).secure)
        flips = sum(1 for i in range(1, len(pattern)) if pattern[i] != pattern[i - 1])
        monotone = flips == 1
        if not monotone:
            mono_note = (f"; security flips {flips} times on a {verify_points}-point "
                         "grid -- the returned value is one boundary, not the boundary")

    return BoundaryResult(
        found=True,
        x_boundary=0.5 * (x_sec + x_ins),
        x_secure=x_sec,
        x_insecure=x_ins,
        direction=direction,
        tol=tol,
        n_eval=len(probes),
        monotone=monotone,
        note=f"bracketed on [{lo:g}, {hi:g}]" + mono_note,
        probes=probes,
    )
