# AGC Design Audit — 100% IBR Islanded Microgrid

**Status:** Active design rationale for `src/env/freq_dynamics.py` and
`src/env/microgrid_env_dual.py` after the audit on 2026-05-15.
**Decision:** Option (a) — AGC is *infrastructure* internal to the simulator;
RL controls only the FFR (Tier 2 droop) channel.

---

## 1. Scope

This document records:
1. The hierarchical frequency-control structure used in the simulator.
2. Which components are kept, which are disabled, and why.
3. Cross-references for every design choice against papers loaded in
   the user's NotebookLM (verified queries, May 2026).
4. Known limitations and what they imply for paper claims.

---

## 2. Final architecture (after the simplification)

| Tier | Function | Bandwidth | Implementation | Status |
|---|---|---|---|---|
| **1 — Primary** | Governor droop `1/R_SYS` + virtual inertia `H_SYS` | tens of Hz | `FrequencyDynamics.step()` swing equation | Kept |
| **2 — FFR** | Event-triggered fast active-power correction | seconds | `microgrid_env_dual.step_fast` droop block (learned by RL) | Kept (learned) |
| **3 — Secondary (AGC)** | Integral restoration of `Δf → 0` | fractions of Hz | Single PI loop **inside `FrequencyDynamics`** | Kept, simplified |
| **3'** — VPP-level distributed AGC | Per-area integral | — | `VPPSecondaryControl`, `BatterySecondaryControl` | **Disabled** by flag `USE_DISTRIBUTED_AGC=False` |

Only Tier 2 is learned by the AM policy. Tiers 1 and 3 are
deterministic and fixed across baselines and across training/eval
topologies.

---

## 3. Why this is correct for a 100% IBR islanded microgrid

### 3.1 Hierarchical timescale separation

> *Anttila et al. (Energies 2022, GFM review)* establishes that in
> 100% IBR microgrids the control hierarchy must respect strict
> bandwidth separation: inner V/I loops < 5 kHz, primary droop /
> virtual inertia at tens of Hz, and **secondary AGC at fractions of
> Hz** so that integral action does not interfere with primary
> response.

In the simplified design only one integrator (inside
`FrequencyDynamics`) acts on `Δf`. The previous design had three
integrators acting on the same `Δf` at the same fast timescale
`dt_fast_s` (`FrequencyDynamics._agc_integral` +
`VPPSecondaryControl.integral` + `BatterySecondaryControl.integral`),
which violates the bandwidth-separation principle and is the root
cause of the cascading-integrator instability identified in the audit.

### 3.2 ACE form — bias × frequency deviation

> *Saxena et al. (J. Energy Storage 2020)* and *Kerdphol et al. (IEEE
> Access 2017)*, both confirmed via NotebookLM, formulate the
> secondary controller as
> $\Delta P_{\text{ACE}} = -\frac{K_i}{s}\,[\beta\,\Delta f]$
> with bias factor $\beta = 1\,\mathrm{pu\,MW/Hz}$ and integral
> gain $K_i\in\{0.09, 22.114\}$ depending on tuning method.

Our `FrequencyDynamics` integrator uses raw `Δf` together with
`K_p`, `K_i` and a deadband, which is functionally equivalent to a
PI-on-ACE with $\beta = 1$. The form is acceptable for a single-area
microgrid; bias other than $\beta = 1$ is meaningful only when there
are interconnections to neighbouring areas, which the islanded
microgrid does not have.

### 3.3 Deadband

> *Anttila et al. (Energies 2022)* and *Adaptive Synthetic Inertia
> paper* (verified Notebook 4) recommend a small deadband on the
> secondary loop to prevent unnecessary integration of measurement
> noise.

The current `FrequencyDynamics.deadband = 0.1` Hz is wider than the
typical islanded value (50 mHz). Recommended tightening to
`0.05` Hz aligns with the Saxena 2020 and Kerdphol 2017 settings.

### 3.4 Anti-windup

> *Salem et al. (IEEE OJ-IES 2025)* on grid-forming converters notes
> that under current limiting, secondary integral action requires
> anti-windup (clamping or back-calculation) to recover after a
> disturbance. Pure clipping of the output without modifying the
> integral state leads to slow recovery (residual integral state).

Currently `FrequencyDynamics` clips `p_ref` to `±pref_max_pu` but
does **not** clip the underlying `_agc_integral`. Recommended fix:
back-calculation, i.e., when `p_ref` is clipped, subtract the
overflow divided by `K_i` from `_agc_integral`. The disabled
`VPPSecondaryControl` already does clipping-style anti-windup on
`integral`; if it is ever re-enabled, it should be upgraded to
back-calculation.

### 3.5 Single-area assumption

> *Ojo et al. (IEEE TCST 2023)* design distributed secondary control
> for IBR microgrids using a Laplacian consensus
> $\dot\chi = -\alpha L \chi + \alpha L k_I \delta$
> across grid-forming inverters. Distributed control is meaningful
> when multiple zones or multiple grid-forming references must agree
> on a common frequency setpoint.

Because the simulator uses a *single aggregated frequency state*
(`FrequencyDynamics` represents the islanded grid as one swing
equation with aggregate $H_{\text{eq}}$ and $D_{\text{eq}}$), there
is no need for a Laplacian-consensus distributed AGC at this level
of fidelity. The previous `VPPSecondaryControl` did *not* implement
consensus — it just kept three independent scalar integrators — so
its removal does not lose any consensus property.

### 3.6 SoC-weighted inertia and topology-aware H_SYS

> *Farmer et al. (cited inline in `freq_dynamics.py:128`) and
> Kerdphol et al. (IEEE Access 2017)* support computing system
> inertia as $H_{\text{SYS}} = \sum_i H_i S_i w_i / S_{\text{base}}$
> with weights $w_i$ reflecting GFM availability.

This is implemented correctly in `_update_system_params` and
`update_topology` and is consistent with the topology-aware narrative
of the paper. Kept as-is.

---

## 4. Parameter table

| Parameter | File | Value | Reference |
|---|---|---|---|
| $K_p$ (primary droop control gain) | `freq_dynamics.py:26` | 0.2 | Kerdphol 2017 (PI tuning) |
| $K_i$ (secondary integral) | `freq_dynamics.py:27` | 0.1 | Saxena 2020 (K_I = 0.09) |
| Deadband | `freq_dynamics.py:28` | 0.1 Hz | Recommend tightening to 0.05 Hz |
| `pref_min/max_pu` | `freq_dynamics.py:29-30` | ±0.05 pu | Saxena-class tuning |
| `R` (droop, derived) | `freq_dynamics.py:97` | from $\sum S_i/R_i$ | Standard inverse droop aggregation |
| `H_SYS` | `freq_dynamics.py:95` | ≈1.2 s | Kerdphol 2017 (H<3 s for high-RES) |
| `D_SYS` | `freq_dynamics.py:96` | from $\sum D_i S_i$ | Aggregate damping |
| Per-GFM `H_i, D_i, R_i` | `freq_dynamics.py:60-66` | 1.0–1.5 s, 0.5–1.0, 0.04–0.05 | Bevrani 2016, Rafiee 2021 |
| **Tier 3' flag `USE_DISTRIBUTED_AGC`** | `microgrid_env_dual.py` (new) | `False` | This document |
| `VPPSecondaryControl.K_i` (when re-enabled) | `microgrid_env_dual.py:27` | 0.1 default | Future work |
| `BatterySecondaryControl.K_i` (when re-enabled) | `microgrid_env_dual.py:53` | 0.5 default — too high | Reduce to 0.1 if re-enabled |

---

## 5. What changed (recommended diff)

### 5.1 `src/env/microgrid_env_dual.py`

Around line 899–910, replace the live AGC sum with a feature-flagged
no-op so the distributed-AGC code path remains reachable for future
ablation but does not run by default:

```python
# Tier 3 (secondary AGC) is owned by FrequencyDynamics. The
# distributed VPP/BESS AGC is preserved as code but disabled by
# default to enforce single-integrator semantics.
USE_DISTRIBUTED_AGC = False

if USE_DISTRIBUTED_AGC:
    agc_term = 0.0
    for vpp_idx in range(self._n_vpps):
        agc_term += self.vpp_agc[vpp_idx].step(f_current, self.dt_fast_s)
    agc_term += self.bess_agc.step(f_current, self.dt_fast_s)
    agc_term = float(np.clip(agc_term, -0.05, 0.05))   # match pref_max_pu
else:
    agc_term = 0.0

support_term = bess_term + agc_term
```

### 5.2 `src/env/freq_dynamics.py` (optional)

If 50 mHz deadband and back-calculation anti-windup are desired:

```python
def __init__(self, ..., deadband: float = 0.05, ...):
    ...

def step(self, dt, delta_P_pu, P_bess_pu=0.0):
    ...
    self._agc_integral += delta_f_for_agc * dt_val
    p_ref_raw = self._p_ref + (self.Kp * delta_f_for_agc
                               + self.Ki * self._agc_integral) * dt_val
    p_ref_new = min(self.pref_max_pu, max(self.pref_min_pu, p_ref_raw))
    # Back-calculation anti-windup: undo integral contribution that
    # was just clipped, so the integral state does not store an
    # unrealizable command.
    if self.Ki > 0:
        self._agc_integral -= (p_ref_raw - p_ref_new) / self.Ki
    self._p_ref = p_ref_new
    ...
```

### 5.3 Paper text — Section III §III-C addendum

Inserted into `paper/section3_math.tex`, frequency-response model
subsection:

> *The implemented frequency-response model includes a single
> internal PI-based secondary control loop that restores $\Delta f_t$
> toward zero on a slow timescale. This secondary loop is part of the
> simulator infrastructure and is not learned by the AM policy; the
> RL controller operates only on the fast FFR action
> $\Delta P^{\mathrm{FFR}}_{i,t}$ defined above.*

### 5.4 Paper text — Section V opening disclaimer

Inserted into `paper/section5_method.tex`, preamble:

> *The RL policy controls only the fast FFR channel
> $\Delta P^{\mathrm{FFR}}_{i,t}$. Steady-state frequency restoration
> is handled by an underlying PI secondary loop with deadband and
> clipped reference, fixed across all baselines and not modified
> during training.*

---

## 6. Issues retained for future work

These are real but out of scope for the current paper:

1. **Per-zone economic AGC.** A future paper may treat the distributed
   AGC as a contribution; in that case `USE_DISTRIBUTED_AGC=True` and
   the back-calculation anti-windup plus per-agent participation
   factors $\alpha_i$ should be re-introduced.
2. **Multi-area ACE.** Currently the simulator is a single area; a
   multi-area extension would benefit from explicit bias $\beta_z$
   per zone.
3. **EMT fidelity.** The aggregate swing equation is not a substitute
   for EMT converter dynamics. This is already disclosed at the end
   of Section III-C.

---

## 7. Cross-reference index — verified via NotebookLM

| Claim | Source | Verified in notebook |
|---|---|---|
| Secondary AGC must run at fractions of Hz | Anttila et al. 2022 *Energies* "Grid Forming Inverters: Review" | Notebook 7 *(100% IBR)* |
| Distributed secondary via Laplacian consensus (5 GFM inverters; 100 ms vs 1 ms timescale) | Ojo et al. 2023 *IEEE TCST* | Notebook 7 |
| Anti-windup (clamping or back-calculation) needed when current limiting binds | Salem et al. 2025 *IEEE OJ-IES* | Notebook 7 |
| ACE = $K_i/s\cdot[\beta\,\Delta f]$; $\beta=1$ pu MW/Hz; $K_I=0.09$ | Saxena et al. 2020 *J. Energy Storage* | Notebook 4 *(VSG/Frequency)* |
| ACE PI tuning by IMC: $K_I=22.114$, $K_P=3.581$, $\beta=1$ | Kerdphol et al. 2017 *IEEE Access* | Notebook 4 |
| DE-tuned ACE controller for low-inertia MG with synthetic inertia | "Adaptive Synthetic Inertia Control Framework" (Notebook 4) | Notebook 4 |
| $H_{\text{SYS}}<3$ s for high-RES islanded MG; metaheuristic VIC with EVs | Mishra et al. 2024; Hamanah et al. 2024; Kerdphol 2017 | Notebook 4 |

---

## 8. Decision log

| Date | Decision | Rationale |
|---|---|---|
| 2026-05-15 | Adopt Option (a): AGC = infrastructure | RL claim is AM/FFR only; cascading integrators were unstable and not part of contribution |
| 2026-05-15 | Disable `VPPSecondaryControl` and `BatterySecondaryControl` by feature flag | Single integrator semantics; preserve code for future work |
| 2026-05-15 | Add Section III-C and Section V disclaimers | Reviewer-facing transparency that RL does not learn AGC |
| 2026-05-15 (pending) | Tighten deadband to 0.05 Hz | Saxena/Kerdphol alignment |
| 2026-05-15 (pending) | Add back-calculation anti-windup in `FrequencyDynamics` | Salem 2025 recommendation |

---

## 9. References

1. Saxena, P., Singh, N., Pandey, A. K. (2020). Enhancing the dynamic
   performance of microgrid using derivative controlled solar and
   energy storage based virtual inertia system. *J. Energy Storage*,
   31, 101613.
2. Kerdphol, T., Rahman, F. S., Mitani, Y., Hongesombut, K., Kufeoglu,
   S. (2017). Robust virtual inertia control of an islanded microgrid
   considering high penetration of renewable energy. *IEEE Access*,
   5, 25958–25969.
3. Anttila, S., Dollón, J. B., Pollanen, M., Martínez, J. A. (2022).
   Grid forming inverters: a review of the state of the art of key
   elements for microgrid operation. *Energies*, 15(15), 5517.
4. Ojo, Y. et al. (2023). A distributed scheme for voltage and
   frequency control and power sharing in inverter-based microgrids.
   *IEEE Trans. Control Systems Technology*.
5. Salem, Q. et al. (2025). Grid forming converters for low-inertia
   systems — capabilities and limitations: a critical review.
   *IEEE Open Journal of the Industrial Electronics Society*, 6.
6. Mishra, S. et al. (2024). A metaheuristic algorithm for regulating
   virtual inertia of a standalone microgrid incorporating electric
   vehicles. *J. Engineering*.
7. Hamanah, W. M. et al. (2024). Realization of robust frequency
   stability in low-inertia islanded microgrids with optimized
   virtual inertia control. *IEEE Access*, 12.
8. "Adaptive Synthetic Inertia Control Framework for Distributed
   Energy Resources in Low-Inertia Microgrid" (Notebook 4 source;
   K_I=146.3, K_P=148.9, K_VI=5.35, D_VI=5.62).
