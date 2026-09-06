# Project Concept v3.2 — Measurement-Revised Research Design

## Closed-Form Frequency Security Limits for 100% Inverter-Based Islanded Microgrids: Validity Envelope and Dimensionless Screening Criteria

> **Title changed on 2026-09-03.** The v3.1 title — *When Do Reduced Frequency Models
> Overestimate Security in 100% Inverter-Based Islanded Microgrids?* — is retired because the
> campaign answered its question in the negative and the answer is now measured, not assumed.
> Inside the energy-feasible set the reduced model does not overestimate security: its nadir
> error is \(\le 1.44\%\) and the closed-form boundary predicts the simulated boundary to
> \(0.26\%\). Retaining a title that asks "when does it fail?" would invite the reviewer
> question *"and your answer is: never, in everything you tried?"* — which reads as a null
> paper. The measured content is the opposite in sign and stronger in kind: the reduced model
> is exact, its validity envelope coincides with the energy-feasibility boundary, and its
> failure outside that envelope is a monotone degradation law rather than a cliff.
> §15 pre-committed to this interpretation as **Outcome B**; the pre-commitment is what makes
> the change legitimate rather than post-hoc. The v3.1 fallback title is superseded by the
> title above, which names the positive content instead of the abandoned question.

**Every claim in this document is now classified.** Section 28 is the authoritative register of
what the paper may claim, what it may **not** claim, the reason, and the experiment that
settles it. No statement anywhere in this document overrides §28.

**Primary target:** IEEE Transactions on Power Systems (TPWRS)  
**Primary quantitative testbed:** modified IEEE 123-bus distribution feeder  
**Real-island motivation/reference:** Bach Long Vy island microgrid  
**Main GFM control mode:** multiloop droop GFM (P–f and Q–V); VSG is outside the scope of the first paper  
**Methodological hierarchy:** established analytical screening model → applicability check → converter-level EMT verification  
**AI status:** not a core contribution in the first paper; only a fallback acceleration layer if physics-based screening + adaptive EMT sampling is insufficient.

---

# 1. Frozen project statement

> **Renewable-rich islands are approaching operating intervals in which the last synchronous diesel unit can be disconnected. This project determines the operating conditions under which the resulting 100% inverter-based islanded microgrid remains frequency-secure, accounting for finite GFM headroom, converter current limits, GFM spatial deployment, and physically reachable feeder reconfiguration.**

The central causal chain is

\[
\boxed{
(P_{\mathrm{head}},\; I_{\max},\; \mathcal G_{\mathrm{GFM}},\; G,\; d)
\rightarrow
\text{deliverable GFM response}
\rightarrow
\text{converter-limited dynamics}
\rightarrow
\text{frequency security}.
}
\]

State of charge is retained as an **energy-feasibility and operating-state variable**, but is no longer assumed to independently dominate sub-second frequency dynamics.

---

# 2. Test-system decision — frozen

## 2.1 Main system: modified IEEE 123-bus feeder

The **modified IEEE 123-bus feeder** is the main quantitative testbed because it already provides the degrees of freedom required by the study:

- 47 physically feasible radial switching configurations;
- a six-GFM deployment plus 2→5→6 GFM comparison;
- explicit topology dependence;
- sufficient spatial resolution for bus-level frequency/voltage analysis;
- a tractable but meaningful EMT benchmark.

All results involving 47 configurations, six GFM units, and the electrical-distance/accessibility metrics belong to this feeder.

## 2.2 Role of Bach Long Vy

Bach Long Vy is **not** the six-GFM reconfiguration testbed. Its role is:

1. empirical motivation for diesel-free island operation;
2. evidence that an EMS may schedule intervals with all diesel units disconnected;
3. a source of realistic island operating profiles/ranges when useful;
4. a motivating case showing why energy feasibility should be followed by a dynamic-security check.

Preferred wording:

> *Motivated by real island microgrids that can schedule diesel-free operating intervals, this work studies the dynamic-security limits of 100% inverter-based operation on a modified IEEE 123-bus feeder with controllable GFM deployment and feasible radial reconfiguration.*

A full Bach Long Vy EMT model is outside the first paper to avoid doubling the dominant implementation burden.

---

# 3. Central event restored: last-diesel-off transition

The paper explicitly models

\[
\boxed{
\text{hybrid diesel + IBR island}
\rightarrow
\text{last diesel OFF}
\rightarrow
\text{100% IBR island}.
}
\]

Before \(t=t_{\mathrm{off}}\), one synchronous diesel generator remains connected, the droop GFMs are already online and synchronized, and GFL resources inject scheduled P/Q. The first paper does not introduce a GFL↔GFM mode-switching problem; the designated GFM units remain in droop-GFM mode before and after diesel disconnection.

Two diesel-off tests are required:

### Controlled diesel-off

\[
P_{\mathrm{DG}}(t_{\mathrm{off}}^-)\approx0.
\]

This minimizes active-power-loss confounding and isolates the loss of the final synchronous machine and its electromechanical support.

### Loaded diesel trip

\[
P_{\mathrm{DG}}(t_{\mathrm{off}}^-)>0.
\]

This combines loss of the synchronous machine with a sudden active-power deficit and is the more severe contingency.

---

# 4. Motivation — wording corrected

The paper must **not** claim that

\[
\text{energy feasible}\not\Rightarrow\text{frequency secure}
\]

is itself new. That statement belongs to the established frequency-constrained scheduling/unit-commitment literature.

The actual question is:

> **When a microgrid reaches a scheduled diesel-free operating point, do finite inverter headroom, current limits, GFM location, and feeder configuration cause the actual converter-level security boundary to differ from the boundary predicted by established reduced analytical models?**

---

# 5. Research gap after novelty audit

The new novelty audit changes the **priority** of the paper. The strongest contribution is no longer the security-region construction by itself; it is the **system-level validity boundary of a reduced analytical frequency model expressed in operational/converter-utilization coordinates**.

## 5.1 Reduced-model validity is now the lead gap

Prior work already addresses converter model reduction, including current limiting, inner-loop effects, EMT-to-RMS reduction, and scenario-dependent model selection. The important distinction is that these studies are predominantly device-level, small-signal, frequency-domain, or model-order-reduction studies.

The target gap is therefore narrower:

> **Can the applicability of an established reduced frequency/voltage model be quantified at the system level using operational indicators such as current utilization and headroom utilization, and linked directly to errors in frequency-security assessment?**

The paper does **not** claim that model-validity boundaries are new. It aims to translate model validity into operator-relevant quantities:

\[
\mu_I,\qquad \mu_P,\qquad
|f_{\mathrm{nadir}}^{\mathrm{ana}}-f_{\mathrm{nadir}}^{\mathrm{EMT}}|,
\]

and ultimately an empirically validated applicability envelope.

## 5.2 Headroom/GFM allocation already has a close competitor

Recent work has already used EMT-trained learning models to estimate required reserved headroom and GFM allocation inside frequency-aware dispatch. Therefore, “required headroom + GFM allocation” cannot be presented as an untouched gap.

Our remaining distinction is:

- no cost/OPF novelty claim;
- an explicit **analytical-model validity question**;
- converter-current/headroom applicability indicators;
- physically reachable feeder reconfiguration;
- controlled and loaded last-diesel-off transitions;
- security boundaries rather than only an optimal dispatch point.

## 5.3 Frequency-constrained scheduling is mature

Frequency-aware scheduling already embeds nadir, RoCoF, reserve, voltage-transient and islanding constraints in optimization formulations. Thus

\[
\text{energy feasible}\not\Rightarrow\text{frequency secure}
\]

remains motivation, not contribution.

## 5.4 100% IBR dynamic security is established

High-fidelity EMT dynamic-security assessment and optimization for 100% IBR island systems already exist. The paper therefore cannot claim the first 100% IBR dynamic-security framework.

The differentiation is the combination of:

\[
\text{reduced-model applicability}
+\text{operational reserve boundary}
+\text{converter constraints}
+\text{reconfiguration/diesel-off transition}.
\]

## 5.5 Current limiting is established

Current limiting can change voltage-source behavior, synchronization and transient stability. The first paper does not propose a new limiter. It studies when limiter/current-headroom proximity makes a reduced analytical security prediction unreliable or conservative/non-conservative.

## 5.6 Security regions already exist

Steady-state inverter security regions and dynamic converter stability regions already exist in power-injection and controller-parameter spaces.

The intended coordinates here are operational:

\[
\boxed{
P_{\mathrm{head}},\;\Delta P,\;\mu_I,\;\mu_P,\;G,\;\mathcal G_{\mathrm{GFM}}
}
\]

with an emphasis on the **minimum secure dynamic reserve** rather than claiming the first inverter security region.

## 5.7 GFM placement is a mature field

The statement “GFM placement matters” is not novel and must not appear as a main scientific discovery. Placement is retained as an experimental factor.

The narrower result of interest is:

> **How much does placement/reconfiguration change the current-constrained minimum headroom requirement once aggregate capacity is controlled?**

The headroom-weighted accessibility index is therefore a diagnostic/explanatory metric, not the central novelty.

# 6. Fast timescale vs. energy timescale — SoC corrected

The previous SoC vs. headroom signature map is **not frozen** because SoC may have little independent influence on sub-second nadir/RoCoF once its impact on power capability is already represented by headroom.

Define the energy-feasible set

\[
\Omega_E=
\left\{
x:
SoC_{\min}\le SoC\le SoC_{\max},\;
E_{\mathrm{avail}}\ge E_{\mathrm{req}},\;
P_{\mathrm{BESS}}\ \text{dispatch feasible}
\right\}.
\]

Define the fast dynamic-security set

\[
\Omega_D(G)=
\left\{
x:
 f_{\mathrm{nadir}}\ge f_{\min},\;
 V\ \text{acceptable},\;
 I\le I_{\max},\;
 \text{synchronization retained}
\right\}.
\]

The diesel-free admissible set is

\[
\boxed{\Omega_{\mathrm{DF}}=\Omega_E\cap\Omega_D.}
\]

Before the main campaign, run a small SoC sensitivity sweep at fixed headroom, disturbance, topology and GFM deployment. If the fast response is insensitive to SoC, keep SoC in \(\Omega_E\) rather than forcing it onto the fast-security map.

> **Measured on 2026-09-03 (T26, `artifacts/T26_phead_dp1p1/`, `artifacts/T26_phead_dp0p6/`).**
> On this system the intersection **collapses**: \(\Omega_D\) does not cut inside \(\Omega_E\).
> The measured reserve multiplier is
> \[
> \kappa=\frac{P_{\mathrm{head}}^{\min}}{|\Delta P|}=1.006\pm0.002
> \qquad
> \left(
> \Delta P=0.6\ \mathrm{MW}\Rightarrow P_{\mathrm{head}}^{\min}=0.5953,\ \kappa=1.0078;\quad
> \Delta P=1.1\ \mathrm{MW}\Rightarrow P_{\mathrm{head}}^{\min}=1.0947,\ \kappa=1.0049
> \right),
> \]
> i.e. the headroom boundary **is** the power-balance feasibility boundary to within 0.6%, and
> the excess dynamic reserve \(P_{\mathrm{excess}}=P_{\mathrm{head}}^{\min}-|\Delta P|\) is
> below the resolution at which it could be reported as an engineering quantity.
>
> **Consequence for the motivation.** The sentence *"energy feasible \(\not\Rightarrow\)
> frequency secure"* is already not a contribution (§4). It is now additionally **not true in a
> quantitatively interesting way on this testbed**: the measured gap between the two sets is
> 0.6%. The manuscript must not motivate the work by asserting a large gap it has measured to
> be small. The honest framing is the reverse and is stronger: *on a 100% GFM island with
> \(\Lambda\gg1\), the fast dynamic-security set is recovered by an energy-feasibility check
> plus one closed-form inequality — and this document states the conditions under which that is
> true and the degradation law for when it is not.* See §28, item N4.

---

# 7. New signature security map

The primary candidate map is

\[
\boxed{P_{\mathrm{head}}\ \text{vs.}\ \Delta P}
\]

for fixed topology, GFM deployment, disturbance location/type, and an energy-feasible SoC band.

The boundary gives

\[
P_{\mathrm{head}}^{\min}(G,\Delta P,\mathcal G_{\mathrm{GFM}}).
\]

Overlay:

- analytical boundary;
- EMT boundary;
- current-limiter activation boundary;
- topology variants;
- 2-, 5-, 6-GFM variants.

> **Revised on 2026-09-03.** Three of the five overlays were measured to carry no information
> on this feeder and must not be drawn as if they did: the current-limiter activation boundary
> does not exist for any spec-compliant limiter (§28 N3), the topology variants are
> coincident to 0.0000 MW (§28 N2), and the 2-GFM member set is unrecorded (§10).
> Drawing five overlapping curves and calling the overlap a result is the weakest possible use
> of the measurement. The signature figure is therefore replaced by **Figure G** of §21, which
> plots the same physics along the one axis on which something actually happens:
> \[
> \frac{P_{\mathrm{head}}}{|\Delta P|}
> \quad\text{vs.}\quad
> f_{\mathrm{nadir}},
> \]
> with the closed-form prediction as a horizontal line, the measured points tracking it above
> \(\kappa=1.006\) and diverging below it, and the infeasible region shaded. That single figure
> carries the closed form, its envelope, and its failure mode.

---

# 8. GFM accessibility metric — canonical definition

Let \(\mathcal L\) be the load-bus set, \(\mathcal G\) the online-GFM set, \(P_{L,k}\) the active load at bus \(k\), \(P_{\mathrm{head},g}\) the upward active-power headroom of GFM \(g\), \(Z_{gk}^{\mathrm{eq}}(G)\) the electrical distance under topology \(G\), and \(Z_{\mathrm{base}}\) a common impedance base.

Define

\[
w_k=\frac{P_{L,k}}{\sum_{j\in\mathcal L}P_{L,j}},
\qquad \sum_k w_k=1,
\]

and

\[
\pi_g=\frac{P_{\mathrm{head},g}}{\sum_{h\in\mathcal G}P_{\mathrm{head},h}},
\qquad \sum_g\pi_g=1.
\]

The normalized GFM accessibility index is

\[
\boxed{
\bar A_{\mathrm{GFM}}(G)
=
\sum_{k\in\mathcal L}
w_k
\sum_{g\in\mathcal G}
\pi_g
\frac{Z_{\mathrm{base}}}{|Z_{gk}^{\mathrm{eq}}(G)|+\epsilon}
}
\]

and is dimensionless.

Interpretation:

- larger \(\bar A_{\mathrm{GFM}}\): headroom is spatially more accessible to load/disturbance locations;
- smaller \(\bar A_{\mathrm{GFM}}\): available GFM headroom is electrically more remote.

A raw diagnostic metric may be retained internally,

\[
A_{\mathrm{GFM}}^{\mathrm{raw}}
=
\sum_k P_{L,k}
\sum_g
\frac{P_{\mathrm{head},g}}{|Z_{gk}^{\mathrm{eq}}|},
\]

but it should not be the main published index because it mixes capacity and placement and carries awkward units.

**Implementation rule:** verify the existing code against the canonical \(\bar A_{\mathrm{GFM}}\) formula before manuscript use.

---

# 9. Existing configuration findings — retained with narrower interpretation

For the six-GFM configuration phase:

- 64 raw switching combinations;
- 47 feasible radial/reachable configurations;
- 0 connectivity failures after adding G6;
- 16 cycle-related rejections;
- 1 duplicate.

Edge-Jaccard distance:

\[
d_J^{\min}=0.0083,\qquad
\bar d_J=0.0253,\qquad
d_J^{\max}=0.0492.
\]

This is a **structurally compact physically reachable reconfiguration space**, not arbitrary graph transfer.

Across the 47 configurations:

| Pair | Pearson | Spearman |
|---|---:|---:|
| \(d_J\) vs. \(|\Delta Z_{\mathrm{avg}}|\) | 0.09 | 0.35 |
| \(d_J\) vs. \(|\Delta Z_{\max}|\) | 0.37 | 0.53 |
| \(d_J\) vs. \(|\Delta Z_{95}|\) | 0.37 | 0.53 |
| \(d_J\) vs. \(|\Delta \bar A_{\mathrm{GFM}}|\) | 0.66 | 0.73 |

> **Flagged by T1 (2026-09-02).** Every number in this section derives from the same untracked,
> unreproducible `artifacts/electrical_distance_analysis.json` as Section 10, and the last row
> uses the \(\bar A_{\mathrm{GFM}}\) implementation that T1 showed does not match Section 8.
> The 47-configuration sweep must be re-run with `src/analytical/accessibility.py` before any
> of it enters the manuscript. Scheduled with the axis-B campaign, not with T1.

The configuration phase therefore does **not** support a dramatic structural/electrical-decoupling claim. The remaining hypothesis is threshold-based: modest electrical changes may still shift the dynamic boundary when one or more GFMs are close to headroom/current limits.

---

# 10. Existing GFM-placement finding — retained

At nominal topology \(G_0\):

| GFM deployment | \(Z_{\mathrm{avg}}\) [Ω] | \(Z_{\max}\) [Ω] | raw \(A_{\mathrm{GFM}}\) |
|---|---:|---:|---:|
| 2 GFM | 0.737 | 1.743 | 680,265 |
| 5 GFM | 0.350 | 0.932 | 2,466,067 |
| 6 GFM | 0.294 | 0.824 | 3,146,148 |

Measured 2→6 changes:

\[
Z_{\mathrm{avg}}\downarrow60.1\%,
\qquad
Z_{\max}\downarrow52.7\%.
\]

> **Superseded on 2026-09-02 by T1** (`artifacts/T01_agfm/`, `experiments/t01_agfm.py`).
> The figures previously reported here — \(+121.2\%\) for 2→6 and \(+16.5\%\) for 5→6, labelled
> "normalized-analysis changes" — are the behaviour of the **raw** index \(A_{\mathrm{GFM}}\),
> not of the canonical dimensionless \(\bar A_{\mathrm{GFM}}\) of Section 8. Recomputing the raw
> index from the current network reproduces them (\(+120.8\%\) and \(+17.8\%\)), which confirms
> the mislabelling. This is exactly the failure Section 8 warns about: the raw index "mixes
> capacity and placement".
>
> The table above and the two \(Z\) reductions are likewise not reproducible: the script that
> produced `artifacts/electrical_distance_analysis.json` is absent from the working tree **and
> from the entire git history**, and the artifact itself is untracked (violates N-2).

Recomputed against the canonical Section 8 definition, at \(G_0\), with \(P_{\mathrm{head},g}\)
taken as the BESS rating and \(\epsilon_g\) as each converter's own coupling impedance
(0.10 pu on the unit's MVA base):

| Quantity | 2→6 GFM | 5→6 GFM |
|---|---:|---:|
| \(\bar A_{\mathrm{GFM}}\), canonical, \(\epsilon\) physical | \(-0.05\%\) | \(-2.6\%\) |
| \(\bar A_{\mathrm{GFM}}\), canonical, \(\epsilon\) numerical | \(+10.4\%\) | \(+8.0\%\) |
| \(Z_{\mathrm{avg}}\) to nearest GFM | \(-46.8\%\) | — |
| \(Z_{\max}\) to nearest GFM | \(-30.7\%\) | — |

Three consequences.

1. \(\bar A_{\mathrm{GFM}}\) is **not monotone** in the number of GFMs. Because \(\pi_g\) is a
   normalized weight, adding a unit redistributes headroom weight rather than adding it, so
   axis A is close to flat in accessibility while being large in capacity.
2. \(\epsilon\) is not a numerical detail. Four of the 85 load buses host a GFM, so
   \(|Z_{gk}|=0\) there; and at 4.16 kV a converter's coupling impedance (0.12–1.30 Ω over
   0.05–0.15 pu) is the same order as the feeder's entire electrical span
   (\(Z_{\max}=0.73\) Ω). The choice of \(\epsilon\) flips the sign of the 2→6 change.
   Section 8 must state it.
3. The 2-GFM membership is unrecorded and dominates the 2→6 number: across all 15 pairs
   \(\Delta\bar A_{\mathrm{GFM}}\) ranges from \(-17.6\%\) to \(+40.9\%\). The 5→6 figure is
   free of that assumption and should lead.

This weakens C3 further, in the direction already anticipated by risk R4.

---

# 11. Modeling hierarchy — hybrid SG/GFM model and applicability gate

The central methodological rule remains:

\[
\boxed{
\text{established analytical model}
\rightarrow
\text{validated applicability gate}
\rightarrow
\text{converter-level EMT reference}.
}
\]

The new version explicitly includes the synchronous diesel generator required by the last-diesel-off transition.

## Level 1 — established hybrid SG/GFM analytical screening model

Before diesel disconnection the network is mixed SG/GFM; after disconnection it becomes all-inverter. Therefore the analytical layer follows the mixed-generation formulation of Trujillo et al. rather than using only the droop-GFM block.

To avoid notation collision with network topology \(G\), the manuscript should denote synchronous-generator indices by \(\mathcal S\) and GFM indices by \(\mathcal I\), while noting that this is the \(\mathcal G/\mathcal I\) partition used in the source model.

### Droop GFM \(i\in\mathcal I\)

\[
\dot{\Delta\delta_i}=\omega_0\Delta\omega_i
\]

\[
\dot{\Delta\omega_i}
=-\frac{1}{T_{c,i}}\Delta\omega_i
+\frac{\alpha_iR_i}{T_{c,i}}\Delta P_{G,i}.
\]

### Synchronous diesel generator \(i\in\mathcal S\)

Following the sign convention of the Trujillo mixed-generation model:

\[
\dot{\Delta\delta_i}=\omega_0\Delta\omega_i
\]

\[
\dot{\Delta\omega_i}
=-\frac{D_i}{M_i}\Delta\omega_i
-\frac{1}{M_i}\Delta P_{M,i}
+\frac{\alpha_i}{M_i}\Delta P_{G,i}
\]

\[
\dot{\Delta P_{M,i}}
=\frac{K_i}{T_{SG,i}R_{SG,i}}\Delta\omega_i
-\frac{1}{T_{SG,i}}\Delta P_{M,i}.
\]

Thus the SG contributes three frequency states \((\Delta\delta,\Delta\omega,\Delta P_M)\), while each droop GFM contributes two.

### Network coupling

\[
\Delta P_G=B_r(G)\Delta\delta_G+B_L(G)\Delta P_L.
\]

The hybrid state-space system is assembled using the SG and GFM index sets. The diesel-off event is represented by changing the online device set and the corresponding network/device matrices at \(t=t_{\mathrm{off}}\).

The analytical model is used for screening only under its stated assumptions, especially adequate headroom and negligible saturation nonlinearities.

## Level 2 — analytical applicability estimates

A distinction is required between **analytical screening estimates** and **EMT-measured utilization**.

### 2.1 Analytical current estimate

The LIFE/LIVE-type reduced model has no inner-loop current state. In per unit, estimate current magnitude from analytical power/voltage outputs:

\[
\boxed{
\hat I_i^{\mathrm{ana}}(t)
=\frac{\sqrt{P_i^2(t)+Q_i^2(t)}}{|V_i(t)|}
}
\]

and define

\[
\boxed{
\hat\mu_I^{\mathrm{ana}}
=\max_{i,t}
\frac{\hat I_i^{\mathrm{ana}}(t)}{I_{i,\max}}.
}
\]

This is **not** assumed to be ground truth. Before it is used as a screening gate, the current estimator must be independently validated against EMT:

\[
e_I=
\left|
\hat\mu_I^{\mathrm{ana}}-\mu_I^{\mathrm{EMT}}
\right|.
\]

Only after this validation can a conservative analytical screening threshold \(\hat\mu_I^{\star}\) be chosen.

### 2.2 Headroom-demand estimate

Define

\[
\boxed{
\hat\mu_P^{\mathrm{ana}}
=\max_i
\frac{\Delta P_{i,\mathrm{req}}^{\mathrm{ana}}}
{P_{i,\mathrm{head}}}.
}
\]

This is a **trigger**, not a trusted extrapolation into saturation. If

\[
\hat\mu_P^{\mathrm{ana}}\ge1,
\]

the analytical model has already reached an excluded regime and the operating point is routed to EMT rather than interpreted quantitatively from the reduced model.

> **Superseded on 2026-09-03 (T26).** This gate is over-conservative by a factor of ~1.4 in
> \(\mu_P\). Measured, peak \(\mu_P\) does not separate the accurate regime from the inaccurate
> one: \(\mu_P^{\mathrm{peak}}=1.067\) gives \(-0.06\%\) error and \(1.434\) gives \(+1.44\%\),
> both with **no unit holding its ceiling at steady state**. A transient excursion above the
> ceiling that the \(K_{P,\mathrm{plim}}\) loop pulls back is harmless. The gate is replaced by
> sustained ceiling occupancy at the post-event steady state, which coincides to 0.6% with
> \(P_{\mathrm{head}}\ge P_{\mathrm{head}}^{\min}\). See §14 H1(iii) for the sweep and §28 N5
> for the scope of the coincidence.

### 2.3 Final EMT utilization indicators

The final validity relation is constructed from EMT-measured quantities:

\[
\mu_I^{\mathrm{EMT}}
=\max_{i,t}\frac{I_i^{\mathrm{EMT}}(t)}{I_{i,\max}}
\]

and the realized headroom utilization.

The intended product is an empirical applicability envelope such as

\[
\boxed{
\mu_I^{\mathrm{EMT}}\le \mu_I^{\star},
\qquad
\mu_P\le \mu_P^{\star}
\Rightarrow
\varepsilon_f\le\varepsilon_f^{\max}
}
\]

for predefined prediction-error tolerances.

## Level 3 — converter-level EMT reference including diesel SG

The EMT model contains two device families.

### GFM-BESS

\[
\mathrm{BESS/DC}
\rightarrow
\mathrm{DC\ link}
\rightarrow
\mathrm{VSC}
\rightarrow
\mathrm{LCL}
\rightarrow
\mathrm{network}
\]

with P/Q measurement filters, P–f droop, Q–V droop, voltage/current PI loops, current limiter, anti-windup, and the selected established virtual-impedance/feed-forward terms.

### Synchronous diesel generator

The pre-transition EMT model must include:

- synchronous machine electromagnetic/electromechanical dynamics;
- diesel engine/turbine-governor dynamics;
- AVR/exciter;
- machine limits needed for the simulated transition;
- breaker/disconnection logic at \(t_{\mathrm{off}}\).

The exact standard SG/governor/AVR model family remains an implementation choice, but omission of the diesel dynamics is no longer permitted because the last-diesel-off transition is a central experiment.

## Level 4 — matched-assumption vs. native-unbalance EMT validation

Because the analytical model is based on balanced/positive-sequence-type network assumptions while the IEEE 123 feeder is inherently unbalanced, the campaign should avoid confounding model-order error with phase-unbalance error.

Use two EMT tiers:

1. **Matched-assumption EMT:** a balanced three-phase / positive-sequence-consistent version of the modified IEEE 123 feeder for the main analytical-vs-EMT validity study.
2. **Native unbalanced EMT subset:** the original three-phase unbalanced feeder for a smaller set of representative boundary cases to test whether the conclusions remain qualitatively valid under feeder unbalance.

The size of the second tier is determined after runtime benchmarking.

# 12. GFM mode — frozen

The main paper uses

\[
\boxed{\text{multiloop droop-controlled GFM}}
\]

for all designated GFM assets.

This matches the six-GFM study, the first-order analytical screening model, and established detailed EMT implementations. VSG is explicitly outside scope.

---

# 13. RoCoF — physical security vs. operational compliance

Large local instantaneous RoCoF in high-GFM systems does not automatically imply physical instability. Therefore define two nested regions.

Physical dynamic-security region:

\[
\Omega_{\mathrm{dyn}}
=
\left\{
x:
\text{stable synchronization},\;
f_{\mathrm{nadir}}\ge f_{\min},\;
V_i\in[V_{\min},V_{\max}],\;
I_i\le I_{i,\max},\;
\text{successful recovery}
\right\}.
\]

Operational/protection-compliant region:

\[
\boxed{
\Omega_{\mathrm{op}}
=
\Omega_{\mathrm{dyn}}
\cap
\{|\mathrm{RoCoF}_{\mathrm{meas}}|\le R_{\mathrm{relay/code}}\}.
}
\]

The RoCoF algorithm must state sampling rate, filtering/window, local vs. center metric, and post-event reporting interval.

---

# 14. Unified hypotheses

> **Status on 2026-09-03.** All four hypotheses have been tested in the positive-sequence layer
> and each carries its outcome below. **Three of the four returned a result opposite in sign to
> the direction the hypothesis anticipated, and the fourth returned a null.** These outcomes are
> retained verbatim in the manuscript together with the measurements that produced them. They
> were written down before the campaign, which converts the campaign from an exploratory search
> into a pre-registered falsification attempt — and that is the strongest defence available
> against the reviewer objection *"your positive result is that a simple formula works; where is
> the contribution?"* The answer is that four independent attempts were designed to break the
> formula and the attempts, not the formula, are what failed. Deleting the refuted hypotheses
> would destroy exactly the evidence that makes the surviving result credible.
>
> All outcomes below are **positive-sequence (ANDES 2.0) measurements**. None is EMT-verified.
> §28 item N9 governs how far they may be stated.

## H1 — reduced-model applicability (lead hypothesis)

\[
\boxed{
\text{The error of the reduced analytical frequency/voltage model can be bounded using converter/headroom utilization indicators once the analytical current estimator has been validated against EMT.}
}
\]

This is outcome-neutral: the result may be a narrow failure region or a broad validated applicability envelope.

> ### H1 — outcome: **broad validated envelope, and the envelope is the feasibility boundary**
>
> Evidence: `artifacts/T20_andes_bisect*/`, `artifacts/T21_*/`, `artifacts/T26_phead_dp1p1/`.
>
> **(i) The closed form.** In the unconstrained regime the steady-state deviation is exact:
> \[
> \Delta f_{ss}=\frac{f_0 R\,\Delta P}{\sum_g S_g}
> \qquad\text{(agreement to four significant figures)},
> \]
> and the nadir is a fixed multiple of it,
> \[
> \boxed{\;f_{\mathrm{nadir}}=f_0-\kappa_{os}\,\Delta f_{ss},\qquad
> \Delta P_{\max}=\frac{(f_0-f_{\min})\sum_g S_g}{\kappa_{os}\,f_0\,R}\;}
> \]
> with \(\kappa_{os}\) measured over 23 operating points spanning a 36\(\times\) range in
> \(\Delta P\). The closed form predicts \(\Delta P_{\max}=1.1881\) MW against a bisected
> \(1.18506\) MW — **0.26%**.
>
> **(ii) \(\kappa_{os}\) is event-class dependent and must not be quoted as one number.**
>
> | event class | \(\kappa_{os}\) | \(\sigma\) |
> |---|---:|---:|
> | non-synchronous source loss (`gen_loss`) | 1.2275 | 0.24% |
> | synchronous machine trip (last diesel) | 1.1888 | 1.7% |
>
> The two differ by 3.2% and the scatter differs by a factor of 8. A pooled constant
> (1.2239 ± 0.33%) exists but is an artefact of averaging two populations and is **not** to be
> published as a single value. The mechanism of the 3.2% gap is **not established** — see
> §28 item N8.
>
> **(iii) The envelope is not the binary condition anticipated.** The applicability limb of H1
> was written expecting a threshold in peak \(\mu_I,\mu_P\). Measured at \(\Delta P=1.1\) MW,
> where the closed form predicts \(f_{\mathrm{nadir}}=59.0714\) Hz:
>
> | \(P_{\mathrm{head}}\) [MW] | \(P_{\mathrm{head}}/|\Delta P|\) | units saturated at steady state | peak \(\mu_P\) | \(f_{\mathrm{nadir}}\) [Hz] | error |
> |---:|---:|---:|---:|---:|---:|
> | 3.414 → 1.3115 (5 pts) | 3.10 → 1.19 | 0 | 0.300 → 1.067 | 59.0720 | −0.06% |
> | 1.1013 | 1.001 | 0 | 1.434 | 59.0581 | +1.44% |
> | 1.0881 | 0.989 | 6 | 1.466 | 59.0493 | +2.39% |
> | 1.0487 | 0.953 | 6 | 1.569 | 59.0133 | +6.26% |
> | 0.9961 | 0.905 | 6 | 1.732 | 58.8845 | +20.1% |
> | 0.8910 | 0.810 | 6 | 2.187 | 58.5229 | +59.1% |
> | 0.7228 | 0.657 | 6 | 3.766 | 57.9443 | +121.4% |
>
> Two corrections follow, and both are stronger than the hypothesis they replace.
>
> **Correction 1 — the gate is sustained saturation, not peak utilisation.** At
> \(P_{\mathrm{head}}=1.3115\) MW the peak \(\mu_P\) is already 1.067 and the error is
> \(-0.06\%\); at 1.1013 MW the peak is 1.434 and the error is 1.44%. A transient excursion
> above the ceiling that the \(K_{P,\mathrm{plim}}\) loop pulls back is **harmless**. The
> correct applicability condition is
> \[
> \boxed{\;\text{no unit holds its active-power ceiling at the post-event steady state.}\;}
> \]
> This is counter-intuitive and could not have been reached by argument; it required the sweep.
> The v3.1 gate \(\hat\mu_P^{\mathrm{ana}}\ge1 \Rightarrow\) route to EMT (§11 Level 2.2) is
> **over-conservative by a factor of ~1.4 in \(\mu_P\)** and is superseded.
>
> **Correction 2 — outside the envelope the failure is a monotone degradation law, not a
> cliff.** Error grows from 2.39% to 121.4% as the feasibility deficit deepens. A single
> threshold understates what is measurable.
>
> **(iv) The envelope coincides with the feasibility boundary.** The 0 → 6 saturation
> transition falls between \(P_{\mathrm{head}}=1.1013\) and \(1.0881\) MW; the independently
> bisected \(P_{\mathrm{head}}^{\min}=1.0947\) MW (H2 below) falls inside that interval.
> **Scope of this identification:** the two boundaries coincide *to within the 0.6% resolution
> of the sweep*. They are **not** proven analytically identical, and the manuscript must say
> "coincide to 0.6%", never "are identical". Two further bisection steps between 1.0881 and
> 1.1013 would tighten this; see §26 task 3.

## H2 — nontrivial dynamic-reserve requirement

Define the reserve multiplier

\[
\boxed{
\kappa(G,d)=
\frac{P_{\mathrm{head}}^{\min}(G,d)}{|\Delta P(d)|}.
}
\]

The trivial balance baseline is \(\kappa=1\). The study tests whether converter/network dynamics create a measurable excess reserve requirement

\[
\kappa-1>0
\]

and identifies the mechanism when they do.

> ### H2 — outcome: **refuted. \(\kappa-1\) is not measurably positive.**
>
> Evidence: `artifacts/T26_phead_dp1p1/`, `artifacts/T26_phead_dp0p6/`.
>
> \[
> \kappa=1.006\pm0.002
> \]
> over the two disturbance magnitudes bisected. The excess dynamic reserve
> \(P_{\mathrm{excess}}=P_{\mathrm{head}}^{\min}-|\Delta P|\) is 0.0047 MW at
> \(\Delta P=0.6\) MW and 0.0053 MW at \(\Delta P=1.1\) MW — quantities no operator would
> schedule against and no reviewer would accept as a "nontrivial dynamic reserve requirement".
>
> **What this kills.** \(\kappa\) was frozen in v3.1 §25 as *"a primary nontrivial metric"* and
> \(P_{\mathrm{excess}}\) was one of the two preferred quantities in Figure B. Both are
> withdrawn as headline metrics (§28 N4). \(\kappa\) survives only in its measured role: as the
> **numerical statement that the headroom boundary is the feasibility boundary**, which is what
> makes the H1(iv) identification meaningful. A degree of freedom that is measured to equal 1 is
> not a degree of freedom; reporting it as one would be the reserve-sizing equivalent of
> rediscovering power balance.
>
> **A prerequisite defect had to be fixed before this number existed, and the defect is itself a
> result.** `REGF1.Pmax` is declared `non_negative`, so an operating point whose computed active
> ceiling is \(\le 0\) — a bidirectional device that is *required to charge*, a physically valid
> constraint — had its ceiling silently replaced by 1.0 pu. The simulation then converged on a
> system that was not the system requested, and those operating points **reported `SECURE`**.
> This is not a numerical inaccuracy; it is a security verdict inverted toward the unsafe side.
> The patch replaces simulation with an analytical verdict — a non-positive ceiling means the
> fleet cannot leave the charging quadrant, so no post-event equilibrium exists, and `solve`
> returns in 3.4 s instead of 20 s of wrong simulation. Flags `ceiling_representable` and
> `pmax_dev` are written to every run index so the condition is auditable from the artifact
> alone. Scope and re-run accounting: §25, 2026-09-03 block.

## H3 — topology/GFM configuration as a modifier, not a novelty by itself

\[
\boxed{
\text{Feasible feeder configuration and GFM spatial deployment modify }\kappa\text{ and }P_{\mathrm{head}}^{\min}\text{ near converter-constrained boundaries.}
}
\]

The paper does not claim that “GFM placement matters” is new; it quantifies the headroom penalty/benefit under explicit current constraints.

> ### H3 — outcome: **refuted, four times independently, and the refutation has a mechanism**
>
> Evidence: `artifacts/T22_topology_sweep/`, `artifacts/T23_event_location_sweep/` (84 runs),
> `artifacts/T24_imaxf15/`, `artifacts/T25_pdgoff_h0p1/`.
>
> | perturbation | measured shift in the boundary |
> |---|---|
> | tie-switch configuration | **0.0000 MW** dispersion |
> | disturbance location (84 runs) | 0.97% raw, **0.07%** after removing the non-physical part |
> | \(I_{\max F}: 2.00\to1.50\) pu | no change |
> | diesel inertia \(H: 1.0\to0.1\) s | no change; all 13 probe points identical |
>
> Neither feasible feeder reconfiguration nor GFM spatial deployment moves
> \(P_{\mathrm{head}}^{\min}\) or \(\kappa\) on this system. H3 as written is dead.
>
> **The mechanism, which is what makes this publishable rather than a null.** The converter
> interface reactance spans 0.158–0.554 pu while the *entire* electrical spread of the 4.16 kV
> feeder is 0.0386 pu. Define
> \[
> \boxed{\;\Lambda=\frac{X_{\mathrm{converter\ interface}}}{X_{\mathrm{feeder\ spread}}}\;}
> \qquad \Lambda\in[4,14]\ \text{here.}
> \]
> When \(\Lambda\gg1\) the feeder is electrically a single point as seen from the converters,
> and no rearrangement of a single point changes anything. The four nulls are then not four
> separate disappointments but **four measurements of one predicted invariance**, and \(\Lambda\)
> is the criterion that says in advance which systems will behave this way and which will not.
> That inverts the epistemic status of the result: an unexplained null is weak, a *predicted*
> null with a dimensionless criterion is a screening rule other groups can apply.
>
> **The second criterion, from the same family.** Peak deviation current is affine in
> disturbance size, \(I_{\mathrm{dev}}(\Delta P)=0.6247\,\Delta P+0.6151\) pu, giving the
> limiter setting at which current would first become the binding constraint,
> \[
> I_{\max F}^{\mathrm{crit}}=1.3555\ \mathrm{pu},
> \]
> which lies **below the floor of the REGFM_A1 parameter range by a factor of 1.107**. No
> spec-compliant limiter makes current binding on this fleet. This is why T24 returned no
> change, and it is a stronger statement than T24 alone: not "we tried 1.5 pu and nothing
> happened" but "no admissible value can make anything happen, and here is the number".
>
> **Scope limits that must accompany both criteria** — see §28 N2 and N3. \(\Lambda\in[4,14]\)
> and \(I_{\max F}^{\mathrm{crit}}=1.356\) pu are measured on **one** feeder with **one** fleet.
> The criteria are proposed as general and derived from a general argument; they are **not**
> validated across systems, and the manuscript must present them as screening rules with a
> stated derivation, not as empirically generalised constants.
>
> **The failed experiment design, recorded.** The \(H=1.0\to0.1\) s sweep was intended to
> separate *loss of a voltage source* from *loss of inertia*. It cannot: the machine is
> disconnected at \(t_{\mathrm{event}}\), so its rotor leaves the system at the same instant,
> and the pre-event state is steady. `GENROU.M` was verified to differ by 10\(\times\)
> (3.0 vs 0.3) in the assembled model, and all 13 probe points were still identical. The run
> separates nothing and **must not be reported as an inertia sensitivity**. Its one residual
> value is real and should be stated: the diesel-off boundary does not depend on the assumed
> inertia constant, which answers the objection *"your result depends on an arbitrary \(H\)"*
> by measurement rather than argument. Measuring an inertia contribution requires keeping the
> diesel online and applying `gen_loss` elsewhere while sweeping \(H\); that experiment has not
> been run.

## H4 — last-diesel-off transition

\[
\boxed{
\text{The secure removal of the final synchronous diesel unit depends on pre-trip diesel loading, available GFM headroom, and the post-transition network/GFM configuration.}
}
\]

Controlled diesel-off and loaded diesel trip are analyzed separately.

> ### H4 — outcome: **resolved, and opposite in sign to the operational assumption**
>
> Evidence: `artifacts/T25_pdgoff_h1p0/`, `artifacts/T27_pdgoff_gast/`.
>
> \[
> P_{DG,\mathrm{off}}^{\max}=1.2086\ \mathrm{MW}
> \quad>\quad
> \Delta P_{\max}^{\mathrm{gen\_loss}}=1.1851\ \mathrm{MW}
> \qquad(+2.0\%)
> \]
>
> **Tripping the last synchronous machine is *easier* than losing a non-synchronous source of
> the same size.** This is a resolved result, not tolerance noise: the bisection brackets do not
> overlap — `gen_loss` is already insecure at 1.1908 MW while the diesel case is still secure at
> 1.2031 MW.
>
> **The measurement that carries the physics is the coordinate split, not the 2.0%.** At matched
> disturbance size (1.181 vs 1.179 MW):
>
> | coordinate | diesel trip | `gen_loss` | direction |
> |---|---:|---:|---|
> | \(f_{\mathrm{nadir}}\) [Hz] | 59.0256 | 59.0044 | 2.1% **better** |
> | RoCoF [Hz/s] | 1.9556 | 1.9989 | 2.2% **better** |
> | \(V_{\min}\) [pu] | 0.9066 | 0.9196 | 0.0130 pu **worse** |
> | \(\mu_I\) | 0.7991 | 0.6759 | 18.2% **worse** |
>
> The bulk coordinates improve while the local coordinates degrade. Losing a synchronous
> *voltage* source shifts the burden from frequency onto voltage and current, because the GFM
> fleet must absorb the reactive power the machine was supplying. This is the first
> positive-direction confirmation of the \(\Lambda\) mechanism in the whole campaign:
> \(\Lambda\) says the *feeder* is invisible; it does not say a *voltage source* is invisible,
> and this is where the two are distinguished. Every previous \(\Lambda\) result was a null.
>
> **Normative consequence.** The operational assumption that the last-diesel-off transition is
> the binding case is refuted, and the reason is that it is binding in a *different family of
> criteria* than the one usually checked. A study that screens only nadir and RoCoF will rank
> the two events in the wrong order. That is the transferable statement; the 2.0% is not.
>
> **Governor-family cross-check (T27).** The full diesel-off boundary was recomputed with
> `GAST` in place of `TGOV1`:
>
> | quantity | \(\max|\text{TGOV1}-\text{GAST}|\) over 13 probe points |
> |---|---:|
> | \(f_{\mathrm{nadir}}\) | \(1.9\times10^{-10}\) |
> | RoCoF | \(6.2\times10^{-9}\) |
> | \(V_{\min}\) | \(7.5\times10^{-11}\) |
> | \(\mu_I\) | \(1.1\times10^{-9}\) |
>
> Identical to machine precision; the boundary is unchanged at 1.20859 MW. The structural reason
> is the same one that voided the \(H\) sweep: **the governor leaves the system with the
> machine.** Therefore the D2b governor-family risk is *eliminated for the diesel-off boundary*
> — but by the same structural argument, T27 has **no power to test any scenario in which the
> diesel stays online**, and \(\kappa_{os}^{\mathrm{gen\_loss}}=1.2275\) is exactly such a
> scenario. See §28 N7: the leading constant of Contribution I still sits inside the
> un-cross-checked region until the discriminating run (§26 task 1) is done.

# 15. Pre-specified interpretation if the analytical–EMT gap is small

The paper remains valid even if

\[
\Omega_{\mathrm{ana}}\approx\Omega_{\mathrm{EMT}}
\]

over most tested conditions.

### Outcome A — material discrepancy near constraints

If error rises strongly as \(\mu_I\to1\) or \(\mu_P\to1\), report the converter-limited EMT boundary and an applicability threshold.

### Outcome B — small discrepancy over a wide region

If analytical predictions remain accurate at high utilization, report a validated applicability envelope

\[
\mu_I\le\mu_I^\star,
\qquad
\mu_P\le\mu_P^\star
\]

for a predefined error tolerance, together with the computational acceleration achieved by analytical screening.

Thus the study does not depend on “finding” a large model failure.

> **Realised on 2026-09-03: Outcome B, with one refinement.** The measured discrepancy is small
> over the entire energy-feasible domain (\(\le1.44\%\)), so the paper takes the Outcome-B
> branch that was pre-committed here before any of it was run. The refinement is that the
> envelope is **not** specified by thresholds \(\mu_I^\star,\mu_P^\star\) as written above.
> Peak \(\mu_P\) does not separate the regimes: 1.067 and 1.434 both sit inside the accurate
> region. The envelope is specified by *sustained ceiling occupancy at the post-event steady
> state*, which coincides to 0.6% with \(P_{\mathrm{head}}\ge P_{\mathrm{head}}^{\min}\)
> (\(\kappa=1.006\)). Outcome A is not reached: no configuration in the tested domain produced a
> materially optimistic analytical prediction while remaining feasible.
>
> **The second half of Outcome B — "the computational acceleration achieved by analytical
> screening" — is claimable and should be stated, but as a phasor-vs-grid comparison, not as an
> EMT saving.** Bisection on the monotone boundary reached each boundary in ~15 runs against
> thousands for a grid sweep (~18 min against ~10 h). No EMT run has yet been made, so no
> analytical-vs-EMT speed-up may be quoted (§28 N9).

---

# 16. Contributions — v3.2, restated on measurement

> **This section replaces the v3.1 set C1/C2/C3 in full.** The v3.1 contributions were written
> before the campaign and two of them are refuted by it: C2 rested on
> \(\kappa-1>0\), measured at \(0.006\pm0.002\) (§14 H2); C3 rested on a topology-conditioned
> shift in \(P_{\mathrm{head}}^{\min}\), measured at 0.0000 MW (§14 H3). The v3.1 text is kept
> below the new set, struck through in substance, so the change is auditable rather than silent.
>
> **Labels are I/II/III, not C1/C2/C3.** The strings "C2", "C3", "C4" are in use in the
> execution log as *experiment* identifiers and mean different things there. The mapping is
> stated with each contribution.
>
> **Count.** Three, per the target journal's expectation and §16's own v3.1 rule. An earlier
> draft carried four; the closed form and its validity envelope were merged into Contribution I
> because they are one proposition and its qualifying clause. Splitting them would be an
> inflated count and a reviewer will see it.

---

## Contribution I — a closed-form frequency-security boundary, its validity envelope, and the degradation law outside it **(lead)**

*Maps to experiment C4 plus the T26 headroom sweep.*

\[
\boxed{
\Delta P_{\max}=\frac{(f_0-f_{\min})\sum_g S_g}{\kappa_{os}\,f_0\,R}
}
\qquad
\kappa_{os}=
\begin{cases}
1.2275\pm0.24\% & \text{non-synchronous source loss}\\
1.1888\pm1.7\% & \text{synchronous machine trip}
\end{cases}
\]

- accurate to **0.26%** against bisection, over a 36\(\times\) range in \(\Delta P\);
- valid exactly on the energy-feasible set, \(P_{\mathrm{head}}\ge P_{\mathrm{head}}^{\min}\)
  with \(\kappa=P_{\mathrm{head}}^{\min}/|\Delta P|=1.006\pm0.002\) — the two boundaries
  coincide to 0.6%;
- the applicability gate is **sustained ceiling occupancy at the post-event steady state**, not
  peak utilisation: \(\mu_P^{\mathrm{peak}}=1.434\) still yields 1.44% error;
- outside the envelope the error is a monotone degradation law reaching 121.4%, not a cliff;
- in that same region a positive-sequence tool returns `SECURE` for operating points that have
  no post-event equilibrium, unless the active-power ceiling is checked explicitly.

**Why this is a contribution and not a rediscovery of droop.** The steady-state expression is
elementary. Three things are not: that a *single* overshoot constant closes it to four figures
across a 36\(\times\) disturbance range in a six-converter island; that its domain of validity
is the feasibility set rather than a utilisation threshold; and that the failure outside that
domain is graded and quantified rather than assumed catastrophic. Prior model-reduction work
(§23 Group G) establishes *how* converter models can be reduced; none of it expresses
security-assessment error in operator-facing coordinates and bounds it.

**Explicit non-claims** — full statements in §28:

- **not** claimed: that reduced models overestimate security (N1);
- **not** claimed: that \(\kappa_{os}\) is a single universal constant (N8);
- **not** claimed: that the envelope and feasibility boundaries are analytically identical
  rather than coincident to 0.6% (N5);
- **not** claimed: EMT validity of any of it (N9).

---

## Contribution II — two dimensionless screening criteria that decide in advance whether topology, disturbance location and current limiting can matter at all

*Maps to experiment C3.*

\[
\Lambda=\frac{X_{\mathrm{converter\ interface}}}{X_{\mathrm{feeder\ spread}}}
\qquad
I_{\max F}^{\mathrm{crit}}
\ \text{from}\
I_{\mathrm{dev}}(\Delta P)=0.6247\,\Delta P+0.6151
\]

On the study system \(\Lambda\in[4,14]\) and \(I_{\max F}^{\mathrm{crit}}=1.356\) pu, the latter
below the REGFM_A1 parameter floor by 1.107\(\times\). The criteria predict, and four
independent sweeps confirm, that the boundary is invariant to tie configuration (0.0000 MW),
disturbance location (0.07% physical, 84 runs) and limiter setting (no change at 1.5 pu).

**This is the contribution with the widest reach.** \(\Lambda\) and \(I_{\max F}^{\mathrm{crit}}\)
are two inequalities another group can evaluate on their own network in minutes, before
committing to a detailed model. A paper that reports "on our feeder, topology did not matter"
is a null; a paper that reports "topology cannot matter when \(\Lambda\gg1\), here is
\(\Lambda\), and here are four measurements confirming it" is a screening rule.

**Explicit non-claims:** the numerical ranges are single-system measurements and the criteria
are **not** cross-validated on a second feeder (§28 N2, N3). They must be presented as derived
screening rules with one supporting system, not as empirically general constants. The
headroom-weighted accessibility index \(\bar A_{\mathrm{GFM}}\) is **not** part of this
contribution and is not published as an explanatory metric (§28 N6).

---

## Contribution III — the last-diesel-off transition is not the worst case, and the reason is an exchange between two families of security criteria

*Maps to experiment C2.*

\(P_{DG,\mathrm{off}}^{\max}=1.2086\) MW against \(\Delta P_{\max}^{\mathrm{gen\_loss}}=1.1851\)
MW: +2.0%, with non-overlapping brackets. At matched disturbance size the bulk coordinates
improve (\(f_{\mathrm{nadir}}\) +2.1%, RoCoF +2.2%) while the local coordinates degrade
(\(V_{\min}\) −0.0130 pu, \(\mu_I\) +18.2%), because the GFM fleet must absorb the reactive
power the synchronous machine was supplying. The boundary is additionally shown independent of
the assumed inertia constant and of the governor family (`TGOV1` vs `GAST`, agreement to
\(10^{-10}\)).

**The transferable statement is the exchange, not the margin.** 2.0% is small. The result that
survives is normative: an assessment that screens only nadir and RoCoF ranks these two events in
the wrong order, and the criterion family — not the criterion value — decides the ranking.

**Explicit non-claims:** the mechanism of the 3.2% \(\kappa_{os}\) gap is **not** established
(§28 N8); the \(H\) sweep is **not** an inertia sensitivity and is not reported as one (§28 N10);
the governor-family invariance does **not** extend to diesel-online scenarios (§28 N7).

---

## v3.1 contributions — superseded, retained for audit

> The three subsections below are the v3.1 text. **They are superseded in full by I/II/III
> above and must not be copied into the manuscript.** C1 is absorbed into Contribution I with
> its applicability gate corrected. C2 is withdrawn: its central quantity \(\kappa-1\) measures
> \(0.006\pm0.002\). C3 is withdrawn: its central quantity \(\Delta P_{\mathrm{head}}^{\mathrm{req}}(G)\)
> measures 0.0000 MW, and its explanatory metric \(\bar A_{\mathrm{GFM}}\) is separately
> unreproducible (§9, §10).

## C1 — system-level applicability envelope for a reduced analytical frequency model **(lead contribution)** *(superseded)*

Quantify when an established reduced SG/GFM frequency-voltage model is sufficiently accurate for system security screening using operational/converter-utilization indicators:

\[
\hat\mu_I^{\mathrm{ana}},\quad
\mu_I^{\mathrm{EMT}},\quad
\hat\mu_P^{\mathrm{ana}},\quad
\varepsilon_f,\quad
\varepsilon_V.
\]

The final product is either:

- a converter/headroom-limited region where the analytical model becomes materially optimistic; or
- a validated applicability envelope within which reduced-model screening can replace a large fraction of EMT simulations.

This differs from prior device-level model-reduction work by expressing validity in **system-operational coordinates tied directly to security-assessment error**.

## C2 — operational reserve boundary for 100% IBR/diesel-off security *(withdrawn — \(\kappa-1=0.006\pm0.002\))*

Characterize

\[
P_{\mathrm{head}}^{\min}(G,d)
\]

and the reserve multiplier

\[
\kappa(G,d)=P_{\mathrm{head}}^{\min}/|\Delta P|
\]

for post-diesel-off and 100% IBR disturbances, while explicitly representing converter current limits in EMT.

The novelty is not “frequency-constrained operation” by itself; it is the converter-constrained operational boundary and its relation to reduced-model validity.

## C3 — topology-conditioned headroom penalty/benefit **(supporting contribution)** *(withdrawn — measured shift 0.0000 MW)*

Quantify how feasible feeder reconfiguration and GFM spatial deployment shift

\[
\Delta P_{\mathrm{head}}^{\mathrm{req}}(G)
\]

and \(\kappa\) after controlling for aggregate capacity.

This contribution must **not** be framed as discovering that GFM placement affects stability. The publishable result is the size of the current-constrained dynamic-reserve shift and the explanatory value of the headroom-weighted accessibility metric.

# 17. Event taxonomy

## A. Transition event

1. controlled last-diesel-off;
2. loaded last-diesel trip.

## B. Post-transition power-imbalance events

- load increase;
- sustained wind reduction;
- PV reduction;
- GFL source trip;
- other credible generation/load imbalance.

## C. Network/device events

- critical feeder outage;
- selected GFM trip if physically meaningful;
- disturbance-induced current-limiter activation;
- motor start if a defensible motor model is available.

Intentional feeder **configuration states** remain separate from N-1 outage contingencies.

---

# 18. Experimental axes

### Axis A — GFM deployment

At fixed \(G_0\), compare 2→5→6 GFM.

### Axis B — feeder configuration

At fixed six-GFM deployment, compare \(G_0,\ldots,G_{46}\).

Define

\[
\Delta P_{\mathrm{head}}^{\mathrm{req}}(G)
=
P_{\mathrm{head}}^{\min}(G)-P_{\mathrm{head}}^{\min}(G_0).
\]

### Axis C — disturbance severity

Sweep \(\Delta P\) or event magnitude.

### Axis D — diesel-off transition severity

Sweep

\[
P_{\mathrm{DG}}(t_{\mathrm{off}}^-).
\]

---

# 19. Computational budget — benchmark first, then commit

No fixed “2 min/run” assumption is allowed in the frozen design.

## 19.1 Stage 0 runtime benchmark

Before the campaign, measure wall-clock time on the actual hardware/software using at least 10–20 representative runs for:

1. single GFM device EMT;
2. small multi-GFM network;
3. modified IEEE 123 matched-assumption EMT;
4. modified IEEE 123 native three-phase unbalanced EMT;
5. one mixed diesel-SG + GFM transition case.

Record:

\[
t_{50},\quad t_{95},\quad \text{simulation horizon},\quad \Delta t,\quad \text{solver},\quad \text{CPU/cores}.
\]

Campaign time is then estimated from **measured median/p95 runtime**, not literature extrapolation.

## 19.2 Stage C — direct EMT boundary study

A representative design remains approximately

\[
6\ \text{configurations}
\times
8\ \text{events}
\times
20\ \text{operating points}
\approx960\ \text{runs},
\]

but the actual wall-clock budget is reported only after Stage 0.

The mixed-SG diesel-off cases must be included in this budget.

## 19.3 Stage D — no brute-force dense EMT grid

A naive

\[
47\times8\times100=37{,}600
\]

run campaign is not the default plan regardless of per-run time.

Use:

1. analytical screening over the full space;
2. adaptive/boundary-focused EMT sampling;
3. parallel execution where available;
4. interpolation/error monitoring;
5. only if necessary, a lightweight data-driven surrogate.

## 19.4 Surrogate decision rule

A surrogate is introduced only if **both** are true:

- measured EMT runtime makes the required boundary density impractical; and
- analytical screening + adaptive EMT sampling cannot reconstruct the desired boundary within the specified tolerance.

Thus GNN is a computational contingency, not a predetermined method.

# 20. AI/GNN decision

GNN is **not required** in the first paper. The topology-dependent analytical model already provides a physics-based screening layer through \(B_r(G)\) and \(B_L(G)\).

A learned surrogate is added only if the EMT-verified boundary cannot be reconstructed adequately from analytical screening plus adaptive sampling.

If needed:

\[
(G,x,d)
\rightarrow
\{\hat f_{\mathrm{nadir}},\widehat I_{\mathrm{peak}},\hat V_{\min},P(\mathrm{secure})\}.
\]

MAPPO remains outside this paper.

---

# 21. Key figures — revised to avoid trivial geometry

## Figure A — research architecture

\[
\text{hybrid SG/GFM state}
\rightarrow
\text{last diesel off}
\rightarrow
\text{100% IBR}
\rightarrow
\text{analytical screening}
\rightarrow
\text{applicability gate}
\rightarrow
\text{EMT verification}.
\]

## Figure B — reserve multiplier, not only raw headroom

A raw map of

\[
P_{\mathrm{head}}^{\min}\ \text{vs.}\ \Delta P
\]

must include the trivial diagonal

\[
P_{\mathrm{head}}=|\Delta P|.
\]

The preferred comparison is

\[
\boxed{
\kappa(\Delta P)=
\frac{P_{\mathrm{head}}^{\min}}{|\Delta P|}
}
\]

or the excess dynamic reserve

\[
\boxed{
P_{\mathrm{excess}}
=P_{\mathrm{head}}^{\min}-|\Delta P|.
}
\]

These quantities isolate the nontrivial penalty caused by network location, converter limits, and dynamic behavior.

## Figure C — analytical current-estimator validation

\[
\hat\mu_I^{\mathrm{ana}}
\quad\text{vs.}\quad
\mu_I^{\mathrm{EMT}}.
\]

This figure must precede any use of analytical current utilization as an applicability gate.

## Figure D — reduced-model applicability

\[
\mu_I^{\mathrm{EMT}}
\quad\text{and/or}\quad
\mu_P
\quad\text{vs.}\quad
|f_{\mathrm{nadir}}^{\mathrm{ana}}-f_{\mathrm{nadir}}^{\mathrm{EMT}}|.
\]

## Figure E — configuration-conditioned dynamic reserve

\[
G_k\quad\text{vs.}\quad \kappa(G_k)
\]

or

\[
G_k\quad\text{vs.}\quad \Delta P_{\mathrm{head}}^{\mathrm{req}}(G_k).
\]

## Figure F — last-diesel-off transition

Waveforms of

\[
f_i,\quad P_{\mathrm{GFM}},\quad I_{\mathrm{GFM}},\quad V_i,\quad P_{\mathrm{DG}}
\]

for controlled and loaded diesel-off events.

## Figure G — signature figure **(new, 2026-09-03; replaces Figure B as the lead)**

One panel carrying all of Contribution I.

- **abscissa:** \(P_{\mathrm{head}}/|\Delta P|\), logarithmic;
- **ordinate:** \(f_{\mathrm{nadir}}\);
- **horizontal line:** the closed-form prediction \(f_0-\kappa_{os}\Delta f_{ss}\);
- **markers:** the measured sweep, tracking the line above \(\kappa=1.006\) and diverging below
  it, with marker shape switching at the 0 → 6 sustained-saturation transition;
- **vertical line with uncertainty band:** \(\kappa=1.006\pm0.002\);
- **shaded region to the left:** energy-infeasible;
- **annotation on that region:** *pre-guard: reported* `SECURE`.

Rationale: the figure shows in one view that the closed form is exact, where its envelope lies,
how it fails outside, and that the simulator's own verdict was wrong in the same region. No
other single figure in the campaign carries three results at once.

## Status of Figures B, C and E — 2026-09-03

- **Figure B** (\(\kappa\) or \(P_{\mathrm{excess}}\) vs \(\Delta P\)) is **demoted**. Its whole
  purpose was to display a nontrivial \(\kappa-1\); measured, \(\kappa-1=0.006\pm0.002\).
  It survives only as a small inset establishing \(\kappa\approx1\), which is the premise of
  Figure G's abscissa, not a result in its own right.
- **Figure C** (\(\hat\mu_I^{\mathrm{ana}}\) vs \(\mu_I^{\mathrm{EMT}}\)) is **not yet
  drawable** — no EMT run exists. The analytical current estimator was measured against the
  phasor layer and is short by 4.6–6.2\(\times\), with 96.9% of one unit's current reactive;
  that comparison is `ana` vs `andes`, not `ana` vs `emt`, and the axis labels must say so.
- **Figure E** (\(G_k\) vs \(\kappa(G_k)\)) plots a flat line at 0.0000 MW dispersion.
  It is **not published as a figure**; the invariance is one sentence and one number in text,
  and the ink belongs to \(\Lambda\) instead.

# 22. Recommended manuscript storyline

## I. Introduction

1. diesel-free operation is becoming practically relevant in renewable-rich islands;
2. frequency-constrained scheduling already addresses reserve/frequency coupling;
3. 100% IBR EMT dynamic-security studies already exist;
4. GFM current limiting and inverter security regions are established;
5. missing operational link: **when a closed-form frequency-security boundary is valid for a
   diesel-free island, what decides in advance whether topology and current limiting can move
   it, and how the boundary depends on the class of the triggering event**;
6. contributions I–III (§16).

> **Revised 2026-09-03.** Item 5 previously asserted that converter limits, GFM deployment and
> feeder configuration *do* shape the boundary. Two of those three were measured not to
> (§28 N2, N3), so the introduction cannot open on that premise. The revised item is the
> question the campaign actually answered. The introduction must also state the four refuted
> hypotheses — they were written before the campaign and they are the paper's evidence that the
> surviving result was not selected after the fact.

## II. System and Transition Problem

- modified IEEE 123 testbed;
- Bach Long Vy as motivation;
- diesel + IBR pre-transition state;
- 100% IBR post-transition state;
- droop GFM/GFL roles;
- configuration set.

## III. Analytical Screening Model

- Trujillo-type frequency/voltage model;
- topology matrices;
- assumptions;
- applicability indicators \(\mu_I,\mu_P\).

## IV. Converter-Level EMT Model

- BESS/DC side;
- multiloop droop GFM;
- VSC/LCL;
- current limiter;
- anti-windup;
- validation.

## V. Security-Region Method

- \(\Omega_E,\Omega_{\mathrm{dyn}},\Omega_{\mathrm{op}}\);
- last-diesel-off event;
- \(P_{\mathrm{head}}^{\min}\);
- analytical/EMT boundary comparison;
- adaptive sampling.

## VI. Results

1. device-model validation;
2. last-diesel-off transition;
3. analytical vs. EMT validity;
4. GFM placement;
5. feeder reconfiguration;
6. computational performance.

## VII. Discussion

- what limits diesel-free operation;
- what the analytical model can/cannot screen;
- RoCoF interpretation;
- implications for scheduling/EMS;
- generalizability beyond the IEEE 123 feeder.

## VIII. Conclusion

---

# 23. Literature map — novelty boundary expanded

**Citation counts below are those reported in the 2026-09-02 novelty audit and should be refreshed before submission.**

## Group A — nearest 100% IBR dynamic-security competitor

[1] **Nan Xue et al.**, “Dynamic Security Optimization for N-1 Secure Operation of Hawai‘i Island System With 100% Inverter-Based Resources,” *IEEE Transactions on Smart Grid*, 2022. DOI: **10.1109/TSG.2021.3135232**.

Novelty boundary: high-fidelity EMT + 100% IBR + DSA/optimization already exists.

## Group B — nearest operational headroom / secure-transition competitors

[2] [**Frequency-Dynamics-Aware Economic Dispatch with Optimal Grid-Forming Inverter Allocation and Reserved Power Headroom**](https://consensus.app/papers/details/d93c6154b0d45b7b9155ccd7d59add2b/?utm_source=claude_desktop) — Fan Jiang et al., 2025, arXiv, 2 citations. DOI: **10.48550/arxiv.2512.01814**.

[3] [**Frequency Constrained Proactive Scheduling for Secure Microgrid Formation in Wind Power Penetrated Distribution Systems**](https://consensus.app/papers/details/f66d23921f6c5782b0c9796161221082/?utm_source=claude_desktop) — Sheng Cai et al., 2025, *IEEE Transactions on Smart Grid*, 30 citations. DOI: **10.1109/TSG.2024.3524557**.

[4] [**Resilient Microgrid Scheduling With Secure Frequency and Voltage Transient Response**](https://consensus.app/papers/details/67bcd2ef28905fbab6fb97e3bcb487e7/?utm_source=claude_desktop) — A. Nakiganda et al., 2023, *IEEE Transactions on Power Systems*, 14 citations. DOI: **10.1109/TPWRS.2022.3207523**.

Novelty boundary: required headroom, secure microgrid formation, and transient-security-aware scheduling are not empty research spaces.

## Group C — frequency-constrained scheduling/planning anchors

[5] **M. Javadi, Yuzhong Gong, C. Chung**, “Frequency Stability Constrained Microgrid Scheduling Considering Seamless Islanding,” *IEEE Transactions on Power Systems*, 2021. DOI: **10.1109/TPWRS.2021.3086844**.

[6] **M. Javadi, Yuzhong Gong, C. Y. Chung**, “Frequency Stability Constrained BESS Sizing Model for Microgrids,” *IEEE Transactions on Power Systems*, 2024. DOI: **10.1109/TPWRS.2023.3284854**.

[7] **Xuebo Liu et al.**, “Frequency Nadir Constrained Unit Commitment for High Renewable Penetration Island Power Systems,” *IEEE Open Access Journal of Power and Energy*, 2024. DOI: **10.1109/OAJPE.2024.3370504**.

[8] **Zhongda Chu, Ning Zhang, Fei Teng**, “Frequency-Constrained Resilient Scheduling of Microgrid: A Distributionally Robust Approach,” *IEEE Transactions on Smart Grid*, 2021. DOI: **10.1109/TSG.2021.3095363**.

## Group D — GFM current limiting

[9] **Nathan Baeckeland et al.**, “Overcurrent Limiting in Grid-Forming Inverters: A Comprehensive Review and Discussion,” *IEEE Transactions on Power Electronics*, 2024. DOI: **10.1109/TPEL.2024.3430316**.

[10] [**Transient Stability Analysis and Enhancement of Inverter-Based Microgrid Considering Current Limitation**](https://consensus.app/papers/details/0d6e7404931c52e18da68476f4e270eb/?utm_source=claude_desktop) — Dashuang Li, Qiuye Sun, Rui Wang, Zhengqi Sui, 2025, *IEEE Transactions on Power Electronics*, 18 citations. DOI: **10.1109/TPEL.2024.3467215**.

[11] [**Current limiting algorithms and transient stability analysis of grid-forming VSCs**](https://consensus.app/papers/details/4a7ed6e9649e5107971283c35537aaaa/?utm_source=claude_desktop) — T. Qoria et al., 2020, *Electric Power Systems Research*, 208 citations. DOI: **10.1016/j.epsr.2020.106726**.

## Group E — inverter/security-region literature

[12] **Jiazuo Hou et al.**, “Security region of inverter-interfaced power systems: Existence, expansion, and application,” *Renewable and Sustainable Energy Reviews*, 2024. DOI: **10.1016/j.rser.2023.114222**.

[13] **Mazhar Ali, Airin Rahman, Wei Sun**, “Power System Resilience Enhancement Through Algebraic Characterization of Steady-State Security Regions with Embedded Inverter Protection Limits,” *IEEE TPEC*, 2026. DOI: **10.1109/TPEC67884.2026.11513033**.

[14] **Rui Ma et al.**, “Generalized Dynamic Security Region of Grid-Following and Grid-Forming Converter-Based Systems by Basin of Attraction Method,” *Applied Sciences*, 2026. DOI: **10.3390/app16042130**.

## Group F — GFM placement literature: mature field, supporting only

[15] [**Placement and Implementation of Grid-Forming and Grid-Following Virtual Inertia and Fast Frequency Response**](https://consensus.app/papers/details/aa44318c69a25e649427f1538aae1173/?utm_source=claude_desktop) — B. Poolla et al., IEEE TPWRS, 353 citations in the audit. DOI: **10.1109/TPWRS.2019.2892290**.

[16] [**Placing Grid-Forming Converters to Enhance Small Signal Stability of PLL-Integrated Power Systems**](https://consensus.app/papers/details/e3b61ac4d9bc52b7b24610f935964a8b/?utm_source=claude_desktop) — Chaoran Yang et al., 2020, *IEEE Transactions on Power Systems*, 219 citations. DOI: **10.1109/TPWRS.2020.3042741**.

[17] [**Strategic Placement of Grid-Forming Inverters Considering Spatiotemporal Dynamics and Composite Stability Index**](https://consensus.app/papers/details/68d90dabb48d5dc7bc8b4e7871df6423/?utm_source=claude_desktop) — Chalitha Liyanage et al., 2025, *IEEE Open Journal of the Industrial Electronics Society*. DOI: **10.1109/OJIES.2025.3538480**.

[18] [**Location of grid forming converters when dealing with multi-class stability problems**](https://consensus.app/papers/details/aec34ae4591d55f698a91d508110f178/?utm_source=claude_desktop) — Francisco Fernandes et al., 2025, *IET Generation, Transmission & Distribution*. DOI: **10.1049/GTD2.13312**.

[19] [**Fundamental-Frequency Bus-Impedance Analysis of Power Grids Dominated by Grid-Forming and Grid-Following Inverters**](https://consensus.app/papers/details/fff8a6bdbd9b57279e60aa287b8862e7/?utm_source=claude_desktop) — Jialu Yuan et al., 2025, *IET Generation, Transmission & Distribution*. DOI: **10.1049/GTD2.70110**.

[20] [**The Impacts of Grid-Forming Inverter Placement on the Small-Signal Stability of Distribution Networks**](https://consensus.app/papers/details/d0bc0f35b5f65f1dace563e7d616c96b/?utm_source=claude_desktop) — Shiwang Gu et al., 2025. DOI: **10.1109/SPET67577.2025.11335949**.

[21] [**Grid-Strength-Aware Placement of Distributed IBRs for Frequency Stability in Integrated T&D Networks**](https://consensus.app/papers/details/8d9049958df751d58c804577d204ceb6/?utm_source=claude_desktop) — Mariam Islam et al., 2026, GPECOM. DOI: **10.1109/GPECOM70462.2026.11578654**.

Novelty consequence: placement is an explanatory/experimental factor. The headroom weighting \(\pi_g\) in \(\bar A_{\mathrm{GFM}}\) is the specific distinction from topology-only or installed-capacity-only placement indices.

## Group G — reduced-model validity / model-order reduction: lead novelty boundary

[22] [**Model Reduction for Inverters With Current Limiting and Dispatchable Virtual Oscillator Control**](https://consensus.app/papers/details/65802e897d2e5d68a36064490b253ea4/?utm_source=claude_desktop) — O. Ajala et al., 2021, *IEEE Transactions on Energy Conversion*, 53 citations. DOI: **10.1109/TEC.2021.3083488**.

[23] [**Impact of Inner Control Loops on Small-Signal Stability and Model-Order Reduction of Grid-Forming Converters**](https://consensus.app/papers/details/f00fc2b56c985fc786f0f9f372096ada/?utm_source=claude_desktop) — S. Eberlein et al., 2023, *IEEE Transactions on Smart Grid*, 29 citations. DOI: **10.1109/TSG.2022.3220723**.

[24] [**Model Order Reduction of Voltage Source Converters Based on the Ac Side Admittance Assessment: From EMT to RMS**](https://consensus.app/papers/details/e70d54fea32c5f9c981f66f713fa752a/?utm_source=claude_desktop) — Goran Grdenić et al., 2023, *IEEE Transactions on Power Delivery*, 8 citations. DOI: **10.1109/TPWRD.2022.3179836**.

[25] [**Reduced-Order Models for Representing Converters in Power System Studies**](https://consensus.app/papers/details/437ad7e65a0a5b7aa9c18e48c3c6690e/?utm_source=claude_desktop) — Yunjie Gu et al., 2018, *IEEE Transactions on Power Electronics*, 148 citations. DOI: **10.1109/TPEL.2017.2711267**.

[26] [**Modeling of Grid-Forming and Grid-Following Inverters for Dynamic Simulation of Large-Scale Distribution Systems**](https://consensus.app/papers/details/8411bff14a15573e9f750f5c19b08ccc/?utm_source=claude_desktop) — W. Du et al., 2021, *IEEE Transactions on Power Delivery*, 288 citations. DOI: **10.1109/TPWRD.2020.3018647**.

**Lead novelty distinction:** these works address how converter models can be reduced/selected. Our paper should ask how **security-assessment error at system scale** maps to measurable utilization coordinates and whether those coordinates can define an operational applicability gate for fast screening.

## Group H — IEEE 123 islanded-microgrid precedent

[27] [**Under-Frequency Load Shedding for Power Reserve Management in Islanded Microgrids**](https://consensus.app/papers/details/bc63dabc15d85d148118f351003ac8af/?utm_source=claude_desktop) — Bei Xu et al., 2023/2024 record in the audit, *IEEE Transactions on Smart Grid*, 13 citations. DOI: **10.1109/TSG.2024.3393426**.

Role: precedent for using a modified IEEE 123 feeder as an islanded microgrid with a limited number of grid-forming resources and real-time/OPAL-RT-style dynamic studies.

# 24. Modeling anchors

**M1.** Marena Trujillo, Amir Sajadi, Jonathan Shaw, Bri-Mathias Hodge, “Analytical Models of Frequency and Voltage in Large-Scale All-Inverter Power Systems,” *IEEE Transactions on Power Systems*, 2026. DOI: **10.1109/TPWRS.2025.3650698**.

**M2.** Marena Trujillo, Amir Sajadi, Jonathan Shaw, Bri-Mathias Hodge, “Computationally Efficient Analytical Models of Frequency and Voltage in Low-Inertia Systems,” accepted in *IEEE Transactions on Power Delivery*, 2026. DOI: **10.1109/TPWRD.2026.3722456**.

**M3.** Rick Wallace Kenyon et al., “Interactive Power to Frequency Dynamics Between Grid-Forming Inverters and Synchronous Generators in Power Electronics-Dominated Power Systems,” *IEEE Systems Journal*, 2023. DOI: **10.1109/JSYST.2023.3257284**.

**M4.** Mohsen Eskandari, Andrey V. Savkin, “On the Impact of Fault Ride-Through on Transient Stability of Autonomous Microgrids: Nonlinear Analysis and Solution,” *IEEE Transactions on Smart Grid*, 2021. DOI: **10.1109/TSG.2020.3030015**.

**M5.** Liansong Xiong, Xiaokang Liu, Yonghui Liu, Fang Zhuo, “Modeling and Stability Issues of Voltage-Source Converter-Dominated Power Systems: A Review,” *CSEE Journal of Power and Energy Systems*, 2022. DOI: **10.17775/CSEEJPES.2020.03590**.

---

# 25. What is frozen vs. open

## Frozen

- main topic: system-level validity limits of reduced frequency/security models in 100% IBR islanded operation;
- primary testbed: modified IEEE 123;
- Bach Long Vy: motivation/reference only;
- multiloop droop GFM only in the first paper;
- 47 feasible configuration phase retained;
- 2/5/6 GFM deployment retained as an experimental axis, not a novelty claim;
- controlled and loaded last-diesel-off events retained;
- hybrid SG/GFM analytical model required before diesel-off;
- synchronous diesel machine + governor + AVR required in EMT;
- analytical current estimate must be validated against EMT before it is used as a gate;
- analytical model used only within stated assumptions;
- EMT is reference near converter constraints;
- RoCoF separated into physical-security and compliance roles;
- ~~reserve multiplier \(\kappa=P_{\mathrm{head}}^{\min}/|\Delta P|\) is a primary nontrivial metric~~ — **unfrozen 2026-09-03**, measured \(\kappa=1.006\pm0.002\); \(\kappa\) is retained only as the numerical statement that the headroom boundary is the feasibility boundary (§14 H2);
- ~~C1 = model applicability; C2 = operational reserve boundary; C3 = supporting topology-conditioned reserve result~~ — **replaced 2026-09-03** by Contributions I/II/III of §16;
- no MAPPO;
- no mandatory GNN.

## Closed on 2026-09-02 (plan v3.1 Part 0; evidence in `artifacts/T01_agfm/`, `artifacts/T02_soc/`)

**D1 — EMT platform.** MATLAB R2025a + **Simscape Electrical**, using the modern
`Simscape/Electrical/Control/SM Control` blocks. The legacy Specialized Power Systems
library is *not* used: MathWorks removes it in R2026a, and `spsConversionAssistant`
degrades converted blocks to plain subsystems. Simulink, Simscape and Simscape Electrical
licences verified present on the study machine; PSCAD and PLECS are not installed.

**D1b — VSC fidelity.** Switching-averaged VSC model for the campaign; full switching
reserved for 2–3 sanity cases only.

**D2b — diesel prime mover and excitation.**

| Element | Selected model | Basis |
|---|---|---|
| Machine | Simscape Electrical synchronous machine, standard parameters | — |
| Governor | **GGOV1** configured for diesel (`Flag=1`, `Teng>0`, `Dm>0`, `Rselect` for isochronous) | IEEE PES WG 2013 turbine-governor report; validated against a 400 kVA CAT C-15 genset in a remote islanded microgrid by Shah et al., *IEEE Access* 10:110537, 2022, where it tracks nadir and rebound markedly better than DEGOV |
| Governor cross-check | **DEGOV1** (Woodward), run as a sensitivity on the loaded-trip case (T11) only | traditional diesel governor; present in PSS®E, PowerWorld, OpenIPSL, HYPERSIM libraries |
| AVR / exciter | **SM DC4C** block set to DC4B | same validated genset model as above; block conforms to IEEE Std 421.5-2016 |
| V/Hz limiter | **out of scope**, declared as a deliberate simplification | real remote-microgrid gensets carry one; omitting it is defensible for the frequency-security question but must be stated |

Neither GGOV1 nor DEGOV1 exists as a shipped MATLAB block; both are built from the
published block diagrams. This does **not** breach N-4, which forbids self-invented models,
not standard-library models rebuilt from their published definition.

**SoC classification.** SoC belongs to the slow set \(\Omega_E\). It has no channel into the
unconstrained frequency dynamics — \(\Delta f_{ss}=f_0 R\,\Delta P/\sum_g S_g\) contains no
\(P_{\mathrm{head}}\) — and reaches security only by moving the saturation boundary, which
for this fleet only begins below SoC ≈ 0.225. Under a uniform SoC, \(\bar A_{\mathrm{GFM}}\)
is exactly invariant (all six units share E/P = 2 h); a single-unit SoC spread moves it by at
most 5.1% on the v3 fleet and 12.1% after the D6 rescale (`artifacts/T02_soc_v4/`), the
growth being an artefact of the epsilon convention rather than a new SoC effect. The
*numbers* attached to the saturation boundary are superseded by that rescale --
$\Delta P_{\mathrm{critical}}$ falls from 17.25 MW to 3.272 MW and $\kappa$ crosses 1 at
$\Delta P = 3.414$ MW, inside the feeder's credible disturbance range -- but the
classification itself is unchanged.
To be reconfirmed in EMT at P1: the assumption that low-SoC derating changes only
\(P_{\max}\), not droop gain or inner-loop bandwidth.

## Closed on 2026-09-03 (evidence in `artifacts/T00_rescale/`, `artifacts/T01_agfm_v4/`, `artifacts/T02_soc_v4/`, `artifacts/T20_andes_bisect/`)

**D5 — a phasor-domain workhorse alongside EMT.** ANDES 2.0 (positive-sequence,
trapezoidal, `REGF1` droop grid-forming converters, `GENROU` + `TGOV1` + `SEXS` for the
diesel) carries the *search* for the boundary; EMT under D1 remains the ground truth and
the only thing a claim is stated on. The split is what makes the campaign affordable: one
ANDES run of an 8 s horizon on the 123-bus feeder costs about 12 s of wall clock, so a
bisection buys a boundary in roughly 15 runs instead of a grid sweep of thousands.

The two platforms disagree by construction and the artifact keeps them apart: the
`*_emt` columns of `metrics.csv` are left empty on ANDES rows, ANDES results live in
`*_andes` columns, and every row carries `platform`.

**D6 — GFM fleet rescaled to 1.25x peak load.** 23 MVA of converters on a 3.49 MW feeder
made the headroom-scarce region physically unreachable; every boundary metric saturated and
the campaign would have returned "secure everywhere", which is a non-result rather than a
null result. Uniform scaling by 0.18967 to 4.362 MVA preserves \(\pi_g\), each unit's E/P
and each unit's MVA/MW. Wind, DPV and EVCS are **not** rescaled and still total ~19 MW on
the same feeder; they set the campaign's \(\Delta P\) magnitudes and need a separate
decision.

**Four properties of the platform, measured rather than assumed.** Each one changes results
by more than the effects the paper reports, so each is recorded here rather than buried in
code:

1. *`REGF1`'s shipped inner-loop gains make this fleet linearly unstable.* Six units on a
   4.16 kV feeder give ten eigenvalues with \(\mathrm{Re}\,\lambda>0\) at 92–260 Hz and
   \(\zeta \approx -0.03\ldots-0.05\); every time-domain run then collapses about 0.25 s
   after any event, at any step size and any disturbance magnitude down to 10 kW. The
   instability is in the *current* loop. Holding \(K_{Pv},K_{Iv}\) at their defaults and
   slowing \(K_{Pi}: 0.5\to0.2\), \(K_{Ii}: 20\to5\) restores \(\mathrm{Re}\,\lambda\le0\)
   for 2/5/6 GFM, every headroom, \(x_f\in[0.05,0.20]\), \(R\in[0.03,0.05]\) and both load
   models. Modes above ~100 Hz are outside a positive-sequence model's validity in any
   case. **These are not the physical gains** — T3 sets those from the EMT device model.
2. *Reactive coupling is the ill-conditioned coordinate, not active power.* With the six
   converters sitting directly on the feeder, measured \(\mathrm{d}Q/\mathrm{d}V\) is
   \(3.7\times10^{4}\) pu/pu: a \(10^{-3}\) pu difference in voltage setpoint swings one
   unit's reactive output by 37 pu of its own rating, and the smallest unit sat at
   **1.82 pu current before any disturbance**. This is the same mechanism reported for the
   100% grid-forming case in the literature, and it says \(V\) and \(Q\) belong inside
   \(\Omega_{\mathrm{dyn}}\) from the start. Modelling each converter behind an explicit
   step-up branch drops the sensitivity to ~1.3e3 and brings the base case inside rating.
   The residual reactive split is still a *dispatch decision that has not been made*, and
   no \(\mu_I\) result should be quoted until it is.
3. *`REGF1`'s droop is referred to the system base.* Passing \(R\) straight through gives
   every unit the same MW/Hz gain irrespective of rating — a 0.38 MVA unit picks up as many
   MW as a 1.33 MVA one. Referring \(R\) to each unit's own rating restores proportional
   sharing and makes the aggregate exactly \(f_0R\,\Delta P/\sum_g S_g\), the §14
   expression.
4. *`REGF1`'s P-limiter integrator silently stiffens the droop by \((1+K_{Iplim}T_{pm})\).*
   Measured across \(K_{Iplim}=30/10/4/1/0\), the realised droop came out
   \(0.0875/0.0625/0.0550/0.0513/0.0500\) against \(R=0.05\) — the formula to four digits,
   a 75% error at the shipped value. Setting \(K_{Iplim}=0\) removes it; the headroom
   ceiling still binds, at \(\max(P_e/P_{\max})=1.0001\).

**Cross-validation of §14 against a dynamic model.** With (3) and (4) corrected, the ANDES
steady-state frequency deviation reproduces the analytical
\(\Delta f_{ss}=f_0R\,\Delta P/\sum_g S_g\) to better than 1% in the unconstrained regime.
That is the first independent check the analytical layer has had.

## Closed on 2026-09-03 — boundary campaign T20–T28

Evidence: `artifacts/T20_andes_bisect*/`, `T20_genloss_constP/`, `T20_genloss_constZ/`,
`T22_topology_sweep/`, `T22_phead_at_dp1p1/`, `T23_event_location_sweep/`, `T24_imaxf15/`,
`T25_pdgoff_h1p0/`, `T25_pdgoff_h0p1/`, `T26_phead_dp1p1/`, `T26_phead_dp0p6/`,
`T27_pdgoff_gast/`, `T28_dpmax_guarded/`.

**Headline numbers.** All positive-sequence; none EMT-verified.

| quantity | value | source |
|---|---:|---|
| \(\Delta P_{\max}^{\mathrm{gen\_loss}}\) | 1.1850585938 MW | T21, reproduced bit-exact in T28 |
| \(P_{DG,\mathrm{off}}^{\max}\) | 1.2086 MW | T25 |
| \(\kappa_{os}\), non-synchronous | 1.2275 (\(\sigma\) 0.24%) | T20/T21, 23 points |
| \(\kappa_{os}\), synchronous trip | 1.1888 (\(\sigma\) 1.7%) | T25 |
| \(\kappa=P_{\mathrm{head}}^{\min}/|\Delta P|\) | 1.006 ± 0.002 | T26 |
| \(\Lambda\) | 4–14 (0.158–0.554 pu vs 0.0386 pu) | network data |
| \(I_{\max F}^{\mathrm{crit}}\) | 1.3555 pu | T24 |
| closed-form error on \(\Delta P_{\max}\) | 0.26% | T21 vs §14 H1 |
| binding coordinates at the boundary | RoCoF + nadir at the corner; \(\mu_I=0.809\), \(\mu_P=0.364\) | T21 |

**A defect that inverted a security verdict, and its guard.** `REGF1.Pmax` is declared
`non_negative`; an operating point requiring a non-positive active ceiling — a bidirectional
device constrained to charge — had that ceiling silently replaced by 1.0 pu, and the resulting
run **reported `SECURE`**. The condition is now detected analytically before any integration:
a non-positive ceiling means no post-event equilibrium exists, and `solve` returns a verdict in
3.4 s instead of 20 s of simulating a different system. Patch sites: `build_case.build_system`
and `metrics.extract`; flags `ceiling_representable` and `pmax_dev` are written to every run
index, so any future artifact can be audited for the condition without re-reading the code.

This is a third instance of the same class already recorded in this section (silent parameter
correction, `pv2pq`/`adjust_upper` widening \(Q_{\max}\), and the \(K_{I,\mathrm{plim}}\) droop
stiffening). The pattern — a positive-sequence library silently substituting an admissible
parameter for the requested one — is the standing reason this project verifies platform
behaviour by measurement rather than by reading documentation.

**Re-run accounting after the patch.** The defect can only be reached by the combination
`gen_loss` **and** a \(p_{\mathrm{head}}\) sweep below \(-p_0\). Exactly one campaign item meets
both conditions, and it was run after the patch.

| artifact | \(p_{\mathrm{head}}\) | guard fired? | re-run needed? |
|---|---|---|---|
| T21 \(\Delta P_{\max}\) | `None` | no | no — confirmed by T28 |
| T22 tie sweep | `None` | no | no |
| T23 disturbance-location sweep | `None` | no | no |
| T24 \(I_{\max F}=1.5\) | `None` | no | no |
| T25 / T27 diesel-off | `None` | no | no |
| T26 \(P_{\mathrm{head}}^{\min}\) | swept | **yes** | already run post-patch |

**T28 — what it does and does not establish.** With the guard active, \(\Delta P_{\max}\)
reproduces to \(0.00\mathrm{e}{+}00\); the brackets \([1.1793,\,1.1908]\) and the 14 probe runs
are identical; all nine metrics (\(f_{\mathrm{nadir}}\), RoCoF window, \(V_{\min}\),
\(V_{\max}\), \(\mu_I\), continuous \(\mu_I\), \(\mu_P\), \(\Delta f_{ss}\), \(f_{ss}\)) agree
to \(0.000\mathrm{e}{+}00\) across all probe points; `secure_flag`, `n_units_saturated` and
`tds_converged` match exactly; and the log contains **zero** occurrences of
`non_negative param <Pmax> corrected to 1.0`. That last count is the load-bearing evidence: in
the \(\Delta P_{\max}\) sweep \(p_{\mathrm{head}}\) is `None`, so
\(P_{\max}=(-0.575+3.414)/4.3624=0.651>0\) at every probe point and the platform never had
anything to correct.

**Stated limit of T28.** It shows the guard changed nothing where nothing needed changing. It is
**not** an independent verification of \(\Delta P_{\max}\) itself. The evidence for
\(\Delta P_{\max}\) remains the four invariance measurements (T22, T23, T24, T25b) plus the
0.26% closed-form agreement.

## Open but explicitly scheduled for resolution

- ~~**T29 — governor family on the `gen_loss` boundary.**~~ — **run 2026-09-06.** The families
  differ by 0.842% (6.4 kW) with the diesel online, against T27's \(10^{-10}\) with it off, so
  the structural-artefact reading of T27 is now measured. Pre-registered verdict
  **inconclusive** (0.842% sits between the 0.5% and 2% thresholds). \(\kappa_{os}\) for the
  `gen_loss` class therefore carries a stated 0.842% governor-family uncertainty rather than an
  assumed zero. Full reading in §28 N7 and P10b. **What remains un-cross-checked is the
  `GGOV1`/`DEGOV1` substitution of D2b, which T29 does not test.**
- **Two bisection steps between \(P_{\mathrm{head}}=1.0881\) and \(1.1013\) MW**, to tighten the
  envelope/feasibility coincidence below the present 0.6% resolution (§14 H1(iv)).
- the \(\epsilon\) convention in the §8 \(\bar A_{\mathrm{GFM}}\) formula — **newly blocking**, see §10;
- the membership of the 2-GFM deployment on axis A — unrecorded, and it swings \(\Delta\bar A_{\mathrm{GFM}}\) from −17.6% to +40.9%;
- exact \(I_{\max}\) and current-limiter strategy;
- controller gains;
- EMT timestep/solver;
- measured wall-clock runtime;
- main campaign size for native three-phase unbalanced EMT;
- numerical security thresholds;
- RoCoF measurement window;
- representative disturbance magnitudes;
- selected detailed configurations;
- ~~final applicability thresholds \(\mu_I^\star,\mu_P^\star\)~~ — **resolved differently**: the
  envelope is not a utilisation threshold. Peak \(\mu_P\) does not separate the regimes
  (1.067 and 1.434 both sit inside the accurate region); the gate is sustained ceiling occupancy
  at the post-event steady state (§14 H1);
- whether adaptive sampling alone is sufficient or a learned surrogate is required — **leaning
  "sufficient"**: bisection on a monotone boundary reached each boundary in ~15 runs (~18 min
  against ~10 h for a grid sweep), and the boundary was measured monotone in every coordinate
  swept. Not closed until EMT run times are known;
- the mechanism of the 3.2% \(\kappa_{os}\) gap between event classes (§28 N8);
- ~~the GFL fleet sizing: wind ≈12 MW, DPV ≈3.85 MW, EVCS ≈2.48 MW~~ — **closed 2026-09-06, see
  §28 N13.** The figures belong to the archived v3 RL layer
  (`archive/src/env/microgrid_env.py:1045`), which no experiment from T20 onward imports. The
  campaign's GFL fleet is 16 aggregated sgen totalling 2.8800 MW against 3.4900 MW of load and
  4.3624 MVA of GFM — internally consistent. No rescale is required;
- **the location of the interior capability optimum** — *newly opened 2026-09-06.* P19
  establishes that one exists; computing it needs the partition MILP, which does not yet
  exist. This is now the shortest path from a set of measurements to a contribution;
- **the \(\zeta\) threshold and the compliant droop range** for the T53 envelope (§28 N16
  items 1 and 2) — both are cheap sweeps and both are load-bearing for P17/P18;
- **the reactive ceiling under partitioning** — *newly opened 2026-09-06.* T52 measures 10 of 32
  islands over `q_max_pu` at the pre-event operating point, with the count moving 17/10/3 across
  the REGFM_A1 range. Which value ships is a modelling decision that is currently unmade, and it
  changes a headline count by 5.7× (§28 N14);
- whether the false-secure mechanism (`non_negative` ceiling substitution) is present in other
  positive-sequence platforms. Demonstrated in ANDES 2.0; the mechanism is generic to
  representing a bidirectional device with a unidirectional generator model, but **not verified
  elsewhere** (§28 N11).

# 26. Immediate next technical tasks — revised execution order

> **Re-ordered on 2026-09-03.** The list below was written when the campaign was expected to
> find a model failure. Tasks 1–2 and 10–11 are done or superseded; the phasor layer has
> answered more than it was expected to, and the binding constraint on the paper is now the
> absence of any EMT run, not the absence of more phasor runs. The revised head of the queue:
>
> 1. ~~**T29 — `GAST` vs `TGOV1` on the `gen_loss` boundary, diesel online.**~~ — **done
>    2026-09-06**, verdict inconclusive at 0.842%; the governor family is now a stated
>    uncertainty on the `gen_loss` boundary, not an assumed zero (§28 N7, P10b).
> 2. **Single-device GFM EMT model + REGFM_A1 conformance.** Everything downstream of
>    Contribution I is phasor-only until this exists. Tasks 3–6 of the old list, unchanged in
>    substance and now the critical path.
> 3. **Two bisection steps in \(P_{\mathrm{head}}\in[1.0881,\,1.1013]\)** to tighten the
>    envelope/feasibility coincidence below 0.6%. Cheap; strengthens the load-bearing claim of
>    Contribution I.
> 4. **Cross-feeder check of \(\Lambda\).** One additional feeder with a materially different
>    \(X_{\mathrm{feeder\ spread}}\) — ideally one where \(\Lambda\lesssim1\) — converts
>    Contribution II from "a criterion with one supporting system" into "a criterion tested on
>    both sides of its threshold". This is the single highest-leverage addition available to
>    the paper and it is not currently scheduled anywhere. Decide explicitly whether to do it or
>    to accept the single-system scope limit of §28 N2.
> 5. ~~**Rescale the GFL fleet** to be consistent with the D6 GFM rescale.~~ — **dropped
>    2026-09-06.** There is nothing to rescale: the figures that motivated the task belong to the
>    archived RL layer, and the campaign's fleet is consistent. §28 N13 records the trace.
> 5b. **Decide `q_max_pu`.** *(new, 2026-09-06)* The reactive ceiling is now a headline
>    constraint — 10 of 32 partitions fail it — and the count moves 17/10/3 across the REGFM_A1
>    range \([0.44,\,1.00]\). It cannot ship as an unexamined 0.60. Either justify the value on
>    the device specification or report the band everywhere (§28 N14).
> 5c. **Bisect the reactive boundary the way the frequency one was bisected.** *(new)* The
>    frequency side has a closed form confirmed to 0.37% (P11–P13); the reactive side has only a
>    pass/fail screen at the pre-event operating point. The obvious next quantity is the largest
>    island a given GFM set can hold within its reactive ceiling, which is what actually decides
>    partition viability (P15, P16) and is the coordinate no one in §23 has a criterion for.
> 6. ~~**Repo hygiene before any code release**~~ — **done, verified 2026-09-06.** The MAPPO /
>    baseline / layer-2 trees are under `archive/`; `artifacts/_lyap_vsg.txt` and both
>    `configs/training_config_*.yaml` are gone; `.gitignore:24` carries `artifacts/*/raw/` and
>    `git ls-files` tracks **0** `*.npz`.
>
> The original list follows, retained for the EMT-phase items which are unchanged.

1. **Verify \(\bar A_{\mathrm{GFM}}\)** against the canonical headroom-weighted definition and recompute the 2/5/6-GFM percentages if needed.
2. **Run the small SoC sensitivity sweep** to decide whether SoC belongs only to \(\Omega_E\) or also materially affects fast dynamics.
3. **Build and validate one single-device multiloop droop-GFM EMT model.**
4. **Validate the analytical current estimator** \(\hat I^{\mathrm{ana}}=|S|/|V|\) against EMT on the single device and a small multi-GFM network.
5. **Reproduce one unconstrained analytical-vs-EMT frequency/voltage response** with current and headroom far from their limits.
6. **Activate the current limiter** and verify the transition into the converter-constrained regime.
7. **Build the synchronous diesel EMT subsystem:** synchronous machine + governor/prime mover + AVR/exciter + breaker logic.
8. **Implement the hybrid SG/GFM analytical model** using the mixed-device index-set formulation from M2.
9. **Validate the mixed SG/GFM pre-transition operating case** before attempting diesel disconnection.
10. **Test controlled last-diesel-off** with \(P_{DG}(t_{off}^-)\approx0\).
11. **Test loaded diesel trip** at several pre-trip diesel loading levels.
12. **Benchmark actual EMT runtime** for matched-assumption and native-unbalanced IEEE 123 cases; replace all assumed computational budgets with measured values.
13. **Run a small balanced-vs-unbalanced robustness subset** so analytical-model error is not confounded with phase-unbalance error.
14. **Only after these checks, launch the topology/GFM deployment boundary campaign.**

# 27A. Final paper identity — v3.2 (2026-09-03, current)

The lead question is now answerable rather than open, and the identity follows the answer:

> **A closed-form frequency-security boundary holds exactly on the energy-feasible set of a
> 100% inverter-based islanded microgrid; two dimensionless criteria decide in advance whether
> topology, disturbance location or current limiting can move it; and the last-diesel-off
> transition is not the worst case.**

Hierarchy:

\[
\boxed{
\textbf{I: closed form + validity envelope + degradation law}
>
\textbf{II: }\Lambda\textbf{ and }I_{\max F}^{\mathrm{crit}}\textbf{ screening criteria}
>
\textbf{III: event-class dependence of the boundary}
}
\]

Title:

> **Closed-Form Frequency Security Limits for 100% Inverter-Based Islanded Microgrids: Validity
> Envelope and Dimensionless Screening Criteria**

The last-diesel-off transition remains the physical operating narrative, and is now also
Contribution III rather than only a setting.

**The honest one-line summary of the campaign, which the introduction should not hide:** four
pre-registered attempts to break a reduced model failed, and the paper reports the model, the
exact conditions under which it holds, and how it degrades when they do not. A paper that
reports refuted hypotheses alongside the surviving result is more credible than one that reports
only confirmations; §14 and §28 exist so that this is visible rather than reconstructible.

---

# 27. Final paper identity — v3.1 *(superseded by §27A)*

> **Superseded on 2026-09-03.** Retained for audit. The v3.1 lead question presupposed that
> converter limits, headroom and feeder configuration *do* move the operational reserve
> boundary; measured, headroom does (trivially, \(\kappa=1.006\)) and the other two do not
> (0.0000 MW, no admissible limiter setting). The v3.1 C1 > C2 > C3 hierarchy is replaced by
> I > II > III of §16.

The project is no longer primarily:

> **“What is the frequency-security region of a 100% IBR microgrid?”**

because frequency-security regions, GFM placement, current limiting and frequency-constrained scheduling all have substantial prior literature.

The lead question is now:

> **“When can a fast reduced SG/GFM analytical model be trusted for security assessment of a diesel-free/100% inverter-based islanded microgrid, and how do converter limits, headroom and feeder configuration move the operational reserve boundary?”**

The intended hierarchy is

\[
\boxed{
\textbf{C1: reduced-model applicability}
>
\textbf{C2: operational reserve/security boundary}
>
\textbf{C3: topology/GFM-conditioned headroom shift}
}
\]

with the last-diesel-off transition providing the physical operating narrative.

The preferred working title is:

> **When Do Reduced Frequency Models Overestimate Security in 100% Inverter-Based Islanded Microgrids?**

If the measured analytical–EMT discrepancy is small over nearly the entire tested domain, use the outcome-neutral title:

> **Validity Limits of Reduced Frequency Models for Security Assessment of 100% Inverter-Based Islanded Microgrids**

This precommits the paper to a publishable interpretation regardless of whether the final result is a large modeling gap or a broad validated reduced-model applicability envelope.

---

# 28. Claims register — what may be claimed, what may not, and why

**Authoritative.** Where any other section of this document conflicts with this one, this one
governs. Every entry names the measurement that settles it. Nothing here rests on argument
alone.

## 28.1 Claimable

| # | Claim | Evidence | Stated scope |
|---|---|---|---|
| P1 | \(\Delta P_{\max}=(f_0-f_{\min})\sum_g S_g/(\kappa_{os}f_0R)\) predicts the simulated boundary to 0.26% | T21 vs closed form, 36× range in \(\Delta P\) | positive-sequence; this fleet and feeder |
| P2 | \(\Delta f_{ss}=f_0R\,\Delta P/\sum_g S_g\) reproduces to four significant figures | T20 | unconstrained regime only |
| P3 | The validity envelope is *sustained ceiling occupancy at the post-event steady state*, not peak utilisation | T26 (\(\mu_P^{\mathrm{peak}}=1.434\) → 1.44% error) | as measured |
| P4 | Envelope and energy-feasibility boundaries coincide **to 0.6%** | T26 \(\kappa=1.006\pm0.002\); saturation transition brackets \(P_{\mathrm{head}}^{\min}\) | resolution-limited, see N5 |
| P5 | Outside the envelope the error is monotone, reaching 121.4% | T26, 7 points | as measured |
| P6 | A positive-sequence platform can return `SECURE` for operating points with no post-event equilibrium | T26 pre-patch behaviour, `REGF1.Pmax` `non_negative` | ANDES 2.0; see N11 |
| P7 | The boundary is invariant to tie configuration, disturbance location and limiter setting, **and \(\Lambda\gg1\) explains why** | T22, T23 (84 runs), T24 | one feeder; see N2, N3 |
| P8 | \(I_{\max F}^{\mathrm{crit}}=1.356\) pu lies below the REGFM_A1 parameter floor, so no compliant limiter binds | T24 + \(I_{\mathrm{dev}}(\Delta P)\) fit | this fleet; see N3 |
| P9 | Tripping the last synchronous machine is 2.0% *easier* than an equal non-synchronous loss, with an exchange between bulk and local coordinates | T25, non-overlapping brackets | positive-sequence |
| P10 | The diesel-off boundary is independent of the assumed \(H\) and of the governor family | T25b, T27 (\(10^{-10}\)) | **diesel-off only** — with the diesel *online* the families differ by 0.842% (T29); see N7, N10 |
| P10b | With the diesel online, the governor family shifts the `gen_loss` boundary by 0.842% (6.4 kW), splits the **transient** but not the steady state (nadir 1.8–5.2 mHz, \(f_{ss}\) 0.03–0.08 mHz), and the boundary does not depend on the pre-trip diesel loading | T29, 12 evals per arm, two loadings | pre-registered verdict **inconclusive**; report the number, not a verdict |
| P11 | The closed form transfers across the partition space: \(\le0.37\%\) relative error, absolute residual \(\le2.65\) kW, over a 5.75× range in \(\sum_g S_g\) | T52, 14/14 islands bisected, all monotone; intact feeder as control (0.7261 against T21's 0.7241) | positive-sequence; islands of this feeder |
| P12 | At fixed \(\sum_g S_g\) the boundary does not move with the **number** of GFM units (3/4/5) or the island load (1.49× range): spread \(\le2.6\) kW, i.e. the bisection resolution | T52, three fixed-\(\sum_g S_g\) groups (2.845, 3.604, 3.983 MVA) | as measured; this is a statement about the *functional form*, not only its accuracy |
| P13 | \(\kappa_{os}=1.00338\pm0.00252\) (0.251% scatter, \(n=14\)) across the partition space | T52 | positive-sequence; supersedes the 1.005 used by the T51 screen |
| P14 | **Partitioning** moves the security boundary where **routing** does not: 282% relative spread in the margin over 32 islands, 8 of which cannot survive the loss of their own largest DER block while all 32 have positive headroom | T51 (closed form over 92 valid switch states), pre-registered H1 \(\ge20\%\) | closed form; the 8 are a screen, not all bisected |
| P15 | The binding constraint on which partitions exist is **reactive, not frequency**: 10/32 islands exceed the GFM reactive ceiling at the pre-event operating point, and only 4 of those are also frequency-insecure | T52 feasibility screen; 15/22/29 feasible at \(q_{\max}=0.44/0.60/1.00\) | must be reported as the band, not a point; see N14 |
| P16 | Reactive demand per MVA of island GFM rating rises monotonically as the fleet fragments: mean 0.383 at 5 GFM → 0.768 at 1 GFM, peak 1.240 | T52, 32 islands | the *ordering* is independent of \(q_{\max}\); the pass/fail count is not |
| P17 | The droop gain is bounded below by a small-signal stability floor that **scales with the capacity it is applied to**: \(R_{\min}\propto(\sum_g S_g)^{0.388}\), so certified capability grows only as \(C_{\max}\propto(\sum_g S_g)^{0.612}\) — a 3.83× increase in assigned converter capacity returns **2.18×** capability | T53, 13 bracketed islands, \(R^2=0.911/0.962\), max fit error 11.3%; exponents sum to 1.000 by construction | positive-sequence small-signal; \(\zeta_{\min}\ge0.02\); see N16 |
| P18 | An island anchored on a **single** GFM unit is not on that curve: it sustains \(\zeta=0.614\) at the stiffest droop swept, never destabilises, and delivers **1.77×** the capability the law predicts | T53; every multi-unit island fails through an oscillatory mode whose violence grows with fleet size (\(\max\mathrm{Re}\) 0.19 → 2.79) | this quantifies, as a droop floor, the parallel-voltage-source interaction that Häberle (TSG 2024) requires qualitatively; the single-unit point is a **lower bound**, see N16 |
| P19 | Fragmenting the GFM fleet **relaxes** the droop stability floor and **tightens** the reactive ceiling; the two mechanisms oppose, so the capability optimum is interior to the configuration space | P16 + P17/P18 | the existence of an interior optimum; its *location* is not yet computed |

## 28.2 Not claimable

**N1 — "Reduced frequency models overestimate security in 100% IBR islanded microgrids."**
*Why not:* measured false. Inside the feasible set the nadir error is \(\le1.44\%\) and the
closed form predicts the boundary to 0.26%. *Settled by:* T21, T26. *What replaces it:* the
validated-envelope statement, which §15 pre-committed to as Outcome B. The v3.1 title asserting
the opposite is retired.

**N2 — "Feeder reconfiguration shifts the security boundary."** — **AMENDED 2026-09-06. The
prohibition holds for *routing* and is withdrawn for *partitioning*; the unqualified sentence
is now the thing that must not be said, in either direction.**
*Why not, for routing:* measured 0.0000 MW dispersion across tie configurations and 0.07%
physical across 84 disturbance locations. *Settled by:* T22, T23. *What replaces it:* the
invariance itself, plus \(\Lambda\) as the criterion predicting it. *Why the amendment:* T22/T23
varied which path carries the power while holding the generator set of a single island fixed,
and the closed form has no routing term — it has \(\sum_g S_g\). A switch state that **splits**
the feeder changes \(\sum_g S_g\) per island directly, and the boundary then moves by
construction: 0.0629 → 0.7234 MW across the 32 islands, confirmed in ANDES to \(\le0.37\%\)
(P11, P14). *Required wording:* name the operation. "Routing is invariant, partitioning is not,
and \(\Lambda\) explains the first while \(\sum_g S_g\) explains the second." Do not write
"reconfiguration" unqualified anywhere in the manuscript. **Residual scope limit:** \(\Lambda\in[4,14]\) is
measured on **one** feeder. The paper may present \(\Lambda\) as a screening criterion with a
stated derivation and one supporting system; it may **not** present the range as an empirically
general constant, and it may **not** claim the criterion has been tested on the \(\Lambda\lesssim1\)
side of its own threshold, because it has not. See §26 revised task 4.

**N3 — "Converter current limits shape the operating boundary."**
*Why not:* \(I_{\max F}^{\mathrm{crit}}=1.3555\) pu is 1.107× **below** the floor of the
REGFM_A1 admissible range; no spec-compliant limiter setting makes current the binding
constraint on this fleet. At the boundary \(\mu_I=0.809\), well inside rating. *Settled by:*
T24 plus the affine \(I_{\mathrm{dev}}\) fit. *What replaces it:* the screening inequality and
the explicit statement that current is not binding here. Consequence: the REGFM_A1
fault-current-limiter implementation is de-prioritised — it would refine a constraint measured
not to bind.

**N4 — "There is a nontrivial excess dynamic reserve requirement, \(\kappa-1>0\)."** —
**AMENDED 2026-09-06: still not claimable *for the intact fleet*, and the motivating sentence it
governs is now claimable *under partitioning*.** T51 measures every one of the 32 islands to have
positive headroom while 8 of them cannot survive the loss of their own largest DER block: the gap
is 0.6% at fleet level and qualitative at island level, because partitioning shrinks
\(\sum_g S_g\) while the block sizes stay fixed. So *"energy feasible \(\not\Rightarrow\)
frequency secure"* may be used **only** with the partitioning mechanism named alongside it, and
never as a statement about the intact feeder, where the measured gap is 0.6%. The headline
metrics \(\kappa\) and \(P_{\mathrm{excess}}\) remain withdrawn.
*Why not:* \(\kappa=1.006\pm0.002\); \(P_{\mathrm{excess}}\) is 4.7–5.3 kW. *Settled by:* T26.
*Consequence:* \(\kappa\) and \(P_{\mathrm{excess}}\) are withdrawn as headline metrics
(v3.1 §25 froze \(\kappa\) as "a primary nontrivial metric" — that freeze is lifted), Figure B
is demoted to an inset, and the motivating sentence *"energy feasible \(\not\Rightarrow\)
frequency secure"* must not be used to imply a large gap on this testbed, where the measured gap
is 0.6%.

**N5 — "The validity envelope and the energy-feasibility boundary are identical."**
*Why not:* they coincide **to within the 0.6% resolution of the sweep**. The 0 → 6 saturation
transition falls in \([1.0881,\,1.1013]\) MW and \(P_{\mathrm{head}}^{\min}=1.0947\) MW falls
inside it; that is coincidence at the available resolution, not analytical identity. *Required
wording:* "coincide to 0.6%". *Tightened by:* two further bisection steps, §26 revised task 3.

**N6 — "The headroom-weighted accessibility index \(\bar A_{\mathrm{GFM}}\) explains the
boundary."**
*Why not:* two independent reasons. (i) There is no dependent variable left to explain — the
boundary is invariant (N2). (ii) The index is separately compromised: the source artifact
`electrical_distance_analysis.json` is untracked and its generating script is absent from the
entire git history; the published 2→6 and 5→6 percentages are the **raw** index mislabelled as
the canonical one; the \(\epsilon\) convention flips the sign of the 2→6 change; and the 2-GFM
member set is unrecorded, swinging \(\Delta\bar A_{\mathrm{GFM}}\) from −17.6% to +40.9%.
*Settled by:* T1 (§9, §10) and T22. *Consequence:* \(\bar A_{\mathrm{GFM}}\) is not published as
an explanatory metric, and the 47-configuration correlation table of §9 does not enter the
manuscript in its present form.

**N7 — "The governor model family does not affect the results."**
*Why not:* T27 replaced `TGOV1` with `GAST` on the **diesel-off** boundary, where the governor
is disconnected together with the machine at \(t_{\mathrm{event}}\) and therefore cannot
influence the outcome by construction. The agreement to \(10^{-10}\) is a consequence of that
structure, not evidence about governor modelling. T27 has **no power** over any scenario in
which the diesel stays online — and \(\kappa_{os}^{\mathrm{gen\_loss}}=1.2275\), the leading
constant of Contribution I, is exactly such a scenario. *Claimable subset:* governor-family
invariance **of the diesel-off boundary**. Separately, `TGOV1`/`GAST` are phasor stand-ins for
the `GGOV1`/`DEGOV1` selected in D2b; that substitution is unverified in either direction.

**T29 has now run (2026-09-06) and the reasoning above is confirmed by measurement.** With the
diesel **online** through a `gen_loss` event, the two families do not agree:

| | TGOV1 | GAST | difference |
|---|---|---|---|
| \(\Delta P_{\max}\), diesel at 0.30 MW | 0.76514 MW | 0.77158 MW | **6.445 kW = 0.842%** |
| \(\Delta P_{\max}\), diesel at 0.50 MW | 0.76514 MW | 0.77158 MW | 6.445 kW = 0.842% |

*Pre-registered verdict: **inconclusive*** — 0.842% falls between H0 (\(\le0.5\%\)) and H1
(\(>2\%\)) and must be reported as such, not rounded into either. It is nonetheless **3.2× the
bisection half-bracket**, so the difference is resolved, not noise, and it is seven orders of
magnitude away from T27's \(10^{-10}\). The structural-artefact reading of T27 is therefore
measured, not merely argued.

Three further readings, all from the 11 matched probe points:

- the split is **transient, not steady-state**: \(f_{ss}\) differs by 0.03–0.08 mHz (the droop is
  the same in both), while the nadir differs by 1.8–5.2 mHz and the gap grows monotonically with
  \(\Delta P\). GAST gives the *higher* nadir throughout;
- the boundary is **independent of the pre-trip diesel loading** — 0.30 and 0.50 MW give
  identical values to five digits — so what the machine contributes here is its presence and its
  droop, not its dispatch;
- **beyond** the boundary the families diverge violently: at \(\Delta P=1.4\) MW the nadirs are
  58.018 and 59.077 Hz, a **1.06 Hz** gap. Nothing stated in the insecure region transfers
  between governor families.

*Consequence for Contribution I:* the governor family is an uncertainty of **0.842%** on the
`gen_loss` boundary — 3.4× the 0.251% scatter of \(\kappa_{os}\) across the partition space
(P13). It must be carried as a stated uncertainty rather than assumed zero. It does not
invalidate the contribution; it bounds it. *Still outstanding:* the `GGOV1`/`DEGOV1`
substitution of D2b, which this run says nothing about.

**N8 — "The 3.2% gap between the two \(\kappa_{os}\) values is explained."**
*Why not:* no mechanism is established. Two candidates have been eliminated by measurement —
inertia (T25b) and, for the diesel side only, governor dynamics (T27). A third remains: a
synchronous machine trip removes a *voltage* source and its reactive support, which is
consistent with the direction of the T25 coordinate split (\(V_{\min}\) −0.0130 pu, \(\mu_I\)
+18.2%). This is a hypothesis with one supporting observation, not a result. *Required wording:*
report the two constants separately with their scatter and state that the mechanism is open.
**Do not** publish the pooled value 1.2239 ± 0.33% as a single constant — it averages two
populations whose scatters differ by a factor of 8.

**N9 — Anything stated on EMT grounds.**
*Why not:* no EMT run exists. Every number in §14, §16 and §25 is ANDES 2.0 positive-sequence.
Decision D5 states that EMT under D1 is "the only thing a claim is stated on"; that rule is
either honoured — in which case the present results are *preliminary* until the EMT phase —
or explicitly relaxed with a declared scope. **It must not be left ambiguous.** Immediate
consequences: no analytical-vs-EMT speed-up may be quoted (only the phasor-vs-grid-sweep figure,
~15 runs / ~18 min against ~10 h); Figure C's axes must read `ana` vs `andes`, not `ana` vs
`emt`; and the analytical current estimator's 4.6–6.2× shortfall is a phasor-layer comparison.

**N10 — "The \(H=1.0\to0.1\) s sweep measures the inertia contribution."**
*Why not:* the design cannot separate the effects. The machine is disconnected at
\(t_{\mathrm{event}}\), so its rotor leaves the system simultaneously and the pre-event state is
steady; `GENROU.M` was verified to differ by 10× and all 13 probe points were nonetheless
identical. *Claimable subset:* the diesel-off boundary does not depend on the assumed inertia
constant — which rebuts *"your result depends on an arbitrary \(H\)"* by measurement.
*To actually measure it:* keep the diesel online, apply `gen_loss` elsewhere, sweep \(H\).
Not run.

**N11 — "Positive-sequence tools in general return false-secure verdicts for bidirectional
devices."**
*Why not:* demonstrated in ANDES 2.0 via `REGF1.Pmax`'s `non_negative` declaration. The
mechanism — representing a bidirectional device with a unidirectional generator model, so that a
physically valid negative ceiling is silently replaced — is generic in kind, but has **not** been
verified in any other platform. *Required wording:* name the platform and the parameter, state
the mechanism as generic in kind, and do not generalise the observation.

**N12 — "T28 verifies \(\Delta P_{\max}\)."**
*Why not:* T28 verifies that the ceiling guard changed nothing in the region where the guard
never fires (\(p_{\mathrm{head}}=\)`None` ⇒ \(P_{\max}=0.651>0\) at every probe point; zero
correction lines in the log). That is a regression check, not an independent verification.
*What does support \(\Delta P_{\max}\):* the four invariance measurements plus the 0.26%
closed-form agreement.

**N13 — Any result depending on the GFL fleet magnitudes as currently configured.** —
**RESOLVED 2026-09-06 by tracing the figures, not by rescaling anything.**
*What the entry asserted:* wind ≈12 MW, DPV ≈3.85 MW and EVCS ≈2.48 MW remain at v3 scale on a
3.49 MW feeder while the GFM fleet was rescaled by D6 to 4.362 MVA.
*What is actually in the campaign:* the phasor case is built by
`build_ieee123_net(mode="feeder123", source_mode="publish")`, whose GFL fleet is **16 aggregated
sgen totalling 2.8800 MW** — 14 × 0.20 MW PV, 0.05 MW wind, 0.03 MW storage — against 3.4900 MW
of load and 4.3624 MVA of GFM. That is GFM at **60.2% of installed converter capacity** and GFL
at **82.5% of energy supply**, and it is internally consistent.
*Where the 12 MW came from:* the only `12.0` wind cap in the tree is
`archive/src/env/microgrid_env.py:1045` and `archive/src/opt/precompute.py:133` — the **archived
v3 RL layer**, which no experiment from T20 onward imports. The 3.85 and 2.48 figures appear
nowhere in code or configuration, only in this document.
*Consequence:* no T20–T52 result depends on the magnitudes the entry names, and §26 revised task
5 ("rescale the GFL fleet") is dropped as unnecessary. **What survives:** the largest single GFL
block is 0.20 MW, an artefact of the aggregation granularity rather than a physical unit, so a
statement of the form *"a credible DER loss of 0.20 MW"* still needs that granularity declared —
which is why T51 also reports the aggregation-free critical trip fraction (P14).

**N14 — "10 of 32 partitions are reactive-infeasible."** *(the count, not the finding)*
*Why not as a point value:* \(\mu_Q>1\) is measured against `q_max_pu = 0.60`, a **choice** inside
the REGFM_A1 admissible range \([0.44,\,1.00]\). The count moves to **17 / 10 / 3** at
\(q_{\max}=0.44/0.60/1.00\). *Required wording:* report the band and name the parameter every
time. *What is claimable regardless:* the monotone ordering of P16, and the fact that the
reactive criterion and the frequency criterion select different island sets (P15).
*Two further scope limits on the same measurement:* the Slack GFM is deliberately unlimited in
the power flow, so \(\mu_Q\) compares **required Q against the declared ceiling** — the violation
surfaces at TDS initialisation, not as a non-converged power flow; and the fleet was sized and
placed for the intact feeder, so this is a statement about **sizing under reconfiguration**, not
about grid-forming converters in general.

**N15 — Any island result produced before 2026-09-06.**
*Why not:* `build_system` computed the GFM dispatch from the whole feeder's load while loading
only the buses present in the case. Identical for every topology up to T50, where the dropped
buses are the zero-load tie ends (measured \(\Delta=0.00\mathrm{e}{+}00\) on G0), and wrong the
moment a switch state leaves a populated section outside the case: non-slack units were pinned at
an impossible \(p_0\) and the slack was driven into over-frequency and over-current at \(t=0\).
*Consequence:* a first reading of T52 that reported a third failure mode ("current limit at
\(\Delta P=0\)", \(\mu_I=1.25{-}3.96\)) was an artefact of that bug and is withdrawn; after the
fix, **22/22 reactive-feasible islands are operable at \(\Delta P=0\)** with \(\mu_I\in[0.049,
0.358]\). There are two failure modes, not three. *Fixed in:* `build_case.py:526`,
`impact(build_system, upstream)` = LOW, commit `fd5047f`.

**N16 — "The capability envelope of T53 is the envelope."** *(the numbers, not the law)*
Three qualifications, all of which must travel with P17/P18:
1. **The floor is defined against a chosen damping threshold.** \(R_{\min}\) is where
   \(\zeta_{\min}\) crosses **0.02**; the sensitivity of the exponents to that choice is
   **not measured**. The exponents are reported with the threshold beside them or not at all.
2. **The envelope has not been intersected with the specification-admissible droop range.**
   The single-GFM island is still stable at \(R=0.006\) — a 0.6% droop, outside ordinary
   grid-forming practice — and its \(R_{\min}\) is the **bottom of the swept grid, not a
   measured floor**, so its 8.33× and 1.77× are **lower bounds**. For small islands the
   binding ceiling may be the specification rather than stability, and that has not been
   checked.
3. **29% of the probes (42/143) emitted TDS-initialisation warnings** from the reactive
   limiter at small \(R\). The \(R_{\min}\) brackets themselves have clean spectra on **both**
   sides — positive damping above, \(\max\mathrm{Re}>0\) and \(\zeta<0\) below — so the
   brackets stand; the warning is recorded because it means the sweep passes through a region
   where the reactive limiter is active, which is the same limiter P15 measures.
*What is claimable regardless:* the direction and the mechanism — the floor rises with
capacity, the single-unit island escapes it, and the two mechanisms of P19 oppose.

## 28.3 Standing rule

A quantity measured to equal its trivial value is reported as such and removed from the
contribution set; it is not restated in units or coordinates that make it look larger.
\(\kappa=1.006\) is the reference case. This rule is what makes the surviving claims in §28.1
worth a reviewer's trust.
