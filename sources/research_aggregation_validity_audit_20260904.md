# Aggregation-validity audit for the 100% IBR VPP concept

Date: 2026-09-04

## Question

Does the current IEEE-123/ANDES campaign show that VPP aggregation is security-preserving, or is the apparent invariance caused by a testbed that stays inside the aggregate model's validity regime? Does T25 already constitute a false-secure counterexample?

## Verdict

The present parameterization is a useful **positive control for aggregation validity**, but it is not a decisive falsification test. It combines a small homogeneous droop-GFM fleet, strong converter-interface impedance relative to feeder electrical spread (`Lambda = 4--14`), an active-power loss event, and operation below the assumed transient-current ceiling. These choices are biased toward coherent aggregate behavior.

The earlier novelty audit was too strong if it concluded that Chen et al. and Song et al. fully close the gap. Chen et al. explicitly assume unsaturated, symmetrical operation and construct/validate an aggregate model; they do not derive a general operational certificate for when aggregation preserves joint dynamic security. Song et al.'s DOSS is a multi-period distribution operating region built from power-flow, voltage, thermal, energy, uncertainty, and privacy constraints; it is not a transient frequency/RoCoF/GFM-synchronization/current-limiter model.

The rebuttal is also too strong in calling T25 a false-secure case. T25 shows a ranking conflict: after the last-diesel trip, frequency nadir and RoCoF improve while voltage and current margins worsen. However, at the reported joint-security boundary, frequency/RoCoF still bind and `V_min` and `mu_I` remain admissible. Thus T25 is a **false-secure precursor / metric-decoupling observation**, not yet a threshold-inverting counterexample.

## Paper checks

### Chen, Liu, and Milano (2021)

- Title: *Aggregated Model of Virtual Power Plants for Transient Frequency and Voltage Stability Analysis*
- Journal: IEEE Transactions on Power Systems, 36(5), 4366--4375
- DOI: https://doi.org/10.1109/TPWRS.2021.3063280
- IEEE record: https://ieeexplore.ieee.org/document/9369128
- Author-hosted PDF: https://faraday1.ucd.ie/archive/papers/vppequiv.pdf
- Key scope statement: the aggregate model assumes all internal DG units operate in an **unsaturated** and **symmetrical** situation.
- The paper neglects fast current- and voltage-controller dynamics in the reduced model. Its validation itself reports lower active-power peaks during voltage changes and larger mismatch after grid-impedance changes because the reduced model omits current-controller transients.
- Therefore, it establishes that a carefully parameterized aggregate model can work in its stated regime; it does not establish a general security-preserving aggregation implication under saturation, heterogeneous controls, or hidden local limits.

### Song et al. (2025)

- Title: *Distribution Dynamic Operating Security Space Characterization for Aggregation and Congestion Capacity Evaluation of Virtual Power Plant*
- Journal: IEEE Transactions on Smart Grid, 16(5), 3904--3918
- DOI: https://doi.org/10.1109/TSG.2025.3570448
- Full-text author version: https://www.researchgate.net/publication/391791707_Distribution_Dynamic_Operating_Security_Space_Characterization_for_Aggregation_and_Congestion_Capacity_Evaluation_of_Virtual_Power_Plant
- The word “dynamic” refers to time-coupled operating space. The paper constrains active/reactive outputs using power-flow, voltage, thermal/current, storage-energy, congestion, uncertainty, and privacy considerations.
- It does not model transient nadir, RoCoF, GFM synchronization, converter control switching, or fault-current limiting. It is adjacent prior art for an operating-envelope story, not a complete neutralization of a transient dynamic-security certificate.

### Relevant boundary literature

- Trujillo et al., *Analytical Models of Frequency and Voltage in Large-Scale All-Inverter Power Systems*, IEEE Transactions on Power Systems, 41(4), 2547--2562, DOI https://doi.org/10.1109/TPWRS.2025.3650698. It already demonstrates that spatially resolved frequency and voltage dynamics can invalidate a single system-frequency trajectory and validates its low-order model against EMT. This narrows novelty: simply adding nodal `f` and `V` is not enough.
- Du et al., *Positive-Sequence Modeling of Droop-Controlled Grid-Forming Inverters for Transient Stability Simulation of Transmission Systems*, IEEE Transactions on Power Delivery, 39(3), 1736--1748, DOI https://doi.org/10.1109/TPWRD.2024.3376245. It shows that positive-sequence GFM models can include P/Q and fault-current limiting and can be accurate when benchmarked against EMT. The local ANDES `REGF1` implementation is only a subset of this capability.
- Haeberle et al., *Grid-Forming and Spatially Distributed Control Design of Dynamic Virtual Power Plants*, arXiv:2202.02057, https://arxiv.org/abs/2202.02057. It coordinates heterogeneous, spatially dispersed DERs and accounts for device limits, so generic “heterogeneous DVPP coordination” is not novel by itself.
- He et al., *Grid-Forming Control of Modular Dynamic Virtual Power Plants*, arXiv:2410.14912, https://arxiv.org/abs/2410.14912. It extends aggregate/disaggregate control design to modular heterogeneous DVPPs.

## Local evidence audit

1. **Topology and event location:** these are genuine dynamic perturbations, but only within one strongly coupled feeder/fleet regime. They support local invariance, not a universal sufficient condition.
2. **T24 current-limit sweep:** `ImaxF` is not passed into the ANDES `REGF1` equations. `mu_I` is computed after simulation by dividing the unconstrained current trace by the assumed ceiling. T24 therefore estimates the current ceiling at which the present trajectory would become inadmissible; it does not simulate limiter engagement or demonstrate invariance to limiter dynamics.
3. **T25 inertia sweep:** the diesel is disconnected at the event and the pre-event state is steady. Varying its inertia cannot isolate post-trip inertial support. The concept file already records this as a failed experiment design. The result only shows insensitivity to an arbitrary inertia value for this exact trip implementation.
4. **T25 coordinate split:** at comparable disturbance power, diesel trip improves `f_nadir` and RoCoF but worsens `V_min` by 0.0130 pu and `mu_I` by 18.2%. Yet the diesel-trip boundary is still limited by nadir/RoCoF, with `mu_I = 0.809` and `mu_P = 0.364`. This is not yet `M_a secure => M_d unsafe`.
5. **Pmax non-negative defect:** this is a real representability/software-domain failure that previously produced a false secure verdict. It is not an aggregation counterexample between two detailed VPP realizations with the same reduced state.
6. **T26 outside-envelope error:** the 121.4% error demonstrates breakdown of the closed form once sustained P saturation/no-equilibrium behavior is entered. It supports an applicability gate for the reduced formula, but it is not an internal-network/controller counterexample.

## Minimal decisive experiment

Keep ANDES as the search platform, but construct paired detailed fleets `A` and `B` satisfying the same reduced-model inputs. Preserve total P/Q, MVA, energy, P/Q headroom, equivalent droop, and the same contingency. Change only hidden coordinates: spatial allocation of headroom/current rating, controller time constants, converter-interface impedance, and/or controller family. Test a strong-coupling homogeneous positive control and at least one weak-coupling or heterogeneous regime. A valid counterexample requires the reduced model to accept both while the detailed joint criteria reject exactly one.

Because the installed ANDES 2.0 `REGF1` has no operative circular/fault-current limiter, current-limiter-induced switching cannot be claimed from the existing campaign. Either add a validated limiter model in ANDES, or remove limiter dynamics and synchronization-loss claims from the central contribution. A small EMT benchmark remains the safest validation response to the original reviewer comment; it need not become the training/evaluation platform.

