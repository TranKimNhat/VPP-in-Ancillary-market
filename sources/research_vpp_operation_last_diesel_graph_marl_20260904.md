# Research note: VPP operation around the last-diesel-off transition

Date: 2026-09-04

## Proposed operational gap

Use the analytical/ANDES campaign to determine the **total** active reserve required for a secure last-diesel-off transition, then pose the VPP problem as the intertemporal and network-aware **disaggregation** of that requirement among spatially distributed DERs. In the present high-coupling regime, feeder topology may have negligible effect on the aggregate frequency boundary while still changing voltage, branch loading, reactive support, device apparent-power margins, losses, and the cost/feasibility of allocating reserve to individual DERs.

The resulting central question is: how should a VPP pre-position energy and allocate active/reactive headroom and fast-response participation before disconnecting the last diesel unit, under renewable/load/DER-availability uncertainty and feasible feeder reconfiguration?

## Literature boundary from the current scan

- *Virtual Power Plants for Frequency Regulation: A Learning-Based Method With Safety Guarantee*, IEEE Transactions on Smart Grid, 2026, DOI https://doi.org/10.1109/TSG.2025.3618896, already combines VPP frequency regulation, sufficient transient-frequency safety conditions, RNN-based reinforcement learning, and reserve-proportional disaggregation. Therefore generic “RL-based safe VPP frequency regulation” is not a novelty claim.
- *Design and Analysis of Parallel-Connected Grid-Forming Virtual Synchronous Machines for Island and Grid-Connected Applications*, IEEE Transactions on Power Electronics, 2022, DOI https://doi.org/10.1109/TPEL.2021.3127463, coordinates parallel converter units so a plant presents a desired aggregate frequency/voltage behavior. Generic aggregate GFM coordination is also established.
- *Testing GFM and GFL Inverters Operating With Synchronous Condensers*, IEEE PESGM 2023, DOI https://doi.org/10.1109/PESGM52003.2023.10252338, includes experimental loss-of-last-synchronous-generator cases. The physical event itself is not novel.
- *Day-ahead Complex Power Scheduling in a Reconfigurable Hybrid-Energy Islanded Microgrid With Responsive Demand Considering Uncertainty and Different Load Models*, Applied Energy, 2022, DOI https://doi.org/10.1016/j.apenergy.2021.118416, already schedules active and reactive power under uncertainty and reconfiguration. Generic P/Q scheduling is not novel.
- EPRI's GFM tutorial documents St. Eustatius operation with diesel-free daytime supply and seamless transfer following simultaneous loss of gensets: https://restservice.epri.com/publicdownload/000000003002025483/0/Product. This supports the practical motivation but not novelty.

## Defensible novelty direction

The paper should not claim novelty in GraphSAGE, MAPPO, safe RL, P/Q scheduling, or the last-synchronous-machine event separately. The narrower operational contribution is a **transition-aware VPP reserve product and control problem** that couples:

1. a pre-transition diesel loading/scheduled switch-off state;
2. an analytically or ANDES-calibrated total active FFR requirement;
3. spatial disaggregation of active and reactive headroom under inverter apparent-power, voltage, branch-current, ramp, and SoC constraints;
4. post-transition GFM participation; and
5. intertemporal operating cost and renewable/load uncertainty.

The physics layer answers “how much aggregate response is required.” The graph-MARL layer answers “which DERs should reserve and deliver it, given their locations, local limits, and future availability.” This is compatible with the measured topology-invariance of the aggregate frequency boundary and gives topology a physically defensible role in the disaggregation/network-feasibility layer.

## Recommended learning role

GraphSAGE/MAPPO can be retained as the implementation of the operational policy, not as the scientific novelty. The old free joint `power-reference + droop-gain` action should be replaced or constrained by a two-timescale parameterization:

- slow supervisory actions: DER P/Q setpoints, active/reactive headroom reservations, and optionally the diesel-off readiness decision;
- fast local response: fixed-form P-f/Q-V droop whose participation coefficients are linked to the reserved headroom rather than independently and arbitrarily selected by MARL.

MARL is justified only by a genuinely sequential problem with SoC evolution, renewable/load uncertainty, changing DER availability, and repeated diesel-free intervals. A one-event static allocation would be better solved by optimization and would not justify MARL.

## Test-system implication

Retain the modified IEEE-123 feeder and the six-GFM fleet, but redesign operating scenarios rather than replacing the platform. Add high-load/low-voltage and reactive-stress conditions, heterogeneous DER availability/headroom, and topology cases that produce distinguishable voltage/current/line-loading consequences. The current low-stress homogeneous cases remain a control regime.

If current-limiter dynamics remain a claim, the ANDES device model must include an operative limiter. Otherwise treat current as an ex-post equipment-margin constraint and avoid claims about limiter switching or synchronization under saturation.

ANDES can remain the sole physical dynamic simulator by generating an offline response/security table for policy training and by evaluating held-out trajectories directly. A small EMT spot-check would be validation of the simulator/model, not a second training or evaluation campaign.

