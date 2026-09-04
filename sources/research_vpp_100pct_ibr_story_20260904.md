# Research scan: VPP storytelling for 100% IBR islanded systems

Date: 2026-09-04

## Research question used for the scan

Which emerging concepts can be combined coherently with virtual power plants (VPPs) and 100% inverter-based resource (IBR) islanded systems, while retaining a system-level dynamic-security contribution appropriate for IEEE Transactions on Power Systems?

## Main findings

1. Recent VPP/DVPP research already covers aggregate frequency control, controller parameterization, and learning-based safety guarantees. A new paper should therefore avoid presenting generic VPP coordination or reinforcement learning as the main novelty.
2. Grid-forming capability is moving from a device feature toward a procured and verified system service. Current operator initiatives distinguish inertia, short-circuit strength, phase-jump response, voltage support, and restoration capability rather than treating reserve as a single MW quantity.
3. Current limiting can change GFM output impedance, synchronization behavior, and large-signal stability. A headroom-only reserve model is therefore insufficient for a 100% IBR system.
4. Dynamic operating envelopes are established mainly for steady-state distribution-network export/import limits. Extending the envelope concept to frequency-voltage-current dynamic security is a plausible research gap.
5. A coherent story is to treat the VPP as a security-bearing dynamic entity after the last synchronous machine is disconnected. The research object becomes a conservative dynamic-security capability envelope that maps an operating state to the set of ancillary-service commitments that remain deliverable under contingencies.

## Selected primary sources

- IEEE Transactions on Smart Grid, "Virtual Power Plants for Frequency Regulation: A Learning-Based Method With Safety Guarantee," published online 2025, DOI 10.1109/TSG.2025.3618896: https://ieeexplore.ieee.org/document/11195783/
- IEEE IAS 2024, "Dynamic Virtual Power Plants with Frequency Regulation Capacity," DOI 10.1109/IAS55788.2024.11023659: https://ieeexplore.ieee.org/document/11023659/
- He, Duarte, Haeberle, and Doerfler, "Grid-Forming Control of Modular Dynamic Virtual Power Plants," 2024: https://arxiv.org/abs/2410.14912
- IEEE Transactions on Power Systems, "Unified Model of Current-Limiting Grid-Forming Inverters for Large-Signal Analysis," published online 2025, DOI 10.1109/TPWRS.2025.3587224: https://ieeexplore.ieee.org/document/11077947/
- IEEE Transactions on Energy Conversion, "Modeling Fault Recovery and Transient Stability of Grid-Forming Converters Equipped With Current Reference Limitation," DOI 10.1109/TEC.2024.3507544: https://ieeexplore.ieee.org/abstract/document/10769568
- Chu and Teng, "Stability Constrained Optimization in High IBR-Penetrated Power Systems—Part II," 2023: https://arxiv.org/abs/2307.12156
- Wang and Geng, "Decentralized Stability-Constrained Optimal Power Flow for Inverter-Based Power Systems," 2026: https://arxiv.org/abs/2604.17603
- ENTSO-E, Phase II Technical Report on Grid Forming Requirements, 2025: https://www.entsoe.eu/news/2025/11/04/entso-e-publishes-phase-ii-technical-report-on-grid-forming-requirements/
- AEMO, Engineering Roadmap Execution Reports and GFM test frameworks: https://www.aemo.com.au/initiatives/major-programs/engineering-roadmap/engineering-roadmap-execution-reports
- AEMO, GFM Technology Access Standards Technical Requirements Review, 2025–2026: https://www.aemo.com.au/consultations/current-and-closed-consultations/grid-forming-technology-access-standards-technical-requirements-review
- NESO, Stability Market: https://www.neso.energy/industry-information/balancing-services/stability-market
- NESO, first grid-forming battery projects under Stability Pathfinder Phase 2, 2025: https://www.neso.energy/great-britains-first-grid-forming-battery-connects-scotland

## Proposed positioning

Working title: "Dynamic-Security Operating Envelopes for Virtual Power Plants in 100% Inverter-Based Islanded Power Systems."

Central thesis: after the last synchronous machine is disconnected, a VPP is no longer only an energy aggregator. It becomes part of the grid-forming and stability infrastructure. Consequently, an offered megawatt of reserve is deliverable only if energy, active-power headroom, converter current, voltage recovery, synchronization, topology, and control mode remain jointly feasible.

The proposed novelty is capability certification rather than controller synthesis: construct and validate a conservative inner approximation of the VPP's secure ancillary-service set, then embed it into system scheduling or service procurement.
