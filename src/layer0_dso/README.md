# Layer-0 reconfiguration implementations

`src/layer0_dso/reconfiguration.py` is the canonical Pyomo/MISOCP formulation and should be treated as the ground-truth research model for equivalence checks.

`src/opt/l0_reconfig.py` is the CVXPY-based production path used by precompute/offline generation. It should match the canonical path on feasibility and downstream voltage behavior, but exact switch choices may differ because solver tie-breaking can select different equivalent radial topologies.

`src/opt/tie_switch_reconfig.py` is a fast heuristic baseline for RL environment sampling. It is not expected to reproduce the MISOCP optimum exactly; it is accepted when it remains connected, power-flow feasible, and close on downstream voltage invariants.

Use `tests/test_reconfig_equivalence.py` to generate `artifacts/reconfig_equivalence_report.json` with Hamming distance and voltage-band metrics for reviewer-facing evidence. The MATPOWER fixture raises placeholder `max_i_ka` ratings to at least 2.0 kA so equivalence is measured on topology behavior rather than an already-overloaded base snapshot.
