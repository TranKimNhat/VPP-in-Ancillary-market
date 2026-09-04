# Peer Review Report — Multi-Perspective Panel Review
**Manuscript:** "Graph Learning for Topology-Adaptive Fast Frequency Response in Inverter-Based Microgrids"
**Review date:** 2026-06-10 · **Mode:** full (5-reviewer panel) · **Reviewer skill:** academic-paper-reviewer v1.10.0

---

## Phase 0 — Field Analysis & Reviewer Configuration

| Attribute | Assessment |
|---|---|
| Primary discipline | Power systems — frequency control in inverter-based microgrids |
| Secondary discipline | Machine learning — graph neural networks, multi-agent RL |
| Paradigm | Quantitative simulation study (controller design + zero-shot generalization evaluation) |
| Target venue tier | IEEE Transactions (Smart Grid / Power Systems / Sustainable Energy class) |
| Maturity | **Pre-submission draft** — contains [TBD] entries and unresolved citations "[?]" |

**Panel:**
- **EIC** — Editor, IEEE Transactions on Smart Grid; expertise in DER coordination and grid-edge ML.
- **R1 (Methodology)** — RL-for-power-systems researcher; focus on statistical rigor, reproducibility, baseline fairness.
- **R2 (Domain)** — Microgrid frequency-dynamics expert (GFM/GFL control, model reduction); focus on physical modeling validity and standards.
- **R3 (Perspective)** — Electricity-markets + distributed-computing researcher; focus on deployment realism, communication architecture, economic layer.
- **DA (Devil's Advocate)** — Challenges the core claims: safety, decentralization, zero-shot generalization.

---

## Reviewer 0 — Editor-in-Chief Assessment

**Summary.** The paper proposes an inductive GraphSAGE encoder + parameter-shared MAPPO controller emitting a joint power-reference/droop-gain action per DER, evaluated zero-shot on 24 unseen reconfigurations of an islanded IEEE 123-bus feeder. Fit to a smart-grid/power-systems ML venue is excellent; the problem (topology-adaptive FFR under 100% IBR) is timely and the zero-shot topology hold-out protocol is a genuinely good evaluation idea.

**Originality.** The authors candidly disclaim novelty of GraphSAGE and of GraphSAGE+RL, locating the contribution in (i) the topology-coupled reduced-order frequency model A_f(G_t,K_t), (ii) the joint (a^P, a^K) action with a shared safety envelope, and (iii) the zero-shot hold-out evaluation. This is a fair and defensible framing — incremental in components, novel in combination and in evaluation discipline.

**Readiness — not submittable in current state:**
1. **Table V contains "[TBD]" entries** (entropy coefficient, update epochs/minibatch). A manuscript with placeholder hyperparameters cannot enter review.
2. **Two unresolved citations "[?]"** in Section II-B-5 (SFR-modelling claim and topology-shapes-COI claim).
3. **Reference list quality**: refs [11], [19], [27], [28], [31] have no authors; [9] is dated 2026 with no venue; [14], [15] have malformed volume fields ("vol. Volume 12 - 2024"); the IEEE Std 1547 ride-through claim on p.7 has no corresponding reference entry.
4. **Internal numerical contradictions between text and figures** (detailed by R1 and DA below) would, if spotted at screening, trigger an immediate return to authors.

**Significance if fixed:** High. The "train on one topology, evaluate zero-shot on 24" protocol with an encoder-only ablation is exactly the kind of evaluation discipline the MARL-for-grids literature lacks.

**Scores:** Fit 9/10 · Originality 7/10 · Significance 7/10 · Readiness 3/10

---

## Reviewer 1 — Methodology Review

**Summary.** The experimental design has good bones — single-topology training, 24-topology disjoint hold-out, encoder-only ablation under an identical learning stack, paired Wilcoxon test, ASHA-tuned shared hyperparameters. But the execution as reported has serious gaps in statistical reporting, reproducibility, and internal consistency.

**Strengths.**
- S1. The encoder-only ablation (MLP-MAPPO, identical stack) is the correct isolation experiment and is honestly framed (GCNN-PPO is acknowledged as a confounded comparison, Sec. IV-B).
- S2. ASHA-tuned hyperparameters shared across all learning baselines removes a common unfairness (capacity/schedule advantage).
- S3. Paired Wilcoxon signed-rank across the 24 hold-out topologies, with the "below next-best on every topology" claim, is an appropriately strong nonparametric result.

**Weaknesses (what / where / fix).**
- W1. **CRITICAL — Text–figure contradictions.** Sec. V claims "the shared safety envelope holds the post-event nadir at the 49.5 Hz alarm for every learning controller" and "caps the over-frequency zenith at 50.5 Hz in S4." Fig. 5 reports nadirs of 49.22 (S1, GraphSAGE), 48.97 (S1, GCNN), and 48.35–48.76 Hz for all learning controllers in S2 — below the 49.5 alarm and, in S2, below the 49.0 Hz UFLS-1 trip — and S4 zeniths of 51.53–51.93 Hz. Either the figures are from a pre-safety-layer run or the claims are wrong; both the "unit FFR success rate" claim and the nadir-projection layer description (δ = 0.5 Hz band) are contradicted by the plotted data. Fix: regenerate all figures and prose from one frozen result set; state explicitly which runs include the projection layer.
- W2. **CRITICAL — Text–text contradiction.** Sec. V intro says fixed droop reaches 48.69 Hz on S2; Fig. 5 says 47.56 Hz; Sec. V-A then computes the 0.81 Hz nadir gain consistent with 47.56. The 48.69/48.43 numbers and the 47.56 numbers cannot both be right.
- W3. **MAJOR — Seeds asymmetry.** Only the proposed controller is stated to be trained from three seeds; baselines apparently use one. Report all methods at ≥3 seeds (or justify), and report seed-level dispersion for baselines — otherwise the "smallest run-to-run spread" claim (σ ≤ 0.005 Hz·s, vs 0.15–6.5) is not a like-for-like comparison. σ ≤ 0.005 Hz·s is also implausibly tight; clarify whether this is per-episode, per-topology, or per-seed dispersion.
- W4. **MAJOR — Reproducibility.** No code/data availability statement; reward weights promised "in Section IV" never appear; Table V has [TBD]s; the supplementary material is cited ~8 times for load-bearing content (device recursions, Kron transfer, Lyapunov derivation, PPO losses, secondary loop) but was not provided. A reviewer cannot verify the certificate or the model.
- W5. **MAJOR — THD audit methodology unspecified.** Table VI is produced at "a representative 50%-rated dispatch with dominant harmonics h ∈ {5,7,11,13}" — one operating point, no harmonic-injection model, no propagation method stated. As reported, the harmonic claim is unverifiable; describe the harmonic source model and sweep operating points, or demote the claim.
- W6. **MINOR — Multiple comparisons.** Several headline deltas (IAE, nadir, settling, cost) are tested or asserted; only one Wilcoxon test is reported. State which comparisons were tested and whether any correction was applied.
- W7. **MINOR — Table III/Fig. 5 sign inconsistency.** S3 line trip is +2.4 MW in Table III, −2.4 MW in Fig. 5.
- W8. **MINOR — "Converged performance" undefined.** State the convergence criterion and evaluation checkpoint selection rule (risk of implicit test-set selection).

**Scores:** Design 7/10 · Statistical validity 4/10 · Reproducibility 3/10 · Internal consistency 2/10

---

## Reviewer 2 — Domain Review (Power Systems)

**Summary.** The modeling chapter is the paper's intellectual core and is mostly sound for a small-signal study, with two physical-validity questions that must be addressed and several standards-handling errors.

**Strengths.**
- S1. Retaining inter-unit angle structure and letting topology enter via the Kron-reduced Jacobian J̃_r(G_t), with COI as a read-out rather than the state, is the right repair of the reconfiguration-invariant COI models common in this literature.
- S2. The located-disturbance input map B_net(G_t) is a real improvement — most MARL-FFR papers spread the imbalance uniformly.
- S3. The RoCoF discussion is physically honest: peak RoCoF is backbone-governed and correctly excluded from the GFL reward.

**Weaknesses (what / where / fix).**
- W1. **MAJOR — Three-phase unbalance vs. Kron reduction.** The IEEE 123-bus feeder is a canonically *unbalanced* three-phase network, yet Eq. (7) Kron-reduces a single active-power Jacobian, implying a balanced positive-sequence model. State explicitly how the unbalanced feeder is mapped to the per-unit single-phase Jacobian (per-phase aggregation? positive-sequence equivalent?) and what error this introduces — especially since the THD section then treats the same feeder as a harmonic network.
- W2. **MAJOR — Δt = 1 s vs. FFR timescale.** With a 1-s control step and Fig. 5's "FFR (0–2 s)" window, the nadir of a severe trip in a H_sys = 1.34 s system occurs within roughly the first control step — before the policy can react to the event it is rewarded for arresting. The paper needs an explicit account of what the learned action can causally influence (post-nadir recovery and settling, pre-positioning via a^P) versus what it cannot (first-swing nadir), and the nadir-improvement claims should be re-expressed accordingly. The pre-positioning argument (Sec. III-A-2) hints at this but is never demonstrated — show one trajectory of anticipatory storage positioning.
- W3. **MAJOR — Standards table conflates jurisdictions.** Table II mixes ENTSO-E SO GL quantities (50 Hz, FCR ±200 mHz) with IEEE Std 1547/519 (North American). The 2.0 Hz/s ride-through is attributed to [38][39] in Table II but to "IEEE Std 1547" in Sec. V-A, and 1547 is absent from the references. Fig. 5's legend labels 49.5 Hz as "UFLS" while Table II defines 49.5 Hz as the alarm and 49.0 Hz as UFLS-1. Clean up: one jurisdictional frame, consistent threshold names, correct citations.
- W4. **MAJOR — Zone count inconsistency.** Table I: |Z| = 34 zones for 41 agents (≈1.2 agents/zone — why zone at all?); Figs. 1–3 show 4 zones. Reconcile; the VPP-zone construct currently does no observable work in the method.
- W5. **MINOR — Literature.** Missing engagement with system-frequency-response/FFR-sizing literature for IBR-dominated islands and with safe-RL-for-frequency work beyond [34][35][37]; refs [14][15] (the closest graph-RL LFC papers) deserve a substantive comparison paragraph, not just baseline status for [14].
- W6. **MINOR — GFL inner loops.** First-order envelope at Δt = 1 s is defensible, but the claimed harmonic injections (Sec. V-D) originate precisely in the dynamics the model discards; acknowledge this tension.

**Scores:** Physical modeling 6/10 · Standards handling 3/10 · Literature 6/10 · Domain contribution 7/10

---

## Reviewer 3 — Perspective Review (Markets / Deployment / Cross-Disciplinary)

**Summary.** The paper's framing as a Layer-2 price-taker inside a DSO–VPP–DER hierarchy is attractive but mostly decorative in the present draft, and the deployment story has an unexamined communication assumption at its core.

**Strengths.**
- S1. The three-layer separation (DSO SOCP-OPF → VPP DRO offer → fast execution) is a credible institutional embedding that most FFR-RL papers lack entirely.
- S2. The DSO cost-per-event metric (capacity payment vs. load-shedding penalty) is a good translation of frequency security into operator-relevant economics.
- S3. Honest scoping: no price impact, no closed-loop re-clearing, price-making left to future work.

**Weaknesses (what / where / fix).**
- W1. **MAJOR — "Decentralized execution" vs. a 2-layer GNN over the full feeder.** Sec. III-A states obs_full ∈ R^(123×20) including per-bus voltages, loads, and agent density; the GraphSAGE encoder aggregates 2-hop neighborhoods over the active 123-bus graph at every 1-s step, and the edge index is "rebuilt at every step from the active graph G_t." Who measures, communicates, and assembles this? Either (a) each agent needs 2-hop neighbor states plus DSO topology telemetry every second — a real but unstated communication architecture — or (b) a central node computes embeddings, in which case execution is not decentralized and the CTDE framing is misleading. State the execution-time information architecture explicitly, and ideally evaluate sensitivity to the delays/dropouts the conclusion already concedes are untested.
- W2. **MAJOR — Economic layer is asserted, not exercised.** The DLMP/SOCP-OPF and Wasserstein-DRO layers (Sec. II-C) never produce a number in the paper; the "common price deck" behind Fig. 9 is uncited and "cost units" are dimensionless. Either give the price deck (sources, €/MW values, VOLL) and make costs interpretable, or compress Sec. II-C to a paragraph and reframe Fig. 9 as a shedding-avoidance result.
- W3. **MINOR — Scalability and runtime.** No inference-time or training-cost numbers. For a 1-s loop the per-step latency of encoder+actor matters; one table row would suffice.
- W4. **MINOR — Fig. 4 caption mismatch.** Caption mentions a "DSO-cost axis"; the radar lists five axes without it.
- W5. **MINOR — Transferability beyond this feeder.** The inductive-encoder argument implies cross-feeder transfer; a single experiment on a second feeder (even small) would substantially raise impact. (Suggestion, not a requirement.)

**Scores:** Practical relevance 7/10 · Deployment realism 4/10 · Cross-disciplinary rigor 5/10

---

## Devil's Advocate Report

**Strongest counter-argument (the case against the paper as written).**
The paper's headline safety claim is contradicted by its own figures. The abstract and Section V assert that a shared safe-by-construction envelope — monotone gains, a Lyapunov certificate, and a closed-form nadir projection holding COI deviation within ±0.5 Hz — yields a unit FFR success rate for every learning controller. Figure 5 shows every learning controller breaching the 49.5 Hz band in S1–S3, breaching the 49.0 Hz UFLS-1 threshold in S2 (proposed: 48.35 Hz — the *worst* nadir among the four learning controllers), and exceeding 51.5 Hz in S4 against a claimed 50.5 Hz cap. If the figures are correct, the safety envelope does not do what the paper says it does and the FFR-success metric is false; if the prose is correct, every frequency figure is stale. Either way, the central empirical narrative — "safety is common, the discriminators are error and settling" — is built on data the paper itself contradicts. Compounding this, the abstract's generalization claim ("10–22% degradation for fixed-adjacency *and non-graph* encoders") is falsified by Fig. 7, where the non-graph MLP-MAPPO degrades by only +0.4% — the text quietly reclassifies this as "an artifact of an already-poor baseline," meaning the correct statement is that MLP is *uniformly worse*, not that it *degrades more*. A camera-ready abstract that survives only if the reader doesn't check Fig. 5 and Fig. 7 is not yet a publishable abstract.

**Issue list.**
- **CRITICAL (consistency · Abstract, Sec. V, Figs. 5–6):** Nadir/zenith/FFR-success prose contradicts Fig. 5 values; nadir-projection (±0.5 Hz) contradicted by plotted excursions up to 1.65 Hz. One result set must be regenerated and every number in abstract/text/figures reconciled.
- **CRITICAL (claim accuracy · Abstract vs. Fig. 7):** "10–22% degradation for fixed-adjacency and non-graph encoders" — the non-graph encoder (MLP) shows +0.4%. The honest claim is "lowest unseen IAE, 12–25% below all learning baselines, with 30× lower across-topology dispersion than MLP"; the degradation-percentage framing only holds for MATD3/GCNN.
- **MAJOR (decentralization · Sec. III):** CTDE "execution from local observation alone" vs. an encoder that consumes the full 123-bus graph each second (see R3-W1). The core deployability premise is unproven as stated.
- **MAJOR (selective emphasis · Sec. V):** The proposed method is best on IAE/settling/consistency but *worst-of-four* on S2 nadir and second-best on S3 nadir and THD (GCNN wins both). The radar's "largest enclosed area" framing and axis normalization smooth this over. State per-metric losses plainly.
- **MAJOR (causality · Sec. V-B):** The "single mechanism" discussion attributes everything to inductive re-aggregation, but the +1.3% vs +0.4% gap comparison shows topology *gap* doesn't separate graph from non-graph methods — absolute level and dispersion do. Tighten the causal story to what the ablation actually supports.
- **MINOR (framing · Sec. V-A):** "1.4–1.9× faster settling than learning baselines" — fixed droop settles comparably fast (admitted in text); against deployed practice the advantage is nadir/IAE, not speed. Say so.
- **MINOR (scope · Sec. IV-A):** Zero-shot hold-out spans Jaccard edge distance ≤ 0.18 — honestly disclosed, but the abstract's unqualified "24 unseen reconfigurations" invites over-reading; carry the scope qualifier into the abstract or conclusion.

**Ignored alternatives.** A per-topology re-tuned droop (oracle gain schedule) baseline would test whether *learning* is needed at all, or merely topology-indexed gain tables; none of the six baselines occupies this cell.

**Missing stakeholder.** The DSO as safety authority: if the nadir projection is a hard actuator-side guard, who certifies and audits it in deployment? The paper treats safety as an environment property — convenient for fair comparison, but it erases the question of who owns the guard.

**"So what?" test: PASS (conditionally).** If the consistency crisis is resolved in favor of the figures-with-projection, the result — one controller trained on one topology that holds its performance within ~2% across 24 reconfigurations while baselines degrade or disperse — is a genuinely useful finding for DSO-reconfigurable feeders.

---

## Phase 2 — Editorial Decision

**Decision: MAJOR REVISION** (Devil's Advocate CRITICAL findings bar acceptance; the underlying contribution and evaluation protocol merit a revision invitation rather than rejection.)

**Consensus across panel (≥3 reviewers):**
1. Text–figure–abstract numerical contradictions on the safety/nadir story (R1-W1/W2, R2-W3 partially, DA-C1) — the single highest-priority item.
2. Manuscript incompleteness: [TBD] hyperparameters, "[?]" citations, malformed references, missing supplementary material (EIC, R1).
3. Decentralized-execution claim vs. global-graph encoder (R3-W1, DA-M1).
4. Statistical reporting: seed asymmetry, dispersion definition, untested comparisons (R1-W3/W6).

**Disagreement and arbitration:**
- R2 rates the modeling contribution highly; DA challenges the causal narrative. Arbitration: both stand — the model is a real contribution *and* the discussion overclaims a "single mechanism"; revise the Discussion, keep Section II.
- R3 would cut Section II-C; EIC values the institutional framing. Arbitration: keep but either instrument it (real price deck) or compress it and soften Fig. 9's framing.

## Revision Roadmap (prioritized)

| # | Priority | Action | Addresses |
|---|---|---|---|
| 1 | BLOCKING | Regenerate all results from one frozen run set; reconcile every number in abstract, Sec. V, Figs. 5–9, Tables. Explicitly state whether the nadir-projection layer was active, and report both with/without if illuminating. | DA-C1, R1-W1/W2 |
| 2 | BLOCKING | Rewrite the abstract's generalization sentence to match Fig. 7 (level + dispersion, not blanket degradation %); carry the Jaccard-scope qualifier. | DA-C2, DA-m2 |
| 3 | BLOCKING | Fill Table V [TBD]s; resolve "[?]" citations; repair refs [9],[11],[14],[15],[19],[27],[28],[31]; add IEEE 1547; attach supplementary material; add reward weights and a code/data statement. | EIC, R1-W4 |
| 4 | HIGH | Specify the execution-time information architecture (who computes embeddings, what is communicated per second); reframe or defend "decentralized execution." | R3-W1, DA-M1 |
| 5 | HIGH | Equalize seeds across methods (≥3) and define the dispersion statistic; clarify the σ ≤ 0.005 Hz·s claim. | R1-W3 |
| 6 | HIGH | Add the unbalance→positive-sequence mapping for the 123-bus Kron reduction and a paragraph on Δt = 1 s vs. first-swing nadir causality (what the policy can and cannot influence); demonstrate pre-positioning once. | R2-W1/W2 |
| 7 | MEDIUM | Fix standards table (jurisdiction, threshold names, Fig. 5 legend "UFLS 49.5"); fix S3 sign; fix zone count (34 vs 4); fix Fig. 4 caption. | R2-W3/W4, R1-W7, R3-W4 |
| 8 | MEDIUM | Document the THD methodology (source model, propagation, operating-point sweep) or demote the compliance claim. | R1-W5 |
| 9 | MEDIUM | Specify the price deck behind Fig. 9 or compress Sec. II-C; add inference-latency row. | R3-W2/W3 |
| 10 | OPTIONAL | Add a per-topology re-tuned droop (oracle) baseline; a second-feeder transfer experiment. | DA-alt, R3-W5 |

*This report is advisory output of a simulated review panel; the manuscript file was not modified.*
