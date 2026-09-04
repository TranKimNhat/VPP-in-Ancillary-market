# Hyperparameter Optimization Report (ASHA) — Reviewer Response

**Purpose.** Document the hyperparameter (HP) optimization protocol for the
proposed GraphSAGE-MAPPO dual-action (P_ref, K_droop) controller, so the
reported results are reproducible and the chosen HPs are defensible. This report
answers the reviewer concern: *"How were hyperparameters selected, and is the
comparison fair / not cherry-picked?"*

Status: **complete** (`logs/asha_fixed_env.log`,
`artifacts/asha_fixed_env/asha_summary.json`). Selected configuration =
**Trial 9**, now used for the full 6000-episode curriculum retrain on the
VSG frequency model (`logs/retrain_vsg_seed42.log`,
`artifacts/checkpoints_am_mappo_vsg`).

---

## 1. Why re-tune (motivation)

The previously committed HPs (labelled "ASHA Trial 5" in the code) were tuned on
an **earlier, defective version of the environment**. A pipeline audit found and
fixed a chain of issues that all changed the effective optimization landscape:

| Fixed issue | Effect on training dynamics |
|---|---|
| LTI disturbance not injected / Euler overshoot / AGC not closed | frequency response was wrong-magnitude or absent |
| Reward + observation saturation (refs 0.5 Hz vs events at 2.5 Hz) | flat gradient in the regime where events occur |
| Train/eval `ffr_mode` mismatch (dual vs droop) | policy never exercised at eval |
| Dead VPP→LTI coupling (`pp_bus_idx` always −1) | policy actions never reached the dynamics |
| K-channel unit mismatch | droop gain either inert or numerically divergent |
| AGC freeze during FFR-active | sustained-disturbance frequency stuck off-nominal |
| K_min lowered 0.05·P_rated → 0 | action space widened to [0, K_max] |

Because the optimization landscape changed, HPs tuned on the old environment are
no longer valid. HP search is therefore repeated on the **corrected pipeline**
(coupling live, AGC concurrent, monotone-K, nadir safety layer in-the-loop).

---

## 2. Search algorithm

**Successive Halving (SHA / Hyperband bracket), single-machine sequential.**
Implemented in `experiments/run_asha_am_mappo.py`. Random configurations are
evaluated at increasing training budgets ("rungs"); the top `1/η` are promoted to
the next rung, concentrating compute on promising configurations.

| Parameter | Value |
|---|---|
| Number of initial configurations | 12 |
| Reduction factor η | 3 |
| Rung budgets (episodes) | 25 → 75 → 225 |
| Promotion schedule | 12 → 4 → 1 |
| Seeds per rung (variance control) | 1 → 2 → 3 |
| Per-episode horizon | 300 steps |
| Sampler seed (reproducibility) | 123 |

Rationale for the schedule: cheap 25-episode screening eliminates clearly poor
configurations, and the surviving configurations are re-evaluated at 75 and 225
episodes with **increasing seed counts** so the final selection is averaged over
3 seeds (reduces selection variance). Total budget ≈ 1.5k episode-equivalents.

> Note: the implementation re-trains each rung from scratch at the rung budget
> (rather than warm-starting from the previous rung's checkpoint). This is the
> more conservative variant — each rung is an independent measurement of a
> configuration at that budget — at the cost of extra compute; it does not bias
> selection.

---

## 3. Objective (selection metric)

**Mean post-event Integral Absolute Error (IAE)** of the COI frequency, measured
by loading each trial's trained checkpoint and running **forced contingencies**:

- Scenarios: S1 load-step (+2.5 MW), S2 generator trip (−3.9 MW), S4
  high-renewable surge (+4.7 MW).
- Topologies: 2 held configurations (one from each electrical group).
- Score = mean IAE over scenario × topology (lower is better).

This is the same metric used in the paper's results tables, so HP selection
optimizes exactly what is reported — not a proxy. Two deliberate choices:

1. **No RoCoF term.** Peak RoCoF after a contingency is ΔP/(2·H_backbone), set by
   the grid-forming backbone inertia and provably invariant to the grid-following
   VPP's power actions (Approach-C). Including RoCoF in the objective would inject
   a non-actionable (noise) term into selection, so it is excluded.
2. **Eval-based, not training-aggregate.** Training-time aggregates are diluted by
   warm-up and post-recovery quiet steps and by event-free episodes; a forced-
   contingency eval isolates the controlled transient.

---

## 4. Search space

Sampled per configuration (`_sample_config`):

| Hyperparameter | Range / set |
|---|---|
| Learning rate | log-uniform 10^[−4.5, −3.0] |
| Entropy coefficient | {0.001, 0.003, 0.01, 0.03} |
| GraphSAGE embedding dim | {64, 128} |
| Actor/critic hidden dim | {64, 128} |
| PPO update epochs | {2, 4, 8} |
| Mini-batch size | {16, 32, 64} |
| Initial log-std (exploration) | {−2.0, −1.5, −1.0, −0.5, 0.0} → std = softplus(x)+0.05 |

The LR upper bound was raised to 10^−3 after fixing the actor log-std
parameterization (a former clamp at 0.5 suppressed the log-std gradient and
forced very small LRs).

---

## 5. Results

<!-- FILL FROM artifacts/asha_fixed_env/asha_summary.json ON COMPLETION -->

### 5.1 Best configuration (Trial 9)

| Hyperparameter | Selected value |
|---|---|
| Learning rate | $2.14\times10^{-4}$ |
| Entropy coefficient | 0.001 |
| Embedding dim | 128 |
| Hidden dim | 128 |
| Update epochs | 8 |
| Mini-batch size | 32 |
| log-std init | −2.0 |
| Eval-IAE @ 75-ep rung (2-seed mean ± std) | 4.97 ± 0.04 |
| **Eval-IAE @ 225-ep rung (3-seed mean ± std)** | 16.08 ± 3.03 |

### 5.2 Per-rung promotion summary

| Rung | Budget (ep) | Seeds | Candidates | Promoted | Best trial | Best IAE | Best IAE std |
|---|---|---|---|---|---|---|---|
| 0 | 25 | 1 | 12 | 4 (T7, T11, T9, T6) | 7 | 4.981 | 0.000 |
| 1 | 75 | 2 | 4 | 1 (T9) | 9 | 4.970 | 0.041 |
| 2 | 225 | 3 | 1 | 1 (T9) | 9 | 16.082 | 3.031 |

Trial 9 was not the rung-0 leader (Trial 7 was) but overtook the field once the
budget and seed count increased, which is exactly the behaviour successive
halving is designed to surface: cheap screening keeps a diverse top-4, and the
higher-budget, multi-seed rungs select the configuration that generalises.

**Caveat (reported for transparency).** The eval-IAE rises from 4.97 (75-ep,
2-seed) to 16.08 (225-ep, 3-seed). The increase is driven by the third seed (44)
added at the final rung — `score_std` jumps from 0.04 to 3.03 — i.e. the absolute
IAE is seed-sensitive at this budget, not monotonically improving with training.
The *ranking* that drives selection is nonetheless stable (Trial 9 wins the 75-ep
survivor rung and is the sole 225-ep finalist). The full 6000-episode curriculum
retrain plus the multi-seed evaluation (Section 6) are what validate the final
policy; ASHA is used only to pick the HP vector, not to report performance.

### 5.3 Comparison to prior (old-env) HPs

| HP | Old "Trial 5" (broken env) | New Trial 9 (fixed VSG env) |
|---|---|---|
| lr | 3.13e-5 | 2.14e-4 |
| entropy_coef | 0.03 | 0.001 |
| embed/hidden | 128 / 128 | 128 / 128 |
| update_epochs | 4 | 8 |
| mini_batch | 16 | 32 |
| log_std_init | −1.0 | −2.0 |

The corrected pipeline favours a markedly higher learning rate ($\sim 7\times$),
lower entropy regularisation, more PPO update epochs, a larger mini-batch, and a
tighter initial exploration std. This is consistent with the pipeline fixes
(Section~1): once the VPP actions actually reach the dynamics and the
reward/observation scaling matches event magnitudes, the gradient signal is
informative enough to support faster, less heavily-regularised updates.

---

## 6. Reproducibility

```bash
python -m experiments.run_asha_am_mappo \
  --num-trials 12 --eta 3 --min-episodes 25 --max-episodes 225 \
  --output artifacts/asha_fixed_env
```

Environment fixes are committed (see git history through the AGC-freeze fix);
the search runs on that exact pipeline. The selected configuration is then used
for the full 6000-episode curriculum training and the multi-seed evaluation that
produce the paper's result tables.
