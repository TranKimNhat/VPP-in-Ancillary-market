# Paper Protocol Gate

Mục tiêu: đảm bảo mọi thay đổi thuật toán được báo cáo với độ tin cậy thống kê trước khi đưa vào paper.

## Gate rules

1. Ablation screening: tối thiểu N=3 seeds.
2. Main results: tối thiểu N=5 seeds.
3. Seed list lấy từ `configs/seeds.yaml`, không cherry-pick.
4. Báo cáo theo format `mean ± std`.
5. Khi so method, báo thêm p-value từ Welch's t-test.

## Checklist trước khi report kết quả

- [ ] Chạy `experiments/run_multi_seed.py` với seed list chuẩn.
- [ ] Có file JSON output trong `artifacts/multi_seed/`.
- [ ] File JSON có `seeds`, `individual`, `mean`, `std`, `n_success`.
- [ ] Chạy `scripts/compare_methods.py` cho baseline vs candidate.
- [ ] Ghi bảng so sánh (mean, std, N, p-value) vào appendix.

## Pre-registered comparison setup (before viewing real runs)

- Baseline config: `configs/training_config_baseline.yaml`.
- Method config: `configs/training_config_method.yaml`.
- Primary comparison metric: `final_reward` from per-seed train result JSON.
- Screening stage: N=3 seeds per config (first 3 seeds in `configs/seeds.yaml`).

### Fixed decision rule to scale from N=3 → N=5

Use **effect-size-first** rule on screening outputs:

- Compute Cohen's d: `(mean_method - mean_baseline) / pooled_std`.
- Scale to N=5 only if `d >= 0.5` and `mean_method > mean_baseline`.
- If rule is not met, stop at N=3 and record a negative result.

This decision rule is fixed in advance and is not changed after observing screening outcomes.

### Compute timebox

Let `T_seed` be measured wall-clock from one real dry run.

- Screening budget cap: `6 × T_seed` (3 seeds × 2 configs).
- Full N=5 additional budget cap: `4 × T_seed` (2 more seeds × 2 configs).
- Hard stop if any stage exceeds its cap by >20% or if `n_success < N` due to failures.

## Screening log (2026-04-24)

Applied the pre-registered N=3 screening rule with seeds `[42, 1337, 2024]`.

- Baseline (`configs/training_config_baseline.yaml`): mean = -1556.1627, std = 71.8497, n=3.
- Method (`configs/training_config_method.yaml`): mean = -1547.2594, std = 71.3172, n=3.
- Cohen's d (pooled): 0.1016.
- Decision: **STOP at N=3** (direction positive but `d < 0.5`, so do not scale to N=5).

Artifacts:
- `artifacts/multi_seed/screening_baseline_n3.json`
- `artifacts/multi_seed/screening_method_n3.json`

This is logged as a negative/non-actionable screening result for the current primary metric (`final_reward`) and is not overridden post hoc.

## Reproducibility rerun log (2026-04-25)

A full rerun was executed with the same pre-registered seeds `[42, 1337, 2024]` after hardening per-arm/per-seed artifact isolation.

- Baseline rerun (`artifacts/multi_seed/papergrade_baseline_n3.json`) is bit-identical to screening on `final_reward`.
- Method rerun (`artifacts/multi_seed/papergrade_method_n3.json`) is bit-identical to screening on `final_reward`.

Interpretation:

- This confirms seed-deterministic reproducibility of the current training pipeline.
- This does **not** constitute additional sampling; effective sample size remains **N=3 per arm**.
- The Phase D STOP decision on pre-registered `final_reward` remains unchanged.

## Exploratory note (non pre-registered)

From the reproducibility rerun outputs:

- `mean_eval_reward` effect size is small (~0.11).
- `best_eval_reward` effect size is also small but larger than `final_reward` (~0.27).

These metrics were not pre-registered primary endpoints for Phase D, so they are logged only as exploratory signals and do not reverse the STOP decision.

## Training-length caveat (Phase D)

Current Phase D runs use `updates=20` and `n_eval_points=4` per seed. Reward traces across seeds show mixed/unstable late-stage trend (no consistent plateau), so asymptotic behavior is unverified at this training length.

## Phase D.5 pre-registration (2026-04-25)

To test whether the exploratory signal survives a clean protocol:

- Baseline config: `configs/training_config_baseline.yaml`.
- Method config: `configs/training_config_method.yaml`.
- Seed list: `configs/seeds_d5.yaml` with fresh seeds `[777, 3141, 2718]` (no reuse of `[42, 1337, 2024]`).
- Primary metric (D.5 only): `best_eval_reward` from per-seed train result JSON.
- Screening stage: N=3 per config.
- Fixed decision rule to scale D.5 from N=3 to N=5: scale only if direction is positive and `d >= 0.3` on D.5 primary metric.

Artifacts to populate:
- `artifacts/multi_seed/d5_baseline_n3.json`
- `artifacts/multi_seed/d5_method_n3.json`

## Phase D.5 screening log (2026-04-25)

Applied D.5 pre-registered N=3 rule with fresh seeds `[777, 3141, 2718]`.

- Baseline (`configs/training_config_baseline.yaml`): `best_eval_reward` mean = -1562.7790, std = 48.1140, n=3.
- Method (`configs/training_config_method.yaml`): `best_eval_reward` mean = -1562.4847, std = 45.9444, n=3.
- D.5 primary effect (Cohen's d, pooled): 0.0063.
- Direction: positive but negligible (`Δ = +0.2943`).
- Decision under fixed D.5 rule (`d >= 0.3` and positive direction): **STOP at N=3** (do not scale to N=5).

Supportive stats (exploratory):

- Welch t-test on D.5 primary metric: `t = 0.0077`, `p = 0.9943`.
- `mean_eval_reward` effect size: `d = 0.0763`.
- `final_reward` effect size: `d = 0.1116`.

## Path A → Path B execution log (2026-04-27)

Decision matrix (pre-registered, append-only):

| Branch | Gate rule | Result | Outcome |
|--------|-----------|--------|---------|
| Path A | `final_reward`, d ≥ 0.5, direction positive, N=3 seeds [42, 1337, 2024] | d = 0.1016, STOP | Frozen as negative result; proceed to Path B |
| Path B | `best_eval_reward`, d ≥ 0.3, direction positive, N=3 seeds [777, 3141, 2718] | d = 0.0063, STOP | STOP — no scale-up |

Path B outcome confirms D.5 screening result (bit-identical; same seeds/configs).

**Final decision: STOP at N=3 on both paths. No scale-up to N=5 or full ablation is justified under pre-registered protocol.**

Artifacts:
- `artifacts/multi_seed/path_a_baseline_n3.json`
- `artifacts/multi_seed/path_a_method_n3.json`
- `artifacts/multi_seed/path_b_baseline_n3.json`
- `artifacts/multi_seed/path_b_method_n3.json`

Gate evaluation tool: `scripts/compare_methods.py` with `--metric {final_reward,best_eval_reward,mean_eval_reward}`.

## Template caption

Results are reported as mean ± std over N seeds from `configs/seeds.yaml`. Statistical comparison versus baseline uses Welch's t-test (unequal variance), with reported p-values.

## Phase E pre-registration (market-linked reward cutover) — 2026-04-27

This section applies to the post-cutover reward (`MarketPrices`-based EM + AM reward) and is append-only.

- Primary metric (Phase E): `mean_market_profit`.
- Confirmatory metric: `best_eval_reward`.
- Primary decision threshold: Cohen's d >= 0.5 with positive direction.
- Confirmatory threshold: Cohen's d >= 0.3 with positive direction.
- Fresh screening seeds (N=3): `[901, 1459, 8087]`.
- Fresh scale-up seeds for N=5 (only if screening gate passes): add `[1613, 9341]`.
- Burned seeds from earlier reward definitions must not be reused: `[42, 1337, 2024]`, `[777, 3141, 2718]`.

Protocol guardrail:

- No paper-grade Phase E training run may start before this pre-registration entry exists in version control.
- Path A/Path B STOP decisions remain valid only for earlier reward definitions and are not reused for Phase E.
