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

## Template caption

Results are reported as mean ± std over N seeds from `configs/seeds.yaml`. Statistical comparison versus baseline uses Welch's t-test (unequal variance), with reported p-values.
