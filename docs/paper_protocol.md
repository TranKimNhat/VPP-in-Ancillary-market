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

## Template caption

Results are reported as mean ± std over N seeds from `configs/seeds.yaml`. Statistical comparison versus baseline uses Welch's t-test (unequal variance), with reported p-values.
