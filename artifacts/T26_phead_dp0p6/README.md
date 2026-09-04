# T26b — biến thể ΔP = 0,6 MW


Bản chính: `artifacts/T26_phead_dp1p1/` — đọc README ở đó.

Chạy này chỉ khác ΔP: 0,6 MW thay vì 1,1 MW.

**`P_head^min` = 0,5953 MW → κ = 1,0078** (bản chính: κ = 1,0049 tại ΔP = 1,1). Hai điểm cho
κ = 1,006 ± 0,002 — biên headroom là biên khả thi, không có dự trữ động học chồng lên.

```
uv run python experiments/t20_andes_bisect.py --what p_head --event gen_loss --dp 0.6 \
    --load-p2z 0.0 --q-max 0.60 --out artifacts/T26_phead_dp0p6
```
