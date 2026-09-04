# T21 — biến thể độ nhạy `Qmax = ±0,44`

Bản chính là `artifacts/T21_genloss_constP_regfm_q060/` — đọc README ở đó.

Chạy này chỉ khác một tham số: `q_max_pu = 0,44` (đáy dải REGFM_A1) thay vì 0,60.

**Biên giống hệt: `ΔP_max = 1,18506 MW`**, vì biên do tần số quyết định chứ không do công
suất phản kháng. Đó chính là điều cần chứng minh: chọn Qmax trong dải đặc tả **không** dịch
biên an ninh.

Hai điểm khiến 0,44 không được chọn làm bản chính:

- Nhu cầu Q thường trực của feeder là 1,921 MVAr trên 4,362 MVA thiết bị = **0,4403 pu**, nên
  trần 0,44 đặt **cả sáu máy đúng trên trần**, không còn dự trữ phản kháng nào.
- GFM Slack kết thúc **cao hơn trần 0,0011 pu** → mỗi lần chạy in `** Initialization FAILED **`
  với thặng dư ~1e-4. Kết quả vẫn dùng được (`TDS.initialized = True`), nhưng là tiếng ồn
  không nên có trong hiện vật của bài báo.

```
uv run python experiments/t20_andes_bisect.py --what dp_max --event gen_loss --load-p2z 0.0 \
    --q-max 0.44 --out artifacts/T21_genloss_constP_regfm_q044
```
