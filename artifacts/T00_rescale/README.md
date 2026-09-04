# T00 — rescale đội GFM về cỡ vật lý khả tín

**Lý do.** Hồ sơ v3 được tối ưu cho bài toán VPP/thị trường, không cho an ninh tần số:
23 MVA converter trên feeder tải đỉnh 3,49 MW. Mọi đơn vị chạy ở ~10% định mức, nên
*toàn bộ vùng khan hiếm headroom là bất khả đạt về mặt vật lý*. Campaign chạy trên v3 sẽ
trả về "an toàn ở mọi nơi" — một kết quả rỗng, không phải một kết quả null.

```
uv run python experiments/rescale_gfm_fleet.py [--ratio 1.25]
```

**Quy tắc.** Scale **đều** cả sáu đơn vị trên `bess_mw`, `bess_mwh`, `inverter_mva`, `pv_mw`.
Điều này giữ nguyên `π_g`, tỉ số E/P (2 h) và tỉ số MVA/MW của từng đơn vị. Tỉ lệ mục tiêu
đặt trên **công suất biểu kiến converter**, đúng cách nhà máy tham chiếu Bạch Long Vĩ được
báo (630 kVA BESS trên tải đỉnh ~0,6–1 MW).

| | trước | sau | hệ số |
|---|---:|---:|---:|
| Tổng inverter | 23,000 MVA | **4,3625 MVA** | ×0,18967 |
| Tổng BESS | 18,000 MW | **3,4140 MW** | |
| Tổng BESS năng lượng | 36,00 MWh | 6,828 MWh | |

Chi tiết từng đơn vị: `gfm_rescale.csv`.

## Điều rescale này **không** làm

Chỉ đội GFM được đụng tới. Toàn bộ nguồn còn lại giữ nguyên cỡ v3:

| Lớp tài sản | MW | Đơn vị lớn nhất |
|---|---:|---:|
| Gió | 12,000 | 3,0 MW |
| DPV | 3,850 | 14 cụm nhỏ |
| EVCS — PV / BESS / V2G | 1,458 / 2,475 / 0,825 | |
| PV đi kèm GFM | 2,000 → 0,379 | |

**Tổng công suất có thể bơm: 39,78 MW (v3) → 23,58 MW (v4), trên tải đỉnh 3,49 MW —
vẫn dư 6,8 lần.** Con số này quan trọng vì các nguồn chưa rescale chính là thứ **đặt độ lớn
`ΔP` cho campaign**: mất một trang trại gió là nhiễu 3,0 MW, tức 0,86 lần tải đỉnh feeder và
0,88 lần `ΔP_critical` của đội GFM mới. Sự kiện lớn nhất do đó gần như đúng bằng biên — thuận
lợi cho việc dò biên, nhưng là kết quả của một tỉ lệ chưa ai chọn có chủ đích.

**Đây là quyết định còn mở**, xem phần cuối.

## Ảnh hưởng đã đo được

| Hệ quả | Ở đâu |
|---|---|
| `ΔP_critical` 17,25 → **3,272 MW**; `κ` cắt 1 tại `ΔP` = 3,414 MW | `artifacts/T02_soc_v4/` |
| Ràng buộc tần số nay siết trước headroom 4,5 lần (`R` = 5%) | `artifacts/T02_soc_v4/` |
| `Ā_GFM` (ε vật lý) 15,83 → 6,01; ε lấn át hoàn toàn khoảng cách lưới | `artifacts/T01_agfm_v4/` |
| R2 trong plan §6 được sửa lần 2 — phương án lùi sống lại | `plan_v3.1_execution.md` |

## Quyết định còn mở

1. **Tỉ lệ 1,25 là giả định của tôi**, đặt trong dải 1,0–1,5 mà bạn nêu. Nó là tham số một
   dòng (`--ratio`), chạy lại toàn bộ chuỗi T01/T02 mất vài giây.
2. **Có rescale gió / DPV / EVCS không.** Nếu không, nhiễu lớn nhất (3 MW) gần trùng biên và
   campaign sẽ chỉ có vài điểm hữu ích ở phía dưới biên. Nếu có, phải quyết định giữ tỉ số
   nguồn/tải nào — và điều đó chạm vào phần VPP/thị trường của bài, chứ không chỉ phần tần số.
3. **`R` = 5% có còn đúng không.** Ở v4 nó cho `Δf_ss` = 0,688 Hz với `ΔP` = 1 MW. Hoặc `R`
   phải giảm, hoặc điều khiển thứ cấp phải nằm trong định nghĩa an toàn ngay từ đầu.
