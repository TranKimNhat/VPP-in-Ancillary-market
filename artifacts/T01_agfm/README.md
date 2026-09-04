# T01 — Ā_GFM recomputed against the canonical §8 definition

> **Chạy trên hồ sơ bố trí v3 (chưa rescale).** Xem `artifacts/T01_agfm_v4/` cho đội GFM
> đã đưa về cỡ thật — và cho phát hiện rằng quy ước ε hiện tại làm `Ā_GFM` đo định mức
> converter nhiều hơn đo vị trí.

**Câu hỏi:** con số `Ā_GFM` và các % thay đổi 2→5→6 GFM trong concept v3.1 §10 có khớp
định nghĩa chuẩn ở §8 không?

**Kết luận một dòng:** **Không.** Con số `+121,2%` (2→6) và `+16,5%` (5→6) là hành vi của
chỉ số **raw** `A_GFM` (đo *dung lượng*), không phải của chỉ số chuẩn hoá không thứ nguyên
`Ā_GFM` (đo *khả năng tiếp cận*); tính lại đúng theo §8 thì trục A gần như phẳng.

## Sinh lại

```
uv run python experiments/t01_agfm.py
```

Cài đặt: `src/analytical/accessibility.py`. Tham số: `config.yaml`. Môi trường: `manifest.json`.

## Bốn phát hiện

**F1 — Số cũ không tái lập được (vi phạm N-2).**
Script sinh ra `artifacts/electrical_distance_analysis.json` không có trong cây làm việc
**và không có trong toàn bộ lịch sử git**; bản thân tệp artifact cũng untracked. Không thể
kiểm chứng nó. Mọi con số §9/§10 dẫn từ tệp này phải coi là chưa có nguồn cho tới khi
được tính lại.

**F2 — `+121,2%` là hiệu ứng dung lượng, không phải khả năng tiếp cận.**

| Chỉ số | 2→6 GFM | 5→6 GFM |
|---|---:|---:|
| `A_raw` (tính lại) | **+120,8%** | +17,8% |
| concept §10 công bố | +121,2% | +16,5% |
| `Ā_GFM` chuẩn, ε số học | +10,4% | +8,0% |
| `Ā_GFM` chuẩn, ε vật lý (x=0,10 pu) | **−0,05%** | **−2,6%** |

Khớp gần như hoàn hảo giữa `A_raw` tính lại và số công bố xác nhận: §10 đang báo cáo chỉ số
raw dưới nhãn "normalized-analysis". Đây đúng là điều §8 cảnh báo — raw "mixes capacity and
placement". Nguyên nhân cấu trúc: `π_g` là trọng số **chuẩn hoá**, nên thêm một GFM chỉ
*phân phối lại* trọng số headroom chứ không cộng thêm; `Ā_GFM` do đó không đơn điệu theo
số lượng GFM.

**F3 — ε không phải chi tiết số học, nó quyết định kết quả.**
4/85 nút tải có GFM đặt ngay trên nó ⇒ `|Z_gk| = 0`. Ở cấp 4,16 kV, trở kháng ghép nối của
chính converter (LCL + máy biến áp nâng, 0,05–0,15 pu trên base thiết bị) là **0,12–1,30 Ω**,
cùng bậc với toàn bộ khoảng cách điện của feeder (`Z_max = 0,73 Ω`). Chọn ε lật dấu kết quả:
2→6 đi từ +10,4% (ε = 1e-6 Ω) xuống −3,5% (ε = 0,15 pu). **§8 phải phát biểu rõ ε là gì.**
Mặc định dùng ở đây: ε_g = 0,10 pu trên base MVA của từng thiết bị (đơn vị lớn hơn ⇒ gần
hơn về điện, đúng chiều vật lý).

**F4 — Tập 2 GFM không được ghi lại ở đâu, và nó chi phối con số 2→6.**
`two_gfm_pair_sweep.csv`: quét cả 15 cặp, `ΔĀ` 2→6 trải từ **−17,6%** (G2+G5) tới **+40,9%**
(G4+G6). Giả định dùng ở đây là G1+G2 (neo slack + giữa feeder theo quy tắc 2/3) cho −0,05%.
Con số 5→6 (**−2,6%**) không phụ thuộc giả định này và nên là con số dẫn.

## Cái vẫn sống

Chẩn đoán khoảng cách thuần tuý vẫn giảm mạnh và đơn điệu theo số GFM:
`Z_avg −46,8%`, `Z_max −30,7%` (2→6). (concept §10 ghi −60,1% / −52,7% — cũng không tái lập
được, cùng gốc F1, nhưng dấu và bậc độ lớn giữ nguyên.)

## Hệ quả cho bản thảo

- Bỏ `+121,2% / +16,5%` khỏi bản thảo, hoặc gắn lại nhãn đúng là chỉ số **raw**.
- C3 (đóng góp cấu hình/bố trí) yếu hơn concept giả định — khớp với R4 trong sổ rủi ro §6.
- Cần chốt ε và tập 2-GFM vào concept §25 trước khi dùng bất kỳ con số trục A nào.

## Tệp

| Tệp | Nội dung |
|---|---|
| `agfm_recomputed.csv` | bảng chính: 2/5/6 GFM ở ε và P_head chuẩn |
| `metrics.csv` | toàn bộ 24 ca (2 quy ước P_head × 4 ε × 3 bố trí) |
| `axis_a_deltas.csv` | % thay đổi trục A cho `Ā` và `A_raw` |
| `two_gfm_pair_sweep.csv` | độ nhạy theo tập 2-GFM |
| `legacy_v3.1_values.csv` | số cũ chép lại để đối chiếu |
