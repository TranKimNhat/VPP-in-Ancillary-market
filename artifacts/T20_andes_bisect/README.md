# T20 — bisection biên an ninh trên ANDES

**Câu hỏi (plan §3-P5, bước 2 của kế hoạch):** vòng bisection có kẹp được một biên hữu hạn ở
một cấu hình duy nhất (G0, 6 GFM, một nhiễu) không? Nếu có, campaign chạy được bằng ~12 lần
chạy một biên thay vì quét lưới vài nghìn điểm.

## Kết luận một dòng

**Đạt.** Bốn biên, mỗi biên kẹp được, **đơn điệu**, 12–13 lần chạy, ~12 s một lần chạy. Nhưng
hai chuyện phải nói ngay: một trong bốn biên **không chứa động học nào**, và ba biên còn lại
do **một quyết định điều độ chưa ai ra** quyết định.

## Bốn biên

| Hiện vật | Đại lượng | Nhiễu | Mô hình tải | Giá trị | Toạ độ siết |
|---|---|---|---|---:|---|
| `T20_andes_bisect/` | `P_head^min` | bậc tải 0,5 MW | P không đổi | **0,5099 MW** | khả thi (không có động học) |
| `T20_andes_bisect_dpmax/` | `ΔP_max` | bậc tải | P không đổi | **0,5340 MW** | `μ_I` (G4) |
| `T20_genloss_constP/` | `ΔP_max` | **mất máy phát** | P không đổi | **0,5801 MW** | `μ_I` (G4) |
| `T20_genloss_constZ/` | `ΔP_max` | mất máy phát | Z không đổi | **0,7184 MW** | `μ_I` (G4) |

Hai hàng cuối là cặp đối chứng mô hình tải: **chỉ riêng giả thiết tải đã dịch biên +23,8%.**

## Sinh lại

```
uv run python experiments/t20_andes_bisect.py --what p_head --event load_step --dp 0.5 \
    --tol 0.05 --t-end 8.0 --out artifacts/T20_andes_bisect
uv run python experiments/t20_andes_bisect.py --what dp_max --event gen_loss \
    --tol 0.05 --t-end 8.0 --dp-lo 0.05 --dp-hi 3.0 --out artifacts/T20_genloss_constP
uv run python experiments/t20_andes_bisect.py --what dp_max --event gen_loss --load-p2z 1.0 \
    --tol 0.05 --t-end 8.0 --dp-lo 0.05 --dp-hi 3.0 --out artifacts/T20_genloss_constZ
```

Cài đặt: `src/phasor/build_case.py` (dựng ca), `src/phasor/metrics.py` (đo + phán quyết
Ω_dyn), `src/campaign/boundary.py` (bisection). Nền tảng ANDES 2.0, `REGF1` droop cho GFM,
`GENROU`+`TGOV1`+`SEXS` cho diesel. Bố trí `official_placement_v4_rescaled.json`.

---

## B1 — `P_head^min` không chứa động học ở nhiễu nhỏ

Cơ chế đúng là điều kiện tồn tại điểm cân bằng, không phải đáp ứng động:

| headroom | trần đội thiết bị | nhu cầu sau nhiễu | có cân bằng? |
|---:|---:|---:|---|
| 0,4968 MW | 1,1068 MW | 1,1100 MW | **không** |
| 0,5231 MW | 1,1331 MW | 1,1100 MW | có |

Khi trần thấp hơn nhu cầu, hệ **không có trạng thái xác lập nào để hội tụ tới**; tần số rơi
không giới hạn. Cận giải tích là `P_head^min = ΔP = 0,500` MW; ANDES cho 0,5099 với dung sai
0,05, tức khoảng kẹp **chứa** cận giải tích.

> **Đọc đúng:** ở ΔP nhỏ, `P_head^min` chỉ là bất đẳng thức bảo toàn công suất — nó không cần
> mô hình động để biết. Muốn `P_head^min` mang nội dung, ΔP phải đủ lớn để nadir / RoCoF /
> `μ_I` siết *trước* điều kiện khả thi. Đó là lý do trục quét chính của campaign nên là ΔP
> (B2), không phải headroom.

Một chi tiết suýt làm hỏng phán quyết: ca vô nghiệm **không "trượt" mà "bò"** — 798 s so với
trung vị 12 s, và tiêu chuẩn phân kỳ của ANDES không kích hoạt. `metrics.py` nay phân biệt
*hết ngân sách nhưng cả đội bão hoà và `μ_P > 1`* (= vô nghiệm, một **kết quả**) với *đứng
máy trong khi mọi toạ độ còn trong dải* (= thất bại nền tảng, **không** tính là bằng chứng).

---

## B2 — `ΔP_max`: biên dòng, không phải biên công suất

Ca `gen_loss` / tải P không đổi (`T20_genloss_constP/`):

| ΔP [MW] | `μ_I` | `μ_P` | f_nadir [Hz] | RoCoF | V_min | an toàn? | lý do |
|---:|---:|---:|---:|---:|---:|---|---|
| 0,4188 | 0,908 | 0,225 | 59,648 | 0,70 | 0,971 | có | — |
| **0,5570** | **0,980** | 0,254 | 59,532 | 0,94 | 0,962 | **có** | — |
| **0,6031** | **1,004** | 0,264 | 59,493 | 1,02 | 0,958 | **không** | chỉ `μ_I` (G4) |
| 0,7875 | 1,100 | 0,306 | 59,337 | 1,33 | 0,946 | không | chỉ `μ_I` (G4) |
| 1,2300 | 1,325 | 0,431 | 58,961 | 2,08 | 0,915 | không | `μ_I`, nadir, RoCoF |

**`μ_P` chỉ đạt 0,26 tại biên.** Dự trữ công suất không hề ràng buộc — biên là biên *dòng*,
và nó nằm ở đơn vị **nhỏ nhất** (G4, 0,3793 MVA). Đây chính là điều C1/C2 phát biểu, đo được
lần đầu trên một mô hình động.

---

## Ba phát hiện đi kèm

### F1 — Bộ ước lượng dòng giải tích hụt 4,6–6,2 lần, vì nó bỏ qua công suất phản kháng

`μ̂_I^ana = |S|/|V|` theo plan §3, nhưng tầng giải tích không dự báo `Q`, nên thực tế nó chỉ
tính dòng tác dụng. Tỉ số `μ_I^andes / μ̂_I^ana` trung bình **5,13** (gen_loss / P không đổi),
**4,64** (gen_loss / Z không đổi), **6,16** (bậc tải).

Phân rã dòng của G4 tại đỉnh, ca ΔP = 0,5 MW:

```
P = 0,1056   Q = 0,4159   |S| = 0,4291   |I| = 0,4443   (pu hệ)
phản kháng chiếm 96,9% công suất biểu kiến
μ_I có Q      = 0,976
μ_I nếu Q = 0 = 0,232
```

> **Ngưỡng nghiệm thu T4 (`p95|μ̂_I^ana − μ_I| ≤ 0,05`) không thể đạt bằng bộ ước lượng chỉ có
> P.** Hoặc tầng giải tích phải sinh ra `Q`, hoặc T4 phải phát biểu lại. Đây là kết quả cho T4
> trước khi T4 chạy — và nó rẻ hơn T4 rất nhiều.

### F2 — Chéo kiểm tầng giải tích: sai số 0,004%

Ở chế độ không bão hoà, `Δf_ss` đo trên ANDES là **−0,343860 Hz**; công thức §14
`f₀·R·ΔP/ΣS_g` cho **0,343847 Hz**. Đây là lần đầu tầng giải tích được kiểm độc lập bằng một
mô hình động — và nó chỉ đúng *sau khi* hai lỗi nền tảng của `REGF1` được sửa (droop quy về
system base; tích phân giới hạn P làm cứng droop 75%). Xem concept §25.

### F3 — Chi phí, và đòn bẩy của bisection

| | trung vị | khoảng |
|---|---:|---:|
| lần chạy an toàn (8 s mô phỏng, dt = 2 ms) | 12,0 s | 11,6–14,2 s |
| lần chạy mất an ninh, hỏng nhanh | — | 3,5–29 s |
| lần chạy vô nghiệm (chạm trần ngân sách 60 s) | 61 s | 61–68 s |
| **một biên** | **12 lần chạy, ~4 phút** | |

Bốn biên trên tốn **49 lần chạy, ~18 phút**. Một quét lưới 3 000 điểm cùng độ phân giải sẽ
tốn ~10 giờ.

---

## Cảnh báo — **chưa được trích dẫn `ΔP_max` như một kết quả**

Cả ba biên B2 do `μ_I` của G4 đặt, và 96,9% dòng của G4 là phản kháng. Lượng phản kháng đó
**không phải một quyết định điều độ** — nó là nghiệm mà bài toán phân bố công suất trả về khi
sáu GFM cùng được đặt `V = 1,00 pu`. Độ nhạy dQ/dV đo được **1,3·10³ pu/pu** ngay cả *sau khi*
đã đưa mỗi converter ra sau nhánh ghép nối tường minh (trước đó **3,7·10⁴**). Con số 0,58 MW
do đó đo *một cách chia công suất phản kháng tuỳ tiện*, không đo năng lực đội thiết bị.

**Phải chốt trước khi biên này thành số trong bản thảo:** chiến lược điều độ `Q` cho đội GFM
— setpoint cố định theo định mức, hay Q–V droop có điểm làm việc xác định. Đây cũng là toạ độ
mà tài liệu về ca 100% GFM báo là nguyên nhân đổ vỡ, nên nó là quyết định vật lý chứ không
phải chi tiết cài đặt. Đã ghi thành **R8** trong plan §6.

Ba giới hạn khác của tầng này, đã ghi trong mã:

1. **`REGF1` không có bộ hạn dòng.** Mọi phán quyết `μ_I > 1` là *sàng lọc*, không phải mô tả:
   converter thật chuyển sang chế độ nguồn dòng và động học đổi. Cột `beyond_platform` trong
   `metrics.csv` đánh dấu chính xác các hàng này. T6 phải xác nhận.
2. **Gain vòng trong là gain ổn định số, không phải gain vật lý** (KPi 0,5→0,2; KIi 20→5).
   T3 mới đặt gain thật từ mô hình EMT thiết bị.
3. **Mô hình tải dịch biên 23,8%.** Hai ca P-không-đổi và Z-không-đổi là hai đầu mút; điểm
   vận hành thật nằm giữa và cần một tỉ lệ ZIP có nguồn.

### Một cái bẫy đã sập, để lại đây cho lần sau

Đối chứng mô hình tải ban đầu chạy bằng nhiễu *bậc tải*, và trả về "an toàn ở cả hai đầu
bracket". Không phải vật lý: `Alter` trên `PQ.Ppf` **hoàn toàn vô hiệu** khi trọng số công
suất không đổi bằng 0 — nhiễu 3 MW dịch tần số **0,0000 Hz**. Guard kiểm tra bracket trong
`boundary.py` bắt được, đúng chức năng của nó. Nhiễu mặc định nay là `gen_loss` (`Toggle` một
thiết bị đang vận hành), hoạt động như nhau với mọi tỉ lệ ZIP; `load_step` kèm `load_p2z > 0`
nay báo lỗi thay vì chạy im lặng.

## Tệp

| Tệp | Nội dung |
|---|---|
| `metrics.csv` | một hàng / một lần chạy, đủ trường bắt buộc của plan §5 |
| `boundaries.csv` | một hàng / một biên: giá trị, khoảng kẹp, số lần chạy, tính đơn điệu |
| `raw/<run_id>.npz` | t, f và RoCoF (các bus đo), P/Q/\|I\| từng GFM, bao V của feeder |
| `manifest.json` | git commit, phiên bản nền tảng, toàn bộ `CaseSpec`, dải an ninh |
| `figures/` | biên trên mặt (x, f_nadir) và (x, μ) |

Các cột `*_emt` **để trống có chủ ý**. ANDES không phải chuẩn tham chiếu; kết quả của nó ở cột
`*_andes`, mọi hàng mang `platform = andes`. Khi campaign EMT chạy, hai tệp ghép được theo
hàng mà không lẫn nguồn gốc.
