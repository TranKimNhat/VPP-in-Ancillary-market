# T01-v4 — `Ā_GFM` trên đội GFM đã rescale, và tại sao quy ước ε phải chốt trước

Cùng mã như `artifacts/T01_agfm/`, chỉ đổi hồ sơ bố trí sang
`official_placement_v4_rescaled.json`.

```
uv run python experiments/t01_agfm.py \
    --placement artifacts/placement/official_placement_v4_rescaled.json \
    --out artifacts/T01_agfm_v4
```

## Kết luận một dòng

Rescale **không** đụng tới `π_g` hay `w_k`, nhưng nó đổi `Ā_GFM` gần 2,6 lần — vì bộ chính
quy hoá `ε_g = x_pu·V²/S_g` tỉ lệ nghịch với công suất converter. Ở quy mô v4, **mọi `ε_g`
đều lớn hơn toàn bộ bề rộng điện của feeder**, nên `Ā_GFM` với ε vật lý đo công suất thiết
bị nhiều hơn đo vị trí. Quy ước ε ở concept §8 là điều kiện chặn cho mọi con số axis-A.

## Bằng chứng 1 — `π_g` bất biến, đúng như thiết kế

Ở `eps_numerical` (ε = 1e-6 Ω, đọc "+ ε" theo nghĩa đen là sàn số học), v3 và v4 cho **cùng
một giá trị đến chữ số cuối cùng**:

| Bố trí | `Ā` v3 | `Ā` v4 | tỉ số |
|---|---:|---:|---:|
| 2 GFM | 99 200,222469 | 99 200,222469 | 1,000000 |
| 5 GFM | 101 463,976496 | 101 463,976496 | 1,000000 |
| 6 GFM | 109 540,170427 | 109 540,170427 | 1,000000 |

Scale đều giữ nguyên `π_g`. Đó là điều duy nhất tôi khẳng định ban đầu, và nó đúng.

## Bằng chứng 2 — nhưng với ε vật lý thì không

| Bố trí | `Ā` v3 | `Ā` v4 | tỉ số | `ε_max` v3 | `ε_max` v4 |
|---|---:|---:|---:|---:|---:|
| 2 GFM | 15,8393 | 6,9203 | 0,4369 | 0,4326 Ω | 2,2810 Ω |
| 5 GFM | 16,2510 | 6,2543 | 0,3849 | 0,8653 Ω | 4,5625 Ω |
| 6 GFM | 15,8321 | 6,0059 | 0,3794 | 0,8653 Ω | 4,5625 Ω |

(`eps_x010`, `p_head_basis = bess_mw`.)

## Bằng chứng 3 — ε đã nuốt trọn lưới

`ε_g` từng đơn vị ở v4, `x_pu` = 0,10:

| | bus | `S_g` [MVA] | `ε_g` [Ω] |
|---|---:|---:|---:|
| G1 | 114 | 1,3277 | 1,303 |
| G2 | 60 | 0,7587 | 2,281 |
| G3 | 105 | 0,7587 | 2,281 |
| G4 | 47 | 0,3793 | **4,563** |
| G5 | 67 | 0,7587 | 2,281 |
| G6 | 1 | 0,3793 | **4,563** |

Đối chiếu với chính lưới đó: khoảng cách điện tới GFM gần nhất có `Z_avg = 0,283 Ω` và
`Z_max = 0,726 Ω`. **`ε` nhỏ nhất vẫn lớn hơn `Z_max` 1,8 lần; `ε` lớn nhất lớn hơn 6,3 lần.**
Mẫu số `|Z_gk| + ε_g` do đó gần như hằng số theo `k`.

Đo trực tiếp mức đóng góp của lưới — so `Ā` thật với `Ā` tính khi ép `Z = 0` (tức chỉ còn
công suất thiết bị):

| | `Ā` (Z thật) | `Ā` (Z = 0) | lưới dịch được |
|---|---:|---:|---:|
| v3 | 15,832 | 46,667 | **−66,1%** |
| v4 | 6,006 | 8,851 | **−32,1%** |

Độ nhạy của chỉ số với bố trí **giảm một nửa** sau rescale. Chỉ số vẫn mang tên
"accessibility" nhưng ở v4 hai phần ba giá trị của nó đến từ định mức converter.

## Bằng chứng 4 — dấu của trục A đảo chiều

| `eps_case` | `ΔĀ` 2→6 GFM, v3 | v4 |
|---|---:|---:|
| `eps_numerical` | +10,42% | +10,42% |
| `eps_x005` | +5,46% | **−8,20%** |
| `eps_x010` | −0,05% | **−13,21%** |
| `eps_x015` | −3,50% | **−15,54%** |

Ở v3 dấu đã đảo một lần khi ε tăng (giữa `x005` và `x010`); ở v4 nó **âm ở mọi ε vật lý**.

Với ε vật lý, **thêm GFM làm `Ā` giảm**. Không nghịch lý: `Ā` là trung bình có trọng số `π_g`
của nhân `Z_base/(|Z|+ε_g)`, mà các đơn vị thêm vào (G4, G6 — 0,3793 MVA) có `ε_g` lớn nhất,
nên chúng kéo trung bình xuống. Nhưng nó có nghĩa là **chỉ số này, ở quy ước ε hiện tại,
không dùng được để lập luận "thêm GFM ⇒ tiếp cận tốt hơn"** — tức đúng cái §10 đang muốn nói.

## Việc phải chốt trước khi có bất kỳ con số axis-A nào lên bản thảo

1. **Quy ước ε.** Ba lựa chọn, phải chọn một và ghi vào §8:
   (a) ε là sàn số học thuần tuý ⇒ `Ā` đo bố trí, nhưng lúc đó `ε` không có nghĩa vật lý và
   `Ā` mang đơn vị lạ (bậc 10⁵);
   (b) ε là trở kháng ghép nối converter như hiện nay ⇒ có nghĩa vật lý, nhưng chỉ số trộn
   định mức với vị trí và không so sánh được giữa các đội thiết bị khác cỡ;
   (c) chuẩn hoá ε trên **base chung** thay vì base riêng từng đơn vị, tách định mức khỏi
   khoảng cách. Đây là hướng tôi nghiêng về, nhưng nó là thay đổi định nghĩa, cần bạn duyệt.
2. **Tập 2 GFM.** Vẫn chưa xác định được — mã sinh ra con số v3.1 không có trong cây làm việc
   lẫn trong lịch sử git. `two_gfm_pair_sweep.csv` cho dải **−13,2% … +84,6%** tuỳ cặp, nên
   mọi phát biểu "2→6 GFM" chưa có nghĩa cho tới khi tập này được ghi lại.
