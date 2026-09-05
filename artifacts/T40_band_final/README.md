# T40 — $\Delta P_{\max}$ tại dải an ninh đã chốt toàn bộ

Dải an ninh nay có nguồn cho **mọi** ngưỡng vào phán quyết. Nguồn gốc, lý do chọn, và BibTeX:
`reference/security_band_provenance.md`.

| ngưỡng | giá trị | nguồn |
|---|---|---|
| `f_min_hz` / `f_max_hz` | **59,5 / 60,5** (±0,5 Hz) | dải phổ biến nhất trong điều độ microgrid ốc đảo; neo: trên UFLS tầng đầu (`song2025`, `rebollal2021`, `xu2023ufls`) |
| `rocof_max_hz_s` | 2,0 | IEEE 1547-2018 như văn liệu microgrid trích (`ali2024rocof`) |
| `v_min_pu` | 0,88 | IEEE 1547-2018 Cat III, sàn Continuous Operation (`ninad2021`, `mahmud2022`) |
| `mu_i_max` | 1,0 vs `ImaxF` | REGFM_A1 PNNL-35110 (`du2023regfm`) |

Cấu hình converter: ship C (`kp_plim` 1,0; vòng trong 0,10/3,0) — T32/T33.

---

## Kết quả

$$\Delta P_{\max} = \mathbf{0{,}7241\ MW}\quad [0{,}7184;\ 0{,}7299],\ 14\text{ lần dò, đơn điệu}$$

**Siết bởi `f_nadir`, dứt khoát.** Mọi điểm mất an ninh quanh biên hỏng vì `f_nadir < 59,5` và
không kèm tiêu chí nào khác.

| tiêu chí tại điểm an toàn cuối | giá trị | ngưỡng | dùng hết | còn lại |
|---|---:|---:|---:|---:|
| **nadir** | 59,5045 | 59,5 | 0,9910 | **0,9%** |
| RoCoF | 0,9680 | 2,0 | 0,4840 | 51,6% |
| $V_{\min}$ | 0,9497 | 0,88 | 0,4195 | 58,1% |
| $\mu_I$ | 0,5149 | 1,0 | 0,5149 | 48,5% |

Thế chân vạc ở dải cũ biến mất hoàn toàn: ba tiêu chí còn lại đều còn **quá nửa** dự phòng.

## Dạng đóng được xác nhận bằng đo

Vì $\kappa_{os}$ = 1,0046 (T36), ràng buộc nadir là đại số thuần:

$$\Delta P_{\max} = \frac{\Delta f_{band}\sum S_g}{\kappa_{os}f_0R} = 1{,}447\,\Delta f_{band}$$

| | dự đoán | đo | lệch |
|---|---:|---:|---:|
| $\Delta f_{band}$ = 0,5 Hz | 0,7235 | **0,7241** | **0,08%** |
| $\Delta f_{band}$ = 1,0 Hz (T39) | 1,447 | 1,4501 | 0,21% |

**Miền hiệu lực:**

| $\Delta f_{band}$ | siết bởi | $\Delta P_{\max}$ |
|---|---|---|
| **< 1,039 Hz** ← ta ở đây | **nadir** | $1{,}447\,\Delta f_{band}$ |
| 1,039 – 1,194 Hz | RoCoF | 1,504 (cố định) |
| > 1,194 Hz | $V_{\min}$ | 1,727 (cố định) |

**Bài nên công bố hệ số và miền hiệu lực, không công bố điểm.** Như vậy dải tần là *đầu vào
được khai báo*, không phải một lựa chọn ẩn, và người đọc đọc ra con số ở bất kỳ dải nào họ dùng.

## Sổ cái $\Delta P_{\max}$ qua toàn chiến dịch

| cấu hình | dải | $\Delta P_{\max}$ | siết bởi |
|---|---|---:|---|
| công bố ban đầu | f ±1,0 · RoCoF 2,0 · V 0,90 | 1,185059 | rocof + nadir |
| ship C (T34) | như trên | 1,438574 | $V_{\min}$ (nadir cách 0,56%) |
| + V có nguồn (T39) | f ±1,0 · RoCoF 2,0 · **V 0,88** | 1,450098 | nadir |
| **+ f có nguồn (T40)** | **f ±0,5 · RoCoF 2,0 · V 0,88** | **0,7241** | **nadir** |

Hai hiệu ứng ngược chiều và cả hai đều lớn: sửa conformance converter **nâng** biên 21,4%; sửa
conformance dải an ninh **hạ** nó 50%. Con số công bố cuối cùng thấp hơn con số ban đầu 39%, và
lần này mọi ngưỡng đặt ra nó đều trích dẫn được.

---

`experiments/t20_andes_bisect.py` với mặc định mới (`--f-band 0.5`, `--v-min 0.88`) · 14 lần dò
