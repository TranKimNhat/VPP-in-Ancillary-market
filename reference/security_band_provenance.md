# Nguồn gốc dải an ninh Ω_dyn — và vì sao mục này tồn tại

**Vấn đề đã phát hiện (T38/§3, 2026-09-05).** `src/phasor/metrics.SecurityBand` khai bốn
ngưỡng trần trụi, không comment, không trích dẫn:

```python
f_min_hz: float = 59.0
f_max_hz: float = 61.0
rocof_max_hz_s: float = 2.0
v_min_pu: float = 0.90
```

Cùng file, docstring module ghi **rất kỹ** nguồn gốc của `mu_I` (vì sao đo theo `ImaxF` chứ
không theo định mức liên tục) và của RoCoF (vì sao cửa sổ trượt chứ không dùng `BusROCOF`).
Bốn ngưỡng trên thì không. Concept doc cũng không định nghĩa dải ở đâu.

Đây là **khiếm khuyết nền tảng thứ năm** cùng lớp với `pv2pq`/`adjust_upper`, `KIplim`,
`Pmax non_negative`, `KPplim` — một giá trị không ai chọn có chủ đích đang quyết định một kết
luận. Ở đây nó tệ hơn bốn cái trước: nó không chỉ dịch một con số, nó quyết định **cơ chế vật
lý mà bài nói là nguyên nhân**.

Đo được (cấu hình ship C, `artifacts/T34_rerun_shipC/dpmax_q060/metrics.csv`):

| ngưỡng $V_{\min}$ | ΔP khi V siết | toạ độ siết thật | $\Delta P_{\max}$ |
|---|---:|---|---:|
| 0,9167 | 1,195 | $V_{\min}$ (nadir cách 21%) | **1,195** |
| 0,90 | 1,438 | $V_{\min}$ (nadir cách **0,56%**) | **1,438** |
| 0,88 | 1,729 | **nadir** (V cách 16,4%) | **≈1,446** ← nadir chặn trên |

nadir siết tại ΔP = 1,44595 MW, **cố định**, không phụ thuộc ngưỡng V. Nên nới ngưỡng V dưới
0,90 gần như không dịch $\Delta P_{\max}$ — nadir tiếp quản. Siết ngưỡng thì dịch mạnh.

---

## Quyết định: $V_{\min}$ = **0,88 pu**

**Lý do:** đây là ngưỡng duy nhất trong ba lựa chọn có neo tài liệu **trực tiếp cho DER**, chứ
không phải cho chất lượng điện áp phía tải.

| ngưỡng | neo | đại lượng nó thật sự định nghĩa |
|---|---|---|
| **0,88 pu** | **IEEE 1547-2018 / 1547.1-2020**, LVRT Category III | **cận dưới vùng Continuous Operation của DER** (dải 0,88–1,00 pu, 5 s và 120 s) |
| 0,90 pu | ANSI C84.1 | cận dưới **utilization voltage Range A** (108 V trên nền 120 V) |
| 0,9167 pu | ANSI C84.1 | cận dưới **service voltage Range B** (110 V trên nền 120 V) |

Biến bài này cần là *ngưỡng mà DER phải duy trì vận hành liên tục khi có nhiễu loạn điện áp*,
không phải *dải điện áp cấp cho tải*. IEEE 1547 quy định trực tiếp yêu cầu hiệu năng của DER
dưới điện áp bất thường (continuous / mandatory / permissive / momentary cessation / trip),
và IEEE 1547.1 là bộ thử xác minh tuân thủ chính các yêu cầu ride-through đó. ANSI C84.1 trả
lời một câu hỏi khác.

### Cách phát biểu trong bài

> Ngưỡng điện áp thấp lấy ở 0,88 pu, cận dưới vùng vận hành liên tục (Continuous Operation)
> của DER theo IEEE 1547-2018 Category III, xác minh bằng bộ thử IEEE 1547.1-2020.

**Không** viết 0,90 pu mà nói là ngưỡng DER — đó là ANSI utilization Range A. **Không** viết
0,9167 pu như ngưỡng DER trừ khi nói rõ đó là quy đổi từ ANSI service Range B minimum.

### Trích dẫn

```bibtex
@article{ninad2021ride,
  author  = {Ninad, Nayeem and Apablaza-Arancibia, Evgueniy and Bui, Michael and Johnson, Jay},
  title   = {Commercial {PV} Inverter {IEEE} 1547.1 Ride-Through Assessments
             Using an Automated {PHIL} Test Platform},
  journal = {Energies},
  year    = {2021},
  doi     = {10.3390/en14216936}
}

@techreport{mahmud2022protection,
  author      = {Mahmud, Rasel and Ingram, Michael},
  title       = {Background Information on the Protection Requirements in {IEEE} Std 1547-2018},
  institution = {NREL},
  year        = {2022},
  doi         = {10.2172/1839049}
}

@techreport{narang2021voltage,
  author      = {Narang, David and Mahmud, Rasel and Ingram, Michael and Hoke, Andy F.},
  title       = {An Overview of Issues Related to {IEEE} Std 1547-2018 Requirements
                 Regarding Voltage and Reactive Power Control},
  institution = {NREL},
  year        = {2021},
  doi         = {10.2172/1821113}
}

@standard{ieee2800,
  title = {{IEEE} Standard for Interconnection and Interoperability of Inverter-Based
           Resources ({IBR}s) Interconnecting with Associated Transmission Electric
           Power Systems},
  number = {IEEE Std 2800-2022},
  year  = {2022},
  pages = {1--180},
  doi   = {10.1109/ieeestd.2022.9762253}
}
```

Vai trò từng nguồn: **ninad2021** mang chính con số 0,88 pu và dải Continuous Operation Cat III
— đây là trích dẫn chính. **mahmud2022** cho khung năm chế độ vận hành dưới điện áp bất thường
của 1547-2018. **narang2021** là nguồn cho *hai giá trị bị loại* (ANSI Range A 108/120,
Range B 110/120) — cần khi giải thích vì sao không dùng chúng. **ieee2800** củng cố rằng
ride-through điện áp là yêu cầu chuẩn hoá cho IBR nhưng **không** nêu ba mốc cụ thể; đừng trích
nó cho con số.

---

## Ba ngưỡng còn lại vẫn chưa có nguồn

`f_min_hz` = 59,0 · `f_max_hz` = 61,0 · `rocof_max_hz_s` = 2,0 — vẫn trần trụi. Với đội hình
mới ($V_{\min}$ = 0,88) **nadir trở thành toạ độ siết**, nên `f_min_hz` = 59,0 giờ là con số
trực tiếp đặt $\Delta P_{\max}$. Nó cần đúng mức trích dẫn như 0,88 vừa nhận, và **chưa có**.

Đây là việc còn mở, không phải việc đã xong.
