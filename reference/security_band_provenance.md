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

# Phần II — `f_min_hz` và `rocof_max_hz_s`: **đã chốt** (2026-09-05)

Với $V_{\min}$ = 0,88, nadir thành toạ độ siết, nên `f_min_hz` đặt trực tiếp $\Delta P_{\max}$.
Hai vòng tra cứu Consensus cho ba kết quả, một trong đó **bác một cảnh báo tôi đã nêu**.

## II.1 ❌ RÚT: "dải trộn Category II với Category III"

Bản trước của mục này cảnh báo rằng dùng $V_{\min}$ = 0,88 (Cat III) cùng RoCoF = 2,0 (Cat II)
là trộn category, và đề nghị nâng RoCoF lên 3,0. **Rút.**

Văn liệu microgrid dùng **2 Hz/s như chính là** giới hạn IEEE 1547-2018. Ali et al. (2024) phát
biểu thẳng cho microgrid converter-interfaced: đảm bảo RoCoF *"remains within the limit of
**2 Hz/s according to IEEE Standard 1547–2018**"*, đặt cạnh giới hạn mất cân bằng áp 2% theo
IEC 61000-3-13. Con số 2,0 hiện dùng **có nguồn và giữ nguyên**.

Cặp 2 Hz/s (Cat II) / 3 Hz/s (Cat III) là ngưỡng **thử ride-through thiết bị** trong quy trình
IEEE 1547.1. Nó không phải là ngưỡng mà một nghiên cứu điều độ microgrid dùng làm tiêu chí —
và khi các nghiên cứu đó trích IEEE 1547-2018 cho RoCoF, con số họ trích là 2 Hz/s.

## II.2 `f_min` — neo đúng là **UFLS**, không phải ngưỡng cắt DER

Không nghiên cứu điều độ microgrid ốc đảo nào dùng ngưỡng cắt thiết bị (57,0 Hz) làm tiêu chí.
Tất cả dùng **dải lệch tần**, và neo vật lý là **kích hoạt UFLS tầng đầu**: nadir phải ở trên
đó thì mới không sa thải tải nào.

Rebollal et al. (2021) xây FCUC cho microgrid ốc đảo với mục tiêu tường minh là *giảm thiểu
kích hoạt UFLS*, và ghi rằng microgrid ốc đảo nhỏ đặc biệt nhạy vì thiếu quán tính. Xu et al.
(2023) thiết kế UFLS cho microgrid ốc đảo **chỉ có một nguồn grid-forming** — đúng cấu hình
của ta. Amraee et al. (2018) cho xung đột giữa rơ-le RoCoF của DG và nadir UFLS, tức lý do hai
ngưỡng phải đặt phối hợp.

Dải mà văn liệu này thực sự dùng:

| nghiên cứu | dải lệch | quy về 60 Hz |
|---|---|---|
| Gao et al. (2025), IEEE Access | ±0,4 Hz | 59,6 |
| **Song et al. (2025), Energies** | **0,5 Hz** | **59,5** |
| Gao et al. (2025), AEET | 49,2 trên nền 50 Hz | 59,2 |
| ~~ta, trước~~ | ~~1,0 Hz~~ | ~~59,0~~ ← **lỏng hơn tất cả** |

## II.3 ✅ QUYẾT ĐỊNH: `f_min_hz` = **59,5** (`f_max_hz` = 60,5), dải ±0,5 Hz

Lý do:

1. **Trong dải văn liệu**, và là giá trị phổ biến nhất trong đó (Song et al. 2025).
2. **Bảo thủ.** 1,0 Hz nằm ngoài dải, về phía lỏng — điểm reviewer sẽ đánh với một bài an ninh.
3. **Nhất quán với cách chọn $V_{\min}$**: cả hai là ngưỡng *vận hành liên tục*, không phải
   điểm cắt thiết bị.
4. Giữ nadir trên mọi sơ đồ UFLS tầng đầu hợp lý cho hệ 60 Hz.

## II.4 Hệ quả — và cách công bố xoá được tính tuỳ tiện

Vì $\kappa_{os}$ = 1,0046 (T36), ràng buộc nadir là **đại số thuần**:

$$\Delta P_{\max} = \frac{\Delta f_{band}\sum S_g}{\kappa_{os}\,f_0 R} = \mathbf{1{,}447\,\Delta f_{band}}\ \text{[MW, } \Delta f_{band}\text{ tính bằng Hz]}$$

Kiểm: $\Delta f$ = 1,0 → 1,447 (đo 1,450 ✓); $\Delta f$ = 0,5 → **0,724**.

**Miền hiệu lực** (cấu hình ship C, khớp tuyến tính 11 điểm T39):

| $\Delta f_{band}$ | toạ độ siết | $\Delta P_{\max}$ |
|---|---|---|
| < 1,039 Hz | **nadir** | $1{,}447\,\Delta f_{band}$ |
| 1,039 – 1,194 Hz | RoCoF (2,0 Hz/s) | 1,504 (cố định) |
| > 1,194 Hz | $V_{\min}$ (0,88) | 1,727 (cố định) |

**Công bố hệ số, không công bố điểm.** Bài nên đưa quan hệ tuyến tính cộng miền hiệu lực chứ
không đưa một con số — như vậy không phải bảo vệ một lựa chọn ngưỡng, và người đọc đọc ra con
số ở bất kỳ dải nào họ dùng. Tại dải đã chốt: $\Delta P_{\max}$ = **0,724 MW**.

## II.5 Bảng nguồn cho **cả bộ** — dùng để cite

| ngưỡng | giá trị | nguồn | loại |
|---|---|---|---|
| `v_min_pu` | **0,88** | IEEE 1547-2018 Cat III, sàn Continuous Operation (0,88–1,00, giữ 5 s và 120 s); thử theo IEEE 1547.1-2020 | `ninad2021`, `mahmud2022` |
| `v_max_pu` | 1,10 | ⚠️ **chưa kiểm** — cần cùng mức 0,88 | — |
| `rocof_max_hz_s` | **2,0** | IEEE 1547-2018, như văn liệu microgrid trích | `ali2024rocof` |
| `rocof_window_s` | 0,5 | đầu dài nhất dải 100–500 ms; lựa chọn bảo thủ theo hướng làm RoCoF khó siết | `brogan2019`, `amraee2018` |
| `f_min_hz` | **59,5** | dải ±0,5 Hz, phổ biến nhất trong điều độ microgrid ốc đảo; neo: trên UFLS tầng đầu | `song2025`, `rebollal2021`, `xu2023ufls` |
| `f_max_hz` | 60,5 | đối xứng | như trên |
| `mu_i_max` | 1,0 vs `ImaxF` | REGFM_A1 (PNNL-35110): `ImaxF` ví dụ 2,0, dải 1,5–3,0 | `du2023regfm` |
| `settle_tol/window` | 0,01 Hz / 2 s | tiêu chí hội tụ số học nội bộ, **không phải chuẩn** — khai báo như vậy | — |

## II.6 Trích dẫn Phần II

```bibtex
@article{ali2024rocof,
  author  = {Ali, Nada and others},
  title   = {Real-time validation of {RoCoF} -- Enhancing advanced coordinated control
             strategy for voltage unbalance mitigation implementing demand side management},
  journal = {Electric Power Systems Research}, year = {2024},
  doi     = {10.1016/j.epsr.2023.110100}
}

@article{song2025projection,
  author  = {Song, Xin and others},
  title   = {Projection-Based Coordinated Scheduling of Distribution--Microgrid Systems
             Considering Frequency Security Constraints},
  journal = {Energies}, volume = {18}, number = {21}, pages = {5707}, year = {2025},
  doi     = {10.3390/en18215707}
}

@article{rebollal2021endogenous,
  author  = {Rebollal, David and others},
  title   = {Endogenous Approach of a Frequency-Constrained Unit Commitment in Islanded
             Microgrid Systems},
  journal = {Energies}, volume = {14}, number = {19}, pages = {6290}, year = {2021},
  doi     = {10.3390/en14196290}
}

@article{xu2023ufls,
  author  = {Xu, Bei and others},
  title   = {Under-Frequency Load Shedding for Power Reserve Management in Islanded
             Microgrids},
  journal = {IEEE Transactions on Smart Grid}, year = {2023},
  doi     = {10.1109/tsg.2024.3393426}
}

@article{gao2025dispatch,
  author  = {Gao, Runsheng and others},
  title   = {Coordinated Dispatch of Microgrids With Multi-Process Industrial Loads
             Considering Frequency Security Under Unplanned Islanding},
  journal = {IEEE Access}, year = {2025},
  doi     = {10.1109/access.2025.3634672}
}

@article{javadi2021fscms,
  author  = {Javadi, Masood and others},
  title   = {Frequency Stability Constrained Microgrid Scheduling Considering Seamless
             Islanding},
  journal = {IEEE Transactions on Power Systems}, year = {2021},
  doi     = {10.1109/tpwrs.2021.3086844}
}
```

Vai trò từng nguồn: **ali2024rocof** mang con số 2 Hz/s gắn với IEEE 1547-2018 cho microgrid —
đây là trích dẫn cho `rocof_max_hz_s`. **song2025** mang dải 0,5 Hz cho điều độ
distribution–microgrid, trích dẫn chính cho `f_min_hz`. **rebollal2021** cho việc neo tiêu chí
tần số vào *giảm thiểu kích hoạt UFLS* trong microgrid ốc đảo — đây là lập luận, không phải con
số. **xu2023ufls** cho UFLS trong microgrid ốc đảo **chỉ có một nguồn grid-forming**, tức đúng
cấu hình này. **gao2025dispatch** cho một dải thứ hai (±0,4 Hz) khi cần cho thấy 0,5 nằm giữa
dải chứ không ở mép. **javadi2021fscms** cho khung ba tiêu chí (RoCoF, nadir, overshoot) mà
điều độ microgrid dùng — dùng khi dựng bối cảnh.

Xem thêm Phần I cho `ninad2021`, `mahmud2022`, `narang2021`, `ieee2800`, `ninad2023`,
`amraee2018`, `brogan2019`, `xu2021`, `ruban2019`.

## II.7 Việc còn mở

- `v_max_pu` = 1,10 chưa được kiểm ở mức mà 0,88 vừa nhận. Trong các sự cố tăng tải nó không
  siết, nên không ảnh hưởng $\Delta P_{\max}$ hiện tại — nhưng sẽ ảnh hưởng nếu có ca mất tải.
- `settle_tol_hz` / `settle_window_s` là tiêu chí số học, phải phát biểu như vậy trong bài,
  không được để người đọc hiểu là ngưỡng chuẩn.
