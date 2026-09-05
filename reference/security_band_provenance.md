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

---

# Phần II — `f_min_hz` và `rocof_max_hz_s`: **quyết định đang mở**, ảnh hưởng 2,4×

Với $V_{\min}$ = 0,88, **nadir thành toạ độ siết**, nên `f_min_hz` giờ đặt trực tiếp
$\Delta P_{\max}$. Tra cứu (2026-09-05) cho ba phát hiện, mỗi cái nặng hơn cái trước.

## II.1 Dải hiện tại **không nhất quán về category**

IEEE 1547-2018 gán ngưỡng RoCoF theo category DER: **2 Hz/s cho Category II, 3 Hz/s cho
Category III**. Ta đang dùng:

| | ta dùng | category tương ứng |
|---|---|---|
| $V_{\min}$ = 0,88 | dải Continuous Operation LVRT | **Category III** |
| RoCoF = 2,0 Hz/s | ngưỡng ride-through | **Category II** ❌ |

Trộn hai category. Đây là khiếm khuyết độc lập với mọi thứ khác, và phải sửa dù chọn gì.

## II.2 `f_min` = 59,0 **không có đối ứng** trong IEEE 1547

Con số của 1547 là **ngưỡng cắt** (underfrequency trip), mặc định **57,0 Hz**, do area EPS
operator chỉnh và phải phối hợp với sơ đồ UFLS diện rộng. 59,0 không phải một mốc của tiêu
chuẩn này.

Với voltage ta chọn **sàn vùng vận hành liên tục** (0,88), không chọn điểm cắt. Đối ứng tần số
của "sàn vận hành liên tục" là **59,5 Hz** — nhưng nguồn hiện có nêu dải 59,5–60,5 Hz là
*"implied by trip ranges"*, tức **suy ra, không phải trích trực tiếp**. Chưa đủ để dùng.

## II.3 Hệ quả định lượng — dải 2,4×

Khớp tuyến tính trên 11 điểm chưa bão hoà của `T39_vmin088` (ΔP 0,05–1,52):

```
f_nadir = -0.6922*dP + 60.0011      rocof = 1.3229*dP + 0.0109
v_min   = -0.0691*dP +  0.9994      mu_I  = 0.2823*dP + 0.3104
```

ΔP mà từng tiêu chí siết:

| tiêu chí | ngưỡng | nguồn | ΔP siết [MW] |
|---|---|---|---:|
| f_nadir | 59,5 | sàn vận hành liên tục *(suy ra)* | **0,724** |
| f_nadir | **59,0** | **không có nguồn** | **1,446** |
| f_nadir | 57,0 | IEEE 1547 mặc định UF trip | 4,336 |
| RoCoF | 2,0 | IEEE 1547 **Cat II** | 1,504 |
| RoCoF | 3,0 | IEEE 1547 **Cat III** | 2,260 |
| $V_{\min}$ | 0,88 | IEEE 1547 Cat III cont. op ✅ | 1,727 |
| $\mu_I$ | 1,0 | ImaxF = 2,0 (REGFM_A1 ví dụ) | 2,443 |

Ba bộ ngưỡng **tự nhất quán**, và chúng cho ba bài báo khác nhau:

| bộ | $\Delta P_{\max}$ | siết bởi |
|---|---:|---|
| hiện tại (f 59,0 + RoCoF 2,0 CatII + V 0,88 CatIII) | **1,446** | nadir |
| toàn Cat III, gốc trip (f 57,0 + RoCoF 3,0 + V 0,88) | **1,727** | $V_{\min}$ |
| gốc vận-hành-liên-tục (f 59,5 + RoCoF 3,0 + V 0,88) | **0,724** | nadir |

**2,4× giữa cao nhất và thấp nhất.** Lớn hơn mọi hiệu ứng vật lý đo được trong chiến dịch này.

## II.4 Chất vấn phải trả lời trước khi chọn

Ngưỡng ride-through của DER trả lời câu *"khi nào một thiết bị **được phép** ngắt"*. Nó **không**
trả lời *"trạng thái vận hành nào chấp nhận được cho một microgrid ốc đảo"*.

57 Hz trên một feeder ốc đảo không phải điểm vận hành chấp nhận được bất kể inverter chịu được
bao nhiêu — chết máy động cơ, UFLS, phối hợp bảo vệ đều tác động trước đó rất lâu. Nên **IEEE
1547 có thể là họ tiêu chuẩn sai cho `f_min`**: neo đúng là một tiêu chí **vận hành/quy hoạch**
(dải tần khẩn cấp trong grid code, hoặc chính sơ đồ UFLS của microgrid), không phải một ngưỡng
ride-through thiết bị.

$V_{\min}$ = 0,88 không dính vấn đề này vì nó là **sàn vận hành liên tục**, tức trạng thái
thiết bị phải *chịu được liên tục* — gần với một tiêu chí vận hành hơn là một điểm cắt.

**Chưa chọn. Chưa đổi code.** `f_min_hz` = 59,0 và `rocof_max_hz_s` = 2,0 giữ nguyên cho tới
khi có quyết định, và mọi con số $\Delta P_{\max}$ công bố phải kèm bộ ngưỡng đã dùng.

## II.5 Trích dẫn bổ sung

```bibtex
@inproceedings{ninad2023mil,
  author    = {Ninad, Nayeem and Couture, E. D.},
  title     = {Assessment of a {DER} Inverter Model for {IEEE} 1547 Ride-Through
               Requirements Using a Model in the Loop Testbed},
  booktitle = {2023 IEEE 50th Photovoltaic Specialists Conference (PVSC)},
  pages     = {1--6}, year = {2023},
  doi       = {10.1109/pvsc48320.2023.10359898}
}

@article{amraee2018ufls,
  author  = {Amraee, Turaj and Darebaghi, M. G. and Soroudi, Alireza and Keane, Andrew},
  title   = {Probabilistic Under Frequency Load Shedding Considering {RoCoF} Relays
             of Distributed Generators},
  journal = {IEEE Transactions on Power Systems}, volume = {33}, pages = {3587--3598},
  year    = {2018}, doi = {10.1109/tpwrs.2017.2787861}
}

@article{brogan2019bess,
  author  = {Brogan, P. and Best, R. and Morrow, D. and McKinley, K. and Kubik, M.},
  title   = {Effect of {BESS} Response on Frequency and {RoCoF} During Underfrequency
             Transients},
  journal = {IEEE Transactions on Power Systems}, volume = {34}, pages = {575--583},
  year    = {2019}, doi = {10.1109/tpwrs.2018.2862147}
}

@article{xu2021support,
  author  = {Xu, Sheng and Xue, Yaosuo and Chang, Liuchen},
  title   = {Review of Power System Support Functions for Inverter-Based Distributed
             Energy Resources -- Standards, Control Algorithms, and Trends},
  journal = {IEEE Open Journal of Power Electronics}, volume = {2}, pages = {88--105},
  year    = {2021}, doi = {10.1109/ojpel.2021.3056627}
}

@inproceedings{ruban2019gridcodes,
  author    = {Ruban, N. and Kinshin, A. and Gusev, A.},
  title     = {Review of grid codes: Ranges of frequency variation},
  booktitle = {HMTTSC 2019}, year = {2019}, doi = {10.1063/1.5120686}
}
```

Vai trò: **ninad2021** mang cặp RoCoF 2/3 Hz/s theo category và mặc định UF trip 57,0 Hz.
**mahmud2022** cho việc UF/OF trip do EPS operator đặt và phải phối hợp UFLS — đây là nguồn cho
lập luận §II.4. **ninad2023** cho việc f_min và RoCoF là hai tiêu chí **song song, không phân
cấp**. **amraee2018** cho xung đột RoCoF-relay vs UFLS nadir — liên quan trực tiếp nếu chọn neo
UFLS. **brogan2019** cho ảnh hưởng của BESS lên f và RoCoF trong quá độ thiếu tần. **ruban2019**
cho dải grid code châu Âu (RoCoF 0,09–1 Hz/s; UF trip 47–48,5 Hz trên nền 50 Hz) — dùng khi cần
đối chiếu quốc tế. **xu2021** cho việc một số grid code dùng RoCoF làm **tín hiệu kích hoạt bổ
sung** cho điều khiển khẩn cấp, và cho nhận định 59 Hz là "biên mềm" trên hệ 60 Hz.

⚠️ **Cửa sổ đo RoCoF quan trọng và ta đã chọn nó:** tài liệu nêu cửa sổ 100–500 ms, cửa sổ ngắn
cho ước lượng nhiễu hơn, và **lựa chọn cửa sổ quyết định tiêu chí nào chạm trước**. Ta dùng
`rocof_window_s` = 0,5 s (`metrics.extract`), tức **đầu dài nhất của dải**. Đó là lựa chọn bảo
thủ theo hướng làm RoCoF *khó* siết hơn, và cũng cần khai báo.
