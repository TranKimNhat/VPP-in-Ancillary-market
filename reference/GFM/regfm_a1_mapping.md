# REGFM_A1 ↔ ANDES `REGF1` — bảng đối chiếu và việc cần làm

**Nguồn chuẩn:** Wei Du et al., *Model Specification of Droop-Controlled, Grid-Forming Inverters (REGFM_A1)*, PNNL-35110, 09/2023.
**Danh sách đóng góp gồm:** PNNL, UW-Madison, **NERC**, **WECC**, **EPRI**, **GE**, **Siemens PTI**, **PowerWorld**, PowerTech Labs, SMA Solar, PG&E, Portland General Electric.
**Nguồn đối chiếu:** `andes/models/renewable/regf1.py` (ANDES 2.0, bản trong `.venv` của dự án).

---

## 0. Kết luận một dòng

Mọi khối của REGFM_A1 **đã có** trong `REGF1`, trừ **một** khối: **giới hạn dòng (fault current limiting)**. Ba tham số hiện đang đặt **ngoài dải khuyến nghị** của bản đặc tả, và chúng là nguyên nhân của kết quả T20.

---

## 1. Câu hỏi "làm sao reviewer không nói đây không phải GFM BESS" — đã có lời giải trong một dòng của bảng

Bản đặc tả nói về BESS **đúng một lần**, ở hàng `Pmin`:

| Symbol | Description | Example Value | Normal Range |
|---|---|---|---|
| `Pmin` | Lower limit of the inverter active power output | 0 pu | **"Should be negative when representing energy storage systems"** |

Đó là toàn bộ. Trong thế giới mô hình positive-sequence đã chuẩn hoá, **một GFM BESS *là* một REGFM_A1 với `Pmin < 0`.** Không có DC link, không có SoC, không có hoá học pin trong đặc tả.

**Cấu hình hiện tại của bạn đã có `Pmin = -1.0`** → đã tuân thủ.

Cách viết trong bản thảo:

> *"Each GFM unit is represented by the REGFM_A1 droop-controlled grid-forming model [Du 2023], with `Pmin < 0` to represent bidirectional battery energy storage as specified therein. DC-link and state-of-charge dynamics are outside the model scope; on the 8-s horizon studied here, energy state enters only through the active-power limits, consistent with [T02 result]."*

Không cần dựng gì thêm. Đây là phần tiết kiệm công sức lớn nhất.

---

## 2. ⚠️ Ba phát hiện làm thay đổi kết quả T20

### 2.1 `I_max = 1.20` nằm **dưới đáy dải** của đặc tả

| | Giá trị bạn dùng | REGFM_A1 |
|---|---|---|
| `ImaxF` — *Inverter maximum transient output current* | **1.20 pu** | ví dụ **2 pu**, dải bình thường **1.5–3 pu** |

Toàn bộ bốn biên trong T20 do `μ_I` chạm 1,0 quyết định. Với `I_max = 2.0` thay vì 1,2, **cùng một dòng điện cho `μ_I = 0,60` thay vì 1,00** — biên dịch ra ngoài ít nhất 67%, và theo chính bảng B2 của bạn (nadir 58,96 và RoCoF 2,08 vi phạm tại ΔP = 1,23 MW), ràng buộc chi phối nhiều khả năng **chuyển sang tần số**.

**Quyết định cần ra và khai báo:** 1,2 pu là một giá trị hợp lý cho **định mức liên tục** (nhiệt). `ImaxF` của đặc tả là **giới hạn quá độ**, dùng bởi bộ giới hạn dòng. Đỉnh dòng trong nghiên cứu của bạn kéo dài vài trăm ms — tức chế độ quá độ. Dùng 1,2 pu và gọi nó là "converter current limit" trong khi trích dẫn REGFM_A1 là **không nhất quán**.

Đề xuất: dùng `ImaxF = 2.0` cho **bộ giới hạn** (theo đặc tả), và nếu vẫn muốn báo cáo mức 1,2 pu thì báo cáo nó như **ngưỡng vượt định mức liên tục**, tách khỏi tiêu chí an ninh.

### 2.2 Ba nguyên nhân của lỗi phân bổ Q — cả ba đều có giá trị chuẩn

Vấn đề G4 ở 82,5% định mức lúc `t = 0` có **ba** nguyên nhân cộng dồn, không phải một:

| # | Nguyên nhân | Bạn đang dùng | REGFM_A1 |
|---|---|---|---|
| **a** | **Khởi tạo power flow ép `v = 1,00` tại cả sáu GFM** | `v0 = 1.0` cho cả sáu (PV/Slack) | Q-V droop quyết định điện áp; `QVFlag = 1` → khởi tạo đặt `Qref = 0`; `QVFlag = 0` → `Qref = Qinv` |
| **b** | **Điện kháng ghép nối quá nhỏ** → ghép cứng → chia Q cực nhạy | `x_f_pu = 0.05` | `X_L` ví dụ **0,15 pu**, dải **0,05–0,25**. Đặc tả nói rõ `X_L` "is important for the droop controller design" và cỡ 0,05–0,15 để **tách P khỏi Q** |
| **c** | **Giới hạn Q quá rộng nên bộ hạn chế không bao giờ tác động** | `Qmax/Qmin = ±1.0` | ví dụ **±0,44 pu**, dải ±0,44 đến ±1 |

Nguyên nhân **(c) là bản vá rẻ nhất**: G4 hiện mang Q = 0,815 pu trên base thiết bị. Đặt `Qmax = 0.44` theo đặc tả thì bộ điều khiển `Qmax/Qmin` (`KPqlim`/`KIqlim`, đã có sẵn trong `REGF1`) sẽ cắt và buộc Q phân bổ lại — **một dòng cấu hình**.

Bản đặc tả nói thẳng về mục đích của Q-V droop:

> *"The Q-V droop control **prevents large circulating reactive power between grid-forming inverters**."*

Nghĩa là hiện tượng bạn quan sát được chính là thứ khối này sinh ra để ngăn — và nó không hoạt động vì cả ba lý do trên.

### 2.3 `KPplim = 5` nằm **5× trên đỉnh dải** — và nó nằm trên đường droop

Bổ sung từ T30/T31/T32. PNNL-35110 cho `kppmax` = 0,01 pu, dải bình thường **0,005–0,05**.

**⚠️ Phải quy đổi thứ nguyên trước khi so.** Sơ đồ P của REGFM_A1 cộng nhánh
`kppmax + kipmax/s` **thẳng vào ω**, cùng điểm với `mp` — nên `kppmax` có thứ nguyên
ω-trên-công-suất, giống `mp` (và đặc tả cho hai tham số này *cùng* dải 0,005–0,05). REGF1 thì
`PIplim_y` là **công suất**, chỉ nhân `w0·wdrp` về sau, nên `KPplim` **vô thứ nguyên**. Đại
lượng so được là

$$k_{ppmax}^{eff}=m_p\cdot K_{Pplim}=R\cdot K_{Pplim}=0{,}05\,K_{Pplim}$$

| `KPplim` | 0,05 | 0,1 | **0,2** | 0,5 | 1,0 | 2,0 | **5,0** |
|---|---|---|---|---|---|---|---|
| $k_{ppmax}^{eff}$ | 0,0025 | 0,005 | **0,010** | 0,025 | 0,050 | 0,100 | **0,250** |
| vs dải 0,005–0,05 | 2× dưới | đáy dải ✅ | **ví dụ đặc tả** ✅ | ✅ | đỉnh dải ✅ | 2× trên | **5× trên** ❌ |

Cửa sổ đúng đặc tả là `KPplim` ∈ **[0,1 – 1,0]**, ví dụ đặc tả là **0,2**. Giá trị 5,0 đang
dùng là mặc định ANDES, lệch **5×** (không phải 100× — bản sửa đầu của mục này so sai thứ
nguyên).

Điều khiến nó nghiêm trọng hơn `KIplim`: đầu vào droop của REGF1 là `PIplim_y − Psen_y`, nên
bộ PI giới hạn P nằm trên **đường thuận của droop kể cả khi không máy nào bão hoà**. Hàm
truyền dẫn xuất (`src/analytical/regf1_droop.py`):

$$\frac{\Delta\omega}{\Delta P_e}=-\,\omega_0 w_{drp}\cdot\frac{1+K_{Iplim}T_{pm}+s\,T_{pm}(1+K_{Pplim})}{(1+sT_r)(1+sT_{pm})}$$

Với `KIplim = 0`, zero nằm ở $-1/(T_{pm}(1+K_{Pplim}))$:

| `KPplim` | 0,1 (đáy dải) | **0,2 (ví dụ đặc tả)** | 1,0 (đỉnh dải) | 5 (mặc định ANDES) |
|---|---|---|---|---|
| zero [rad/s] | −36,4 | **−33,3** | −20,0 | **−6,67** |
| so với pole −40 | lệch 9% | lệch 17% | thấp hơn 2× | **thấp hơn cả hai pole** |

Zero chỉ triệt tiêu pole $-1/T_{pm}$ ở giới hạn $K_{Pplim}\to0$, tức **dưới** dải đặc tả. Trong
cửa sổ đặc tả nó vẫn là lead-lag nhẹ — nhưng ANDES đo $\kappa_{os}$ = 1,003–1,006 ở cả cửa sổ,
tức **vị trí zero hở mạch không dự đoán được ngưỡng overshoot**; ngưỡng nằm giữa `KPplim` = 2
và 5 và là tương tác vòng trong × đường droop (T32 §2③). Tại mặc định 5,0, overshoot đo được
22,7% — một đại lượng mọi mô hình GFM rút gọn dự đoán bằng 0.

$\kappa_{os}$ và RoCoF đều là hàm của `KPplim`, nên $\Delta P_{\max}=1{,}1851$ MW ứng với một
mặc định công cụ lệch 5× khỏi đặc tả. Phép tách cơ chế:
`experiments/t31_kpplim_mechanism.py`; bản đồ ổn định: `experiments/t32_eig_map.py`.
Chi tiết: `artifacts/T30_ducoin_crosscheck/README.md`, `artifacts/T32_eig_map/README.md`.

---

## 3. Bảng đối chiếu đầy đủ

| Khối REGFM_A1 | Tham số đặc tả (ví dụ / dải) | Tương ứng trong `REGF1` | Bạn đang dùng | Trạng thái |
|---|---|---|---|---|
| Điện kháng ghép nối | `X_L` = 0,15 (0,05–0,25) | `xf` (mặc định 0,2) | **0,05** | ⚠️ đáy dải |
| P–f droop | `m_p` = 0,01 (0,005–0,05) | `wdrp` (mặc định 0,033) | 0,05 → quy đổi qua `_wdrp` | ✅ trong dải |
| Q–V droop | `m_q` = 0,05 (0–0,20) | `Qdrp` (mặc định 0,045) | 0,045 | ✅ |
| Bộ điều khiển điện áp | `k_pv` = 0 (0–0,01), `k_iv` = 5,86 pu/s (3–15) | `KPv` (3), `KIv` (10) | KPv 3,0 / KIv 10,0 | ⚠️ `KPv` = 3 vs đặc tả 0–0,01 — cần kiểm |
| Giới hạn đầu ra vòng áp | `E_max` = 1,15 / `E_min` = 0 | — (không có giới hạn E tường minh) | — | ⚠️ thiếu |
| Giới hạn P + overload mitigation | `P_max` = 0,9 / `P_min` **âm cho BESS**; `k_ppmax` = 0,01 (dải **0,005–0,05**), `k_ipmax` = 0,1 pu/s (dải 0,05–0,2) | `Pmax`/`Pmin`, `KPplim`/`KIplim` (5 / 30) | Pmax 1,0 / Pmin −1,0; **KPplim 5**, **KIplim 0** | ❌ **so đúng thứ nguyên: $k_{ppmax}^{eff}=m_p K_{Pplim}=0{,}05\times5=0{,}25$, nằm 5× trên đỉnh dải** — xem §2.3 |
| Bộ điều khiển điện áp (thứ nguyên) | `k_pv`, `k_iv` lái thẳng `E_droop` (**áp→áp**, vô thứ nguyên) | `KPv`/`KIv` lái `Idref` (**áp→dòng**, dẫn nạp) | KPv 3,0 / KIv 10,0 | ⛔ **không so được** — REGFM_A1 là biến thể *không có vòng dòng trong*; hai gain tham số hoá hai plant khác nhau. Cờ "cần kiểm" ở hàng dưới đã giải quyết theo nghĩa này, không phải sai lệch |
| Giới hạn Q | `Q_max/Q_min` = **±0,44** (±0,44…±1); `k_pqmax` = 3, `k_iqmax` = 20 pu/s | `Qmax`/`Qmin`, `KPqlim` (0,1) / `KIqlim` (1,5) | **±1,0** | ⚠️ xem §2.2(c) |
| Lọc đo P/Q/V | `T_Pf`/`T_Qf`/`T_Vf` = 0,01 s (0,01–0,1) | `Tr` (0,005), `Tpm` (0,025) | Tpm 0,025 | ✅ trong dải |
| Quy đổi base | `S_base/M_base` (eq. 4–6) | `Sn` + `gammap`/`gammaq` | có | ✅ |
| Cờ chế độ | `VFlag` (POI hay E), `QVFlag` (Qref hay Vref) | — (không có cờ tương đương) | — | ⚠️ thiếu, ảnh hưởng khởi tạo |
| **Giới hạn dòng sự cố** | **`ImaxF` = 2 pu (1,5–3)** | ❌ **KHÔNG CÓ.** Có `PQFLAG` (ưu tiên P/Q), `Vdip`, `Tfrz` — nhưng **không có tham số `ImaxF` và không có giới hạn độ lớn dòng dạng vòng tròn** | 1,2 (hậu nghiệm) | ❌ **đây là khoảng trống duy nhất** |

**Lưu ý cấu trúc:** `REGF1` là mô hình **có vòng điều khiển trong** (`Id`, `Iq`, `PIvd`, `PIvq`, vòng dòng, `udref`/`uqref`). Bản đặc tả REGFM_A1 ở trên là biến thể **không có vòng trong** (nguồn áp sau `X_L`, giao tiếp mạng qua tương đương Norton). Bài TPWRD 2024 [2] bao phủ **cả hai** — *"inverters with and without inner control loops"* — nên `REGF1` vẫn nằm trong phạm vi trích dẫn được, chỉ cần nói rõ dùng biến thể nào.

---

## 4. Khối cần cài — thuật toán đầy đủ, không cần dẫn xuất lại

REGFM_A1 §3.3, và nó **thuần đại số, không thêm trạng thái**:

Ở mỗi bước, tính dòng ra từ điện áp trong của droop:

$$I\angle\varphi=\frac{E_{\text{droop}}\angle\delta_{\text{droop}}-V\angle\delta_V}{jX_L}$$

Rồi:

$$E\angle\delta_E=\begin{cases}
E_{\text{droop}}\angle\delta_{\text{droop}}, & |I| < I_{\max F}\\[4pt]
V\angle\delta_V + jX_L\,\bigl(I_{\max F}\angle\varphi\bigr), & |I| \ge I_{\max F}
\end{cases}$$

Đặc điểm của lựa chọn này: **giữ nguyên góc pha $\varphi$**, chỉ kẹp độ lớn. Khi sự cố qua đi, $|I|$ tụt xuống dưới $I_{\max F}$ và mô hình **tự động** quay lại chế độ droop — không cần logic thoát riêng.

Với `REGF1` (có vòng trong), điểm chèn tương đương là kẹp vector tham chiếu dòng:

$$\bigl(I_{d,\text{ref}},I_{q,\text{ref}}\bigr) \leftarrow \bigl(I_{d,\text{ref}},I_{q,\text{ref}}\bigr)\cdot\min\!\left(1,\ \frac{I_{\max F}}{\sqrt{I_{d,\text{ref}}^2+I_{q,\text{ref}}^2}}\right)$$

tức áp lên `Idref`/`Iqref` (là `AliasAlgeb` của `PIvd_y`/`PIvq_y` trong `REGF1`). Neo lý thuyết cho biến thể này: Bo Fan et al., *Equivalent Circuit Model of Grid-Forming Converters With Circular Current Limiter*, **IEEE TPWRS 2022, 135 trích dẫn** — chứng minh vòng trong khi bão hoà rút gọn thành nguồn áp sau một điện trở tương đương, nên phép kẹp này hợp lệ ở mức positive-sequence.

Ghi chú: `PQFLAG` của `REGF1` (0 = ưu tiên Q, 1 = ưu tiên P) là **chiến lược phân bổ khi kẹp** — nó giả định đã có một giới hạn để phân bổ. Hiện không có giới hạn nào nên cờ này không có tác dụng.

---

## 5. Việc cần làm, theo thứ tự

| # | Việc | Công sức | Tác động |
|---|---|---|---|
| **1** | Đặt `Qmax/Qmin = ±0.44` theo đặc tả | 1 dòng | Bộ hạn chế Q kích hoạt, ép Q phân bổ lại — bản vá rẻ nhất cho §2.2 |
| **2** | Nâng `x_f_pu` từ 0,05 lên **0,15** | 1 dòng | Giảm mạnh độ nhạy chia Q; vào giữa dải đặc tả |
| **3** | Đặt `ImaxF = 2.0`; tách "vượt định mức liên tục 1,2 pu" thành chỉ tiêu báo cáo riêng | 1 dòng + sửa `metrics.py` | Biên nhiều khả năng chuyển từ dòng sang tần số |
| **4** | Chạy lại bisection `gen_loss/constP` với (1)(2)(3) | 30 phút | **Cho biết kết quả T20 nào là vật lý** |
| **5** | Cài khối giới hạn dòng vào `REGF1` (custom model) | 1–2 tuần | `μ_I` thành **động** thay vì hậu nghiệm; mô hình đủ tư cách gọi là REGFM_A1-conformant |
| **6** | Kiểm `KPv = 3` so với đặc tả `k_pv ∈ [0; 0,01]` | 1 giờ | Có thể là khác biệt giữa hai biến thể (có/không vòng trong) — cần đọc [2] để xác nhận, không sửa mù |
| **7** | Bảng tham số trong bản thảo, mọi hàng dẫn về PNNL-35110 | — | Phòng thủ trước phản biện |

**Bước 1–4 làm được trong một buổi và có thể đảo ngược kết luận số 1 của T20.** Bước 5 là việc thật nhưng chỉ là một khối.

---

## 6. Ghi vào concept §25 / plan

- **D7 — mô hình GFM:** REGFM_A1 (PNNL-35110), biến thể có vòng điều khiển trong theo [2]; `Pmin < 0` biểu diễn BESS; DC-link và SoC ngoài phạm vi, lý do là tách thang thời gian, bằng chứng là T02.
- **Mọi tham số GFM phải nằm trong dải của Bảng 1 PNNL-35110**, và lệch khỏi dải phải có lý do ghi trong bài.
- **R9 (mới):** kết quả T20 được đo ở `I_max = 1,2` (dưới dải đặc tả) và `Qmax = ±1,0` (rộng hơn đặc tả) → không trích dẫn cho tới khi chạy lại theo §5 bước 1–4.

---

## 7. Kết quả chạy lại (bước 1–4 đã làm xong)

Hiện vật: `artifacts/T21_genloss_constP_regfm_q060/` (chính) và `..._q044/` (độ nhạy).
Cài đặt: `src/phasor/build_case.py`, `src/phasor/metrics.py`, `experiments/t20_andes_bisect.py`.

**`ΔP_max` (gen_loss, P không đổi): 0,5801 → 1,1851 MW (+104%), toạ độ siết chuyển từ `μ_I`
sang tần số** — nadir chạm 59,00 Hz và RoCoF chạm 2,00 Hz/s cùng lúc, trong khoảng ±0,006 MW.
Tại biên `μ_I`(ImaxF) = 0,68 và `μ_P` = 0,32. Dự đoán ở §2.1 đúng.

Ba đính chính đối với §2 và §5, đều do phép đo:

**(i) §5 bước 1 — `Qmax = 0,44` một mình là lệnh rỗng.** ANDES mặc định (a) `PV.pv2pq = 0`
nên power flow bỏ qua `qmax`, và (b) `REGF1.config.adjust_upper = 1` nên lúc khởi tạo TDS nó
**nới** `Qmax` lên đúng Q mà power flow đưa vào, im lặng, theo từng máy: khai báo 0,44 trở
thành `[0,44 0,49 0,44 0,82 0,46 0,66]`. Phải tắt cả hai (`CaseSpec.enforce_q_limits`) thì
trần mới tồn tại. Đây không phải "một dòng cấu hình".

**(ii) §2.2(b) sai — `x_f` không dính đến chia Q.** Ở cả `x_f = 0,05` và `0,15`, Q ban đầu
giống nhau đến bốn chữ số: `[0,253 0,494 0,389 0,822 0,464 0,663]` pu. Đầu cực converter là
nút PV/Slack ghim ở v = 1,0, nên chia Q do mạng *phía ngoài* đầu cực quyết định (`x_tr` =
0,06 + feeder), không do điện kháng nằm *sau* đầu cực. Nguyên nhân thật chỉ có (a). Nâng
`x_f` lên 0,15 vẫn nên làm — nhưng vì tuân thủ Bảng 1, không vì phân bổ Q.

**(iii) 0,44 là đáy dải theo nghĩa vật lý, không chỉ theo bảng.** Nhu cầu Q thường trực của
feeder ở điểm điều độ này là 1,921 MVAr trên 4,362 MVA thiết bị = **0,4403 pu**, trùng gần
đúng trần ví dụ. Đặt 0,44 làm cả sáu máy nằm đúng trên trần (độ lệch chia Q 0,569 → 0,001 pu,
nhưng bằng bão hoà), và Slack vượt trần 0,0011 pu → `Initialization FAILED` mỗi lần chạy.
Dùng **0,60**. Biên `ΔP_max` giống hệt ở cả hai giá trị, nên lựa chọn này không ảnh hưởng kết
luận.

**R9 gỡ được một phần:** kết quả T20 cũ (`ΔP_max = 0,58 MW`) **không** phải biên ổn định — nó
là ngưỡng vượt định mức dòng *liên tục* của một máy bị chia Q lệch. Không trích dẫn. Thay bằng
T21. Bước 5 (khối limiter) vẫn còn nợ: `μ_I` vẫn hậu nghiệm, nên mọi điểm `μ_I > 1` (ΔP ≳ 1,8
MW) vẫn ngoài miền hợp lệ — nhưng chúng đã nằm ngoài biên tần số nên không đụng tới `ΔP_max`.
