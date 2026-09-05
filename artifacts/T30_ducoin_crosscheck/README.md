# T30 — đối chiếu Ducoin: cổng Task 0 và phán quyết K1–K4

**Nguồn chuẩn:** E. A. S. Ducoin, Y. Gu, B. Chaudhuri, T. C. Green, *Analytical Design of
Contributions of Grid-Forming and Grid-Following Inverters to Frequency Stability*,
IEEE Trans. Power Systems **39**(5), 09/2024 — `reference/Analytical Design of
Contributions of Grid-Forming and Grid-Following Inverters to Frequency Stability.pdf`.
Eq. (2)–(5), (21), (32), (33)–(34).

**Nguồn đối chiếu:** `andes/models/renewable/regf1.py` (ANDES 2.0, bản trong `.venv`),
`andes/core/block.py` (`PIController`, `GainLimiter`).

**Không có run ANDES mới.** Mọi số lấy từ `artifacts/T23_event_location_sweep/metrics.csv`
và `artifacts/T26_phead_dp1p1/raw/*.npz`.

---

## 0. Kết luận một dòng

**Chuỗi kế thừa chết ở tầng mô hình, không phải tầng tham số.** Đường droop của REGF1 là
lead-lag; mô hình GFM của Ducoin là lag một pole. Không tồn tại $\omega_{Pf}$ nào để lấy giá
trị, nên $H^{GFM}=1/(2m_P\omega_{Pf})$ không có đối tượng và K2/K3 mất căn cứ đối chứng.
K1 và K4 vẫn đứng. Và cổng Task 0 bắt được **bốn** sai trong thiết kế T30 trước khi viết
dòng code nào.

**Hệ quả bắt buộc:** không được viết *"ANDES triển khai mô hình đã được Ducoin kiểm chứng
bằng EMT."* Câu đó sai.

---

## 1. Cổng Task 0 đã bắt được gì

| đề xuất trong thiết kế T30 | thực tế |
|---|---|
| $\omega_{Pf}=1/T_{pm}$ "nhiều khả năng" | Sai **vị trí**. `Tpm` nằm trong vòng giới hạn P; `Tr` mới là $T_{Pf}$ của REGFM_A1 |
| K2 ngưỡng đạt 20% | Không có đại lượng để đo — $H_T^{GFM}$ không tồn tại |
| K3 là đối chứng với Ducoin | Tự quy chiếu: $\tau_{eq}$ nhận dạng từ ANDES rồi kiểm bằng chính nó |
| K4 dùng lại T23 | T23 là tản theo **vị trí sự cố**, không theo bus, và không có `raw/` |

---

## 2. Hàm truyền dẫn xuất — tài sản tái dùng

Mã: `src/analytical/regf1_droop.py`. Dẫn xuất từ `regf1.py:242–266, 308`:

```
Psen   = Lag(Pe, T=Tr)                                  # bộ lọc đo (REGFM_A1 T_Pf)
Psig   = LagAntiWindup(Psen + Paux, T=Tpm, Pmin, Pmax)  # trong vòng giới hạn P
PIplim = PIController(Psig - Psen, kp=KPplim, ki=KIplim, x0=Psen)
dw     = w0 * wdrp * (PIplim_y - Psen_y)
```

Đầu vào droop là hiệu `PIplim_y − Psen_y`, **không** phải `Pref − Pe`: bộ PI giới hạn P nằm
trên đường thuận của droop kể cả khi không máy nào bão hoà. `PIController.x0` chỉ là điều
kiện đầu của tích phân, không phải feedforward (`block.py:374–379`); `GainLimiter(K=1,R=1)`
khi chưa bão hoà là passthrough.

$$\frac{\Delta\omega}{\Delta P_e}=-\,\omega_0 w_{drp}\cdot\frac{1+K_{Iplim}T_{pm}+s\,T_{pm}(1+K_{Pplim})}{(1+sT_r)(1+sT_{pm})}$$

**Kiểm chứng độc lập.** DC gain cho droop thực hiện $R(1+K_{Iplim}T_{pm})$, trùng **5/5**
điểm đo đã có trong docstring `build_case._wdrp` tới 4 chữ số:

| $K_{Iplim}$ | 30 | 10 | 4 | 1 | 0 |
|---|---|---|---|---|---|
| mô hình | 0,0875 | 0,0625 | 0,0550 | 0,0512 | 0,0500 |
| đo | 0,0875 | 0,0625 | 0,0550 | 0,0513 | 0,0500 |

Hàm truyền đúng, không phải giả thuyết.

**Cấu trúc** với cấu hình đang chạy ($K_{Iplim}=0$, $K_{Pplim}=5$): zero tại $-6{,}67$ rad/s,
poles tại $-40$ và $-200$ rad/s. Zero **thấp hơn cả hai pole** ⇒ lead-lag, khuếch đại dải
giữa **4,2× DC**. Ducoin Eq. (3)–(5) là lag một pole thuần — gain giảm đơn điệu. Hai cấu
trúc khác nhau, không phải cùng cấu trúc khác tham số.

**Điểm suy biến.** Tại $K_{Pplim}=0$ zero rơi đúng $-1/T_{pm}$ và **triệt tiêu pole đó**:

$$\frac{\Delta\omega}{\Delta P_e}\Big|_{K_{Pplim}=K_{Iplim}=0}=-\frac{\omega_0 w_{drp}}{1+sT_r}$$

tức REGF1 **trở thành đúng** mô hình Ducoin với $\omega_{Pf}=1/T_r=200$ rad/s. Tại giá trị
đặc tả REGFM_A1 $k_{ppmax}=0{,}01$ zero ở $-39{,}6$ rad/s — triệt tiêu tới 1%. Đây là cơ sở
của T31.

---

## 3. Ánh xạ tham số — ba ứng viên, và khoảng cách

Fit **chính Eq. (33) của Ducoin** (dạng chính xác cho hệ toàn IBR không PLL, tức đúng cấu
hình này) lên 13 điểm chưa bão hoà, đã settled của T23, trải 36× về $\Delta P$:

| | $2H_T^{GFM}/D_T$ | $H_T^{GFM}$ trên 4,362 MVA | lệch |
|---|---:|---:|---:|
| **nhận dạng từ ANDES** | **0,2613 ± 0,0015 s** | **2,613 s** | — |
| $\omega_{Pf}=1/T_{pm}$ (giả thuyết T30) | 0,025 s | 0,250 s | **10,4×** |
| $\omega_{Pf}=1/T_r$ (REGFM_A1 $T_{Pf}$) | 0,005 s | 0,050 s | 52× |
| $\omega_{Pf}=(1{+}K_{Pplim})/T_r$ (RoCoF $t=0^+$ đúng) | 0,00083 s | 0,0083 s | 315× |

$D_T$ không phải vấn đề: $\sum S_g/(f_0R)=1{,}4540$ MW/Hz, và K1 dưới đây xác nhận nó.

---

## 4. Phán quyết K1–K4

| | phán quyết | số đo |
|---|---|---|
| **K1** $\Delta f_{ss}$, Eq. (21) | ✅ Chạy được. **Không tính điểm** — đồng nhất cấu trúc như đã đăng ký, vì $D_T^{GFL}=D^{SG}=K_T=0$ khiến Eq. (21) và công thức của ta trùng nhau theo đại số | max\|df_ss_andes − df_ss_ana\| = **5,9 × 10⁻⁴ Hz** / 15 điểm |
| **K2** RoCoF, Eq. (32) | ❌ **Không chạy được.** Eq. (32) cần $H_T^{SG+GFM}$; không có giá trị nào để đưa vào | — |
| **K3** settling, Eq. (34) | ⚠️ Chạy được nhưng **tự quy chiếu**: $\tau_{eq}$ nhận dạng từ ANDES rồi kiểm bằng chính nó. Là kiểm nhất quán nội bộ, không phải đối chứng | $\tau_{eq}=0{,}2613\pm0{,}0015$ s |
| **K4** tản theo bus, Eq. (14) | ✅ **Đạt**, trên nền vững hơn thiết kế giả định | **0,034%** của $\Delta f_{ss}$ |

**K4 vì sao vững hơn.** Lý do không phải $D_i/H_i$ đều. Cả sáu máy dùng chung
$T_r/T_{pm}/K_{Pplim}/K_{Iplim}$ và $w_{drp}\propto 1/S_i$ ⇒ compensator chuẩn hoá **đồng
nhất** ⇒ số hạng $\sum_i D_i\omega_{osc,i}$ triệt tiêu với **mọi** $C(s)$, không riêng bậc
một. Đây là dự đoán cấu trúc duy nhất của Ducoin mà ANDES tái tạo, và nó không phụ thuộc
ánh xạ đã chết.

**Sửa thiết kế:** K4 **không** dùng lại được T23. Con số 0,07% của T23 là phân tán qua *vị
trí nhiễu*, không phải qua bus, và `T23_event_location_sweep/` không có `raw/`. Số 0,034% ở
trên đo từ `T26_phead_dp1p1/raw/*.npz`, nơi `f_hz` là (time × 10 bus).

---

## 5. Cái thu được, lớn hơn cái mất

**$\kappa_{os}$ có ý nghĩa cấu trúc, không phải hằng số kinh nghiệm.**

$$\kappa_{os}=\frac{f_0-f_{nadir}}{f_0-f_{ss}}=1{,}227$$

ổn định 1,220–1,235 qua 13 điểm; đo trên COI thật (tản theo bus chỉ 0,034% nên không phải
hiệu ứng $\omega_{osc}$). Ducoin Eq. (33) là **bậc một** cho đúng cấu hình này ⇒ đáp ứng bước
đơn điệu ⇒ $\kappa_{os}\equiv 1{,}000$.

$$\boxed{\kappa_{os}-1=0{,}2275\ \text{là phép đo định lượng mức vi phạm giả thiết}}$$

Giả thiết bị vi phạm được Ducoin phát biểu ngay trên Eq. (2): *"It is assumed that the inner
control loops of inverters track their references perfectly."* REGF1 có vòng trong thật, và
nghiên cứu này đã cố ý làm chậm nó (KPi 0,5→0,20; KIi 20→5,0) để giữ $\mathrm{Re}(\lambda)\le0$.

Mọi mô hình GFM rút gọn hiện hành mô hình GFM bằng cặp quán tính–damping, và cặp đó **không
sinh overshoot theo cấu trúc**. Nên đây là một đại lượng mà cả lớp mô hình dự đoán bằng 0.

### ⚠️ Cảnh báo diễn giải — bắt buộc kèm khi trích $\tau_{eq}$

$\kappa_{os}=1{,}227>1$ chứng minh hệ **ít nhất bậc hai**, nên khớp mô hình bậc một Eq. (33)
vào nó là *misspecified*. $\tau_{eq}$ phải báo cáo là **hằng số thời gian pole trội**, không
phải $H_{eq}$. **Không được gọi nó là quán tính.** Xem docstring
`regf1_droop.tau_from_windowed_rocof`.

Mệnh đề còn lại vẫn đứng và chạm đúng luồng *aggregation validity*: **động học tổng hợp của
đội không tính được từ tham số của các thành viên.**

---

## 6. Phát hiện phụ, mức nền tảng: $K_{Pplim}=5$ nằm ngoài đặc tả 5×

PNNL-35110 (`reference/GFM/Model Specification of Droop-Controlled, Grid-Forming Inverters
(REGFM_A1).pdf`), bảng tham số: `kppmax` = 0,01 pu (dải **0,005–0,05**), `kipmax` = 0,1 pu/s
(dải 0,05–0,2).

**⚠️ Sửa sau T32 — phải quy đổi thứ nguyên trước khi so.** Bản đầu của mục này ghi lệch 100×
bằng cách so thẳng $K_{Pplim}$ với `kppmax`. Sai: REGFM_A1 cộng nhánh overload **thẳng vào ω**
(cùng điểm với $m_p$, nên đặc tả cho hai tham số *cùng* dải 0,005–0,05), còn REGF1 cộng vào
**công suất** rồi mới nhân $\omega_0 w_{drp}$. Đại lượng so được:

$$k_{ppmax}^{eff}=m_p\cdot K_{Pplim}=R\cdot K_{Pplim}=0{,}05\,K_{Pplim}$$

| $K_{Pplim}$ | **0,1** | **0,2** | **1,0** | 2,0 | **5,0** |
|---|---|---|---|---|---|
| $k_{ppmax}^{eff}$ | 0,005 | **0,010** | 0,050 | 0,100 | **0,250** |
| vs 0,005–0,05 | đáy ✅ | **ví dụ đặc tả** ✅ | đỉnh ✅ | 2× trên | **5× trên** ❌ |

Cửa sổ đúng đặc tả là $K_{Pplim}\in[0{,}1;\,1{,}0]$; giá trị đang dùng (5,0) là **mặc định của
ANDES**, lệch **5×**. `regfm_a1_mapping.md` §3 đã liệt kê cặp gain này nhưng chỉ gắn cờ
`KIplim`; `KPplim` lọt lưới.

Đường lan truyền tới biên an ninh là trực tiếp:

$$\Delta P_{\max}=\frac{(f_0-f_{\min})\sum S_g}{\kappa_{os}\,f_0R}$$

$\kappa_{os}$ nằm ở mẫu số. Nếu $\kappa_{os}=\kappa_{os}(K_{Pplim})$ thì
$\Delta P_{\max}=1{,}1851$ MW là hàm của một mặc định công cụ, và cùng với nó là T22–T26.
**Đây là lý do T31 được ưu tiên trên mọi việc đang mở.**

---

## 7. Hồ sơ phòng thủ thành cái gì

Yếu hơn thiết kế, hình dạng tốt hơn. Không còn *"chúng tôi khớp Ducoin"*, mà:

> ANDES tái tạo dự đoán cấu trúc về triệt tiêu tản tần số theo bus của mô hình đã được EMT
> kiểm chứng (0,034%), trùng biểu thức xác lập theo đồng nhất đại số (5,9 × 10⁻⁴ Hz trên 15
> điểm), và lệch khỏi nó ở **một chỗ định vị và quy trách nhiệm được**: dạng rút gọn ép
> overshoot bằng 0 theo cấu trúc do giả thiết vòng trong lý tưởng; chúng tôi đo 22,7% và
> [T31 nói cơ chế].

Một sai khác được giải thích mạnh hơn một sự trùng khớp không giải thích.

**Dấu vết trong bài:** một câu trong mục giới hạn phạm vi + một đoạn response letter. Không
sửa `concept`, không vào §16.

---

## 8. Hiện vật

| | |
|---|---|
| `src/analytical/regf1_droop.py` | hàm truyền + điểm suy biến + nhận dạng $\tau$ |
| `artifacts/T30_ducoin_crosscheck/README.md` | tài liệu này |
| `experiments/t31_kpplim_mechanism.py` | phép tách cơ chế (T31) |

**Chưa làm, cố ý:** không có `ducoin_swing.py` triển khai Eq. (32)/(34) như phép kiểm — vì
không có $H^{GFM}$ để đưa vào. Nếu T31 cho $\kappa_{os}\to1$ tại $K_{Pplim}$ đặc tả thì ánh
xạ Ducoin **sống lại** ở cấu hình conformant, và khi đó module đó mới có nghĩa.
