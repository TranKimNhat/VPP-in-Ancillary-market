# T34 — chạy lại toàn bộ biên an ninh tại cấu hình ship C

**Đổi duy nhất so với các artifact đã công bố:** ba default trong `src/phasor/build_case.py`.

```
kp_plim  5.0  -> 1.0     kppmax_eff = m_p*KPplim = 0.050, trong dải REGFM_A1 0,005-0,05
kp_i     0.20 -> 0.10    (5.0 cũ nằm 5x trên dải)
ki_i     5.0  -> 3.0
```

Cả ba **không phải CLI argument**, nên mọi chiến dịch dưới đây tự lấy từ `CaseSpec`. Mọi cờ
khác tái lập nguyên bản gốc, khôi phục từ chính `config.yaml` của từng chiến dịch. Artifact cũ
**không bị ghi đè** — chúng là số đã công bố và phép so là sản phẩm.

Cơ sở: `artifacts/T30_ducoin_crosscheck` (phát hiện), `T31_kpplim_mechanism` (cơ chế),
`T32_eig_map` (chọn cấu hình), `T33_inner_revalidation` (cổng ổn định).

---

## 1. Bảng kết quả

| chiến dịch | đại lượng | công bố | ship C | đổi |
|---|---|---:|---:|---:|
| `dpmax_q060` | $\Delta P_{\max}$ | 1,185059 | **1,438574** | **+21,4%** |
| `dpmax_imaxf15` | $\Delta P_{\max}$ ($I_{\max F}$=1,5) | 1,185059 | **1,438574** | +21,4% |
| `phead_dp1p1` | $P_{head}^{\min}$ @ΔP=1,1 | 1,094680 | **1,094680** | **0,0%** |
| `phead_dp0p6` | $P_{head}^{\min}$ @ΔP=0,6 | 0,595336 | **0,595336** | **0,0%** |
| `pdgoff_h1p0` | $P_{DG,off}^{\max}$ (H=1,0) | 1,208594 | **1,296094** | +7,2% |
| `pdgoff_h0p1` | $P_{DG,off}^{\max}$ (H=0,1) | 1,208594 | **1,296094** | +7,2% |
| `pdgoff_gast` | $P_{DG,off}^{\max}$ (GAST) | 1,208594 | **1,296094** | +7,2% |

## 2. Cái sống nguyên vẹn

**$P_{head}^{\min}$ và bao hiệu lực $\kappa$ không đổi một chữ số nào** — 1,094680 và 0,595336,
với $\kappa$ = 1,004860 và 1,007834 y hệt. Đây là **xác minh**, không còn là giả định: biên
headroom là ràng buộc **khả thi năng lượng**, không phải động học, nên nó miễn nhiễm với thay
đổi bộ điều khiển. Giả thuyết này từng được nêu mà chưa kiểm; giờ đã kiểm.

**Quán tính và họ governor của diesel vẫn không đổi biên**: H = 1,0 / H = 0,1 / GAST cho
**cùng** 1,296094, đúng như cả ba cho cùng 1,208594 trước đây. Kết luận D2b sống.

**Giới hạn dòng vẫn không siết**: $I_{\max F}$ = 1,5 cho cùng biên như 2,0.

## 3. Cái đổi tính chất — tiêu chí siết

Tại điểm SECURE cuối, tỉ lệ ngưỡng đã dùng:

| | công bố | ship C |
|---|---:|---:|
| RoCoF | **0,9994** | 0,9512 |
| nadir | 0,9956 | 0,9908 |
| **v_min** | 0,8038 | **0,9965** |
| $\mu_I$ | 0,6759 | 0,7139 |

Điểm insecure đầu tiên ở ship C (ΔP = 1,4443) hỏng **chỉ vì** `v_min` = 0,8996 < 0,90, với
RoCoF còn ở 1,918/2,0.

Cơ chế: đổi bộ điều khiển **nới hai tiêu chí tần số nhưng không nới tiêu chí điện áp**, nên
biên đi ra cho tới khi điện áp bắt kịp — v_min từ 80% ngưỡng lên 99,65%.

**Phát biểu đúng không phải "biên thành giới hạn điện áp"** mà: biên giờ do một đại lượng
**cục bộ** siết, sát ngay sau là một đại lượng khối (nadir kém 0,6%, RoCoF kém 4,5%). Vì
khoảng cách dưới 1%, vị trí sự cố hoặc topology **có thể lật tiêu chí siết giữa các case** —
T35 phải trả lời, và vì thế nó không còn là kiểm tra tuỳ chọn.

## 4. ❌ Một đóng góp đã công bố **đảo dấu**

T25 (`T25_pdgoff_h1p0/README.md:15`) phát biểu:

> *"Mất diesel cuối **dễ hơn** mất một nguồn phi đồng bộ cùng cỡ 2,0% — ngược dấu với dự đoán."*

| | $P_{DG,off}^{\max}$ | $\Delta P_{\max}$ | hiệu |
|---|---:|---:|---:|
| công bố | 1,208594 | 1,185059 | **+1,99%** — diesel **dễ hơn** |
| ship C | 1,296094 | 1,438574 | **−9,90%** — diesel **khó hơn** |

Kết luận "ngược dấu với dự đoán" **là hệ quả của một tham số ngoài đặc tả**. Ở cấu hình đúng
đặc tả, dấu quay về đúng chiều trực giác. Đóng góp này phải viết lại, không phải chỉnh số.

## 5. $I_{\max F}^{crit}$ tính lại

Định nghĩa: giá trị $I_{\max F}$ mà tại đó giới hạn dòng bắt đầu siết đúng ở biên.

| | $\mu_I$ tại biên | $I_{dev}$ = $I_{\max F}^{crit}$ |
|---|---:|---:|
| công bố | 0,6759 | **1,3519** (số trên hồ sơ: 1,356 — khớp 0,3%) |
| ship C | 0,7173 | **1,4346** |

**+6,1%.** Vẫn dưới đáy dải REGFM_A1 (1,5–3,0), nên kết luận §28 N3 *"không limiter hợp lệ nào
siết"* **sống** — và sống với biên rộng hơn trước.

## 6. Bảng trạng thái đóng góp

| | |
|---|---|
| $\Delta f_{ss}$ dạng đóng | ✅ nguyên vẹn — `KPplim` không có trong DC gain |
| $\Lambda \in [4,14]$ | ✅ nguyên vẹn — thuộc tính mạng |
| $P_{head}^{\min}$, bao $\kappa$ | ✅ **không đổi một chữ số** — nay là xác minh, không phải giả định |
| $I_{\max F}$ không siết (T24) | ✅ sống, biên rộng hơn (1,4346 vs 1,5) |
| Diesel H / governor không đổi biên | ✅ sống |
| $\Delta P_{\max}$ = 1,1851 MW | ⚠️ **→ 1,4386 MW (+21,4%)** |
| $P_{DG,off}^{\max}$ = 1,2086 MW | ⚠️ → 1,2961 MW (+7,2%) |
| $\kappa_{os}$ = 1,2275 | ⚠️ → 1,0046; phải viết thành $\kappa_{os}(K_{Pplim})$ |
| $I_{\max F}^{crit}$ = 1,356 | ⚠️ → 1,4346 |
| **Diesel-off "+2,0%, ngược dấu"** | ❌ **đảo dấu → −9,90%** |
| Bất biến topology / vị trí (T22/T23) | ⏳ **T35** — tiền đề đã mất |

---

`experiments/t34_rerun_ship_c.sh` · 7 biên, mỗi biên 13–14 lần dò
