# T50 — dựng cơ chế Yang vào mô hình, rồi đo. $\lambda_{\min}$ **không có nội dung** trên hệ này

**Điều T49 chưa trả lời được.** T49 đo tương quan $\lambda_{\min}$ với trị riêng ANDES và được
+0,14. Nhưng đó là đo **sự vắng mặt của một cơ chế**, không phải sự vắng mặt của một hiệu ứng:
Yang, Xu, Zhang & Sun (IEEE TPWRS 2020) chứng minh đặt GFM tương đương nâng độ cứng lưới **cho
converter dựa PLL**, và mô hình khi đó không có PLL nào. 16 `sgen` — mang **2,88 MW trên 3,49 MW
tải, tức 82,5% nguồn** — là tải âm hằng công suất, **không động học gì cả**.

`CaseSpec.gfl_dynamic` nay dựng 16 đơn vị đó thành **REGCP1 + PLL1** trên PV static gen ghim Q.

| | tĩnh (T1–T49) | động (T50) |
|---|---|---|
| trạng thái | 108 | **220** (= 108 + 16×3 REGCP1 + 16×4 PLL1) |
| $\sum|V|$ sau power flow | 129,949976 | **129,949976** — trùng từng chữ số |
| Q của đội GFL | 0 | **−0,000000 MVAr** (ghim `qmax = qmin = q0`) |

Điểm vận hành **không đổi**, nên mọi chênh lệch trị riêng là do PLL, không do power flow khác.

---

## Ngưỡng đăng ký trước, trên nhánh động

| | điều kiện | nghĩa |
|---|---|---|
| **H1** đúng | $\mathrm{corr} \ge +0{,}7$ **và** tản $\zeta_{\min} \ge 20\%$ trung bình | Yang đứng ở đây; vị trí có nội dung định hướng; **bài toán phân bổ tồn tại** |
| **H2** sai dấu | $\mathrm{corr} \le -0{,}7$, cùng tản | cơ chế có, chạy ngược Yang trên feeder này |
| **H0** không nội dung | $|\mathrm{corr}| < 0{,}4$ **hoặc** tản $< 5\%$ | $\lambda_{\min}$ nhập bọn với bốn khung đã chết |
| — | còn lại | **không phân giải được ở độ phân giải này** — khác với "không có" |

Trục thứ hai không phải tham số tự do: PLL1 tích phân $2\pi f_n(K_p e + K_i\!\int\! e)$, nên
crossover $\approx 60 K_p$ Hz. $K_p = 0{,}5$ là PLL 30 Hz (dải thật của inverter phân phối),
1,0 và 5,0 là điểm ép. $K_i = K_p$, đúng tỉ số PLL1 ship.

## Kết quả — 16 cách đặt, $\lambda_{\min}$ trải **13,2×** (2,96 → 39,21)

| nhánh | $\zeta_{\min}$ | tản/trung bình | pearson | spearman | phán quyết |
|---|---|---:|---:|---:|---|
| tĩnh (đối chứng T49) | 0,1559 – 0,1589 | 0,019 | +0,122 | +0,053 | **H0** |
| PLL $K_p$ = 0,5 | 0,1551 – 0,1551 | **0,000** | −0,060 | −0,176 | **H0** |
| PLL $K_p$ = 1,0 | 0,1095 – 0,1095 | **0,000** | −0,096 | −0,179 | **H0** |
| PLL $K_p$ = 5,0 | 0,0489 – 0,0489 | **0,000** | −0,279 | −0,491 | **H0** |

Nhánh động **bất biến chặt hơn nhánh tĩnh**: bốn chữ số thập phân giống hệt nhau qua cả 16 cách
đặt. Đây không phải nhiễu quanh một xu hướng yếu — không có xu hướng nào.

## Và lần này có **cơ chế**, không phải một chỉ số nữa

Cơ chế **có mặt thật** trong mô hình, không bị vô hiệu hoá: `zam` = 1 ở cả 16 thiết bị, khối PLL
ghép với phần còn lại ở chuẩn $\lVert A_{\text{pll},\text{oth}}\rVert_\infty$ = 6,0 và
$\lVert A_{\text{oth},\text{pll}}\rVert_\infty$ = 12,4 s⁻¹ — **khác không**.

Nhưng mode tới hạn ở 13,7 Hz **sống gọn trong một thiết bị**:

| tham gia | trạng thái |
|---:|---|
| 0,47 | `ae PLL1 PLL 34` |
| 0,47 | `am PLL1 PLL 34` |
| 0,01 | `ae PLL1 PLL 42` |

**95% trong ba trạng thái, 94% trong hai trạng thái của một PLL duy nhất.** Đó là mode cục bộ
của thiết bị, không phải mode mạng. (16 PLL cùng tham số nên 16 bộ mode suy biến; "bus 34" là
lựa chọn tuỳ ý của vector riêng, ý nghĩa nằm ở chỗ nó **đơn thiết bị**.)

Lý do, có số kèm theo:

$$\underbrace{2\pi f_n K_p = 377\ \mathrm{s^{-1}}}_{\text{tự khuếch đại của PLL}}
\;\gg\; \underbrace{12{,}4\ \mathrm{s^{-1}}}_{\text{ghép mạng lớn nhất}}
\qquad \text{tỉ số } \mathbf{30{,}4\times}$$

Mode do chính bộ chỉnh định PLL đặt ra. Mạng không đủ lực để dịch nó.

## Trở kháng nào đặt ra ghép đó — **không phải feeder**

| lever | ghép s⁻¹ | tỉ số | $\zeta_{\min}$ | f Hz | mode nằm ở |
|---|---:|---:|---:|---:|---|
| gốc | 12,40 | 30,4 | 0,10953 | 13,7 | `ae PLL1 PLL 34` |
| **$Z_{feeder}$ ×100** | 14,56 | 25,9 | **0,10954** | 13,7 | `ae PLL1 PLL 34` |
| $x_{tr}$ 0,06 → 2,00 (×33) | 12,60 | 29,9 | 0,10953 | 13,7 | `ae PLL1 PLL 34` |
| $x_f$ 0,15 → 1,00 (ngoài đặc tả) | 33,85 | 11,1 | **−0,05566** | 7,1 | `uqLag_y REGF1 G1` |
| ↑ như trên, **nhánh tĩnh (đối chứng)** | — | — | **−0,05071** | 7,1 | `uqLag_y REGF1 G1` |

Nhân trở kháng feeder lên **100 lần** dịch $\zeta_{\min}$ được $10^{-5}$. Nhân $x_{tr}$ lên 33 lần:
không gì. Chỉ $x_f$ động đậy — **và dòng cuối cho thấy đó không phải cơ chế Yang**: nhánh tĩnh mất
ổn định ở đúng mode đó, đúng tần số đó, với tham gia chi phối bởi vòng áp và góc của GFM G1. PLL
chỉ thêm 0,5 pp. Đó là mode vòng áp grid-forming, một chuyện khác, và $x_f$ = 1,00 vốn đã **4×
ngoài dải REGFM_A1** (0,05–0,25).

> **Feeder chưa bao giờ là trở kháng chi phối.** Độ cứng lưới mà đội GFL nhìn thấy do điện kháng
> ghép của chính converter đặt ra, không do mạng. Nên $\lambda_{\min}$ — một đại lượng thuần
> **cấu trúc mạng** — không thể mang thông tin ở đây.

## Kết luận

$\lambda_{\min}$ là chỉ số vay mượn thứ năm không có miền hiệu lực trên hệ này. Khác biệt: **lần
này cơ chế đã được dựng vào mô hình và vẫn đo được H0**, kèm lý do định lượng (30,4×) thay vì một
tương quan trống.

Kết luận vị trí GFM nay đứng trên **bốn phép đo độc lập**, không qua chỉ số trung gian nào:

| tầng | phép đo | kết quả |
|---|---|---|
| biên an ninh (T41) | 4 topology × 6 vị trí | tản **0,000%**, `margin_gap` 0,476 |
| chia công suất quá độ (T44) | G2/G3/G5 cùng định mức | **0,300 pp**; định mức 40× hơn |
| trị riêng, GFL tĩnh (T49) | 8 cách đặt | corr **+0,14**, mode cố định 38,6 Hz |
| **trị riêng, GFL có PLL (T50)** | 16 cách đặt, $\lambda_{\min}$ 13,2× | corr **−0,10**, tản **0,000** |

## Cái này **không** trả lời

- **Không kiểm được ở cỡ đội hay feeder khác.** Cơ chế 30,4× phụ thuộc $x_f$ và định mức
  converter; một hệ với $x_f$ hiệu dụng lớn hơn nhiều có thể cho $\lambda_{\min}$ nội dung thật.
  T50 nói về **hệ này**, không bác Yang.
- **16 PLL cùng tham số.** Đội không đồng nhất (Kp khác nhau theo thiết bị) chưa kiểm; nó phá suy
  biến và có thể mở mode liên thiết bị. Chưa đo.
- **Ipcmd/Iqcmd hằng** (không REECA1, không REPCA1). Đây là mô hình rút gọn chuẩn cho mất ổn định
  PLL lưới yếu, nhưng vòng điều khiển công suất ngoài chưa có.
- **Không chạy miền thời gian.** T50 hoàn toàn là tuyến tính hoá quanh một điểm vận hành.
- $x_f$ = 1,00 làm mất ổn định vòng áp GFM. Nằm ngoài đặc tả nên không phải phát hiện về cấu hình
  đang ship, nhưng nó **chưa được quét** trong cổng robustness T33 ở nhánh động.

---

`experiments/t50_pll_placement_validation.py` · `results.json` · 16 placement `pl_NN_*.json`
Thay đổi mã: `src/phasor/build_case.py` — `gfl_dynamic`, `pll_kp/ki/tf/tp`, `gfl_x_conv_pu`.
Mặc định **off**; T1–T49 tái lập không đổi.
