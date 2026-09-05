# T44 — lever $\Lambda$ hoạt động, nhưng $\Lambda$ **không phải biến điều khiển**

**Giả thuyết được kiểm.** T43 cho $\Lambda > 1$ ở mọi cách đặt tại cỡ đội hiện tại, và đề xuất
lever rẻ nhất: $x_f$ 0,15 → 0,05 (đáy dải $X_L$ REGFM_A1, và là giá trị dự án dùng trước khi
đổi) cộng đặt lại ở ly cách tối đa (T43), cho $\Lambda$ = 0,96–3,37 mà **không** đổi cỡ đội,
feeder, hay tuân thủ đặc tả.

Ba bước, mỗi bước gác bước sau. Bước 3 là phép thử có đối chứng: **G2, G3, G5 cùng định mức
0,7587 MVA**, chỉ khác vị trí điện — nên tản tỉ lệ chia công suất quá độ của ba máy này **là**
phép đo mức độ vị trí có ảnh hưởng.

---

## Kết quả

| cấu hình | $X_{feeder}$ tb | $\Lambda$ | ổn định | **tản do vị trí** | tản do định mức | tỉ số |
|---|---:|---|---|---:|---:|---:|
| shipped ($x_f$ 0,15, bus hiện tại) | 0,04183 | 3,78 – 13,24 | ✅ | 0,300 pp | 12,06 pp | 40,2× |
| $x_f$ = 0,05 | 0,04183 | 1,98 – 6,93 | ✅ | **0,143 pp** | 15,55 pp | 108,7× |
| max-separation | 0,08614 | 1,84 – 6,43 | ✅ | **0,461 pp** | 11,92 pp | 25,9× |
| **$x_f$ 0,05 + max-sep** | 0,08614 | **0,96 – 3,37** | ✅ | **0,174 pp** | 15,55 pp | 89,1× |

Bước 1 **đạt**: $\Lambda$ xuống đúng 0,96–3,37, tức $\le 1$ cho máy lớn nhất.
Bước 2 **đạt**: cả bốn cấu hình ổn định ở bộ điều khiển ship ($\max\mathrm{Re}\lambda$ −1,81
đến −2,02). $x_f$ = 0,05 hạ $\zeta_{\min}$ 0,156 → 0,029, còn ổn định nhưng damping kém 5×.
Bước 3 **trượt, và ngược chiều**.

## $\Lambda$ không điều khiển hiệu ứng vị trí

| lever | $\Lambda$ | tản do vị trí |
|---|---|---|
| max-separation | ↓ 2,06× | **↑** 0,300 → 0,461 pp ✓ đúng chiều |
| $x_f$ 0,15 → 0,05 | ↓ 1,91× | **↓** 0,300 → 0,143 pp ✗ **ngược chiều** |
| ghép cả hai | ↓ 3,93× → $\Lambda$ = 0,96 | 0,174 pp — **tệ hơn cấu hình đang ship** |

Hai lever dịch $\Lambda$ cùng chiều và dịch kết quả **ngược chiều nhau**. Đạt $\Lambda \le 1$
không làm phân bổ trở nên nhìn thấy được.

$$\textbf{Khung thiết kế } \Lambda=(x_f+x_{tr})\,n/(\alpha\zeta) \textbf{ dự đoán sai.}$$

Cơ chế chưa xác lập. Quan sát: hạ $x_f$ làm tản **do định mức** tăng 12,06 → 15,55 pp, vì
$X_{conv,i} = (x_f+x_{tr})/S_i$ co lại **nhiều hơn ở máy nhỏ**, khiến phần chia do định mức lấn
át phần do vị trí. Nhưng đây là quan sát, không phải cơ chế đã kiểm — **không dùng làm lập luận
cho tới khi có bằng chứng**.

## Kết luận

Không cấu hình nào trong bốn làm vị trí vượt **0,5 pp**. Tốt nhất là max-separation một mình
(0,461 pp), vẫn kém định mức **26 lần** và nhỏ tới mức không có ý nghĩa vận hành.

Cộng với T35/T41 (biên tổng hợp: tản 0,000% qua 4 topology và 6 vị trí, `margin_gap` 0,476), kết
luận nay đứng ở **ba tầng độc lập**:

| tầng | phép đo | kết quả |
|---|---|---|
| biên an ninh tổng hợp | 4 topology × 6 vị trí | tản **0,000%**, phân giải thật |
| chia công suất quá độ, cấu hình ship | G2/G3/G5 cùng định mức | **0,300 pp**, kém định mức 40× |
| chia công suất quá độ, cấu hình thuận lợi nhất đạt được | như trên, $\Lambda \le 1$ | **0,461 pp**, kém định mức 26× |

> **Lưới này, ở cỡ này, về mặt điện là một nút đơn.** Không phải vì đặt sai vị trí, không phải
> vì chọn sai $x_f$, và không sửa được bằng bất kỳ lever nào giữ nguyên feeder và cỡ đội.

Đây là **kết quả**, không phải thất bại — nhưng nó đóng lối cuối cùng để cấu trúc mạng ảnh
hưởng an ninh trên hệ này.

## Cái này không trả lời

- Cỡ đội 15–20 MVA của T43 §2 **chưa kiểm** ở phép thử vị trí này. T44 chỉ kiểm các lever giữ
  nguyên cỡ đội. Cửa sổ sizing của T43 vẫn mở về mặt số học.
- Một feeder khác với $\zeta$ cao hơn hẳn chưa kiểm. Nhưng vì $\Lambda$ vừa được chứng minh là
  **không phải biến điều khiển**, $\zeta$ cũng không còn là tiêu chí chọn feeder đáng tin —
  cần một tiêu chí khác trước khi đi tìm feeder khác.
- $x_f$ = 0,05 hạ $\zeta_{\min}$ xuống 0,029. Nếu vì lý do khác mà chọn giá trị đó thì phải
  chạy lại cổng robustness T33; T44 chỉ kiểm một điểm vận hành danh định.

---

`experiments/t44_lambda_lever_test.py` · `results.json` · placement đã dời:
`placement_maxsep.json` (bus G1 114, G2 85, G3 151, G4 33, G5 96, G6 66; định mức, E/P, $\pi_g$
không đổi)
