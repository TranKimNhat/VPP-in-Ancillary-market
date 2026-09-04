# T27 — (c) họ governor: rủi ro TGOV1 bị **loại**, không phải thu hẹp

`DieselSpec` cảnh báo TGOV1 là bản thế thân miền phasor, không phải GGOV1/DEGOV1 của D2b. Câu
hỏi: chênh lệch 3,2% của $\kappa_{os}$ ca đồng bộ (1,1888 so với 1,2275) có tái lập được từ
trễ governor không?

## Kết luận

**Không — và không thể.** Chạy lại toàn bộ biên tắt diesel với `GAST` (họ governor có nhánh
load-limiter gần máy sơ cấp thật hơn) thay vì `TGOV1`:

| đại lượng | max \|TGOV1 − GAST\| qua 13 điểm dò khớp |
|---|---:|
| `f_nadir` | 1,9 × 10⁻¹⁰ |
| RoCoF | 6,2 × 10⁻⁹ |
| `V_min` | 7,5 × 10⁻¹¹ |
| `μ_I` | 1,1 × 10⁻⁹ |

Biên y hệt: `P_DG,off^max = 1,20859 MW` ở cả hai. **Trùng khớp ở mức sai số máy.**

Lý do cấu trúc, cùng lý do đã giết phép quét `H` ở T25b: **governor rời hệ cùng với máy**. Nó
chỉ đặt trạng thái dừng trước sự cố, mà trạng thái đó là tĩnh; toàn bộ quá độ đo được xảy ra
sau khi máy đã bị ngắt.

## Hai hệ quả

**① Rủi ro (c) bị loại cho biên tắt diesel** — không phải thu hẹp xuống "chờ EMT T7/T9". Lựa
chọn họ governor **không ảnh hưởng gì** tới `P_DG,off^max`, và điều đó được đo chứ không được
lập luận. Cảnh báo TGOV1 **quay lại đầy đủ** cho bất kỳ kịch bản nào giữ diesel *online* qua
sự cố — ở đó governor có tác dụng và chưa được kiểm chéo.

**② Giả thuyết "3,2% do trễ governor" bị bác.** Hai ứng viên giải thích chênh lệch
$\kappa_{os}$ giữa ca đồng bộ và phi đồng bộ nay đều bị loại **bằng phép đo**: quán tính
(T25b) và governor (T27). Còn lại ứng viên điện: cắt máy đồng bộ lấy đi cả **nguồn áp và công
suất phản kháng** của nó, và đó là chỗ `V_min` xấu đi 0,013 pu và `μ_I` xấu đi 18% trong T25.
Chưa xác lập, nhưng không gian giả thuyết đã hẹp lại từ ba xuống một.

## Sinh lại

```
uv run python experiments/t20_andes_bisect.py --what p_dg_off --load-p2z 0.0 --q-max 0.60 \
    --diesel-bus 76 --diesel-mva 1.5 --diesel-h 1.0 --governor GAST \
    --dg-lo 0.0 --dg-hi 1.4 --out artifacts/T27_pdgoff_gast
```
