# T25 (C2) — biên tắt diesel cuối: **kết quả dương đầu tiên của loạt này**

**Câu hỏi:** ngắt máy diesel cuối là thay đổi *cấu trúc* (mất nguồn áp đồng bộ, mất quán tính,
đổi mạng đồng bộ hoá), không phải thay đổi tham số. Cơ chế Λ nói *feeder* vô hình — nó không
nói việc mất một máy đồng bộ là vô hình. Vậy biên có dịch không?

Diesel 1,5 MVA tại bus 76, GENROU + TGOV1 (R = 0,05) + SEXS, H = 1,0 s. Bisection trên
`P_DG(t_off⁻)`. So sánh với `ΔP_max = 1,18506 MW` của T21 (gen_loss cùng cỡ, cùng dải, cùng
điểm điều độ: đội GFM nạp ~0,57 MW trước sự cố ở cả hai ca).

## Kết luận một dòng

$$P_{DG,\text{off}}^{\max} = \mathbf{1{,}2086\ MW} \quad\text{so với}\quad \Delta P_{\max}^{\text{gen\_loss}} = 1{,}1851\ \text{MW}$$

**Mất diesel cuối *dễ hơn* mất một nguồn phi đồng bộ cùng cỡ 2,0%** — ngược dấu với dự đoán.
Và quan trọng hơn con số: **hai họ toạ độ dịch ngược chiều nhau**, lần đầu tiên trong loạt này.

## ① Phân rã tại cùng công suất (1,181 MW so với 1,179 MW)

| toạ độ | diesel trip | gen_loss | chênh |
|---|---:|---:|---|
| `f_nadir` [Hz] | 59,0256 | 59,0044 | **tốt hơn 2,1%** (độ lệch nhỏ hơn) |
| RoCoF [Hz/s] | 1,9556 | 1,9989 | **tốt hơn 2,2%** |
| `V_min` [pu] | 0,9066 | 0,9196 | **xấu hơn 0,0130 pu** |
| `μ_I` | 0,7991 | 0,6759 | **xấu hơn 18,2%** |

Toạ độ **khối** (tần số) tốt lên; toạ độ **cục bộ** (áp, dòng) xấu đi — đúng phân đôi mà cơ
chế Λ dự đoán, và là phép đo đầu tiên trong loạt cho thấy hai họ **không** đi cùng chiều. Diễn
giải vận hành: mất nguồn áp đồng bộ chuyển gánh nặng từ *tần số* sang *điện áp và dòng*, vì
đội GFM phải gánh thêm cả công suất phản kháng mà diesel đang cấp.

Khác biệt **được phân giải, không phải nhiễu dung sai**: hai khoảng kẹp **không giao nhau** —
gen_loss mất an ninh tại 1,1908 trong khi diesel còn an ninh tại 1,2031 (khe hở 0,0123 MW).

## ② Hằng số vọt lố **kém chặt hơn** cho sự cố đồng bộ

| ca | $\kappa_{os}$ | σ | n |
|---|---:|---:|---:|
| gen_loss | 1,2275 | **0,24%** | 10 |
| diesel trip | 1,1888 | **1,7%** | 12 |

Nhỏ hơn 3,2% và **tản gấp 8 lần**. Đây là giới hạn phạm vi thật cho C4: dạng đóng
$\Delta f_{\text{nadir}} = \kappa_{os} f_0 R \Delta P/\sum S_g$ **chặt bốn chữ số cho sự cố
nguồn phi đồng bộ, nhưng chỉ chặt ~2% cho sự cố máy đồng bộ**. Phải phát biểu kèm, không được
gộp hai ca vào một hằng số.

Cơ chế của chênh lệch 3,2% **chưa xác lập** bằng các lần chạy này. Không suy diễn trong bài
cho tới khi có bằng chứng.

## ③ Quán tính của diesel cuối **không đóng góp gì** vào việc sống sót qua chính nó

Chạy lại toàn bộ với `H = 0,1 s` thay vì `1,0 s` (`M = 0,3` so với `3,0`, đã xác minh trong
mô hình dựng): **cả 13 điểm dò giống hệt nhau đến bốn chữ số**, biên y hệt `1,2086 MW`
(`artifacts/T25_pdgoff_h0p1/`).

Lý do hiển nhiên khi đã thấy: máy bị ngắt tại `t_event`, nên rotor của nó rời hệ cùng lúc;
trước đó hệ ở trạng thái dừng nên quán tính không có gì để tác động. **Phép chạy này không
phân tách được "mất nguồn áp" khỏi "mất quán tính" — thiết kế thí nghiệm sai, ghi lại để không
lặp.** Muốn đo đóng góp của quán tính thì phải **giữ diesel online** và đánh một `gen_loss`
chỗ khác, rồi quét `H`.

Điều nó *có* xác lập, sạch: biên tắt diesel **không phụ thuộc lựa chọn `H`**, nên phản biện
"kết quả của bạn phụ thuộc một hằng số quán tính tuỳ chọn" bị loại bằng phép đo.

## ④ Toạ độ siết không đổi

Biên vẫn do **RoCoF + nadir** quyết định (tại 1,2141: `f_nadir 58,9977 < 59,0; rocof 2,0117 >
2,0`). `μ_I` = 0,809 và `μ_P` = 0,364 vẫn xa ngưỡng. Điểm góc tần số vẫn đứng.

## Sinh lại

```
uv run python experiments/t20_andes_bisect.py --what p_dg_off --load-p2z 0.0 --q-max 0.60 \
    --diesel-bus 76 --diesel-mva 1.5 --diesel-h 1.0 --dg-lo 0.0 --dg-hi 1.4 \
    --out artifacts/T25_pdgoff_h1p0
```

Cảnh báo phạm vi từ `DieselSpec`: TGOV1 là bản thế thân miền phasor, không phải GGOV1/DEGOV1
của D2b. Chênh lệch 2,0% và độ tản $\kappa_{os}$ 1,7% đều nằm trong cỡ mà một họ governor khác
có thể dịch — phải đóng bằng đối chiếu EMT ở T7/T9, không được giả định hai họ trùng nhau.
