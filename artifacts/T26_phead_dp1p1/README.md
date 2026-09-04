# T26 — `κ` có giá trị, và bao hiệu lực của dạng đóng có điểm chạm biên

Gỡ chỗ chặn (b), rồi (b) mở khoá (a) trong cùng một lần chạy.

## 0. Bản vá chặn: trần P âm là **không biểu diễn được**, không phải sai công thức

`REGF1.Pmax` khai `non_negative=True`; ANDES thay mọi giá trị **< 0** bằng mặc định 1,0 —
im lặng, theo từng máy, tại khởi tạo TDS. Với `gen_loss` ΔP > tải ròng 0,61 MW, đội GFM nạp
trước sự cố (`p₀ = 0,61 − ΔP < 0`) nên `Pmax = (p₀ + head)/S_n` **âm thật**. Đó là một ràng
buộc vận hành hợp lệ (pin buộc phải nạp), chỉ là REGF1 không diễn đạt được.

Không có "sửa công thức" nào cứu được — mô hình thật sự không biểu diễn nổi. Bản vá đúng là
**phát hiện và ra phán quyết giải tích**: `build_system` tính `pmax_dev`, đặt cờ
`ceiling_representable`; `solve` trả về ngay không mô phỏng; `metrics.extract` ghi
`no_equilibrium = True`, `secure = False`. Trần dưới 0 nghĩa là đội thiết bị không rời được
góc phần tư nạp — **không có điểm cân bằng sau sự cố để mô phỏng tới**, và đó là kết quả chứ
không phải giới hạn nền tảng.

Chọn lối 3 (sửa) chứ không lối 1 (đổi sang `load_step`) vì **tính so sánh được**: `ΔP_max`,
T22, T23, T24, T25 đều đo trên `gen_loss`. Đổi nhiễu cho riêng `P_head^min` sẽ làm `κ` không
so được với `ΔP_max`.

Hiệu lực bản vá, ΔP = 1,1 MW: head ≤ 0,40 → trần âm, phán quyết giải tích trong **3,4 s**
thay vì 20 s mô phỏng sai. Trước bản vá, cùng những điểm đó báo **SECURE**.

## 1. `κ` — có giá trị, và nó bằng 1

| ΔP [MW] | `P_head^min` [MW] | **κ = ΔP / P_head^min** |
|---:|---:|---:|
| 0,6 | 0,5953 | **1,0078** |
| 1,1 | 1,0947 | **1,0049** |

$$\kappa = 1{,}006 \pm 0{,}002$$

**Biên headroom là biên khả thi và không gì hơn** — xác nhận B1, lần này với trần đúng thay vì
trần bị ANDES xoá. Không có dự trữ động học chồng lên. `κ` giờ là một số đọc được, và giá trị
của nó là: nó không phải một bậc tự do.

## 2. Bao hiệu lực **đã có điểm chạm biên** — vế thứ hai của Đóng góp I

Dự đoán dạng đóng tại ΔP = 1,1: $\Delta f = \kappa_{os} f_0 R \Delta P/\sum S_g = 0{,}9286$ Hz
→ nadir 59,0714.

| `P_head` [MW] | máy bão hoà | `μ_P` | nadir đo | sai số dạng đóng |
|---:|---:|---:|---:|---:|
| 3,414 → 1,3115 (5 điểm) | 0 | 0,300 → 1,067 | 59,0720 | **−0,06%** |
| 1,1013 | 0 | 1,434 | 59,0581 | +1,44% |
| 1,0881 | **6** | 1,466 | 59,0493 | +2,39% |
| 1,0750 | 6 | 1,499 | 59,0387 | +3,53% |
| 1,0487 | 6 | 1,569 | 59,0133 | +6,26% |
| 0,9961 | 6 | 1,732 | 58,8845 | **+20,1%** |
| 0,8910 | 6 | 2,187 | 58,5229 | **+59,1%** |
| 0,7228 | 6 | 3,766 | 57,9443 | **+121,4%** |

Đây là **đáp ứng-liều đơn điệu**, không phải một điểm vỡ. Dạng đóng chính xác −0,06% trên
năm điểm không bão hoà trải 2,6 lần theo `P_head`, rồi suy giảm đều khi `μ_P` vượt 1.

**Tinh chỉnh quan trọng cho tiêu chí áp dụng:** ngưỡng không phải "đỉnh `μ_P` < 1". Tại
`P_head` = 1,3115 đỉnh `μ_P` = 1,067 > 1 mà sai số vẫn −0,06%; và tại 1,1013 đỉnh `μ_P` = 1,434
mà **không máy nào bão hoà lúc ổn định** và sai số chỉ 1,44%. Điều kiện đúng là **không máy
nào *duy trì* trần lúc ổn định** — vượt trần thoáng qua rồi được bộ hạn chế `KPplim` kéo về
là vô hại. Phát biểu bao hiệu lực phải dùng bão hoà ổn định, không dùng đỉnh.

## Sinh lại

```
uv run python experiments/t20_andes_bisect.py --what p_head --event gen_loss --dp 1.1 \
    --load-p2z 0.0 --q-max 0.60 --out artifacts/T26_phead_dp1p1
uv run python experiments/t20_andes_bisect.py --what p_head --event gen_loss --dp 0.6 \
    --load-p2z 0.0 --q-max 0.60 --out artifacts/T26_phead_dp0p6
```
