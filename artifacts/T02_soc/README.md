# T02 — SoC thuộc Ω_E hay có ảnh hưởng nhanh?

> **Chạy trên hồ sơ bố trí v3 (chưa rescale).** Kết luận về SoC vẫn đúng; các con số về
> biên (`ΔP_critical`, `κ`, ngưỡng SoC) đã bị thay bởi `artifacts/T02_soc_v4/`.

**Câu hỏi:** SoC có kênh tác động nhanh lên an ninh tần số không, hay chỉ là biến chậm
thuộc tập Ω_E?

**Kết luận một dòng (đưa vào concept §25):**
**SoC ∈ Ω_E.** Nó không có kênh nào chạm vào động học tần số không-ràng-buộc; nó chỉ dịch
*biên bão hoà* qua `P_head`, và với đội thiết bị hiện tại biên đó chỉ bắt đầu dịch khi
**SoC < 0,225** — xa mọi điểm vận hành thực tế.

## Sinh lại

```
uv run python experiments/t02_soc.py
```

Cài đặt: `src/analytical/headroom.py`. Cố định: topology `G0`, bố trí 6 GFM, `R = 5%`.

## Mô hình derating (tham số, cần nhóm nghiên cứu thẩm định)

```
P_head(SoC) = min(  taper(SoC) · P_rated ,  (SoC − SoC_min) · E_rated / t_hold  )
taper: 0 tại SoC_min = 0,10 → 1 tại SoC_taper = 0,20
t_hold ∈ {5, 15, 30} phút
```

## Bốn quan sát

**O1 — Kênh nhanh không tồn tại.** Ở chế độ droop không bão hoà,
`Δf_ss = f₀·R·ΔP / ΣS_g` — công thức **không chứa `P_head`**. Cột
`df_ss_hz_unconstrained` trong `soc_kappa.csv` bất biến hoàn toàn theo SoC ở mọi `ΔP`.
SoC chỉ vào được qua bão hoà.

**O2 — Với SoC đồng nhất, `Ā_GFM` bất biến *chính xác*.** Cả 6 GFM có cùng tỉ số
E/P = 2 h, nên derating nhân mọi `P_head,g` với cùng hệ số ⇒ `π_g` không đổi ⇒ `Ā` không đổi
(15,832117 ở mọi SoC ≥ 0,2). Đây là tính chất của đội thiết bị này, không phải của định nghĩa.

**O3 — Với SoC lệch nhau, `Ā_GFM` dịch tối đa ±5,1%.** Hạ SoC của **một** đơn vị xuống
trong khi các đơn vị khác ở 0,80 (`soc_spread.csv`): biên độ 1,3% (G3) → 5,1% (G1). Nhỏ,
và chỉ xuất hiện dưới SoC ≈ 0,225.

**O4 — Biên công suất gần như không bao giờ ràng buộc.** Ở SoC đủ, `ΔP_critical = 17,25 MW`
so với **tổng tải feeder chỉ 3,49 MW**. Nghĩa là `μ_P ≤ 0,2` cho mọi nhiễu khả tín, kể cả
mất toàn bộ một trang trại gió 3 MW (`μ_P = 0,17`).

> **~~Hệ quả cho §15/R2~~ — ĐÃ BỊ THAY THẾ ngày 2026-09-03.** Kết luận cũ ("`μ̂_P` không bao
> giờ chạm 1, nên phương án lùi ở R2 là rỗng") coi định mức đội GFM là hằng số. Nó không phải:
> `P_head` là **biến quyết định** của campaign. Sau khi rescale về 1,25 lần tải đỉnh,
> `ΔP_critical` = 3,272 MW và `μ_P` = 0,92 tại `ΔP` = 3 MW — biên vào tầm với, `μ̂_P` dùng làm
> cổng sàng lọc được. Xem `artifacts/T02_soc_v4/` và R2 bản sửa lần 2 trong plan §6.
> Toàn bộ phần còn lại của tệp này vẫn đúng **cho hồ sơ bố trí v3**.

## Ngưỡng SoC an toàn (`soc_thresholds.csv`)

SoC thấp nhất mà không đơn vị nào bão hoà, theo `t_hold` và `ΔP`:

| ΔP [MW] | 5 phút | 15 phút | 30 phút |
|---:|---:|---:|---:|
| 0,5–3,0 | 0,125 | 0,125 | 0,125–0,150 |
| 5,0 | 0,150 | 0,150 | 0,175 |
| 8,0 | 0,150 | 0,175 | 0,225 |

## Cần xác nhận bằng EMT ở P1

Kết luận trên đúng **trong tầng giải tích**. Điều EMT phải kiểm là giả định ngầm rằng
derating theo SoC chỉ đổi `P_max` chứ không đổi *độ lợi droop* hay động học vòng trong.
Nếu bộ điều khiển BESS thật giảm băng thông ở SoC thấp thì có kênh nhanh và kết luận này
phải mở lại.

## Tệp

| Tệp | Nội dung |
|---|---|
| `soc_sweep.csv` | `P_head`, `ΔP_critical`, `Ā` theo SoC × `t_hold` |
| `soc_kappa.csv` | `μ_P`, `κ`, `Δf_ss` theo SoC × `t_hold` × `ΔP` |
| `soc_thresholds.csv` | SoC thấp nhất còn an toàn |
| `soc_spread.csv` | ca SoC không đồng nhất (một đơn vị lệch) |
