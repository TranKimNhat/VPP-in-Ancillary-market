# T02-v4 — chạy lại T2 trên đội GFM đã rescale

Cùng mã, cùng mô hình derating như `artifacts/T02_soc/`; **chỉ đổi hồ sơ bố trí**:
`official_placement_v3.json` → `official_placement_v4_rescaled.json` (đội GFM ×0,18967,
tổng inverter 23 → 4,362 MVA = 1,25 × tải đỉnh 3,49 MW).

```
uv run python experiments/t02_soc.py \
    --placement artifacts/placement/official_placement_v4_rescaled.json \
    --out artifacts/T02_soc_v4
```

## Kết luận về SoC **không đổi**

`Δf_ss = f₀·R·ΔP / ΣS_g` vẫn không chứa `P_head`; cột `df_ss_hz_unconstrained` vẫn bất biến
tuyệt đối theo SoC. **SoC ∈ Ω_E** giữ nguyên. Rescale không mở ra kênh nhanh nào.

Cái *đổi* là mọi thứ khác.

## V1 — biên đã vào tầm với

| | v3 | v4 |
|---|---:|---:|
| `ΔP_critical` (SoC ≥ 0,30) | 17,25 MW | **3,272 MW** |
| Tải đỉnh feeder | 3,49 MW | 3,49 MW |
| `κ` tại `ΔP` = 3 MW | 6,00 | **1,138** |
| `μ_P` (đơn vị siết) tại `ΔP` = 3 MW | 0,174 | **0,917** |

`κ` cắt qua 1 tại `ΔP = 3,414 MW` — nằm ngay trong dải nhiễu khả tín của feeder này.
Ở `ΔP` = 5 và 8 MW **không tồn tại SoC an toàn nào**: hai mức này biến mất khỏi
`soc_thresholds.csv` vì mọi SoC đều bão hoà. Đây chính là điều v3 không làm được.

## V2 — nhưng ràng buộc *tần số* siết trước ràng buộc *headroom* 4,5 lần

Với `R = 5%` trên `ΣS_g = 4,362 MVA`, độ lợi sơ cấp của cả đội là

```
ΔP/Δf = ΣS_g /(f₀·R) = 1,454 MW/Hz
```

nên dải cho phép ±0,5 Hz chỉ nuốt được **0,727 MW**, trong khi headroom cho phép tới 3,272 MW.

| Ngưỡng | `ΔP` tối đa |
|---|---:|
| `Δf_ss` ≤ 0,2 Hz | 0,291 MW |
| `Δf_ss` ≤ 0,5 Hz | 0,727 MW |
| `Δf_ss` ≤ 1,0 Hz | 1,454 MW |
| `μ_P` ≤ 1 (headroom) | 3,272 MW |

**Hệ quả:** trong tầng giải tích, toạ độ chi phối ở quy mô v4 là **tần số**, không phải
headroom công suất — trừ khi điều khiển thứ cấp được tính vào, hoặc `R` được hạ. Điều này
biến câu hỏi "ràng buộc nào chi phối" từ suy đoán thành một con số cần EMT xác nhận.
Nó cũng đặt ra một quyết định chưa chốt: **`R = 5%` có còn đúng cho đội thiết bị đã rescale
không** — ở v3 nó cho `Δf_ss` = 0,13 Hz với `ΔP` = 1 MW, ở v4 cho **0,688 Hz**.

## V3 — ngưỡng SoC an toàn dịch lên rõ rệt

SoC thấp nhất còn an toàn (`soc_thresholds.csv`); ô trống = **không SoC nào an toàn**:

| ΔP [MW] | 5 phút | 15 phút | 30 phút | (v3, 30 phút) |
|---:|---:|---:|---:|---:|
| 0,5 | 0,125 | 0,125 | 0,150 | 0,125 |
| 1,0 | 0,150 | 0,150 | 0,200 | 0,125 |
| 2,0 | 0,175 | 0,200 | 0,275 | 0,125 |
| 3,0 | 0,200 | 0,225 | **0,350** | 0,150 |
| 5,0 | — | — | — | 0,175 |
| 8,0 | — | — | — | 0,225 |

## V4 — `Ā_GFM` nhạy hơn với SoC lệch

Hạ SoC một đơn vị trong khi các đơn vị khác ở 0,80: biên độ **0,45% (G3) → 12,07% (G1)**,
so với 1,3% → 5,1% ở v3. Lý do là `ε_g ∝ 1/S_g` nên sau rescale các đơn vị nhỏ bị đẩy ra
rất xa về mặt điện; xem `artifacts/T01_agfm_v4/README.md`. Đây là *tác dụng phụ của quy ước
ε*, không phải một hiệu ứng vật lý mới của SoC.

## Tệp

Giống `artifacts/T02_soc/`. `manifest.json` nay ghi cả đường dẫn hồ sơ bố trí và
`P_rated / E_rated / S_rated` từng đơn vị, để hai lần chạy phân biệt được (N-2).
