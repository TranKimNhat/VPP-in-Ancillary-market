# reporttopolog

## 1) Topology tổng quan IEEE 123 hiện tại
- Nguồn chuẩn: `data/grid_IEEE123_complete.m` (MATPOWER bus numbering), `artifacts/placement/official_placement_v3.json`, `artifacts/l0l1_stats/bus_zone_map.csv`, `src/opt/tie_switch_reconfig.py`.
- **Quy ước bus trong toàn bộ tài liệu này: dùng trực tiếp MATPOWER bus ID từ file `.m`** (không dùng index nội bộ của pandapower DataFrame).
- Số zone vận hành: 4 (Z1, Z2, Z3, Z4).
- Tie-switch điều khiển trong reconfiguration: line IDs `108, 110, 112, 114, 116` (đánh theo thứ tự branch trong `.m`).
- Số agent MARL: 41 (9 EVCS-PV + 9 EVCS-BESS + 9 EVCS-V2G + 14 DPV).

## 2) Tie-switch điều khiển (reconfiguration)
| Line ID | From bus | To bus | Vai trò |
|---:|---:|---:|---|
| 108 | 113 | 61 | Tie/bridge switch điều khiển topology |
| 110 | 135 | 35 | Tie/bridge switch điều khiển topology |
| 112 | 149 | 1 | Tie/bridge switch điều khiển topology |
| 114 | 152 | 52 | Tie/bridge switch điều khiển topology |
| 116 | 160 | 67 | Tie/bridge switch điều khiển topology |

> Trong `TieSwitchReconfiguration`, tập này là các tie-line thao tác mở/đóng để sinh topology khả thi.

## 3) Cấu trúc Zone
- **Zone 1**: 21 buses
  - Buses: `[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 34, 149, 150, 152]`
- **Zone 2**: 37 buses
  - Buses: `[18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 135, 151, 250, 251]`
- **Zone 3**: 23 buses
  - Buses: `[97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 197, 300, 350, 450, 451]`
- **Zone 4**: 48 buses
  - Buses: `[52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 128, 160, 610]`

## 4) Cấu trúc VPP theo Zone
- **VPP_1 ↔ Zone 1**
- **VPP_2 ↔ Zone 2**
- **VPP_3 ↔ Zone 4**
- **Zone 3**: wind corridor / balancing support (không phải VPP bidding zone chính).

## 5) DER placement theo bus / VPP / Zone

### 5.1 EVCS (mỗi EVCS có PV + BESS + V2G)
| EVCS | Bus | Zone | VPP | PV (MW) | BESS (MW/MWh) | V2G (MW) |
|---|---:|---:|---|---:|---:|---:|
| E1 | 34 | 1 | VPP_1 | 0.200 | 0.325/0.650 | 0.100 |
| E2 | 1 | 1 | VPP_1 | 0.050 | 0.325/0.650 | 0.100 |
| E3 | 2 | 1 | VPP_1 | 0.050 | 0.325/0.650 | 0.100 |
| E4 | 49 | 2 | VPP_2 | 0.200 | 0.275/0.550 | 0.100 |
| E5 | 50 | 2 | VPP_2 | 0.200 | 0.275/0.550 | 0.100 |
| E6 | 48 | 2 | VPP_2 | 0.200 | 0.275/0.550 | 0.100 |
| E7 | 52 | 4 | VPP_3 | 0.200 | 0.225/0.450 | 0.075 |
| E8 | 53 | 4 | VPP_3 | 0.158 | 0.225/0.450 | 0.075 |
| E9 | 66 | 4 | VPP_3 | 0.200 | 0.225/0.450 | 0.075 |

### 5.2 DPV
| DPV | Bus | Zone | VPP | MW |
|---|---:|---:|---|---:|
| PV1 | 6 | 1 | VPP_1 | 0.275 |
| PV2 | 17 | 1 | VPP_1 | 0.275 |
| PV3 | 3 | 1 | VPP_1 | 0.275 |
| PV4 | 4 | 1 | VPP_1 | 0.275 |
| PV5 | 43 | 2 | VPP_2 | 0.275 |
| PV6 | 27 | 2 | VPP_2 | 0.275 |
| PV7 | 47 | 2 | VPP_2 | 0.275 |
| PV8 | 30 | 2 | VPP_2 | 0.275 |
| PV9 | 18 | 2 | VPP_2 | 0.275 |
| PV10 | 45 | 2 | VPP_2 | 0.275 |
| PV11 | 54 | 4 | VPP_3 | 0.275 |
| PV12 | 96 | 4 | VPP_3 | 0.275 |
| PV13 | 55 | 4 | VPP_3 | 0.275 |
| PV14 | 56 | 4 | VPP_3 | 0.275 |

### 5.3 Wind (corridor)
| Wind | Bus | MW |
|---|---:|---:|
| W1 | 67 | 3.000 |
| W2 | 105 | 3.000 |
| W3 | 98 | 3.000 |
| W4 | 101 | 3.000 |

### 5.4 GFM (ổn định / reconfiguration)
| GFM | Bus | Cấu hình | Mode |
|---|---:|---|---|
| G1 | 114 | BESS 6.0 MW/12.0 MWh | VSG |
| G2 | 60 | PV 2.0 MW, BESS 3.0 MW/6.0 MWh | Droop |
| G4 | 67 | BESS 2.0 MW/4.0 MWh | Droop |
| G5 | 36 | BESS 2.0 MW/4.0 MWh | Droop |
| G6 | 101 | BESS 2.0 MW/4.0 MWh | Droop |

## 6) Sơ đồ để vẽ nhanh
```mermaid
flowchart LR
  Z1[Zone 1 / VPP_1]
  Z2[Zone 2 / VPP_2]
  Z3[Zone 3 / Wind Corridor]
  Z4[Zone 4 / VPP_3]

  Z1 ---|Tie 112: 149-1| Z1
  Z1 ---|Tie 114: 152-52| Z4
  Z2 ---|Tie 110: 135-35| Z2
  Z3 ---|Tie 108: 113-61| Z4
  Z3 ---|Tie 116: 160-67| Z4

  E1[E1@34]
  E2[E2@1]
  E3[E3@2]
  Z1 --- E1
  Z1 --- E2
  Z1 --- E3

  E4[E4@49]
  E5[E5@50]
  E6[E6@48]
  Z2 --- E4
  Z2 --- E5
  Z2 --- E6

  E7[E7@52]
  E8[E8@53]
  E9[E9@66]
  Z4 --- E7
  Z4 --- E8
  Z4 --- E9

  W1[W1@67]
  W2[W2@105]
  W3[W3@98]
  W4[W4@101]
  Z3 --- W1
  Z3 --- W2
  Z3 --- W3
  Z3 --- W4
```

## 7) Tóm tắt vận hành
- VPP bidding/dispatch tập trung ở Z1/Z2/Z4.
- Z3 chủ yếu mang nguồn wind và hỗ trợ cân bằng lưới.
- Reconfiguration thao tác 5 tie-switch để giữ connectivity và chọn topology vận hành tốt.
- Placement DER hiện tại được đọc trực tiếp từ `official_placement_v3.json` trong train/eval env.
