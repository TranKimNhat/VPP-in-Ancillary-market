# Topology-Adaptive Voltage Control via Graph Attention MARL in Tri-Level DSO-VPP Coordination

**Target Journal:** IEEE Transactions on Smart Grid (TSG)

**Working Title:** *"Topology-Adaptive Multi-Agent Voltage Control Using Graph Attention Networks for Coordinated DSO-VPP Dispatch in Active Distribution Networks"*

---

## 1. Executive Summary

Dự án phát triển một khung điều khiển phân cấp 3 lớp (Tri-Level Hierarchical Framework) cho sự phối hợp giữa Nhà vận hành lưới phân phối (DSO) và Nhà máy điện ảo (VPP), trong đó **đóng góp chính (main contribution)** là thuật toán điều khiển điện áp thời gian thực tại Layer 2 sử dụng **Graph Attention Network - Multi-Agent Proximal Policy Optimization (GAT-MAPPO)**.

### Tuyên bố Novelty (Novelty Claims)

| # | Claim | Scope | Cơ sở |
|---|-------|-------|-------|
| **C1** | Kết hợp **network reconfiguration** như một layer chiến lược trong DSO-VPP coordination framework | Architectural | Không có bài báo nào (2022-2025) kết hợp reconfiguration switching với multi-level DSO-VPP market mechanism |
| **C2** | **GAT-MAPPO** cho distribution voltage control dưới dynamic topology | Algorithmic | GAT và MAPPO đã tồn tại riêng lẻ; combination cụ thể này cho voltage control chưa được published |
| **C3** | **Honest evaluation** của zero-shot topology generalization cho GNN-based controllers | Empirical | Literature hiện tại (2024-2025) cho thấy 3-15% performance degradation trên unseen topology; chưa có evaluation có hệ thống trong bối cảnh DSO-VPP |

### Positioning vs. State-of-the-Art

| Paper | Approach | Thiếu gì so với framework này |
|-------|----------|-------------------------------|
| Xue et al. (2024, *Applied Energy*) | Hierarchical Safe DRL cho DSO-VPP, đạt 1.46% of optimal | Không có reconfiguration, không dùng GNN |
| Lin et al. (2025, *Applied Energy*) | Stackelberg-cooperative game DSO-VPP với shared energy storage | Không có topology adaptation, không real-time control |
| Sun et al. (2024, *IET GTD*) | Multi-agent DRL + AC-OPF cho DSO-VPP | Không có graph-based architecture, topology cố định |
| Mi et al. (2024, *SSRN*) | Tri-level VPP-Prosumer robust game | Không có reconfiguration, không có AI-based real-time layer |

---

## 1.1 Current Implementation Status (Updated: 2026-02)

Các thay đổi kỹ thuật đã được triển khai trong codebase hiện tại:

- **Physics-first, fail-closed Layer 0 quality gate** đã bật mặc định:
  - `enforce_radiality=True`, `radiality_slack=0`
  - `ac_tolerance=0.01` (publish-grade default)
  - Pipeline fail-closed: nếu AC validation không đạt thì không xuất dataset hợp lệ cho Layer 1.
- **Bus-82 calibration fixes** đã được áp dụng ở Layer 0:
  - Siết transformer tap handling và SOC reference logic.
  - Thêm ràng buộc calibration theo `voltage_reference_upper_band` để neo điện áp SOCP quanh AC reference.
  - Kết quả sau calibration trong diagnostics: `fails=0/288`, `max_gap≈0.00482 p.u.`.
- **Bootstrap tri-layer path đã đồng bộ với cấu hình publish pass**:
  - Trong `experiments/train_mappo.py`, bootstrap hiện dùng
    - `ac_tolerance=0.01`
    - `voltage_reference_upper_band=0.005`
  - Tránh tình trạng bootstrap fail do threshold quá chặt không tương thích calibration.
- **Zone partition scoring** đã được bổ sung ở Layer 0 để chọn partition học thuật tốt hơn:
  - `ZoneScoringConfig`, `score_zone_partition(...)`, `select_best_zone_partition(...)`
  - Chấm theo connectivity, bus/load balance, DER penetration, boundary cuts, và penalties mất cân bằng.

---

## 2. Problem Statement & Research Gaps

### Gap chính (Primary Gap): Topology-Adaptive Real-Time Control trong DSO-VPP Coordination

Các framework DSO-VPP hiện tại (bi-level hoặc tri-level) đều **giả định cấu trúc lưới cố định**. Khi DSO thực hiện network reconfiguration (đóng/cắt switch), các agent điều khiển real-time được train trên topology cũ sẽ **mất hiệu lực** — hiện tượng "topology distribution shift". Đây là khoảng trống cụ thể:

> **Không có framework nào tích hợp network reconfiguration như biến quyết định chiến lược (Layer 0) với real-time voltage control topology-adaptive (Layer 2) thông qua cơ chế GNN nhận thức cấu trúc.**

### Gap phụ (Supporting Gaps)

**Gap A — Reconfiguration-embedded coordination:** Các bài reconfiguration (Jabr 2012, Qiao 2022) xử lý như bài toán single-agent DSO. Các bài DSO-VPP coordination (Xue 2024, Lin 2025) không xét reconfiguration. Chưa ai kết hợp cả hai trong một framework thống nhất.

**Gap B — Zonal pricing under reconfiguration:** DLMP là lĩnh vực trưởng thành (Bai et al. 2022 review trên *Proc. IEEE*, 155 references). Tuy nhiên, zonal pricing **thay đổi theo cấu hình lưới** (vì congestion pattern thay đổi khi switch topology) — khía cạnh này chưa được khai thác.

**Gap C — Honest zero-shot evaluation:** GNN-based controllers claim "zero-shot generalization" nhưng evidence gần đây cho thấy performance giảm đáng kể (de Jong et al. 2025; ACM e-Energy 2025). Cần evaluation có hệ thống trong bối cảnh cụ thể của DSO-VPP coordination.

### Lưu ý quan trọng cho viết paper

> ⚠️ **KHÔNG claim** rằng "VPP topology blindness" là gap mới — đã có >10 bài bi-level DSO-VPP gần đây xử lý vấn đề này.
>
> ⚠️ **KHÔNG claim** rằng "DLMP is underexplored" — đây là mature field.
>
> ⚠️ **KHÔNG claim** rằng "hybrid optimization-RL is novel" — đây là one of the most active areas in power systems AI.
>
> ✅ **NÊN claim**: Sự **kết hợp cụ thể** của reconfiguration + zonal pricing + GAT-MAPPO trong một framework thống nhất là chưa từng có, và evaluation trung thực về topology generalization là contribution có giá trị.

---

## 3. System Architecture

```
┌─────────────────────────────────────────────────────┐
│                    LAYER 0: DSO                      │
│         "Market Maker & Grid Architect"              │
│                                                      │
│  Input:  Load forecast, DER forecast, grid state     │
│  Solve:  MISOCP (Reconfiguration + OPF)              │
│  Output: Topology A_t, Zonal prices Λ_t             │
│  Cycle:  Every 15-60 minutes                         │
│                                                      │
│  ┌─────────────┐    ┌──────────────────┐            │
│  │ Binary vars │───▶│ Fix topology     │            │
│  │ (switches)  │    │ Solve LP → DLMP  │            │
│  └─────────────┘    │ Aggregate → Zone │            │
│                     └──────────────────┘            │
└──────────────┬────────────────┬──────────────────────┘
               │ Λ_t            │ A_t
               ▼                ▼
┌──────────────────────┐  ┌────────────────────────────┐
│     LAYER 1: VPP     │  │       LAYER 2: LOCAL       │
│  "Profit Maximizer"  │  │     "Grid Guardian"        │
│                      │  │                            │
│  Input:  Λ_t         │  │  Input:  P*_ref, A_t,      │
│  Solve:  Wasserstein  │  │          V_i, SoC_i        │
│          DRO          │  │  Solve:  GAT-MAPPO         │
│  Output: P*_ref       │  │  Output: P_i, Q_i per node │
│  Cycle:  15 min       │  │  Cycle:  <1 second         │
│                      │  │                            │
│  NO power flow       │  │  Topology-aware via GNN    │
│  constraints         │  │  Safety: voltage clipping   │
└──────────┬───────────┘  └────────────────────────────┘
           │ P*_ref                ▲
           └───────────────────────┘
```

### Luồng thông tin (Information Flow)

1. **Top-down signals:** DSO → (prices Λ_t, topology A_t) → VPP & Local Controllers
2. **Economic dispatch:** VPP → (reference power P\*_ref) → Local Controllers
3. **Real-time adjustment:** Local Controllers tự động curtail/adjust Q nếu phát hiện voltage violation
4. **Feedback (optional iteration):** Nếu Layer 2 phải curtail quá nhiều → signal ngược lên Layer 1 để re-optimize

### Tần suất hoạt động

| Layer | Cycle Time | Solver | Quyết định |
|-------|-----------|--------|------------|
| 0 | 15-60 min | Gurobi (MISOCP) | Switch states, zonal prices |
| 1 | 15 min | Gurobi/CPLEX (DRO) | P_ref per zone |
| 2 | 0.1-1 s | GAT-MAPPO (inference) | P_i, Q_i per inverter |

---

## 4. Mathematical Formulation

### 4.1 Layer 0: Co-Optimized Reconfiguration & Pricing

#### Objective Function

$$\min_{P, Q, V, \alpha} \sum_{t \in \mathcal{T}} \left[ C_{loss} \cdot P_{loss,t} + C_{sw} \sum_{l \in \mathcal{L}_{sw}} |\alpha_{l,t} - \alpha_{l,t-1}| + C_{vuf} \cdot V_{idx,t} \right]$$

Trong đó:
- $\alpha_{l,t} \in \{0,1\}$: Trạng thái switch $l$ tại thời điểm $t$
- $P_{loss,t}$: Tổn thất công suất tác dụng
- $V_{idx,t}$: Chỉ số mất cân bằng điện áp (voltage unbalance index)
- $C_{sw}$: Chi phí chuyển mạch (switching cost) — penalize thao tác switch quá nhiều

#### Ràng buộc chính

**Branch Flow Model (SOCP relaxation):**

$$P_{ij} = p_{ij} - r_{ij} l_{ij}, \quad Q_{ij} = q_{ij} - x_{ij} l_{ij}$$

$$v_j = v_i - 2(r_{ij} p_{ij} + x_{ij} q_{ij}) + (r_{ij}^2 + x_{ij}^2) l_{ij}$$

$$l_{ij} v_i \geq p_{ij}^2 + q_{ij}^2 \quad \text{(relaxed to SOC)}$$

**Radiality constraints:**

$$\sum_{l \in \mathcal{L}} \alpha_l = |\mathcal{N}| - 1 \quad \text{(spanning tree)}$$

$$\alpha_l \cdot \underline{S}_l \leq |S_l| \leq \alpha_l \cdot \overline{S}_l \quad \text{(big-M for open switches)}$$

**Voltage limits:** $\underline{V}^2 \leq v_i \leq \overline{V}^2, \quad \forall i \in \mathcal{N}$

**Thermal limits:** $p_{ij}^2 + q_{ij}^2 \leq \overline{S}_{ij}^2 \cdot \alpha_{ij}$

#### Zonal Pricing Mechanism

**Bước 1 — Fix topology, solve LP for DLMPs:**

Sau khi cố định $\alpha^*$ (biến nhị phân), bài toán trở thành LP/SOCP liên tục. Dual variables tại ràng buộc power balance cho DLMPs:

$$\lambda^{DLMP}_{i,t} = \lambda^{energy}_{i,t} + \lambda^{loss}_{i,t} + \lambda^{congestion}_{i,t} + \lambda^{voltage}_{i,t}$$

**Bước 2 — Aggregate to zonal prices:**

$$\lambda^{En}_{z,t} = \frac{\sum_{i \in z} P_{load,i} \cdot \lambda^{DLMP}_{i,t}}{\sum_{i \in z} P_{load,i}}$$

> ⚠️ **Known limitation:** Load-weighted averaging phá hủy tín hiệu congestion/voltage cục bộ. Cần thực hiện **sensitivity analysis** so sánh:
> - (a) Load-weighted average (baseline hiện tại)
> - (b) Max-DLMP trong zone (conservative)
> - (c) Congestion-weighted average (proposed improvement)
> - (d) Full nodal pricing (upper bound benchmark)
>
> **TODO trong paper:** Quantify welfare loss từ aggregation bằng Price of Aggregation metric:
> $$PoA = \frac{SW_{nodal} - SW_{zonal}}{SW_{nodal}} \times 100\%$$

**Bước 3 — Reserve pricing:**

$$\lambda^{Res}_{z,t} = \mu_{sys,t} + \eta_{z,t}$$

- $\mu_{sys,t}$: System-wide reserve price (dual of system reserve constraint)
- $\eta_{z,t}$: Zonal scarcity premium (dual of zonal contingency constraint)

#### SOCP Exactness Verification

> ⚠️ **SOCP relaxation exactness** (Gan-Li-Low 2014) holds cho radial networks dưới các điều kiện:
> - Không có upper voltage bound binding
> - Không có reverse power flow lớn
>
> **Implementation requirement (current code):** Sau mỗi lần solve MISOCP, chạy **AC power flow check** (Newton-Raphson trong Pandapower) để verify:
> $$\|V^{SOCP} - V^{AC}\|_\infty < \epsilon_{tol} \quad (\epsilon_{tol} = 0.01 \text{ p.u.},\ publish\ mode)$$
> Nếu gap > threshold, pipeline vận hành theo cơ chế **fail-closed** (không phát hành output hợp lệ cho Layer 1), đồng thời lưu diagnostics để calibration.

### 4.2 Layer 1: Distributionally Robust VPP Bidding

#### Why DRO over Stochastic Optimization

Scenario-based SO (bản gốc) có **69% out-of-sample reliability** theo Li et al. (2022). Wasserstein DRO đạt **>85%** trên cùng test case. Đổi sang DRO là improvement có cơ sở.

#### Formulation (Wasserstein DRO)

$$\max_{P_{inj}, R} \min_{\mathbb{Q} \in \mathcal{B}_\epsilon(\hat{\mathbb{P}})} \mathbb{E}_{\mathbb{Q}} \left[ \sum_{t=1}^{T} \sum_{z \in \mathcal{Z}} \left( \lambda^{En}_{z,t} P_{inj,z,t} + \lambda^{Res}_{z,t} R_{z,t} - C_{deg}(P_{inj}) \right) \right]$$

Trong đó:
- $\mathcal{B}_\epsilon(\hat{\mathbb{P}})$: Wasserstein ball bán kính $\epsilon$ quanh empirical distribution $\hat{\mathbb{P}}$
- $P_{inj,z,t}$: Công suất bơm vào zone $z$ tại thời điểm $t$
- $R_{z,t}$: Dung lượng dự phòng cung cấp cho zone $z$
- $C_{deg}$: Chi phí degradation pin (battery aging cost)

#### Ràng buộc VPP

**Virtual battery model:**

$$SoC_{t+1} = SoC_t + \eta_{ch} P^{ch}_t \Delta t - \frac{P^{dis}_t}{\eta_{dis}} \Delta t$$

$$\underline{SoC} \leq SoC_t \leq \overline{SoC}, \quad 0 \leq P^{ch}_t \leq \overline{P}^{ch}, \quad 0 \leq P^{dis}_t \leq \overline{P}^{dis}$$

**Inverter capacity:** $P_{inj,z,t}^2 + Q_{inj,z,t}^2 \leq \overline{S}_{inv,z}^2$

**Reserve delivery guarantee:** $R_{z,t} \leq \overline{P}^{dis} - P^{dis}_{z,t} + P^{ch}_{z,t}$

> **Lưu ý thiết kế:** Layer 1 **cố tình loại bỏ** power flow constraints. Đây là design choice (không phải oversight):
> - VPP hoạt động như "Commercial VPP" — chỉ quan tâm kinh tế
> - Physical feasibility được đảm bảo bởi Layer 2 (curtailment nếu cần)
> - Nếu Layer 2 phải curtail >5% P_ref → trigger **iterative feedback** (xem Section 6)

#### DRO Tractable Reformulation

Wasserstein DRO với 1-norm distance và linear objective admits LP/SOCP reformulation (Mohajerin Esfahani & Kuhn, 2018):

$$\max_{P, R, \lambda_0, s} \quad \lambda_0 - \epsilon \sum_\omega s_\omega$$

$$\text{s.t.} \quad f(P, R, \xi_\omega) \geq \lambda_0 - s_\omega \|\xi_\omega\|_1, \quad s_\omega \geq 0, \quad \forall \omega$$

Với $\xi_\omega$ là price scenario và $f$ là profit function.

### 4.3 Layer 2: GAT-MAPPO Real-Time Voltage Control ⭐ (Main Contribution)

#### Multi-Agent Setup

- **Agents:** Mỗi DER inverter node $i$ là một agent
- **Paradigm:** Centralized Training, Decentralized Execution (CTDE)
- **Algorithm:** MAPPO (Multi-Agent PPO) với shared GAT encoder

#### Observation Space cho Agent $i$

$$o_i = \left[ \underbrace{V_i, \theta_i, P_{load,i}, Q_{load,i}, P_{gen,i}, SoC_i}_{\text{Local state } s_i^{local}}, \underbrace{P^*_{ref}, \lambda^{Res}_{z(i),t}}_{\text{Global command}}, \underbrace{A_t, X_t}_{\text{Graph structure}} \right]$$

Trong đó:
- $A_t \in \{0,1\}^{N \times N}$: Ma trận kề **thay đổi theo thời gian** (từ Layer 0 reconfiguration)
- $X_t \in \mathbb{R}^{N \times d}$: Node feature matrix (chứa $V$, $P$, $Q$ cho tất cả các nút)

#### GAT Encoder Architecture

```
Input: Graph G_t = (X_t, A_t)
  │
  ▼
┌─────────────────────────────────────┐
│  GAT Layer 1 (K=4 heads, d=32)     │
│  h_i^(1) = ║_{k=1}^{K} σ(Σ_j α^k_ij W^k x_j)  │
│                                     │
│  Attention weights α_ij:            │
│  α_ij = softmax_j(LeakyReLU(       │
│         a^T [W h_i ║ W h_j]))      │
│                                     │
│  Key insight: α_ij tự động học      │
│  "nút nào quan trọng" — nút cổ chai│
│  sẽ có attention weight cao         │
└─────────────┬───────────────────────┘
              │
              ▼
┌─────────────────────────────────────┐
│  GAT Layer 2 (K=1 head, d=64)      │
│  h_i^(2) = σ(Σ_j α_ij W h_j^(1))  │
└─────────────┬───────────────────────┘
              │
              ▼
┌─────────────────────────────────────┐
│  Output: Node embedding z_i ∈ R^64 │
│  → Feed vào Actor & Critic heads    │
└─────────────────────────────────────┘
```

#### Actor-Critic Architecture

```
Actor (decentralized):
  Input:  z_i (node embedding) + s_i^local
  Hidden: MLP [128, 64]
  Output: μ_i, σ_i (Gaussian policy for continuous P_i, Q_i)

Critic (centralized, training only):
  Input:  [z_1, z_2, ..., z_N] (all node embeddings) + global state
  Hidden: MLP [256, 128]
  Output: V(s) (state value)
```

#### Action Space

Agent $i$ outputs continuous actions:

$$a_i = [P_i^{act}, Q_i^{act}] \in [-\overline{P}_i, \overline{P}_i] \times [-\overline{Q}_i, \overline{Q}_i]$$

Với safety clipping:

$$P_i^{final} = \text{clip}(P_i^{act}, P_i^{min}, P_i^{max})$$

$$Q_i^{final} = \begin{cases} Q_i^{act} & \text{if } V_i \in [V_{min}+\epsilon, V_{max}-\epsilon] \\ Q_i^{emergency} & \text{otherwise (droop control fallback)} \end{cases}$$

#### Reward Function

$$R_t = \underbrace{-\alpha \left(\sum_i P_i - P^*_{ref}\right)^2}_{\text{Tracking (how well agents follow L1)}} - \underbrace{\beta \sum_{j \in \mathcal{N}} \max(0, |V_j - 1| - \epsilon_V)^2}_{\text{Voltage violation (L2 penalty)}} + \underbrace{\gamma \cdot \lambda^{Res}_{z,t} \cdot R_{avail,i}}_{\text{Reserve provision bonus}}$$

> ⚠️ **Design tension:** Reserve bonus xung đột với voltage regulation (giữ reserve = giảm Q available).
>
> **Giải pháp đề xuất:**
> - Dùng **adaptive weighting**: $\gamma$ giảm khi voltage violation tăng
> - Hoặc **separate critic heads** cho economic vs safety objectives
> - **Ablation study cần thiết**: Test từng reward component riêng

#### Hyperparameters (Starting Point)

| Parameter | Value | Notes |
|-----------|-------|-------|
| GAT heads (Layer 1) | 4 | Standard cho power systems graphs |
| GAT hidden dim | 32 → 64 | 2-layer progressive |
| Actor MLP | [128, 64] | Decentralized |
| Critic MLP | [256, 128] | Centralized |
| Learning rate | 3e-4 | Adam optimizer |
| Clip ratio (ε) | 0.2 | Standard PPO |
| GAE λ | 0.95 | Generalized Advantage Estimation |
| Discount γ_RL | 0.99 | Near-sighted enough for voltage control |
| Reward weights | α=1.0, β=10.0, γ=0.1 | **Cần tune** — β >> α to prioritize safety |
| Mini-batch size | 256 | |
| Training episodes | 50,000-100,000 | Expect ~24-48h on single GPU |
| N agents | ~15-20 | DER nodes trên IEEE 123-bus |

---

## 5. Experimental Design

### 5.1 Test System

**IEEE 123-bus distribution network** (modified)

> ⚠️ **Single-phase equivalent** — Known limitation:
> - IEEE 123-bus là inherently 3-phase unbalanced
> - Single-phase mất voltage unbalance, phase-specific congestion
> - **Mitigation**: Acknowledge rõ trong paper, note rằng BFM SOCP proofs dùng single-phase models
> - **Nếu muốn mạnh hơn**: Dùng IEEE 33-bus (naturally single-phase) hoặc giữ IEEE 123-bus full 3-phase (phức tạp hơn nhiều)

**Modifications cho reconfiguration và zonal operation:**
- Thêm **5 tie-lines** (normally-open switches) → tạo loop cho reconfiguration.
- **~15-20 DER nodes** (PV + Battery) phân bố trên lưới.
- Zone/VPP partition theo nguyên tắc topology-aware, mỗi zone gồm **nhiều bus** (khuyến nghị 8-25 bus/zone cho IEEE 123-bus), thay vì chia quá nhỏ 2-3 bus.
- Tích hợp **zone partition scoring** để chọn partition tốt nhất theo tiêu chí định lượng (connectivity, load/DER balance, boundary edges, imbalance penalty).

### 5.2 Scenarios

| Scenario Set | Mục đích | Số lượng |
|--------------|----------|----------|
| **S1: Normal operation** | Baseline performance | 100 episodes |
| **S2: High PV penetration** | Voltage rise stress test | 100 episodes |
| **S3: Congestion events** | Zonal pricing activation | 50 episodes |
| **S4: Topology changes** | Zero-shot generalization test | 50 episodes × N topologies |
| **S5: Combined stress** | Worst-case robustness | 50 episodes |

### 5.3 Topology Generalization Experiment ⭐ (Critical Experiment)

Đây là experiment quan trọng nhất — phải thiết kế cẩn thận:

**Training topologies:** Train GAT-MAPPO trên **5 base topologies** (5 cấu hình switch khác nhau của IEEE 123-bus)

**Test protocol:**
```
Level 1 — Interpolation:  Topologies "gần" với training set
                          (thay đổi 1 switch so với base)
                          → Expected: <3% degradation

Level 2 — Extrapolation:  Topologies "xa" với training set
                          (thay đổi 2-3 switches)
                          → Expected: 5-15% degradation

Level 3 — Extreme shift:  Topologies rất khác
                          (thay đổi 4+ switches, different feeder structure)
                          → Expected: >15% degradation

Level 4 — Fine-tuning:    Từ Level 3, fine-tune 100-500 episodes
                          → Measure recovery speed
```

**Metrics cho generalization:**
- **Voltage Violation Rate (VVR):** % timesteps có $|V_i - 1| > 0.05$ p.u.
- **Tracking Error (TE):** $\frac{|P_\Sigma - P^*_{ref}|}{P^*_{ref}} \times 100\%$
- **Generalization Gap (GG):** $\frac{\text{Reward}_{train\_topo} - \text{Reward}_{unseen\_topo}}{\text{Reward}_{train\_topo}} \times 100\%$

> ⚠️ **Honest reporting requirement:**
> - Report GG cho **tất cả** levels, kể cả khi kết quả xấu
> - So sánh với MLP baseline (không có GNN) để isolate GNN contribution
> - So sánh với "retrained from scratch" để đánh giá transfer learning value

### 5.4 Ablation Study Design

| Experiment | Remove/Replace | Đo gì |
|------------|---------------|-------|
| **A1: No GNN** | Thay GAT bằng MLP | Topology awareness contribution |
| **A2: No attention** | Thay GAT bằng GCN (uniform weights) | Attention mechanism value |
| **A3: No reconfiguration** | Fix topology (Layer 0 disabled) | Reconfiguration benefit |
| **A4: No DRO** | Thay DRO bằng deterministic opt | DRO value under uncertainty |
| **A5: No reserve signal** | Remove γ term từ reward | Reserve provision contribution |
| **A6: Single-agent** | Thay MAPPO bằng single PPO | Multi-agent coordination value |
| **A7: No iterative feedback** | Single-pass (no L2→L1 feedback) | Iteration benefit |

### 5.5 Benchmark Comparisons

| Method | Type | Source | Implementation |
|--------|------|--------|----------------|
| **Centralized OPF** | Optimal benchmark (upper bound) | Pandapower AC-OPF | Giải full OPF mỗi timestep — impractical nhưng cho optimal reference |
| **Rule-based droop** | Traditional baseline | IEEE 1547 standard | Droop control Q(V) trên mỗi inverter |
| **Single-agent PPO** | RL baseline (no graph) | Standard PPO | Một agent điều khiển tất cả, MLP policy |
| **MAPPO (no GNN)** | MARL baseline | In-repo MAPPO baseline | Multi-agent nhưng MLP, không có topology awareness |
| **Safe DRL** | Closest competitor | Reproduce Xue et al. (2024) approach | Hierarchical constrained DRL (Lagrangian method) |

### 5.6 Key Performance Indicators

| KPI | Formula | Target |
|-----|---------|--------|
| Voltage Violation Rate | $\frac{\sum_t \mathbb{1}[|V_i-1|>0.05]}{\text{total timesteps}}$ | <1% |
| Average Voltage Deviation | $\frac{1}{NT}\sum_{i,t} |V_{i,t} - 1|$ | <0.02 p.u. |
| Power Tracking Error | $\frac{1}{T}\sum_t |P_\Sigma - P^*_{ref}|/P^*_{ref}$ | <3% |
| Reserve Delivery Rate | $\frac{\text{Delivered reserve}}{\text{Committed reserve}}$ | >95% |
| Inference Time | Wall clock per decision | <100ms |
| Social Welfare Gap vs OPF | $(SW_{OPF} - SW_{proposed})/SW_{OPF}$ | <5% |
| Generalization Gap (Level 2) | See 5.3 | <10% |

---

## 6. Implementation Details

### 6.1 Tech Stack

| Component | Technology | Version | Role |
|-----------|-----------|---------|------|
| Grid Simulation | **Pandapower** | ≥2.13 | AC power flow, feeder model, switch status |
| Optimization L0 | **Pyomo + MOSEK** | Pyomo 6.x, MOSEK | MISOCP reconfiguration + SOCP dual extraction |
| Optimization L1 | **SciPy (linprog)** + fallback greedy | SciPy ≥1.10 | Wasserstein-style DRO scheduling approximation |
| Deep Learning | **PyTorch** | ≥2.0 | Dense GAT encoder + actor-critic |
| RL Training | **Custom MAPPO loop** | In-repo implementation | Multi-agent rollout/update without RLlib dependency |
| Data Processing | **NumPy, Pandas, YAML** | Standard | Profiles, scenarios, configs, metrics export |

### 6.2 Project Structure

```
project/
├── README.md                      # This file
├── configs/
│   ├── grid_ieee123.json          # IEEE 123-bus network data
│   ├── der_placement.json         # DER locations and capacities
│   ├── zone_definition.json       # Zone boundaries
│   ├── training_config.yaml       # RL hyperparameters
│   └── experiment_config.yaml     # Experiment scenarios
│
├── src/
│   ├── layer0_dso/
│   │   ├── reconfiguration.py     # MISOCP formulation in Pyomo
│   │   ├── dlmp_calculator.py     # Extract DLMPs from dual variables
│   │   ├── zonal_pricing.py       # Aggregate DLMPs to zones
│   │   └── socp_validator.py      # AC power flow verification
│   │
│   ├── layer1_vpp/
│   │   ├── dro_bidding.py         # Wasserstein DRO formulation
│   │   ├── scenario_generator.py  # Price scenario generation
│   │   └── virtual_battery.py     # Aggregated battery model
│   │
│   ├── layer2_control/
│   │   ├── gat_encoder.py         # GAT network (PyG)
│   │   ├── actor_critic.py        # Actor-Critic with GAT backbone
│   │   ├── mappo_policy.py        # Custom MAPPO policy for RLlib
│   │   ├── safety_layer.py        # Voltage clipping + droop fallback
│   │   └── reward.py              # Reward function components
│   │
│   ├── environment/
│   │   ├── grid_env.py            # Gym-compatible multi-agent env
│   │   ├── pandapower_backend.py  # Pandapower wrapper
│   │   └── topology_manager.py    # Switch state management
│   │
│   └── utils/
│       ├── graph_utils.py         # Adjacency matrix from Pandapower
│       ├── data_loader.py         # Load profiles, DER profiles
│       └── metrics.py             # KPI calculations
│
├── experiments/
│   ├── train_mappo.py             # Main training script
│   ├── eval_generalization.py     # Topology generalization test
│   ├── run_ablation.py            # Ablation study runner
│   ├── run_benchmarks.py          # Baseline comparisons
│   └── analyze_results.py         # Generate tables & figures
│
├── data/
│   ├── load_profiles/             # 24h load curves (scaled real data)
│   ├── pv_profiles/               # Solar irradiance profiles
│   ├── price_scenarios/           # Generated price scenarios
│   └── topologies/                # Pre-computed valid topologies
│
├── notebooks/
│   ├── 01_grid_visualization.ipynb
│   ├── 02_dlmp_analysis.ipynb
│   ├── 03_training_curves.ipynb
│   └── 04_results_visualization.ipynb
│
├── tests/
│   ├── test_reconfiguration.py
│   ├── test_dro.py
│   ├── test_gat_encoder.py
│   └── test_environment.py
│
└── requirements.txt
```

### 6.3 Critical Integration Points

**PyG ↔ RLlib Bridge (quan trọng nhất):**

RLlib expects Dict observation space. PyG expects `torch_geometric.data.Data` objects. Cần custom wrapper:

```python
# Pseudo-code for the bridge
class GATRLlibPolicy(TorchPolicy):
    def __init__(self, ...):
        self.gat_encoder = GATEncoder(in_dim, hidden_dim, heads)

    def compute_actions(self, obs_batch):
        # obs_batch is Dict with keys: 'local', 'global', 'adj_matrix', 'node_features'
        # Convert to PyG Data object
        edge_index = adj_to_edge_index(obs_batch['adj_matrix'])
        data = Data(x=obs_batch['node_features'], edge_index=edge_index)
        # GAT forward
        node_embeddings = self.gat_encoder(data)
        # Extract this agent's embedding
        agent_embedding = node_embeddings[agent_id]
        # Concat with local obs → Actor MLP
        action = self.actor(torch.cat([agent_embedding, obs_batch['local']]))
        return action
```

**Pandapower ↔ Gym Environment:**

```python
# Pseudo-code for environment step
class GridEnv(MultiAgentEnv):
    def step(self, action_dict):
        # Apply agent actions to Pandapower network
        for agent_id, action in action_dict.items():
            bus_idx = self.agent_to_bus[agent_id]
            self.net.sgen.at[bus_idx, 'p_mw'] = action[0]
            self.net.sgen.at[bus_idx, 'q_mvar'] = action[1]

        # Run power flow
        pp.runpp(self.net, algorithm='nr')

        # Extract observations and rewards
        voltages = self.net.res_bus.vm_pu.values
        ...
```

### 6.4 Implementation Roadmap

```
Phase 1: Foundation (Tuần 1-3)
├── Setup IEEE 123-bus trong Pandapower
├── Implement switch control + topology manager
├── Verify SOCP formulation cho 2-3 topologies
├── Create Gym environment with fixed topology
└── Milestone: Environment chạy được, power flow converge

Phase 2: Layer 0 + Layer 1 (Tuần 4-6)
├── Implement MISOCP trong Pyomo
├── Extract DLMPs, implement zonal pricing
├── Implement Wasserstein DRO (hoặc scenario-based SO trước)
├── Verify economic dispatch results
└── Milestone: L0+L1 chạy end-to-end, prices hợp lý

Phase 3: Layer 2 Core (Tuần 7-10)  ⭐ Focus
├── Implement GAT encoder trong PyG
├── Build PyG-RLlib bridge
├── Train MAPPO trên FIXED topology (sanity check)
├── Verify voltage regulation converges
└── Milestone: Agents học được voltage control trên 1 topology

Phase 4: Integration & Topology (Tuần 11-14)
├── Connect L0 → L2 (dynamic topology)
├── Train trên 5 base topologies
├── Implement safety layer (droop fallback)
├── Implement iterative feedback L2 → L1
└── Milestone: Full 3-layer pipeline chạy end-to-end

Phase 5: Experiments (Tuần 15-18)
├── Run generalization experiments (Level 1-4)
├── Run ablation studies (A1-A7)
├── Run benchmark comparisons
├── Generate figures and tables
└── Milestone: Tất cả experiments hoàn thành

Phase 6: Paper Writing (Tuần 19-22)
├── Draft introduction + literature review
├── Write methodology (re-use phần lớn từ doc này)
├── Write results & discussion
├── Revise, get feedback, submit
└── Milestone: Paper submitted to IEEE TSG
```

---

## 7. Expected Results & Honest Predictions

### What we expect to show:

1. **GAT-MAPPO achieves <1% voltage violation rate** trên training topologies, competitive với centralized OPF nhưng nhanh hơn 100-1000x.

2. **Generalization gap 5-15%** trên unseen topologies (Level 2), **giảm xuống <5% với 100-500 episodes fine-tuning** — consistent với de Jong et al. (2025) findings.

3. **GAT outperforms MLP baseline by 20-40%** trên unseen topologies (ablation A1), demonstrating GNN's structural inductive bias.

4. **Reconfiguration + pricing** (Layers 0+1) **giảm total cost 5-10%** so với fixed-topology baselines, consistent với reconfiguration literature.

5. **Iterative feedback** giảm curtailment rate 30-50% so với single-pass coordination.

### What might NOT work (honest risks):

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| MAPPO training unstable with dynamic topology | Medium | Curriculum learning: start fixed, gradually add changes |
| SOCP relaxation gap large under high DER | Low-Medium | AC power flow verification + penalty method |
| DRO formulation too conservative | Medium | Tune Wasserstein radius ε via cross-validation |
| Inference time >100ms | Low | Smaller GAT, quantization if needed |
| Reward function imbalance (safety vs economy) | High | Extensive hyperparameter sweep, consider constrained RL |

---

## 8. Paper Outline (IEEE TSG Format)

```
I.   Introduction (1.5 pages)
     - Motivation: DER integration, DSO-VPP coordination challenge
     - Literature: Bilevel DSO-VPP frameworks → gap in topology adaptation
     - Contribution summary (C1, C2, C3)

II.  System Model (2 pages)
     - Tri-level architecture
     - Layer 0: MISOCP formulation (brief — cite conference paper foundation)
     - Layer 1: DRO formulation (brief — cite conference paper)

III. GAT-MAPPO for Topology-Adaptive Voltage Control (3 pages)  ⭐
     - GAT encoder design
     - MAPPO with CTDE
     - Reward design and safety layer
     - Training procedure with topology curriculum

IV.  Simulation Setup (1.5 pages)
     - Test system, scenarios, baselines

V.   Results and Discussion (3 pages)
     - Performance comparison (Table + Figures)
     - Generalization experiment (key contribution)
     - Ablation study
     - Computational performance

VI.  Conclusion (0.5 pages)

     Total: ~12 pages (IEEE TSG limit: 12-14 pages)
```

---

## 9. References to Cite (Must-cite)

### Direct competitors (benchmark against these):
- Xue et al. (2024), "Privacy-preserving multi-level co-regulation of VPPs via hierarchical safe DRL," *Applied Energy*
- Lin et al. (2025), "Coordinated DSO-VPP operation framework with energy and reserve," *Applied Energy*
- Sun et al. (2024), "Collaborative operation optimization of DSO-VPP using MADRL," *IET GTD*

### DLMP foundation (do NOT overclaim novelty here):
- Bai et al. (2022), "Distribution LMP: Fundamentals and Applications," *Proc. IEEE*
- Papavasiliou (2018), "Analysis of DLMP," *IEEE Trans. Power Systems*

### GNN for power systems (position our work):
- Donon et al. (2020), "Neural networks for power flow," *PSCC*
- Owerko et al. (2025), "PowerGNN: Topology-aware GNN for electricity grids," *arXiv*
- de Jong et al. (2025), "Generalizable GNN for robust power grid topology control," *arXiv*

### MARL for power systems:
- Wang et al. (2021), "Multi-agent RL for active voltage control," *NeurIPS*
- CommonPower framework (2024), "MAPPO for safe grid control"

### Reconfiguration:
- Jabr et al. (2012), "Minimum loss reconfiguration via MICP," *IEEE Trans. Power Systems*
- Gan, Li, Low (2014), "Exact convex relaxation of OPF," *IEEE Trans. Power Systems*

### DRO:
- Mohajerin Esfahani & Kuhn (2018), "Data-driven DRO using Wasserstein metric," *Math. Programming*
- Li et al. (2022), "DRO vs SO for VPP scheduling comparison"