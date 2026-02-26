# Topology-Adaptive Voltage Control via Graph Attention MARL in Tri-Level DSO-VPP Coordination

**Target Journal:** IEEE Transactions on Smart Grid (TSG)

**Working Title:** *"Topology-Adaptive Multi-Agent Voltage Control Using Graph Attention Networks for Coordinated DSO-VPP Dispatch in Active Distribution Networks"*

---

## 1. Executive Summary

Dự án phát triển một khung điều khiển phân cấp 3 lớp (Tri-Level Hierarchical Framework) cho sự phối hợp giữa Nhà vận hành lưới phân phối (DSO) và Nhà máy điện ảo (VPP), trong đó **đóng góp chính (main contribution)** là thuật toán điều khiển điện áp thời gian thực tại Layer 2 sử dụng **Graph Attention Network - Multi-Agent Proximal Policy Optimization (GAT-MAPPO)**. Dynamic zone partitioning là hướng mở rộng đang được triển khai từng phần (hiện đã có scoring/selection API).

### Novelty Claims

| # | Claim | Scope | Cơ sở |
|---|-------|-------|-------|
| **C1** | Kết hợp **network reconfiguration** như một layer chiến lược trong DSO-VPP coordination framework | Architectural | Không có bài báo nào (2022–2025) kết hợp reconfiguration switching với multi-level DSO-VPP market mechanism |
| **C2** | **GAT-MAPPO** cho distribution voltage control dưới dynamic topology | Algorithmic | GAT và MAPPO đã tồn tại riêng lẻ; combination cụ thể này cho voltage control chưa được published |
| **C3** | **Honest evaluation** của zero-shot topology generalization cho GNN-based controllers | Empirical | Literature (2024–2025) cho thấy 3–15% degradation trên unseen topology; chưa có evaluation có hệ thống trong bối cảnh DSO-VPP |
| **C4** | **Dynamic zone partitioning** thích ứng theo topology — zone boundaries tự động thay đổi khi DSO reconfigure lưới | Methodological (planned extension; partial implementation) | Các bài VPP zoning hiện tại dùng partition cố định; code hiện có scoring/selection API, dynamic re-partitioning theo từng chu kỳ đang in progress |

### Positioning vs. State-of-the-Art

| Paper | Approach | Thiếu gì so với framework này |
|-------|----------|-------------------------------|
| Xue et al. (2024, *Applied Energy*) | Hierarchical Safe DRL cho DSO-VPP, đạt 1.46% of optimal | Không có reconfiguration, không dùng GNN, zone cố định |
| Lin et al. (2025, *Applied Energy*) | Stackelberg-cooperative game DSO-VPP với shared energy storage | Không có topology adaptation, không real-time control |
| Sun et al. (2024, *IET GTD*) | Multi-agent DRL + AC-OPF cho DSO-VPP | Không có graph-based architecture, topology cố định |
| Mi et al. (2024, *SSRN*) | Tri-level VPP-Prosumer robust game | Không có reconfiguration, không có AI-based real-time layer |

### Lưu ý quan trọng cho viết paper

> ⚠️ **KHÔNG claim** rằng "VPP topology blindness" là gap mới — đã có >10 bài bi-level DSO-VPP gần đây.
>
> ⚠️ **KHÔNG claim** rằng "DLMP is underexplored" — đây là mature field (Bai et al. 2022, *Proc. IEEE*, 155 refs).
>
> ⚠️ **KHÔNG claim** rằng "hybrid optimization-RL is novel" — đây là one of the most active areas in power systems AI.
>
> ✅ **NÊN claim**: Sự **kết hợp cụ thể** của reconfiguration + dynamic zoning + GAT-MAPPO trong framework thống nhất là chưa từng có; evaluation trung thực về topology generalization là contribution có giá trị.

---

## 2. Problem Statement & Research Gaps

### Primary Gap: Topology-Adaptive Real-Time Control trong DSO-VPP Coordination

Các framework DSO-VPP hiện tại (bi-level hoặc tri-level) đều **giả định cấu trúc lưới cố định**. Khi DSO thực hiện network reconfiguration (đóng/cắt switch), hai hệ quả chưa được giải quyết:

1. **Agent control failure:** Các RL agent được train trên topology cũ mất hiệu lực — "topology distribution shift".
2. **Zone boundary invalidation:** Zone partition cố định có thể bị phá vỡ (zone mất tính liên thông) khi switch thay đổi.

> **Không có framework nào tích hợp: (a) network reconfiguration như biến quyết định chiến lược, (b) dynamic zone partitioning thích ứng theo topology, và (c) real-time voltage control topology-adaptive thông qua GNN.**

### Supporting Gaps

**Gap A — Reconfiguration-embedded coordination:** Các bài reconfiguration (Jabr 2012, Qiao 2022) xử lý như bài toán single-agent DSO. Các bài DSO-VPP coordination (Xue 2024, Lin 2025) không xét reconfiguration. Chưa ai kết hợp cả hai.

**Gap B — Dynamic zoning under reconfiguration:** DLMP là lĩnh vực trưởng thành (Bai et al. 2022). Tuy nhiên, khi topology thay đổi, congestion pattern thay đổi → zone boundaries tối ưu cũng thay đổi. Hiện tại mọi bài VPP zoning dùng partition cố định.

**Gap C — Honest zero-shot evaluation:** GNN-based controllers claim "zero-shot generalization" nhưng evidence gần đây (de Jong et al. 2025; ACM e-Energy 2025) cho thấy 3–15% performance degradation. Cần evaluation có hệ thống trong bối cảnh DSO-VPP.

---

## 3. System Architecture

```
┌──────────────────────────────────────────────────────────┐
│                      LAYER 0: DSO                         │
│           "Market Maker & Grid Architect"                 │
│                                                           │
│  Input:  Load forecast, DER forecast, grid state          │
│  Solve:  MISOCP (Reconfiguration + OPF)                   │
│          + Graph Partitioning (Dynamic Zoning)             │
│  Output: Topology A_t, Zone partition Z_t, Prices Λ_t    │
│  Cycle:  Every 15–60 minutes                              │
│                                                           │
│  ┌──────────┐   ┌──────────────┐   ┌────────────────┐   │
│  │ MISOCP   │──▶│ Fix topology │──▶│ Re-partition   │   │
│  │ (switch  │   │ Solve SOCP   │   │ zones on A_t*  │   │
│  │  vars)   │   │ → DLMPs      │   │ (spectral/     │   │
│  └──────────┘   └──────────────┘   │  scoring)      │   │
│                                     └────────────────┘   │
└────────────┬──────────────┬──────────────┬───────────────┘
             │ Λ_t          │ A_t          │ Z_t
             ▼              ▼              ▼
┌────────────────────────┐  ┌──────────────────────────────┐
│      LAYER 1: VPP      │  │        LAYER 2: LOCAL        │
│   "Profit Maximizer"   │  │      "Grid Guardian"         │
│                        │  │                              │
│  Input:  Λ_t, Z_t     │  │  Input:  P*_ref, A_t, Z_t,  │
│  Solve:  Wasserstein   │  │          V_i, SoC_i          │
│          DRO           │  │  Solve:  GAT-MAPPO           │
│  Output: P*_ref/zone   │  │  Output: P_i, Q_i per node  │
│  Cycle:  15 min        │  │  Cycle:  < 1 second          │
│                        │  │                              │
│  NO power flow         │  │  Topology-aware via GAT      │
│  constraints           │  │  Safety: droop fallback      │
└────────────┬───────────┘  └──────────────────────────────┘
             │ P*_ref                ▲
             └───────────────────────┘
             (Iterative feedback nếu curtail > 5%)
```

### Information Flow

1. **Top-down:** DSO → (prices Λ_t, topology A_t, zone partition Z_t) → VPP & Local Controllers
2. **Economic dispatch:** VPP → (reference power P\*_ref per zone) → Local Controllers
3. **Real-time adjustment:** Local Controllers tự động curtail/adjust Q nếu phát hiện voltage violation
4. **Iterative feedback:** Nếu Layer 2 phải curtail >5% P_ref → signal lên Layer 1 để re-optimize

### Operating Frequency

| Layer | Cycle Time | Solver | Decision Variables |
|-------|-----------|--------|-------------------|
| 0 | 15–60 min | MOSEK (MISOCP) + graph partitioning | Switch states, zone boundaries, zonal prices |
| 1 | 15 min | SciPy/HiGHS (LP/DRO) | P_ref per zone |
| 2 | 0.1–1 s | GAT-MAPPO (inference) | P_i, Q_i per inverter |

---

## 4. Mathematical Formulation

### 4.1 Layer 0: Co-Optimized Reconfiguration, Zoning & Pricing

#### 4.1.1 Reconfiguration (MISOCP)

**Objective:**

$$\min_{P, Q, V, \alpha} \sum_{t \in \mathcal{T}} \left[ C_{loss} \cdot P_{loss,t} + C_{sw} \sum_{l \in \mathcal{L}_{sw}} |\alpha_{l,t} - \alpha_{l,t-1}| + C_{vuf} \cdot V_{idx,t} \right]$$

- $\alpha_{l,t} \in \{0,1\}$: Switch $l$ state at time $t$
- $P_{loss,t}$: Active power losses
- $V_{idx,t}$: Voltage unbalance index
- $C_{sw}$: Switching cost penalty

**Branch Flow Model (SOCP relaxation):**

$$P_{ij} = p_{ij} - r_{ij} l_{ij}, \quad Q_{ij} = q_{ij} - x_{ij} l_{ij}$$

$$v_j = v_i - 2(r_{ij} p_{ij} + x_{ij} q_{ij}) + (r_{ij}^2 + x_{ij}^2) l_{ij}$$

$$l_{ij} v_i \geq p_{ij}^2 + q_{ij}^2 \quad \text{(relaxed to SOC cone)}$$

**Radiality:** $\sum_{l \in \mathcal{L}} \alpha_l = |\mathcal{N}| - 1$ (spanning tree)

**Big-M switching:** $\alpha_l \cdot \underline{S}_l \leq |S_l| \leq \alpha_l \cdot \overline{S}_l$

**Voltage/thermal limits:** $\underline{V}^2 \leq v_i \leq \overline{V}^2$, $\;p_{ij}^2 + q_{ij}^2 \leq \overline{S}_{ij}^2 \cdot \alpha_{ij}$

**SOCP Exactness Verification:**

> Sau mỗi lần solve MISOCP, chạy AC power flow (Newton-Raphson, Pandapower) để verify:
> $$\|V^{SOCP} - V^{AC}\|_\infty < \epsilon_{tol} = 0.01 \text{ p.u.}$$
> Pipeline **fail-closed**: nếu gap > threshold, không xuất output cho Layer 1.
> Tolerance 0.01 p.u. ≈ 1% voltage error — chấp nhận được cho distribution-level studies; tighter tolerance (0.001) không đạt được ổn định trên IEEE 123-bus single-phase equivalent do SOCP relaxation gap at high-DER buses.

**Current calibration status:** Bus-82 calibration fix đã áp dụng (siết transformer tap handling + voltage reference band 0.005 p.u.), đạt `fails=0/288, max_gap≈0.00482 p.u.`.

#### 4.1.2 Dynamic Zone Partitioning (NEW — C4)

Sau khi cố định topology $A_t^*$, Layer 0 chạy **graph partitioning** trên lưới mới để tìm zone boundaries tối ưu.

**Formulation — Spectral clustering trên weighted power-flow graph:**

Cho graph $G_t = (\mathcal{N}, \mathcal{E}, W_t)$ với edge weight $w_{ij} = |P_{ij}| + |Q_{ij}|$ (power flow magnitude):

$$\min_{\mathcal{Z}} \sum_{(i,j) \in \text{cut}(\mathcal{Z})} w_{ij,t}$$

$$\text{s.t.} \quad \frac{\max_z |\mathcal{N}_z| - \min_z |\mathcal{N}_z|}{|\mathcal{N}|/K} \leq \delta_{bal} \quad \text{(size balance)}$$

$$\sum_{i \in z} P_{DER,i} > 0 \quad \forall z \in \mathcal{Z} \quad \text{(mỗi zone có DER)}$$

$$G[\mathcal{N}_z] \text{ is connected} \quad \forall z \in \mathcal{Z} \quad \text{(zone connectivity)}$$

**Implementation:** Giải bằng spectral clustering (Laplacian eigenvectors) hoặc multi-criteria scoring:

| Criterion | Weight | Description |
|-----------|--------|-------------|
| Connectivity | Mandatory | Mỗi zone phải là subgraph liên thông trên $A_t^*$ |
| Bus/load balance | 0.25 | Cân bằng số bus và tổng tải giữa các zone |
| DER penetration | 0.25 | Mỗi zone phải có DER; phân bố DER cân bằng |
| Boundary cuts | 0.30 | Minimize inter-zone power flow (tight coupling → same zone) |
| Imbalance penalty | 0.20 | Phạt zone quá lớn hoặc quá nhỏ |

> **Key insight:** Khi DSO đóng/cắt switch, graph $G_t$ thay đổi → spectral clustering cho partition khác → zone boundaries **tự động thích ứng**. Đây là điểm khác biệt so với mọi bài VPP zoning hiện tại (zone cố định).

**Recommended zone size:** 8–25 buses/zone cho IEEE 123-bus (tổng ~120 buses, 4–6 zones).

#### 4.1.3 Zonal Pricing

**Bước 1 — DLMP extraction:** Fix $\alpha^*$, solve SOCP → dual variables at power balance constraints:

$$\lambda^{DLMP}_{i,t} = \lambda^{energy}_{i,t} + \lambda^{loss}_{i,t} + \lambda^{congestion}_{i,t} + \lambda^{voltage}_{i,t}$$

**Bước 2 — Aggregate to zonal prices:**

$$\lambda^{En}_{z,t} = \frac{\sum_{i \in z} P_{load,i} \cdot \lambda^{DLMP}_{i,t}}{\sum_{i \in z} P_{load,i}}$$

> ⚠️ **Known limitation:** Load-weighted averaging phá hủy tín hiệu congestion cục bộ. Sensitivity analysis cần so sánh:
> - (a) Load-weighted average (baseline)
> - (b) Max-DLMP trong zone (conservative)
> - (c) Congestion-weighted average
> - (d) Full nodal pricing (upper bound)
>
> **Quantify bằng Price of Aggregation:**
> $$PoA = \frac{SW_{nodal} - SW_{zonal}}{SW_{nodal}} \times 100\%$$

**Bước 3 — Reserve pricing:**

$$\lambda^{Res}_{z,t} = \mu_{sys,t} + \eta_{z,t}$$

- $\mu_{sys,t}$: System-wide reserve price (dual of system reserve constraint)
- $\eta_{z,t}$: Zonal scarcity premium (dual of zonal contingency constraint)

### 4.2 Layer 1: Distributionally Robust VPP Bidding

#### Why DRO over Stochastic Optimization

Li et al. (2022) showed: scenario-based SO achieves 69% out-of-sample reliability; Wasserstein DRO achieves >85% on the same test case.

#### Formulation (Wasserstein DRO)

$$\max_{P_{inj}, R} \min_{\mathbb{Q} \in \mathcal{B}_\epsilon(\hat{\mathbb{P}})} \mathbb{E}_{\mathbb{Q}} \left[ \sum_{t=1}^{T} \sum_{z \in \mathcal{Z}_t} \left( \lambda^{En}_{z,t} P_{inj,z,t} + \lambda^{Res}_{z,t} R_{z,t} - C_{deg}(P_{inj}) \right) \right]$$

> Lưu ý: $\mathcal{Z}_t$ (zone set) có thể thay đổi theo $t$ do dynamic zoning từ Layer 0.

- $\mathcal{B}_\epsilon(\hat{\mathbb{P}})$: Wasserstein ball radius $\epsilon$ around empirical distribution
- $C_{deg}$: Battery degradation cost

#### VPP Constraints

**Virtual battery:**

$$SoC_{t+1} = SoC_t + \eta_{ch} P^{ch}_t \Delta t - \frac{P^{dis}_t}{\eta_{dis}} \Delta t$$

$$\underline{SoC} \leq SoC_t \leq \overline{SoC}, \quad 0 \leq P^{ch}_t \leq \overline{P}^{ch}, \quad 0 \leq P^{dis}_t \leq \overline{P}^{dis}$$

**Inverter capacity:** $P_{inj,z,t}^2 + Q_{inj,z,t}^2 \leq \overline{S}_{inv,z}^2$

**Reserve delivery:** $R_{z,t} \leq \overline{P}^{dis} - P^{dis}_{z,t} + P^{ch}_{z,t}$

> **Design choice:** Layer 1 **cố tình loại bỏ** power flow constraints (Commercial VPP paradigm). Physical feasibility do Layer 2 đảm bảo. Nếu Layer 2 curtail >5% P_ref → trigger iterative feedback.

#### Tractable Reformulation

Wasserstein DRO với 1-norm + linear objective → LP (Mohajerin Esfahani & Kuhn, 2018):

$$\max_{P, R, \lambda_0, s} \quad \lambda_0 - \epsilon \sum_\omega s_\omega$$

$$\text{s.t.} \quad f(P, R, \xi_\omega) \geq \lambda_0 - s_\omega \|\xi_\omega\|_1, \quad s_\omega \geq 0, \quad \forall \omega$$

> **Implementation note:** Giải bằng SciPy `linprog` (HiGHS backend). Đủ cho proof-of-concept (~100 scenarios × 24 timesteps). Production-scale sẽ cần Gurobi/MOSEK.

### 4.3 Layer 2: GAT-MAPPO Real-Time Voltage Control ⭐ (Main Contribution)

#### Multi-Agent Setup

- **Agents:** Mỗi DER inverter node $i$ là một agent (~15–20 agents trên IEEE 123-bus)
- **Paradigm:** Centralized Training, Decentralized Execution (CTDE)
- **Algorithm:** MAPPO (custom implementation, không phụ thuộc RLlib)
- **GNN backbone:** GAT (implemented: dense PyTorch; planned option: PyG migration)

#### Observation Space

$$o_i = \left[ \underbrace{V_i, \theta_i, P_{load,i}, Q_{load,i}, P_{gen,i}, SoC_i}_{\text{Local state}}, \underbrace{P^*_{ref}, \lambda^{Res}_{z(i),t}}_{\text{Global command}}, \underbrace{A_t, X_t}_{\text{Graph structure}} \right]$$

- $A_t \in \{0,1\}^{N \times N}$: Adjacency matrix **thay đổi theo thời gian** (từ Layer 0)
- $X_t \in \mathbb{R}^{N \times d}$: Node feature matrix ($V$, $P$, $Q$ tại mọi bus)

#### GAT Encoder Architecture

```
Input: Graph G_t = (X_t, A_t)
  │
  ▼
┌─────────────────────────────────────────┐
│  GAT Layer 1 (K=4 heads, d_hidden=32)   │
│  h_i^(1) = ║_{k=1}^K σ(Σ_j α^k_ij W^k x_j)  │
│                                          │
│  Attention: α_ij = softmax_j(            │
│    LeakyReLU(a^T [Wh_i ║ Wh_j]))       │
│                                          │
│  → α_ij cao = nút j quan trọng cho i   │
│  → tự học "bottleneck nodes"            │
└──────────────┬──────────────────────────┘
               ▼
┌─────────────────────────────────────────┐
│  GAT Layer 2 (K=1 head, d_out=64)       │
│  h_i^(2) = σ(Σ_j α_ij W h_j^(1))      │
└──────────────┬──────────────────────────┘
               ▼
┌─────────────────────────────────────────┐
│  Output: z_i ∈ ℝ^64 (node embedding)    │
│  → Feed into Actor & Critic heads       │
└─────────────────────────────────────────┘
```

#### Actor-Critic Architecture

```
Actor (decentralized — chạy khi inference):
  Input:  z_i (GAT embedding) ⊕ s_i^local (6-dim)
  Hidden: MLP [128, 64]
  Output: μ_i, σ_i → Gaussian policy cho [P_i, Q_i]

Critic (centralized — chỉ dùng khi training):
  Input:  [z_1, ..., z_N] (all embeddings) ⊕ global state
  Hidden: MLP [256, 128]
  Output: V(s) (state value estimate)
```

#### Action Space & Safety Layer

Agent $i$ outputs continuous: $a_i = [P_i^{act}, Q_i^{act}]$

**Safety clipping:**

$$P_i^{final} = \text{clip}(P_i^{act}, P_i^{min}, P_i^{max})$$

$$Q_i^{final} = \begin{cases} Q_i^{act} & \text{if } V_i \in [V_{min}+\epsilon, V_{max}-\epsilon] \\ Q_i^{droop}(V_i) & \text{otherwise (droop control fallback)} \end{cases}$$

Droop fallback đảm bảo an toàn ngay cả khi GAT-MAPPO policy cho action xấu (e.g., trên unseen topology).

#### Reward Function

$$R_t = \underbrace{-\alpha \left(\sum_i P_i - P^*_{ref}\right)^2}_{\text{Tracking error}} - \underbrace{\beta \sum_{j \in \mathcal{N}} \max(0, |V_j - 1| - \epsilon_V)^2}_{\text{Voltage violation (L2)}} + \underbrace{\gamma_t \cdot \lambda^{Res}_{z,t} \cdot R_{avail,i}}_{\text{Reserve bonus}}$$

**Adaptive weighting** để giải quyết xung đột safety vs economy:

$$\gamma_t = \gamma_0 \cdot \max\left(0, 1 - \frac{\text{ViolationCount}_t}{\text{ViolationThreshold}}\right)$$

Khi voltage violation tăng → $\gamma_t$ giảm → agent ưu tiên safety hơn reserve provision.

#### Hyperparameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| GAT heads (Layer 1) | 4 | Standard for power-system-sized graphs |
| GAT hidden dim | 32 → 64 | 2-layer progressive |
| Actor MLP | [128, 64] | Decentralized |
| Critic MLP | [256, 128] | Centralized |
| Learning rate | 3e-4 | Adam |
| PPO clip ratio | 0.2 | Standard |
| GAE λ | 0.95 | |
| Discount γ_RL | 0.99 | |
| Reward weights | α=1.0, β=10.0, γ₀=0.1 | β >> α → safety first |
| Mini-batch | 256 | |
| Training episodes | 50k–100k | ~24–48h single GPU |
| N agents | 15–20 | DER nodes on IEEE 123-bus |

---

## 5. Experimental Design

### 5.1 Test System

**IEEE 123-bus distribution network** (single-phase equivalent, modified)

**Modifications:**
- 5 tie-lines (normally-open switches) cho reconfiguration
- ~15–20 DER nodes (PV + Battery)
- 4–6 zones via dynamic partitioning (8–25 buses/zone)
- Zone partition scoring: connectivity (mandatory), load/DER balance, boundary cuts, imbalance penalty

> ⚠️ **Known limitation:** IEEE 123-bus là inherently 3-phase unbalanced. Single-phase equivalent mất voltage unbalance effects, phase-specific congestion.
> **Mitigation:** Acknowledge rõ trong paper; BFM SOCP proofs sử dụng single-phase models; IEEE 33-bus/69-bus là alternatives nếu cần natural single-phase.

### 5.2 Scenarios

| Set | Purpose | Count |
|-----|---------|-------|
| **S1:** Normal operation | Baseline | 100 episodes |
| **S2:** High PV penetration | Voltage rise stress | 100 episodes |
| **S3:** Congestion events | Zonal pricing activation | 50 episodes |
| **S4:** Topology changes | Zero-shot generalization | 50 episodes × N topologies |
| **S5:** Combined stress | Worst-case robustness | 50 episodes |

### 5.3 Topology Generalization Experiment ⭐

**Training:** GAT-MAPPO trained trên **5 base topologies** (5 cấu hình switch khác nhau).

**Test protocol:**

| Level | Description | Expected Degradation |
|-------|-------------|---------------------|
| **L1 — Interpolation** | Thay đổi 1 switch vs base | <3% |
| **L2 — Extrapolation** | Thay đổi 2–3 switches | 5–15% |
| **L3 — Extreme shift** | Thay đổi 4+ switches | >15% |
| **L4 — Fine-tuning** | Từ L3, fine-tune 100–500 episodes | Measure recovery → expect <5% |

**Metrics:**
- **VVR (Voltage Violation Rate):** % timesteps có $|V_i - 1| > 0.05$ p.u.
- **TE (Tracking Error):** $|P_\Sigma - P^*_{ref}|/P^*_{ref} \times 100\%$
- **GG (Generalization Gap):** $(\text{Reward}_{train} - \text{Reward}_{unseen})/\text{Reward}_{train} \times 100\%$

> **Honest reporting:** Report GG cho **tất cả** levels kể cả khi xấu. So sánh vs MLP baseline để isolate GNN contribution. So sánh vs retrained-from-scratch để đánh giá transfer value.

### 5.4 Ablation Studies

| ID | Remove/Replace | Measures |
|----|---------------|----------|
| **A1** | GAT → MLP (no GNN) | Topology awareness contribution |
| **A2** | GAT → GCN (no attention) | Attention mechanism value |
| **A3** | Fix topology (disable L0 reconfig) | Reconfiguration benefit |
| **A4** | DRO → deterministic optimization | DRO value under uncertainty |
| **A5** | Remove reserve bonus (γ=0) | Reserve signal contribution |
| **A6** | MAPPO → single-agent PPO | Multi-agent coordination value |
| **A7** | Single-pass (no L2→L1 feedback) | Iterative feedback benefit |
| **A8** | Fixed zones vs scoring-selected zones | Zone scoring value (implemented) |
| **A9** | 3 zones vs 5 zones vs nodal pricing | Zone granularity trade-off |
| **A10** | Scoring-selected vs dynamic re-partitioning | Dynamic zoning incremental value (planned) |

### 5.5 Benchmark Comparisons

| Method | Type | Implementation |
|--------|------|----------------|
| **Centralized OPF** | Upper bound (optimal) | Pandapower AC-OPF every timestep |
| **Rule-based droop** | Traditional baseline | IEEE 1547 Q(V) droop |
| **Single-agent PPO** | RL baseline (no graph) | One agent, MLP policy |
| **MAPPO (no GNN)** | MARL baseline | Multi-agent MLP (in-repo) |
| **Safe DRL (Xue et al.)** | Closest competitor | Reproduce hierarchical constrained DRL |

### 5.6 Key Performance Indicators

| KPI | Target |
|-----|--------|
| Voltage Violation Rate (VVR) | <1% |
| Avg Voltage Deviation | <0.02 p.u. |
| Power Tracking Error | <3% |
| Reserve Delivery Rate | >95% |
| Inference Time | <100 ms |
| Social Welfare Gap vs OPF | <5% |
| Generalization Gap (Level 2) | <10% |
| Price of Aggregation (dynamic zoning) | <3% (improvement over fixed) |

---

## 6. Tech Stack & Implementation

### 6.1 Technology Choices

| Component | Technology | Role |
|-----------|-----------|------|
| Grid simulation | **Pandapower** ≥2.13 | AC power flow, network model, switch control |
| Layer 0 optimization | **Pyomo + MOSEK** | MISOCP reconfiguration + SOCP dual extraction |
| Layer 0 zoning | **In-repo scoring API (NumPy/Pandas)** | Implemented: zone scoring/selection in `layer0_dso.py`; planned: spectral clustering dynamic re-partition |
| Layer 1 optimization | **SciPy** `linprog` (HiGHS) | DRO LP reformulation. Sufficient for ~100 scenarios × 24h. Production: Gurobi/MOSEK |
| GNN engine | **PyTorch (dense GAT implementation)** | Implemented in `src/layer2_control/gat_encoder.py`; planned migration path: PyG if needed |
| RL training | **Custom MAPPO loop** (in-repo, no RLlib) | Multi-agent rollout/update. Avoids RLlib dependency hell |
| Experiment tracking | **CSV logs (implemented)** / W&B (planned) | Current experiments log to artifacts CSV; online tracking optional |
| Data processing | **NumPy, Pandas, PyYAML** | Profiles, configs, metrics |

### 6.2 Project Structure (Implemented vs Planned)

```
project/
├── configs/
│   ├── grid_ieee123.json           # IEEE 123-bus network data
│   ├── der_placement.json          # DER locations and capacities
│   ├── zone_config.yaml            # Zone scoring weights, constraints
│   ├── training_config.yaml        # RL hyperparameters
│   └── experiment_config.yaml      # Scenario definitions
│
├── src/
│   ├── layer0_dso/
│   │   ├── reconfiguration.py      # Implemented: MISOCP formulation (Pyomo + MOSEK)
│   │   ├── dlmp_calculator.py      # Implemented: DLMP extraction from SOCP duals
│   │   ├── layer0_dso.py           # Implemented: pipeline + zone scoring/selection API
│   │   ├── zonal_pricing.py        # Implemented: aggregate DLMPs to zone prices
│   │   └── socp_validator.py       # Implemented: AC verification (fail-closed gate)
│   │
│   ├── layer1_vpp/
│   │   ├── dro_bidding.py          # Implemented: Wasserstein-style DRO via SciPy linprog
│   │   ├── scenario_generator.py   # Implemented: price scenario generation
│   │   └── virtual_battery.py      # Implemented: aggregated battery model
│   │
│   ├── layer2_control/
│   │   ├── gat_encoder.py          # Implemented: dense GAT in pure PyTorch
│   │   ├── actor_critic.py         # Implemented: actor-critic heads (PyTorch MLP)
│   │   ├── mappo_policy.py         # Implemented: custom MAPPO policy/training update
│   │   ├── safety_layer.py         # Implemented: voltage clipping + droop fallback
│   │   └── reward.py               # Implemented: reward with adaptive γ
│   │
│   ├── environment/
│   │   ├── grid_env.py             # Gym-compatible multi-agent env
│   │   ├── pandapower_backend.py   # Pandapower wrapper for step/reset
│   │   └── topology_manager.py     # Switch state + zone update management
│   │
│   └── utils/
│       ├── graph_utils.py          # Adjacency matrix ↔ PyG edge_index
│       ├── data_loader.py          # Load/PV/price profiles
│       └── metrics.py              # KPI calculations
│
├── experiments/
│   ├── train_mappo.py              # Main training (bootstrap uses ac_tol=0.01)
│   ├── eval_generalization.py      # Topology generalization (L1–L4)
│   ├── run_ablation.py             # Ablation A1–A9
│   ├── run_benchmarks.py           # Baseline comparisons
│   └── analyze_results.py          # Tables & figures generation
│
├── data/
│   ├── load_profiles/
│   ├── pv_profiles/
│   ├── price_scenarios/
│   └── topologies/                 # Pre-computed valid topologies + zone partitions
│
├── notebooks/
│   ├── 01_grid_visualization.ipynb
│   ├── 02_dlmp_analysis.ipynb
│   ├── 03_zone_sensitivity.ipynb   # NEW: zone partition analysis
│   ├── 04_training_curves.ipynb
│   └── 05_results_visualization.ipynb
│
├── tests/
│   ├── test_reconfiguration.py
│   ├── test_dro.py
│   ├── test_gat_encoder.py
│   ├── test_environment.py
│   └── test_training_smoke.py
│
└── requirements.txt
```

### 6.3 Key Integration: Dense-GAT (implemented) ↔ Custom MAPPO

> **Implemented now:** Dense adjacency GAT bằng PyTorch thuần + custom MAPPO loop (`mappo_policy.py`).
>
> **Planned extension:** chuyển encoder sang PyG (`torch_geometric`) khi cần tối ưu hóa cho graph lớn hoặc batching chuyên sâu.

```python
# gat_encoder.py — implemented dense GAT (pure PyTorch)
class GATEncoder(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gat1 = _DenseGATLayer(config.in_dim, config.hidden_dim, config.heads_l1, config.dropout)
        self.gat2 = _DenseGATLayer(config.hidden_dim * config.heads_l1, config.output_dim, 1, config.dropout)

    def forward(self, obs):
        x = to_tensor(obs.node_features)
        a = to_tensor(obs.adjacency)
        h1 = F.elu(self.gat1(x, a))
        h2 = self.gat2(h1, a)
        return h2  # [N, output_dim] node embeddings
```

```python
# mappo_policy.py — custom loop, KHÔNG dùng RLlib
class MappoPolicy(torch.nn.Module):
    def act(self, obs):
        node_features = obs['node_features']
        adjacency = obs['adjacency']
        local_state = obs['local_state']
        agent_index = obs['agent_index']

        embeddings = self.encoder.encode(GraphObservation(node_features, adjacency))
        node_embedding = embeddings[agent_index]
        actor_out = self.actor_critic.actor(node_embedding=node_embedding, local_state=local_state)
        action = sample_and_clip(actor_out)
        return action
```

```python
# grid_env.py — Pandapower backend
class GridEnv:
    def step(self, action_dict):
        for agent_id, action in action_dict.items():
            bus = self.agent_to_bus[agent_id]
            self.net.sgen.at[bus, 'p_mw'] = action[0]
            self.net.sgen.at[bus, 'q_mvar'] = action[1]

        pp.runpp(self.net, algorithm='nr')
        voltages = self.net.res_bus.vm_pu.values
        # ... compute rewards, obs, done
```

### 6.4 Implementation Roadmap

```
Phase 1: Foundation (Weeks 1–3)                           ✅ DONE
├── ✅ IEEE 123-bus setup in Pandapower
├── ✅ Switch control + topology manager
├── ✅ SOCP formulation verification (2–3 topologies)
├── ✅ Bus-82 calibration fix (max_gap ≈ 0.00482)
├── ✅ Fail-closed quality gate (ac_tol=0.01)
└── Milestone: Environment runs, power flow converges

Phase 2: Layer 0 + Layer 1 (Weeks 4–6)                   ✅ MOSTLY DONE
├── ✅ MISOCP in Pyomo + MOSEK
├── ✅ DLMP extraction
├── ✅ Zone partition scoring (ZoneScoringConfig)
├── 🔄 Dynamic zone partitioning (spectral clustering)    ← IN PROGRESS
├── 🔄 Wasserstein DRO via SciPy linprog                  ← IN PROGRESS
└── Milestone: L0+L1 end-to-end, prices reasonable

Phase 3: Layer 2 Core (Weeks 7–10)                        ⭐ NEXT FOCUS
├── ✅ Dense-GAT encoder (pure PyTorch) implemented
├── ✅ Actor-Critic heads (pure PyTorch) implemented
├── ✅ Custom MAPPO policy/update loop implemented
├── 🔄 Stabilize training on FIXED topology (sanity check)
├── 🔄 Verify voltage regulation convergence across seeds
└── Milestone: Robust and reproducible control on 1 topology

Phase 4: Integration & Dynamic Topology (Weeks 11–14)
├── Connect L0 → L2 (dynamic topology + dynamic zones)
├── Train on 5 base topologies (curriculum learning)
├── Safety layer (droop fallback)
├── Iterative feedback L2 → L1
└── Milestone: Full 3-layer pipeline end-to-end

Phase 5: Experiments (Weeks 15–18)
├── Generalization experiments (L1–L4)
├── Ablation studies (A1–A9, including zone sensitivity)
├── Benchmark comparisons (5 baselines)
├── Zone sensitivity analysis (fixed vs dynamic, granularity)
└── Milestone: All experiments complete

Phase 6: Paper Writing (Weeks 19–22)
├── Draft (reuse most from this document)
├── Results & discussion
├── Revise, feedback, submit to IEEE TSG
└── Milestone: Paper submitted
```

---

## 7. Expected Results & Honest Predictions

### What we expect to show

1. **GAT-MAPPO achieves <1% VVR** trên training topologies, competitive với centralized OPF nhưng 100–1000× faster.

2. **Generalization gap 5–15%** trên unseen topologies (Level 2), **giảm <5% với 100–500 episodes fine-tuning** — consistent với de Jong et al. (2025).

3. **GAT outperforms MLP baseline 20–40%** trên unseen topologies (A1), demonstrating GNN structural inductive bias.

4. **Scoring-selected zoning improves PoA vs fixed heuristic zoning** (A8), trong khi dynamic re-partitioning được kỳ vọng cải thiện thêm khi hoàn thiện (A10).

5. **Reconfiguration + pricing giảm total cost 5–10%** so với fixed-topology baselines (A3).

6. **Iterative feedback giảm curtailment 30–50%** so với single-pass (A7).

### Honest risks

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| MAPPO unstable with dynamic topology | Medium | Curriculum learning: fixed → gradual changes |
| SOCP gap large under high DER | Low–Medium | Fail-closed gate + penalty method |
| DRO too conservative | Medium | Tune ε via cross-validation |
| Inference >100ms | Low | Smaller GAT, quantization |
| Reward imbalance (safety vs economy) | High | Adaptive γ, constrained RL if needed |
| Spectral clustering unstable | Low | Fallback to scoring-based partitioning |

---

## 8. Paper Outline (IEEE TSG, ~12 pages)

```
I.   Introduction (1.5 pp)
     - Motivation, literature, gap: topology-adaptive control + dynamic zoning
     - Contributions C1–C4

II.  System Model (2 pp)
     - Tri-level architecture with dynamic zoning
     - Layer 0: MISOCP + zone partitioning (brief)
     - Layer 1: DRO formulation (brief)

III. GAT-MAPPO for Topology-Adaptive Voltage Control (3 pp)  ⭐
     - GAT encoder, MAPPO with CTDE
     - Reward design with adaptive γ
     - Safety layer, training with topology curriculum

IV.  Simulation Setup (1.5 pp)
     - IEEE 123-bus, scenarios, baselines

V.   Results and Discussion (3 pp)
     - Performance (Table + Figs)
     - Topology generalization (L1–L4) — key contribution
     - Ablation (A1–A9 including zone sensitivity)
     - Computational performance

VI.  Conclusion (0.5 pp)
```

---

## 9. Must-Cite References

**Direct competitors (benchmark against):**
- Xue et al. (2024), "Hierarchical safe DRL for DSO-VPP," *Applied Energy*
- Lin et al. (2025), "DSO-VPP coordination with shared energy storage," *Applied Energy*
- Sun et al. (2024), "MADRL for DSO-VPP collaborative operation," *IET GTD*

**DLMP (established — do not overclaim):**
- Bai et al. (2022), "Distribution LMP: Fundamentals and Applications," *Proc. IEEE*
- Papavasiliou (2018), "Analysis of DLMP," *IEEE Trans. Power Systems*

**GNN for power systems:**
- Donon et al. (2020), "Neural networks for power flow," *PSCC*
- Owerko et al. (2025), "PowerGNN: Topology-aware GNN," *arXiv*
- de Jong et al. (2025), "Generalizable GNN for grid topology control," *arXiv*

**MARL for power systems:**
- Wang et al. (2021), "Multi-agent RL for active voltage control," *NeurIPS*
- CommonPower (2024), "MAPPO for safe grid control"

**Reconfiguration & SOCP:**
- Jabr et al. (2012), "Minimum loss reconfiguration via MICP," *IEEE Trans. Power Systems*
- Gan, Li, Low (2014), "Exact convex relaxation of OPF," *IEEE Trans. Power Systems*

**DRO:**
- Mohajerin Esfahani & Kuhn (2018), "Data-driven DRO using Wasserstein metric," *Math. Programming*
- Li et al. (2022), "DRO vs SO for VPP scheduling"

**Graph partitioning / Zoning:**
- Karypis & Kumar (1998), "METIS graph partitioning," *SIAM J. Scientific Computing*
- Von Luxburg (2007), "A tutorial on spectral clustering," *Statistics and Computing*