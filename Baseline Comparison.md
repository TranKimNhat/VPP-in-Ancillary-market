# Baseline Comparison Plan — Section VI Design

**Paper focus:** GraphSAGE-MAPPO provides effective FFR **and** adapts to topology changes in 100% renewable islanded microgrid.

**Target journal:** IEEE Transactions on Smart Grid

---

## 1. Method Lineup (6 methods)

Each step up the ladder adds exactly one component, so reviewers can trace contribution.

| #  | Method                        | Role          | GNN          | RL algo   | Multi-agent | What it isolates                                       | Reference                                                   |
|----|-------------------------------|---------------|--------------|-----------|-------------|--------------------------------------------------------|-------------------------------------------------------------|
| 1  | **GraphSAGE-MAPPO (Ours)**    | Proposed      | GraphSAGE    | PPO       | MAPPO       | —                                                      | This work                                                   |
| 2  | MLP-MAPPO                     | Ablation      | None (MLP)   | PPO       | MAPPO       | GraphSAGE encoder contribution                         | Ablation of (1): same RL, remove graph                      |
| 3  | GCNN-PPO                      | Baseline      | Spectral GCN (2 layers) | PPO | Single centralized | GraphSAGE vs GCN (inductive vs transductive) + MA vs single | Guo et al. (2024), Front. Energy Res., doi:10.3389/fenrg.2024.1517861 |
| 4  | MATD3                         | Baseline      | None (MLP)   | TD3       | CTDE (MA)   | On-policy vs off-policy + GNN contribution             | Base algo from Li & Zhou (2025), IEEE/CAA JAS, vol. 12, no. 7 (EIE-MATD3 paper) |
| 5  | Fixed Droop                   | Rule-based    | None         | None      | None        | RL vs conventional control                              | Standard k-droop = 0.05 pu                                  |
| 6  | No FFR                        | Null baseline | None         | None      | None        | FFR is essential (not optional)                         | —                                                           |

### Comparison ladder (narrative for paper)

```
No FFR  →  Fixed Droop    : FFR is essential (H_SYS = 1.18s, ALL events exceed RoCoF limit)
Fixed Droop  →  MATD3     : RL outperforms rule-based (adaptive vs fixed gain)
MATD3  →  GCNN-PPO        : GNN encoder helps perceive grid state (GCN+PPO centralized > MLP+TD3 CTDE)
GCNN-PPO  →  MLP-MAPPO    : Multi-agent credit assignment matters (MAPPO CTDE > single-agent PPO)
MLP-MAPPO  →  GraphSAGE-MAPPO : Inductive graph learning enables topology adaptation ★

Note: ladder is not strictly single-variable (MATD3→GCNN-PPO changes both encoder and RL algo).
The ablation MLP-MAPPO→GraphSAGE-MAPPO is the clean single-variable comparison for GNN contribution.
```

---

## 2. Baseline Descriptions and Implementation

### 2.1 GCNN-PPO — Guo et al. (2024)

**Source:** "Learning-driven load frequency control for islanded microgrid using graph networks-based deep reinforcement learning," Frontiers in Energy Research, vol. 12, 2024. doi: 10.3389/fenrg.2024.1517861

**Original algorithm (as published):**
- Graph Convolution Neural Networks (spectral GCN, Kipf & Welling) + PPO (Actor-Critic)
- **Single centralized agent** controlling the entire microgrid
- GCN propagation rule (Eq. 16 in paper): `X' = σ(D̂^{-1/2} Â D̂^{-1/2} X W)`, where `Â = A + I`
- Actor network: 2 GCN layers → ReLU → summation pooling → 3 MLP layers → policy output
- Critic network: same as Actor but with **global** summation pooling behind GCN → value estimate

**Original MDP (as published, Section 3.2):**
- State (Eq. 18): `s = [Δf, ∫Δf dt, ΔP_total_G]` (frequency deviation, integral of Δf, total unit output)
- Action (Eq. 17): `a = ΔP_order` (total dispatch command, dimension ≈ number of controllable units)
- Reward (Eq. 19–20): `r = -μ2|Δf| + μ3·ΣCi + A`, where penalty `A = -10 if |Δf| ≥ 0.05 Hz, else 0`
- Multi-objective: minimizes frequency deviation AND generation cost simultaneously (Eq. 11)

**Original training setup:**
- Tested on China Southern Grid (CSG) isolated microgrid model
- Episode = 1 day of operation, time step = 3-min interval, 480 steps/episode
- Compared against: PPO, TRPO, SAC, TD3, DDPG, DDQN, DQN, DMPC, MPC, Fuzzy-FOPI, Fuzzy-PI, PSO-PI
- Simulation: 7200s (2h) for step disturbance case, full day for renewable disturbance case

**Adaptation for our environment (IEEE 123-bus islanded microgrid):**
- **Keep faithful:** GCN spectral encoder (2 layers) + summation pooling + MLP + PPO algorithm
- **Adapt to our env:** Single centralized agent with flattened action: `action_dim = N_agents × action_per_agent` (e.g., 41 × 3 = 123 if 41 agents)
- **Adapt MDP:** Use our state representation (bus features as graph node features), our reward function (frequency-based), our episode structure
- **Adjacency:** Fixed `A` matrix from grid topology (GCN requires fixed graph at train time)

**Key differences from Proposed:**
- GCN is **transductive** (spectral filters tied to specific graph Laplacian eigendecomposition) vs GraphSAGE **inductive** (learns aggregator functions that generalize to unseen graphs)
- **Single centralized** agent (flattened action space) vs **multi-agent** with per-agent credit assignment (MAPPO)
- No topology adaptation capability: changing `A` invalidates the trained spectral filters

**Implementation:** `src/baselines/gcnn_ppo.py` (existing)

**What comparison proves:**
- On seen topologies: GCNN-PPO may match or beat on mild scenarios (S1-like, as shown in report.md)
- On unseen topologies: GraphSAGE-MAPPO should show significantly less OOD degradation because GraphSAGE's inductive nature generalizes aggregation functions, while GCN's spectral filters are topology-specific
- Multi-agent advantage: MAPPO enables per-agent credit → better coordination under severe events

**Literature justification for GCN limitation on topology:**
- Hamilton et al. (2017, GraphSAGE paper): spectral approaches do not naturally generalize to inductive settings because they factorize a particular graph
- PPI benchmark in GraphSAGE paper: GraphSAGE-LSTM achieves 0.612 F1 on completely unseen graphs vs baselines < 0.42

---

### 2.2 MATD3 — Li & Zhou (2025)

**Source:** "A Robust Large-Scale Multiagent Deep Reinforcement Learning Method for Coordinated Automatic Generation Control of Integrated Energy Systems in a Performance-Based Frequency Regulation Market," IEEE/CAA Journal of Automatica Sinica, vol. 12, no. 7, pp. 1475–1488, July 2025.

**Original algorithm (EIE-MATD3, as published):**

The full paper proposes EIE-MATD3 (Efficient Integration Exploration MATD3), a complex framework consisting of:

- **CTDE architecture:** Decentralized actors (local obs → action), centralized critics (global state + all actions → Q)
- **FOPI controller tuning:** Each agent tunes an adaptive Fractional-Order PI controller. The RL agent does NOT output power directly — it outputs FOPI coefficients that then control the plant.
- **Twin Q-networks** (Q1, Q2) with clipped double-Q (Eq. 23): `y_t = r + γ·min_{i=1,2} Q_{θ'_i}(s_{t+1}, π_{ϕ}(s_{t+1}))`
- **Delayed policy updates** (actor updates every 2 critic updates, standard TD3)
- **12 parallel systems** with diverse exploration:
  - Systems 1–2: ε-greedy exploration (Eq. 16)
  - System 3: Ornstein-Uhlenbeck noise (Eq. 17)
  - System 4: Gaussian noise (Eq. 18)
  - Systems 9–12: Maximum entropy (SAC-based) exploration (Eq. 19–21)
- **Imitation learning:** Demonstrators (PSO-fuzzy-PI, GA-fuzzy-PI, TS-fuzzy-PI, Type-II fuzzy PI) generate demonstration samples into a separate experience pool (pool 2)
- **Classified experience replay:** Leaders sample from pool 1 (explorer samples) and pool 2 (demonstrator samples) with different probabilities

**Original MDP (as published, Section III-A):**
- State (Eq. 13): `s_j = [e_ACE^j, ∫e_ACE^j dt, ΔP^j]` (area control error, integral of ACE, total regulated output)
- Action (Eq. 12): `a_j = [k_p^j, k_i^j, λ^j]` (FOPI controller parameters: proportional, integral, fractional order)
- Reward (Eq. 14–15): `r_j = -(μ1|e_ACE^j| + μ2·ΣD_i^j) + A`, where penalty `A = -0.8 if |e_ACE| ≥ 7 MW, else 0`
- Objective (Eq. 8): multi-area coordinated AGC minimizing ACE + regulation mileage payment

**Original training setup:**
- 4-area China Southern Grid (CSG) real-time digital system (total installed capacity 160 GW)
- 4 agents (one per area), each tuning a FOPI controller
- Case 1: stochastic step disturbance in Area A (4800s)
- Case 2: stochastic + renewable disturbance in all 4 areas (86400s = 24h)
- Case 3: large-scale renewable in all 4 areas (86400s, 12 repeated experiments with different random seeds)
- Compared against: DMATD3-FOPI, MATD3-FOPI, MADDPG-FOPI, Fuzzy-FOPI, FOPI, PSO-Fuzzy-PI, GA-Fuzzy-PI, TS-Fuzzy-PI, Fuzzy-PI, PSO-PI, GA-PI, standalone MATD3, standalone MADDPG

**Adaptation for our environment — using base MATD3 (justified simplification):**

We implement the **core MATD3 algorithm** (the base that EIE-MATD3 builds upon), NOT the full EIE-MATD3. Justification:

1. The full EIE-MATD3 requires 12 parallel systems with 4 demonstrator controllers (PSO-fuzzy-PI, GA-fuzzy-PI, TS-fuzzy-PI, Type-II fuzzy PI) — these are specific to the CSG-AGC problem and would need to be redesigned entirely for our FFR problem. Implementing them would introduce a confounding "how well did we re-design the demonstrators" variable.
2. The paper itself compares EIE-MATD3 against standalone MATD3 (Table II–IV), treating MATD3 as a meaningful baseline. By using MATD3, we compare against the same algorithmic family.
3. Our goal is to isolate: **off-policy deterministic multi-agent (MATD3) vs on-policy stochastic multi-agent with GNN (Proposed)**. The EIE enhancements (imitation learning, multi-exploration) are orthogonal to this axis and could in principle be added to any base algorithm.

**Adapted MATD3 for our env:**
- **Keep faithful:** CTDE framework, twin Q-networks, clipped double-Q, delayed policy updates, deterministic actors with exploration noise
- **Adapt:** Direct power output (not FOPI tuning), use our state/action/reward (see Section 2.6)
- **Per-agent architecture (standard MATD3 hyperparameters):**
  - Actor: MLP(obs_dim → 256 → 256 → action_dim), deterministic, tanh output
  - Critic × 2: MLP(global_obs_dim + all_actions_dim → 256 → 256 → 1)
  - Target networks: soft update τ = 0.005
- **Training (standard TD3 hyperparameters from Fujimoto et al. 2018):**
  - Replay buffer: 100k transitions
  - Batch size: 256
  - Delayed policy update: every 2 critic updates
  - Exploration: Gaussian noise σ = 0.3 → 0.05 (linear decay over training)
  - Target policy smoothing: clip(N(0, 0.2), -0.5, 0.5)

> **Note for paper text:** We cite Li & Zhou (2025) as the source of MATD3 for frequency regulation, but clarify that we implement the base MATD3 algorithm without the EIE enhancements (imitation learning, parallel exploration systems). This is standard practice — e.g., comparing against MADDPG rather than the full QMIX+MADDPG variant.

**What comparison proves:**
- MATD3 is a strong multi-agent baseline (same CTDE paradigm as MAPPO)
- Isolates: off-policy deterministic MLP (MATD3) vs on-policy stochastic GraphSAGE (Proposed)
- If Proposed beats MATD3 on unseen topology: **graph-aware on-policy > MLP off-policy for topology adaptation**
- If Proposed beats MATD3 on severe events: **stochastic exploration (PPO) + cooperative GAE > deterministic policy + replay** for frequency emergencies

**Literature justification for MATD3 as baseline:**
- IEEE/CAA JAS is top-tier (IF ~15.3)
- MATD3 represents state-of-the-art in multi-agent frequency regulation (2025)
- Same CTDE paradigm makes comparison fair on the multi-agent axis
- The original paper shows standalone MATD3 is already competitive (Table II: MATD3 |Δf|_avg = 0.0147 Hz vs EIE-MATD3-FOPI = 0.0054 Hz in Area A) — a meaningful baseline
- Adding GNN to TD3 variants has not been explored for islanded microgrids → our work fills this gap

---

### 2.3 MLP-MAPPO (Ablation)

**Source:** Ablation of our own method — removes GraphSAGE encoder, replaces with MLP of equivalent parameter count.

**Algorithm:**
- Identical to GraphSAGE-MAPPO except:
- 
  - Replace `FeederGraphSAGEAgentEncoder` → `MLPAgentEncoder`
  - MLP encoder: `Linear(obs_feat, hidden) → ReLU → Linear(hidden, embed_dim)`
  - Agent embeddings from concatenated local features (no message passing)
  - edge_index is **ignored** during forward pass

**Implementation plan:** `src/baselines/mlp_mappo.py` (new, ~100 LOC, mostly wrapper)

```python
class MLPAgentEncoder(nn.Module):
    """MLP encoder that ignores graph structure."""
    def __init__(self, obs_feat, hidden_dim=64, embed_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_feat, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim),
        )
    
    def forward(self, x_full, edge_index, agent_bus_idx):
        # edge_index is IGNORED — no message passing
        node_embeds = self.net(x_full)
        return node_embeds[agent_bus_idx]
```

**What comparison proves:**
- **The** most important ablation: isolates GraphSAGE contribution with everything else held constant
- Same RL algorithm (PPO), same multi-agent framework (MAPPO), same critic, same reward, same training schedule
- Only difference: graph message passing vs no message passing
- Expected result on topology generalization:
  - Train topologies: MLP-MAPPO may be competitive (memorizes training distribution)
  - Unseen topologies: MLP-MAPPO degrades significantly → proves GraphSAGE's inductive generalization
  - IAE degradation vs d_E plot (Fig. 6): MLP-MAPPO shows steep curve, GraphSAGE-MAPPO shows flat curve

**Literature justification for MLP ablation:**
- Standard in GNN-RL papers: Graph-PPO vs IPPO vs MAPPO (c Chen et al. 2022, PowerNet)
- GraphSAGE-D3QN vs D3QN (MLP baseline) for undervoltage load shedding (notebook "AI & RL")
- Key principle: "swap encoder, keep RL identical" to isolate graph contribution

---

### 2.4 Fixed Droop (Rule-based)

**Standard proportional droop controller:**

```
ΔP = -k_droop × Δf / f_deadband
k_droop = 0.05 pu (5% droop, standard IEEE/ENTSO-E)
f_deadband = ±0.02 Hz (20 mHz, ENTSO-E FCR deadband)
```

**What comparison proves:**
- All RL methods should beat Fixed Droop on severe events (S2, S4)
- Fixed Droop may partially handle moderate events (S1)
- On line_trip (S3): Fixed Droop cannot adapt to topology change (fixed gain regardless of network state)

---

### 2.5 No FFR (Null baseline)

**Zero injection: all agents output zero action.**

**What comparison proves:**
- With H_SYS = 1.18s, even 1.5 MW event exceeds IEEE 1547 Cat III RoCoF limit (2.0 Hz/s)
- **ALL** scenarios violate frequency limits without FFR
- This proves FFR from VPP agents is **essential**, not optional, in 100% IBR islanded system
- Establishes performance floor for all other methods

---

### 2.6 Fair Comparison Protocol — Unified MDP and Training

The two baseline papers operate on **different systems** (CSG multi-area grid vs our IEEE 123-bus islanded microgrid) with **different MDP formulations**. To ensure fair comparison, we follow the standard practice in RL comparison papers: **keep the algorithm architecture faithful, adapt the MDP interface to our environment, train all methods under identical conditions.**

#### 2.6.1 What we keep from each paper (algorithm fidelity)

| Component | GCNN-PPO (Guo et al.) | MATD3 (Li & Zhou) |
|-----------|----------------------|-------------------|
| Encoder | Spectral GCN, 2 layers, summation pooling | MLP (no graph encoder) |
| RL algorithm | PPO (Actor-Critic, clipped objective) | TD3 (twin Q, delayed policy update) |
| Agent structure | Single centralized agent | Multi-agent CTDE |
| Policy type | Stochastic (Gaussian) | Deterministic + noise |
| Training paradigm | On-policy (rollout buffer) | Off-policy (replay buffer) |

#### 2.6.2 What we unify across ALL methods (fair comparison axes)

**Unified environment:**
- All methods train and evaluate on the **same** IEEE 123-bus islanded microgrid environment
- Same S_BASE = 15.705 MW, same H_SYS = 1.18 s, same GFM units (G1–G6)
- Same topology cache (20 configs, 15 train / 5 test via farthest-point split)

**Unified MDP interface:**
- **Observation:** All methods receive the same per-bus feature vector (voltage, frequency deviation, power injection, etc.) as graph node features. For GCN-based methods: fed through graph convolution with adjacency `A`. For MLP-based methods: concatenated/flattened.
- **Action:** All methods output the same VPP power injection ΔP per controllable agent. For centralized methods (GCNN-PPO): flattened `[ΔP_1, ..., ΔP_N]`. For CTDE methods (MATD3, MAPPO): per-agent ΔP.
- **Reward:** All methods use the **same** reward function (our FFR reward with frequency deviation penalty, RoCoF penalty, etc.) — NOT the original reward functions from the respective papers. This ensures we compare algorithms, not reward engineering.

**Unified training budget:**
- Same number of environment interactions (total timesteps)
- Same number of episodes with same scenario distribution
- Same random seeds for reproducibility (N=20 runs per configuration)

**Unified evaluation:**
- Same 4 scenarios (S1–S4) × same topology splits (15 train / 5 test)
- Same FFR metrics (IAE_post, RoCoF_max, nadir, settling_time, FFR_SR)
- Per-scenario reporting (no averaging across scenarios of different severity)

#### 2.6.3 What differs (the variables we compare)

The ONLY differences between methods are the algorithm components listed in Section 2.6.1. This ensures that performance gaps are attributable to the algorithm design, not to differences in environment, reward, or training budget.

#### 2.6.4 MDP adaptation rationale

| Original paper | Original MDP | Our adaptation | Why |
|---------------|-------------|---------------|-----|
| Guo et al. — state = [Δf, ∫Δf dt, ΔP_G] | AGC-oriented, 3D | Per-bus features as graph nodes | Our env has 41 agents on a distribution network; the GCN needs node features to operate |
| Guo et al. — action = ΔP_order (scalar dispatch) | Single command | Flattened [ΔP_1,...,ΔP_N] | Direct VPP injection control, not AGC dispatch |
| Guo et al. — reward = -μ2\|Δf\| + μ3·ΣCi | Freq + cost | Our FFR reward | No generation cost in VPP FFR context |
| Li & Zhou — state = [e_ACE, ∫e_ACE dt, ΔP_j] | AGC/ACE-oriented | Per-bus features (local obs for CTDE) | ACE is an inter-area concept; our islanded system uses local frequency |
| Li & Zhou — action = [k_p, k_i, λ] (FOPI tuning) | Controller tuning | Direct ΔP per agent | No FOPI layer in our env; direct power injection |
| Li & Zhou — reward = -(μ1\|e_ACE\| + μ2·ΣD_i) | ACE + payment | Our FFR reward | No regulation market in islanded microgrid |

> **Key principle:** We compare **learning algorithms** (GCN+PPO vs MLP+TD3 vs GraphSAGE+MAPPO), not system models or reward formulations. All adaptations are made to ensure the comparison is about algorithmic capability, not about who has the better reward function or environment interface.

---

## 3. Evaluation Scenarios (4 levels)

### System parameters

| Parameter    | Value     | Source                                         |
|-------------|-----------|------------------------------------------------|
| S_BASE      | 15.705 MW | IEEE 123-bus modified                          |
| H_SYS       | 1.18 s    | Aggregated from 5 GFM units (G1-G6)           |
| D_SYS       | 0.73 pu   | Load damping coefficient                       |
| R_SYS       | 0.048 pu  | Aggregated droop coefficient                   |
| f_0         | 50.0 Hz   | Nominal frequency                              |
| Max dP for RoCoF ≤ 2.0 Hz/s | 1.49 MW (9.5% S_BASE) | Analytical from swing equation |

### Scenario design

| Scenario | Event      | ΔP (MW) | % S_BASE | RoCoF (Hz/s) | Nadir est. (Hz) | Purpose                                     | Literature ref                               |
|----------|-----------|---------|----------|--------------|-----------------|----------------------------------------------|----------------------------------------------|
| **S1**   | load_step | +2.5    | 16%      | -3.36        | 49.41            | Moderate N-1; droop can partially handle     | Bitew et al. (2025) - load surge             |
| **S2**   | gen_trip  | -3.9    | 25%      | +5.24        | 50.92            | **Sweet spot:** droop fails, RL must act     | Seneviratne et al. (2016, 131 cit)           |
| **S3**   | line_trip | -2.4    | 15%      | +3.23        | 50.57            | **KEY:** edge_index changes mid-episode      | Farooq et al. (2022) - transmission contingency |
| **S4**   | gen_trip  | -5.5    | 35%      | +7.39        | 51.30            | Extreme N-1; graceful degradation test       | Knap et al. (2016, 368 cit)                  |

### Why these 4 (not 5 as in old comparison_runner)

Old S1 mild (load_step 0.8 MW, RoCoF ~1.07 Hz/s) was **below IEEE 1547 Cat III limit**. System self-recovers, all methods perform similarly → zero differentiation. This caused GCNN to beat Proposed on overall IAE mean (report.md finding). Removing it and reporting per-scenario eliminates this artifact.

### Contingency magnitudes literature basis

| Event type | Magnitudes (MW)      | % S_BASE | Literature reference                                                  |
|------------|---------------------|----------|-----------------------------------------------------------------------|
| load_step  | 1.6, 2.5, 3.9      | 10-25%   | Bitew et al. (2025) - load surge testing in 500kW MG                  |
| gen_trip   | -2.4, -3.9, -5.5   | 15-35%   | Seneviratne et al. (2016, 131 cit); Knap et al. (2016, 368 cit)      |
| line_trip  | -2.4                | ~15%     | Farooq et al. (2022) - transmission contingency                       |
| high_ren   | 3.1, 4.7, 6.3      | 20-40%   | Kerdphol et al. (2019, 186 cit) - RES surge                           |

---

## 4. Section VI Structure

Following IEEE TSG pattern from NotebookLM survey (Xia et al., Yan et al., Liu et al., Chen et al.):

| Subsection            | Content                                                                                                      | Outputs                              |
|-----------------------|--------------------------------------------------------------------------------------------------------------|--------------------------------------|
| **VI-A. Setup**       | IEEE 123-bus, 41 agents, 5 GFM units, topology cache (20 configs, 15 train / 5 test), baseline descriptions | Table I (system params), Table II (method comparison) |
| **VI-B. FFR comparison** | FFR metrics per-scenario × per-method. 4 scenarios × 6 methods × 20 runs. Primary: IAE_post, RoCoF, nadir, settling time, ITAE, ffr_success_rate | Table III, Fig. 4 (f-traces), Fig. 5 (bar chart) |
| **VI-C. Topology adaptation** | Train vs unseen topology FFR. Farthest-point split. Per-topology metrics. IAE degradation vs edge distance d_E | Table IV, Table V (split quality), **Fig. 6 ★ KEY** |
| **VI-D. GNN ablation** | GraphSAGE-MAPPO vs MLP-MAPPO vs GCNN-PPO (same RL paradigm, different encoder). Train IAE gap, test IAE gap → graph contribution % | Table VI, Fig. 7 (training curves) |
| **VI-E. Severity scaling** | FFR success rate vs contingency magnitude (1.6 → 2.5 → 3.9 → 5.5 MW). Graceful degradation curve per method | Table VII, Fig. 8 |
| **VI-F. Computation** | Inference time (ms/step), parameter count, training wall-clock (h). Real-time feasibility ≤ 50ms | Table VIII |

### Total: 8 tables + 5 figures (within IEEE TSG standard range of 4-8 tables + 4-7 figures)

---

## 5. FFR Metrics (aligned with IEEE 1547 / ENTSO-E)

| Metric              | Symbol          | Definition                                             | Unit   | Source standard              |
|---------------------|-----------------|-------------------------------------------------------|--------|------------------------------|
| Frequency nadir     | f_nadir         | min(f(t)) during transient                            | Hz     | IEEE 1547, ENTSO-E           |
| Frequency zenith    | f_zenith        | max(f(t)) during transient                            | Hz     | IEEE 1547                    |
| Max RoCoF           | RoCoF_max       | max(|df/dt|) at t=0+                                  | Hz/s   | IEEE 1547 Cat III: ≤ 2.0 Hz/s |
| Max frequency deviation | Δf_max      | max(|f(t) - f_0|)                                     | Hz     | ENTSO-E: ≤ 0.8 Hz           |
| IAE (post-event)    | IAE_post        | ∫|Δf(t)|dt from t_event to t_event + T_window         | Hz·s   | Bevrani et al. (2016)        |
| ITAE                | ITAE            | ∫ t·|Δf(t)|dt (penalizes slow recovery)               | Hz·s²  | Standard control theory       |
| Settling time       | t_settle        | Time until |Δf(t)| ≤ 2% of Δf_ss permanently          | s      | Standard control theory       |
| FFR success rate    | FFR_SR          | Fraction of episodes where nadir ≥ 49.5 AND RoCoF ≤ 2.0 | %   | IEEE 1547 Cat III             |
| Time in violation   | t_viol          | Total time |Δf| > 0.5 Hz                              | s      | ENTSO-E FCR activation range  |

### Metric hierarchy for claims

- **Primary** (headline results): IAE_post, FFR_SR, nadir — these directly prove FFR effectiveness
- **Secondary** (supporting): RoCoF_max, settling_time, ITAE — these characterize response quality
- **Diagnostic** (appendix only): THD (currently invalid_rate = 1.0, do not claim)

---

## 6. Topology Evaluation Protocol

### 6.1 Farthest-point split (replace random split)

Current split has d_min ≈ 0.008 → test topologies nearly identical to train. Use farthest-point selection:

```
1. Compute pairwise Jaccard edge distance: d_E(G_a, G_b) = 1 - |E_a ∩ E_b| / |E_a ∪ E_b|
2. Start with all 20 topologies in train set
3. Iteratively move the topology with largest d_min to test set
4. Repeat until |test| = 5
5. Report: mean d_min, min d_min, max d_min
```

Target: mean d_min ≥ 0.02 (2.5× current distance)

### 6.2 Per-topology evaluation

For each method × each topology (train + test) × each scenario:
- Run N=10 episodes with different random seeds
- Compute mean ± std of all FFR metrics
- Track: ffr_success_rate, iae_post, nadir, rocof_max

### 6.3 Fig. 6 — The key figure ★

**X-axis:** Jaccard edge distance d_E (from test topology to nearest train topology)

**Y-axis:** IAE degradation (%) = (IAE_test - IAE_train_mean) / IAE_train_mean × 100

**Plot:** Scatter + regression line for each method

**Expected result:**
- GraphSAGE-MAPPO: **flat line** (low degradation regardless of distance) → inductive generalization
- MLP-MAPPO: **steep positive slope** (degradation increases with distance) → no topology awareness
- GCNN-PPO: **moderate slope** (GCN has some graph awareness but spectral filters are topology-specific)
- MATD3: **steep slope** (MLP + off-policy → worst generalization)

This single figure is the strongest visual evidence for the topology adaptation claim.

---

## 7. Claim-Safe Wording (for paper text)

### Claim 1 — FFR Effectiveness
> "The proposed GraphSAGE-MAPPO achieves [X]% higher FFR success rate and [Y]% lower post-event IAE compared to the state-of-the-art MATD3 and GCNN-PPO baselines under severe contingencies (25-35% of S_BASE) in a 100% inverter-based islanded microgrid with H_SYS = 1.18 s."

### Claim 2 — Topology Adaptation
> "On held-out feeder reconfigurations not seen during training, the proposed method retains [Z]% of its FFR success rate (retention ratio), while the MLP-MAPPO ablation degrades by [W]%. The inductive nature of GraphSAGE—learning aggregator functions rather than fixed spectral filters—enables topology-invariant frequency control without retraining."

### Claim 3 — GNN Contribution (from ablation)
> "Replacing the GraphSAGE encoder with an equivalent-parameter MLP increases the topology generalization gap from [A]% to [B]% (Table VI), confirming that graph message-passing is necessary for adapting to distribution network reconfiguration."

### What NOT to claim
- Do NOT claim "arbitrary topology transfer" — only local feeder reconfiguration within IEEE 123-bus
- Do NOT claim overall IAE superiority if not per-scenario stratified
- Do NOT claim RoCoF superiority unless reward weight is tuned (current rocof_weight = 0.15)
- Do NOT claim THD performance (harmonic_invalid_rate = 1.0)

---

## 8. Implementation Priority and Training Budget

| Task                             | Code effort | Training time | Priority | Notes |
|----------------------------------|-------------|---------------|----------|-------|
| MLP-MAPPO ablation               | ~100 LOC    | ~8.5h (same as proposed) | **P0** — most important for claim | |
| MATD3 baseline (base MATD3, no EIE) | ~300 LOC | ~6h (off-policy, faster) | **P1** — new IEEE baseline | Standard TD3 hyperparameters (Fujimoto 2018), NOT EIE-MATD3's 12 parallel systems |
| GCNN-PPO verify/update           | ~50 LOC     | retrain if MDP changed | **P1** — existing code may need MDP alignment | Ensure GCN architecture matches Guo et al.: 2 GCN layers + summation pooling + 3 MLP layers |
| Farthest-split implementation    | ~80 LOC     | 0 (eval only)  | **P0** — required for Fig. 6 | |
| ITAE metric in eval              | ~10 LOC     | 0 (eval only)  | **P0** — trivial addition | |
| Unified reward function          | ~20 LOC     | 0 (config)     | **P0** — required for fairness | All methods MUST use identical reward; document in paper Section VI-A |
| RoCoF reward weight sweep        | 0           | ~3 × 8.5h      | **P3** — optional improvement | |

**Total new training:** ~23h (MLP-MAPPO 8.5h + MATD3 6h + proposed retrain if needed 8.5h)

---

## 9. File Structure

```
src/eval/
├── eval_ffr_topology.py     ← REWRITE: main eval runner for Section VI
│   ├── FFRMetrics            (dataclass: nadir, RoCoF, IAE_post, ITAE, settling, ffr_success)
│   ├── compute_ffr_metrics() (from f_trace + IEEE 1547 gates)
│   ├── compute_farthest_split() (maximize d_min for topology split)
│   ├── FFRTopologyEvaluator  (main class)
│   │   ├── build_table_ffr_comparison()   → Table III
│   │   ├── build_table_topology()         → Table IV
│   │   ├── build_table_split_quality()    → Table V
│   │   ├── build_table_ablation()         → Table VI
│   │   ├── build_table_severity()         → Table VII
│   │   ├── build_table_computation()      → Table VIII
│   │   ├── plot_freq_traces()             → Fig. 4
│   │   ├── plot_iae_bars()                → Fig. 5
│   │   ├── plot_degradation_vs_distance() → Fig. 6 ★ KEY
│   │   ├── plot_training_curves()         → Fig. 7
│   │   ├── plot_severity_curve()          → Fig. 8
│   │   └── run_all()                      → orchestrate everything
│   └── MLPMAPPOPolicy       (MLP ablation wrapper)
├── comparison_runner.py     ← KEEP (backward-compatible, legacy)
├── evaluate_dual.py         ← KEEP (DeterministicDualPolicy, etc.)
├── figures.py               ← KEEP
└── harmonic_analysis.py     ← KEEP

src/baselines/
├── gcnn_ppo.py              ← EXISTING (Guo et al. 2024)
├── sgac.py                  ← EXISTING (Wu et al. 2025)
├── matd3.py                 ← NEW (~300 LOC, Li & Zhou 2025)
├── mlp_mappo.py             ← NEW (~100 LOC, ablation)
└── graph_ppo.py             ← EXISTING (keep for backward compat)
```
