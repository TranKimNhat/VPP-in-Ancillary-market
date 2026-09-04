# VPP Operation Flowchart for Islanded Microgrid with Ancillary Services

## System Context
- **Grid Type**: 100% Renewable Islanded Microgrid (IEEE 123-bus)
- **Market**: Ancillary Market (FFR) + Energy Market (P2P)
- **Assets**: 3 VPPs with PV, Wind, BESS, EVCS
- **Control**: Dual-timescale MAPPO (Fast 1s + Slow 15min)

---

```mermaid
graph TD
    Start((Start of<br/>Operating Day)) --> L0

    subgraph L0 [LAYER 0: DSO - DISTRIBUTION SYSTEM OPERATOR]
        L0_Input[Inputs:<br/>- Load forecast (24h)<br/>- RES generation forecast<br/>- Network topology state<br/>- Voltage/thermal limits]
        
        L0_Reconfig[Tie-Switch Reconfiguration<br/>MISOCP Optimization:<br/>- Minimize losses<br/>- Ensure radiality<br/>- Voltage feasibility]
        
        L0_Zonal[Zonal Pricing Calculation:<br/>- DLMP per bus<br/>- Zone price aggregation<br/>- Congestion signals]
        
        L0_Publish[Publish to VPPs:<br/>- Zone prices (96 intervals)<br/>- Topology schedule<br/>- P_ref per zone]
        
        L0_Input --> L0_Reconfig
        L0_Reconfig --> L0_Zonal
        L0_Zonal --> L0_Publish
    end

    L0_Publish --> L1

    subgraph L1 [LAYER 1: VPP - MARKET PARTICIPATION]
        L1_Input[Inputs per VPP:<br/>- Zone prices from L0<br/>- Asset capacities (PV/Wind/BESS/EVCS)<br/>- SOC states<br/>- Uncertainty sets (DRO)]
        
        L1_DRO[DRO Bidding Optimization:<br/>- Worst-case expected profit<br/>- Reserve capacity allocation<br/>- Energy schedule]
        
        L1_Split[Market Bid Splitting:<br/>- AM: FFR reserve capacity (MW)<br/>- EM: Energy schedule (MWh)]
        
        L1_Submit[Submit to Markets:<br/>- FFR reserve bids → AM<br/>- Energy bids → EM (P2P)]
        
        L1_Clear{Market Clearing}
        
        L1_Award[Receive Awards:<br/>- Cleared FFR capacity<br/>- Cleared energy schedule<br/>- Settlement prices]
        
        L1_Input --> L1_DRO
        L1_DRO --> L1_Split
        L1_Split --> L1_Submit
        L1_Submit --> L1_Clear
        L1_Clear --> L1_Award
    end

    L1_Award --> L2_Slow

    subgraph L2 [LAYER 2: REAL-TIME CONTROL - DUAL MAPPO]
        
        subgraph L2_Slow [SLOW LOOP - 15 min intervals]
            S_Obs[Observation:<br/>- Current SOC states<br/>- RES actual generation<br/>- Load actual<br/>- Market prices]
            
            S_Policy[Policy Slow π_slow:<br/>- P/Q setpoints per agent<br/>- Energy tracking vs schedule]
            
            S_Dispatch[Dispatch to Assets:<br/>- BESS charge/discharge<br/>- EVCS power allocation<br/>- PV/Wind curtailment]
            
            S_Track{Tracking Error<br/>vs DA Schedule?}
            
            S_Penalty[Imbalance Penalty<br/>Update reward]
            
            S_Obs --> S_Policy
            S_Policy --> S_Dispatch
            S_Dispatch --> S_Track
            S_Track -- High --> S_Penalty
            S_Penalty --> S_Obs
        end

        S_Track -- Low --> F_Monitor

        subgraph L2_Fast [FAST LOOP - 1 sec intervals]
            F_Monitor[Frequency Monitor:<br/>- Δf measurement<br/>- RoCoF calculation<br/>- H_sys (topology-aware)]
            
            F_Trigger{FFR Activation?<br/>|Δf| > 0.1 Hz or<br/>|RoCoF| > 0.5 Hz/s}
            
            F_Policy[Policy Fast π_fast:<br/>- k_droop per VPP (3)<br/>- P setpoints (41 agents)]
            
            F_FFR[FFR Response:<br/>ΔP = -k_droop × Δf<br/>Distribute to GFM assets]
            
            F_Execute[Execute on Assets:<br/>- BESS fast response<br/>- VSG/GFM injection<br/>- Update frequency state]
            
            F_Check{Frequency<br/>Stabilized?<br/>|Δf| < 0.5 Hz}
            
            F_Escalate[Escalate:<br/>- Increase k_droop<br/>- Load shedding signal<br/>- Alert to DSO]
            
            F_Monitor --> F_Trigger
            F_Trigger -- No --> F_Monitor
            F_Trigger -- Yes --> F_Policy
            F_Policy --> F_FFR
            F_FFR --> F_Execute
            F_Execute --> F_Check
            F_Check -- No --> F_Escalate
            F_Escalate --> F_Policy
            F_Check -- Yes --> F_Monitor
        end
    end

    F_Monitor --> Settlement

    subgraph Settlement [SETTLEMENT & LEARNING]
        Settle_Calc[Calculate Settlement:<br/>- FFR delivery vs commitment<br/>- Energy delivery vs schedule<br/>- Imbalance penalties]
        
        Settle_Revenue[Revenue Calculation:<br/>- AM revenue = FFR_delivered × λ_FFR<br/>- EM revenue = Energy × zone_price<br/>- Penalties subtracted]
        
        Settle_Profit[Profit = Revenue - OPEX<br/>- PV: 12 €/kW/year<br/>- Wind: 30 €/kW/year<br/>- BESS: 7.5 €/kWh/year]
        
        Settle_Update[Update RL Policy:<br/>- Reward = Profit + Voltage + Tracking<br/>- PPO update for both loops]
        
        Settle_Calc --> Settle_Revenue
        Settle_Revenue --> Settle_Profit
        Settle_Profit --> Settle_Update
    end

    Settle_Update --> End((End of<br/>Operating Day))

    %% Cross-layer feedback
    F_Escalate -.-> L0_Reconfig
    S_Penalty -.-> L1_DRO
```

---

## Key Design Decisions

### 1. FFR Activation Trigger (NEW)
```
IF |Δf| > 0.1 Hz OR |RoCoF| > 0.5 Hz/s:
    Activate FFR response
    k_droop_effective = policy_fast(obs)
ELSE:
    Normal operation (slow loop only)
```

### 2. Topology-Aware Frequency Dynamics
```
H_sys = Σ(H_i × S_i × w_i) / S_BASE
where w_i = 1 if GFM_i connected, else 0

RoCoF = -f0 × ΔP / (2 × H_sys)
```

### 3. VPP-Level FFR (3 droops instead of 18)
```
VPP_1: agents [9-11, 18-20] → k_droop_1
VPP_2: agents [12-14, 21-23] → k_droop_2  
VPP_3: agents [15-17, 24-26] → k_droop_3
```

### 4. Dual-Timescale Coordination
```
Slow Loop (15 min):
  - Market tracking
  - SOC management
  - Energy optimization

Fast Loop (1 sec):
  - FFR response
  - Frequency stabilization
  - Droop control
```

---

## Improvements Over Reference Flowchart

| Aspect | Reference (Grid-Connected) | Ours (Islanded) |
|--------|---------------------------|-----------------|
| Frequency | Infinite bus (stable) | GFM/VSG (dynamic H_sys) |
| Market | KPX centralized | Zonal P2P + AM |
| Control | MPC | MAPPO (adaptive) |
| Topology | Fixed | Reconfigurable (tie-switches) |
| FFR | Not mentioned | Core feature |

---

## Implementation Status

| Component | File | Status |
|-----------|------|--------|
| L0 Reconfiguration | `src/layer0_dso/reconfiguration.py` | ✅ |
| L0 Zonal Pricing | `src/layer0_dso/zonal_pricing.py` | ✅ |
| L1 DRO Bidding | `src/layer1_vpp/dro_bidding.py` | ✅ |
| L2 Slow Policy | `src/rl/train_dual.py` (policy_slow) | ✅ |
| L2 Fast Policy | `src/rl/train_dual.py` (policy_fast) | ✅ |
| Frequency Dynamics | `src/env/freq_dynamics.py` | ✅ |
| Topology-aware H_sys | `freq_dynamics.update_topology()` | ✅ |
| VPP-level FFR | `microgrid_env_dual.py` (k_droop_vpp) | ✅ |
| Settlement | `src/eval/evaluate_dual.py` | ✅ |

---

## Future Enhancements (Optional)

1. **Inter-VPP Coordination**: Share FFR burden based on SOC
2. **Adaptive Trigger Thresholds**: Learn optimal Δf trigger
3. **Online Model Update**: Update RES forecast with actual data
4. **Load Shedding Integration**: Last-resort frequency protection
