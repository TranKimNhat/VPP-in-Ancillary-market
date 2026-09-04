# Tri-Layer VPP Operation Framework for Islanded Microgrid with Ancillary Services

## Flowchart for Paper

```mermaid
flowchart TD
    subgraph Layer0 ["<b>Layer 0: Distribution System Operator (Day-Ahead)</b>"]
        L0_Start([Day-Ahead Planning]) --> L0_Forecast
        L0_Forecast[/"Load & RES Forecast<br/>(24-hour horizon)"/]
        L0_Forecast --> L0_Reconfig
        L0_Reconfig["Optimal Network Reconfiguration<br/>(MISOCP)"]
        L0_Reconfig --> L0_Price
        L0_Price["Zonal Pricing Calculation<br/>(DLMP-based)"]
        L0_Price --> L0_Publish
        L0_Publish[/"Publish Zone Prices<br/>& Topology Schedule"/]
    end

    subgraph Layer1 ["<b>Layer 1: Virtual Power Plant Aggregator (Day-Ahead)</b>"]
        L1_Receive[/"Receive Zone Prices<br/>& Asset Forecasts"/]
        L1_Receive --> L1_DRO
        L1_DRO["Distributionally Robust<br/>Bidding Optimization"]
        L1_DRO --> L1_AM
        L1_DRO --> L1_EM
        L1_AM["Ancillary Market:<br/>FFR Reserve Capacity Bid"]
        L1_EM["Energy Market:<br/>P2P Energy Schedule"]
        L1_AM --> L1_Submit
        L1_EM --> L1_Submit
        L1_Submit[/"Submit Bids<br/>to Market Operator"/]
        L1_Submit --> L1_Clear
        L1_Clear{"Market<br/>Clearing"}
        L1_Clear --> L1_Award
        L1_Award[/"Receive Cleared<br/>Schedule & Prices"/]
    end

    subgraph Layer2 ["<b>Layer 2: Real-Time Distributed Control (MAPPO)</b>"]
        
        subgraph SlowLoop ["Slow Control Loop (15-minute intervals)"]
            S_Obs[/"Observe:<br/>SOC, Generation, Load"/]
            S_Obs --> S_Policy
            S_Policy["Slow Policy π<sub>slow</sub>:<br/>Energy Dispatch Optimization"]
            S_Policy --> S_Dispatch
            S_Dispatch["Dispatch Setpoints<br/>to DER Assets"]
            S_Dispatch --> S_Track
            S_Track{"Schedule<br/>Tracking?"}
            S_Track -->|Deviation| S_Adjust
            S_Adjust["Adjust Dispatch<br/>& Update Reward"]
            S_Adjust --> S_Obs
        end

        subgraph FastLoop ["Fast Control Loop (1-second intervals)"]
            F_Monitor[/"Monitor:<br/>Frequency Deviation Δf<br/>Rate of Change (RoCoF)"/]
            F_Monitor --> F_Trigger
            F_Trigger{"FFR<br/>Activation?<br/>|Δf| > ε<sub>f</sub>"}
            F_Trigger -->|No| F_Monitor
            F_Trigger -->|Yes| F_Policy
            F_Policy["Fast Policy π<sub>fast</sub>:<br/>VPP-Level Droop Control"]
            F_Policy --> F_Response
            F_Response["FFR Power Injection<br/>ΔP = -k<sub>droop</sub> × Δf"]
            F_Response --> F_Execute
            F_Execute["Execute on GFM Assets:<br/>BESS, VSG Inverters"]
            F_Execute --> F_Stable
            F_Stable{"Frequency<br/>Stabilized?"}
            F_Stable -->|No| F_Escalate
            F_Escalate["Escalate Response:<br/>Increase Droop Gain"]
            F_Escalate --> F_Policy
            F_Stable -->|Yes| F_Monitor
        end

        S_Track -->|Compliant| F_Monitor
    end

    subgraph Settlement ["<b>Settlement & Learning</b>"]
        Set_Calc["Calculate Delivered Services:<br/>• FFR Energy Delivered<br/>• Schedule Compliance"]
        Set_Calc --> Set_Revenue
        Set_Revenue["Revenue Calculation:<br/>R = R<sub>AM</sub> + R<sub>EM</sub> - Penalties"]
        Set_Revenue --> Set_Profit
        Set_Profit["Profit = Revenue - OPEX"]
        Set_Profit --> Set_Update
        Set_Update["Policy Update:<br/>PPO Gradient Descent"]
    end

    L0_Publish --> L1_Receive
    L1_Award --> S_Obs
    F_Monitor --> Set_Calc
    Set_Update --> L0_Start

    style Layer0 fill:#e6f3ff,stroke:#0066cc
    style Layer1 fill:#fff0e6,stroke:#cc6600
    style Layer2 fill:#e6ffe6,stroke:#006600
    style Settlement fill:#f5f5f5,stroke:#666666
    style SlowLoop fill:#d4edda,stroke:#28a745
    style FastLoop fill:#fff3cd,stroke:#ffc107
```

---

## Simplified Version (Single-Page)

```mermaid
flowchart LR
    subgraph DA ["Day-Ahead Stage"]
        direction TB
        DSO["DSO:<br/>Topology &<br/>Zonal Pricing"]
        VPP["VPP:<br/>DRO Bidding<br/>(AM + EM)"]
        DSO --> VPP
    end

    subgraph RT ["Real-Time Stage"]
        direction TB
        Slow["Slow Loop<br/>(15 min):<br/>Energy Tracking"]
        Fast["Fast Loop<br/>(1 sec):<br/>FFR Response"]
        Slow --> Fast
    end

    subgraph OUT ["Outcomes"]
        direction TB
        Freq["Frequency<br/>Stability"]
        Rev["Market<br/>Revenue"]
    end

    DA --> RT
    RT --> OUT

    style DA fill:#e6f3ff,stroke:#0066cc
    style RT fill:#e6ffe6,stroke:#006600
    style OUT fill:#fff0e6,stroke:#cc6600
```

---

## Notation Table

| Symbol | Description | Unit |
|--------|-------------|------|
| Δf | Frequency deviation from nominal | Hz |
| RoCoF | Rate of Change of Frequency | Hz/s |
| ε_f | FFR activation threshold | Hz |
| k_droop | Droop coefficient per VPP | MW/Hz |
| π_slow | Slow-timescale policy (MAPPO) | - |
| π_fast | Fast-timescale policy (MAPPO) | - |
| R_AM | Ancillary market revenue | € |
| R_EM | Energy market revenue | € |
| OPEX | Operational expenditure | €/year |
| H_sys | System equivalent inertia constant | s |
| SOC | State of Charge | % |

---

## Key Innovations Highlighted

1. **Tri-Layer Hierarchical Control**
   - Layer 0: System-level optimization (DSO)
   - Layer 1: Market participation (VPP)
   - Layer 2: Real-time control (DERs)

2. **Dual-Timescale MAPPO**
   - Slow loop: Energy market tracking (15 min)
   - Fast loop: Ancillary service delivery (1 sec)

3. **VPP-Level FFR Aggregation**
   - Aggregate DER responses at VPP level
   - Reduce action space complexity
   - Market-realistic bidding unit

4. **Topology-Aware Frequency Dynamics**
   - H_sys adapts to network configuration
   - GFM/VSG assets provide virtual inertia

5. **Distributionally Robust Optimization**
   - Handle RES uncertainty in day-ahead
   - Worst-case expected profit maximization
