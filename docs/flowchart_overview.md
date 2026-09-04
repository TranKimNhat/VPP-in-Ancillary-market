# VPP Operation Framework Overview

## Dual-Market Participation: AM + EM Integration

```mermaid
graph TB
    subgraph DSO ["Layer 0: Distribution System Operator"]
        Topology["Network<br/>Reconfiguration"]
        Pricing["Zonal<br/>Pricing"]
        Topology --> Pricing
    end

    subgraph Markets ["Dual Market Structure"]
        subgraph AM ["Ancillary Market"]
            AM_DA["Day-Ahead:<br/>Reserve Bidding"]
            AM_RT["Real-Time:<br/>FFR Activation"]
            AM_DA --> AM_RT
        end
        
        subgraph EM ["Energy Market"]
            EM_DA["Day-Ahead:<br/>Energy Scheduling"]
            EM_RT["Real-Time:<br/>Dispatch Tracking"]
            EM_DA --> EM_RT
        end
    end

    subgraph Control ["Layer 2: Dual-Timescale Control"]
        Fast["Fast Loop (1s)<br/>FFR Response"]
        Slow["Slow Loop (15min)<br/>Energy Dispatch"]
    end

    subgraph Assets ["DER Assets"]
        BESS["BESS"]
        EVCS["EVCS"]
        PV["PV"]
        Wind["Wind"]
    end

    DSO --> Markets
    AM_RT --> Fast
    EM_RT --> Slow
    Fast --> Assets
    Slow --> Assets

    style DSO fill:#e6f3ff,stroke:#0066cc
    style AM fill:#fff2e6,stroke:#cc6600
    style EM fill:#e6ffe6,stroke:#006600
    style Control fill:#f5e6ff,stroke:#6600cc
    style Assets fill:#f5f5f5,stroke:#666666
```

---

## Timescale Separation

| Market | Timescale | Control Loop | Primary Objective |
|--------|-----------|--------------|-------------------|
| **AM** | 1 second | Fast (π_fast) | Frequency stability |
| **EM** | 15 minutes | Slow (π_slow) | Energy balance & profit |

---

## Coordination Mechanism

```mermaid
sequenceDiagram
    participant DSO as Layer 0: DSO
    participant VPP as Layer 1: VPP
    participant Fast as Fast Loop
    participant Slow as Slow Loop
    participant DER as DER Assets

    Note over DSO,DER: Day-Ahead Phase
    DSO->>VPP: Zone prices, Topology
    VPP->>VPP: DRO Optimization
    VPP->>DSO: AM Reserve Bid + EM Energy Bid
    DSO->>VPP: Cleared Awards

    Note over DSO,DER: Real-Time Phase (every 15 min)
    Slow->>DER: Energy setpoints (P, Q)
    DER->>Slow: SOC, Generation status
    
    Note over DSO,DER: Real-Time Phase (every 1 sec)
    Fast->>Fast: Monitor Δf, RoCoF
    alt FFR Activation (|Δf| > 0.1 Hz)
        Fast->>DER: Droop response (ΔP)
        DER->>Fast: Power delivered
    end

    Note over DSO,DER: Settlement
    DER->>VPP: Delivered energy & FFR
    VPP->>DSO: Settlement data
```

---

## File References

| Flowchart | File | Description |
|-----------|------|-------------|
| AM Operation | `docs/flowchart_AM.md` | Ancillary market FFR participation |
| EM Operation | `docs/flowchart_EM.md` | Energy market P2P trading |
| Overview | `docs/flowchart_overview.md` | This file - integration view |
