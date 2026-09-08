# Exact valid-time comparison and temporal diagnostics

```mermaid
flowchart LR
    A[Archive snapshots<br/>exact UTC valid_time] --> O[Select origin]
    C[Frozen MoE / Manifold MoE<br/>or saved ensemble NPZ] --> F[M x H x D forecast]
    O --> K[origin + physical lead_hours]
    F --> K
    K --> J{Exact timestamp join}
    A --> J
    J --> V[Fixed t2m color limits<br/>same u/v scale and m/s key]
    J --> D[delta state / delta physical hour]
    J --> E[RMSE CRPS u/v error<br/>anomaly and lag diagnostic]
    V --> M[MP4 or GIF]
    D --> M
```

`tau` is not on this timeline: `dq/dtau` is the velocity inside each generative
solve at a fixed physical lead. The comparison uses `dX/dt` from adjacent valid
times. Wind arrows show `u10/v10` in their source units and use one fixed scale
and key for generated and ERA5 panels. Lag is reported only; truth is never
shifted or fed to the prediction.
