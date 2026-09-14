# Joint A+B + Loss V2

```mermaid
flowchart LR
  H[Observed origin/history only] --> E[Encoder + fixed affine q]
  E --> Q[q at physical step j]
  N[Persistent independent member noise] --> R[Residual FM tau ODE]
  Q --> R
  Q --> B[latent drift q/day]
  R --> F[expert fields, same q, fused each tau]
  B --> S[q next = q + dt/24 (drift + residual endpoint)]
  F --> S
  S --> D[decoder + origin anchored offset]
  D --> Q
  D --> L[one B,M,T+1,D graph]
  L --> C1[state CRPS, origin excluded]
  L --> C2[transition CRPS, actual dt and train-only scale]
  L --> C3[endpoint + increment Energy]
  L --> C4[ensemble mean state/tendency]
```

The FM teacher target is detached; generated rollout, decoder Jacobian and
projection graphs are not.  q/day, q/hour, physical units/hour and generative
dr/dtau are never interchanged. Geometry is reusable only within one physical
step at the identical q tensor.
