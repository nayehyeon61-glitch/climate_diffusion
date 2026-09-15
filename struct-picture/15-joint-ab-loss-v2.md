# A+B 공동 학습 + Loss V2: 실제 단계·시간·gradient 계약

기준 코드 `3fc5328`, 2026-09-15 점검. [전체 실행 순서](../flow-matching_moe/JOINT_AB_TRAINING_MANUAL.md).
이 그림은 연결된 코드의 구조이지 ERA5 성능 검증 결과가 아닙니다.
V2 tendency 단위 환산/gradient logger는 [미해결 항목](../docs/results/joint-ab-loss-v2/manual-audit-2026-09-15.md)을 확인하세요.

## 학습 단계와 best checkpoint

```mermaid
flowchart TB
  DATA["Exact 6h archive / canonical msl,t2m,u10,v10"] --> AUDIT["Train-only stats / purged five-way split"]
  AUDIT --> A["Warmup A: train<br/>reconstruction + PI + decoded AE delta + finite-step drift"]
  A --> AV["expert_validation selects best A"]
  AV --> SEAL["Once: train-only affine q, charts, reference encoder"]
  SEAL --> AB["joint_ab: train<br/>encoder,decoder,drift + experts,gate,history jointly trainable"]
  AB --> ABV["expert_validation selects best AB"]
  ABV --> C["Legacy C: calibration<br/>small LR + warmup reference anchor<br/>FM + marginal scores + explicit temporal weights"]
  C --> CV["validation selects best C"]
  CV --> EVAL["Compare AB/C on validation<br/>state, uncertainty, tendencies, geometry"]
  EVAL --> NPZ["One stored native 6h ensemble forecast"]
  NPZ --> VIDEO["All members: 6h and 12h videos<br/>same samples, fixed scales, exact valid times"]
  EVAL --> TEST["Freeze choices, then final test"]
```

ABでは1本のrecurrent graphを全V2 scoreで共有します。Cは別実装で、marginalとtemporalの
生成が別です。ABのprofileがそのままCに適用されるわけではありません。
affine/chart/referenceはAB後・C後に再sealしません。Cのreferenceもwarmup由来です。

## 同一 member の physical recurrence

```mermaid
flowchart TB
  H["Observed history only<br/>B x 6 x D; origin included"] --> HE["History encoder: fixed origin context h"]
  H --> Q0["Origin encoder + fixed affine -> q_origin"]
  Q0 --> Q["Current physical q_j<br/>B x M x r"]
  N["Persistent noise epsilon_b,m<br/>independent across members"] --> TAU["Residual FM solve: tau 0 to 1"]
  Q --> GEO["Decoder Jacobian + tangent metric<br/>reused within this physical step only"]
  Q --> TAU
  HE --> TAU
  CLOCK["Physical lead clock j / trained H"] --> TAU
  GEO --> EX["All full-state experts at same q_j and residual r_tau"]
  TAU --> EX
  EX --> FUSE["Project candidates; simplex gate fusion<br/>dr/dtau, not physical drift"]
  FUSE --> TAU
  TAU --> R["Residual endpoint r_1 in q/day"]
  Q --> DR["Raw z/day drift divided by latent scale -> q/day"]
  R --> STEP["q_next = q_j + dt_hours/24 * (drift + r_1)"]
  DR --> STEP
  Q --> STEP
  STEP -->|"Next physical step: direct q feedback"| Q
  STEP --> DECODE["origin + decode(q_next) - decode(q_origin)"]
  DECODE --> OUT["One generated trajectory B x M x 21 x D<br/>20 physical steps = 120h"]
```

decoder 출력은 기상장/score 경로입니다. 다음 recurrent 입력으로 decode→encode하지 않습니다.
비선형 decoder에 velocity를 state처럼 넣지 않습니다. `dr/dtau`, `dq/day`, `ds/hour`,
u/v(m/s)는 다른 양입니다. 12h 출력은 원래 6h trajectory의 부분 선택입니다.

## 감독과 gradient 경계

```mermaid
flowchart LR
  OBS["Observed adjacent pair: supervision only"] --> LABEL["Current encoder coordinates<br/>residual target, stop-gradient; no EMA"]
  OBS --> QCOND["Teacher current q: live gradient"]
  LABEL --> FM["Fused/expert FM"]
  QCOND --> FM
  GEN["Generated trajectory from recurrent rollout"] --> SCORES["V2: state/transition CRPS<br/>joint endpoint+increment Energy<br/>small mean state/tendency"]
  TRUTH["Future truth: label only"] --> SCORES
  OBS --> AUX["Geometry / decoded AE delta / finite-step drift / anchor"]
  SCORES -. "backward through physical steps, decode and projection" .-> PARAM["AB: encoder/decoder/drift/experts/gate/history"]
  FM -. backward .-> PARAM
  AUX -. backward .-> PARAM
```

full-windowは`trajectory_edges=0`。陽なscore blockは小さくてもoriginからprefixを積分します。
同じnoiseをlead間で使うだけでjoint trajectory lawの学習成功を主張しません。
member-MSEは新AB/Cで0。finite-M mean MSEも分散への圧力があるため、coverage/spreadと併せて評価します。
**意図する physical tendency は `diff(x_normalized) * state_scale / dt_hours / tendency_scale`**。
現在V2呼び出しではこの`state_scale`復元が欠けているため、図中のtransition scoreが
物理単位まで検証済みという意味ではありません。モデル修正は今回の文書作業に含めていません。
