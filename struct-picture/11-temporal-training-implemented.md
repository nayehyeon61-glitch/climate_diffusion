# 구현된 State + Dynamics 학습 (2026-09-10)

[실행 README](../flow-matching_moe/RETRAIN_120H.md). 기존 network와 latent/manifold 차원은 유지합니다.
아래 그래프는 `train_manifold_moe._epoch`와 `ManifoldMoE.sample_trajectory`의 실제 계산 경로입니다.

```mermaid
flowchart TB
    RAW["6h ERA5 archive: msl / t2m / u10 / v10"] --> CHECK["시간 / schema / unit / observed_mask 검사"]
    CHECK --> SPLIT["5-way split + future-target purge"]
    SPLIT --> STATS["Train의 unique observations / pairs<br/>state mean-scale + tendency channel scale + area weights"]
    SPLIT --> WIN["120h sliding window: x0 포함 21 states"]
    WIN --> HIST["Causal history: B,L,D<br/>origin까지 관측만"]
    WIN --> TARGET["감독 전용 trajectory / delta / dt<br/>B,21,D / B,20,D / B,20"]
    STATS --> NORM["고정 normalization"]
    HIST --> NORM
    NORM --> A["Stage A: 새 PI manifold 초기화<br/>reconstruction + metric + physics + latent dynamics"]
    A --> SEAL["Best A 재로드 / seal<br/>latent scale / local centers / reference encoder"]
    NORM --> CONTEXT["History-time DCT + encoder: h"]
    SEAL --> B["Stage B: manifold 동결<br/>experts / gate / history 학습"]
    TARGET --> BLOCK["연속 sub-block E=2 또는 full E=20<br/>integer physical leads j"]
    BLOCK --> SUP["관측 endpoints + adjacent increments"]
    BLOCK --> LEAD["condition s=j / trained H<br/>신규 H20, 기존 H120 유지"]
    NOISE["독립 member noise z: B,M,r<br/>같은 member는 모든 lead에서 재사용"] --> Q["member / lead의 현재 q_tau"]
    CONTEXT --> EXP["K full-state local experts"]
    Q --> EXP
    LEAD --> EXP
    EXP --> PROJ["공통 decoder Jacobian으로 tangent projection<br/>각 expert intrinsic candidate"]
    Q --> GATE["Manifold local prior + learned gate<br/>simplex fusion weights"]
    CONTEXT --> GATE
    LEAD --> GATE
    PROJ --> MIX["각 ODE step에서 candidate vector fields 가중 결합"]
    GATE --> MIX
    MIX --> ODE["Member / lead별 하나의 최종 ODE<br/>dq / dtau"]
    ODE -->|"다음 생성 step"| Q
    ODE -->|"tau=1"| DEC["State decoder<br/>입력 q의 gradient 유지"]
    DEC --> EP["예측 endpoints: B,M,E+1,D<br/>lead 0은 observed origin"]
    EP --> DIFF["같은 member의 endpoint 차분<br/>actual dt와 train tendency scale로 정규화"]
    SUP --> LOSSES["loss_trajectory: joint endpoint-increment fair Energy<br/>loss_delta: ensemble-mean tendency MSE<br/>선택 wind-speed / bounded sin-cos loss"]
    EP --> LOSSES
    DIFF --> LOSSES
    B --> OLD["기존 FM / PI / specialization objective 유지"]
    OLD --> TOTAL["L_existing + ramp(epoch) * weighted new losses"]
    LOSSES --> TOTAL
    TOTAL -. "gradient: decoder 입력을 통과" .-> EXP
    TOTAL -. "gradient" .-> GATE
    TOTAL -. "gradient" .-> CONTEXT
    TOTAL --> BESTB["Best B checkpoint<br/>frozen A hash 확인"]
    BESTB --> C["Stage C: calibration split<br/>작은 manifold LR + anchor<br/>기존 marginal Energy / CRPS + 새 losses"]
    C --> VAL["Validation full-window score<br/>변수별 state / tendency / spread / coverage"]
    VAL --> CK["Best C + config / stats / parent hashes"]
```

Gradient: B에서 A parameter의 `requires_grad=False`와 `decoder(q)`에 대한 gradient는 별개입니다.
decoder를 `no_grad`로 둘러 새 loss를 끊지 않습니다. C는 manifold도 작은 LR로 업데이트하지만
reference encoder와 좌표 anchor를 보존합니다. 비선형 decoder에 velocity를 state처럼 넣지 않습니다.
새 objective에서 정의한 physical tendency는 **ds/dphysical-hour**이며 FM의 dq/dtau 또는 wind m/s가 아닙니다.

Data shape와 leak 방지:

```mermaid
flowchart LR
    PAST["raw i ... origin o<br/>o=i+history_span-1"] --> H["history B,L,D"]
    H --> CONDITION["모델 condition h"]
    ORIGIN["x0 = raw o"] --> PATH["trajectory B,21,D"]
    FUTURE["raw o+1 ... o+20<br/>valid=o_time+6j hours"] --> PATH
    PATH --> DELTA["delta j = raw o+j+1 - raw o+j<br/>B,20,D"]
    DELTA --> RATE["tendency = delta / actual dt_hours"]
    RATE --> ONLY["감독 loss / diagnostics 전용"]
    PATH --> ONLY
    CONDITION --> GENERATE["새로운 독립 member 시나리오"]
    GENERATE --> ONLY
```

미래 supervision에서 condition으로 향하는 edge가 없습니다. 전체 window 옵션이 아닌 sub-block
학습에서는 선택된 인접 구간만 joint score를 받습니다. M member 각각을 같은 정답에 MSE로
강제하지 않으며, mean MSE의 유한-M 분산 penalty도 작은 weight/ramp/ablation으로 검사합니다.
