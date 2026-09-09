# DataLoader·두 시간축·실험 순서

2026-09-09 설계, 아직 제안된 dynamics loader/loss 및 12h 출력 CLI는 구현하지 않았습니다.
[전체 설계](../docs/training-mechanism/README.md).

## 데이터 shape와 정답 누수 방지

```mermaid
flowchart TB
    W["Window start i; origin o = i + S - 1"] --> H["History B x L x D, indices no later than o"]
    W --> X0["Observed origin B x D"]
    W --> T["Raw trajectory B x H+1 x D, indices o through o+H"]
    TIMES["Actual UTC timestamps B x H+1"] --> DT["dt_hours B x H; strictly positive and 6h"]
    T --> DELTA["Adjacent differences B x H x D"]
    MASK["Observed mask at both endpoints"] --> PM["Pair mask: logical AND; fail-fast if missing"]
    DELTA --> V["Physical tendency = delta / dt"]
    DT --> V
    PM --> V
    H --> COND["Only causal conditioning into model"]
    X0 --> COND
    COND --> PRED["Generated endpoints B x M x P x D"]
    V --> SUP["Supervision and evaluation only"]
    T --> SUP
    PRED --> SCORE["Same-member pair Energy"]
    SUP --> SCORE
```

`D=C*Y*X`, `S=(L-1)*history_stride+1`, `P=2` endpoints이며 j=0의 왼쪽은 관측값입니다.
역정규화 후 차분하며 future trajectory는 inference의 입력으로 넘기지 않습니다.
통계는 train raw span 내부의 고유 인접쌍에서 fit합니다. C calibration에서 다시 fit하지 않습니다.

## Physical time과 generation time

```mermaid
flowchart LR
    O["j=0: observed origin"] --> P1["j=1: +6h, s=1/120"]
    P1 --> P2["j=2: +12h, s=2/120"]
    P2 --> PN["j=120: +720h, s=1"]
    Z["One z per member"] --> F1["tau 0 to 1: solve for +6h"]
    Z --> F2["tau 0 to 1: solve for +12h"]
    F1 --> P1
    F2 --> P2
    P1 --> D["Compare physical adjacent endpoint difference / 6h"]
    P2 --> D
```

위 가로 화살표는 **물리 시각의 순서**이지 예측 state를 다음 lead의 입력으로 다시 넣는
autoregressive 경로가 아닙니다. 각 생성 ODE의 lead 조건은 고정됩니다.

## 실험과 후속 12h 출력

```mermaid
flowchart TB
    AUDIT["Hash / split / A best reconstruction and tendency audit"] --> OK{"A geometry adequate?"}
    OK -->|yes| REUSE["Reuse same A best"]
    OK -->|no| AFIT["Retrain A with same architecture; new B/C required"]
    AFIT --> REUSE
    REUSE --> E0["E0: existing baseline"]
    E0 --> E1["E1: paired member noise only"]
    E1 --> E2["E2: add joint endpoint-increment score"]
    E2 --> E3["Separate optional ablations: magnitude OR longer block"]
    E2 --> OPT["Separate sampling / optimizer ablations"]
    E3 --> V["Validation: state + dynamics + ensemble; not test tuning"]
    OPT --> V
    V --> TEST["Locked final candidates: test once and report provenance"]
    V --> OUT["Preserve 6h outputs; select +12h, +24h, ..."]
    OUT --> TRAJ["Same-member trajectory, ERA5, mean and spread"]
    OUT --> D12["12h differences / actual 12h; retain 6h diagnostics"]
```

12h output는 `[M,120,D][:,1::2,:]`이며 origin을 별도 anchor로 포함할 수 있습니다.
Checkpoint H=120/step=6은 유지합니다. 같은 투영에서 그린 state-space trajectory는 공기
입자의 geographic 이동 경로가 아닙니다. 날짜를 shift하거나 fps/풍속을 키워 성능을 보정하지 않습니다.
