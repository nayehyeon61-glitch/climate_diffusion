# 최종 Manifold MoE: 설계와 A/B/C 학습

[실행 순서·수식·단위·한계](../flow-matching_moe/MANIFOLD_README.md),
[member별 추론](07-manifold-inference.md), [실제 결과](../docs/results/manifold-smoke/README.md).
이 그림은 `train-climate-manifold-moe`용입니다. 이전 Meta160 경로는 04/05 문서에 보존합니다.

## A: Physics-informed 좌표 학습

```mermaid
flowchart TB
    DATA["연속 6h surface archive와 observed mask"] --> CHECK["Schema, 시간 간격, 관측 완전성 검사"]
    CHECK --> SPLIT["시간순 five-way split과 future-target purge"]
    SPLIT --> TRAIN["Train 관측과 인접한 실제 다음 관측"]
    TRAIN --> NORM["Train-only state 및 physics 통계"]
    NORM --> ENC["공간 DCT와 PI encoder E"]
    ENC --> Z["잠재 좌표 z"]
    Z --> DEC["PI decoder와 공간 IDCT"]
    DEC --> REC["State reconstruction"]
    DEC --> PHY["Divergence, vorticity, gradients, KE 진단 재구성"]
    Z --> MET["물리 거리 보존 및 실제 시간 latent dynamics"]
    REC --> LA["A loss: rec, physics, invariant proxy, metric, dynamics"]
    PHY --> LA
    MET --> LA
    LA --> SELECT["Expert-validation PI loss로 A 선택"]
    SELECT --> SEAL["Train latent 통계와 chart centers 고정"]
    SEAL --> CACHE["Stage A checkpoint와 reference encoder"]
    CACHE --> VIS["Train-fit PCA, chart, 물리 신호 시각화"]
```

물리항은 관측과 재구성의 진단 비교입니다. 대기를 divergence=0으로 만들거나 primitive
equation을 정확히 풀지 않습니다. PCA 색 분리는 학습한 기상 regime의 존재를 증명하지 않습니다.

## B: 고정 geometry에서 국소 expert 분업

```mermaid
flowchart TB
    A["A checkpoint: PI manifold와 chart centers 동결"] --> Q["미래 정답의 intrinsic 좌표 q_s"]
    H["Origin 이하 history"] --> CTX["PI encode, history-time DCT, context h"]
    N["독립 noise z와 생성시간 tau"] --> PAIR["q_tau와 target velocity u"]
    Q --> PAIR
    PAIR --> EXP["K full-state experts: bottleneck 64와 별도 velocity head"]
    CTX --> EXP
    EXP --> ID["후보 velocity 공간 IDCT"]
    ID --> PROJ["공통 decoder Jacobian으로 후보별 tangent lift"]
    A --> PROJ
    PAIR --> GATE["고정 local prior와 bounded condition correction"]
    CTX --> GATE
    PROJ --> ERR["Expert별 intrinsic FM error"]
    ERR --> RESP["Detached responsibility: local prior와 expert error"]
    GATE --> CE["Responsibility에 대한 gate CE"]
    RESP --> CE
    RESP --> EF["Responsibility-weighted expert FM"]
    ERR --> EF
    PROJ --> FUSE["Simplex pi로 intrinsic vector fields 결합"]
    GATE --> FUSE
    FUSE --> LF["Fused FM"]
    CE --> LB["B loss와 expert, gate, history optimizer"]
    EF --> LB
    LF --> LB
    PROJ --> REG["Projection, weak diversity, balance, entropy guard"]
    GATE --> REG
    REG --> LB
    LB --> SEL["생성 Energy와 CRPS로 B 선택 및 저장"]
    SEL --> AUDIT["영역별 expert error와 실제 ODE routing audit"]
```

Projection 자체는 담당 영역을 만들지 않습니다. 영역 prior와 responsibility가 분업을 유도합니다.
Expert 출력의 공통 dynamics를 무조건 직교시키지 않습니다. 교차 영역에서도 overlap을 허용합니다.

## C: geometry anchor를 유지하는 공동 보정

```mermaid
flowchart TB
    B["선택한 B checkpoint 재로드"] --> FIT["Calibration split과 독립 member noise"]
    FIT --> FM["B의 FM, gate, projection 목적함수"]
    FIT --> ODE["모든 expert를 step마다 평가하고 fusion 후 intrinsic ODE"]
    ODE --> OUT["Decoder로 실제 ensemble endpoint 생성"]
    OUT --> ENS["Energy, CRPS, bounded spread penalty"]
    B --> REF["고정 A reference encoder와 chart 통계"]
    REF --> ANCHOR["좌표 anchor와 PI reconstruction, metric, dynamics"]
    FM --> TOTAL["C total loss"]
    ENS --> TOTAL
    ANCHOR --> TOTAL
    TOTAL --> OPT["Experts, gate, history: LR x 0.1"]
    TOTAL --> SLOW["PI encoder와 decoder: LR x 0.01"]
    OPT --> SELECT["별도 validation Energy와 CRPS로 선택"]
    SLOW --> SELECT
    SELECT --> CKPT["Final checkpoint, schema, 통계, manifest"]
    CKPT --> TEST["Untouched test와 uniform-fusion 비교"]
    TEST --> PLOTS["RMSE, CRPS, spread-skill, rank, 지역별 전문화 그림"]
```

C는 encoder도 보정하므로 FM target 좌표를 detach하고 A reference anchor를 유지합니다.
Jacobian과 ODE를 통한 decoder/input gradient는 유지합니다. 고정 기준 좌표/physics 통계는
다시 fit하지 않습니다. B/C의 validation 구간이 달라 stage별 selection score의 높이를
서로 직접 비교하지 말고 동일 test 조건의 forecast skill을 비교하세요.
