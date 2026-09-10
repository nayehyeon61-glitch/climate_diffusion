# Full-state MoE: 학습 구조

구현: `moe.py`, `train_moe.py`, `moe_data.py`. 기존 physical latent-dynamics trainer와
**별도 학습 경로**입니다. [실행 README](../flow-matching_moe/README.md).

## 데이터·좌표·네트워크 전체

```mermaid
flowchart TB
    A["Fixed-step fields + schema + observed_mask"] --> CHECK["6h/time/grid 검사<br/>결측 또는 integrated feature: fail-fast"]
    CHECK --> SPLIT["시간순 5-way split<br/>경계마다 horizon-aware purge"]
    SPLIT --> NORM["Expert train span mean/std만 추정<br/>두 단계·추론 공통 정규화"]
    NORM --> HIST["Causal history B × L × D"]
    HIST --> HDCT["Orthonormal DCT-II<br/>history 시간축 L"]
    HDCT --> H["History context MLP → h"]
    NORM --> Y["학습 split의 미래 정답 y_s"]
    Z["독립 Gaussian z<br/>member마다 1개"] --> INTERP["X_tau = (1-tau)z + tau y_s<br/>u = y_s - z"]
    Y --> INTERP
    T["Flow time tau ∈ 0..1<br/>Physical lead s는 별도 조건"] --> INTERP
    INTERP --> SDCT["현재 full state의 공간 DCT<br/>각 변수의 lat/lon 축"]
    H --> EX["K full-state conditional experts<br/>각 encoder bottleneck 64"]
    SDCT --> EX
    T --> EX
    EX --> REC["별도 state reconstruction heads<br/>Stage 1 auxiliary loss"]
    EX --> V["별도 supervised velocity heads<br/>공간 주파수 vector fields"]
    V --> IDCT["IDCT: Meta Learner 직전<br/>표준화 physical-grid velocity로 복귀"]
    H --> ROUTER["Warm-up router g<br/>softmax simplex"]
    SDCT --> ROUTER
    T --> ROUTER
    ROUTER --> L1["Stage 1: router-weighted FM<br/>+ reconstruction + balance + bounded diversity"]
    IDCT --> L1
    INTERP --> L1
    L1 -. "gradient: expert/history/router만" .-> EX
    L1 -. gradient .-> H
    L1 -. gradient .-> ROUTER
    IDCT --> META["Meta learner<br/>encoder bottleneck 160"]
    H --> META
    INTERP --> META
    T --> META
    META --> ALPHA["alpha = softmax<br/>K candidate weights"]
    META --> RES["r = 2 tanh residual head"]
    META --> MREC["Meta state reconstruction<br/>auxiliary loss"]
    ALPHA --> FUSE["v_final = sum alpha_k v_k + r<br/>공통 full-field 좌표"]
    RES --> FUSE
    IDCT --> FUSE
    FUSE --> ODE["Member별 최종 ODE 적분<br/>매 midpoint 평가에 모든 expert + meta 재실행"]
    Z --> ODE
    ODE --> ENS["M generated endpoints"]
    FUSE --> L2["Stage 2: fused FM + Energy + CRPS<br/>+ spread band + meta reconstruction + residual L2"]
    ENS --> L2
    Y --> L2
    MREC --> L2
    L2 -. "gradient: meta parameters만<br/>frozen experts의 입력 미분은 유지" .-> META
```

`D=C×Y×X`는 전체 archive field 크기입니다. `64/160`은 내부 AE bottleneck이며
ODE state dimension이 아닙니다. Expert의 state reconstruction decoder와 velocity head는
서로 다릅니다. IDCT는 선형 velocity 변환입니다. 최종 state의 물리 단위 변환은 별도
`x = mean + scale * X` 역정규화이며 IDCT라고 부르지 않습니다.

그림의 FM interpolation 입력 경로는 학습용입니다. Ensemble ODE loss 경로는
정답 endpoint를 state로 넣지 않고 독립 noise에서 생성합니다. 정답은 비교 loss에만 갑니다.

## 두 단계 optimizer·checkpoint 생명주기

```mermaid
flowchart LR
    TR["train windows"] --> S1["Stage 1 optimizer<br/>Experts + History + Router"]
    S1 --> EV["expert_validation<br/>실제 ensemble Energy+CRPS"]
    EV --> W["선택된 experts.pt<br/>schema / norm / split / SHA"]
    W --> RELOAD["선택 epoch 재로드<br/>experts/history/router freeze"]
    CAL["calibration windows<br/>Stage 1 fit에서 제외"] --> S2["Stage 2 optimizer<br/>Meta learner만"]
    RELOAD --> S2
    S2 --> VAL["validation<br/>고정 noise/lead로 선택"]
    VAL --> FINAL["선택된 moe.pt<br/>모든 모듈 + 동일 norm + warm-up SHA"]
    FINAL --> TEST["Untouched test 평가<br/>router-only / uniform / meta"]
```

체크포인트 선택이 끝나기 전에는 test 점수로 epoch를 고르지 않습니다. Stage 1과 Stage 2의
validation 데이터가 다르므로 곡선 연결만으로 성능 개선을 주장하지 않습니다.

## 실제 학습 그래프

![Expert warm-up](../docs/figures/moe-smoke/experts-training.svg)

![Meta learner](../docs/figures/moe-smoke/meta-training.svg)

이 그래프는 합성 데이터 실행 로그입니다. **실제 ERA5/30일 예측 학습 결과가 아닙니다.**
