# Full-state MoE: 저장 모델·member별 추론

## 저장되는 checkpoint

```mermaid
flowchart TB
    CK["moe.pt: flow_matching_moe.v1"] --> CFG["Grid/schema + model config<br/>AE dims 64/160 + horizon + cadence"]
    CK --> STATE["History encoder + temporal/spatial DCT buffers<br/>K expert encoders / reconstruction / velocity heads<br/>warm-up router + meta encoder / gate / residual / reconstruction"]
    CK --> PROV["Train-only mean/std<br/>5-way split + archive SHA + warm-up SHA<br/>stage + validation-selected epoch"]
    MAN["manifest.json: checkpoint SHA / forecast step"] --> LOAD["LatentFlowForecaster<br/>format dispatch + verification + eval/freeze"]
    CK --> LOAD
    LOAD --> RUNNER["FlowMatchingWeatherRunner<br/>기존 rollout contract"]
    LOAD --> CLI["forecast-climate-flow<br/>evaluate-climate-flow"]
```

Warm-up checkpoint는 `stage=experts`, 최종 checkpoint는 `stage=meta`입니다. 최종 파일에는
모든 expert가 포함되므로 추론 시 warm-up 파일을 별도로 요구하지 않습니다. 원본
monthly/dynamics checkpoint도 기존 dispatch로 계속 동작합니다.

## 한 member·한 physical lead의 최종 ODE

```mermaid
flowchart TB
    HIST["Origin 이하 history<br/>학습 cadence와 grid로 선택"] --> NORM["저장된 mean/std로 정규화"]
    NORM --> H["History-time DCT → frozen context h"]
    NOISE["z_m ~ N(0,I)<br/>다른 member와 독립"] --> INIT["X_m at tau=0"]
    INIT --> LOOP["현재 공통 X_m, tau<br/>같은 member 안의 expert들은 동일 입력"]
    H --> LOOP
    LEAD["고정 physical lead s<br/>예: +720h = 30일"] --> LOOP
    LOOP --> DCT["Spatial DCT of full state"]
    DCT --> EX["E1 ... EK<br/>모든 full-state velocity 후보를 평가"]
    EX --> IDCT["각 후보 IDCT<br/>Meta 직전 공통 physical-grid 좌표"]
    IDCT --> META["Meta AE160<br/>X_m + 후보들 + h + tau + s"]
    LOOP --> META
    META --> GATE["alpha simplex + bounded residual"]
    IDCT --> FUSE["v_final_m = sum alpha_k v_k + r"]
    GATE --> FUSE
    FUSE --> STEP["하나의 midpoint ODE update<br/>full/midpoint state에서 각각 fusion"]
    STEP --> DONE{"tau = 1 ?"}
    DONE -->|아니오| LOOP
    DONE -->|예| OUT["한 member의 endpoint<br/>mean + scale × X_m"]
    OUT --> PACK["모든 M member와 H lead를 모음<br/>forecast shape M × H × D"]
    PACK --> RESTORE["변수 schema로 xarray 복원<br/>또는 NPZ predictions + lead_hours"]
```

그림의 update 블록 안에서 중간점의 vector field도 다시 모든 expert/meta를 통과합니다.
가중치를 시작점에서 한 번만 구해 끝까지 고정하지 않습니다. Expert별 trajectory를 따로
적분한 뒤 endpoint를 평균하는 경로는 없습니다.

## 두 종류의 gate와 두 시간축

| 항목 | Warm-up | 최종 meta |
|---|---|---|
| `g_psi` | expert별 FM 책임과 router-only velocity | 동결; baseline 및 진단용 |
| `alpha_phi` | 초기 균등값; 학습 안 함 | 각 현재 state/후보/조건에 따른 fusion |
| `r_phi` | 0 초기화; 학습 안 함 | bounded residual correction |
| `tau` | interpolation/생성 flow time | 매 ODE evaluation에서 갱신 |
| Physical lead | 조건 `s=(j+1)/H` | 해당 lead solve 동안 고정 |

Meta 추론에서 `g`와 `alpha`를 곱하지 않습니다. `--moe-mode experts`와 `uniform`은
동일 ODE 구현에서 결합 정책만 바꾸는 ablation입니다. `uniform`도 expert endpoint 평균이 아닙니다.

여러 lead에는 같은 member noise를 재사용합니다. 그러나 lead별 ODE를 각각 풀고 학습 loss는
lead별 분포에 계산하므로, 현재 출력은 **common-random-number coupling을 가진 조건부
marginals**입니다. Joint stochastic trajectory 학습 또는 물리 시간 latent ODE라고 해석하면
안 됩니다. 미래 정답·GPT state·WeatherNext API는 이 추론 경로의 입력이 아닙니다.

## 실제 held-out 진단

![Test and routing diagnostics](../docs/figures/moe-smoke/test-diagnostics.svg)

합성 smoke의 routing 값과 meta alpha는 **실제 생성 ODE의 midpoint state**에서 측정했습니다.
Teacher-forced FM interpolation에서의 gate 사용률과 혼동하지 마세요. Hand-made toy mode와
router의 관련성은 기상 regime 분업이 입증됐다는 뜻이 아닙니다.
