# 분리형 A information/process 학습과 B/C 연결

구현은 `information_process.py`, 실행은 [전체 매뉴얼](../flow-matching_moe/A_INFORMATION_TRAINING_MANUAL.md).
**A+B 공동학습 그림이 아니다.** 기존4dd9bf4에서 별도로 확장했으며 output은 surface4변수다.

```mermaid
flowchart TB
    H["관측 surface history / origin"] --> E["A surface encoder: spatial DCT + MLP"]
    C["관측 origin C: upper fields + static terrain"] --> I["A information encoder"]
    E --> Z["raw z = E(x) + I(C_origin)"]
    I --> Z
    Z --> R["Surface decoder / physical drift"]
    Z --> AUX["A 전용 small context + residual FM sampler"]
    N["member별 독립 persistent noise"] --> AUX
    AUX --> AP["A full20 physical recurrence / state decode"]
    R --> AP
    AP --> AL["fair state CRPS + transition CRPS + trajectory Energy"]
    R --> DL["reconstruction / PI / decoded AE delta / finite-step drift"]
    Z --> IH["학습된 information reconstruction head"]
    IH --> IL["static L2 / paired fixed-target alignment / dynamic info CRPS"]
    AP --> IH
    FT["future surface / upper observations: labels only"] -. "감독만" .-> AL
    FT -. "감독만" .-> IL
    AL -. "gradient: A만" .-> Z
    AL -. "gradient" .-> AUX
    DL --> BEST["phase6 이후 best A + quality audit"]
    IL --> BEST
    AL --> BEST
    BEST --> SEAL["train-only affine / chart / reference seal 1회"]
    SEAL --> B["B: 새 experts / gate / history 학습; A frozen"]
    B --> CAL["C: calibration split; 작은 representation LR + anchor"]
    CAL --> OUT["동일 저장 forecast → 모든 member 6h / 12h 영상"]
```

A sampler/context/info head는 auxiliary다. B/C forecast는 이 sampler 대신 기존 full-state MoE를 사용한다.
그래서 A의 확률적 목적이 B 최종 불확실성에 자동으로 전달되었다고 말할 수 없다. B 재학습 및 비교가 필요하다.

## 두 시간축 / 하나의 member 경로

```mermaid
flowchart LR
    C0["origin C / history 고정"] --> CT["조건 context"]
    Q["member m의 현재 physical q_j"] --> GEO["decoder/Jacobian 공통 geometry"]
    N["같은 member의 base noise epsilon_m"] --> RT["잔차 공간 r_tau"]
    RT --> K["B/C K experts: 동일 q_j, r_tau, tau"]
    CT --> K
    GEO --> K
    K --> G["projection + gate simplex로 vector fields 결합"]
    G --> TAU["tau midpoint ODE: 0 → 1"]
    TAU --> RT
    TAU --> R1["잔차 endpoint: q/day"]
    Q --> D["A drift z/day → q/day"]
    R1 --> STEP["q_next = q_j + dt_hours/24 * drift_plus_residual"]
    D --> STEP
    STEP --> NEXT["다음 physical step의 실제 입력"]
    NEXT --> Q
    STEP --> DEC["surface state decode + 고정 origin offset"]
```

`tau` 적분은 분포 생성 시간이고 `j*6h`는 실제 예보시간이다. `dr/dtau`를 물리 drift로 더하지 않는다.
잔차 endpoint만 같은 단위로 더한다. A auxiliary는 raw z/day로 계산해 seal에 불변이고 B/C는 standardized q/day다.
비선형 decoder는 state를 복원한다. Jacobian 방향 oracle과 유한6h decoded secant는 같은 것이 아니다.

## 입력 shape / 미래 정보의 경계

```mermaid
flowchart TB
    OBS["관측: history B,L,4HW / C_origin B,FHW"] --> ENC["z_origin / 고정 context"]
    ENC --> PATH["생성 B,M,21,4HW: origin + 20 future endpoints"]
    PATH --> DIFF["같은 member 차분 B,M,20,4HW"]
    DT["actual dt B,20 = 6h"] --> DIFF
    DIFF --> SCALE["state_scale 복원 / dt / train tendency_scale"]
    TRUTH["future surface B,20,4HW / future C B,20,FHW"] --> LOSS["감독 graph: CRPS / joint Energy / 작은 anchor"]
    PATH --> LOSS
    SCALE --> LOSS
    LOSS -. "단계별 trainable 모듈에 gradient" .-> ENC
```

미래 labels→encoder conditioning 화살표는 없다. teacher-forced FM은 별도 one-step 감독으로만 현재/다음 정답 pair를 쓴다.
미래 upper label을 history/router에 넣지 않는다. 알려진 static geometry와 동적 upper fields를 구분하고
inference 중 dynamic C도 origin snapshot으로 고정한다. B에서는 A가 frozen이어도 decoder 입력 gradient는 유지한다.
