# 같은 member를 유지하는 120h 출력

```mermaid
flowchart TB
    CK["Checkpoint: trained H / step_hours / schema<br/>A-B-C provenance / train-only statistics"] --> LOAD["LatentFlowForecaster"]
    RAW["고정 archive의 causal history + 정확한 origin UTC"] --> LOAD
    LOAD --> H["History context h"]
    Z["z[b,m,r]: member별 독립 noise"] --> SHARED["같은 member의 모든 lead에 같은 초기 z"]
    H --> SOLVE["sample_trajectory<br/>각 member / lead에서 모든 expert 평가<br/>project + simplex fusion 후 하나의 ODE 적분"]
    SHARED --> SOLVE
    J["요청 prefix j=1..20<br/>s=j / checkpoint H"] --> SOLVE
    SOLVE --> DECODE["최종 q를 state decode + 역정규화"]
    DECODE --> SAVE["forecast-native-6h.npz<br/>M,20,D + origin + lead_hours + valid_times"]
    SAVE --> JOIN["ERA5 truth와 exact UTC valid-time join<br/>shift / reforecast / member reorder 없음"]
    RAW --> JOIN
    JOIN --> SIX["기본 +6..+120h: 20 future states<br/>origin 포함 21 states"]
    JOIN --> TWELVE["선택 +12..+120h: 10 future states<br/>origin 포함 11 states"]
    SIX --> ALL["저장된 모든 member ID 일괄 export"]
    TWELVE --> ALL
    ALL --> MOVIE["member-000 / 001 / ... GIF 또는 MP4<br/>왼쪽 generated member, 오른쪽 ERA5<br/>t2m K colorbar + u/v m/s quiver + coastlines"]
    ALL --> JSON["member별 state / tendency RMSE<br/>진폭비 null + valid count / wind speed-direction / lag"]
    ALL --> AGG["별도 ensemble summary<br/>mean tendency와 member temporal spread"]
    ALL --> LINE["변수별 member tendency 시계열 그림"]
```

`120h`와 `120 steps`는 다릅니다. 신규 H20 모델은 s=j/20, 기존 H120 모델의 120h prefix는
s=j/120입니다. 출력 길이로 checkpoint의 물리 lead 조건을 다시 정규화하지 않습니다.
6h 원본을 유지하고12h는 정확한 index 선택만 합니다. origin은 별도 state로 저장합니다.

Quiver는 Eulerian wind: 격자 위치는 고정되고 u/v 방향과 길이만 업데이트됩니다.
이는 입자 궤적이나 FM 생성 velocity가 아닙니다. 각 member 영상끼리 동일한 온도/화살표 scale을
사용하며 FPS 또는 화살표 scale 조정이 모델 동역학 개선이라는 뜻은 아닙니다.
mean 영상은 기존 renderer로 요청할 때만 선택적으로 생성합니다.

실행은 [RETRAIN_120H.md의 8번](../flow-matching_moe/RETRAIN_120H.md#8-같은-forecast에서-모든-member-출력),
합성 예시는 [member 0](../docs/results/temporal-120h-smoke/members-12h/member-000.gif),
[member 1](../docs/results/temporal-120h-smoke/members-12h/member-001.gif),
[member 2](../docs/results/temporal-120h-smoke/members-12h/member-002.gif)입니다. ERA5 결과가 아닙니다.
