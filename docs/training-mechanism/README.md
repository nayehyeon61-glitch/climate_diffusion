# State + Dynamics Matching 설계

**후속 구현 완료:** [120h 실행 README](../../flow-matching_moe/RETRAIN_120H.md),
[실제 계산 그래프](../../struct-picture/11-temporal-training-implemented.md),
[CPU 합성 검증 결과](../results/temporal-120h-smoke/README.md)를 확인하세요.
아래는 2026-09-09 당시의 근거·설계 기록이며, 당시의 미구현 설명은 역사적 상태입니다.

검토일: 2026-09-09. 코드 기준: `a3003d1fb53f5c1fecae21ff7a8cdfcf87069096`,
`feature/latent-dynamics-flow`. **이 문서는 설계이며 새 dynamics loss의 구현 완료 보고가 아닙니다.**
실제 ERA5 재학습·RTX 4090 실행은 이번 작업에서 수행하지 않았습니다.

첫 변경은 **동일 member의 인접 physical lead 두 개를 함께 생성하고, 두 endpoint와 변화량의
결합 분포를 감독하는 것**으로 정합니다. 기존 FM·PI·expert·ensemble objective를 보존합니다.
모델 크기, expert 수, manifold 구조를 늘리지 않습니다.

- [결과와 원본 근거](../results/manifold-run-001-review/README.md)
- [상세 학습·gradient Mermaid](../../struct-picture/09-dynamics-supervision-design.md)
- [DataLoader·시간·단계 Mermaid](../../struct-picture/10-dynamics-data-and-stages.md)
- [구현 순서·4090 실험 계획·현재 실행 가능한 명령](RUNBOOK.md)
- [수학적 계약 probe](probe_design.py): 실제 기상 학습이 아닌, 설계의 반례·정규화 검사

## 1. 지금 관찰한 것과 아직 모르는 것

사용자가 올린 `manifold-run-001-results(1).zip`에는 영상·PNG만 있습니다. Metrics JSON,
예측 NPZ, ERA5 archive, checkpoint, 실행 인자는 없습니다. 아래 수치는 그림의 표기이며
원시 로그에서 새로 계산한 값이 아닙니다. 업로드 원본과 SHA는 근거 문서에 기록했습니다.

| 근거 | 확인한 관찰 | 해석과 다음 확인 |
|---|---|---|
| A 학습 곡선 | train latent-dynamics 감소, validation loss/rec/metric은 초기 최저점 뒤 상승. 선택 epoch 8 | A 후기 일반화 악화에 부합. 계절 분포 차이·metric 표본 구성·압축 손실도 검사 |
| B 학습 곡선 | Energy/CRPS/RMSE 거의 정체, spread 감소. 선택 epoch 2 | 추가 epoch의 효용이 작음. 직접 dynamics 감독 및 gradient 유효성 확인 |
| C 학습 곡선 | Energy/CRPS 소폭 개선. 선택 epoch 10 | 모든 단계가 같은 방식으로 과적합한다고 단정할 수 없음 |
| expert audit | cosine 0.951, local expert best 40.6%, gate best 40.6%, 각 영역 n=10/8/7/7 | 후보 유사성이 높음. 영역 할당이 기상 전문화의 증거는 아님 |
| +366h 화면 | mean과 member 0 모두 t2m delta/hour 패널이 거의 중립색 | 평균화만으로 설명하기 부족. 색 범위·실제 배열·다른 member를 함께 검사 |
| 영상 시각 | origin 2009-04-04 06:00, +366h의 valid time 2009-04-19 12:00 | 해당 화면의 날짜 산술은 일치. 전체 배열/모든 frame 계약 검증은 별도 |

**실제 B/C가 A의 마지막 epoch 50을 사용했다고 추정하지 않습니다.** 현재 trainer는 단계가
끝나면 저장된 best state를 재로드하고, A는 그 state에서 geometry를 seal합니다.
실제 실행 weight가 이 코드와 일치하는지는 checkpoint hash/metadata가 있어야 검증됩니다.

학습 부족·과적합·decoder 표현 손실·시간 조건 무시·ensemble 평균화는 서로 다른 가설입니다.
파라미터 수가 충분하다는 판단만으로 `r=16` decoder image의 표현력을 확인할 수 없습니다.
우선 validation에서 `decode(encode(x_j))`의 state 및 **인접 차분 재구성 오차**를 측정합니다.
이 재구성조차 변화량을 없애면 B/C만의 loss 조정으로 해결 가능한 범위가 작습니다.

## 2. 코드에서 확인한 학습상의 빈틈

| 위치 | 현재 동작 | 설계에 주는 영향 |
|---|---|---|
| `dynamics.py:TrajectoryWindowDataset.__getitem__` | history, origin, targets만 반환 | 실제 timestamp·delta·tendency 감독 계약 추가 필요 |
| `train_manifold_moe.py:_pairs` | 인접 lead 선택, FM source 공유, target code detach | 좋은 시작이나 lead별 FM 회귀만으로 시간 coupling을 학습하지 않음 |
| `train_manifold_moe.py:_sample` | flatten한 lead마다 독립 member noise | 추론의 lead간 동일 noise 계약과 다름. paired endpoint sampler 필요 |
| `train_manifold_moe.py:_epoch` | B train은 local FM·specialization 중심, B validation/C는 endpoint 확률 점수; 차분 loss 없음 | B에 새 differentiable pair score를 넣고 C에도 동일 계약 유지 |
| `manifold_moe.py:manifold_loss` | A의 drift가 6h=0.25일 latent next-state를 맞춤 | 실제 observed-time 보조항은 있으나 field/forecast에서 drift를 호출하지 않음 |
| `manifold_moe.py:forecast` | 각 lead를 같은 초기 noise로 별도 생성 ODE solve | 물리 시간 autoregressive rollout이 아님. `integrate`는 tau 적분 |
| `manifold_moe.py:manifold_loss` | batch size 1이면 metric loss를 0으로 만듦 | A/C에서 batch 1 또는 마지막 singleton batch를 조용히 허용하지 않기 |
| `train_manifold_moe.py:train_manifold_moe` | AdamW weight_decay=0, 고정 LR, 지정 epoch 전부 실행 | 실제 weight decay/scheduler/early-stop 설정 필요. best 저장은 이미 있음 |
| 같은 trainer | 기본 validation 최대 32 windows, epoch마다 고정 RNG | 비교 재현성은 있으나 전체 계절/lead 검증을 대신하지 못함 |
| `model.py:sinusoidal_time_embedding` | 시간 입력에 1000을 곱한 여러 주파수 | lead가 1/H로 작다는 이유만으로 둔화를 단정하지 말고 조건 민감도 측정 |
| `time_alignment.py:temporal_diagnostics` | 전체 state의 원시 단위 혼합 RMS/ratio | Pa/K/m/s를 합친 숫자를 한 물리 속도로 해석하지 않기 |
| `evaluation.py`, `manifold_diagnostics.py` | recorded test split 사용 | 설계 보정 중에는 validation용 audit 경로를 별도로 추가 |

현재 `load_moe_archive`는 시간 간격·finite·fully observed mask를 검사하고, five-way split은
future target 겹침을 거부하며 normalization은 train span에서 계산합니다. 이 검토에서
미래 정답이 history conditioning에 직접 들어가는 코드 경로는 찾지 못했습니다.
사용자 실험의 archive와 checkpoint가 없으므로 실제 데이터 누수 없음까지 인증하지는 않습니다.

가장 우선할 변경은 새 블록을 추가하는 것이 아니라 **실제 물리 차분으로 가는 gradient의
연결과 train/inference의 member coupling 일치**입니다.

## 3. 시간과 좌표의 계약

| 기호 | 정의 / 단위 |
|---|---|
| `t0` | 마지막 관측의 UTC origin |
| `j=0..H` | 물리 예측 시점. j=0은 관측 origin, j=1은 +6h |
| `s=j/H` | 모델의 physical lead 조건. H는 checkpoint의 고정 전체 horizon |
| `tau∈[0,1]` | Gaussian에서 특정 lead state를 생성하는 flow time |
| `dq/dtau` | intrinsic 생성 vector field; 시간당 기상 변화량이 아님 |
| `v_j=(x_{j+1}-x_j)/dt_j` | Eulerian 기상장 tendency. 변수별 Pa/h, K/h, (m/s)/h |
| `u10,v10` | 바람 성분, m/s. 물리 tendency와 다른 양 |

FM은 조건부 확률 경로의 vector field를 회귀하는 방법입니다. 실제 대기의 시간 미분을
얻으려면 별도의 물리 시간 계약과 감독이 필요합니다.
[원논문: Lipman et al.](https://arxiv.org/abs/2210.02747)

각 expert는 현재 q에서 공통 decoder Jacobian의 접공간으로 투영된 후보를 생성합니다.
동일 member/lead의 한 integration step에서 모든 expert가 같은 q를 보고 gate로 결합한
field를 적분합니다. 생성 뒤 expert별 endpoint를 평균하는 방식으로 바꾸지 않습니다.

`xhat_j = sigma_x * decode(q_j(tau=1)) + mu_x`를 먼저 계산한 뒤 물리 시간 차분을 취합니다.
비선형 decoder에 `dq/dtau`를 state처럼 넣거나, FM target `q_target-z`를 `dx/dt`와 직접
MSE로 비교하지 않습니다. `--integration-steps`는 tau 적분 정밀도이지 예보 간격이 아닙니다.

## 4. DataLoader: causal input과 감독용 trajectory

기존 fixed-step archive를 그대로 사용하고 dynamics는 window view에서 파생합니다.
기존 `TrajectoryWindowDataset`를 사용하는 다른 trainer를 깨지 않도록 선택적 확장 또는
manifold 전용 wrapper `TemporalSupervisionDataset`을 제안합니다. 새 archive를 의무화하지 않습니다.

`S=(history_steps-1)*history_stride+1`, window start `i`, origin index `o=i+S-1`로 둡니다.
관측 history index는 `i + arange(history_steps)*history_stride`, target j는 `o+j`입니다.

| 키 | Batch shape | 사용처 |
|---|---|---|
| `history` | `[B,L,D]` | 기존 state 통계로 정규화, 모델 conditioning |
| `origin` | `[B,D]` | 정규화 x0, 첫 차분의 알려진 왼쪽 endpoint |
| `targets` | `[B,H,D]` | 기존 정규화 FM targets 유지 |
| `trajectory_raw` | `[B,H+1,D]` 또는 선택 block view | x0 포함 원래 단위, 감독용 |
| `delta_raw` | `[B,H,D]` | `trajectory_raw[:,1:]-trajectory_raw[:,:-1]` |
| `tendency_raw` | `[B,H,D]` | `delta_raw/dt_hours[...,None]` |
| `dt_hours` | `[B,H]` | 실제 timestamp 차이; 현재 loader는 모든 값 6 요구 |
| `pair_observed_mask` | `[B,H,D]` | 양쪽 endpoint mask의 AND |
| `origin_time_ns`, `valid_time_ns` | `[B]`, `[B,H+1]` | UTC int64; batch.to(device)와 호환되게 수치 metadata 분리 |
| `variable_names`, `units`, `grid` | dataset/schema-level | 문자열을 현재 `.to(device)` 루프에 넣지 않음 |

여기서 `D=C*Y*X`이며 H는 horizon, Y/X는 격자 크기입니다. Future target/delta는 encoder의
감독 branch에서만 사용하며 history·inference gate의 조건으로 전달하지 않습니다.

첫 감독 구간 `j=0`은 `(x0,x1)`, 이후 j는 `(x_j,x_{j+1})`입니다. Future pair j≥1에서는
두 state를 모두 생성해야 하며, 실제 x_j를 conditioning에 주고 x_{j+1}만 생성하면 이 설계의
open-loop 검증과 달라집니다. `xhat_0=x0`는 관측으로 고정하고 s=0 생성은 호출하지 않습니다.

H+1 전체 tensor를 세 벌 저장하지 않고 raw state view와 선택 pair를 이용해 delta/tendency를
batch에서 계산할 수 있습니다. Dynamics 통계 fit은 전체 train의 **고유 인접 관측쌍**에서
한 번만 수행하여 겹치는 window가 동일 쌍을 중복 가중하지 않게 합니다.

### Split과 mask

기존 `train → expert_validation → calibration → validation → test`를 유지합니다.
A/B는 train, C는 calibration으로 학습합니다. 새 tendency 통계도 최초 train에서만 fit하고
C/validation/test에서 다시 맞추지 않습니다. 통계 fit의 양 endpoint는 모두 기록된 train raw
span 내부여야 합니다. timestamp gap을 건너뛴 차분이나 다른 storm/year 파일 경계 차분은 금지합니다.

기존 purge H−1은 **future label disjointness**입니다. 다음 split의 관측 history/origin이
앞 split의 과거 label 시각과 겹칠 수 있는데, 예측 시점에 이미 알려진 관측으로 쓰는 계약입니다.
모든 raw history까지 독립이어야 하는 별도 실험이라면 purge≥S+H−1로 변경하고 A부터 재학습합니다.
계약을 바꾸지 않은 실험에서는 split을 그대로 둬 비교 조건을 보존합니다.

첫 구현은 현재 fully-observed 정책을 유지해 결측쌍을 fail-fast합니다. 추후 부분 관측을 허용하면
pair mask를 loss의 분자/분모 모두 적용하고 유효 쌍 수 0은 skip/error로 기록해야 합니다.
NaN을 0으로 채운 뒤 관측으로 취급하지 않습니다. 기존 coarsening이 skipna 평균을 쓰는 점도
확인해 coarse-cell 완전 관측과 원해상도 전체 관측을 혼동하지 않습니다.

### State와 dynamics 정규화

```math
y_{j,c,p}=\frac{x_{j,c,p}-\mu_{x,c,p}}{\sigma_{x,c,p}},\qquad
\Delta x_{j,c,p}=\sigma_{x,c,p}(y_{j+1,c,p}-y_{j,c,p}).
```

Dynamics scale은 변수별 train tendency의 면적 가중 표준편차로 고정합니다.

```math
b_c=\max\{\operatorname{Std}_{train,area}(v_c),\,\epsilon_c\},\qquad
w_{c,p}=\frac{a_p}{C\sum_p a_p},\qquad
d_{j,c,p}=v_{j,c,p}/b_c.
```

`a_p`는 실제 grid 간격과 cos(latitude)의 양의 면적. 통계는 float64 streaming accumulator로
fit하며 `mu_v`도 진단용으로 저장하되 첫 loss는 양쪽을 같은 b로 나눠 차이를 비교합니다.
floor `epsilon_c`는 train-only state scale/6h의 작은 비율과 변수 단위를 반영해 기록합니다.
예: `1e-3 * area_RMS(sigma_x,c)/6h`; 상수 변수에서는 물리 단위별 양의 최소값을 명시하거나
관측 dynamics가 전혀 없다는 오류를 반환합니다. 검증 집합으로 floor를 맞추지 않습니다.
고정 dt에서는 delta/std(delta)와 tendency/std(tendency)가 같으므로 두 loss를 중복 추가하지 않습니다.

## 5. Ensemble을 유지하는 최소 loss

하나의 ERA5 실현 y에 모든 member를 독립적으로 MSE 회귀시키면

```math
\frac1M\sum_m\|d^{(m)}-d^*\|^2
=\|\bar d-d^*\|^2+\frac1M\sum_m\|d^{(m)}-\bar d\|^2.
```

즉 정확도 항에 더해 ensemble 분산 자체를 벌점으로 줍니다. 단순 memberwise tendency MSE를
주 목적함수로 채택하지 않습니다. Ensemble-mean MSE도 finite M에서는 기대값에
`Var(d)/M`이 포함되므로 무조건 collapse-free라고 부르지 않습니다.

### 선택: paired endpoint + increment Energy score

먼저 `z[b,m,r] ~ N(0,I)`를 생성해 같은 b,m의 선택 lead 둘에 broadcast합니다.
각 lead의 현재 q는 적분 중 서로 달라져도 됩니다. **같은 member의 모든 expert가 q를 공유**하는
것과 **서로 다른 lead가 항상 같은 q여야 한다**는 것은 다릅니다.

각 쌍 j에 대해 `A=diag(sqrt(w))`와 다음 감독 벡터를 만듭니다.

```math
\Psi_j^{(m)}=
\left[\frac{Ay_j^{(m)}}{\sqrt2},\quad
      \frac{Ay_{j+1}^{(m)}}{\sqrt2},\
      A\frac{x_{j+1}^{(m)}-x_j^{(m)}}{dt_j b}\right],\qquad
\Psi_j^*=\Psi_j(x_j^*,x_{j+1}^*).
```

고정 scale에서 두 endpoint를 보존하는 injective 선형 변환이므로 변화량뿐 아니라 state 수준도
감독합니다. Increment만 비교하면 `(x_j+c,x_{j+1}+c)`의 공통 bias를 구별하지 못합니다.
j=0에서는 첫 block이 고정된 관측값이며, 모든 predicted member가 같은 x0를 사용합니다.

```math
L_{pair}=\mathbb E_{b,j}\left[
\frac1M\sum_m\|\Psi_j^{(m)}-\Psi_j^*\|_2
-\frac{1}{2M(M-1)}\sum_{m\ne n}\|\Psi_j^{(m)}-\Psi_j^{(n)}\|_2\right].
```

M≥2, 학습은 independent member draws의 off-diagonal estimator를 사용합니다. 서로 다른
member 번호를 lead마다 섞거나, member 평균을 먼저 내고 차분 score에 넣지 않습니다.
위 A가 이미 면적/차원을 정규화하므로 기존 `fair_energy_score`의 `/sqrt(D)`를 다시 적용하지
않도록 별도 metric-aware helper를 정의합니다. 이중 정규화는 새 항의 gradient를 약하게 만듭니다.

Energy score의 확률 예측 평가 근거는 [Gneiting & Raftery, §4.3](https://sites.stat.washington.edu/raftery/Research/PDF/Gneiting2007jasa.pdf)
입니다. 위 endpoint/increment 특징 선택은 이 프로젝트의 제안입니다. 기대 score의 성질이
한 데이터셋에서의 최적화 성공·기상 skill·ensemble 보정을 보장하지는 않습니다.

```math
L_B^{new}=L_B^{existing}+\lambda_{pair}(e)L_{pair},\qquad
L_C^{new}=L_C^{existing}+\lambda_{pair}(e)L_{pair}.
```

현재 B에는 생성 endpoint train score가 없으므로 L_pair가 처음으로 실제 적분을 통한 감독을
B에 연결합니다. C의 기존 marginal Energy/CRPS도 같은 paired samples에서 재사용합니다.
FM target code 및 책임도 target은 기존처럼 detach합니다. L_pair는 decoder와 적분 경로를
거쳐 experts/gate/history로 역전파합니다. B에서는 PI 파라미터가 frozen이어도 decoder의
입력 q에 대한 gradient는 필요하므로 `no_grad()`로 decode/integrate를 감싸면 안 됩니다.
C는 기존 작은 LR·reference anchor 하에서 PI 파라미터에도 gradient를 허용합니다.

### 추가 항은 분리된 ablation

- Magnitude: `s_m=||A d_m||`, `s*=||A d*||`의 **scalar CRPS**를 작은 가중치로 시험합니다.
  모든 member magnitude를 같은 숫자로 강제하는 MSE보다 분포 평가 목적에 맞습니다.
  양 endpoint의 부호/위상/위치 오류는 norm으로 사라지므로 L_pair를 대신하지 않습니다.
- Mean tendency MSE: 짧은 lead에서 작은 보조항으로만 별도 비교. finite-M 분산 벌점과
  marginal/joint score 악화 여부를 함께 기록합니다.
- Direction cosine: true/pred tendency가 train 기반 임계값보다 큰 쌍만 진단하고, 유효 비율도
  기록합니다. 근거가 생기기 전 학습 항에 넣지 않습니다.
- Rollout: 이 모델에서는 physical autoregression이 아니라 여러 lead의 생성 endpoint입니다.
  제안 `L_block`은 길이 P=3~4 endpoint와 P−1 increment를 함께 쌓은 Energy score로 정의합니다.
  P=2이면 L_pair와 같으므로 두 이름으로 중복 가중하지 않습니다. 가속도 감독은 후속 선택입니다.

단순 음의 spread 보상은 쓰지 않습니다. 기존 C spread band는 유지하되 새 score의 효과와
분리해 기록합니다. B에서의 ensemble spread도 측정합니다. 공통 latent noise가 유도하는
시간 coupling의 표현 한계는 남으며, adjacent pair 적합만으로 30일 전체 joint law가 검증되지는 않습니다.

## 6. Sampling과 optimizer 설계

**비교의 첫 기준은 현재 full-horizon uniform pair sampling**입니다. 시간·loss curriculum은
그 다음 ablation으로 추가해 loss 자체의 효과와 섞지 않습니다.

1. Train window를 epoch마다 shuffle하되 6h 연속 pair 내부 순서는 보존합니다. 현재
   `window_stride=4`가 매일 같은 UTC hour origin만 선택할 수 있으므로 stride offset을
   epoch마다 순환하거나 모든 origin에서 균일 추출하는 실험을 추가합니다.
2. 모든 인접 edge j=0..H−1이 포함될 기회를 줍니다. FM anchor는 전체 lead에 균일 분포로
   유지합니다. 같은 pair의 FM tau도 공유하는 옵션은 수치 상관을 줄이는 별도 계약으로 기록합니다.
3. 계절/시간/변화량 분위수는 train만으로 정합니다. Rare fast-change oversampling은 예보의
   기후 빈도 자체를 바꿀 수 있으므로 원래 확률 p와 sampling 확률 q의 p/q 가중치를
   **원본 sample 전체 loss**에 적용합니다. 표본 없는 계절을 만들거나 validation을 섞지 않습니다.
4. 과도한 p/q 가중치가 필요하면 q를 uniform과 혼합하고 bin 수를 줄입니다. clipping은 목적분포를
   바꾸므로 초기안에서 피합니다. batch 안의 겹치는 window가 주는 낮은 유효 표본 수를 기록합니다.
5. Curriculum 선택안: 처음 2 epoch는 짧은 1~2일 pair, 다음 2 epoch는 7일, 이후 30일로
   확장하되 전체 pair 확률을 항상 양수로 남깁니다. **s=j/120은 그대로**이며 7일 단계에서
   j/28로 바꾸지 않습니다. 검증 panel은 처음부터 전 lead 고정입니다.
6. L_pair 가중치는 0에서 시작해 3~5 epoch 동안 예비값 0.1로 ramp하는 것을 작은 후보로 둡니다.
   이는 검증된 최적값이 아닙니다. 초반 layer별 `||grad_pair||/||grad_existing||`를 기록해
   새 항이 사라지거나 지배하는지 판단하고 후보를 한 번에 하나씩 바꿉니다.
7. AdamW weight decay 0/1e-4, 현재 LR와 절반 LR를 작은 사전 정의 grid로 비교합니다.
   scheduler patience 2, early-stop patience 5, relative min_delta 0.1%는 출발 후보입니다.
   Early stopping과 이미 있는 best checkpoint 재로드를 함께 유지합니다.

선택 지표는 고정된 held-out panel의 `legacy marginal Energy+CRPS + eta*pair Energy`로
고정합니다. eta는 pilot의 **train-only scale**로 정한 뒤 validation에서 매 epoch 다시 맞추지 않습니다.
학습 중 ramp되는 lambda를 selection score에 그대로 써 서로 다른 epoch 목적을 비교하지 않습니다.
State score/변수별 RMSE가 사전 tolerance 이상 나빠지면 새 모델을 채택하지 않는 gate도 둡니다.

A/B selection은 expert_validation, C selection은 validation. 모든 stage의 검증을 32개
window에만 의존하지 말고 예산에 맞는 고정 multi-season origin/lead panel을 기록합니다.
6h pair의 첫/중간/긴 lead와 저/고 변화량 bin이 충분한지 확인합니다. 현재 A validation metric은
batch 내 `roll(1)` pair 구성에도 의존하므로 fixed validation pair 목록을 별도로 만들어
GPU batch 크기를 바꿔도 같은 metric을 비교하도록 설계합니다.

### A 재사용 판단

먼저 동일 schema/archive/split의 A best를 사용한 AE reconstruction+tendency validation과
한-step physical latent-drift auxiliary를 측정합니다. 단순히 후기 validation이 상승했다는
이유만으로 정상적으로 선택된 epoch 8을 폐기하지 않습니다. Reconstruction 차분도 매우 작거나
metric 일반화가 부족하면 **같은 architecture로 A의 학습률·pair sampling·regularization을
보정하여 A를 다시 학습**합니다. A를 바꾸면 latent scale/chart centers가 달라지므로 B/C도
새로 학습합니다. 서로 다른 A 좌표계의 B/C weight를 무검증 재사용하지 않습니다.

현재 CLI는 C→C resume와 optimizer/RNG 복원을 지원하지 않습니다. 첫 구현에서 frozen B의
dynamics fine-tune 재시작을 허용하려면 명시적인 warm-start 옵션과 provenance를 추가합니다.
이를 exact resume라고 부르지 말고 scheduler/optimizer 재초기화를 기록합니다.

## 7. 평가로 가설을 구별하기

기존 `time_alignment`의 날짜를 shift하지 않고 per-variable 원래 단위 및 train dynamics scale
두 기준을 저장합니다. `ratio≈1` 하나를 성공으로 정의하지 않습니다. 큰 틀의 기후장을 외우는
예측도 전지구 온도 RMSE가 작을 수 있으므로 anomaly/변화량/확률 score를 함께 봅니다.

| 측정 | 정의 / 해석 |
|---|---|
| State RMSE | 변수별 area-weighted physical RMSE, 기존 normalized 전체 지표도 별도 보존 |
| ACC | train-only day-of-year/hour climatology를 뺀 prediction/true anomaly의 면적 가중 비중심 cosine. norm=0이면 N/A. 전체-period mean은 seasonal ACC라 부르지 않음 |
| Tendency RMSE | 동일 j, dt의 `(vhat-v*)/b_c` 및 변수별 physical RMSE |
| 진폭 비 | `||A_c vhat||/||A_c v*||`; true norm이 train 기반 threshold 이하이면 N/A와 유효 count |
| Mean vs member | mean tendency와 member tendency 크기 분포를 따로 비교. mean의 비율 1을 강제 목표로 두지 않음 |
| 확률 score | marginal state CRPS/Energy, pair joint Energy, scalar increment CRPS; 학습 fair/평가 empirical estimator 이름과 M 기록 |
| 시간 상관·PSD | variable별 anomaly의 고정 지역/area summary, detrend/window 방법·주파수 단위 명시. 30일 영상 한 개로 장기 spectrum을 추정하지 않음 |
| Lag | physical-hour lag 진단만; 정답 이동 후 점수를 성능으로 보고하지 않음 |
| Ensemble | state 및 increment spread, coverage/rank, 동일 member lag covariance. 다른 member의 시점들을 이어 붙이지 않음 |
| 전문화 | validation 지역별 candidate skill, gate-best agreement, counts, usage, cosine. 마지막 lead만 아니라 여러 lead audit |
| Baselines | persistence, train-only climatology, AE-reconstructed truth 차분, 동일 A의 기존 B/C |

신뢰구간은 cell 수를 독립 표본 수로 쓰지 않고 겹치지 않는 origin 기간을 block bootstrap합니다.
실제 데이터 계절 범위가 좁으면 추가 계절의 일반화 평가는 미검증으로 보고합니다.
현재의 test 기반 expert audit 그림은 이미 열람한 결과이므로 반복 튜닝에 사용하지 않습니다.
새 validation audit를 추가하고 최종 test 반복 사용 이력을 기록합니다.

## 8. 단계별 대조 실험과 채택 조건

| 실험 | 바뀌는 한 가지 | 목적 |
|---|---|---|
| E0 | 현재 a3003d1 재현: sampled_leads=2, 기존 `_sample` | 재현 baseline; 실제 run metadata 없으면 기존 첨부와 동일 실험이라고 주장하지 않음 |
| E1 | `_sample`도 member noise를 lead 사이에 공유 | coupling mismatch 효과만 측정 |
| E2 | E1 + L_pair, 고정 full-horizon sampling | dynamics supervision 자체의 효과 |
| E3a | E2 + scalar magnitude CRPS | 진폭 보조의 효용과 부작용 |
| E3b | E2 + P=3~4 block score로 확장 | 더 긴 coupling; E3a와 별개 실행 |
| E4 | 선택된 최소 loss + sampling 또는 optimizer 보정 하나 | 과적합/데이터 단조로움 개선 |

같은 A hash, archive hash, split, 최초 seed, member 수, 적분 step, validation panel과
최대 optimizer update 수를 비교합니다. 성능뿐 아니라 wall time/메모리/field evaluation 수도
보고합니다. Loss 추가로 RNG 소비가 달라지는 문제를 피하려면 FM source/tau, pair sampler,
ensemble noise를 독립 RNG stream으로 관리합니다. 최종 후보는 최소 여러 seed로 재검사합니다.

채택 조건은 validation의 pair Energy·변수별 tendency 오차 개선, marginal score/state 오차의
허용 범위 유지, increment coverage/spread의 심각한 악화 없음입니다. 허용 범위는 E0의
seed/origin 변동을 보고 **test를 열기 전에** 확정합니다. 모든 lead에서 변화량을 키우는 것,
candidate cosine을 낮추는 것, train loss만 낮추는 것은 단독 성공 기준이 아닙니다.

이번에 제공하는 [probe](probe_design.py)는 같은 marginal인데 시간 coupling이 다른 예측을
구별하는지, memberwise MSE의 분산 벌점, 단위/차분 계약을 검사합니다. ERA5 학습이나
새 PyTorch loss의 backward 성공을 대신하지 않습니다. 실제 구현 acceptance test와 실행
순서는 [RUNBOOK](RUNBOOK.md)에 있습니다.

## 9. 후속 출력: 12시간 trajectory 뷰

학습 관측 간격은 우선 **6h를 유지**하고 표시/진단용 출력에서 +12,+24,…,+720h를 고릅니다.
현재 `[M,120,D]` 예측의 lead index `1,3,…,119`가 60개 미래 시점입니다. 관측 origin을
별도 anchor로 붙이면 61개 시점이며, 모든 시점에서 member 번호를 그대로 유지합니다.
Checkpoint의 `step_hours=6`, `H=120`, `s=j/120`을 12h/60으로 바꾸거나 +6h를 +12h로
다시 이름 붙이지 않습니다. 저장 metadata에서 model/source interval과 output interval을 구분합니다.

12h tendency는 선택된 양 endpoint와 실제 timestamp에서 `(x(t+12h)-x(t))/12h`로 계산합니다.
시간 간격을 늘린 그림은 변화를 보기 쉽게 할 수 있지만 dynamics를 학습시키거나 시간 지연을
자동으로 고치지는 않습니다. 6h 원본 진단을 병행해 빠른 변화가 downsampling으로 가려지는지 봅니다.

예정 출력은 (1) ERA5/mean/동일 member 기상장과 12h 차분, (2) 변수별 고정 지역 anomaly
시계열과 member 구간, (3) train-only EOF 또는 기존 고정 latent projection 위 시간색 trajectory입니다.
ERA5와 예측은 같은 projection을 사용하며 test에 새 PCA를 fit하지 않습니다. 이는 기상 **상태의
시간 경로**이지 바람 화살표로 만든 공기 입자 경로나 태풍 중심 경로가 아닙니다.
현재 `time_alignment`에는 12h 출력 선택 flag가 없습니다. 이 절은 후속 구현 계약입니다.
