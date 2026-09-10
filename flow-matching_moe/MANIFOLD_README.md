# 최종 구조: Physics-informed Manifold MoE Ensemble

**2026-09-10:** 모델 구조를 유지한 dynamics loss / 120h 전면 재학습 / 각 member 영상은
[RETRAIN_120H.md](RETRAIN_120H.md)를 사용합니다. `scripts/smoke_temporal_moe.py`가 새 loss까지
활성화한 합성 A/B/C 재현 진입점입니다. 아래는 기반 manifold architecture 설명입니다.

진입점은 **`train-climate-manifold-moe`**입니다. 기존 `train-climate-moe`/`smoke_moe.py`는
이전 AE64 + Meta160 모델을 실행합니다. 최종 구조 재현에는 **`smoke_manifold_moe.py`**를
사용하세요. 이전 코드·checkpoint·실험은 보존하며 자동 변환하지 않습니다.

- **[처음부터 따라 하는 학습 README: 설치 → 데이터 → A/B/C → 영상 → 평가 → 추론](TRAINING_README.md)**
- [설계와 A/B/C 학습 Mermaid](../struct-picture/06-manifold-training.md)
- [접공간 투영과 member별 추론 Mermaid](../struct-picture/07-manifold-inference.md)
- [실제 합성 학습 결과·한계](../docs/results/manifold-smoke/README.md)

## 1. 설계: 왜 지역 prior와 tangent projection을 함께 쓰는가

모든 expert에 동일한 FM 정답만 주면 같은 함수를 학습하는 것이 합리적인 해가 됩니다.
Projection만 추가해도 이 대칭성은 사라지지 않습니다. 이번 구조는 역할을 분리합니다.

1. **PI-AE**는 재구성·물리 진단·물리 거리·인접 관측 dynamics를 이용해 상태 좌표를 만듭니다.
2. Train 좌표에서 고정한 **국소 영역 prior**가 각 expert의 담당 영역을 정합니다.
3. **영역 prior + 실제 expert 오차의 responsibility**가 담당 영역의 학습 비중을 높입니다.
4. **Decoder Jacobian projection**은 각 full-state 후보를 같은 decoder 접공간에 맞춥니다.
5. 국소 gate로 후보를 합친 **하나의 intrinsic ODE**가 member별 최종 기상장을 생성합니다.

여기서 영역은 지구상의 공간 구획이 아니라 **전체 기상 상태의 잠재공간 영역**입니다.
한 expert가 특정 변수만 출력하지 않습니다. K개 expert 모두 모든 변수를 출력합니다.
K는 전문가 수, M은 독립 초기 noise로 생성하는 ensemble member 수입니다.

첨부 `Final_Physics_Informed_Manifold_MoE_Ensemble_Flow_Matching(2).txt`의 선택지 중
**PI-AE + 초기 noise ensemble**을 구현했습니다. FNO/Wavelet/DeepONet을 동시에 추가하거나
categorical expert sampling을 추가한 구조는 아닙니다. 이전 Meta160의 자유로운 full-field
residual 대신 **국소 gate 자체를 fusion 모델**로 사용해 지역 책임을 우회하지 못하게 합니다.
64는 expert 내부 bottleneck, 160은 기본 gate correction MLP 폭입니다. 새 manifold의
intrinsic 차원 `r`은 별도이며 기본 16입니다. Smoke는 `r=6`, gate 폭 64를 사용했습니다.

## 2. 좌표·DCT·엄밀한 projection의 범위

원자료 기상장 `x`는 expert train 구간에서만 계산한 변수/격자별 mean/std로 표준화합니다.
아래 `x`와 velocity는 그 **표준화된 기상장 좌표**입니다. Pa/K/m/s는 최종 역정규화 후 단위입니다.

| 기호 | Shape | 의미 |
|---|---|---|
| x | B × D | D=C×Y×X, 선택한 전체 변수와 격자 |
| E(x), q | B × r | PI encoder 출력과 train 통계로 표준화한 intrinsic 좌표 |
| h | B × context_dim | 과거 E(x) 시계열의 시간 DCT → history MLP |
| J | B × D × r | 현재 q에서 decoder의 Jacobian |
| v_raw | B × K × D | 각 expert의 별도 velocity head → 공간 IDCT |
| a | B × K × r | 투영 후보를 표현하는 intrinsic velocity |
| pi | B × K | 현재 상태·history·두 시간 조건의 simplex fusion weights |
| forecast | M × H × D | 최종 역정규화 ensemble |

공간 DCT는 각 변수의 Y, X 축에 orthonormal DCT-II를 적용합니다. 시간 DCT는 각 history
관측을 PI encoder로 인코딩한 뒤 history 축 L에 적용합니다. 변수 축 변환이나 주파수 절단은
없습니다. `Q[k,n]=a_k cos(pi k(n+1/2)/N)`, `a_0=sqrt(1/N)`, 나머지 `sqrt(2/N)`이며
역변환은 `Q.T`입니다. **각 후보 velocity의 IDCT는 projection/fusion 직전**입니다.
Decoder도 최종 기상장 복원을 위해 공간 IDCT를 별도로 사용합니다.

```math
q=\frac{E(x)-\mu_z}{\sigma_z},\qquad \bar D(q)=D(\mu_z+\sigma_z q),\qquad
J(q)=\frac{\partial\bar D(q)}{\partial q}.
```

W는 면적에 비례하는 양의 대각 가중치입니다. Lat/lon 간격과 cos(latitude)를 반영하며
표준화 좌표에서 사용합니다. 차트 거리 prior는 표준화 q의 Euclidean 거리입니다.
PI metric loss가 물리 거리와 latent 거리를 근사하도록 유도하지만, 이것이 정확한 지구
물리 metric 또는 전역 geodesic 거리라는 보장은 없습니다.

```math
G=J^T WJ,\quad
a_k=(G+\epsilon I)^{-1}J^T Wv_{\mathrm{raw},k},\quad
v_{\mathrm{tan},k}=Ja_k,\quad
\epsilon=\text{ridge}\cdot\max(\operatorname{mean}\operatorname{diag}G,10^{-8}).
```

각 후보마다 위 계산을 적용하되 **현재 지점의 J와 접공간은 공통**입니다. 서로 무관한
expert별 manifold를 학습해서 합치는 구조가 아닙니다. Decoder에 latent velocity를
state처럼 넣지 않습니다. `torch.func.jacfwd` + `vmap`으로 J를 계산합니다.

J가 full column rank이고 epsilon=0이면 W-orthogonal tangent projector입니다.
실제 학습은 ridge>0의 안정화된 최소제곱이므로 출력은 J의 span에 속하지만 **정확히
idempotent인 projector는 아닙니다**. PI-AE decoder image의 injectivity나 regularity도
증명하지 않았습니다. 본 구현을 엄밀한 대기 해공간 전체의 manifold 발견으로 해석하면 안 됩니다.

전체 D차원 Gaussian에서 rank-r tangent-only ODE를 시작하면 normal 성분을 운반하지
못하는 문제가 생깁니다. 따라서 **초기 noise와 최종 ODE를 r차원 q에 정의**합니다.
물리 state는 항상 decoder로 복원합니다. 이는 decoder image 밖의 기상 변동을 직접
표현할 수 없다는 압축 한계를 가지므로 reconstruction 오차를 별도로 측정합니다.

## 3. Local responsibility와 두 시간축

Stage A 선택 모델의 train 좌표만으로 deterministic farthest-first + Lloyd 중심 `c_k`를
맞춥니다. 중심이 중복되면 fail-fast합니다. 좌표 평균/scale·중심·radius는 B/C에서 고정합니다.
`R²`는 최근접 중심까지 mean-square distance의 train 평균입니다.

```math
\ell_k(q)=-\lambda_{loc}\,\operatorname{mean}(q-c_k)^2/R^2,\quad
p_k(q)=\operatorname{softmax}(\ell/T_g),\quad
\pi_k=\operatorname{softmax}((\ell+b\tanh f_\psi(q,h,\tau,s))/T_g).
```

기본 correction bound b=0.5는 condition MLP가 먼 영역의 prior를 무제한 덮어쓰지 못하게
합니다. Soft overlap은 경계에서 허용합니다. 담당 영역은 학습된 태풍/제트류 이름이 아닙니다.

- **Physical lead s**: `(lead_index+1)/H`. 6h 데이터에서 H=120은 +6h…+720h=30일입니다.
- **생성 flow time tau**: noise에서 해당 lead 상태로 가는 `[0,1]`. 대기의 물리 시간과 다릅니다.
- **A의 latent dynamics**: 인접한 실제 관측을 `step_hours/24` 일 간격으로 예측하는 auxiliary loss.

```math
q_\tau=(1-\tau)z+\tau q_s,\quad u=q_s-z,\quad z\sim N(0,I_r),\quad
r_k=\operatorname{stopgrad}\left[\operatorname{softmax}(\log p_k-\|a_k-u\|^2/T_r)\right].
```

FM target의 encoder gradient는 끊습니다. 첨부의 error-only responsibility에 **고정 local
prior를 더한 것**은 한 expert가 전역적으로 모든 표본을 차지하는 현상을 줄이기 위한 설계 결정입니다.
Gate는 `CE(r,pi)`로 학습하고 expert는 `sum r_k MSE(a_k,u)`로 학습합니다.
공통 대기 dynamics까지 서로 직교시키지 않습니다. Cosine>0.95인 후보에만 작은 overlap-weighted
penalty를 주며, spread를 무한히 증가시키는 음의 분산 보상은 없습니다.

## 4. A → B → C → validation 영상 → 최종 test 평가

| 단계 | 학습 대상 | 목적함수 / 선택 |
|---|---|---|
| A manifold | PI encoder/decoder/latent drift | rec + .1 physics + .05 invariant proxy + .1 metric + .1 real-time dynamics; PI validation 최소 |
| B specialize | experts + gate + history MLP, PI manifold 동결 | fused FM + responsibility FM + .2 gate CE + .05 balance + .001 weak diversity + .05 projection + .01 entropy guard |
| C joint | B 모듈 + PI manifold, reference encoder 고정 | B loss + .5 Energy + .5 CRPS + .5 PI loss + 1 anchor + .01 spread band; 전체 LR×.1, PI LR 추가×.1 |

B/C 선택 지표는 고정 validation noise와 lead sampling의 `Energy+CRPS`입니다. C의
reference-encoder anchor는 A의 영역 좌표를 유지하도록 제한합니다. 정답 좌표 detach만으로
geometry collapse가 해결된다고 가정하지 않습니다. Reconstruction/metric/anchor도 함께 유지합니다.

Ensemble loss는 **differentiable midpoint ODE의 실제 endpoint**에서 계산합니다. Energy/CRPS는
기존 off-diagonal fair estimator, Energy는 `sqrt(D)`로 정규화합니다. Spread band는 표준화
좌표 표준편차가 `[.02,3]` 밖일 때만 제곱 penalty입니다. 원하는 spread 수치를 정답으로 쓰지 않습니다.

기존 five-way split을 재사용합니다: `train → expert_validation → calibration → validation → test`.
인접 구간의 future targets가 겹치지 않게 최소 H−1 window를 purge합니다. A/B는 train으로
학습하고 expert_validation으로 선택합니다. C는 calibration으로 학습하고 validation으로
선택합니다. Test는 최종 평가만 사용합니다. A도 validation과 겹치지 않는 인접 관측 pair를 씁니다.
Normalization, physics scales, latent scales, chart centers, PCA는 train에서만 fit합니다.
B/C는 동일 archive SHA/schema를 요구하고 학습·추론·평가가 같은 통계를 재사용합니다.

### 실행 명령은 단계별 학습 README를 사용하세요

설치·archive 검사부터 A/B/C·시간 비교 영상·평가·추론까지의 명령은
**[TRAINING_README.md](TRAINING_README.md)**에서 순서대로 실행합니다.
해당 안내는 B/C에 `--sampled-leads 2`를 명시해 연속 lead의 FM source noise를 공유합니다.
이 처리는 별도의 `Δstate/Δtime` trajectory loss를 추가한 것은 아닙니다.

| 현재 단계 | 입력 → 출력 | 다음 단계 전 확인 |
|---|---|---|
| A manifold | 고정 archive → `stage_a.pt` | PI validation, reconstruction/physics/metric |
| B specialize | 같은 archive + A → `stage_b.pt` | FM/expert loss, gate 사용률, expert_validation Energy/CRPS |
| C joint | 같은 archive + B → `final.pt` | validation Energy/CRPS, anchor/PI와 ensemble spread |
| Validation 영상 | C + 과거 validation origin → NPZ/MP4/JSON | valid time, u/v, 변화량/hour, mean과 member 차이 |
| 최종 평가 | 설정 확정 후 C → test 지표·전문화 그림 | local/uniform 비교, 영역별 expert 오차 |

현재 `manifold_diagnostics`는 test split을 사용하므로 최종 평가에 배치했습니다.
PCA의 색 분리만으로 전문화를 주장하지 않습니다. Teacher-forced FM 오차와 정답을 보지 않는
생성 ODE의 usage/cosine/spread는 서로 다른 측정입니다. 전문화 진단은 최종 physical lead
표본에 한정되며, 모든 lead의 전문화를 입증하지 않습니다.

입력은 같은 lat/lon grid의 2D fields, `msl` Pa, `t2m` K, `u10/v10` m/s입니다.
원자료의 변수 attrs는 archive schema에 보존하지만 단위 검증·변환은 자동 수행하지 않습니다.
격자는 단조롭고 각 축 2칸 이상이며 정확한 극점을 제외해야 합니다. 관측 mask나 연속 시간
계약을 만족하지 못하면 fail-fast합니다. 상층 pressure-level 축은 자동 지원하지 않습니다.

B/C는 직전 phase의 설정과 같은 archive SHA/schema를 계승합니다. `--stage all`은 A/B/C를
자동 연결하며, 단독 실행의 metrics는 해당 phase만 담으므로 전체 학습 그림에는 세 파일을
합칩니다. Stage 전환 재로드는 지원하지만 optimizer/RNG를 복원하는 exact resume는 없습니다.
Checkpoint는 model/schema/normalization/geometry/physics 통계/phase와 manifest SHA를 저장합니다.
기존 weather adapter의 `rollout(initial_state,horizon_hours)`에서도 읽을 수 있습니다.

## 5. 물리 제약·계산량·이후 보정 범위

Physics loss는 지구 반경과 실제 lat/lon 간격으로 계산하는 divergence/vorticity, 기압·온도
기울기, specific kinetic energy의 **동시각 재구성 오차**입니다. 면적평균 msl/온도/KE도
같은 시각의 proxy를 비교합니다. Surface msl을 대기의 정확한 기둥 질량으로 취급하지 않습니다.
대기를 비압축으로 가정해 divergence=0으로 만드는 항도 없습니다. 상층 구조·연직 속도·forcing·
수지 경계가 없으므로 primitive equations의 PDE residual이나 엄밀한 질량/에너지 보존을
구현했다고 주장하지 않습니다.

Fixed-step midpoint는 step당 2번 전체 K expert와 Jacobian을 평가합니다. 한 평가의 J는
`effective_batch × D × r`, Gram은 `effective_batch × r²`입니다. C의 ensemble 경로에서는
effective_batch=`batch × sampled_leads × M`이며 역전파에 여러 step의 graph도 필요합니다.
학습은 lead를 sampling하므로 매 batch에서 120 lead를 모두 펼치지 않습니다. 추론은 lead별로
solve합니다. 현재 dense MLP·명시적 Jacobian·전체 archive RAM 로드를 사용해 native ERA5
해상도용으로 검증되지 않았습니다. Physics 통계 fit은 train span을 한번에 device에 올립니다.
먼저 coarse grid와 작은 batch로 확인하고 필요하면 batch/M/r/적분 step을 조정하세요.
GPU 성능은 측정하지 않았습니다.

실측 smoke는 D128/r6/K3, 약14.3만 parameter, CPU1thread, A50+B40+C10 및 평가 포함 약36초,
process peak RSS 약427MiB입니다. 이를 실데이터나 GPU 요구량으로 비례 환산할 수 없습니다.

같은 member의 expert는 현재 q를 공유하며 member끼리는 독립 noise를 사용합니다. Lead 사이에
같은 초기 noise를 재사용하지만 학습은 leadwise marginal입니다. **학습된 joint physical-time
trajectory 분포·15～30일 예측 skill·불확실성 보정은 미검증**입니다. 기존 MoE보다 반드시 정확하지
않습니다. 관측 embedding의 PCA뿐 아니라 생성 endpoint·reconstruction·calibration도 확인하세요.
Rank histogram은 상관된 cell/lead를 합산하므로 독립 표본의 신뢰구간을 제공하지 않습니다.

이번 구조를 고정하고 이후 데이터·loss 계수·차원·온도·ridge·학습률·적분 정밀도를 보정하면서
held-out 비교를 진행하는 기준입니다. 새로운 모델 구성요소를 추가하는 것을 전제하지 않습니다.

| 보정 대상 | 코드 위치 / 인자 |
|---|---|
| PI 진단·단위·면적·물리 거리 | `src/climate_diffusion/manifold_physics.py`: `SurfacePhysics` |
| AE 차원·재구성·latent dynamics | `manifold_moe.py`: `PhysicsManifoldAE`, `manifold_loss`; `--manifold-dim` |
| 투영 안정성 | `manifold_moe.py`: `tangent_lift`, `jacobian`; `--projection-ridge` |
| 영역 선명도와 책임 배분 | `LocalManifoldGate`, `specialization_loss`; `--gate-temperature`, `--responsibility-temperature`, `--locality-weight` |
| Stage 전환·freeze·LR·Ensemble loss | `train_manifold_moe.py`: `train_manifold_moe`, `_epoch` |
| 전문화·생성 경로·PCA 진단 | `manifold_diagnostics.py`: `diagnose_manifold` |
| 공통 추론·평가·adapter | `inference.py`, `evaluation.py`, `weather_adapter.py` |

검증 범위는 DCT roundtrip, 투영/Jacobian 수치와 gradient, simplex/locality, member coupling,
A→B freeze, C joint gradient, 단독 phase reload, checksum, CLI/adapter, 시간 분할·mask 계약,
synthetic end-to-end 학습입니다. 실ERA5 장기 학습·RunPod 실행·GPU memory/throughput은 미실행입니다.
