# Full-state Flow Matching MoE

**물리시간 재귀 Flow 신규 경로:** [RECURRENT_TRAINING_MANUAL.md](RECURRENT_TRAINING_MANUAL.md) — drift+residual FM, full120h, member loss와 진단. 기존 checkpoint를 자동 전환하지 않습니다.

**전체 실행 매뉴얼:** [TRAINING_MANUAL.md](TRAINING_MANUAL.md) — 환경·ERA5 준비부터 새 A→B→C, validation/test, member별 영상까지 순서대로 실행합니다.

**새 dynamics loss + 120시간 trajectory + member별 영상:**
[RETRAIN_120H.md](RETRAIN_120H.md)의 처음부터 재학습 순서를 사용하세요.
실제 구현·합성 A/B/C 실행·비교 로그는 [검증 보고](../docs/results/temporal-120h-smoke/README.md)에 있습니다.

**최종 manifold 구조는 [MANIFOLD_README.md](MANIFOLD_README.md)에 있습니다.**
처음 설치하고 실제 데이터를 학습하는 순서는 **[단계별 학습 README](TRAINING_README.md)**를
따라가세요. A → B → C checkpoint 연결과 시간 비교 영상·최종 평가까지 포함합니다.
`train-climate-manifold-moe` / `smoke_manifold_moe.py`를 사용합니다. 아래는 보존한
이전 `train-climate-moe`의 2-stage Meta160 경로이며 최종 manifold 학습 명령과 다릅니다.

`feature/latent-dynamics-flow`의 추가 학습 경로입니다. 기존 dynamics/monthly 모델,
WeatherNext runner, 이전 실험 결과는 유지합니다. 학습 진입점은 **`train-climate-moe`**입니다.
기존 `train-climate-dynamics` 명령은 MoE를 학습하지 않습니다.

소스는 기존 Python package 안의 `src/climate_diffusion/moe.py`, `train_moe.py`,
`moe_data.py`에 있습니다. 이 폴더는 설계·실행 안내입니다.

- [상세 학습 Mermaid](../struct-picture/04-moe-training.md)
- [추론·저장 모델 Mermaid](../struct-picture/05-moe-inference.md)
- [실제 smoke 결과와 해석](../docs/results/moe-smoke/README.md)

## 1. 구현한 구조와 설계 해석

Expert는 **변수별 모델이 아닙니다**. 모든 expert가 archive의 전체 기상장을 입력받고
전체 기상장의 conditional vector field를 출력합니다. Regime/mode에 대한 분업은
router-weighted FM 학습으로 유도하지만, 특정 expert가 특정 기상 regime을 담당한다는
명명이나 보장은 하지 않습니다. Dense MoE이므로 매 평가마다 모든 expert를 실행합니다.

설계 원문의 IDCT 위치 충돌은 상세 bullet인 **“Meta Learner 직전”**을 우선했습니다.
좌표계가 섞이지 않도록 다음을 분리했습니다.

| 구성 | 실제 좌표·shape | 의미 |
|---|---|---|
| Archive state | `[B, D]`, `D=C×Y×X` | 선택한 모든 변수와 격자, train 통계로 표준화 |
| History | `[B, L, D]` | origin 이하의 관측만 사용 |
| History DCT | 시간축 `L`의 orthonormal DCT-II | 변수·격자 순서는 유지; flatten → context MLP → `h` |
| Expert 입력 state | 공간축 `Y, X`에 변수별 DCT-II | 모든 expert에 같은 현재 `X_tau`와 조건 전달 |
| Expert AE | bottleneck **64** | 조건부 state encoder + reconstruction head |
| Expert velocity head | `[B, D]`, 공간 DCT 좌표 | AE state decoder와 **별도 파라미터**; velocity 정답에 직접 감독 |
| IDCT | 각 후보 velocity의 `Y, X` 역변환 | Meta 입력 직전 표준화된 물리 격자 좌표로 복귀 |
| Meta AE | bottleneck **160** | `[X_tau, v_1,...,v_K, h, tau, lead]` 인코딩 |
| Meta 출력 | `alpha: [B,K]`, `r: [B,D]` | softmax simplex + bounded residual |
| 최종 ODE state | `[B×M,D]` | **64/160 차원의 latent가 아닌** 전체 표준화 state |
| 최종 forecast | `[M,H,D]` | 역정규화된 원래 변수 단위, schema로 xarray 복원 |

64와 160은 **서로 다른 네트워크 내부 압축 폭**입니다. 두 latent를 직접 더하지 않습니다.
비선형 state decoder에 latent velocity를 넣어 물리 velocity라고 해석하지 않습니다.
각 expert의 별도 velocity head가 선형 IDCT를 거쳐 공통 좌표의 vector field를 생성합니다.
Reconstruction head는 auxiliary loss용이며 추론 출력 decoder가 아닙니다.

DCT 행렬은 `Q[k,n]=a_k cos(pi*k*(n+1/2)/N)`,
`a_0=sqrt(1/N)`, 나머지는 `sqrt(2/N)`입니다. 역변환은 `Q.T`입니다.
계수를 자르지 않으며 변수 축에 DCT를 적용하지 않습니다. 공간 IDCT는 history의 시간 DCT를
역변환하는 연산이 아니라 **expert가 출력한 공간 주파수 velocity**의 역변환입니다.
공간 DCT의 경계 가정은 구면 대기의 물리 경계조건을 보장하지 않습니다.

## 2. 두 시간축과 member별 적분

- **Physical lead**: origin에서 몇 시간 뒤인가. 6h archive에서 lead index `j=0`은 +6h,
  `j=119`는 +720h=30일. 네트워크 조건은 `s=(j+1)/H`와 sinusoidal embedding입니다.
- **Flow time `tau`**: Gaussian noise에서 해당 physical lead의 기상장 분포로 이동하는
  생성 시간 `[0,1]`. 단위는 일/시간이 아닙니다.

학습 pair는 표준화 정답 `y_s`, `z ~ N(0,I)`로 구성합니다.

```math
X_\tau=(1-\tau)z+\tau y_s, \qquad u_\tau=y_s-z.
```

각 ODE 평가에서 동일 member의 현재 state로 모든 expert를 실행합니다.

```math
\alpha_k^{(m)}=\operatorname{softmax}_k G_\phi(X_\tau^{(m)},v_1^{(m)},\ldots,v_K^{(m)},h,\tau,s),
\quad v_{\rm final}^{(m)}=\sum_k\alpha_k^{(m)}v_k^{(m)}+2\tanh r_\phi^{(m)}.
```

`integrate()`는 fixed-step explicit midpoint solver입니다. 한 step에 2회의 vector-field
평가를 하며 각 평가에서 gate도 다시 계산합니다. **Expert별 ODE endpoint를 따로 생성한 뒤
평균하지 않습니다.** Member끼리는 독립 noise를 쓰고, 동일 member 안에서는 noise/current
state가 공통입니다. History normalization과 context도 공통입니다.

이 MoE v1은 **physical lead에 조건화된 생성 ODE**입니다. 기존 `dynamics.py`의 물리 시간
latent ODE를 내부에 다시 넣은 모델은 아닙니다. 여러 lead의 추론은 같은 member 초기 noise를
재사용하지만, lead별 조건부 분포를 독립적인 ODE solve로 계산합니다. 따라서 **학습된 joint
시간 경로 분포, 시간 일관성 또는 보존 법칙을 보장하지 않습니다**. 이것이 필요하면
trajectory-level loss와 lead를 연결하는 dynamics 조건 경로를 추가해야 합니다.

## 3. 두 단계 학습과 정답 누수 방지

시간순 window split은 다음과 같습니다.

```text
train → purge → expert_validation → purge → calibration → purge → validation → purge → test
  │                  │                        │                    │                │
expert fitting   expert checkpoint         meta fitting        meta 선택       최종 평가만
```

Held-out 비율은 전체 valid windows의 10%, 15%, 10%, 10%이고 나머지에서 네 개의
purge 구간을 제외해 train을 만듭니다. 각 경계는 최소 `H-1`개 start windows를 제거하므로
**양쪽 future target 시각이 겹치지 않습니다**. 과거 history가 이전 split의 관측을 포함하는 것은
causal forecasting에서 허용됩니다. 분할 이후 `window_stride`로 subsample합니다.
전체 통계가 아닌 **expert train의 history+target raw span만** mean/std에 사용합니다.
Meta 단계·추론·평가에서 같은 mean/std를 재사용하며 meta 단계는 동일 archive SHA/schema를
요구합니다. Validation/test는 optimizer에 전달하지 않습니다.

Stage 1:

```math
L_1=\mathbb E\sum_k g_k\operatorname{MSE}(v_k,u)
 +0.1 L_{\rm expert-rec}+\lambda_{bal}(K\|\bar g\|_2^2-1)
 +\lambda_{div}\mathbb E_{k\ne l}[\max(\cos(v_k,v_l)-0.8,0)^2].
```

각 expert는 독립 파라미터를 가지고 history encoder/router와 함께 optimizer로 학습합니다.
균형 항은 평균 사용률을 조절하고 bounded cosine 항은 velocity 크기를 무한히 키우는
보상이 아닙니다. 다만 정확한 expert들이 동일 target 방향을 학습하면 유사한 출력이 생길 수
있으므로 **diversity 숫자만 낮추는 것이 예측 품질 목표는 아닙니다**.

Stage 2: 선택된 expert checkpoint를 재로드하고 **experts + history encoder + warm-up router**를
동결합니다. 새 optimizer에는 meta 파라미터만 들어갑니다.

```math
L_2=\lambda_{FM}\operatorname{MSE}(v_{final},u)
 +\lambda_{Energy}L_{Energy}+\lambda_{CRPS}L_{CRPS}
 +\lambda_{div}L_{spread-band}+0.05L_{meta-rec}+10^{-4}\|r\|_2^2.
```

기본 가중치는 FM=1, Energy=0.5, CRPS=0.5, balance=0.05, diversity=0.01입니다.
Ensemble loss는 **differentiable ODE를 통과한 실제 생성 endpoint**에 계산합니다.
Energy는 전체 변수·격자의 multivariate score를 `sqrt(D)`로 정규화하고, CRPS는 scalar
좌표별 score를 평균합니다. 둘 다 학습에서는 off-diagonal fair estimator를 사용합니다.
Member 수는 학습 시 최소 2입니다. Spread band는 표준화 좌표의 표준편차가 `[0.02,3]`
범위를 벗어나는 경우에만 제곱 penalty를 줍니다. 보정된 불확실성을 보장하는 항은 아닙니다.

Frozen expert는 `requires_grad=False`지만 ODE 중에 `no_grad()`/`detach()`로 끊지 않습니다.
**Meta → 이전 step state → frozen expert의 입력 Jacobian → 다음 step** gradient는 필요합니다.
검증/test 추론만 gradient 없이 실행합니다. Warm-up router `g`와 최종 fusion `alpha`는
별도 네트워크이며 Stage 2/최종 meta 추론에서 `g`가 `alpha`를 곱해 재가중하지 않습니다.

각 단계 checkpoint는 고정 validation noise/lead 샘플의 **생성 Energy+CRPS 최솟값**으로
선택합니다. 두 validation 구간이 다르므로 단계 사이 loss 높이를 직접 성능 비교하지 마세요.
`--stage meta`는 단계 전환 재로드이지 optimizer/RNG까지 복원하는 exact epoch resume는 아닙니다.

## 4. 설치 및 빠른 재현

기존 환경/checkout을 보존하고 필요한 경우 새 clone에서 실행하세요. GPU 환경은 해당
RunPod 이미지에 맞는 PyTorch를 먼저 준비합니다. 아래 설치는 자원을 새로 구매하지 않습니다.

```bash
git clone --branch feature/latent-dynamics-flow --single-branch \
  https://github.com/nayehyeon61-glitch/climate_diffusion.git
cd climate_diffusion
python -m pip install -e '.[io,test,plots]'
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest -q
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python scripts/smoke_moe.py
python scripts/visualize_moe.py
```

가상환경을 권장합니다. User install 때문에 CLI가 PATH에 없으면
`python -m climate_diffusion.train_moe`, `python -m climate_diffusion.inference`,
`python -m climate_diffusion.evaluation`로 같은 명령을 실행할 수 있습니다.

Smoke는 `outputs/moe-smoke/`에 480개의 합성 6h state, warm-up/meta checkpoint를 만들고,
`docs/results/moe-smoke/`의 JSON 및 `docs/figures/moe-smoke/`의 SVG/PNG를 재생성합니다.
같은 경로의 재실행은 그 smoke 산출물을 덮어씁니다. 실험 보존 시 `--work-dir`, `--report-dir`,
plot의 `--metrics`, `--summary`, `--output-dir`에 새 경로를 지정하세요.

## 5. ERA5 archive와 RunPod 명령

현재 MoE archive 지원 범위는 **동일 lat/lon grid의 완전 관측된 2D 변수들**입니다.
`full state`는 선택한 archive 전체를 뜻하며 WeatherNext의 모든 상층 변수/pressure level이
자동 추가된다는 뜻이 아닙니다. Integrated IBTrACS 표 또는 추가 pressure-level 축은
현재 명시적으로 거부합니다. Pressure level별 필드를 별도 2D 변수로 준비하거나 향후
`field_grid()`/DCT/schema 계약을 확장해야 합니다. 이 실행은 GPT/API/WeatherNext 학습을
호출하지 않습니다.

```bash
prepare-climate-fixed-step-data \
  --fields /workspace/data/era5_surface_6h.zarr \
  --variables msl t2m u10 v10 \
  --step-hours 6 --target-lat-points 18 --target-lon-points 36 \
  --output /workspace/data/era5_moe_6h.npz
```

실제 존재하는 ERA5 파일 경로로 바꾸세요. 연속된 충분한 장기 기록이 필요합니다. Archive의
`observed_mask`가 없거나 0인 셀이 있으면 학습을 중단합니다. 과거 zero-filled archive를
그대로 학습하지 말고 원자료에서 결측 처리 정책을 정해 다시 만드세요. 현재 검사 대상은
coarsening 후의 archive cells입니다. Inference adapter도 **같은 평균 pooling**을 적용하고
저장된 격자 좌표가 정확히 일치해야 통과합니다. 임의 보간으로 다른 전처리를 섞지 않습니다.
Pooling 후 비유한 값이 남으면 거부합니다. 원자료와 다른 grid를 쓰려면 일치하는
전처리/관측 정책을 먼저 준비하세요. 기존 non-MoE adapter의 보간 동작은 보존합니다.

처음에는 작은 batch/member/solver step으로 메모리를 확인합니다. 아래 예는 입력 6개를
하루 간격으로 선택하고 전체 +6h~+30d를 학습합니다. 기존 기본 `history_stride=120`은
30일 간격, 150일 lookback이므로 의도에 맞게 명시적으로 선택하세요.

```bash
train-climate-moe \
  --archive /workspace/data/era5_moe_6h.npz \
  --output /workspace/experiments/moe-v1/moe.pt \
  --stage all --history-steps 6 --history-stride 4 --horizon-steps 120 \
  --num-experts 4 --expert-latent-dim 64 --meta-latent-dim 160 \
  --hidden-dim 256 --context-dim 128 \
  --expert-epochs 50 --meta-epochs 20 --learning-rate 0.0001 \
  --batch-size 2 --window-stride 48 --expert-leads 4 --meta-leads 1 \
  --ensemble-size 4 --integration-steps 4 --max-validation-windows 32 \
  --device cuda --seed 7
```

`meta-leads=1`은 매 batch에서 origin별 lead 하나를 무작위 샘플한다는 뜻이지 마지막 lead만
학습한다는 뜻이 아닙니다. 필요한 15~30일 결과는 6h 기준 lead +360h~+720h에서 선택합니다.
Data가 짧으면 purge를 끄지 말고 더 긴 archive를 준비하거나 실험 horizon을 줄이세요.

두 단계를 따로 실행하려면:

```bash
train-climate-moe --archive /workspace/data/era5_moe_6h.npz \
  --output /workspace/experiments/moe-v2/experts.pt --stage experts \
  --history-steps 6 --history-stride 4 --horizon-steps 120 \
  --expert-epochs 50 --batch-size 2 --expert-leads 4 --device cuda

train-climate-moe --archive /workspace/data/era5_moe_6h.npz \
  --output /workspace/experiments/moe-v2/meta.pt --stage meta \
  --init-checkpoint /workspace/experiments/moe-v2/experts.pt \
  --meta-epochs 20 --batch-size 2 --meta-leads 1 \
  --ensemble-size 4 --integration-steps 4 --device cuda
```

Standalone meta 단계의 model dimensions는 warm-up checkpoint에서 읽습니다. 서로 다른
구성이나 archive로 바꾸면 오류가 납니다. 각 `.pt` 옆에 `.manifest.json`, `.metadata.json`,
`.metrics.json`이 저장됩니다. Checkpoint에는 모든 expert/router/history/meta 및 정규화,
schema, split, SHA, 선택 epoch가 들어갑니다. 기존 monthly/dynamics checkpoint를 이 모델로
부분 로딩하거나 자동 fine-tuning하지 않습니다. **MoE expert는 새로 학습합니다.**

## 6. 추론·평가·그래프

```bash
forecast-climate-flow --checkpoint /workspace/experiments/moe-v1/moe.pt \
  --archive /workspace/data/era5_moe_6h.npz --forecast-steps 120 \
  --ensemble-size 8 --integration-steps 32 \
  --output /workspace/experiments/moe-v1/forecast.npz

evaluate-climate-flow --checkpoint /workspace/experiments/moe-v1/moe.pt \
  --archive /workspace/data/era5_moe_6h.npz \
  --ensemble-size 8 --integration-steps 32 --max-cases 32 \
  --moe-mode meta --device cuda --output /workspace/experiments/moe-v1/eval-meta.json

python scripts/visualize_moe.py \
  --metrics /workspace/experiments/moe-v1/moe.metrics.json --summary '' \
  --output-dir /workspace/experiments/moe-v1/figures
```

`--moe-mode experts`는 저장된 warm-up router의 가중 vector field, `uniform`은 동일 expert의
균등 vector field, `meta`는 학습된 fusion+residual을 사용합니다. 세 경우 모두 **결합 후 적분**입니다.
기본은 checkpoint stage입니다. Warm-up checkpoint에 meta 추론을 요청하면 오류입니다.
Test 비교는 같은 checkpoint·seed·case 수·member 수·solver step을 사용하세요.
`--max-cases`를 생략하면 저장된 모든 test windows를 평가합니다.

평가 JSON에는 normalized RMSE/MAE/bias, CRPS, Energy, spread, persistence/climatology,
raw-unit 변수별 오차, lead별 RMSE가 들어갑니다. 평가의 확률 score는 기존 계약에 맞춘
**empirical ensemble score**이며, 학습/선택의 fair estimator와 숫자가 그대로 같지는 않습니다.
현재 grid-cell 동일 가중치이며 cosine-latitude 면적 가중 score는 미구현입니다.
`FlowMatchingWeatherRunner`는 이 checkpoint를 자동 인식해 기존 `rollout()` 계약으로 복원합니다.

## 7. 계산 요구와 실행 한계

Smoke 실측: CPU 1 thread, D=128, K=3, M_train=4, midpoint steps=4,
expert/meta latent=64/160, 324,070 parameters. 학습·평가·routing audit 약 **10.2초**,
해당 process peak RSS 약 **407 MiB**. 설치·pytest·plotting 시간은 포함하지 않습니다.
이 수치는 이 실행 환경에 한정되며 GPU benchmark가 아닙니다.

4×18×36, K=4, hidden=256, context=128, history=6의 모델은 **18,043,368 parameters**,
FP32 weight만 약 **68.8 MiB**입니다. Gradients/Adam states/activations는 별도이며
freeze된 weights도 GPU memory를 차지합니다. 이를 최소 GPU 사양으로 해석하지 마세요.
Stage 2는 differentiable solver graph를 보관하므로 대략
`B × sampled_leads × M × solver_steps × K`에 비례해 activation 비용이 늘어납니다.
Midpoint는 solver step당 2번 모든 expert를 호출합니다. 추론은 lead를 순차 처리하지만
전체 비용은 예측 lead 수에 비례합니다. Dense MLP이므로 전지구 0.25° 원해상도 모델은
현실적인 목표가 아닙니다. 먼저 coarse grid에서 검증하고 patch/conv/spectral operator로
확장하세요. 이 구현은 GPU AMP/DDP/activation checkpointing/adjoint를 아직 적용하지 않았습니다.

실제 ERA5 archive, 연결된 RunPod, GPU 장기 학습은 이 작업에서 실행하지 않았습니다.
기존 LFS checkpoint와 실험을 변경하지 않았고 새 toy `.pt`는 Git에 넣지 않습니다.
테스트와 synthetic 결과는 구조·수치 경로의 검증이지 15~30일 예측 성능 인증이 아닙니다.

## 8. 후속 메커니즘 실험 위치

| 바꿀 항목 | 코드 위치 | 확인할 지표 |
|---|---|---|
| Regime router, 온도/entropy/balance | `moe.py: FlowMatchingMoE.field / warmup_loss` | expert별 오차·실사용률·regime별 조건부 skill, 단순 평균 사용률만으로 결론 금지 |
| Fusion gate와 residual 제약 | `moe.py: FlowMetaLearner` | uniform/router-only/meta 비교, alpha 분산, residual 크기 |
| AE 크기와 full-field velocity head | `moe.py: FullStateExpert`, `MoEConfig` | FM 및 reconstruction, 실제 생성 score, 파라미터/메모리 |
| 학습 lead 분포·장기 가중치 | `train_moe.py: _pairs` | lead별 15~30일 skill; target을 history에 넣지 않기 |
| Energy/CRPS·면적 가중·시간 공동 loss | `moe.py: meta_loss / ensemble_scores` | reliability/rank histogram, spread-error, trajectory consistency |
| Differentiable ODE solver | `moe.py: integrate` | step 수에 따른 score/안정성; frozen expert 입력 gradient 회귀 테스트 |
| 결측 지원·pressure-level 채널 | `moe_data.py: field_grid / load_moe_archive / align_moe_grid`, `data.py: vectorize_dataset` | mask-aware loss/encoder와 추론 pooling 계약을 함께 확장 |
| temporal split·정규화 | `moe_data.py`, `train_moe.py` | purge 및 데이터 hash; test로 hyperparameter 조정 금지 |

우선순위는 **독립 validation에서 meta 개선 재현 → 실제 regime 조건부 skill/ablation →
장기 시간 일관성 및 물리 제약**입니다. Smoke의 meta 최적 epoch가 1인 만큼 epoch 수만
늘리기보다 calibration 구간·meta learning rate·regularization을 validation으로 점검해야 합니다.
