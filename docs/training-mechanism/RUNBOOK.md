# 구현 순서와 실험 실행 계획

2026-09-09, 검토 기준 `a3003d1`. **설계 문서입니다. E0 명령만 현재 CLI로 실행 가능하며
새 paired dynamics score/12h trajectory 기능은 아직 구현하지 않았습니다.** 새 GPU 자원을
만들거나 실제 ERA5를 재학습하지 않았습니다. 원본 실험과 checkpoint를 덮어쓰지 않습니다.

## 1. 우선순위: 데이터 → sampler → loss → 평가 → 학습 보정

| 순서 | 현재 파일·함수 | 최소 수정 / 완료 조건 |
|---|---|---|
| 0 | `moe_data.py:load_moe_archive`, `build_moe_split`; `train_manifold_moe.py:_save` | archive/schema/hash, A best hash, five-way split, UTC origin panel 고정. metadata와 실제 weight provenance 확인 |
| 1 | `dynamics.py:TrajectoryWindowDataset.__getitem__`; `moe_data.py` | 기존 3-key 사용자를 보존하는 wrapper; x0 포함 trajectory, 실제 dt, pair mask, 선택 edge 번호. train-only 변수 tendency 통계 sidecar 및 checkpoint buffer |
| 2 | `train_manifold_moe.py:_pairs`, `_sample` | `_pairs`에 group/lead index 보존; `_sample_pairs`에서 `[B,M,r]` noise를 lead축 broadcast. j=0 왼쪽은 관측 고정. 독립 RNG stream |
| 3 | `manifold_moe.py:integrate`, `decode`, `field`, `set_stage` | architecture 변경 없음. 새 sampler에서 기존 differentiable solve 호출. 단계 B parameter freeze를 input gradient detach와 혼동하지 않음 |
| 4 | `moe_losses.py` 또는 작은 training loss helper | endpoint+increment 특징과 metric-aware fair Energy 구현. 기존 `/sqrt(D)` 이중 적용 방지, M>=2와 fully-observed fail-fast |
| 5 | `train_manifold_moe.py:_epoch`, `train_manifold_moe`, `main` | B/C에 새 score, 고정 validation selection, loss/RNG/통계 hash 저장. 처음엔 optimizer/sampling baseline 유지 |
| 6 | `evaluation.py`; `time_alignment.py:temporal_diagnostics`; `manifold_diagnostics.py` | explicit validation split/panel, 변수별 tendency/near-zero mask, joint score, member covariance, 전문화 표본 수. test 기본 audit를 튜닝용으로 호출하지 않기 |
| 7 | 같은 trainer; `scripts/visualize_manifold_moe.py` | E2 이후 LR/weight decay/early-stop/sampling ablation을 하나씩. loss·gradient·유효 pair 수·tendency/spread 그래프 |
| 8 | `time_alignment.py:load_saved_forecast`, `align_forecast`, `render_animation`, `main`; `inference.py` | 별도 후속 작업: 12h 출력 selection과 trajectory plot. 6h 모델 시간 계약 유지 |

미래 target은 supervision에만 씁니다. `origin` 대신 future x_j를 history input에 끼우거나
router 조건에 true tendency를 넣지 않습니다. 신규 문자열 metadata는 현재 batch `.to(device)`
루프에 직접 넣지 않고 dataset-level schema 또는 별도 metadata container로 관리합니다.

## 2. 새 differentiable paired sampler의 의사코드

다음은 **설계 의사코드**이며 현재 import 가능한 API 이름이 아닙니다.

```text
edge j[b] ~ sampler(train partition), j in 0..H-1
history_context = model.context(history)                # causal only
z = independent_normal([B, M, r], ensemble_rng)          # independent members
for endpoint a in {j, j+1}:
    if a == 0: predicted[a] = observed_origin             # no generated origin
    else:
        q0 = z                                          # same member across leads
        s = a / checkpoint.horizon_steps                # fixed H even in curriculum
        q1 = model.integrate(q0, context, s, N_steps)     # solve in tau, with gradient
        predicted[a] = model.decode(q1)                 # normalized state, not velocity
physical = predicted * state_scale + state_mean
tendency = (physical[j+1] - physical[j]) / actual_dt_hours
features = concatenate(weighted endpoints, weighted scaled tendency)
L_pair = fair_energy(features[M], target_features)
loss = old_loss + scheduled_lambda_pair * L_pair
loss.backward()
```

공통 z만으로 joint physical law를 학습한 것은 아닙니다. 새 score가 두 endpoint를 동시에
비교하는 것이 핵심입니다. 배치 일부에서 j=0이면 해당 항만 관측으로 대체하고 나머지 경로의
gradient를 끊지 않습니다. FM branch와 생성 branch를 따로 샘플링해도 seed/RNG 소비를 기록합니다.

Gradient 범위: B는 expert/gate/history에 전달하고 manifold/reference/scale은 고정;
C는 기존 anchor 하에서 manifold까지 전달합니다. Frozen decoder의 `requires_grad=False`는
가능하지만 `torch.no_grad()` 또는 생성 endpoint `.detach()`는 새 loss를 무효화합니다.
ODE 안의 decoder Jacobian과 tangent solve에 필요한 고차 autodiff도 유지해야 합니다.

## 3. 구현 전에 실행 가능한 baseline E0

실제 ERA5 archive와 완료된 동일 계약의 A checkpoint가 있는 기존 RunPod에서 실행합니다.
아래 경로는 **사용자가 보유한 파일로 지정**해야 합니다. A가 없다면
[기존 학습 README의 1~4단계](../../flow-matching_moe/TRAINING_README.md)를 먼저 수행합니다.
현재 baseline은 새 dynamics 감독이 없으므로 epoch를 늘린다고 E2가 되지 않습니다.

```bash
export MANIFOLD_ARCHIVE=/workspace/data/era5_manifold_6h.npz
export A_CHECKPOINT=/workspace/outputs/manifold_run_001/stage_a.pt
export E0_RUN=/workspace/outputs/dynamics-e0-seed7
mkdir "$E0_RUN"

python -m climate_diffusion.train_manifold_moe \
  --archive "$MANIFOLD_ARCHIVE" --output "$E0_RUN/stage_b.pt" \
  --stage specialize --init-checkpoint "$A_CHECKPOINT" \
  --expert-epochs 10 --batch-size 2 --window-stride 4 \
  --ensemble-size 4 --sampled-leads 2 --integration-steps 4 \
  --learning-rate 0.001 --seed 7 --device cuda

python -m climate_diffusion.train_manifold_moe \
  --archive "$MANIFOLD_ARCHIVE" --output "$E0_RUN/final.pt" \
  --stage joint --init-checkpoint "$E0_RUN/stage_b.pt" \
  --joint-epochs 10 --batch-size 2 --window-stride 4 \
  --ensemble-size 4 --sampled-leads 2 --integration-steps 4 \
  --learning-rate 0.001 --joint-lr-factor 0.1 --encoder-lr-factor 0.1 \
  --seed 7 --device cuda
```

이는 **10 epoch 파일럿 설정**이며 기존 40 epoch 실험의 정확한 재현 주장이 아닙니다.
원본 설정/seed/선택 checkpoint가 확보되면 E0/E1/E2 모두 같은 update budget으로 맞춥니다.
`mkdir`가 실패하면 다른 실험 이름을 사용하고 이후 명령을 진행하지 않습니다.
Current C→C resume는 지원되지 않으므로 C 비교는 같은 B에서 다시 시작합니다.

Validation 영상은 기존 학습 README 7단계의 `MANIFOLD_RUN`을 E0 경로로 지정해 만듭니다.
반복 튜닝 중 `climate_diffusion.evaluation`/`manifold_diagnostics`의 test audit를 실행하지 않습니다.
현재 전체 validation 정량 평가 CLI가 없으므로 새 explicit split/panel 기능이 우선입니다.

### 아직 실행할 수 없는 제안 옵션

| 제안 flag/config | 의미 | 현재 지원 |
|---|---|---|
| `--temporal-score pair-energy` | endpoint+increment score | 없음 |
| `--temporal-weight`, `--temporal-warmup-epochs` | 새 score 비중/ramp | 없음 |
| `--paired-member-noise` | `_sample` lead간 noise 공유 | 없음 |
| `--dynamics-stats` | train-only tendency scale provenance | 없음 |
| `--validation-panel`, `--eval-split` | 고정 validation origin/lead | 없음 |
| `--weight-decay`, `--early-stop-patience` | optimizer/중단 정책 | 없음 |
| `--output-interval-hours 12` | 기존 6h 예측의 12h trajectory 뷰 | 없음 |

위 옵션을 현재 명령에 붙이면 오류가 납니다. 구현 뒤 `--help`, config serialization,
checkpoint roundtrip 및 end-to-end smoke 테스트를 통과한 시점에만 실행 README에 추가합니다.

## 4. RTX 4090: 측정 후 늘리는 초기 예산

RTX 4090의 메모리는 24GB입니다. [NVIDIA 공식 사양](https://www.nvidia.com/en-us/geforce/graphics-cards/40-series/rtx-4090/)
이 저장소는 dense full-state head와 명시적 decoder Jacobian을 사용하므로 native 0.25° 전지구
학습이 이 메모리에 맞는다고 보장할 수 없습니다. GPU 실행 시간이 측정되지 않은 상태에서
분/시간 또는 최적 batch를 확정하지 않습니다.

| 항목 | 작은 파일럿 후보 | 이유/주의 |
|---|---|---|
| grid/variables | 기존 archive와 일치하는 18×36, 4개 변수 | D=2592; 새 grid면 A부터 새 실험 |
| r/K | 기존 A의 16/4 유지 | 구조 변경 금지 |
| B/P/M | 2 windows × 2 endpoints × 4 members | 유효 생성 경로 수 최대 16; j=0은 1 endpoint |
| midpoint N | train 4, 검증 8→16 수렴 확인 | 1 solve당 field 호출 2N; 시간 간격과 별개 |
| precision | Jacobian/solve 우선 FP32 | AMP는 jacfwd·solve/backward finite/오차 검사 이후 |
| gradient | 기존 clip norm 1 유지, 항별 norm 기록 | 새 loss가 FM을 압도하거나 사실상 0인지 검사 |

FP32 Jacobian 단독 크기는 `B*P*M*D*r*4 bytes`입니다. 위 조건이면 약 2.53MiB지만, 각
midpoint stage에서 역전파 그래프와 Jacobian 미분·MLP activation·optimizer state가 추가되므로
이 숫자를 전체 VRAM 요구라고 해석하지 않습니다. 같은 식으로 4×721×1440 native grid면
Jacobian 한 묶음만 약 3.96GiB이며 dense head weight도 크게 증가합니다.
PI 통계 fit은 현재 train span 전체를 device에 올리므로 학습 batch보다 먼저 OOM이 날 수 있습니다.
필요하면 동일 수식을 보존하는 streaming train-stat fit부터 구현합니다.

1. CPU 분석 probe → 기존 pytest/synthetic smoke → 새 paired backward smoke 순서.
2. 보유 GPU에서 train 20 update + 고정 validation 소표본을 profile. synchronized wall time,
   `torch.cuda.max_memory_allocated/reserved`, finite gradient, field 호출 수를 기록.
3. 24GB 중 충분한 여유(예: peak 20GB 미만의 출발 목표)가 확인되면 1 epoch 파일럿.
4. 기준을 통과한 설정만 5~10 epoch E0/E1/E2에 사용. 다음 3 seeds와 계절 panel 확장.
5. 메모리 부족 시 먼저 M/적분 그래프 비용을 점검하되 M>=2, P=2는 유지. batch=1은 현재
   A/C metric loss가 0이므로 무조건 권하지 않습니다. gradient accumulation도 batch 내 metric
   pair를 복원하지 못합니다. explicit metric pair buffer 또는 batch>=2/drop-last 계약 필요.

Stage A best의 reconstruction+tendency ceiling이 나쁘면 A를 동일 구조로 먼저 보정합니다.
A를 그대로 쓸 수 있다면 E1/E2는 B부터 진행합니다. A의 scale/centers를 바꾸고 기존 B/C를
그대로 연결하는 것은 허용하지 않습니다. Validation 결과 없이 'C만 다시 하면 된다'고 정하지 않습니다.

## 5. 구현 acceptance tests (아직 미실행)

- 데이터: history 최종 시각=origin, 첫 pair origin→+6h, 미래 j≥1 pair, 마지막 +720h,
  timestamp gap/중복/음수 dt 거부, mask 양쪽 AND, all-missing fail, 단위 mismatch 오류.
- 누수: target를 임의 교체해도 동일 history의 inference/gate 조건 불변; held-out 값을
  바꿔도 train stats/hash 불변; five-way label disjoint; 필요한 경우 raw-disjoint 별도 계약.
- 정규화: 비균일 channel/cell scale에서 raw↔normalized 차분 일치; delta에 μ를 다시 더하지 않기;
  near-constant channel floor; 6h/12h 실제 dt 나눗셈 일치.
- coupling: 같은 b,m의 lead별 초기 z 일치, 서로 다른 m 독립; 같은 member/step의 모든 expert
  현재 q 일치; member 순서를 모든 lead에서 함께 바꾸면 score 불변, lead 하나만 섞으면 joint score 변화.
- loss: M>=2/shape 검사; off-diagonal fair estimator; 독립 noise 분포의 MC 수렴; mask/area reduction;
  marginal은 같지만 시간 covariance가 다른 반례를 joint score가 구별하는지 확인.
- gradient: L_pair만 켜서 expert/gate/history gradient finite/nonzero; B manifold hash 불변;
  C manifold gradient 및 anchor 유효; frozen decoder에서도 q gradient 전달; zero lambda legacy parity.
- 학습/저장: A→B→C synthetic 2~3 updates, best reload, sidecar 통계/RNG/split/config roundtrip;
  지원하지 않는 C resume fail-fast; epoch 간 lambda 변화가 validation selection 정의를 바꾸지 않음.
- 평가: validation/test 선택 명시, near-zero ratio N/A+count, 단위별 수치, mean/member 별도,
  fixed seed/panel, batch 크기 바뀌어도 동일 A metric pair 평가.
- 12h 출력: index 1,3,…,119의 정확한 valid time, 선택 후 M 불변, source6h/output12h 분리,
  원본 6h 결과 보존, 첫 origin→+12h 차분, 짝수 길이가 아닌 horizon 정책 명시.

## 6. 이번에 실제 실행한 범위

현재 실행 환경에는 NumPy/Matplotlib/ffmpeg가 있지만 PyTorch/pytest, 실제 ERA5 archive,
실험 checkpoint, 연결된 GPU는 없습니다. 다음 probe는 NumPy만 사용합니다.

```bash
python docs/training-mechanism/probe_design.py
```

결과는 [probe-results.json](probe-results.json)에 저장됩니다. **6개 수학적 probe 통과**를 확인했습니다.
이는 새 loss 구현·PyTorch backward·ERA5 학습 테스트가 아닙니다. 추가 정적 검증 결과:
새 Mermaid 5개 문법 통과(mermaid 11.12.0), 로컬 문서 링크 41개 확인, Bash block 13개
`bash -n` 통과, E0 학습 명령 2개를 현재 trainer AST에서 추출한 argparse로 parse했습니다.
Mermaid의 이미지 렌더링 검증은 별도로 수행하지 않았습니다. 원본 첨부 4개의 SHA256도 재확인했습니다.
원본 영상/그림 해석과 이 합성 수학 검사를 구분합니다.
