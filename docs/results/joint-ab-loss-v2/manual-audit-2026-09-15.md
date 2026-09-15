# A+B 실행 매뉴얼/CLI 감사 — 2026-09-15

검사한 원격 코드: `feature/joint-ab-loss-v2`의 `3fc532884b0758e6ff63849fe5296b2e139556fb`.
기반 optimization `4dd9bf4051aa8c277099c980313bdde91d48f854`.
이번 변경 범위는 **문서, 최소 runner/읽기 전용 helper, 그 테스트**입니다.
`src/climate_diffusion`의 모델·loss·평가 구현, weights, 사용자 결과 파일은 변경하지 않았습니다.

## 완성한 실행 연결

- [한국어 전체 매뉴얼](../../../flow-matching_moe/JOINT_AB_TRAINING_MANUAL.md):
  환경 → archive → A → AB → legacy C → validation → 한 forecast → member 6h/12h → frozen test.
- `scripts/run_joint_ab_120h.sh`: 명시적 phase, 새 output 보호, pilot1epoch 기본,
  `horizon_steps=20`, AB memberMSE0, C temporal loss 명시, 실제 forecast/render CLI 연결.
- `scripts/inspect_joint_run.py`: archive/checkpoint SHA·split·통계·시간계약 검증과
  실제 archive의 split별 UTC 범위/validation origin 출력. 데이터/weights를 수정하지 않음.
- [Mermaid](../../../struct-picture/15-joint-ab-loss-v2.md): generated q의 직접 recurrence,
  AB target만 detach, C legacy 경로, warmup reference와 고정 affine/chart 표시.

기존 runner/문서에는 H20 누락(기본H120=720h), C temporal loss flag 누락,
renderer의 존재하지 않는 `--checkpoint/--output` 조합, 지원되지 않는 same-stage continuation
안내가 있었습니다. 이번 보정은 이를 실제 argparse/실행 분기에 맞췄습니다.

## 장기 재학습 전 해결해야 할 코드 문제

### 1. V2 transition score의 물리 단위 환산 누락 — 재현됨

관련: [`train_manifold_moe._epoch`](../../../src/climate_diffusion/train_manifold_moe.py)의 joint_ab 분기,
[`joint_objective.normalized_tendencies/trajectory_scores`](../../../src/climate_diffusion/joint_objective.py),
[`evaluation.evaluate_flow_checkpoint`](../../../src/climate_diffusion/evaluation.py).

trainer가 V2에 넘기는 generated/truth는 state-normalized 좌표입니다. 반면 tendency_scale은
train 관측쌍의 **물리 단위/hour** scale입니다. 현재 V2는
`diff(x_normalized)/dt_hours/tendency_scale`만 계산합니다. 필요한 식은
`diff(x_normalized)*state_scale/dt_hours/tendency_scale`입니다.
state mean은 차분에서 소거되지만 scale은 소거되지 않습니다.

검사 예: normalized delta=1, state_scale=1000, dt=6h, physical tendency_scale=100.
현재 결과 `0.0016666667`, 물리 환산 기대값 `1.6666667`.
이 예의 mismatch는1000배이며 실제 영향은 grid/변수별 state scale에 따라 다릅니다.
transition CRPS/mean tendency/joint increment score와 V2 평가의 단위를 검토해야 합니다.
legacy `TemporalObjective`와 A/AB decoded AEdelta/drift 보조항에는 state_scale 복원이 있어
동일한 오류라고 일반화하면 안 됩니다.

재현(모델 수정/학습 없음):

```bash
python - <<'PY'
import torch
from climate_diffusion.joint_objective import normalized_tendencies
x = torch.tensor([[[[0.], [1.]], [[0.], [1.]]]])
y = torch.tensor([[[0.], [1.]]])
v, _ = normalized_tendencies(x, y, torch.tensor([[6.]]), torch.tensor([100.]))
print('current:', v[0,0,0,0].item(), 'expected:', 1000/6/100)
PY
```

**명령 파싱·기존 테스트 통과로 이 단위 문제까지 검증되었다고 주장하지 않습니다.**
이번 매뉴얼 요청은 모델/loss 의미 변경을 금지했으므로 문제를 보고하고 장기 재학습을 보류합니다.

### 2. `--log-gradient-norms`의 parameter generator 소진 — 재현됨

관련: `_epoch`에서 `.parameters()` generator를 group으로 구성한 뒤
`module_gradient_diagnostics`가 loss마다 반복 소비합니다. 첫 loss 뒤 다음 loss의 parameter가
빈 목록이 되어 `RuntimeError: inputs argument to grad() cannot be empty`가 발생합니다.
raw→weighted 두 호출 사이에서도 같은 group을 재사용합니다.

```bash
python - <<'PY'
import torch
from climate_diffusion.joint_objective import module_gradient_diagnostics
p = torch.nn.Parameter(torch.ones(2))
module_gradient_diagnostics({'one': p.square().sum(), 'two': p.sum()},
                            {'encoder': iter([p])})
PY
```

이 명령은 현재 오류를 재현하는 진단입니다. 학습 runner에서는 flag를 제거했습니다.
후속 보수는 parameter 목록을 재사용 가능하게 만들고 실제 raw/weighted 각 loss의 backward를
검사해야 합니다. 이번 작업에서 gradient norm/cosine 측정을 완료했다고 보고하지 않습니다.

### 3. Loss 스케일/선택적 calibration 검토

`trajectory_scores`의 mean-state/mean-tendency는 합1 metric을 곱한 뒤 전체 mean을 취해
legacy weighted sum보다 추가1/D 스케일이 생깁니다. joint Energy에도 이미 좌표/시간
가중한 feature에 추가 sqrt(feature_dim) 정규화가 있습니다. 가중치를 비교할 때 고려해야 합니다.
이를 모두 오류라 확정한 것은 아니지만 control/V2/legacy의 total을 동일한 물리 score로
비교할 수 없으며, 문서의 후보 계수를 검증된 최적값으로 사용할 수 없습니다.

spread calibration은 실제로 per-cell 비율 후 평균이며 문서에 있던 pooled estimator와 다릅니다.
`(1+1/M)*squared_mean_error`의 finite-M 방향도 검토가 필요합니다. iid ensemble과 독립 truth가
같은 조건부분포를 따르면 기대 mean-error MSE는 `(1+1/M)*conditional_variance`입니다.
그 역산과 batch/cell별 강제 일치의 차이를 확인해야 합니다. 현재 모든 profile의 calibration은0이며
runner는 이를 활성화하지 않습니다. detach만으로 shortcut이 모두 방지되는 것은 아닙니다.

## 단계별 경계 재확인

| 항목 | 코드에서 확인한 현재 상태 |
|---|---|
| AB | 새 profile, generated trajectory 한 번 + 별도 teacher-FM, 전 모듈 공동 최적화 |
| C | legacy objective; generic temporal weights를 명시해야 delta/trajectory/wind 활성 |
| AB/C member-MSE | AB 및 AB parent의 C는0 요구 |
| target | residual label만 stop-gradient; conditioning q는 live gradient |
| 좌표 | best A 뒤 seal 한 번, AB/C affine/chart 고정, warmup reference 유지 |
| resume | optimizer/RNG state 없음; AB→AB/C→C init 금지; 허용 parent에서 새 stage만 가능 |
| AB ramp/block | generic temporal ramp 미적용; validation에도 trajectory_edges 사용 |
| 선택 | stage별 best score 정의 다름; 같은 평가 지표·seed로 비교 필요 |

## 검증 범위

현재 CPU 환경에서 실제 CLI parser와 모든 phase의 인자를 검사했습니다. DRY_RUN은
파일/학습을 만들지 않으며, 공백이 있는 경로·test 확정 flag·동일 NPZ의 6h/12h 선택을 검사합니다.
320시점의 **합성 archive**로 실제 preflight/UTC calendar/21-state·20-delta shape,
train-only 통계와 SHA mismatch 거절을 검사합니다. 실제 ERA5 연도나 skill 측정이 아닙니다.

`scripts/smoke_joint_ab.py`는 random tensor score/backward probe입니다. 이를 A→AB→C
120h from-scratch 학습이라고 소개하지 않습니다. 기존 regression의 작은 학습 테스트와
이번 명령 검증을 실제 ERA5 재학습 성능으로 대체하지 않습니다.

환경: Python3.12.14, PyTorch2.14.0+cpu, pytest9.1.1. ffmpeg 존재.

- 전체 `python -m pytest -q`: **86 passed, 3 warnings, 35.60초**.
- 새 runbook 검사12개: 실제 argparse, 공백 경로, full20/member0/C loss, 한 forecast의6h/12h,
  test확정 조건, preflight/UTC/hash와 작은 **H20 A 1epoch checkpoint** 저장·읽기 검사.
- 정상 state mean/scale SHA 확인을 helper에 추가한 후 해당 runbook 검사12개를 다시 통과.
- trainer/evaluation/time_alignment/trajectory_output/prepare/inspect/visualize의 `--help`7개 성공,
  `bash -n` 및 `git diff --check` 성공. Mermaid는 연결/시간/gradient 내용을 검토했으며
  별도 Mermaid 렌더 엔진 문법 검사는 이번 실행에 포함하지 않았습니다.

작은 A 학습은 합성4×4×8 field, manifold3/hidden12 등으로 검사 비용을 제한했습니다.
4090 설정의 전체 A→AB→C 장기 학습이나 성능 대조를 새로 실행한 것은 아닙니다.
warnings는 기존 NumPy/netCDF, torch.jit 및 테스트 tensor→scalar 관련 경고입니다.

실제 ERA5 archive, 연결된 RunPod/4090에서의 새 장기 재학습, GPU VRAM/속도 측정은
이번 작업에서 미실행입니다. 모델 성능 개선이나 ensemble 확장을 확인한 새 결과는 없습니다.
