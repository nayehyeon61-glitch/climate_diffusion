# 처음부터 실행하는 Manifold MoE 학습 순서

**2026-09-10 갱신:** 새 dynamics loss + 120시간 전면 재학습은
[RETRAIN_120H.md](RETRAIN_120H.md)를 순서대로 실행하세요. 동일 member noise, loader의 delta/tendency,
joint trajectory loss, 변수별 wind 보조항, 모든 member 출력이 구현됐습니다.
아래는 기존 H=120(720시간/30일), 새 loss weight=0 profile을 보존한 문서입니다.

대상은 `feature/latent-dynamics-flow`의 **Flow Matching + Manifold MoE + Ensemble**입니다.
2026-09-09의 시간 정렬/영상 기능(`70ee926`)까지 코드와 CLI를 확인해 작성했습니다.
현재 네트워크 구조를 유지하며 아래 순서대로 진행합니다.

| 순서 | 먼저 할 일 | 다음 단계에 전달하는 결과 |
|---|---|---|
| 1 | 설치와 합성 smoke 확인 | 학습·저장·영상 생성이 되는 환경 |
| 2 | 실제 ERA5를 고정 6시간 archive로 준비 | `era5_manifold_6h.npz` + `.schema.json` |
| 3 | 관측 mask·격자·시간·split 가능 여부 확인 | 사용할 하나의 고정 archive |
| 4 | A: PI manifold 학습 | `stage_a.pt` |
| 5 | B: A를 불러와 expert와 gate 학습 | `stage_b.pt` |
| 6 | C: B를 불러와 ensemble 공동 보정 | `final.pt` |
| 7 | Validation 시각에서 영상과 변화율 확인 | 예측 배열·비교 MP4·시간 진단 JSON |
| 8 | 설정 확정 후 test 평가·전문화 그림 | 최종 지표와 학습 그림 |
| 9 | 실제 최신 관측에서 미래 ensemble 생성 | 운영/후단용 `forecast-latest.npz` |

**현재 완료 범위:** 정확한 valid-time 정렬, 시간 변화율 진단/영상, 여러 lead의 FM pair를
연속 시점으로 뽑고 source noise를 공유하는 처리가 있습니다. 아래 B/C는 이를 사용하도록
`--sampled-leads 2`를 명시합니다. 기본값1에서는 연속 lead pair가 생기지 않습니다.
**이 기존 profile에서 활성화하지 않는 기능:** 새 `loss_delta`/`loss_trajectory`와 dynamics labels는
구현됐지만 아래 명령의 기본 새 loss weight는0입니다. 새 profile에서 명시적으로 활성화하세요.
기존 checkpoint는 코드 갱신만으로 재학습되지 않습니다.

2026-09-09 추가: [State + Dynamics Matching 설계](../docs/training-mechanism/README.md)에
사용자 실험 해석, paired endpoint/increment 확률 감독, 수정할 함수와 E0→E1→E2 실험을 정리했습니다.
**아직 새 loss 구현은 아니며**, 아래 명령은 기존 objective를 학습합니다.

## 1. 처음 한 번: 코드와 실행 환경 준비

RunPod의 기존 GPU 환경에서 새 폴더를 사용하는 예입니다. 원자료 다운로드나 유료 자원
생성은 포함하지 않습니다. 다음 명령들은 같은 Bash 세션에서 순서대로 실행합니다.

```bash
cd /workspace
git clone --branch feature/latent-dynamics-flow --single-branch \
  https://github.com/nayehyeon61-glitch/climate_diffusion.git climate_diffusion_manifold
cd climate_diffusion_manifold

python -m venv --system-site-packages .venv
source .venv/bin/activate
python -m pip install -e '.[io,test,plots]'
python -c "import torch; print('torch=', torch.__version__, 'cuda=', torch.cuda.is_available())"
ffmpeg -version
```

`--system-site-packages`는 기존 RunPod 이미지의 PyTorch를 사용할 수 있게 합니다.
이미 checkout이 있고 작업 내용을 보존해 둔 경우에는 `git pull --ff-only` 후 재설치하면 됩니다.
충돌이 나면 `reset --hard` 대신 새 clone으로 진행하세요. `ffmpeg`가 없다면 Ubuntu/RunPod의
root 터미널에서 아래를 실행합니다. GIF 출력은 Pillow를 사용합니다.

```bash
apt-get update
apt-get install -y ffmpeg
```

먼저 실제 ERA5 없이 합성 데이터로 연결을 확인합니다. 이 단계의 weight를 실데이터 A/B/C에
그대로 가져오지 않습니다. 아래 폴더가 이미 사용 중이면 새 이름으로 바꾸세요.

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest -q
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python scripts/smoke_manifold_moe.py \
  --work-dir outputs/check-manifold --report-dir outputs/check-manifold/report
python scripts/visualize_manifold_moe.py \
  --diagnostics outputs/check-manifold/report/diagnostics-joint.json \
  --metrics outputs/check-manifold/report/training-metrics.json \
  --evaluation outputs/check-manifold/report/evaluation-local.json \
  --output-dir outputs/check-manifold/figures
python scripts/smoke_time_alignment.py
```

확인할 결과는 `outputs/check-manifold/model.pt`, `report/summary.json`, `figures/`와
`outputs/time-alignment-synthetic/comparison.mp4`입니다. 첫 smoke는 **48시간·sampled_leads=1**의
기존 전체 연결 검사이고, 두 번째는 일부러 변화량이 작은 moving-field를 진단하는 검사입니다.
둘 다 실제 ERA5 장기 예측 성능 검증은 아닙니다.

## 2. 실제 ERA5와 이번 실험 경로 지정

`MANIFOLD_FIELDS`를 **실제로 보유한** NetCDF 또는 Zarr 파일/폴더로 변경하세요.
예시 변수는 `msl`(Pa), `t2m`(K), `u10/v10`(m/s), 같은 lat/lon 격자의 2D field입니다.
상층 pressure-level 축이나 IBTrACS 표를 이 명령에 추가하지 않습니다.

```bash
export MANIFOLD_FIELDS=/workspace/data/era5_surface_6h.zarr
export MANIFOLD_ARCHIVE=/workspace/data/era5_manifold_6h.npz
export MANIFOLD_RUN=/workspace/outputs/manifold_run_001
export MANIFOLD_DEVICE=cuda

mkdir -p /workspace/outputs
mkdir "$MANIFOLD_RUN"
git rev-parse HEAD > "$MANIFOLD_RUN/code-commit.txt"

python -m climate_diffusion.fixed_step_data \
  --fields "$MANIFOLD_FIELDS" --variables msl t2m u10 v10 \
  --step-hours 6 --target-lat-points 18 --target-lon-points 36 \
  --output "$MANIFOLD_ARCHIVE"
```

`mkdir "$MANIFOLD_RUN"`에서 이미 있는 폴더라고 나오면 새 실험 번호를 사용하세요.
완료된 동일 계약의 archive가 있으면 변환을 건너뛰고 그 경로를 사용합니다. A/B/C 사이에는
archive를 다시 생성하거나 변경하지 않습니다. Checkpoint가 byte-level SHA와 schema를 확인합니다.
`.npz`와 같은 이름의 `.schema.json`을 항상 함께 보관하세요.

CPU 환경에서는 `MANIFOLD_DEVICE=cpu`로 바꿉니다. 현재 구현은 dense MLP·명시적 Jacobian을
사용하므로 우선 위 coarse grid에서 확인하세요. 전체 archive를 RAM에 읽고 PI 통계 fit은
train span을 device에 올립니다. 실제 0.25° 전지구 장기 학습의 메모리 요구는 검증하지 않았습니다.

## 3. 학습 전에 archive와 시간 설정 확인

아래 사전 확인은 다음 A 명령과 같은 설정을 사용합니다. 모델을 학습하지 않고 실제 loader와
split 함수를 호출합니다. 결측 mask, 불연속 시간, 잘못된 grid, 너무 짧은 기록은 여기서 수정하세요.

```bash
python - <<'PY'
import os
import numpy as np
from climate_diffusion.moe_data import load_moe_archive, field_grid, build_moe_split
from climate_diffusion.manifold_moe import ManifoldMoEConfig
from climate_diffusion.manifold_physics import SurfacePhysics

states, times, schema = load_moe_archive(os.environ['MANIFOLD_ARCHIVE'])
assert schema['forecast_step_hours'] == 6, '이 예시는 6시간 archive를 사용합니다'
config = ManifoldMoEConfig(state_dim=states.shape[1], grid=field_grid(schema),
    step_hours=6, history_steps=6, history_stride=4, horizon_steps=120,
    num_experts=4, manifold_dim=16)
SurfacePhysics(schema, np.zeros(states.shape[1]), np.ones(states.shape[1]))
count = len(states) - config.history_span_steps - config.horizon_steps + 1
split = build_moe_split(count, config.horizon_steps)
print('shape/time:', states.shape, times[0], times[-1])
print('history span steps:', config.history_span_steps, 'horizon hours:', config.horizon_hours)
print('windows:', {k: len(split[k]) for k in
      ('train', 'expert_validation', 'calibration', 'validation', 'test')})
print('variable units:', {v['name']: v.get('attrs', {}).get('units', '원자료 확인 필요')
                         for v in schema['variables']})
PY
```

| 설정 | 이 예시에서의 뜻 |
|---|---|
| `step-hours 6` | archive 관측/예측 간격6시간 |
| `history-steps 6`, `history-stride 4` | 24시간 간격 관측6개, 5일 전부터 origin까지 |
| `horizon-steps 120` | +6h부터 +720h=30일까지 예측 |
| `window-stride 4` | 학습 window 시작점을4개마다 사용 |
| `sampled-leads 2` | 한 window에서 연속 미래 시점2개를 FM 학습에 사용 |
| `num-experts 4` | 전체 기상장을 다루는 국소 expert4개 |
| `ensemble-size 4` | 학습의 확률 점수에 사용할 member4개 |

15일 horizon은60 step이지만, 바꾸려면 사전 확인과 A부터 같은 설정을 사용해야 합니다.
30일 horizon의 five-way split은 경계마다 최소119개 window를 purge하므로 며칠/몇 주 자료로는
부족합니다. 짧은 자료에서 split 검사를 우회하지 마세요. 단위 attrs는 보존하지만 자동 단위
변환은 하지 않습니다. NaN을 숫자0으로 바꿔 관측했다고 표시하는 방식도 사용하지 않습니다.

## 4. Stage A: 상태 좌표계부터 학습

PI encoder/decoder와 인접 관측 latent dynamics를 학습합니다. 이 단계의 checkpoint는
아직 예측용 expert flow가 없으므로 forecast를 실행하지 않습니다.

```bash
python -m climate_diffusion.train_manifold_moe \
  --archive "$MANIFOLD_ARCHIVE" --output "$MANIFOLD_RUN/stage_a.pt" \
  --stage manifold --history-steps 6 --history-stride 4 --horizon-steps 120 \
  --num-experts 4 --manifold-dim 16 --expert-latent-dim 64 --gate-hidden-dim 160 \
  --manifold-epochs 50 --batch-size 8 --window-stride 4 \
  --learning-rate 0.001 --seed 7 --device "$MANIFOLD_DEVICE"

python - "$MANIFOLD_RUN/stage_a.metadata.json" <<'PY'
import json, sys
t = json.load(open(sys.argv[1]))['training']
print('stage:', t['stage'], 'best epoch:', t['best_epoch'], 'validation:', t['best_selection_score'])
PY
```

다음 단계로 넘기는 것은 **`stage_a.pt`**입니다. 함께 생성되는 `.metadata.json`,
`.metrics.json`, `.manifest.json`도 보존합니다. Reconstruction/physics/metric validation을
확인하세요. 마지막 epoch가 아닌 가장 좋은 validation checkpoint가 저장됩니다.

## 5. Stage B: A를 불러와 expert 전문화 학습

PI manifold를 동결하고 experts·local gate·history encoder를 학습합니다. 연속6시간 lead2개가
같은 FM source noise를 사용하도록 `--sampled-leads 2`를 명시합니다.

```bash
python -m climate_diffusion.train_manifold_moe \
  --archive "$MANIFOLD_ARCHIVE" --output "$MANIFOLD_RUN/stage_b.pt" \
  --stage specialize --init-checkpoint "$MANIFOLD_RUN/stage_a.pt" \
  --expert-epochs 40 --batch-size 2 --window-stride 4 \
  --ensemble-size 4 --sampled-leads 2 --integration-steps 4 \
  --learning-rate 0.001 --seed 7 --device "$MANIFOLD_DEVICE"

python - "$MANIFOLD_RUN/stage_b.metadata.json" <<'PY'
import json, sys
t = json.load(open(sys.argv[1]))['training']
print('stage:', t['stage'], 'best epoch:', t['best_epoch'], 'validation:', t['best_selection_score'])
print('lead sampling:', t['training_lead_sampling'], 'sampled_leads:', t['loss_options']['sampled_leads'])
PY
```

다음 단계의 입력은 **`stage_b.pt`**입니다. 차원·history·horizon 설정은 A checkpoint에서
이어받습니다. B에서 임의로 다른 차원을 지정하지 않습니다. FM/expert loss, gate 사용률,
validation Energy/CRPS를 확인하세요. Epoch 수는 시작 설정이며 수렴을 보장하는 값은 아닙니다.

## 6. Stage C: B를 불러와 ensemble 공동 보정

실제로 ODE를 적분해 만든 ensemble에 Energy/CRPS를 계산하고, A 기준 좌표의 anchor와
PI loss를 함께 사용합니다. Flow Matching과 member별 독립 시나리오는 유지합니다.

```bash
python -m climate_diffusion.train_manifold_moe \
  --archive "$MANIFOLD_ARCHIVE" --output "$MANIFOLD_RUN/final.pt" \
  --stage joint --init-checkpoint "$MANIFOLD_RUN/stage_b.pt" \
  --joint-epochs 10 --batch-size 2 --window-stride 4 \
  --ensemble-size 4 --sampled-leads 2 --integration-steps 4 \
  --learning-rate 0.001 --joint-lr-factor 0.1 --encoder-lr-factor 0.1 \
  --seed 7 --device "$MANIFOLD_DEVICE"
```

이 설정의 C 학습률은 experts/gate/history `1e-4`, PI manifold `1e-5`입니다.
최종 예측 모델은 **`final.pt`**입니다. Validation 지표로 C checkpoint를 선택합니다.
메모리가 부족해도 A/C에서 무조건 batch2→1로 줄이지 마세요. 현재 metric loss는 batch1이면
0이 됩니다. [자원·metric pair 주의사항](../docs/training-mechanism/RUNBOOK.md)을 먼저 확인하세요.
연속 lead를 쓰려면 sampled-leads2를 유지합니다.

현재 `_pairs()`는 연속 lead의 FM source를 공유합니다. C의 ensemble-score용 `_sample()`은
flatten된 lead별로 새 noise를 뽑으므로 전체 ensemble 학습이 joint trajectory loss인 것은
아닙니다. 추론은 같은 member의 초기 noise를 lead간 재사용합니다.

## 7. 실제 미래 정답이 있는 validation 시점에서 영상 확인

최신 관측 뒤 미래를 생성하면 같은 archive 안에 정답이 없습니다. 비교 영상은 아래처럼
**archive 내부의 과거 origin**을 선택합니다. 날짜를 외워 입력하지 않고 checkpoint에 기록된
validation split에서 자동으로 골라 history/target 인덱스를 맞춥니다.

```bash
python - <<'PY'
import json, os
from pathlib import Path
import numpy as np
run = Path(os.environ['MANIFOLD_RUN'])
t = json.loads((run / 'final.metadata.json').read_text())['training']
origin_index = t['split']['validation'][0] + t['history_span_steps'] - 1
with np.load(os.environ['MANIFOLD_ARCHIVE'], allow_pickle=False) as a:
    origin = str(a['times'][origin_index].astype('datetime64[ns]')) + 'Z'
(run / 'validation-origin.txt').write_text(origin + '\n')
print('validation origin:', origin)
PY
read -r MANIFOLD_ORIGIN < "$MANIFOLD_RUN/validation-origin.txt"

python -m climate_diffusion.time_alignment \
  --checkpoint "$MANIFOLD_RUN/final.pt" --archive "$MANIFOLD_ARCHIVE" \
  --origin-time "$MANIFOLD_ORIGIN" --forecast-steps 120 \
  --ensemble-size 8 --integration-steps 16 --seed 83 --device "$MANIFOLD_DEVICE" \
  --moe-mode local --variable t2m --u-name u10 --v-name v10 --fps 2.5 \
  --forecast-output "$MANIFOLD_RUN/validation-forecast.npz" \
  --output "$MANIFOLD_RUN/validation-mean.mp4" \
  --report "$MANIFOLD_RUN/validation-time.json"

python -m climate_diffusion.time_alignment \
  --forecast "$MANIFOLD_RUN/validation-forecast.npz" --archive "$MANIFOLD_ARCHIVE" \
  --member 0 --variable t2m --fps 2.5 \
  --output "$MANIFOLD_RUN/validation-member-0.mp4" \
  --report "$MANIFOLD_RUN/validation-member-0-time.json"
```

두 번째 명령은 재추론 없이 같은 ensemble의 member 0을 그립니다. `--member`는 0부터
시작하며, JSON 진단은 두 명령 모두 저장된 **전체 ensemble**을 대상으로 계산합니다.
표시한 member만의 RMSE로 바뀌지는 않습니다. 6h×120 lead는 30일이고 2.5fps 영상은 48초입니다.

확인 순서는 **같은 valid time → t2m과 u/v → error/spread → 변화량/hour → member와 평균의 차이**입니다.
변화량 비율이 작으면 시간 지연·진폭 감소·평균화 효과를 구분하세요. fps나 풍속을 임의로
키워 보정하지 않습니다. `time_alignment`의 전체 state RMS/CRPS는 원래 단위가 서로 다른
변수를 합친 보조 진단입니다. msl 등의 크기에 영향을 받으므로 변수별 지도,
`uv_vector_rmse_mps`, 다음 단계의 정규화 평가를 함께 확인하세요.
보정할 때는 validation까지로 설정을 결정하고 새 실험 폴더에 결과를 저장합니다.

## 8. 설정 확정 후 test 평가와 학습·전문화 그림 생성

```bash
python -m climate_diffusion.evaluation \
  --checkpoint "$MANIFOLD_RUN/final.pt" --archive "$MANIFOLD_ARCHIVE" \
  --output "$MANIFOLD_RUN/evaluation-local.json" --moe-mode local \
  --ensemble-size 8 --integration-steps 16 --max-cases 32 --seed 83 --device "$MANIFOLD_DEVICE"
python -m climate_diffusion.evaluation \
  --checkpoint "$MANIFOLD_RUN/final.pt" --archive "$MANIFOLD_ARCHIVE" \
  --output "$MANIFOLD_RUN/evaluation-uniform.json" --moe-mode uniform \
  --ensemble-size 8 --integration-steps 16 --max-cases 32 --seed 83 --device "$MANIFOLD_DEVICE"

python -m climate_diffusion.manifold_diagnostics \
  --checkpoint "$MANIFOLD_RUN/final.pt" --archive "$MANIFOLD_ARCHIVE" \
  --output "$MANIFOLD_RUN/manifold-diagnostics.json" --members 4 --integration-steps 8 \
  --seed 83 --device "$MANIFOLD_DEVICE"

python - <<'PY'
import json, os
from pathlib import Path
run = Path(os.environ['MANIFOLD_RUN'])
rows = []
for stage in ('stage_a', 'stage_b', 'final'):
    rows.extend(json.loads((run / f'{stage}.metrics.json').read_text()))
(run / 'training-abc.metrics.json').write_text(json.dumps(rows, indent=2) + '\n')
PY
python scripts/visualize_manifold_moe.py \
  --diagnostics "$MANIFOLD_RUN/manifold-diagnostics.json" \
  --metrics "$MANIFOLD_RUN/training-abc.metrics.json" \
  --evaluation "$MANIFOLD_RUN/evaluation-local.json" --output-dir "$MANIFOLD_RUN/figures"
```

단독 A/B/C 실행의 metrics에는 각 단계만 들어 있으므로 위에서 합칩니다. `figures/`에는
`training-abc.png`, `routing-learning.png`, `manifold-geometry.png`,
`expert-specialization.png`, `rank-histogram.png`와 SVG가 생성됩니다.

`evaluation`과 현재 `manifold_diagnostics`는 test split을 사용합니다. 반복해서 보면서
hyperparameter를 고르는 용도로 사용하지 않습니다. PCA 색 분리만으로 판단하지 말고
영역별 expert 오차와 실제 예측 정확도를 확인하세요. Local/uniform은 같은 seed·member 수·
적분 step으로 비교합니다. 현재 전문화 진단은 최종 physical lead의 표본을 사용하므로
모든 lead에서 전문화가 일어났다는 증거는 아닙니다.

## 9. 마지막으로 최신 관측에서 미래 ensemble 저장

```bash
python -m climate_diffusion.inference \
  --checkpoint "$MANIFOLD_RUN/final.pt" --archive "$MANIFOLD_ARCHIVE" \
  --forecast-steps 120 --ensemble-size 8 --integration-steps 16 \
  --moe-mode local --seed 83 --device "$MANIFOLD_DEVICE" \
  --output "$MANIFOLD_RUN/forecast-latest.npz"
```

이 NPZ에는 `predictions[M,H,D]`, `origin_time`, `lead_hours`, `valid_times`,
`forecast_step_hours`가 들어갑니다. M개 시나리오를 유지한 채 후단에 전달합니다.
최신 origin의 미래를 같은 archive와 비교하면 미래 정답이 없어 실패하므로 비교에는
7단계의 과거 origin 방식을 사용하세요.

## 중단 후·기존 weight·보정 시 시작 위치

| 현재 보유한 것 | 다음에 실행할 단계 |
|---|---|
| ERA5 원자료만 있음 | 2단계부터 |
| 올바른 6h archive만 있음 | 3단계부터 |
| 완료된 manifold Stage A checkpoint | 같은 archive로 5단계 |
| 완료된 manifold Stage B checkpoint | 같은 archive로 6단계 |
| 완료된 manifold Stage C checkpoint | 7/8/9단계 평가·추론 |
| 이전 MoE(meta) 또는 dynamics checkpoint | 현재 manifold는 A부터 새로 학습, 기존 weight는 보관 |

단계 도중 중단했을 때 optimizer/RNG까지 복원하는 exact resume는 미구현입니다. A를 중간에
멈추면 geometry seal이 완료되지 않아 B 입력으로 사용할 수 없습니다. 완료된 직전 단계의
checkpoint를 입력으로 해당 단계를 새 출력 이름으로 다시 실행하세요. A 자체를 다시 실행할
때는 초기 checkpoint 없이 시작합니다. C checkpoint에서 C를 이어 학습하는 것도 현재 CLI는
지원하지 않습니다. C 재시도는 B에서 시작합니다. Archive/차원/horizon을 바꾸면 A부터
일관된 새 설정으로 학습합니다.

보관할 파일은 `.pt`와 sidecar 3종, 사용한 archive/schema, `code-commit.txt`, 실행 인자,
평가 JSON/영상입니다. SSH를 다시 열었으면 가상환경과 2단계 환경 변수를 재설정하고,
archive 생성과 `mkdir "$MANIFOLD_RUN"`은 반복하지 말고 기존 실험 경로를 지정하세요.

이 안내를 작성하면서 Bash 블록 11개와 Python 예제 5개의 문법, 실행 명령 13개의 실제
CLI 인자, 문서 링크를 확인했습니다. 이번 문서 변경으로 실제 ERA5 학습을 새로 실행한 것은
아닙니다. 실제 데이터의 사전 검사는 3단계, 학습·검증은 4단계 이후에 실행합니다.

[구조·loss 상세](MANIFOLD_README.md) / [학습 Mermaid](../struct-picture/06-manifold-training.md) /
[시간 진단 Mermaid](../struct-picture/08-time-alignment.md) /
[기존 합성 시간 진단 결과](../docs/results/time-alignment-synthetic/README.md).
