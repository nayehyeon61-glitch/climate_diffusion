# 전체 학습 실행 매뉴얼: ERA5 120시간 State + Dynamics

이 문서는 코드 `0e641b2`에서 구현된 학습을 **새 A 초기화부터 member 영상까지** 순서대로 실행하는 운영 매뉴얼입니다. 기존 checkpoint를 다시 렌더링하는 것만으로 새 loss를 학습한 것이 아닙니다.

현재 모델은 Flow Matching + Physics-Informed Manifold MoE + Ensemble입니다. 동일 member의 noise를 lead 사이에 공유하고 endpoint 차분을 감독하지만, **이전 예측 state를 다음 physical step 입력으로 넣는 recurrent 모델은 아직 아닙니다**. 새 recurrent 제안은 [별도 원문](../docs/training-mechanism/physical-time-recurrent-user-proposal.md)에 있으며 아래 명령이 그 제안을 구현했다고 해석하지 마세요.

| 순서 | 실행 | 다음 단계로 넘어가는 조건 |
|---|---|---|
| 1 | 환경 설치·합성 smoke | 명령·저장·학습·영상 경로가 실행됨 |
| 2 | ERA5 archive·preflight | 실제 6h cadence, 변수·단위·mask·split 확인 |
| 3 | A manifold 새 학습 | best A 재로드·seal·preflight 일치 |
| 4 | B experts/gate/history 학습 | A 동결, 새 loss 활성화·gradient 확인 |
| 5 | C joint calibration | 작은 LR·anchor 유지, best C 저장 |
| 6 | validation·member별 영상 | state와 tendency·spread를 함께 검사 |
| 7 | 설정 고정 후 test | test를 보고 계수 변경하지 않음 |

120시간은 **6h × 20 step = 5일**입니다. `horizon_steps=120`은 이 archive에서 30일이므로 혼동하지 마세요. 자세한 loss 수식·데이터 shape는 [RETRAIN_120H.md](RETRAIN_120H.md), 기존 30일 안내는 [TRAINING_README.md](TRAINING_README.md)에 있습니다.

## 1. 별도 clone과 실행 환경

아래 shell 블록은 동일한 Bash 세션에서 위에서 아래 순서로 실행합니다. 기존 실험 폴더를 덮어쓰지 않습니다.

```bash
git clone --branch feature/latent-dynamics-flow --single-branch \
  https://github.com/nayehyeon61-glitch/climate_diffusion.git climate_diffusion_temporal
cd climate_diffusion_temporal
python -m venv --system-site-packages .venv
source .venv/bin/activate
python -m pip install -e '.[test,plots,io]'
python - <<'PY'
import torch
print('PyTorch:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU:', torch.cuda.get_device_name(0))
PY
ffmpeg -version
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest -q
```

RunPod의 CUDA PyTorch가 준비되어 있다면 위 venv가 이를 재사용합니다. CUDA가 false이면 GPU 설정을 먼저 확인하세요. CPU 확인에는 이후 `TEMPORAL_DEVICE=cpu`를 사용할 수 있습니다. MP4에는 ffmpeg가 필요합니다. 학습 자체와 GIF 출력은 별개지만 아래 통합 smoke는 MP4도 생성하므로 ffmpeg를 먼저 준비하세요.

## 2. 실제 데이터 전에 합성 end-to-end 확인

기본 report 경로는 저장소에 이미 있으므로 **새 경로를 반드시 지정**합니다. 같은 이름으로 재실행하면 출력 보호 때문에 실패합니다.

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python scripts/smoke_temporal_moe.py \
  --work-dir outputs/manual-smoke-001 \
  --report-dir outputs/manual-smoke-report-001
python scripts/visualize_temporal_moe.py \
  --report-dir outputs/manual-smoke-report-001 \
  --checkpoint outputs/manual-smoke-001/new-c.pt \
  --archive outputs/manual-smoke-001/synthetic-states.npz
```

이 smoke는 새 A→baseline/new B→C, 별도 full-window B backward, validation, 6h/12h member 출력을 확인합니다. 합성 데이터의 작은 epoch 결과는 실제 ERA5 성능이 아닙니다. 실패하면 그 오류부터 해결한 뒤 실데이터 학습으로 넘어갑니다.

## 3. 실험 경로와 데이터 계약 고정

아래 두 데이터 경로를 실제 보유 파일에 맞게 수정하세요. 새 archive가 필요할 때만 4단계 변환을 실행합니다. 이미 올바른 archive가 있다면 재생성하지 않습니다.

```bash
export TEMPORAL_FIELDS=/workspace/data/era5-surface.nc
export TEMPORAL_ARCHIVE=/workspace/data/era5-temporal-6h.npz
export TEMPORAL_RUN=/workspace/experiments/temporal-120h-run-002
export TEMPORAL_DEVICE=cuda
export TEMPORAL_EDGES=2
test ! -e "$TEMPORAL_RUN" || { echo '새 TEMPORAL_RUN 경로를 지정하세요'; exit 1; }
mkdir -p "$TEMPORAL_RUN"
git rev-parse HEAD > "$TEMPORAL_RUN/code-commit.txt"
python -m pip freeze > "$TEMPORAL_RUN/environment.txt"
set -o pipefail
```

`TEMPORAL_EDGES=2`는 전체 120h window 중 연속 두 구간을 뽑는 초기 메모리 예산입니다. **120h 전체 joint score 학습은 `TEMPORAL_EDGES=0`**으로 실행해야 합니다. 2로 학습하고 120h를 출력하는 것과 full-window 학습은 다릅니다. 같은 실험의 B/C에서는 이 설정을 고정하고, 비교 실험은 새 폴더에 만드세요.

## 4. ERA5 → 6시간 archive → preflight

```bash
test -f "$TEMPORAL_FIELDS" || { echo '실제 ERA5 입력 경로를 확인하세요'; exit 1; }
test ! -e "$TEMPORAL_ARCHIVE" || { echo '기존 archive는 보존하고 이 변환 블록을 건너뛰세요'; exit 1; }
prepare-climate-fixed-step-data \
  --fields "$TEMPORAL_FIELDS" --variables msl t2m u10 v10 \
  --step-hours 6 --target-lat-points 18 --target-lon-points 36 \
  --output "$TEMPORAL_ARCHIVE"
```

이미 archive가 있으면 위 블록 전체를 건너뛰고 다음 검사를 실행합니다.

```bash
python scripts/prepare_temporal_120h.py \
  --archive "$TEMPORAL_ARCHIVE" --output "$TEMPORAL_RUN/preflight.json" \
  --history-steps 6 --history-stride 4
```

확인할 항목:

- `msl` Pa, `t2m` K, `u10/v10` m/s를 모두 포함하며 archive의 실제 schema/grid를 확인합니다. 요청한 grid 수만 보고 state dimension을 추정하지 마세요.
- 실제 timestamp 간격이 6h이고 gap/중복/NaT/결측 pair를 통과하지 않아야 합니다. 현재 temporal 경로는 미관측 pair를 조용히 0으로 채우지 않고 실패합니다.
- history는 origin까지의 24h 간격 관측 6개입니다(`history_stride=4`). 미래는 **연속 6h 20개**, origin 포함 21개 state/20개 tendency입니다.
- five-way split은 train → expert_validation → calibration → validation → test입니다. 미래 target overlap을 purge하며 origin/history의 인과적 과거 공유와 구분합니다.
- state/dynamics 통계는 train 관측·고유 인접쌍만으로 fit합니다. validation/test에서 재계산하지 않습니다.

변수 weight를 바꾸려면 이 preflight와 새 A의 `--variable-weights`를 동일하게 지정하세요. B/C만 따로 바꾸지 않습니다. 모든 변수의 원시 Pa/K/m/s 크기를 한 norm으로 비교하지 않습니다.

## 5. A: manifold를 처음부터 학습하고 best seal

```bash
train-climate-manifold-moe --archive "$TEMPORAL_ARCHIVE" \
  --output "$TEMPORAL_RUN/a.pt" --stage manifold \
  --history-steps 6 --history-stride 4 --horizon-steps 20 \
  --num-experts 4 --manifold-dim 16 --expert-latent-dim 64 --gate-hidden-dim 160 \
  --manifold-epochs 50 --batch-size 8 --window-stride 4 \
  --learning-rate 0.001 --weight-decay 0.0001 --early-stop-patience 8 \
  --seed 7 --device "$TEMPORAL_DEVICE" 2>&1 | tee "$TEMPORAL_RUN/a.console.log"
python scripts/prepare_temporal_120h.py --archive "$TEMPORAL_ARCHIVE" \
  --output "$TEMPORAL_RUN/preflight-a-verified.json" --checkpoint "$TEMPORAL_RUN/a.pt"
```

A에는 `--init-checkpoint`를 넣지 않습니다. reconstruction/PI/metric/latent-dynamics를 학습하고 best validation A를 재로드한 뒤 latent 통계·지역 중심·reference encoder를 seal합니다. 여기서는 생성 trajectory의 새 loss를 학습하지 않습니다.

`a.metrics.json`의 전체 validation뿐 아니라 변수별 `reconstruction_mse_*`/`reconstruction_tendency_mse_*`도 확인하세요. 온도만 좋아지는지, u/v 보존이 약한지 분리해서 봅니다. A/C batch 1은 metric loss가 조용히 0이 되는 문제를 막기 위해 허용하지 않습니다. 마지막 singleton을 이전 batch에 합쳐 실제 최대 batch는 설정값+1일 수 있습니다.

## 6. B: A 고정, experts/gate/history 학습

```bash
train-climate-manifold-moe --archive "$TEMPORAL_ARCHIVE" \
  --output "$TEMPORAL_RUN/b.pt" --stage specialize --init-checkpoint "$TEMPORAL_RUN/a.pt" \
  --expert-epochs 40 --batch-size 2 --window-stride 4 \
  --ensemble-size 4 --sampled-leads 2 --integration-steps 4 \
  --trajectory-edges "$TEMPORAL_EDGES" --validation-trajectory-edges 0 \
  --trajectory-weight 0.1 --delta-weight 0.02 \
  --wind-speed-weight 0.01 --wind-direction-weight 0.005 \
  --temporal-warmup-epochs 5 --trajectory-selection-weight 0.1 --log-gradient-norms \
  --learning-rate 0.001 --weight-decay 0.0001 --early-stop-patience 8 \
  --seed 7 --device "$TEMPORAL_DEVICE" 2>&1 | tee "$TEMPORAL_RUN/b.console.log"
```

A parameters는 동결하지만 decoder 입력 q의 gradient는 유지합니다. 기존 FM/PI/전문화/균형/투영 항에 새 objective가 더해집니다. `--sampled-leads 2`는 FM sampling이며 trajectory block 크기와 다른 옵션입니다.

| 새 로그 | 의미 | 점검 |
|---|---|---|
| `loss_delta` | ensemble-mean tendency 오차 MSE | weight와 warmup을 함께 확인 |
| `loss_trajectory` | endpoint+increment joint fair Energy | full20인지 sub-block인지 기록 |
| `loss_wind_speed` | 파생 풍속의 fair CRPS | u/v 기본 loss와 중복 가중 고려 |
| `loss_wind_direction` | toward cos/sin mean 보조항 | truth 기반 calm mask 사용 |
| `temporal_output_grad_rms_*` | 변수별 새 loss endpoint gradient | finite/nonzero 여부, 크기 불균형 |

raw loss가 존재하는 것과 weighted contribution이 활성화되는 것은 다릅니다. B/C 콘솔/metadata의 weight·ramp와 함께 확인합니다. mean MSE에도 유한 member 수에서 variance penalty가 있으므로 가중치를 크게 올리는 것만으로 해결하려 하지 말고 0-weight ablation과 spread/coverage를 같이 봅니다.

## 7. C: best B 재로드, 작은 LR로 calibration

```bash
train-climate-manifold-moe --archive "$TEMPORAL_ARCHIVE" \
  --output "$TEMPORAL_RUN/c.pt" --stage joint --init-checkpoint "$TEMPORAL_RUN/b.pt" \
  --joint-epochs 10 --batch-size 2 --window-stride 4 \
  --ensemble-size 4 --sampled-leads 2 --integration-steps 4 \
  --trajectory-edges "$TEMPORAL_EDGES" --validation-trajectory-edges 0 \
  --trajectory-weight 0.1 --delta-weight 0.02 \
  --wind-speed-weight 0.01 --wind-direction-weight 0.005 \
  --temporal-warmup-epochs 5 --trajectory-selection-weight 0.1 --log-gradient-norms \
  --learning-rate 0.001 --joint-lr-factor 0.1 --encoder-lr-factor 0.1 \
  --weight-decay 0.0001 --early-stop-patience 5 --seed 7 --device "$TEMPORAL_DEVICE" \
  2>&1 | tee "$TEMPORAL_RUN/c.console.log"
```

C는 calibration split에서 기존 probabilistic score와 PI/anchor 경계를 유지합니다. manifold LR은 base×joint factor×encoder factor입니다. 각 단계의 `.pt`, `.manifest.json`, `.metadata.json`, `.metrics.json`을 함께 보관합니다. checkpoint는 best weights·통계·설정·상위 SHA를 보존하며 optimizer/RNG 중단점의 bitwise resume 파일은 아닙니다. 중단 후 같은 파일을 덮어쓰며 이어가는 resume 명령을 가정하지 마세요.

```bash
python - <<'PY'
import json, os
from pathlib import Path
root = Path(os.environ['TEMPORAL_RUN'])
for stage in ('a', 'b', 'c'):
    rows = json.loads((root / f'{stage}.metrics.json').read_text())
    print(stage, 'epochs:', len(rows))
    print('last train:', rows[-1]['train'])
    print('last validation:', rows[-1]['validation'])
PY
```

마지막 epoch 지표와 실제 best checkpoint 성능을 구분하세요. validation total 상승만으로 선택된 best도 같은 정도로 과적합됐다고 단정하지 않습니다.

## 8. Validation: state와 dynamics를 함께 평가

```bash
evaluate-climate-flow --checkpoint "$TEMPORAL_RUN/c.pt" --archive "$TEMPORAL_ARCHIVE" \
  --split validation --output "$TEMPORAL_RUN/validation.json" \
  --ensemble-size 8 --integration-steps 16 --max-cases 32 --seed 83 --device "$TEMPORAL_DEVICE"
python -m climate_diffusion.manifold_diagnostics \
  --checkpoint "$TEMPORAL_RUN/c.pt" --archive "$TEMPORAL_ARCHIVE" \
  --split validation --output "$TEMPORAL_RUN/validation-routing.json" --device "$TEMPORAL_DEVICE"
```

32 cases는 시작 검사이며 전체 split 평가가 아닙니다. persistence/climatology 대비 state RMSE, Energy/CRPS, coverage/spread-skill, 변수별 tendency RMSE·진폭 비, member/mean temporal spread, gate 사용률·후보 유사성·지역별 skill을 같이 확인하세요. 균등 usage만으로 전문화 성공을 뜻하지 않습니다. test 수치를 보고 계수나 epoch를 선택하지 않습니다.

## 9. 한 번 예측 → 동일 member 전부 렌더링

학습 metadata에서 validation 첫 origin을 선택합니다. 실제 운영의 최신 시각은 다른 목적이므로 여기의 정답 비교와 구분합니다.

```bash
export TEMPORAL_ORIGIN=$(python - <<'PY'
import os, torch
from climate_diffusion.moe_data import load_moe_archive
ck = torch.load(os.environ['TEMPORAL_RUN']+'/c.pt', map_location='cpu', weights_only=False)
_, times, _ = load_moe_archive(os.environ['TEMPORAL_ARCHIVE'])
i = ck['training']['split']['validation'][0] + ck['training']['history_span_steps'] - 1
print(times[i])
PY
)
diagnose-climate-time --checkpoint "$TEMPORAL_RUN/c.pt" --archive "$TEMPORAL_ARCHIVE" \
  --origin-time "$TEMPORAL_ORIGIN" --forecast-steps 20 --ensemble-size 8 \
  --integration-steps 16 --seed 83 --device "$TEMPORAL_DEVICE" \
  --forecast-output "$TEMPORAL_RUN/forecast-6h.npz" --forecast-only
render-climate-trajectories --forecast "$TEMPORAL_RUN/forecast-6h.npz" --archive "$TEMPORAL_ARCHIVE" \
  --output-dir "$TEMPORAL_RUN/members-6h" --horizon-hours 120 --interval-hours 6 --extension mp4
render-climate-trajectories --forecast "$TEMPORAL_RUN/forecast-6h.npz" --archive "$TEMPORAL_ARCHIVE" \
  --output-dir "$TEMPORAL_RUN/members-12h" --horizon-hours 120 --interval-hours 12 --extension gif
```

6h는 미래 20 frame, 12h는 미래 10 frame입니다. origin은 NPZ에 별도 저장되어 포함하면 각각21/11 state입니다. 두 출력은 같은 forecast에서 나오므로 member를 재추첨하지 않습니다. `--members 0 2`를 추가하면 일부 ID만 렌더링합니다.

`member-NNN.json`은 그 member, `summary.json`은 ensemble aggregate입니다. 첫 origin→+6h와 이후 구간을 분리하고 t2m/msl/u/v의 `amplitude_ratio`를 확인하세요. 작은 true tendency는 null/valid count로 해석합니다. lag는 진단일 뿐 ERA5 시간을 이동해 맞추지 않습니다. 고정 quiver는 Eulerian 풍속이며 입자 이동이 아닙니다. FPS/화살표 배율 변경은 학습 개선이 아닙니다.

## 10. 설정 고정 → 최종 test → 보존

```bash
evaluate-climate-flow --checkpoint "$TEMPORAL_RUN/c.pt" --archive "$TEMPORAL_ARCHIVE" \
  --split test --output "$TEMPORAL_RUN/test.json" \
  --ensemble-size 8 --integration-steps 16 --max-cases 32 --seed 83 --device "$TEMPORAL_DEVICE"
```

코드 commit, environment, archive hash/preflight, A/B/C와 각 sidecar, validation/test JSON, forecast NPZ, member JSON/영상, 실제 실행 명령을 하나의 실험으로 보존하세요. 다른 seed·loss·batch·full-window 비교는 새 실험 폴더로 만듭니다. 동일 split/A 초기화/member 수/적분 step을 고정해 비교하고 결과 파일만 바뀐 것을 새 학습으로 기록하지 않습니다.

## 자원·실패 대응·현재 범위

- 위 batch/epoch/계수는 시작 예산이며 최적값이 아닙니다. RTX4090에서 먼저 별도 출력의 1epoch로 시간·peak VRAM·gradient를 측정하세요. 실제 큰 archive는 host RAM과 A seal 단계 비용도 확인합니다.
- full20은 sub-block2보다 많은 ODE graph를 보존합니다. OOM이면 batch/member/적분 step을 하나씩 줄이고 변경을 기록하세요. A/C batch1로 우회하지 않습니다.
- loss NaN/시간 gap/단위·schema 불일치가 발생하면 입력 계약부터 수정하세요. 결측 label을 0으로 바꾸거나 test 통계로 normalization을 재계산하지 않습니다.
- 기존 H120 checkpoint의 120h prefix는 `--forecast-steps 20`으로 추출하되 trained condition `j/120`을 유지합니다. 새 H20 학습의 `j/20`과 weights를 혼용하지 않습니다.
- 생성 ODE 시간 τ, physical tendency per hour, u/v m/s는 서로 다릅니다. 이 버전은 lead-conditioned 생성이며 physical recurrent drift 연결은 별도 작업입니다.
- 이 매뉴얼 작성 중 새 ERA5 재학습을 실행하지 않았습니다. 기존 합성 검증과 사용자가 별도 수행한 [실제 ERA5 run-001](https://github.com/nayehyeon61-glitch/climate_diffusion/tree/results/temporal-120h-era5-run-001/docs/results/temporal-120h-era5-run-001)을 구분합니다. run-001의 약한 후속 변화량은 다음 recurrent 보강 실험의 비교 기준이며 위 명령만으로 해결됐다는 주장은 아닙니다.
