# 새 A+B 공동 학습 전체 실행 매뉴얼 — 6h × 20 = 120h

검토 기준: `feature/joint-ab-loss-v2`, 코드 `3fc532884b0758e6ff63849fe5296b2e139556fb`.
이번 문서/runner 보정은 모델·loss 구현을 변경하지 않습니다.

> **현재 장기 ERA5 재학습은 보류하세요.** 명령 연결과 모델 성능 검증은 다릅니다.
> 검토 중 AB V2 transition score의 **state 정규화 → 물리 단위 환산 누락**과
> `--log-gradient-norms` 실행 오류를 재현했습니다. 아래 명령은 현재 CLI에 맞춘
> pilot/검증 절차입니다. 단위 오류를 먼저 수정·재검증한 새 커밋에서 장기 실행하세요.
> 자세한 근거는 [검증 보고](../docs/results/joint-ab-loss-v2/manual-audit-2026-09-15.md)에 있습니다.

## 0. 가장 짧은 실행 순서

기존 run을 덮어쓰지 않는 **절대 경로**를 사용합니다. 다음은 이미 준비된 ERA5 archive가
있을 때의 순서입니다. 설치는 1절, archive가 없으면 2절부터 진행하세요.
runner는 한 단계씩만 실행하며, 기본값은 **A/AB/C 각각 1 epoch pilot**입니다.

```bash
GIT_LFS_SKIP_SMUDGE=1 git clone --single-branch --branch feature/joint-ab-loss-v2 \
  https://github.com/nayehyeon61-glitch/climate_diffusion.git climate_diffusion_joint_manual
cd climate_diffusion_joint_manual
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[test,plots,io]'

export ARCHIVE=/workspace/data/era5-temporal-6h.npz
export RUN=/workspace/experiments/joint-ab-v2-pilot-001
export JOINT_DEVICE=cuda
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

# 먼저 실행될 명령만 보기: 파일 생성/데이터 접근/학습 없음
DRY_RUN=1 bash scripts/run_joint_ab_120h.sh warmup "$ARCHIVE" "$RUN"
bash scripts/run_joint_ab_120h.sh prepare "$ARCHIVE" "$RUN"

# 알려진 loss 단위 문제를 승인된 수정 커밋에서 해결한 뒤 아래 pilot 진행
bash scripts/run_joint_ab_120h.sh warmup "$ARCHIVE" "$RUN"
bash scripts/run_joint_ab_120h.sh AB "$ARCHIVE" "$RUN"
bash scripts/run_joint_ab_120h.sh C "$ARCHIVE" "$RUN"
bash scripts/run_joint_ab_120h.sh validation "$ARCHIVE" "$RUN"
bash scripts/run_joint_ab_120h.sh forecast "$ARCHIVE" "$RUN"
bash scripts/run_joint_ab_120h.sh render "$ARCHIVE" "$RUN"
```

각 명령이 성공하고 아래 단계별 검사를 통과해야 다음 명령으로 넘어갑니다.
`prepare` 이후 코드를 바꾸면 runner가 거절하므로 **수정 커밋에는 새 RUN으로 prepare부터**
시작하세요. `test`는 이 목록에 자동 포함하지 않습니다. 설정을 확정한 뒤 8절에서 실행합니다.
이전 `bash scripts/run_joint_ab_120h.sh ARCHIVE RUN` 두 인자 일괄 실행은 폐기되었습니다.

## 1. 설치·환경 고정

Python 3.10 이상과 PyTorch가 필요합니다. RunPod/4090에서는 제공 이미지의 CUDA/driver와
호환되는 PyTorch를 사용하고 아래에서 실제 GPU 인식을 확인하세요. CPU wheel을 설치한
환경에 `JOINT_DEVICE=cuda`만 지정한다고 GPU가 활성화되지는 않습니다.

```bash
python -c 'import sys, torch; print(sys.version); print(torch.__version__, torch.version.cuda); print("CUDA:", torch.cuda.is_available())'
command -v ffmpeg
python -m pytest -q
bash -n scripts/run_joint_ab_120h.sh
```

실제 도움말은 다음 명령입니다.

```bash
bash scripts/run_joint_ab_120h.sh --help
python -m climate_diffusion.train_manifold_moe --help
python -m climate_diffusion.time_alignment --help
python -m climate_diffusion.trajectory_output --help
```

MP4에는 ffmpeg가 필요합니다. Debian/Ubuntu의 설치 권한이 있다면
`apt-get update` 후 `apt-get install -y ffmpeg`를 실행합니다. GIF는 Pillow를 사용합니다.
`plots`에는 matplotlib/Pillow, `io`에는 NetCDF/xarray 입출력 관련 의존성이 포함됩니다.
원격 LFS 과거 영상은 이번 학습의 필수 입력이 아니므로 clone에서 smudge를 건너뜁니다.

입력: 새 clone, 기존 ERA5 파일. 출력: 패키지가 설치된 환경.
통과 조건: import/pytest 성공, GPU를 쓸 때 CUDA=True, MP4를 쓸 때 ffmpeg 존재.
실패 시: 환경부터 고치고 학습하지 않습니다. GPU/유료 자원을 자동 생성하지 않습니다.

## 2. ERA5 archive 준비 → 통계·시간·split 검사

### 2.1 원자료에서 archive를 만들 때만

변수는 `msl`(Pa), `t2m`(K), `u10`, `v10`(m/s) 전체입니다. 이미 만든 동일 archive가 있으면
변환하지 말고 2.2로 갑니다. 아래 grid는 pilot 후보이며 검증된 최적 해상도가 아닙니다.

```bash
export FIELDS=/workspace/data/era5-fields.nc
test ! -e "$ARCHIVE"
prepare-climate-fixed-step-data --fields "$FIELDS" \
  --variables msl t2m u10 v10 --step-hours 6 \
  --target-lat-points 18 --target-lon-points 36 --output "$ARCHIVE"
```

원자료 경로/기간은 사용자가 가진 파일로 대체합니다. 변수 이름만 같다고 단위가 맞는 것은
아닙니다. 원자료 metadata가 없으면 canonical 단위를 가정하는 경로가 있으므로 hPa/섭씨를
Pa/K로 자동 변환해 준다고 믿지 마세요. schema/원자료 단위를 먼저 확인합니다.
coarsening 이후 fully observed pooled cell만 허용하며 결측·불규칙 시간은 fail-fast합니다.

### 2.2 새 RUN 고정

```bash
bash scripts/run_joint_ab_120h.sh prepare "$ARCHIVE" "$RUN"
python scripts/inspect_joint_run.py --archive "$ARCHIVE" --preflight "$RUN/preflight.json"
```

출력: `preflight.json`, `calendar.log`, `code-commit.txt`, `environment.txt`.
실제 archive의 첫/끝 UTC, split별 origin/target UTC, unique train pair 수가 표시됩니다.
연도나 평가 origin을 임의로 고정하지 않습니다. 같은 RUN에 prepare를 다시 하지 마세요.

| 계약 | 값·검사 |
|---|---|
| 예측 시간 | archive exact 6h, horizon_steps=20, 120h=5일 |
| history | 6개 관측, stride=4 archive steps: origin 포함 24h 간격, 과거 span=120h |
| sample | normalized history `[6,D]`, origin `[D]`, targets `[20,D]`; raw trajectory `[21,D]`, delta/tendency `[20,D]` |
| batch | trajectory `[B,21,D]`, generated `[B,M,21,D]`; `D=C×H×W`이며 grid는 schema 사용 |
| 첫 차분 | origin→+6h; 마지막은 +114→+120h; actual dt_hours 사용 |
| mask | 양 endpoint 관측 필요; archive 결측을 미래 정답으로 보간하지 않음 |
| split | train → expert_validation → calibration → validation → test, future target 겹침 제거 |
| purge | 기본 경계 embargo는 H−1=19 window 이상; causal history의 과거 split 겹침은 허용 |
| 통계 | train 범위 고유 관측·인접쌍에서만 state mean/scale, tendency scale/floor·면적·변수 가중 계산 |

통과 조건: exact 6h, 유효한 다섯 split, canonical 변수/단위, 충분한 표본, finite 통계.
archive SHA256/schema가 고정돼야 합니다. 데이터가 짧으면 split을 억지로 겹치지 말고 기간을 늘립니다.
풍향은 u/v에서 파생하며 calm-mask/풍속 scale은 train 통계를 사용합니다.
미래 trajectory/delta는 감독용이며 history/router conditioning에 넣지 않습니다.

## 3. Warmup A — 새 초기화, best 선택, 좌표 seal

```bash
JOINT_WARMUP_EPOCHS=1 bash scripts/run_joint_ab_120h.sh warmup "$ARCHIVE" "$RUN"
```

입력: archive/preflight. 출력: `warmup.pt`, `.metadata.json`, `.metrics.json`, `.manifest.json`,
`warmup.console.log`, `preflight-warmup.json`. 학습기는 처음부터 새 모델을 초기화합니다.

runner는 `--stage manifold --forecast-dynamics recurrent_residual --horizon-steps 20`
및 history6/stride4를 명시합니다. trainer에는 `--step-hours` flag가 없고 archive에서 6h를 읽습니다.
기존 H120(720h) checkpoint를 H20 모델로 묵시 재해석하지 않습니다.

reconstruction/physics/invariant/metric/latent dynamics에
`--ae-delta-weight .05 --finite-step-drift-weight .05`를 추가합니다. 이 값은 실행 후보이지 최적값이
아닙니다. validation은 `expert_validation`을 사용합니다. best 가중치를 reload한 뒤 **train-only**
latent mean/scale, chart center/radius, reference encoder를 한 번 seal합니다.

통과 조건: preflight-warmup의 `checkpoint_verified=true`, finite loss, best epoch 확인,
변수별 state/AE 변화량/decoded drift가 zero-tendency·persistence와 비교해 타당한지 별도 검사.
파일이 저장됐다는 것만으로 A의 dynamics 품질을 통과한 것은 아닙니다. 자동 품질 gate의
임계값을 지정하지 않은 run은 검증된 품질 gate 통과라고 보고하지 않습니다.

## 4. A+B 공동 최적화 — A를 삭제하거나 영구 동결하지 않음

```bash
JOINT_AB_EPOCHS=1 JOINT_LOSS_PROFILE=v2_minimal \
  bash scripts/run_joint_ab_120h.sh AB "$ARCHIVE" "$RUN"
```

입력: `warmup.pt` best. 출력: `ab.pt`와 sidecar/`ab.console.log`.
`--stage joint_ab`는 encoder/decoder/drift + full-state experts/gate/history를 공동 학습합니다.
base LR=.001, manifold LR factor=.3, anchor=.05, weight decay=.0001, seed7입니다.
LR/epoch/계수는 baseline 후보이며 4090이나 ERA5에 최적이라고 단정하지 않습니다.

| loss 항 | ab_control | v2_minimal(기본) | v2_full |
|---|---:|---:|---:|
| fused FM / expert FM | 1 / 1 | 1 / 1 | .3 / 1 |
| state CRPS / Energy | .5 / .5 | 0 / 0 | 1 / 0 |
| transition CRPS | 0 | .25 | .75 |
| joint trajectory Energy | .1 | .3 | .75 |
| mean state / tendency | 0 / .02 | 0 / .02 | .1 / .1 |
| decoded AE delta / finite-step drift | 0 / 0 | .05 / .05 | .05 / .05 |
| member tendency MSE / spread calibration | 0 / 0 | 0 / 0 | 0 / 0 |

기존 PI geometry/specialization/anchor 항도 유지합니다. **generic `--delta-weight`,
`--trajectory-weight`, `--wind-*-weight`는 AB profile의 계수를 바꾸지 않습니다.**
`--temporal-warmup-epochs`도 현재 AB profile score에 ramp를 적용하지 않습니다.
선택적 calibration은 기본 off이며 현재 pooling/finite-M 정책 검토가 끝나지 않았습니다.

한 recurrent rollout `[B,M,21,D]`에서 V2 score를 계산합니다. teacher-forced FM pass는 별개입니다.
같은 member의 persistent noise, 현재 q를 모든 expert가 공유합니다. 매 tau step에서 후보 field를
결합하고 residual endpoint(q/day)와 drift(q/day)를 더해 `dt_hours/24`로 physical Euler 적분합니다.
다음 입력은 **생성 q_next**이며 origin history는 고정입니다. 출력은
`origin + decode(q_j) - decode(q_origin)`입니다. 생성 `dr/dtau`, physical `ds/dhours`, 풍속m/s는 다릅니다.

FM label은 현재 encoder 좌표로 재계산한
`stopgrad((encode(next)-encode(prev))/(dt/24)-drift(encode(prev)))`입니다.
label만 detach하며 conditioning q 및 rollout/decode/Jacobian gradient는 유지합니다. EMA는 없습니다.
AB에서 affine/charts를 재fit하거나 최종 reseal하지 않습니다. reference encoder는 warmup 때의 것입니다.
stop-gradient 하나만으로 latent collapse가 방지되는 것은 아닙니다.

`JOINT_EDGES=0`(기본)은 full20 physical-step BPTT와 전체 endpoint/increment score입니다.
양수는 contiguous score sub-block이지만 origin부터 해당 블록까지 recurrent prefix도 생성하므로
비용이 edge 수에만 비례하지 않습니다. AB의 validation 역시 현재 `trajectory_edges`를 사용하므로
`validation_trajectory_edges=0`만으로 양수 block의 AB validation이 full20이 되지는 않습니다.
공식 비교에는 양쪽 모두0을 사용합니다. 단순 noise 공유만으로 joint law를 입증하지 않습니다.

통과 조건: AB best/checkpoint parent hash, 고정 affine/chart, finite raw/weighted loss,
latent 규모/AE·drift 품질, state와 uncertainty 지표를 함께 확인합니다.
**현재 V2 단위 오류는 먼저 수정해야 합니다.** `--log-gradient-norms`는 재현된 오류 때문에
runner에서 제외했습니다. gradient 기여가 측정됐다고 간주하지 마세요.

## 5. C — 기존 calibration 보정 경로(AB Loss V2 전체가 아님)

```bash
JOINT_C_EPOCHS=1 bash scripts/run_joint_ab_120h.sh C "$ARCHIVE" "$RUN"
```

입력: `ab.pt`. 출력: `c.pt`와 sidecar, `c.console.log`, `checkpoint-audit.log`.
C는 `calibration`에서 학습하고 `validation`에서 best를 고릅니다. AB를 시작점으로 하지만
`--stage joint`는 **legacy C objective**입니다. `--loss-profile v2_full`을 추가해도 C가 V2가 되지 않습니다.

| 단계 | 학습 parameter | LR(현재 runner) | 주요 loss/통계 정책 |
|---|---|---|---|
| A | manifold encoder/decoder/drift | .001 | geometry + decoded delta/drift; best 뒤 train seal |
| AB | manifold + experts/gate/history | manifold .0003 / 나머지 .001 | 선택 V2 + geometry/anchor; 고정 affine/chart |
| C | manifold + experts/gate/history | manifold .00001 / 나머지 .0001 | legacy 보정 + anchor1; affine/chart/reference warmup 유지 |
| legacy B 비교 | experts/gate/history만 | .001 | A manifold frozen, specialize 경로 |

C runner는 marginal Energy=.5/CRPS=.5, mean delta=.02, trajectory Energy=.1,
wind speed=.01/direction=.005, member tendency MSE=0을 **명시적으로** 지정합니다.
temporal 항에는 5epoch ramp가 있으므로 1epoch pilot은 최종 가중치에 도달하지 않습니다.
C에는 marginal sampling과 temporal sampling이 별도로 있고 V2 transition CRPS/mean-state profile을
그대로 쓰지 않습니다. AB의 decoded AEdelta/drift profile 계수도 C auxiliary PI에 그대로 전달되지 않습니다.

통과 조건: parent가 AB이고 `delta_member_weight=0`, temporal loss가 로그에서 nonzero/활성,
best epoch가 실제 checkpoint에 저장됨, state/coverage/spread가 검토 가능함.
AB보다 C가 반드시 좋다는 보장은 없습니다. `ab.pt`를 보존해 같은 조건으로 비교합니다.

## 6. Validation — 숫자와 member trajectory를 함께 검사

```bash
bash scripts/run_joint_ab_120h.sh validation "$ARCHIVE" "$RUN"
python scripts/inspect_joint_run.py --archive "$ARCHIVE" \
  --preflight "$RUN/preflight.json" --checkpoint "$RUN/c.pt"
```

출력: `validation-ab.json`, `validation-c.json`, `validation-routing.json`, `ab-losses.png`.
동일 seed83/M4/tau4/최대4origin 조건입니다. 더 넓은 검증을 하려면 첫 실행 전에
`JOINT_EVAL_MEMBERS`, `JOINT_EVAL_TAU`, `JOINT_EVAL_CASES`를 정하고 provenance와 함께 기록하세요.
기존 JSON을 덮어쓰지 않습니다. 추가 비교는 별도 출력 이름으로 evaluation CLI를 실행합니다.

검사: state RMSE/CRPS/Energy, `temporal_overall`의 변화율·trajectory 지표,
coverage/spread-skill, routing/geometry/candidate 유사도, 변수별 member tendency 및 bias.
V2 transition 관련 평가도 현재 단위 오류가 있어 보정 전 성능 근거로 사용하면 안 됩니다.
`ab-losses.png`는 현재 raw loss/selection 곡선입니다. weighted loss는 metrics JSON으로 확인하며
module-gradient graph는 현재 검증 미완료입니다. 넓은 spread만으로 성공이라고 판단하지 않습니다.

AB의 selection은 고정 score 조합(state Energy+state CRPS+transition CRPS+.1 trajectory Energy),
C는 legacy selection(Energy+CRPS+선택 가중 trajectory)입니다. 서로 다른 selection/total의 크기를
그대로 성능 비교하지 말고 같은 평가 정의·split·member/tau/seed로 비교합니다.
마지막 epoch 로그와 저장된 best epoch를 구분합니다.

## 7. 한 번 예측 저장 → 동일 member 전체 6h/12h 영상

```bash
bash scripts/run_joint_ab_120h.sh forecast "$ARCHIVE" "$RUN"
bash scripts/run_joint_ab_120h.sh render "$ARCHIVE" "$RUN"
```

입력: `c.pt`, archive의 첫 유효 validation origin(자동 추출). 출력: `forecast-6h.npz` **한 개**.
특정 시각은 archive 내부 UTC로 `JOINT_ORIGIN=...`을 설정합니다. 끝에서 120h 정답이 없는 시각은
비교용으로 쓰지 마세요. 렌더러에는 checkpoint가 아니라 **저장한 NPZ**를 넣습니다.

```bash
# 원한다면 이미 저장한 같은 forecast를 GIF로도 출력(새 폴더)
python -m climate_diffusion.trajectory_output --forecast "$RUN/forecast-6h.npz" \
  --archive "$ARCHIVE" --output-dir "$RUN/members-12h-gif" \
  --horizon-hours 120 --interval-hours 12 --extension gif --reference-label 'Actual ERA5'
```

`members-6h/`, `members-12h/`에 모든 member별 MP4+JSON, summary JSON이 저장됩니다.
기본은 ensemble mean이 아니라 member입니다. +6…+120h는20미래프레임,
+12…+120h는10미래프레임입니다. origin을 포함한 시계열 상태 수는 각각21/11이나
영상은 미래 프레임만 표시합니다. 12h 출력은 같은 6h 예측의 선택이며 재추론/재학습이 아닙니다.
온도K/검정 u-v quiver/같은 colorbar와 vector scale/m/s key/origin/valid time을 씁니다.
고정 격자의 Eulerian 화살표는 위치가 움직이는 입자 궤적이 아닙니다.

통과 조건: 두 출력이 같은 forecast SHA/member identity를 사용, 정확한 valid_time,
120h 끝점, 전 변수 finite, JSON에서 small-truth tendency ratio의 N/A/유효표본 수 확인.
FPS/벡터 확대를 dynamics 개선이라고 해석하지 않습니다.

## 8. 설정 고정 후 final test

```bash
# validation으로 선택한 설정과 AB/C 비교 계획을 고정한 뒤 한 번만
JOINT_TEST_CONFIRMED=1 bash scripts/run_joint_ab_120h.sh test "$ARCHIVE" "$RUN"
```

출력: `test-ab.json`, `test-c.json`. 동일 조건의 사전 지정 비교입니다.
test 결과로 loss/epoch/member 수를 다시 고르지 않습니다. 최종 모델 선택은 validation에서 합니다.
checkpoint/manifest/metrics와 archive/code SHA, 실행 환경, seed, 모든 CLI/console log를 함께 보존합니다.
사용자 ERA5 결과와 합성 테스트의 수치를 섞지 않습니다.

## 9. 중단/재개와 구버전 비교

optimizer/scheduler/RNG state를 저장하지 않으므로 **exact resume는 지원하지 않습니다.**
새 output 이름이 필요하며, 현재 `--init-checkpoint`는 아래 단계 전이만 허용합니다.

| 실행 stage | 허용 init stage | 용도 |
|---|---|---|
| manifold | 없음 | 새 A 초기화 |
| joint_ab | manifold | best A에서 새 optimizer로 AB 시작 |
| specialize | manifold | frozen A→B 비교 |
| joint | specialize 또는 joint_ab | best B/AB에서 새 optimizer로 C 시작 |

**AB→AB, C→C 이어 학습은 현재 거절됩니다.** 임의로 stage metadata를 고치지 마세요.
중단된 AB는 원래 `warmup.pt`에서 새 폴더로 AB를 다시 시작하는 것이며 이어 학습이 아닙니다.
중단된 C는 원래 `ab.pt`에서 C를 다시 시작합니다. run runner는 기존 log/output을 보호하므로
부분 실행 복구는 별도 새 output 이름의 명시적 trainer 명령을 사용합니다. 예:

```bash
python -m climate_diffusion.train_manifold_moe --archive "$ARCHIVE" \
  --init-checkpoint "$RUN/warmup.pt" --output "$RUN/ab-restart-001.pt" --stage joint_ab \
  --joint-ab-epochs 1 --loss-profile v2_minimal --manifold-lr-factor .3 --anchor-weight .05 \
  --ensemble-size 4 --integration-steps 4 --sampled-leads 2 --trajectory-edges 0 \
  --validation-trajectory-edges 0 --delta-member-weight 0 --batch-size 2 \
  --window-stride 4 --max-validation-windows 4 --seed 7 --device "$JOINT_DEVICE"
```

기존 frozen A→B 비교는 **별도 파일**에 실행합니다. H20이 명시된 같은 warmup을 재사용할 수 있습니다.

```bash
python -m climate_diffusion.train_manifold_moe --archive "$ARCHIVE" \
  --init-checkpoint "$RUN/warmup.pt" --output "$RUN/legacy-b.pt" --stage specialize \
  --expert-epochs 1 --ensemble-size 4 --integration-steps 4 --sampled-leads 2 \
  --delta-weight .02 --trajectory-weight .1 --trajectory-edges 0 --validation-trajectory-edges 0 \
  --delta-member-weight 0 --batch-size 2 --window-stride 4 --max-validation-windows 4 \
  --seed 7 --device "$JOINT_DEVICE"
```

legacy C는 5절의 실제 C 명령에서 init을 `legacy-b.pt`, output을 `legacy-c.pt`로 바꿉니다.
AB/C를 새 모델로, 이전 weights 재사용을 from-scratch로 바꿔 부르지 않습니다.
기존 H120 checkpoint의120h prefix는 trained H120 의미를 유지해야 하며 새 H20 warmup parent로
쓰지 않습니다. 이 runner는 새 H20 profile만 다룹니다.

## 10. RTX4090 pilot과 전체 학습 예산

4090 실측 VRAM/소요시간은 **이번 환경에서 측정하지 않았습니다.** CPU 검증을 GPU 예산으로
환산하지 않습니다. 첫 pilot은 위 기본1epoch, batch2/M4/tau4/full20입니다. batch 마지막
singleton 병합 때문에 실제 마지막 batch는 설정값+1이 될 수 있습니다.

```bash
# loss 단위 보수/재검증 후 별도 새 RUN에서만 시작할 장기 후보(최적 설정 아님)
export RUN=/workspace/experiments/joint-ab-v2-full-001
export JOINT_WARMUP_EPOCHS=8 JOINT_AB_EPOCHS=30 JOINT_C_EPOCHS=10
export JOINT_LOSS_PROFILE=v2_minimal JOINT_BATCH=2 JOINT_EDGES=0
bash scripts/run_joint_ab_120h.sh prepare "$ARCHIVE" "$RUN"
bash scripts/run_joint_ab_120h.sh warmup "$ARCHIVE" "$RUN"
bash scripts/run_joint_ab_120h.sh AB "$ARCHIVE" "$RUN"
bash scripts/run_joint_ab_120h.sh C "$ARCHIVE" "$RUN"
# 이후 6→7→8절을 순서대로 실행
```

학습기 metrics/runtime에서 `runtime_seconds`, `max_rss_kib`, `cuda_peak_memory_bytes`를
확인합니다. OOM이면 (1) 다른 GPU process 확인 (2) batch2 유지 (3) 별도 pilot에서
`JOINT_EDGES=2` score block을 실험하고 full20과 구분 (4) CPU/작은 grid의 계약 검증으로 되돌립니다.
sub-block도 prefix 비용이 있고 C는 별도 marginal rollout이 있으므로 OOM 해결을 보장하지 않습니다.
M/tau/grid/loss를 동시에 바꾸지 말고 변경한 조건을 기록합니다. 새 유료 자원 구매/장기 실행은 자동화하지 않습니다.

## 현재 남은 보수 경계

1. **AB V2 물리 tendency 단위 환산**: normalized state diff를 physical tendency scale로 나누기 전
   state_scale 복원이 누락됨. transition CRPS/mean tendency/joint increment score 및 해당 평가에 영향.
2. **gradient logging runtime**: parameter generator 소진으로 여러 loss/group의 autograd 검사 실패.
   runner는 flag를 제외했으며 module별 gradient 예산 검사는 미완료입니다.
3. mean MSE/trajectory Energy의 차원별 추가 정규화, 선택적 spread calibration의 pooling/finite-M
   정책은 별도 검토 필요. calibration은 현재 모든 profile에서0입니다.
4. C는 legacy loss, AB profile ramp 없음, 동일 stage exact resume 미지원.
5. 실제 ERA5 from-scratch 장기 학습/성능·coverage 개선과 4090 메모리 검증은 이번 문서 작업에서 미실행.

[학습/gradient Mermaid](../struct-picture/15-joint-ab-loss-v2.md) ·
[이번 CLI·단위 검증 근거](../docs/results/joint-ab-loss-v2/manual-audit-2026-09-15.md).
