# A 보강형 분리 학습: 처음부터 실행하는 매뉴얼

브랜치: `feature/a-manifold-information-process`. 기반은 **4dd9bf4의 분리형 A→B→C**다.
`joint_ab`/Loss V2 브랜치를 사용하거나 A와 B를 합치지 않았다. 기존 trainer/CLI/weights는 그대로다.
새 경로는 별도 checkpoint 형식 `climate_diffusion.separate_a_information_process.v1`을 사용한다.

## 0. 복사해서 실행할 순서

아래 경로는 본인의 실제 파일로 바꾼다. 추가 변수 NetCDF는 **surface archive와 동일한 시각·격자**여야 한다.
이 스크립트는 ERA5를 다운로드하거나 RunPod/GPU를 생성하지 않는다.

```bash
bash <<'BASH'
set -euo pipefail
# 이 두 원본 데이터는 먼저 준비한다. 다운로드/자동 재격자화 명령이 아니다.
export ARCHIVE=/workspace/data/era5-temporal-6h.npz
export INFO_FIELDS=/workspace/data/era5-extra-aligned.nc
export INFO=/workspace/data/era5-information-v1.npz
RUN_TAG="$(date -u +%Y%m%dT%H%M%SZ)"
export RUN="/workspace/experiments/a-information-full-${RUN_TAG}"

GIT_LFS_SKIP_SMUDGE=1 git clone --single-branch --branch feature/a-manifold-information-process \
  https://github.com/nayehyeon61-glitch/climate_diffusion.git "climate_A_information_${RUN_TAG}"
cd "climate_A_information_${RUN_TAG}"
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test,plots,io]'
if ! command -v ffmpeg >/dev/null 2>&1; then
  if [ "$(id -u)" -eq 0 ]; then
    apt-get update
    apt-get install -y ffmpeg
  else
    sudo apt-get update
    sudo apt-get install -y ffmpeg
  fi
fi
python -c 'import torch; print(torch.__version__); assert torch.cuda.is_available(), "CUDA 환경을 확인하세요"'

export MODE=enriched PROFILE=process DEVICE=cuda
export M=4 TAU=4 BATCH=2
export MAX_WINDOWS=0 WINDOW_STRIDE=1 SEED=7 LR=0.001
export A_EPOCHS=60 CURRICULUM_INTERVAL=4 B_EPOCHS=30 C_EPOCHS=10
export B_MEMBER_WEIGHT=0.001 EVAL_CASES=0 AUDIT_SPLIT=expert_validation
mkdir -p "$RUN"
run_stage() { bash scripts/run_a_information_120h.sh "$1" 2>&1 | tee "$RUN/$1.log"; }

run_stage preflight
run_stage A
run_stage audit
python -m json.tool "$RUN/a-audit.json"
read -r -p "A 감사 결과 확인 후 B→C를 진행하려면 yes: " answer </dev/tty
if [ "$answer" != yes ]; then echo "A까지 저장: $RUN"; exit 0; fi
run_stage B
run_stage C
run_stage validation
run_stage render
echo "학습·validation·member 영상: $RUN"
# Test는 계수 선택에 사용하지 않는다.
read -r -p "설정을 확정하고 최종 test를 실행하려면 yes: " answer </dev/tty
if [ "$answer" = yes ]; then run_stage test; fi
echo "결과 위치: $RUN"
BASH
```

기본 epoch A60/B30/C10, A curriculum 간격4는 **실행 가능한 후보**이지 ERA5 검증 최적값이 아니다.
위 블록은 전체 origin stride1, 학습 window 상한0(제한 없음), 전체 evaluation case0을 명시한다.
runner 단독 기본값은 stride4/EVAL_CASES32다. 전체 평가도 겹치는 origin을 포함하므로 독립 사건 수가 아니다.
`read`는 interactive terminal에서만 사용한다. 비대화형 작업에서는 단계별 명령을 따로 실행하고
A 감사/최종 test 승인은 작업자가 별도로 결정한다. 모델은 **6h×20=120h**로 고정된다.
선행 synthetic 확인은 GPU 없이 다음으로 실행한다. 출력 폴더는 반드시 새 이름을 사용한다.

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python scripts/smoke_information_process.py \
  --output outputs/a-info-smoke-new --report outputs/a-info-report-new
python -m pytest -q
```

## 1. 입력 준비 / 검증

surface 출력은 기존 `msl(Pa), t2m(K), u10/v10(m/s)` 네 변수다. 추가 정보는 출력 차원을 늘리는 것이 아니라
**별도 sidecar conditioning**이다. msl을 sidecar에 또 넣지 않는다.

새 surface archive가 필요한 경우, canonical `time,lat,lon` NetCDF에서 기존 준비기를 사용한다.
격자 크기는 예시이며 실제 원본과 비교 기준에 맞춰 결정한다. 기존 4×16×32=2048 archive와 18×36을 혼동하지 않는다.

```bash
python -m climate_diffusion.fixed_step_data --fields /workspace/data/surface.nc \
  --variables msl t2m u10 v10 --step-hours 6 \
  --target-lat-points 16 --target-lon-points 32 --output "$ARCHIVE"
```

enriched 필수: `z850,z500,z250,u850,v850,terrain_height`; `terrain_slope`는 준비기가 계산한다.
상층 이름은 각 이름의 3D 변수이거나 `z/u/v`와 `pressure_level/level/isobaricInhPa` 차원을 사용할 수 있다.
pressure 좌표에는 **Pa/hPa/millibars** 단위를 반드시 선언한다.

| 항목 | 입력/내부 단위 및 검증 |
|---|---|
| z850/500/250 | geopotential `m2 s-2`, `m**2 s**-2`, `m^2/s^2`는 9.80665로 나눠 height m로 변환; m도 허용 |
| terrain_height | static `[lat,lon]`, m 또는 위 geopotential 단위; pressure-level height와 구별 |
| terrain_slope | 구면 지구 반경 6371000m, 북/동 거리 미분 크기; dimensionless; 경도 주기경계 |
| u850/v850 | m/s, `m s-1`, `m s**-1`; 내부 m/s |
| 옵션 | t850,t500(K), u500,v500(m/s), sst(K), q850(질량분율); 없는 변수는 생성하지 않음 |
| time/grid | dynamic은 exact UTC 6h 및 surface와 좌표/시간 완전 동일; static에는 time 차원 없음 |
| mask | sidecar와 surface 모두 fully observed; 결측/시간 gap/격자 불일치는 fail-fast |

terrain slope는 최소3×3, 위도 오름차순, 중복끝점 없는 등간격 전지구 경도, 정확한 극점 없는 격자를 요구한다.
재격자화는 준비 전 사용자가 결정하며 이 코드가 묵시 보간하지 않는다. 저장된 static 배열이 시간에 따라 변해도 거부한다.
optional 변수는 다음처럼 명시한다(기본 runner는 필수만 준비).

```bash
python -m climate_diffusion.physical_information --archive "$ARCHIVE" \
  --fields "$INFO_FIELDS" --output "$INFO" --optional t850 t500 u500 v500 sst q850
```

`preflight.json`은 surface를 검증한다. sidecar는 preflight와 A/B/C load 시 hash/단위/정합을 재검증한다.
`code-commit.txt`, `environment.txt`에 실행 commit과 package 환경을 기록한다.
파일이 없거나 단위가 모호하면 중단하고 실제 데이터를 준비한다. **surface-only baseline**은 `MODE=surface`로 실행하며
상층 정보를 사용했다고 보고하면 안 된다. 이미 있는 checkpoint/output을 덮어쓰지 않는다.

## 2. 분할·shape·통계

- history `[B,L,4HW]`, origin `[B,4HW]`, trajectory `[B,21,4HW]`, delta/tendency `[B,20,4HW]`.
- 추가 정보 `C_origin[B,FHW]`, future information labels `[B,20,FHW]`; 후자는 loss에서만 사용한다.
- history 기본6시점·stride4이므로 120h history span. 마지막 history가 origin이다.
- 모든 예측은 **6h×20=120h(5일)**. 120 step/30일 checkpoint를 새 의미로 읽지 않는다.
- window origin index = start + history_span_steps − 1. 첫 차분은 origin→+6h, 마지막은 +114→+120h.
- train/expert_validation/calibration/validation/test five-way chronological split과 기존 purge 규칙을 재사용한다.
  미래 target 구간은 분리한다. history/경계의 관측 origin까지 전부 격리하는 별도 사건 독립 split은 아니다.
- state mean/scale은 고유 train 시간 구간에서 기존 per-cell 방식으로 fit. tendency는 같은 구간의 고유 인접 관측쌍,
  실제 dt, 면적가중·변수별 scale floor로 fit. sidecar는 train 공간/시간 면적가중 per-channel 통계와 floor.
  겹치는 window를 여러 번 세어 통계를 fit하지 않는다. 실제 연도는 archive timestamp와 checkpoint split에서 확인한다.
- B/C는 A의 split/statistics/hash를 그대로 상속하며 archive/sidecar가 바뀌면 거부한다.

## 3. A: deterministic 정보 보존 + 실제 stochastic process 보조 학습

`z=E(DCT(x_norm))+I(C_origin_norm)`, `q=(z−mu_z)/sigma_z`, surface decoder `D(z)`.
Origin의 C를 **history의 각 encoding과 전체 120h**에 고정한다. 미래 상층 정답으로 conditioning을 갱신하지 않는다.
추가 MLP는 I, 정보 복원 head, 작은 A-context, 작은 A residual-FM sampler다. 대형 backbone은 추가하지 않았다.

A의 auxiliary sampler는 **B experts가 아니다**. A에서는 B experts/gate/history optimizer가 완전히 꺼져 있다.
A의 실제 확률 경로:

```text
z_j, 고정 origin history/C, member별 persistent 독립 noise
→ raw-z/day residual 공간의 tau ODE
→ residual endpoint R_j
→ z_(j+1) = z_j + (dt_hours/24) × [b(z_j) + R_j]
→ D(z_(j+1)) + [x_origin − D(z_origin)]
```

Teacher-forced FM label은 `stopgrad((z_next−z_current)/(dt/24)−b(z_current))`이다.
source N(0,residual_noise_std² I), path `(1−tau)source+tau×label`, target `label−source`.
정답은 target에만 들어간다. 생성 rollout은 항상 이전 예측 state를 다음 입력으로 쓴다.
FM의 `dr/dtau`, physical `dz/day`, 출력의 `ds/hour`, 풍속 m/s는 다르다.
state decoder에 latent velocity를 state인 것처럼 넣지 않는다. drift loss는 **실제 6h Euler 후 decoded secant**다.

### 여섯 curriculum phase

| phase | 처음 활성화되는 항 (기본 plateau 계수) |
|---|---|
| 1 | reconstruction1, forecast-anchor .1, physics .1, invariant .05, metric .1, static L2 .05, dynamic 정보 복원 .05 |
| 2 | latent-dynamics .1, decoded AE tendency .05, finite-step decoded drift tendency .05 |
| 3 | A residual FM1, stochastic surface state fair CRPS .25 |
| 4 | paired 정보 거리 alignment .02, dynamic 정보 state/transition fair CRPS 평균 .1 |
| 5 | surface transition fair CRPS .25, ensemble-mean tendency MSE .02 |
| 6 | 동일 member endpoint+increment fair trajectory Energy .1 |

phase는 `1+(epoch−1)//interval`을6으로 제한한다. 이는 부드러운 ramp 대신 **명시적 단계 activation**이다.
모든 phase에서 단일 full20 recurrent graph를 계산하며 detach/truncated BPTT하지 않는다.
최소 epoch는 `5×interval+1`. process profile은 **phase6 도달 후 epoch만 best 후보**로 선택한다.
활성화 계수 수정 예: `A_LOSS_WEIGHTS='{"static_l2":0.01,"transition_crps":0.1}'`; 0 ablation도 가능하다.
이는 최적 계수 추천이 아니다. `--profile baseline/dynamics/information`은 새 wrapper 내부 ablation이며
과거 ERA5 baseline checkpoint를 재현하는 명령이 아니다.

### 확률 loss와 중복/붕괴에 대한 한계

fair CRPS = mean_m |sample_m−truth| − sum_(m≠n)|sample_m−sample_n|/[2M(M−1)]. M<2는 거부한다.
같은 member의 두 physical endpoint를 빼서 transition score를 계산한다. normalized state 차분에 state_scale을 곱하고
actual dt와 train-only tendency_scale로 나눈다. delta와 tendency를 독립 손실로 중복 가중하지 않는다.
한 조건·한 구간의 state와 delta는 알려진 이전 state의 translation 관계이므로 통계적 중복이 있으며 계수로 조절한다.
CRPS는 marginal 점수이고 joint path 의존성을 충분히 식별하지 않는다. endpoint+increment fair Energy가 이를 보완하지만
고차원에서 모든 의존성 오류를 잘 검출한다는 보장은 없다. pressure는 네 변수 중 한 번 가중하고 msl/Δmsl을 따로 로그한다.

static은 **학습된 정보 head의 복원값**과 관측 terrain을 비교한다. 입력을 그대로 복사해서0이 되는 loss가 아니다.
그러나 전지구 단일 지형을 암기할 수 있으므로 이것만으로 지형 물리 제약/일반화를 입증하지 않는다.
alignment는 고정 관측 C의 **짝이 맞는 pair distance(stopgrad target)**를 사용한다. 두 projector의 무조건 MMD나 histogram
matching을 쓰지 않는다. collapse를 막는 수학적 보장은 아니며 state/info reconstruction, latent variance, gradient도 함께 본다.
정적 지형 거리 그 자체는 시간 차이를 만들지 않으며 다른 동적 정보/기상장 감독과 구분한다.

A에는 member별 truth MSE가 없다. 작은 ensemble-mean tendency MSE도 finite-M에서 분산 부담이 있다.
noise identity만 공유한다고 joint law/Markov성/보정된 ensemble이 보장되지 않는다. 이 모델은 history와 persistent noise를
포함한 augmented condition에 의존하며 Wiener noise SDE가 아니다. 임의 entropy/무한 spread 보상은 없다.

## 4. Best A 검사 → seal → 새 B

`a.pt`, `.metadata.json`, `.metrics.json`, `.manifest.json`이 출력된다. validation 선택 score는 curriculum total과 별개로
stateCRPS + transitionCRPS + .1 pathEnergy + .1 mean-state-MSE, A는 reconstruction + .05(AEdelta+drift)를 추가한다.
train은 train split, A/B 선택은 expert_validation, C 학습은 calibration, C 선택은 validation이다.

`audit`는 기본 expert_validation 고유 관측쌍으로 **AE delta → drift-only delta → tangent oracle**을 분해한다.
A 튜닝/진행 판단에는 이 split을 사용하고, C 선택용 validation을 미리 소비하지 않는다.
과거 validation 감사 재현이 필요할 때만 `AUDIT_SPLIT=validation` 또는 audit CLI의 `--split validation`을 명시한다.
AE 양 endpoint에 정답을 넣는 것은 geometry 감사이지 inference가 아니다. tangent least-squares는 instantaneous Jb가
담을 수 있는 방향을 검사하며 learned 6h drift skill로 해석하지 않는다. full MoE는 뒤의 validation/member JSON으로 비교한다.

다음 단계 통과 조건: 재구성·AE delta·drift가 zero-tendency/persistence 대비 어떤지, 비정상 진폭/latent collapse,
새 score의 nonzero gradient, static gradient 지배, spread/coverage와 bias를 함께 검토한다.
`QUALITY_MAX`를 설정하면 best A의 AE/drift normalized error가 임계값보다 큰 경우 **seal/checkpoint 저장 전에 실패**한다.
기본값에는 검증된 임계값이 없으므로 저장 성공이 quality-pass라는 뜻은 아니다.

best A reload 후 train 관측으로 affine mean/scale(floor .05), chart centers/radius, encoder reference를 **한 번만** seal한다.
A aux sampler는 raw z 좌표라 seal 전후 physical 출력이 일치한다. B/C는 affine/chart를 재fit하지 않는다.
변한 A와 과거 B/C weights를 섞을 수 없다. 새 A에서 **B를 새로 학습**해야 한다.

## 5. B / C 학습 파라미터와 실제 loss

| 모듈 | A | B | C |
|---|---|---|---|
| surface encoder/decoder/drift + information encoder | 학습 | 완전 동결 | 작은 LR |
| A aux sampler/context/info reconstruction head | 학습 | 동결·최종 forecast 미사용 | 동결·최종 forecast 미사용 |
| B full-state experts/gate/history | 동결 | 새 학습 | 학습 |
| affine/chart/reference encoder | best A 후 train seal | 고정 | 고정 |

B는 기존 common-coordinate projection, responsibilities/expertFM/fusedFM/gate/balance/bounded-diversity를 재사용한다.
동일 member의 모든 expert가 같은 physical q와 tau residual을 보고 **각 tau step에서 field 결합 후 한 번 적분**한다.
A aux sampler weights를 B expert로 복사하지 않는다. A 분포 학습의 이득이 B로 전달될지는 실제 검증 대상이다.
geometry는 한 physical step의 tau solve 안에서만 in-graph 재사용한다.

B의 기존 trajectory .1/mean-delta .02/wind speed .01/direction .005, member tendency .001을5epoch ramp로 사용한다.
`B_MEMBER_WEIGHT=0` 비교가 가능하다. member MSE=mean MSE+ensemble variance 벌점이며 spread 축소를 감시한다.
C는 member weight0, temporal ramp3, 기존 marginal Energy .5+CRPS .5, PI .5, reference anchor1을 사용한다.
**A의 새 fair state/transition/info CRPS curriculum 전체를 C에 옮긴 Loss V2가 아니다.**
기본 LR .001이면 C process .0001, representation .00001. Reference anchor에 information encoder도 포함한다.
B의 frozen A tensor는 저장 전 bitwise 비교하며, decoder 입력 q의 gradient는 차단하지 않는다.
로그의 `weighted_*`를 합하면 실제 `loss`가 된다(부동소수점 반올림 허용).
`weighted_specialization`은 기존 expert/FM/gate objective 전체다. C에는 추가로
`weighted_marginal_energy/crps`, `weighted_pi`, `weighted_anchor`가 기록된다.
계산된 `state_crps`, `transition_crps`, `static_l2`, `decoded_drift` 등의 이름만으로
B/C에서 그 항이 직접 최적화된다고 해석하면 안 된다. 예를 들어 C의 static L2는 진단용이다.

## 6. 평가 / 한 번 예측 / 모든 member 출력

`validation` 명령은 checkpoint best weights를 reload하고 전체20lead를 생성한다. `forecast.npz`는 첫 validation origin의
모든 member를 한 번만 저장한다. `validation.json`은 선택된 여러 origin의 aggregate이며 첫 member 지표와 구별한다.
per-variable state/tendency error, fair state/transition CRPS, joint Energy, mean/member 분산, coverage80,
drift/residual/final q/day norm, generated routing/entropy/candidate cosine/projection ratio/chart distance를 확인한다.
2026-09-21 보수 이후 aggregate RMSE는 `sqrt(mean(case mean_state))`, spread는
`sqrt(mean(case ensemble_variance))`다. 기존 case별 RMS의 산술평균은 `mean_case_rmse`,
`mean_case_spread`, `mean_case_persistence_rmse`에 보존한다. 이 집계 변경은 모델 성능 개선이 아니다.
RMSE는 train state scale로 정규화된 변수·면적 가중 값이며 Pa/K/m/s의 원시 단위를 섞은 norm이 아니다.
풍향은 기존 u/v 기반 convention/calm mask를 재사용하며 출력은 u/v를 유지한다.

```bash
python -m climate_diffusion.information_forecast --checkpoint "$RUN/c.pt" --archive "$ARCHIVE" \
  --information "$INFO" --output "$RUN/validation-extra.json" --split validation \
  --members 8 --tau-steps 16 --max-cases 32 --device cuda
python -m climate_diffusion.trajectory_output --forecast "$RUN/forecast.npz" --archive "$ARCHIVE" \
  --output-dir "$RUN/members-12h-gif" --horizon-hours 120 --interval-hours 12 --extension gif
```

M/tau를 바꾼 추가 평가는 같은 조건의 baseline과 비교한다. 6h 출력은20개 미래 시점, 12h는10개이며 origin은 별도 metadata다.
두 출력은 같은 저장 NPZ를 subset하므로 member가 바뀌지 않으며 120h lead 정규화를 재설정하지 않는다.
t2m K 색지도+u/v 고정 quiver/m/s key를 사용한다. 화살표 위치가 고정된 Eulerian field이며 FPS/배율 변경은 개선 근거가 아니다.
`member-xxx.json`에 물리 변화율·풍속오차·near-zero ratio 유효 수가, `summary.json`에 전체 ensemble spread가 기록된다.
마지막 test는 설정을 고정한 뒤에만 실행하고 test를 보고 계수를 튜닝하지 않는다.

## 7. Pilot, 메모리, 중단/재개

4090의 VRAM/시간은 이 실행에서 **미측정**이다. 먼저 작은 관측 수로 full20 BPTT 자체를 검증한다.

```bash
export RUN=/workspace/experiments/a-information-pilot-001
export MAX_WINDOWS=4 A_EPOCHS=6 CURRICULUM_INTERVAL=1 B_EPOCHS=1 C_EPOCHS=1
export BATCH=2 M=4 TAU=4
# 위 preflight→A→audit→B→C→validation 순서 반복
```

full 실행은 새 RUN에 `MAX_WINDOWS=0`, A60/interval4/B30/C10 등 후보를 설정한다. 작은 pilot은 full20graph이지만
전체 ERA5를 학습한 것은 아니다. 이 trainer에는 sub-block/truncated BPTT 옵션이 없으며 20step을 모두 역전파한다.
singleton 마지막 batch는 이전 batch에 합쳐져 peak batch가 설정값+1일 수 있다.
OOM이면 BATCH를2까지, 보조 실험으로 M2/tau2 또는 입력격자를 줄이고 **새 run과 변경 설정**으로 비교한다.
window 수를 줄이면 총시간은 줄지만 단일 batch peak-memory는 거의 줄지 않는다.
`MAX_WINDOWS=1` 또는 stride 선택 후 window가1개뿐인 split은 명시적으로 실패한다.
기존처럼 임의로 첫2개 window로 대체하지 않는다. `MAX_WINDOWS=0` 또는2이상, 더 작은 stride를 사용한다.
loss/gradient audit는 첫 batch에서 추가 backward graph 조회 비용이 든다. 직접 CLI에서 `--gradient-audit`를 생략할 수 있다.
매 epoch runtime/maxRSS/CUDA peak bytes를 기록한다. CPU MaxRSS는 프로세스 누적 최대이며 GPU 예산으로 환산하지 않는다.

Checkpoint는 best weights/설정/통계/parent hash를 저장하고 optimizer/RNG state는 저장하지 않는다.
**exact resume와 같은-stage warmstart는 지원하지 않는다.** 중단한 stage는 새 출력 폴더에서 처음부터 재실행한다.
완료한 A→B 또는 B→C 전환은 `--init`으로 가능하며 새 optimizer다. 이전/legacy/joint-AB checkpoint를 조용히 재해석하지 않는다.
학습 코드 SHA와 데이터 SHA는 provenance다. 소스 변경 뒤 같은 output을 이어쓰기 하지 않는다.

## 8. 구현 위치 / 아직 입증되지 않은 것

- `physical_information.py`: sidecar schema/단위·정합/terrain slope/train stats.
- `information_process.py`: A aux FM, 조건 인코딩, physical recurrence, loss/curriculum/gradient.
- `train_information_process.py`: 별도 A/B/C optimizer/freeze/seal/checkpoint/best.
- `information_forecast.py`: 새 checkpoint 전용 평가/export. legacy `forecast-climate-flow`에 새 weights를 넣지 않는다.
- `audit_information_process.py`: 관측쌍 AE/drift/tangent 감사.
- [구조 그림](../struct-picture/16-separate-a-information-process.md), [실행 결과](../docs/results/a-information-process-smoke/README.md).

원문 직관의 “L2=유한 정보, 분포=infinite 규칙”을 정리로 사용하지 않았다. 추가 C는 관측 부족 완화 가설이지 Markov성 보장이 아니다.
보존된 GIF는 과거 ERA5 결과이고 새 synthetic 성능의 증거가 아니다. 실제 ERA5 재학습·다중 seed/계절 검증·최종 calibration은
아직 실행하지 않았다. 새 loss가 연결되었다는 사실과 accuracy/reliability 개선은 다르다.
