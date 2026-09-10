# 120시간 State + Dynamics: 처음부터 재학습하기

이 문서는 **구현된** `Flow Matching + Physics-Informed Manifold MoE + Ensemble`의 새 학습 profile입니다.
기존 네트워크/전문가 수/투영 구조를 유지하고 데이터·loss·sampling·평가·출력만 변경했습니다.
120h는 5일이며 **6h × 20 step**입니다. 기존 `TRAINING_README.md`의 H=120은 720h/30일입니다.
코드 업데이트만으로 기존 checkpoint가 새 loss를 학습하지 않습니다.

- [실행 결과와 한계](../docs/results/temporal-120h-smoke/README.md)
- [학습·gradient 그림](../struct-picture/11-temporal-training-implemented.md)
- [시간·member 출력 그림](../struct-picture/12-member-trajectory-output.md)
- [설계 당시 근거](../docs/training-mechanism/README.md): 원인 단정이 아닌 가설입니다.

## 1. 설치 → CPU 검증

기존 작업/weights를 보존하려면 별도 clone과 새 출력 폴더를 사용하세요.

```bash
git clone --branch feature/latent-dynamics-flow --single-branch \
  https://github.com/nayehyeon61-glitch/climate_diffusion.git climate_diffusion_temporal
cd climate_diffusion_temporal
python -m venv --system-site-packages .venv
source .venv/bin/activate
python -m pip install -e '.[test,plots,io]'
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest -q
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python scripts/smoke_temporal_moe.py
python scripts/visualize_temporal_moe.py \
  --checkpoint outputs/temporal-120h-smoke/new-c.pt \
  --archive outputs/temporal-120h-smoke/synthetic-states.npz
```

RunPod 이미지의 기존 CUDA PyTorch를 유지하세요. MP4에는 OS의 `ffmpeg`가 필요하며,
없으면 GIF를 사용할 수 있습니다. 스크립트는 기존 출력 폴더/checkpoint를 덮어쓰지 않습니다.
같은 smoke를 다시 실행할 때는 `--work-dir`와 `--report-dir`를 새 이름으로 지정하세요.
이 smoke의 작은 grid/epoch를 실제 기상 예측 성능으로 해석하지 않습니다.

## 2. 실제 ERA5 archive 준비 → 시간·mask·통계 검사

이미 확보한 ERA5 NetCDF/Zarr를 사용합니다. 아래 경로는 **본인의 실제 파일 위치로 설정**하세요.
새 다운로드/유료 GPU를 자동으로 생성하는 명령은 없습니다.

```bash
export TEMPORAL_FIELDS=/workspace/data/era5-surface.nc
export TEMPORAL_ARCHIVE=/workspace/data/era5-temporal-6h.npz
export TEMPORAL_RUN=/workspace/experiments/temporal-120h-run-001
export TEMPORAL_DEVICE=cuda
mkdir -p "$TEMPORAL_RUN"
prepare-climate-fixed-step-data \
  --fields "$TEMPORAL_FIELDS" --variables msl t2m u10 v10 \
  --step-hours 6 --target-lat-points 18 --target-lon-points 36 \
  --output "$TEMPORAL_ARCHIVE"
python scripts/prepare_temporal_120h.py --archive "$TEMPORAL_ARCHIVE" \
  --output "$TEMPORAL_RUN/preflight.json" --history-steps 6 --history-stride 4
```

최초 비교에서는 기존 grid를 유지합니다. 18×36은 실행 예시이며 wind dynamics를 충분히 해상한다는
주장이 아닙니다. 원래 archive가 있으면 재생성하지 말고 동일 파일을 검사합니다.
`msl` Pa, `t2m` K, `u10/v10` m/s가 모두 full-state 입력/출력입니다.
명시된 단위가 다르면 실패하며, units attrs가 없는 기존 archive는 canonical 변수명 단위를
가정하고 manifest에 기록합니다. 실데이터에서는 source attrs를 먼저 확인하세요.

**DataLoader 계약** (`TemporalWindowDataset`, flatten D=C×Y×X):

| 항목 | batch shape | 의미 |
|---|---|---|
| history | B,L,D | origin까지의 인과적 관측; 미래 없음 |
| origin | B,D | 마지막 관측 x0의 state normalization |
| targets | B,20,D | +6,…,+120h normalized future, 감독 전용 |
| trajectory_raw | B,21,D | x0와 20개 미래 physical state |
| delta_raw / tendency_raw | B,20,D | 두 endpoint 차분 / 실제 시간 간격으로 나눈 값 |
| dt_hours | B,20 | timestamp에서 계산한 actual dt=6h |
| pair_observed_mask | B,20,D | 양 endpoint 관측 여부; 현재는 모두 true 또는 fail-fast |
| origin_time_ns / valid_time_ns | B / B,21 | UTC nanoseconds; 변수·grid·unit은 dataset.schema |

window start=i, span=(L−1)×stride+1, origin index=o=i+span−1입니다.
첫 변화량은 raw[o+1]−raw[o], 이후는 raw[o+j+1]−raw[o+j]입니다.
history stride=4이면 24h 간격 6개 관측을 사용하지만 **미래 trajectory는 연속 6h**입니다.
윈도우를 겹쳐 저장하지 않고 원래 archive에서 잘라 읽습니다.
future/차분은 router/history condition에 넣지 않습니다.

시간 gap·중복·NaT·결측 pooled cell을 건너 차분하거나 zero-filled label로 학습하지 않습니다.
기존 archive의 spatial pooling/observed_mask 계약을 그대로 사용하므로 원본 고해상도 cell의
완전 관측까지 보증하는 것은 아닙니다. five-way split은
train → expert_validation → calibration → validation → test이고 future target overlap을
H−1 이상 purge합니다. 다음 split의 origin/history에 과거 관측이 들어가는 것은 인과적으로 허용됩니다.
normalization과 dynamics 통계는 train의 유일 관측/인접쌍에서만 fit합니다(겹친 window를 중복 집계하지 않음).
검증·calibration·test로 통계를 다시 fit하지 않습니다.

state는 기존 per-cell mean/std를 유지하고 작은 std(≤1e-6)는 1로 대체합니다.
tendency는 channel별 area-weighted train std b_c를 별도로 씁니다.
floor=max(1e-3×area-RMS(state_scale_c)/6h,1e-8)입니다.
면적은 정규화한 cos(latitude)×dlat×dlon이고, 동일 grid의 channel weight 합도 정규화합니다.
단위가 서로 다른 Pa/K/m/s 원시 크기를 하나의 norm으로 비교하지 않습니다.

## 3. A: 새 manifold 초기화·학습 → best seal

아래 계수/epoch/batch는 **시작값**이며 검증으로 확정해야 합니다. A에 기존 checkpoint를 넣지 않습니다.

```bash
train-climate-manifold-moe --archive "$TEMPORAL_ARCHIVE" \
  --output "$TEMPORAL_RUN/a.pt" --stage manifold \
  --history-steps 6 --history-stride 4 --horizon-steps 20 \
  --num-experts 4 --manifold-dim 16 --expert-latent-dim 64 --gate-hidden-dim 160 \
  --manifold-epochs 50 --batch-size 8 --window-stride 4 \
  --learning-rate 0.001 --weight-decay 0.0001 --early-stop-patience 8 \
  --seed 7 --device "$TEMPORAL_DEVICE"
python scripts/prepare_temporal_120h.py --archive "$TEMPORAL_ARCHIVE" \
  --output "$TEMPORAL_RUN/preflight-a-verified.json" --checkpoint "$TEMPORAL_RUN/a.pt"
```

A는 기존 reconstruction/PI/metric/latent-dynamics objective를 유지합니다.
새 로그의 `reconstruction_mse_{msl,t2m,u10,v10}`와 `reconstruction_tendency_mse_*`로 변수별
보존 여부를 확인합니다. 이것은 아직 생성 ODE의 새 loss를 학습하는 단계가 아닙니다.
best validation A를 재로드한 뒤 latent scale/지역 중심/reference encoder를 seal합니다.
과거 A 재사용은 ablation이며 **전면 from scratch라고 부르지 않습니다**.

`batch-size=1`은 A/C metric이 조용히 0이 되는 것을 막기 위해 거부합니다.
마지막 singleton을 앞 batch에 합치므로 최대 batch는 설정값+1입니다.
validation도 최소 두 관측쌍을 요구합니다. 작은 split은 데이터/stride를 조정해야 합니다.

## 4. B: A 고정, 새 experts/gate/history 학습

```bash
train-climate-manifold-moe --archive "$TEMPORAL_ARCHIVE" \
  --output "$TEMPORAL_RUN/b.pt" --stage specialize --init-checkpoint "$TEMPORAL_RUN/a.pt" \
  --expert-epochs 40 --batch-size 2 --window-stride 4 \
  --ensemble-size 4 --sampled-leads 2 --integration-steps 4 \
  --trajectory-edges 2 --validation-trajectory-edges 0 \
  --trajectory-weight 0.1 --delta-weight 0.02 \
  --wind-speed-weight 0.01 --wind-direction-weight 0.005 \
  --temporal-warmup-epochs 5 --trajectory-selection-weight 0.1 --log-gradient-norms \
  --learning-rate 0.001 --weight-decay 0.0001 --early-stop-patience 8 \
  --seed 7 --device "$TEMPORAL_DEVICE"
```

기존 FM/지역 responsibility/균형/diversity/투영 objective에 새 loss가 더해집니다.
기존 diversity는 유한 margin 부족분 penalty이며 무한한 spread에 보상을 주는 항을 추가하지 않습니다.
A weights를 동결해도 **ODE q → decoder state의 미분은 유지**합니다. 새 loss는
ODE/투영을 거쳐 experts, gate, history encoder에 gradient를 보냅니다.

`--sampled-leads 2`는 FM pair용입니다. 새 trajectory score의 선택은 별도로
`--trajectory-edges`가 제어합니다. `2`는 전체 120h window 안에서 연속 2구간(3 endpoint)을
뽑아 학습합니다. 이 경우 전체 120h joint law를 학습했다고 주장하지 않습니다.
`--trajectory-edges 0`이면 **origin 포함 21 endpoint/20구간 전체의 joint loss**를 학습합니다.
full-window는 실제 backward smoke를 별도로 실행했고, 장기 수렴은 아직 검증하지 않았습니다.
validation은 기본 `0`으로 전체 window를 사용하며 훈련용 sub-block과 기록을 구분합니다.

## 5. C: B 재로드 → 작은 LR로 joint calibration

```bash
train-climate-manifold-moe --archive "$TEMPORAL_ARCHIVE" \
  --output "$TEMPORAL_RUN/c.pt" --stage joint --init-checkpoint "$TEMPORAL_RUN/b.pt" \
  --joint-epochs 10 --batch-size 2 --window-stride 4 \
  --ensemble-size 4 --sampled-leads 2 --integration-steps 4 \
  --trajectory-edges 2 --validation-trajectory-edges 0 \
  --trajectory-weight 0.1 --delta-weight 0.02 \
  --wind-speed-weight 0.01 --wind-direction-weight 0.005 \
  --temporal-warmup-epochs 5 --trajectory-selection-weight 0.1 --log-gradient-norms \
  --learning-rate 0.001 --joint-lr-factor 0.1 --encoder-lr-factor 0.1 \
  --weight-decay 0.0001 --early-stop-patience 5 --seed 7 --device "$TEMPORAL_DEVICE"
```

C는 기존 calibration split에서 PI+anchor와 marginal Energy/CRPS를 유지하면서 새 loss를 더합니다.
manifold LR은 base LR×joint factor×encoder factor입니다. A 좌표 anchor/지역 경계는 유지합니다.
새 optimizer를 단계마다 생성하며 seed와 독립 RNG stream 정책은 metadata에 저장합니다.
현재 checkpoint는 best weights/통계/설정/상위 checkpoint SHA256을 저장하며 **optimizer 중단점의
bitwise resume 파일은 아닙니다**. 같은 파일에 학습을 이어 덮어쓰지 말고 별도 실험으로 관리하세요.
A/B/C의 `.manifest.json`, `.metadata.json`, `.metrics.json`을 함께 보존합니다.

## 6. 구현한 loss와 두 시간축

ODE 시간 τ∈[0,1]의 dq/dτ, 물리 lead t=j×6h의 dx/dt, u/v 풍속 m/s는 다른 값입니다.
전문가가 latent velocity를 state decoder에 직접 넣는 shortcut은 없습니다.
투영된 vector fields를 **각 step에서 결합한 뒤** member별 최종 intrinsic ODE를 적분하고,
마지막 q를 state decoder에 넣어 얻은 physical endpoint들의 차분을 감독합니다.

정규화 state y=(x−μ)/σ, 선택 endpoint 개수 P, train tendency scale b를 쓰면

```math
d_j^{(m)}=\frac{\sigma\odot(y_{j+1}^{(m)}-y_j^{(m)})}{\Delta t_j\,b},\qquad
d_j^*=\frac{x_{j+1}^*-x_j^*}{\Delta t_j\,b}.
```

W는 위 면적·channel 가중치이며 sum(W)=1입니다.

```math
L_{\rm delta}=\frac1{P-1}\sum_j\|M^{-1}\sum_m d_j^{(m)}-d_j^*\|_W^2,
\quad F^{(m)}=\left[\frac{W^{1/2}y^{(m)}}{\sqrt P},\frac{W^{1/2}d^{(m)}}{\sqrt{P-1}}\right].
```

`loss_delta`는 명시적 **ensemble-mean 변화량 MSE**이며 실제 dt로 tendency scaling한 이름입니다.
고정 dt에서 raw delta MSE와 tendency MSE는 상수배이므로 둘을 중복 가중하지 않습니다.

```math
L_{\rm trajectory}=\frac1M\sum_m\|F^{(m)}-F^*\|_2
-\frac1{2M(M-1)}\sum_{m\ne n}\|F^{(m)}-F^{(n)}\|_2,
\qquad L=L_{\rm existing}+a_e(\lambda_{\rm traj}L_{\rm trajectory}
+\lambda_\Delta L_{\rm delta}+\lambda_s L_{\rm wind-speed}+\lambda_d L_{\rm wind-direction}).
```

Joint endpoint/increment Energy는 독립 member draw에 대한 fair estimator(M≥2)입니다.
기존 marginal Energy/CRPS 평가 값은 기존 empirical estimator를 유지해 새 joint score와 구분합니다.
`a_e=min(1,epoch/warmup)`이고 loss별 raw 값과 ramp를 따로 기록합니다.
best 선택은 ramp에 따라 바뀌지 않는 `validation Energy+CRPS+fixed_weight×trajectory`입니다.

모든 member를 관측 realization 하나에 MSE로 강제하지 않습니다. 그래도 유한 M의 **mean MSE에도
분산 penalty**가 있으므로 delta weight를 작게 시작하고 0 ablation과 spread/coverage를 비교합니다.
방향 보조항도 mean bounded feature MSE라 같은 주의가 필요합니다.
별도 magnitude reward/acceleration/direction cosine ratio/AR rollout loss를 추가하지 않았습니다.
전체 window joint score가 여기서의 trajectory supervision이며 동일한 loss를 `rollout` 이름으로 중복하지 않습니다.

풍속은 sqrt(u²+v²), 방향은 수학적 **toward**, cos=u/sqrt(u²+v²+ε²), sin=v/… 입니다.
기상학적 from 방향과 180° 다르며 각도 직접 뺄셈을 하지 않아 359°/1° wrap에 연속입니다.
calm threshold=max(0.05×train wind-speed RMS,1e-3 m/s), ε=threshold×0.01입니다.
관측 풍속으로만 calm mask를 만들므로 예측 풍속을 0으로 만들어 방향 loss를 숨길 수 없습니다.
풍속은 area-weighted fair CRPS, 방향은 bounded cos/sin ensemble mean 보조항입니다.
기본 msl/t2m/u/v 각각 weight=1이고 `--variable-weights` JSON으로 **새 temporal objective**의
channel weight를 조절할 수 있습니다. 예: u/v를 각0.5로 하면 wind group 합1입니다.
기존 A/PI/FM의 수학적 가중치는 그대로입니다. u/v+speed+direction 중복을 고려해 새 wind 계수는 작게 둡니다.
가중치/통계는 A에서 정해 B/C에 고정하며 중간에 바꾸려면 별도 A 실험을 생성합니다.

Noise는 z[b,m,r], 같은 member의 모든 physical lead에 같은 source를 재사용합니다.
lead별 q는 적분 중 달라질 수 있지만 같은 member/lead/step의 모든 expert는 동일 q를 봅니다.
member 재추첨/정렬은 하지 않습니다. FM source·tau·lead 선택·block 선택·ensemble·shuffle의 RNG를 분리합니다.
공유 noise 자체가 올바른 joint trajectory 분포나 누적오차 방지를 보장하지 않습니다.

## 7. Validation → 설정 고정 → test

```bash
evaluate-climate-flow --checkpoint "$TEMPORAL_RUN/c.pt" --archive "$TEMPORAL_ARCHIVE" \
  --split validation --output "$TEMPORAL_RUN/validation.json" \
  --ensemble-size 8 --integration-steps 16 --max-cases 32 --seed 83 --device "$TEMPORAL_DEVICE"
python -m climate_diffusion.manifold_diagnostics \
  --checkpoint "$TEMPORAL_RUN/c.pt" --archive "$TEMPORAL_ARCHIVE" \
  --split validation --output "$TEMPORAL_RUN/validation-routing.json" --device "$TEMPORAL_DEVICE"
# 위 validation으로 설정을 고정한 다음에만 최종 test를 실행합니다.
evaluate-climate-flow --checkpoint "$TEMPORAL_RUN/c.pt" --archive "$TEMPORAL_ARCHIVE" \
  --split test --output "$TEMPORAL_RUN/test.json" \
  --ensemble-size 8 --integration-steps 16 --max-cases 32 --seed 83 --device "$TEMPORAL_DEVICE"
```

원래 baseline → corrected paired noise(새 weights=0) → trajectory+delta → wind 보조항 순으로 비교하세요.
같은 archive/split/A 초기화/seed/member 수/ODE steps를 고정합니다. 본 smoke의 baseline은
이미 noise가 수정된 버전이며 이전 commit 전체 재현이 아닙니다.
state RMSE, marginal/joint Energy·CRPS, coverage, 변수별 tendency RMSE·진폭 비,
member/mean spread·lag, A 변수별 reconstruction, 지역별 expert skill을 함께 봅니다.
전문화 진단은 이제 `--split validation`으로 test를 열지 않고 사용할 수 있습니다.
usage 균등이나 높은 candidate cosine만으로 전문화 성공/실패를 단정하지 않습니다.
ACC/PSD/계절별 sampling 및 별도 curriculum은 이번 CLI에 구현하지 않았으므로 지원 flag처럼 사용하지 마세요.

## 8. 같은 forecast에서 모든 member 출력

validation split의 첫 origin을 정확히 선택합니다. 원래 시간축을 늦추거나 truth를 shift하지 않습니다.

```bash
export TEMPORAL_ORIGIN=$(python - <<'PY'
import os, torch
from climate_diffusion.moe_data import load_moe_archive
ck=torch.load(os.environ['TEMPORAL_RUN']+'/c.pt',map_location='cpu',weights_only=False)
_,times,_=load_moe_archive(os.environ['TEMPORAL_ARCHIVE'])
i=ck['training']['split']['validation'][0]+ck['training']['history_span_steps']-1
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

default는 **전체 member**, `--members 0 2`이면 그 ID만 렌더링합니다. 한 번 저장한 forecast를
두 출력에 재사용하므로 member identity가 유지됩니다. mean 영상은 기존 `diagnose-climate-time`
명령을 명시적으로 실행할 때만 보조 출력으로 사용하세요.
6h 출력은 +6,…,+120의20frame,12h 출력은+12,…,+120의10frame입니다.
NPZ에는 origin_state를 별도 보관하므로 합치면 각각21/11개 state입니다. 원본 6h forecast는 보존합니다.
old H=120 checkpoint에서도 `--forecast-steps 20`으로120h prefix를 추출할 수 있지만 **조건 s=j/120**을
유지합니다. H20 모델의 s=j/20으로 기존 weights를 재해석하지 않습니다. prefix 일관성 테스트가 있습니다.

각 영상은 왼쪽 generated member m, 오른쪽 ERA5, 동일 K colorbar와 m/s quiver scale,
해안선, origin/lead/UTC valid-time label을 사용합니다. 화살표는 고정 격자의 **Eulerian 풍속**입니다.
입자/pathline 이동이나 τ velocity가 아닙니다. FPS/화살표 배율을 높여 예측 개선을 주장하지 않습니다.
`member-NNN.json`은 해당 member 지표, `summary.json`은 ensemble aggregate이고 섞지 않습니다.
near-zero true tendency ratio는 null과 valid count를 기록합니다. lag는 진폭 시계열의 탐색 진단이며
truth 정렬을 바꾸지 않습니다. 원시 source가 없는 GIF 한 장에서 동역학 수치를 복원하지 않습니다.

## 9. 계산 예산과 실행 범위

4090 최초 점검은 B/C batch2(실제 최대3), M=2~4, integration2~4, 2-edge sub-block으로
1epoch를 별도 경로에 실행해 peak memory/step time/finite gradient를 측정하세요.
숫자는 시작 예산이지 측정된 최적값/메모리 보장이 아닙니다. 이후 M/steps/window를 하나씩 늘립니다.
full20은 sub-block2의3endpoint보다 최대 약20/3배 많은 lead integration graph를 유지합니다.
midpoint ODE는 lead마다2×integration_steps회의 field 평가를 하며, Jacobian/tangent solve와
역전파 graph가 메모리의 주요 비용입니다. 향후 checkpointing이 필요할 수 있습니다.
현재 archive/normalized array는 host RAM에 적재되고 A physics fit/seal도 train span 전체를 사용하므로
전지구 고해상도/수년 자료를 곧바로 넣기 전 RAM/GPU 사용을 측정해야 합니다.

이번 완료 범위는 코드·테스트·CPU synthetic A/B/C·full-window backward·member 영상입니다.
**실제 ERA5 archive와 연결된 RunPod/GPU가 없어 실제 ERA5 전면 재학습은 실행하지 않았습니다.**
사용자가 확보한 데이터와 실행 환경에서 위 순서로 새 A부터 실행해야 합니다.

수정할 코드 위치:

| 파일/함수 | 수정 영역 |
|---|---|
| `temporal_supervision.py`: dataset/statistics/select_block/TemporalObjective | 시간·차분·train 통계·새 loss |
| `manifold_moe.py`: sample_trajectory/forecast | train/inference 공통 noise·ODE 경로 |
| `train_manifold_moe.py`: _pairs/_sample/_epoch/train_manifold_moe | RNG·loss ramp·단계·checkpoint |
| `evaluation.py`, `manifold_diagnostics.py` | held-out split 선택·joint score·지역 전문화 |
| `time_alignment.py`, `trajectory_output.py` | 정확한 prefix/valid-time·member 출력·변수별 변화율 |
| `scripts/prepare_temporal_120h.py`, `smoke_temporal_moe.py` | 실데이터 preflight·처음부터 재현 |
