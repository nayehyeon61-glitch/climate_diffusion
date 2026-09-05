# 검증 보고서: 0.25° spatial Flow Matching 데이터 경로 및 학습/평가

- 대상 브랜치: `feature/spatial-025-flow-autoencoder` (HEAD `6e1764e`)
- 관련 PR: [#1](https://github.com/nayehyeon61-glitch/climate_diffusion/pull/1)
- 검증 범위: 아카이브 → 윈도우 데이터셋 → Flow Matching loss 입력까지의 데이터 경로,
  그리고 학습 중 model selection과 held-out evaluation의 타당성
- 검증 방법: 코드 정독 + 합성 아카이브로 end-to-end 실행 + 전체 테스트 스위트 실행

## 1. 결론

**데이터 경로(아카이브 → Flow Matching 입력)는 정상이다.** 시간 정렬, 정규화, 결측 처리,
train/validation/test 분리 모두 재현 검증에서 통과했다.

**학습 evaluation은 아직 신뢰할 수 없다.** 두 가지 이유다.

1. 체크포인트 선택 지표(validation loss)가 확률적이라 같은 모델을 6번 평가해도
   총 loss가 5.5%, flow 항은 75% 흔들린다. `best.pt`가 "가장 좋은 epoch"이라는 보장이 없다.
2. 학습 중에는 예보 성능을 전혀 측정하지 않는다. 실제 예보 지표(RMSE/CRPS/persistence 대비)는
   학습이 끝난 뒤 test에서 딱 한 번 계산되고, 그 계산에도 위도 가중치와 계절 climatology가 빠져 있다.

추가로 **RunPod 기본 설정(`mixed_precision=True`)에서는 장시간 학습이 죽을 수 있는 경로가 있다.**
아래 P0-1이 가장 먼저 고쳐야 할 항목이다.

### 1.1 수정 상태

이 브랜치에서 아래 네 건은 수정 완료했고, 각 항목에 재측정 결과를 붙였다.

| 항목 | 상태 | 재측정 |
| --- | --- | --- |
| P0-1 AMP overflow가 run을 죽임 | ✅ 수정 | overflow step을 skip·집계, run 유지. fp32 경로는 hard fail 유지 |
| P0-2 patch 경도 가짜 seam | ✅ 수정 | 전 폭 입력만 wrap, 좁은 patch는 replicate(NEAR) |
| P0-3 validation loss 재샘플링 | ✅ 수정 | 6회 반복 편차 5.5% → **0.0** (bit-identical) |
| P1-4 flow 항 detach 없음 | ✅ 수정 | `d(flow loss)/d(target input)` = **0.0** |
| P1-7 tile마다 독립 noise | ✅ 수정 | seam에서 눌리던 spread가 전역 샘플링과 일치 (0.99배) |
| P1-5 학습 중 예보 성능 미측정 | ✅ 수정 | 샘플링 기반 skill로 `best.pt` 선택. loss 기준과 실제로 다른 epoch을 고름 |
| P1-6 위도 가중·계절 baseline 없음 | ✅ 수정 | cos(lat) 가중 + 월별 climatology(3.4배 어려운 baseline) + ACC |
| P2-8 위치 정보 없음 | ✅ 수정 | sin(lat)/cos(lon)/sin(lon) 정적 채널, patch·tile 크롭 추종 |
| P2-9 시작 I/O 낭비 | ✅ 수정 | window 필터 mask 읽기 435,456 cells → **0**, 로드 시 전량 Inf 스캔 제거 |

**10건 전부 수정 완료.** 테스트는 74 passed (최초 33 passed).

## 2. 통과한 항목 (재현 검증됨)

| 항목 | 결과 |
| --- | --- |
| history/target 시간 정렬 | 통과. 값이 `[0,1,2]`인 월을 history로, `3`을 target으로 정확히 집는다 (`start + history + lead - 1`) |
| raw-month split 누수 | 통과. train `[0,12)` / validation `[13,18)` / test `[19,24)`가 교집합 0, purge 월이 실제로 배제됨 |
| 윈도우가 split 경계를 넘지 않음 | 통과. `starts()`가 `bounds[1] - span`까지만 생성 |
| 정규화 통계 | 통과. `_train_only_statistics`가 train raw 월만 사용, observation mask 반영 |
| 결측 처리 | 통과. NaN은 train mean으로 대치 후 정규화, ±Inf는 fail-fast |
| 학습/추론 전처리 일치 | 통과. `LatentFlowForecaster._normalise`가 학습과 동일하게 impute → normalize |
| Flow Matching 수식 | 통과. `x_t = (1-t)x_0 + t x_1`, target velocity `x_1 - x_0`, `t ~ U(0,1)` — rectified flow와 일치 |
| 샘플링 적분 방향 | 통과. `randn`에서 시작해 `t=0→1`로 midpoint 적분 |
| CRPS 추정식 | 통과. `mean|x-y| - (1/2m²)Σ|x_i-x_j|`의 정렬 기반 항등식이 정확 |
| 계약 지문(fingerprint) 검증 | 통과. 평가 아카이브가 학습 아카이브와 다르면 거부 |

## 3. 발견된 문제

### P0-1. AMP gradient overflow가 학습 전체를 중단시킨다

`src/climate_diffusion/train.py:188-192`

```python
if scaler is not None and scaler.is_enabled():
    scaler.unscale_(optimizer)
for name, parameter in model.named_parameters():
    if parameter.grad is not None:
        require_finite_tensor(parameter.grad, f"gradient {name}")   # FloatingPointError
```

`GradScaler`는 fp16 gradient가 inf/NaN이 되는 것을 **정상 이벤트로 간주**하고
`scaler.step()`에서 그 step만 건너뛰고 scale을 낮추도록 설계돼 있다. 그런데 이 코드는
`unscale_` 직후에 finite 검사를 하고 `require_finite_tensor`가 `FloatingPointError`를 던지므로,
학습 초반에 흔한 fp16 overflow 한 번으로 run 전체가 죽는다.

`runpod_train.py`는 `mixed_precision=True`가 기본값이므로 500GB 장시간 run에 그대로 노출된다.

**✅ 수정 완료**: AMP가 켜져 있으면 overflow를 감지해 clip/step을 건너뛰고 `scaler`가 scale을
낮추도록 맡긴다. 횟수는 `amp_overflow_steps`로 epoch metric에 남아 사라지지 않는다.
loss scaling이 없는 fp32 경로에서는 non-finite gradient가 여전히 실제 결함이므로 hard fail을 유지한다.

회귀 테스트 2건: `test_amp_gradient_overflow_skips_the_step_instead_of_ending_the_run`
(overflow 1회 집계, 파라미터 불변, scale 감소 확인),
`test_non_finite_gradient_without_loss_scaling_still_stops_the_run`.

### P0-2. patch 학습 / tiled 추론에서 경도 padding이 가짜 seam을 만든다

`spatial.py:41`, `data.py:610-612`, `runpod_train.py:232`

`PeriodicConv2d`는 경도 방향을 항상 `mode="circular"`로 패딩한다. 이는 **입력이 전 지구
경도를 덮을 때만** 옳다. 그런데 운영 기본값은 1440 중 256 경도 window를 잘라 쓴다
(`--patch-width 256`, `--tile-overlap 64`).

실측 (32칸 격자에서 8칸 patch를 잘라 global conv 결과와 비교):

```
patch interior matches global conv : True
patch left-edge  error vs global   : 0.676144
patch right-edge error vs global   : 0.833084
```

즉 patch 내부는 맞지만 양 끝은 틀린다. circular padding이 patch 왼쪽 끝과 오른쪽 끝을
이웃으로 연결하는데, 실제로는 약 296° 떨어진 지점이다. 이 오염이 **모든 conv layer, 모든
학습 샘플, 모든 추론 tile**에 들어간다.

테스트는 이 경로를 전혀 덮지 않는다. `test_periodic_longitude_convolution_is_roll_equivariant`는
전역 격자로만 검증하고, `test_clean_spatial_archive_trains_reloads_and_evaluates`는
`patch_width=8`(= 전체 격자 폭)로 실행한다.

**✅ 수정 완료**: `PeriodicConv2d`에 `wrap_longitude` 플래그를 두고,
`MonthlyLatentFlow`가 loss/sample 진입 시 입력 폭이 `config.grid_width`와 같을 때만 wrap을 켠다.
좁은 patch와 tile은 경도도 **replicate(NEAR)** 로 패딩하므로 가짜 seam이 사라진다.
downsample 이후 폭이 절반이 되어도 플래그는 최상위에서 한 번 결정되므로 깊은 층까지 일관된다.

재측정: 전 폭(32/32) 입력 → wrap `True`, 좁은 patch(8/32) → wrap `False`.
회귀 테스트: `test_sub_global_patch_pads_by_replication_instead_of_wrapping`,
`test_model_wraps_longitude_only_for_globe_spanning_input`.

**이 수정이 덮지 않는 것**: `spatial_operator` backend의 `SpectralOperator2d`는 patch에
`rfft2`를 걸므로 여전히 patch를 주기 신호로 간주한다. conv padding과 같은 종류의 가정이지만
FNO 계열에서는 patch를 그 자체로 하나의 도메인으로 보는 관행이라 별도 설계 판단이 필요하다.
`spatial_conv` backend는 이 경로를 타지 않는다.

남은 선택지: 경도를 항상 전체 폭으로 유지하고 위도만 자르는 patch 전략으로 가면
seam이 원천 제거되고 P2-8(위치 정보 부재)도 함께 줄며 위 spectral 문제도 사라지지만,
patch 메모리가 커진다.

### P0-3. 체크포인트 선택 지표가 확률적이다

같은 모델·같은 validation loader로 `_epoch`을 6번 반복한 결과:

```
total loss        : 11.515, 11.714, 11.732, 11.168, 11.709, 11.806   (편차 5.5%)
flow_matching_mse :  1.196,  1.395,  1.413,  0.849,  1.390,  1.487   (편차 75%)
reconstruction_mse: 10.319 (6회 모두 동일 — 결정론적)
```

원인은 `model.loss`가 평가 시에도 `torch.randn_like(target_latent)`와 `torch.rand(batch)`를
매번 새로 뽑기 때문이다(`model.py:239-240`). `train.py:477`과 `runpod_train.py:372`의
`validation_metrics["loss"] < best_validation` 비교는 이 노이즈 위에서 이뤄진다.

부수적으로, 결정론적인 reconstruction 항이 확률적인 flow 항보다 훨씬 크므로
**모델 선택이 사실상 autoencoder 재구성 성능만 보고 이뤄진다.** flow matching 품질은
선택에 거의 반영되지 않는다.

**✅ 수정 완료**: `MonthlyLatentFlow.loss`에 `generator` 인자를 추가하고, `_epoch`이 평가일 때만
batch마다 `eval_seed * 1_000_003 + batch_index`로 시드한 generator를 넘긴다. 학습 경로는
독립 추출을 그대로 쓴다. 재샘플링 구간 자체도 줄였다: generator가 주어지면 `t`를
독립 추출 대신 `[0,1)` 위에 **stratified**(`(i + u) / batch_size`)로 배치해 같은 batch 크기에서
flow loss 추정 분산이 줄어든다.

재측정 (동일 모델·동일 loader 6회):

```
total loss x6     : 11.843657 (6회 전부 동일)
flow_matching_mse : 1.535734  (6회 전부 동일)
편차              : 0.0
```

회귀 테스트: `test_seeded_loss_is_reproducible_and_unseeded_is_not`.

남은 문제: 이 수정은 **비교를 재현 가능하게** 만들 뿐, "선택이 사실상 AE 재구성만 본다"는
구조는 그대로다. 그건 P1-5(샘플링 기반 skill 지표)로 풀어야 한다.

### P1-4. flow loss가 encoder로 역전파된다 (detach 없음)

`model.py:228, 243`

```python
target_latent = self.autoencoder.encode(target)
...
target_velocity = target_latent - source_latent   # detach 없음
flow_loss = F.mse_loss(predicted_velocity, target_velocity)
```

실측: flow 항만 남기고 backward하면 encoder 파라미터 **12/12개**가 gradient를 받고
decoder는 0/12개가 받는다.

즉 encoder는 "예측하기 쉬운 latent"를 만드는 방향으로도 학습된다. 여기에
`latent_regularization = target_latent.square().mean()`이 latent를 0으로 미는 압력을
더하므로, 재구성 항이 버텨주지 못하면 latent collapse로 갈 수 있다. 표준 latent flow matching은
autoencoder를 먼저 학습해 freeze하거나, 최소한 flow 항에서 `target_latent.detach()`를 쓴다.

**✅ 수정 완료**: flow branch 전체가 `flow_target_latent = target_latent.detach()`를 쓴다.
`target_velocity`뿐 아니라 `interpolated`의 입력에도 detach된 latent를 넣는 것이 중요하다 —
velocity만 detach하면 `x_t` 경로로 gradient가 그대로 새어 encoder가 여전히 "정답이 들어간
입력"을 만들 수 있다.

재구성 항과 latent 정규화 항은 detach하지 **않는다**(AE는 그쪽으로 학습돼야 한다).
history를 통한 conditioning 경로도 살아 있으므로 encoder는 계속 학습된다 —
없어진 것은 "자기가 예측할 target을 쉽게 만드는" 인센티브뿐이다.

재측정: `d(flow loss)/d(target input)` = **0.0** (수정 전에는 encoder 12/12개가 gradient 수신).
회귀 테스트: `test_flow_term_does_not_backpropagate_through_the_target_encoder`.

### P1-5. 학습 중에 예보 성능을 한 번도 측정하지 않는다

validation에서 계산하는 것은 `reconstruction_mse + flow_matching_mse + latent_l2`뿐이다.
velocity MSE가 낮다고 샘플 품질이 좋다는 보장은 없다.

실제 예보 지표(ensemble 샘플링 후 RMSE/MAE/bias/CRPS, persistence·climatology 대비)는
`evaluation.py`에만 있고, **학습이 끝난 뒤 test split에서 한 번** 계산된다. 그래서
- 조기 종료·하이퍼파라미터 선택을 예보 성능으로 할 수 없고,
- 학습 곡선만 보고는 모델이 persistence보다 나은지조차 알 수 없다.

실제로 합성 데이터 1-epoch 체크포인트를 평가하면 이렇게 나온다:

```json
{ "rmse": 5.016, "crps": 4.945, "persistence_rmse": 0.290, "climatology_rmse": 4.927 }
```

모델이 persistence보다 17배 나쁘지만 평가는 그냥 통과한다. skill score도, 회귀 게이트도 없다.

**✅ 수정 완료**: `train.forecast_skill()`이 validation 윈도우 앞쪽 N개에 대해 추론과 동일한
방식으로 작은 ensemble을 샘플링하고, ensemble 평균을 **같은 셀에서** persistence와 비교한다.
위도 가중은 그 윈도우가 실제로 내주는 center crop 기준으로 계산한다(`patch_latitude_weights`).
`skill_every_epochs`마다 돌며 epoch metric·체크포인트에 기록되고 선택을 주도한다.
측정한 epoch만 후보가 되며, `skill_every_epochs=0`이면 기존 validation loss 선택으로 돌아간다.

**두 기준이 실제로 다른 epoch을 고른다.** 합성 계절 아카이브에서 4 epoch 학습:

```
epoch=1 validation=2.375828  forecast_rmse=0.727721  skill=+0.0360
epoch=2 validation=2.365888  forecast_rmse=0.727454  skill=+0.0363
epoch=3 validation=2.355011  forecast_rmse=0.726934  skill=+0.0370   ← 예보 최적
epoch=4 validation=2.349326  forecast_rmse=0.728977  skill=+0.0343   ← loss 최적
```

validation loss는 매 epoch 단조 감소해 epoch 4를 고르지만, 실제 예보 성능은 epoch 3이 최적이다.

기본값은 의도적으로 저렴하다 — 4 windows / 2 members / 8 steps, RunPod 프로파일은 2 windows —
그리고 모든 값이 양쪽 CLI에 노출된다.

회귀 테스트 6건: `tests/test_forecast_skill.py`.

### P1-6. 평가에 위도 가중치와 계절 climatology가 없다

`evaluation.py:34-43, 140-151`

- `_error_metrics`는 격자점을 균등 가중한다. 0.25° 격자에서 위도 89.875°의 격자 셀 면적은
  적도의 약 1/460인데 RMSE에는 똑같은 무게로 들어간다. 극지방이 전역 점수를 지배한다.
  WeatherBench 계열 표준은 `cos(lat)` 가중 RMSE/ACC다.
- `climatology_rmse`는 정규화 공간의 0, 즉 **전 기간 평균** 하나다. 월별 기후값
  (month-of-year climatology)이 아니다. 월 단위 기후 예측에서 계절 주기는 가장 큰 신호이므로,
  전체 평균 대비 승리는 거의 자동으로 얻어진다. 계절 주기를 못 잡는 baseline과 비교하면
  모델 성능이 과대평가된다.
- ACC(anomaly correlation), spread-skill ratio, rank histogram이 없다.
- test 윈도우는 아카이브 끝의 연속 구간 하나라서 계절이 편향되고 표본이 강하게 자기상관인데,
  신뢰구간이나 부트스트랩이 없다.

`TRAINING_MANUAL.md:158`에 위도 가중 RMSE/ACC가 follow-up으로 이미 적혀 있으니 인지는 되어 있었다.

**✅ 수정 완료**: RMSE/MAE/bias/CRPS 모두 `cos(lat)` 가중을 받는다(vector 아카이브는 격자가
없으므로 제외). 극 행은 정확히 0이 되므로 `1e-6`으로 clip해 표현 가능하게만 남긴다.
학습 월만 사용하는 월별(month-of-year) climatology를 추가했고, 기존 전 기간 평균 baseline도
비교용으로 남겼다. ACC(그 climatology 기준), `spread_skill_ratio`,
`skill_vs_persistence`, `skill_vs_seasonal_climatology`를 함께 낸다. 리포트 형식은
`climate_diffusion.evaluation.v2`.

**baseline 교체 효과가 크다.** 합성 계절 아카이브에서:

```
climatology_rmse           1.4297   ← 전 기간 평균 (기존)
seasonal_climatology_rmse  0.4218   ← 월별 climatology (신규, 3.4배 어려움)
모델 rmse                  1.3244
```

기존 baseline 기준으로는 모델이 "climatology를 이겼다"(1.32 < 1.43)고 나오지만,
제대로 된 계절 baseline 대비로는 2.1배 나쁘다. 과대평가가 실제로 일어나고 있었다.

아직 없는 것: rank histogram, 그리고 test 윈도우가 아카이브 끝의 연속 구간 하나라
계절 편향·자기상관이 있는데 신뢰구간/부트스트랩이 없다는 점은 그대로다.

회귀 테스트 9건: `tests/test_evaluation_metrics.py`. 극지 오차가 적도 오차의 1/1000 미만으로
계산되는지, 가중 없는 경로가 기존 평균과 정확히 일치하는지 포함.

### P1-7. tiled 샘플링은 tile마다 독립 noise를 뽑는다

`model.py:349-355`

```python
def predict(patch):
    return self.sample(patch, ..., generator=generator)   # tile마다 새 randn
```

`sample()`이 tile마다 latent noise를 새로 뽑으므로, 이어붙인 전역 장은 **하나의 결합 분포에서
뽑은 샘플이 아니라 독립 샘플들의 모자이크**다. overlap 구간은 독립 draw의 가중 평균이라
그 부분만 분산이 눌린다. ensemble spread와 CRPS가 위치에 따라 다르게 왜곡된다.

`evaluation.py`는 체크포인트의 `patch_size`를 그대로 tile 크기로 쓰므로(`inference.py:149-153`),
0.25° 운영 설정에서는 **평가가 항상 이 경로를 탄다**.

**✅ 수정 완료**: `sample()`에 `noise` 인자를 추가하고, `sample_tiled`가 전역 latent 해상도로
noise를 **한 번만** 뽑은 뒤 각 tile이 자기 위치의 slice를 쓰게 했다. 이를 위해 `tiled_apply`가
`predict(patch, lat_start, lon_start)`로 tile 원점을 넘긴다. 겹치는 tile은 겹친 영역에서
같은 noise를 적분하므로 blending이 독립 draw를 평균내지 않는다.

latent 좌표 변환은 downsample factor `2**levels`로 나눈 값을 쓴다. 경도는 modulo로 감고,
위도는 감을 수 없으므로 `min(lat_start // factor, latent_height - rows)`로 slice를 안쪽에 붙인다.
운영 형상(721×1440, patch 256, overlap 64, levels 3)에서 마지막 위도 tile 시작점 465는
8의 배수가 아니라 latent 격자에 정확히 맞지 않는다. noise는 i.i.d.라 1픽셀 어긋남 자체는
무해하고, 중요한 성질(겹친 영역이 같은 noise를 본다)은 latent 해상도에서 그대로 성립한다.

재측정 (동일 모델, 48 멤버, 열별 spread):

| | 평균 spread | 최소 열 | max/min |
| --- | --- | --- | --- |
| 수정 전 (tile마다 독립) | 0.11927 | 0.08477 | 1.730 |
| **수정 후 (전역 공유)** | **0.12751** | **0.11125** | **1.312** |
| 전역 샘플링 기준값 | 0.12904 | 0.11218 | 1.611 |

수정 전에는 seam 부근 열의 spread가 기준값의 76%까지 눌렸고, 수정 후에는 전역 샘플링과
사실상 같아진다(99%).

회귀 테스트: `test_overlapping_tiles_share_one_global_noise_field`,
`test_tiled_sampling_is_reproducible_and_seed_dependent`,
`test_tiled_apply_reports_each_tile_origin`,
`test_sample_rejects_noise_that_does_not_match_the_latent_grid`.

### P2-8. patch에 위치 정보가 없다

모델 입력에 위도/경도 채널이 없고 정규화도 채널별 전역 상수다. 무작위로 잘린 256×256 patch만
보고는 그것이 적도인지 극지인지 알 수 없는데, 물리는 위도에 크게 의존한다.
`data.py:605`의 `random_crop`은 위도 시작점을 무작위로 뽑으므로 이 모호성이 학습 전체에 퍼진다.

**✅ 수정 완료**: `sin(lat)`, `cos(lon)`, `sin(lon)` 정적 채널을 encoder 입력에 concat한다.
경도를 sin/cos 쌍으로 넣으므로 날짜변경선에 불연속이 없다.

핵심은 **좌표가 크롭을 따라가야 한다**는 점이다. patch가 무작위로 잘리므로 좌표를 모델이
스스로 알 수 없다. 그래서 좌표는 명시적 인자로 흐른다:
- 학습: `MonthlyWindowDataset`이 history/target과 **동일한 크롭**을 좌표에도 적용해 함께 내보낸다.
- tiled 추론: 좌표를 history 채널에 실어 `tiled_apply`의 기존 크롭 로직이 그대로 자르게 하고,
  tile 안에서 다시 분리한다. tile마다 자기 위치의 좌표를 받는지 회귀 테스트로 고정했다
  (날짜변경선을 넘는 wrap 포함).

encoder 입력만 넓어지고 decoder 출력은 데이터 채널 그대로다(좌표는 재구성 대상이 아니다).
`SpatialAutoencoder`에 `input_channels`를 추가해 양 끝을 분리했고, `tiled_apply`는 이제
출력 채널 수를 입력이 아니라 **첫 예측**에서 가져온다.

**구버전 호환**: `positional_channels` 기본값은 0이라 키가 없는 v1/v2/v3 체크포인트는
기존 구조 그대로 로드된다(확인함). 신규 학습은 양쪽 CLI에서 3이 기본이며 `--positional-channels 0`으로 끌 수 있다.

운영 형상(721×1440, patch 256, overlap 64, 32 tiles)에서 좌표를 실은 tiled 샘플링을 직접 돌려 확인했다.

회귀 테스트 9건: `tests/test_positional_channels.py`. 같은 필드를 극지 좌표와 적도 좌표로
인코딩했을 때 latent가 달라지는지(= 모델이 위도를 실제로 구분하는지)를 포함한다.

### P2-9. `observed_fraction.npy`를 만들어놓고 쓰지 않는다 — 시작 I/O 낭비

`data.py:391-397`과 `era5_streaming.py:203-207`이 `[month, channel]` 관측 비율을 저장하는데,
**어디에서도 읽지 않는다.** 정작 `MonthlyWindowDataset._window_is_observed`(`data.py:565-574`)는
윈도우마다 `observed_mask` memmap에서 `history+1`개월을 다시 읽어 같은 값을 재계산한다.

29채널·721×1440·480개월 아카이브 기준 추정:

| 작업 | 읽는 양 |
| --- | --- |
| `load_monthly_archive` → `_require_no_inf_monthly` (states 전체 스캔) | ≈ 55 GiB |
| train dataset 생성 시 mask 재스캔 (7개월 × 378 윈도우) | ≈ 74 GiB |
| validation dataset 생성 시 mask 재스캔 | ≈ 8 GiB |
| **첫 gradient step 이전 합계** | **≈ 137 GiB** |

그리고 이 비용은 **resume할 때마다 전액 다시** 발생한다. `runpod_train.py`가 자랑하는
resume-safe 설계가 실제로는 재시작마다 100GB+ 스캔을 앞에 달고 있는 셈이다.

**수정 방향**: `_window_is_observed`가 `observed_fraction.npy`를 읽어
`fraction[start:start+history+1, :].mean(axis=0)`로 판정하게 바꾼다 (수학적으로 완전히 동일하다).
`_require_no_inf_monthly`의 전량 스캔은 아카이브 생성 시 한 번만 하고 schema에
`inf_checked: true`로 기록해 로드 때는 건너뛴다.

**✅ 수정 완료**: `MonthlyWindowDataset`이 기록된 fraction으로 판정한다. 근사가 아니라 **동일한
값**이다 — 저장된 각 값이 이미 그 월·채널의 공간 평균이고 모든 월이 같은 셀 수를 덮는다.
임계값 0.0/0.5/0.8/0.95에서 두 경로가 **완전히 같은 윈도우를 고르는지** parametrized 테스트로
고정했고, 접근하면 예외를 던지는 mask를 넘겨 빠른 경로가 mask를 아예 건드리지 않는지도 확인했다.

Inf 전량 스캔은 두 writer가 이미 월별로 검사하며 쓰므로 아카이브가 `inf_checked`를 기록하고
로드는 그것을 신뢰한다. 이 키가 없는 기존 아카이브는 예전대로 스캔한다(테스트로 고정).

실측: 소형 아카이브에서 윈도우 필터가 읽는 mask cell이 **435,456 → 0**.
운영 형상(29채널·721×1440·480개월·history 6) 환산으로 mask 재스캔 ≈ 93 GiB,
Inf 스캔 ≈ 54 GiB가 사라지고 0.36 MiB 읽기로 대체된다.

회귀 테스트 7건: `tests/test_archive_io.py`.

### P2-10. PR #1의 CI가 red이고 PR 본문의 테스트 수치가 낡았다

```
tests/test_era5_vertical.py:59: AssertionError
E  assert 'era5.v2' == 'era5.v1'
1 failed, 32 passed
```

`e4de3a3`가 `era5.py:162`의 adapter 라벨을 `era5.v1` → `era5.v2`로 올리면서
`tests/test_era5_vertical.py:59`를 같이 고치지 않았다. PR 본문은 "25 passed"라고 적혀 있지만
현재 HEAD에서는 32 passed / 1 failed다. GitHub의 `pytest` 체크도 `failure` 상태다.

**✅ 수정 완료**: assertion을 아카이브가 실제로 쓰는 `era5.v2`로 맞췄다. 아카이브 계약 자체는
바뀐 게 없고 테스트만 라벨을 따라가지 못한 것이다.

PR #1은 `main`과도 conflict(`mergeable_state: dirty`) 상태다. `main`이 `30f0fff` 이후
`fixed_step_data.py` / `large_data.py` 라인으로 갈라져 나갔기 때문이다.

## 4. 남은 작업

보고한 10건은 모두 이 브랜치에서 수정했다. 이후 과제로 남는 것:

1. **rank histogram과 신뢰구간.** test 윈도우가 아카이브 끝의 연속 구간 하나라 계절이 편향되고
   표본이 강하게 자기상관인데, 부트스트랩 신뢰구간이 없다. ensemble calibration도 spread-skill
   비율 하나로만 본다.
2. **`SpectralOperator2d`의 patch 주기성 가정.** P0-2는 conv padding만 고쳤다. `spatial_operator`
   backend는 patch에 `rfft2`를 걸어 여전히 patch를 주기 신호로 본다. FNO 계열의 관행이라
   별도 설계 판단이 필요하다. `spatial_conv`는 이 경로를 타지 않는다.
3. **경도 전 폭 patch 전략.** 경도를 자르지 않고 위도만 자르면 P0-2의 seam과 위 spectral 문제가
   원천 제거되지만 patch 메모리가 커진다.
4. **실데이터 검증.** 모든 수치는 합성 아카이브에서 얻었다. ERA5 실데이터와 GPU 학습으로
   재확인이 필요하다.

## 5. 재현 방법

```bash
python -m venv .venv && .venv/bin/pip install -e '.[test,io]'
.venv/bin/python -m pytest -q          # 74 passed
```

tiled 경로는 운영 형상(721×1440, patch 256, overlap 64, levels 3 → 32 tiles)에서도
직접 확인했다. `_latent_extent`가 실제 encoder 출력과 일치하는지는 downsample level 1~4 ×
크기 1~1440 조합으로 대조했고 불일치는 없었다. 어긋나면 `sample()`이
`Expected noise shape ...`로 즉시 실패하므로 조용히 잘못될 여지는 없다.

P0-3, P0-2, P1-4의 수치는 합성 4×8 격자 아카이브로 `train_flow_model` →
`LatentFlowForecaster` → `evaluate_flow_checkpoint`를 돌려 얻었다. 절차는 본문 각 항목에
그대로 적어두었다.
