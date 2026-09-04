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

**수정 방향**: AMP가 켜져 있을 때는 이 검사를 건너뛰고 `scaler.step()`의 skip 메커니즘에 맡기거나,
`scaler.step()` 이후 `optimizer` state의 `found_inf`를 보고 "skip 횟수"를 metric으로 기록만 한다.
fp32 경로에서만 hard fail을 유지하는 것이 안전하다.

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

**수정 방향**: patch 폭이 `grid_width`보다 작으면 경도 padding을 `replicate`로 전환하거나,
경도는 항상 전체 폭(1440)을 유지하고 위도만 자르는 patch 전략으로 바꾼다.
후자는 seam 문제와 P2-8(위치 정보 부재)을 동시에 줄인다.

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

**수정 방향**: validation 경로에서 `(t, x_0)`를 고정한다. epoch마다 같은 시드의
`torch.Generator`를 쓰거나, `t`를 `[0,1]` 등간격 격자로 고정하고 `x_0`를 윈도우 인덱스로
시드해 재현 가능하게 만든다. 이것만으로 위 편차가 0이 된다.

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

**수정 방향**: `target_velocity = target_latent.detach() - source_latent` (와
`history_latents.detach()`)로 flow 항을 분리하거나, AE 사전학습 → freeze → flow 학습의
2단계로 나눈다. 후자가 P0-3의 "선택이 AE만 본다" 문제도 함께 해소한다.

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

**수정 방향**: 몇 epoch마다 validation 윈도우 일부에 대해 작은 ensemble 샘플링 rollout을
돌려 normalized RMSE와 persistence 대비 skill score를 metric에 기록하고, 그 값으로
`best.pt`를 고른다. 평가 JSON에는 `skill_vs_persistence = 1 - rmse/persistence_rmse`를 추가한다.

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

`TRAINING_MANUAL.md:158`에 위도 가중 RMSE/ACC가 follow-up으로 이미 적혀 있으니 인지는 되어 있다.
다만 "학습 evaluation이 잘 되어 있는가"라는 질문에 대한 답으로는, **현재 숫자는 전역 예보 성능을
대표하지 않는다**는 점을 분명히 해둔다.

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

**수정 방향**: 전역 latent noise를 한 번 뽑아두고 각 tile이 그 slice를 쓰도록 `sample()`에
`noise` 인자를 추가한다.

### P2-8. patch에 위치 정보가 없다

모델 입력에 위도/경도 채널이 없고 정규화도 채널별 전역 상수다. 무작위로 잘린 256×256 patch만
보고는 그것이 적도인지 극지인지 알 수 없는데, 물리는 위도에 크게 의존한다.
`data.py:605`의 `random_crop`은 위도 시작점을 무작위로 뽑으므로 이 모호성이 학습 전체에 퍼진다.

**수정 방향**: `sin(lat)`, `cos(lon)`, `sin(lon)` 정적 채널을 입력에 concat한다.

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

### P2-10. PR #1의 CI가 red이고 PR 본문의 테스트 수치가 낡았다

```
tests/test_era5_vertical.py:59: AssertionError
E  assert 'era5.v2' == 'era5.v1'
1 failed, 32 passed
```

`e4de3a3`가 `era5.py:162`의 adapter 라벨을 `era5.v1` → `era5.v2`로 올리면서
`tests/test_era5_vertical.py:59`를 같이 고치지 않았다. PR 본문은 "25 passed"라고 적혀 있지만
현재 HEAD에서는 32 passed / 1 failed다. GitHub의 `pytest` 체크도 `failure` 상태다.

한 줄 수정이다:

```diff
--- a/tests/test_era5_vertical.py
+++ b/tests/test_era5_vertical.py
@@
-    assert schema["source_metadata"]["adapter"] == "era5.v1"
+    assert schema["source_metadata"]["adapter"] == "era5.v2"
```

PR #1은 `main`과도 conflict(`mergeable_state: dirty`) 상태다. `main`이 `30f0fff` 이후
`fixed_step_data.py` / `large_data.py` 라인으로 갈라져 나갔기 때문이다.

## 4. 권장 처리 순서

1. **P0-1** AMP finite guard 완화 — 안 고치면 500GB run이 재현성 있게 죽는다.
2. **P0-3** validation의 `(t, x_0)` 고정 — 한 줄 수준이고, 이걸 고쳐야 이후 실험 비교가 성립한다.
3. **P2-10** stale test 수정 — CI를 green으로 되돌린다.
4. **P0-2** patch 경도 seam — 경도 전체 폭 유지 또는 조건부 replicate padding.
5. **P1-4** flow 항에서 `target_latent.detach()` (또는 AE 2단계 학습).
6. **P1-5 / P1-6** 학습 중 샘플링 기반 skill 지표 + 위도 가중 RMSE/ACC + 월별 climatology baseline.
7. **P1-7 / P2-8 / P2-9** tiled noise 공유, 위치 채널, `observed_fraction` 활용.

## 5. 재현 방법

```bash
python -m venv .venv && .venv/bin/pip install -e '.[test,io]'
.venv/bin/python -m pytest -q          # 32 passed, 1 failed (P2-10)
```

P0-3, P0-2, P1-4의 수치는 합성 4×8 격자 아카이브로 `train_flow_model` →
`LatentFlowForecaster` → `evaluate_flow_checkpoint`를 돌려 얻었다. 절차는 본문 각 항목에
그대로 적어두었다.
