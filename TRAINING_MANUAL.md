# 0.25° Spatial Flow Matching 학습 매뉴얼

이 문서는 저해상도 `vector_mlp`와 0.25° 전지구 `spatial_conv` /
`spatial_operator` 경로를 구분합니다. 원해상도 전체 grid는 절대 dense MLP에
flatten하지 않습니다.

## 1. 데이터 준비

ERA5/HRES 입력은 동일 시간축과 규칙적인 latitude/longitude 좌표를 가져야 합니다.
pressure-level 변수는 `(time, level, lat, lon)`, surface 변수는
`(time, lat, lon)` 형태를 권장합니다.

실제 ERA5가 변수별 NetCDF로 분리되어 있다면 먼저 아래 명령을 사용합니다.

```bash
prepare-era5-climate-flow \
  --source data/era5/ \
  --variables msl t2m u10 v10 z t q u v \
  --target-lat-points 721 --target-lon-points 1440 \
  --output data/monthly_climate_spatial_025
```

생성된 archive부터는 아래의 기존 학습·rollout·evaluation 명령이 동일합니다.

HRES 분석장 또는 forecast archive는 init/step을 하나의 valid-time 축으로 선택한 뒤
동일한 spatial archive로 만듭니다. 서로 다른 lead를 한 archive에 혼합하지 않습니다.

```bash
prepare-hres-climate-flow \
  --source data/hres/ --lead-hours 0 \
  --variables msl t2m u10 v10 z t q u v \
  --target-lat-points 721 --target-lon-points 1440 \
  --output data/monthly_hres_spatial_025
```

```bash
prepare-climate-monthly-data \
  --fields data/era5_025_history.zarr \
  --integrated data/integrated.parquet \
  --variables msl t2m u10 v10 z t q u v \
  --layout spatial \
  --target-lat-points 721 \
  --target-lon-points 1440 \
  --output data/monthly_climate_spatial_025
```

출력 directory에는 `states.npy`, `observed_mask.npy`, `times.npy`, `auxiliary.npy`,
`observed_fraction.npy`, `schema.json`이 생깁니다. `states.npy`는 memory map이므로
전체 연도를 RAM에 올리지 않고 필요한 month/patch만 읽습니다. 저장공간은 대략
`months × channels × 721 × 1440 × 4 bytes`이며, 120개월·32채널이면 약 15.9 GB입니다.

### 필수 데이터 계약

- 시간은 정렬 후 유일하고 정확히 연속된 calendar month여야 합니다.
- `1월, 2월, 4월`을 세 개의 연속 행으로 처리하지 않습니다. 결측 월이 있으면
  archive 생성 또는 로딩이 실패합니다.
- `+Inf/-Inf`는 물리적으로 보간하지 않으며 변수·월·첫 위치·개수를 포함해 즉시
  실패합니다. 따라서 `np.nan_to_num`의 float extrema가 저장되지 않습니다.
- `NaN`은 원본 archive와 `observed_mask`에 보존합니다. raw-month split을 만든 후
  train 구간의 finite 값만으로 feature/channel 평균과 scale을 계산합니다.
- validation/test의 `NaN`도 train mean으로만 보간합니다. spatial 입력에서는 이 값이
  normalization 후 0이 됩니다.
- train 구간의 feature/channel이 전부 결측이면 학습을 중단합니다.
- `--min-observed-fraction 0.95`처럼 window별 최소 관측률을 설정할 수 있습니다.

## 2. 작은 smoke test

대규모 job 전에 작은 channel 집합과 convolutional backend로 입출력·loss·checkpoint를
검증합니다.

```bash
train-climate-flow \
  --archive data/monthly_climate_spatial_025 \
  --model-backend spatial_conv \
  --history-months 6 \
  --patch-height 128 --patch-width 128 --tile-overlap 32 \
  --spatial-base-channels 16 --spatial-latent-channels 8 \
  --batch-size 1 --epochs 2 \
  --min-observed-fraction 0.95 \
  --output download/flow-matching/smoke/spatial-smoke.pt
```

## 3. Operator 사전학습

```bash
train-climate-flow \
  --archive data/monthly_climate_spatial_025 \
  --model-backend spatial_operator \
  --history-months 6 --lead-months 1 \
  --spatial-base-channels 32 \
  --spatial-latent-channels 16 \
  --spatial-downsample-levels 3 \
  --operator-modes-lat 12 --operator-modes-lon 24 \
  --patch-height 256 --patch-width 256 --tile-overlap 64 \
  --batch-size 1 --mixed-precision \
  --gradient-accumulation-steps 8 \
  --gradient-checkpointing \
  --num-workers 4 \
  --epochs 100 \
  --output download/flow-matching/monthly-spatial-025/climate-flow-spatial-025.pt
```

학습은 raw month를 먼저 train/validation/test로 나눈 뒤 각 구간 안에서만 window를
생성합니다. 따라서 history와 target의 원시 월은 split 사이에서 공유되지 않습니다.
정규화·결측 보간은 train month에서 계산한 channel별 통계만 사용합니다. validation loss가 가장 낮은
v3 checkpoint, metrics, metadata와 SHA-256 manifest가 저장됩니다.

manifest의 split에는 raw-month index/time 범위와 실제 window index가 함께 저장됩니다.
checkpoint에는 archive schema·shape·coordinate·time 계약의 SHA-256 fingerprint가
들어가며, evaluation archive가 다르면 추론 전에 실패합니다.

## 4. Frozen rollout과 평가

```bash
forecast-climate-flow \
  --checkpoint download/flow-matching/monthly-spatial-025/climate-flow-spatial-025.pt \
  --archive data/monthly_climate_spatial_025 \
  --months 1 --ensemble-size 8 --integration-steps 32 \
  --output outputs/spatial-025-ensemble.npz

evaluate-climate-flow \
  --checkpoint download/flow-matching/monthly-spatial-025/climate-flow-spatial-025.pt \
  --archive data/monthly_climate_spatial_025 \
  --ensemble-size 8 --integration-steps 32 \
  --output outputs/spatial-025-evaluation.json
```

loader는 checksum을 확인한 다음 `eval()`, `requires_grad_(False)`와
`inference_mode()`를 적용합니다. checkpoint의 training patch와 overlap 설정을
rollout에도 사용하며, 전지구 출력은 원래 latitude/longitude 좌표로 복원합니다.

## 실제 0.25° 학습 전 안전 검사

```bash
python -m pytest -q

prepare-climate-monthly-data \
  --fields data/era5_025_history.zarr \
  --integrated data/integrated.parquet \
  --variables msl t2m u10 v10 z t q u v \
  --layout spatial \
  --target-lat-points 721 --target-lon-points 1440 \
  --output data/monthly_climate_spatial_025
```

archive 준비가 끝나지 않거나 `Inf`, 완전 결측 channel, 중복·결측 월 오류가 나오면
장기 학습을 시작하지 마십시오. 학습 중에는 normalized input, auxiliary, latent,
loss와 unscaled gradient를 검사하고, rollout/evaluation도 non-finite 값을 거부합니다.
evaluation JSON은 표준 JSON으로만 저장되어 `NaN` literal을 허용하지 않습니다.

## 자원 및 실험 주의사항

- 먼저 128² patch smoke test, 다음 256² profiling, 마지막으로 전체 epoch 순서로
  확장합니다.
- GPU OOM이면 base/latent channel을 줄이고 gradient accumulation을 늘립니다.
- Fourier mode 수는 patch latent 크기보다 클 수 없으며 런타임에서 자동으로 잘립니다.
- patch validation은 빠른 model selection용입니다. 논문 결과는 frozen checkpoint로
  전체 test grid rollout을 수행해 위도 가중 RMSE/ACC 등 후속 metric도 계산해야 합니다.
- 현재 저장소에는 실제 ERA5 0.25° 데이터와 학습 weight가 포함되지 않습니다.
