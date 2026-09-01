# 0.25° Spatial Flow Matching 학습 매뉴얼

이 문서는 저해상도 `vector_mlp`와 0.25° 전지구 `spatial_conv` /
`spatial_operator` 경로를 구분합니다. 원해상도 전체 grid는 절대 dense MLP에
flatten하지 않습니다.

## 1. 데이터 준비

ERA5/HRES 입력은 동일 시간축과 규칙적인 latitude/longitude 좌표를 가져야 합니다.
pressure-level 변수는 `(time, level, lat, lon)`, surface 변수는
`(time, lat, lon)` 형태를 권장합니다.

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

출력 directory에는 `states.npy`, `times.npy`, `auxiliary.npy`,
`observed_fraction.npy`, `schema.json`이 생깁니다. `states.npy`는 memory map이므로
전체 연도를 RAM에 올리지 않고 필요한 month/patch만 읽습니다. 저장공간은 대략
`months × channels × 721 × 1440 × 4 bytes`이며, 120개월·32채널이면 약 15.9 GB입니다.

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

학습은 temporal train/validation/test split과 purge boundary를 유지합니다. 정규화는
train month에서 계산한 channel별 통계만 사용합니다. validation loss가 가장 낮은
v3 checkpoint, metrics, metadata와 SHA-256 manifest가 저장됩니다.

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

## 자원 및 실험 주의사항

- 먼저 128² patch smoke test, 다음 256² profiling, 마지막으로 전체 epoch 순서로
  확장합니다.
- GPU OOM이면 base/latent channel을 줄이고 gradient accumulation을 늘립니다.
- Fourier mode 수는 patch latent 크기보다 클 수 없으며 런타임에서 자동으로 잘립니다.
- patch validation은 빠른 model selection용입니다. 논문 결과는 frozen checkpoint로
  전체 test grid rollout을 수행해 위도 가중 RMSE/ACC 등 후속 metric도 계산해야 합니다.
- 현재 저장소에는 실제 ERA5 0.25° 데이터와 학습 weight가 포함되지 않습니다.
