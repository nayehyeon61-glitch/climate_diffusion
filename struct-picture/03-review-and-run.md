# 코드 검토와 실행

검토 시작 commit: `5479591a9ea1783549294e262a9fa17aa1caaab3`.

## 수정 사항

1. Adjoint ODE의 GRU condition gradient가 빠지던 경로를 연결했습니다. Direct ODE와의
   trajectory-only gradient 비교 테스트로 검증합니다.
2. Origin latent의 EMA scale 갱신을 history encoding보다 먼저 실행해 한 forward 안의
   history/origin/target latent 좌표계를 통일했습니다.
3. Dynamics split에서 effective purge를 `max(requested, horizon_steps - 1)`로 적용합니다.
   예를 들어 H=120이면 최소 119 start windows를 경계에서 제외합니다. Window stride는
   split 후 적용되므로 purge는 원 archive step 단위입니다. 과거 context 재사용은 허용하고
   미래 target 구간을 분리합니다. 정규화 통계도 validation 미래 target을 포함하지 않습니다.
4. Dynamics format을 frozen loader, forecast CLI, weather adapter, held-out evaluator에
   연결했습니다. Strided history, horizon 제한, lead 시간과 다중 lead metric을 처리합니다.
5. Schema/정확한 시간 간격 검사, 음수 ensemble/잘못된 lead 검사, 비유한 loss/gradient 오류,
   batch 크기를 고려한 epoch 평균, 150일 history 설명을 추가·수정했습니다.

## RunPod 명령

저장소 루트에서 실행합니다. 기존 환경에 수정 코드를 다시 설치합니다.

```bash
python -m pip install -e '.[test,io]'

train-climate-dynamics \
  --archive data/era5_6h_states.npz \
  --history-steps 6 --history-stride 120 --horizon-steps 120 \
  --latent-dim 512 --autoencoder-hidden-dim 768 --autoencoder-blocks 3 \
  --autoencoder-weight-decay 0 --dynamics-solver rk4 \
  --epochs 150 --batch-size 16 \
  --output download/flow-matching/dynamics/reviewed-h120.pt

forecast-climate-flow \
  --checkpoint download/flow-matching/dynamics/reviewed-h120.pt \
  --archive data/era5_6h_states.npz \
  --forecast-steps 120 --ensemble-size 8 --integration-steps 32 \
  --output outputs/reviewed-h120-forecast.npz

evaluate-climate-flow \
  --checkpoint download/flow-matching/dynamics/reviewed-h120.pt \
  --archive data/era5_6h_states.npz \
  --ensemble-size 8 --integration-steps 32 \
  --output outputs/reviewed-h120-evaluation.json
```

Archive는 미리 준비한 파일을 사용합니다. Forecast는 archive의 마지막 시점을 origin으로,
evaluation은 학습 때와 같은 시간 범위·schema를 사용합니다. 평가는 전체 test windows ×
전체 lead × ensemble에 대해 실행하므로 오래 걸릴 수 있습니다. 15일은
`--forecast-steps 60`, 30일은 120입니다. H=24 모델에서 30일 예측은 지원하지 않습니다.

기존 checkpoint도 예측용으로 로드할 수 있지만 gradient/latent-scale 수정이 이미 학습한
가중치까지 복구하지는 않습니다. Target이 겹치는 기존 split의 점수를 수정 후 점수로
취급하지 말고, 새로운 출력 이름으로 재학습·재평가하세요. 기존 LFS checkpoint와 실험
JSON·이미지는 보존했습니다.

## 검증 범위와 남은 과제

- CPU synthetic archive에서 1 epoch 학습, checkpoint 재로드, 재현성, adapter 시간 계약,
  CLI와 held-out 평가를 통과하는 회귀 테스트를 추가했습니다. 전체 테스트 18개가 통과했습니다.
  기존 RunPod dynamics checkpoint 5개도 모두 로드하고, train mean을 반복한 synthetic history로
  1-lead 예측의 shape와 유한값을 확인했습니다. RunPod 실데이터 재학습이나 예측 skill 개선을
  이 결과로 주장하지는 않습니다.
- Joint training의 latent collapse는 별도 실험 과제입니다. AE pretrain/freeze, latent 분산
  제약, ensemble 지표를 통한 checkpoint 선택은 다음 후보이며 현재 그림의 구현 기능에는
  포함하지 않습니다.
- Dynamics trainer는 archive 전체를 메모리에 읽습니다. Sharded preparation이 있더라도
  이 trainer가 out-of-core 전지구 고해상도 학습을 지원한다는 의미는 아닙니다.
- 기존 RunPod PCA 수치는 선형 재구성 비교 기준입니다. 비선형 AE의 오차 하한이나 forecast
  skill의 증명이 아닙니다. 작은 latent MSE만으로 분산 설명률 99.6%라고 단정할 수도 없습니다.
