# Physical recurrent residual flow — 실제 ERA5 첫 A→B→C→validation→render 실행

2026-09-11 ~ 2026-09-13, 코드 커밋 `f897046` (`feature/physical-time-recurrent-flow`)에서
`scripts/run_recurrent_120h.sh`를 실제 ERA5 6시간 archive와 실제 GPU(RTX 4090,
torch 2.4.1+cu124)로 prepare→A→B→C→validation→render까지 처음부터 끝까지 실행한 결과입니다.

## 데이터·설정

- Archive: `data/era5-temporal-6h.npz` (원자료 `data/era5_wb2_6h_full.nc`, msl/t2m/u10/v10,
  6시간 간격), `state_dim=2048`.
- Preflight: history 6 step/stride4, horizon 20 step(120h), train pairs=50,567.
- `forecast_dynamics=recurrent_residual` (Stage A 명령에서 지정, checkpoint에 저장되어 B/C에 전달).
- 공통: `batch-size=2`, `ensemble-size=4`(학습)/`8`(validation), `integration-steps=4`(학습)/`16`(validation),
  `window-stride=4`, `learning-rate=0.001`, `seed=7`.
- **Stage C만 별도 가중치 사용** (`run_recurrent_120h.sh`에 `temporal_c` 배열을 추가):
  `--delta-member-weight 0.0`(B는 0.001), `--temporal-warmup-epochs 3`(B는 5). 나머지
  (`trajectory-weight=0.10`, `delta-weight=0.02`, `wind-speed-weight=0.01`,
  `wind-direction-weight=0.005`, `trajectory-edges=0`=전체 20구간)는 B와 동일.

## 단계별 실행 결과

| 단계 | epoch(실행/최대) | best epoch | best selection_score | 소요 시간 |
|---|---:|---:|---:|---:|
| A manifold | 15/50 (early-stop patience8) | 7 | 0.5692 | 약1분 |
| B specialize | 35/40 (early-stop patience8) | 27 | 1.3484 | 약56시간 |
| C joint | 7/10 (early-stop patience5) | 2 | 1.3552 | 약5.7시간 |

B는 epoch당 약1.4~2시간이 걸려 전체 파이프라인의 대부분(56시간/총62시간)을 차지했습니다.
C는 best(epoch2) 이후 5epoch 연속 개선이 없어 조기 종료됐고, best score(1.3552)가
B의 best(1.3484)보다 오히려 약간 높습니다(단, C는 energy/CRPS/anchor 등 B에 없는 항이
loss에 섞여 있어 selection_score를 절대 비교하기는 어렵습니다).

## Validation 평가 (`evaluate-climate-flow`, ensemble8/steps16/max-cases32/seed83)

| 지표 (정규화 좌표) | 값 |
|---|---:|
| state RMSE ↓ | 1.0480 |
| persistence RMSE (기준) | 0.9810 |
| climatology RMSE (기준) | 1.0150 |
| CRPS | 0.6062 |
| ensemble spread / rms spread | 0.6166 / 0.7512 |
| spread-skill ratio | 0.7168 |
| coverage@80% (목표 0.80) | 0.4947 |

**핵심 문제: 이번 실행의 모델 RMSE(1.048)는 persistence(0.981)·climatology(1.015) baseline보다
나쁩니다.** Coverage@80%도 0.49로 명목값(0.80)에 크게 못 미쳐 ensemble이 실제 분포를
과소하게 커버합니다. A/B/C 모두 patience에 걸려 조기 종료된 것과 함께 보면, 이번 설정으로는
아직 baseline을 넘어설 만큼 수렴하지 못한 상태로 판단됩니다. 이 실행을 "검증된 예측 모델"의
근거로 사용하지 않습니다.

## Member 영상/궤적

`members-6h/`(20 frame, +6~+120h, MP4+진단 GIF/PNG)와 `members-12h/`(10 frame, +12~+120h, GIF)
각각 8개 독립 noise member. `render-climate-trajectories`가 checkpoint에 기록된 validation split의
첫 origin으로 생성했습니다. 각 `member-NNN.json`은 해당 member 지표입니다.

## 코드 변경

`scripts/run_recurrent_120h.sh`에 Stage C 전용 `temporal_c` 하이퍼파라미터 배열을 추가했습니다
(Stage B의 `temporal` 배열과 분리, 위 "Stage C만 별도 가중치 사용" 참고). `scripts/_run_full_pipeline.sh`는
prepare→A→B→C를 순차 실행하는 러너 스크립트입니다(신규 추가).

## 파일

`training-{a,b,c}.metrics.json`: 단계별 전체 epoch 로그. `{a,b,c}.metadata.json`: 학습 설정/체크섬.
`{a,b,c}.pt`: 단계별 checkpoint (약13MB씩, Git LFS 추적). `preflight.json`/`preflight-a-verified.json`:
archive·시간·split 사전 검사. `validation.json`/`validation-routing.json`: 평가/routing 진단 원본 출력.
`forecast-6h.npz`: render에 사용된 원시 예측(Git LFS 추적). `code-commit.txt`: 이 실행에 사용한 코드 커밋.
`environment.txt`: `pip freeze` 스냅샷.

원본 ERA5 archive(`era5-temporal-6h.npz`, 690MB)는 용량 때문에 이 저장소에 포함하지 않았고
실행 환경의 `/workspace/data/era5-temporal-6h.npz`에 남아 있습니다.

## 한계 / 다음 시도 방향

- 단일 seed, 단일 origin 비교 영상입니다. 여러 seed/origin 반복은 하지 않았습니다.
- Stage C의 `delta-member-weight=0`/`temporal-warmup-epochs=3` 조정이 실제로 유효했는지는
  이번 1회 실행만으로는 판단할 수 없습니다(비교 대상인 기존 가중치로의 재실행 없음).
- RMSE가 baseline보다 나쁘고 coverage가 낮은 원인은 검증하지 않았습니다. 다음 실험에서는
  (a) B/C epoch 수·patience 확대, (b) trajectory-weight/delta-weight 재조정,
  (c) 더 많은 validation origin/seed로 재현성 확인을 권장합니다.
