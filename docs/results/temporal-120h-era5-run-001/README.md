# 120시간 Physics-informed Manifold MoE: 실제 ERA5 + GPU 첫 재현

2026-09-10, `flow-matching_moe/RETRAIN_120H.md`를 코드 커밋 `0e641b2`에서 그대로 따라
**실제 ERA5 6시간 archive**와 **실제 GPU(RTX 4090, torch 2.4.1+cu124)**로 A→B→C→validation→test→
member 영상까지 처음부터 끝까지 실행한 결과입니다. `docs/results/temporal-120h-smoke/`는 CPU 합성
toy 데이터였고, 이 문서가 실데이터/실GPU로 처음 돌린 결과입니다.

## 데이터·설정

- 원자료: `data/era5_wb2_6h_full.nc` (msl/t2m/u10/v10, 1959-01-01~2021-12-31, 6시간 간격,
  native grid 32×64) → `prepare-climate-fixed-step-data`로 18×36 coarse grid archive 생성
  (`era5-temporal-6h.npz`, `state_dim=2048`).
- Preflight (`scripts/prepare_temporal_120h.py`): history 6 step/stride4 (24h 간격 관측6개),
  horizon 20 step(120h/5일), five-way split purge=19 window. 실측 window 수:
  train 50,528 / expert_validation 9,200 / calibration 13,800 / validation 9,200 / test 9,200.
- 나머지 구조/차원(num-experts4, manifold-dim16, expert-latent-dim64, gate-hidden-dim160)과
  loss 계수는 `RETRAIN_120H.md` 예시값을 그대로 사용했습니다. **batch-size만 문서 예시(2)에서
  16으로 올렸습니다** — batch2에서 GPU 메모리가 638MiB/24.5GB만 쓰이고 epoch당 약16분이 걸려
  (커널 호출 오버헤드가 병목으로 추정) batch16으로 올리자 epoch당 약3.4분으로 단축됐습니다
  (동일 첫 epoch 기준 약4.7배). 다른 하이퍼파라미터는 문서와 동일합니다.

## 단계별 실행 결과

| 단계 | epoch(실행/최대) | best epoch | best validation score | 소요 시간 |
|---|---:|---:|---:|---:|
| A manifold | 14/50 (early-stop patience8) | 6 | 0.57570 | 약3.5분 |
| B specialize | 9/40 (early-stop patience8) | 1 | 1.34167 | 약30분 |
| C joint | 10/10 (전부 실행, 개선 지속) | 10 | 1.32268 | 약25분 |

B는 train loss(2.686→2.082)는 계속 줄었지만 validation은 epoch1 이후 더 나아지지 않아 조기 종료됐습니다.
C는 10epoch 예산을 다 채울 때까지 validation이 계속(느리게) 개선 중이었으므로(1.343→1.323),
**더 많은 epoch을 주면 더 개선될 여지가 있습니다** — 이번 실행은 문서 기본 예산을 그대로 따른 결과입니다.

## Validation / Test 평가 (`evaluate-climate-flow`, ensemble8/steps16/max-cases32/seed83)

| 지표 (정규화 좌표) | validation | test |
|---|---:|---:|
| state RMSE ↓ | 0.8799 | 0.8929 |
| persistence RMSE (기준) | 0.9810 | 0.9785 |
| climatology RMSE (기준) | 1.0150 | 1.0335 |
| CRPS ↓ | 0.5562 | 0.5685 |
| empirical central80% coverage | 33.5% | 32.4% |
| spread/skill ratio | 0.423 | 0.414 |

Test는 validation과 거의 같은 범위이므로 validation으로 고른 설정이 명백히 overfit된 것은 아닙니다.
**모델 RMSE가 persistence와 climatology 둘 다보다 낮습니다** — 5일 앙상블 예측이 두 단순 기준선보다
정확하다는 첫 실데이터 증거입니다. 다만 **central80% coverage가 이상적인 80%에 크게 못 미치고
(약33%) spread/skill ratio도 1보다 훨씬 작아(~0.42), 앙상블이 뚜렷하게 under-dispersed합니다** —
불확실성 정량화는 이번 실행에서 보정되지 않았습니다.

## Expert 전문화 진단 (`manifold_diagnostics`, validation split, 실제 생성 ODE 경로)

- Gate usage 4개 expert에 걸쳐 [13.1%, 25.5%, 31.4%, 30.0%] — 한 expert로 붕괴하지는 않았습니다.
- **Candidate cosine similarity 0.933** — expert들이 내는 velocity 후보가 서로 상당히 비슷합니다
  (완전한 expert collapse의 경계에 가깝습니다).
- Teacher-forced 감사에서 "지역적으로 가장 가까운 expert"와 "실제 gate가 고른 expert" 모두 "실제
  최소오차 expert"와 일치하는 비율이 **18.75%**에 불과합니다 — 국소 gate/geometry가 실제로 어떤
  expert가 더 정확한지와 강하게 연결되어 있다는 증거는 약합니다.
- Anchor RMSE 0.041(낮음, A의 좌표계를 C가 잘 유지함), manifold reconstruction RMSE 0.826(다른
  state-space 오차와 비슷한 수준).

**요약**: 이번 첫 실데이터 실행은 (a) 5일 앙상블 예측이 persistence/climatology보다 낫다는 것,
(b) 학습·저장·평가·영상 파이프라인이 실제 ERA5+GPU에서 끝까지 작동한다는 것을 보여줍니다.
동시에 (c) 불확실성 보정과 (d) expert 전문화 모두 아직 약하다는 것도 같이 보여줍니다.
이 수치들을 "물리적으로 의미 있는 regime 분리"나 "잘 보정된 확률 예측"의 증거로 해석하지 않습니다.

## 첫 스텝 이후 사실상 정체(quasi-static) — member 영상에서 확인된 문제

`members-12h/member-000.json`의 `amplitude_ratio`(예측 변화량 RMS / 실제 변화량 RMS,
`prediction_tendency_rms`/`truth_tendency_rms`)를 보면, **+12h(첫 스텝)의 변화량 크기는
실제와 비슷하지만(t2m 0.94, msl/u10/v10은 1.9~2.2로 다소 과함), +24h부터 +120h까지는
실제 변화량의 3~8%만 만듭니다** (예: t2m 0.94 → 0.010~0.036, msl 2.16 → 0.033~0.084).
즉 학습된 모델은 첫 물리적 lead에서만 의미 있게 움직이고 이후로는 거의 정지한 state
근처에 머뭅니다 — 영상에서 "거의 안 움직인다"고 보이는 것은 착시가 아니라 이 현상입니다.
`by_variable_ensemble.*.mean_tendency_rms`(summary.json)에서도 8개 member 평균으로
같은 패턴이 재현됩니다.

가능한 원인(검증하지 않은 가설):
- B가 validation 개선 없이 epoch1에서 조기 종료됨 — expert가 다단계 dynamics를 충분히
  학습하기 전에 멈췄을 수 있습니다.
- `--trajectory-edges 2`로 학습 중에는 항상 전체 20구간(120h) 중 2구간짜리 sub-block만
  보므로, 긴 구간에 걸쳐 움직임을 유지하도록 직접 감독하는 신호가 약합니다.
- `trajectory-weight`(0.1)/`delta-weight`(0.02)가 작고 C가 10epoch뿐이라 전반적으로
  undertrained일 수 있습니다.

다음 실험에서는 trajectory-weight/delta-weight를 올리거나 B/C epoch을 늘려서, 그리고
가능하면 `--trajectory-edges 0`(전체 20구간 joint loss)로 이 현상이 줄어드는지 확인하는
것을 권장합니다.

## 그림

`figures/training-abc.png`(A→B→C train loss·validation selection score, best epoch 표시),
`figures/evaluation.png`(validation/test normalized RMSE vs persistence/climatology,
CRPS/coverage/spread-skill), `figures/routing.png`(생성 경로 gate usage, candidate cosine
등 전문화 진단). `scripts/visualize_era5_run.py`로 위 JSON만 읽어 생성했습니다 — 기존
`visualize_manifold_moe.py`는 구버전 진단 스키마(`test_pca` 등)를 기대해서 현재
`manifold_diagnostics.py` 출력과 맞지 않아 새로 작성했습니다.

## Member 영상/궤적

`members-6h/`(20 frame, +6~+120h, MP4)와 `members-12h/`(10 frame, +12~+120h, GIF) 각각 8개
독립 noise member. Origin은 checkpoint에 기록된 validation split의 첫 origin
(`2009-05-19T06:00:00Z`)이며 같은 archive 안의 실제 미래 관측과 나란히 렌더링합니다.
각 `member-NNN.json`은 해당 member 지표, `summary.json`은 ensemble aggregate입니다.

## 파일

`preflight.json`/`preflight-a-verified.json`: archive·시간·split 사전 검사.
`training-{a,b,c}.metrics.json`, `{a,b,c}.metadata.json`: 단계별 전체 epoch 로그와 학습 설정/체크섬.
`validation.json`/`test.json`: `evaluate-climate-flow` 원본 출력. `validation-routing.json`:
`manifold_diagnostics` 원본 출력. `code-commit.txt`: 이 실행에 사용한 코드 커밋.

체크포인트(`a.pt`/`b.pt`/`c.pt`, 약13MB씩)와 원본 archive(`era5-temporal-6h.npz`, 690MB)는
용량 때문에 이 저장소에는 커밋하지 않았고 실행 환경의 `/workspace/experiments/temporal-120h-run-001/`
및 `/workspace/data/era5-temporal-6h.npz`에 남아 있습니다.

## 실행하지 않은 것 / 한계

- 단일 seed, 단일 origin 비교 영상입니다. 여러 seed/origin에 걸친 반복은 하지 않았습니다.
- Batch size를 16으로 올린 것 외에는 하이퍼파라미터 탐색을 하지 않았습니다(계수/온도/ridge 등은
  `RETRAIN_120H.md` 시작값 그대로).
- ACC/PSD, 계절별 sampling, 15~30일 장기 horizon은 이번 실행 범위가 아닙니다(이 문서의 H=120은
  120시간/5일이며, 기존 `TRAINING_README.md`의 H=120인 720시간/30일과 다릅니다).
- 불확실성 보정(coverage/spread-skill)과 expert 전문화 모두 약하다고 위에서 이미 밝혔습니다 —
  이 실행을 "검증된 예측 모델"의 근거로 사용하지 않습니다.
