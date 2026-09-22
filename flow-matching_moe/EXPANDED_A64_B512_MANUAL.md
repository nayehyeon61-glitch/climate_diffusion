# A64 / B512 확장 모델 실행 안내

브랜치: `feature/a64-b512-expanded`  
기반: `feature/a-hybrid-pinn-physics`의 `efdf085`

## 1. 확장하는 차원의 의미

| 설정 | 기존 기본값 | 이 브랜치 기본값 | 의미 |
|---|---:|---:|---|
| `manifold_dim` | 16 | **64** | A의 전역 상태 좌표, drift 출력, B의 intrinsic 잔차 좌표와 투영 결과 |
| `expert_latent_dim` | 64 | **512** | 각 B 전문가 내부의 feature code |
| `hidden_dim` | 128 | **512** | A/B에서 공유 설정으로 사용하는 MLP 중간층 폭 |
| `context_dim` | 64 | 64 | history context |
| `gate_hidden_dim` | 160 | 160 | router의 중간층 폭 |

여기서 **512는 A의 최종 좌표 차원이 아니라 B 전문가 내부 차원**이다.
A는 `전체 상태 → 512 → 64`, 디코더는 `64 → 512 → 512 → 전체 상태`를 사용한다.
전문가 내부를 512로 늘리면서 중간층이 128에 머물지 않도록 `hidden_dim`도 512로 늘린다.

B의 입력 전체가 64개 숫자로 바뀌는 것은 아니다. B는 기존처럼 A가 복원한 전체 상태와
history context, 시간 embedding, 64차원 intrinsic 잔차 조건을 받는다.
B가 생성하는 후보는 전체 상태 차원의 벡터장이며, A의 Jacobian으로 투영한 intrinsic 결과가
64차원이다. Jacobian은 `state_dim × 64`, metric은 `64 × 64`가 된다.
지상 정보와 추가 정보 encoder의 출력도 동일한 64차원으로 합쳐진다.

**공간 AE로의 변경이나 원본 관측을 B에 직접 전달하는 bypass는 포함하지 않는다.**
전역 DCT/MLP 표현, 접공간 투영식, loss 및 학습 단계는 기존 구조를 유지한다.
B에서는 A를 동결하고, C에서는 기존대로 A/B 일부를 함께 미세조정한다.
C가 A/B 전체를 동결하는 구조는 아니다.

## 2. 기존 데이터로 새 실행 시작

저장소 최상위에서 실행한다. 아래 `ARCHIVE`와 `INFO`는 이미 준비된 실제 경로로 바꾸고,
`RUN`은 이전 checkpoint가 없는 새 실행 디렉터리로 지정한다.
필요한 데이터와 PINN 수식·단위는 [Hybrid PINN 안내](A_HYBRID_PINN_MANUAL.md)를 따른다.

```bash
git switch feature/a64-b512-expanded
python -m pip install -e '.[test,plots,io]'

export ARCHIVE=/workspace/data/era5-temporal-6h.npz
export INFO=/workspace/data/era5-information-pinn.npz
export RUN=/workspace/experiments/a64-b512-pinn-new
export MODE=enriched PROFILE=process DEVICE=cuda
export PINN=1 PINN_LEVELS='500 850'
export MANIFOLD_DIM=64 EXPERT_LATENT_DIM=512 HIDDEN_DIM=512
export A_EPOCHS=60 B_EPOCHS=30 C_EPOCHS=10 CURRICULUM_INTERVAL=4
export M=4 TAU=4 BATCH=2

bash scripts/run_a_information_120h.sh preflight
bash scripts/run_a_information_120h.sh A
bash scripts/run_a_information_120h.sh audit
```

PINN은 계속 **명시적으로 `PINN=1`을 지정해야 켜진다.**
상층 U/V/T/Z/omega와 실제 surface pressure 등을 포함한 PINN용 sidecar가 필요하며,
기존의 최소 정보만 있는 `INFO`에 옵션만 추가해서 사용할 수는 없다.
`INFO`가 아직 없고 정렬된 추가 변수 NetCDF가 있다면 `INFO_FIELDS`를 설정하면 된다.
`preflight`는 그 파일로 sidecar를 생성하며 다운로드는 수행하지 않는다.

`a.metrics.json`과 `a-audit.json`에서 복원·변화량·drift와 persistence 비교를 확인한 뒤
같은 `RUN`에서 B/C를 실행한다.

```bash
bash scripts/run_a_information_120h.sh B
bash scripts/run_a_information_120h.sh C
bash scripts/run_a_information_120h.sh validation
```

A는 위 환경변수로 모델 크기를 정한다. **B/C는 부모 checkpoint의 모델 크기를 상속**하므로
B 실행 직전에 환경변수를 바꿔도 A의 64차원 좌표를 다른 크기로 교체하지 않는다.
직접 Python 학습 명령을 사용할 때 대응 옵션은
`--manifold-dim 64 --expert-latent-dim 512 --hidden-dim 512`이다.
streaming runner도 A 학습 시 같은 환경변수를 사용한다.

차원이 바뀐 기존 weights를 자동으로 resize하거나 이식하지 않는다.
**새 A → 그 A를 상속한 새 B → 그 B를 상속한 C 순서로 재학습**해야 한다.
과거 16차원 A/B checkpoint와 확장 모델 weights를 섞지 않는다.

## 3. 합성 동작 확인과 성능 비교

실제 확장 크기 `64 / 512 / 512`의 작은 합성 A/B/C 실행:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python scripts/smoke_hybrid_pinn.py \
  --expanded --output /tmp/a64-b512-pinn-smoke-new
```

`--expanded`가 없으면 기존의 가벼운 합성 설정을 사용한다.
이 검증은 차원 연결, PINN gradient, checkpoint 계승과 단계별 학습의 동작을 확인하기 위한
것이며, ERA5에서의 예측력 향상을 입증하지 않는다.

이 브랜치에서는 위 확장 smoke와 관련 회귀 테스트 52개를 통과했다.
실행 설정과 확인한 범위는 [합성 검증 기록](../docs/results/expanded-a64-b512/README.md)에 있다.

확장 모델은 기존 모델보다 메모리와 투영 계산량이 늘어난다.
동일한 데이터 분할에서 우선 A의 재구성과 6시간 변화량을 평가하고,
B/C에서는 24·72·120시간 RMSE·CRPS와 ensemble spread를 함께 확인한다.

차원 비교가 필요하면 목적에 따라 다음 둘을 구분한다.

| 비교 목적 | `manifold_dim / hidden_dim / expert_latent_dim` |
|---|---|
| 기존 전체 모델과 확장 모델 비교 | 기존 `16 / 128 / 64` 대 확장 `64 / 512 / 512` |
| A 차원만 바꾸는 통제 실험 | `16 / 512 / 512`, `64 / 512 / 512`, `128 / 512 / 512` |

첫 비교에서 개선되더라도 그 원인을 A 차원 증가만으로 해석할 수는 없다.
통제 실험에서는 각 설정에 별도 `RUN`을 사용하고, 동일한 데이터·loss·학습 예산으로 비교한다.
