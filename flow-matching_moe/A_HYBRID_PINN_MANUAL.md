# 분리형 A에 실제 대기 방정식을 연결하는 Hybrid PINN

브랜치: `feature/a-hybrid-pinn-physics`.
기반: `feature/a-manifold-information-process`의 `d73d01f`.
기존 A→B→C 분리 학습과 A의 Z·상층 바람·지형 conditioning을 유지하고,
**A의 실제 6시간 latent drift가 복원하는 기상장에 PDE residual을 추가**한다.
설계 근거는 사용자 첨부 `2-1-2A에PINN구조까지?`의 **§§29–48 구면·기압좌표 방정식**이다.
구조 그림은 [18-a-hybrid-pinn.md](../struct-picture/18-a-hybrid-pinn.md)를 참조한다.
참고 원문 SHA-256: `748f1b83972f725179a735564c467f6c6261cd780d3c90b3c8d9f49478696fc1`.

`--pinn`/`PINN=1`을 지정할 때만 활성화된다. 이 옵션이 없는 기존 학습과
surface 출력 `msl, t2m, u10, v10`의 네 변수 계약은 유지한다.
학습된 PINN closure는 A loss의 보조 모듈이며 B/C 예측기의 새 시간 적분기가 아니다.

## 1. 필요한 실제 데이터

Z500/Z850와 10m 바람·2m 온도만으로 상층 운동량식을 계산하지 않는다.
선택한 **동일 기압면의 u/v/T/Z/omega**와 실제 지표기압 `sp`가 모두 필요하다.

| 구분 | PINN 입력과 단위 |
|---|---|
| 기본 conditioning 유지 | `z850,z500,z250,u850,v850,terrain_height,terrain_slope` |
| 기본 PINN 기압면 | 500·850 hPa; `--pinn-levels 500 850` |
| 각 기압면 추가/확인 | `u,v`: m/s, `t`: K, `z`: 지위고도 m, `w`: omega Pa/s |
| 실제 지표기압 | `sp`: Pa; 지하 기압면 및 미분 stencil 제외에 사용. `msl`로 대체하지 않음 |
| 정적 지형 | `terrain_height`: m, `terrain_slope`: 무차원. 기존 conditioning·static L2 유지 |
| 선택 확장 | `--pinn-levels 250 500 850`; 세 면 모두 u/v/T/Z/omega 필요 |
| 습도 | 기본 다운로드에 없음. 수동 제공 시 선택한 모든 기압면의 `q`: kg/kg 필요 |

ERA5 원본 `z`는 geopotential일 수 있다. 준비 단계는 선언된 단위를 검사해
`m²/s² ÷ 9.80665 → m`로 정규화하고, PDE 모듈이 **한 번만** `Phi=gZ`로 복원한다.
`w`는 기하학적 수직속도 m/s가 아니다. 단위가 m/s인 값을 omega로 받아들이지 않는다.

기존 필수 변수만 가진 INFO는 PINN 입력으로 부족하다. **새 INFO 파일/디렉터리**를 만들고
surface archive와 동일한 6h UTC 시각·격자를 사용한다. 결측/단위 모호함/시간 불일치를
0 대입이나 시간 보간으로 숨기지 않는다. 준비기는 기존과 같이 fully observed 자료를 요구하며,
PDE mask는 유효 자료 내부의 지하 기압면과 극점 주변 미분 stencil을 제외한다.

## 2. 실제 A 계산과 gradient

관측 origin의 surface와 정보로 `z0=E(surface0)+I(C0)`를 만들고,
기존 latent drift `b(z0)`로 다음 상태를 계산한다.

$$
z_1=z_0+\frac{\Delta t_{\rm hours}}{24}b(z_0),\qquad
\widehat C_0=D_{\rm info}(z_0),\quad
\widehat C_1=D_{\rm info}(z_1).
$$

정보 복원값을 train 통계로 역정규화하고, 시간차분은 **초 단위**로 계산한다.
공간 PDE는 두 복원 endpoint의 중간값에서 평가한다.
따라서 PINN gradient가 정보 decoder뿐 아니라 surface/information encoder와
latent drift까지 전달된다. 별도 surface tendency 감독이 surface decoder도 연결한다.
미래 관측 정보는 tendency 정답과 mask로만 사용하며 생성 예측의 conditioning을 갱신하지 않는다.

이 구현의 PINN은 **origin→+6h의 deterministic A drift**를 제약한다.
기존 A residual-FM의 120h member 경로에는 기존 CRPS·transition·trajectory 손실을 유지하지만,
모든 member/모든 시각에 PDE residual을 적용한 구현은 아니다.
Flow Matching의 생성 시간 `tau` 미분과 실제 기상 시간 미분을 구분한다.

## 3. 구현된 방정식과 범위

경도 `lambda`, 위도 `phi`는 radian, 기압 `p`는 Pa다.
`a=6371000 m`, `f=2 Omega sin(phi)`, `kappa=Rd/cp`로 두고,

$$
\mathcal A(Y)=\frac{u}{a\cos\phi}\partial_\lambda Y
 +\frac{v}{a}\partial_\phi Y+\omega\partial_pY
$$

를 사용한다. 학습 closure를 각각 `C_u,C_v,C_T`라 하면 잔차는 다음과 같다.

$$
\begin{aligned}
R_u&=\frac{\widehat u_1-\widehat u_0}{\Delta t}
 +\mathcal A(u)-\left(f+\frac{u\tan\phi}{a}\right)v
 +\frac{\partial_\lambda\Phi}{a\cos\phi}-C_u,\\
R_v&=\frac{\widehat v_1-\widehat v_0}{\Delta t}
 +\mathcal A(v)+fu+\frac{u^2\tan\phi}{a}
 +\frac{\partial_\phi\Phi}{a}-C_v,\\
R_T&=\frac{\widehat T_1-\widehat T_0}{\Delta t}
 +\mathcal A(T)-\kappa\frac{T}{p}\omega-C_T,\\
R_{\rm cont}&=\frac{\partial_\lambda u+\partial_\phi(v\cos\phi)}{a\cos\phi}
 +\partial_p\omega.
\end{aligned}
$$

두 인접 기압면 `p_i < p_(i+1)`의 정역학 일관성은 두께식으로 제약한다.

$$
R_{\rm thick}=Z_i-Z_{i+1}
 -\frac{R_d}{g}\frac{T_{v,i}+T_{v,i+1}}2
 \log\frac{p_{i+1}}{p_i}.
$$

습도가 있으면 `Tv=T(1+0.61q)`, 없으면 `Tv=T`인 건조 근사다.
위 식은 log-pressure 적분의 endpoint 사다리꼴 근사이며,
두 면만으로 상세한 연직구조나 정확한 층평균 온도를 복원했다는 의미는 아니다.

| 항목 | 구현/한계 |
|---|---|
| 수평 운동량 | 수평·연직 이류, Coriolis, 구면 곡률, geopotential gradient 포함 |
| 온도 | 수평·연직 이류와 압축 가열 포함; 미해상 가열 등은 closure가 보완 |
| 연속·정역학 | 기압좌표 연속식과 인접층 두께식. 전층 질량 보존 보장은 아님 |
| 미분 | 전지구 경도 주기 중앙차분, 위도/기압 좌표 간격을 사용하는 유한차분 |
| mask | 두 관측 endpoint의 `sp`와 이웃 stencil 기준. 모델이 예측한 sp로 loss를 회피하지 않음 |
| 지하층 | 선택한 전체 기압면과 이웃 stencil이 지상에 있는 칸만 사용하는 보수적 mask |
| closure | raw latent에서 u/v/T의 작은 MLP forcing; 마지막 층 0 초기화와 크기 L2 penalty |
| 정적 지형 | 기존 입력·복원 보존. 새 지형 상승류 경계식은 적용하지 않음 |

현재 포함하지 않은 항: `sp` 전층 질량 수지, 지표 omega 경계조건, 지형 상승류,
별도 thermal-wind residual, 수증기 예후식, 잠열·복사·에너지 수지.
드문 기압면 자료로 전층 적분이나 임의의 상단 omega 경계조건을 가정하지 않는다.
이 구현은 완전한 primitive-equation solver나 엄밀한 보존형 기후모델이 아니다.

## 4. Loss와 학습 단계

SI 단위 residual을 고정된 특성 크기로 나눈 뒤 유효 mask와 격자 면적으로 가중한다.
기본 크기는 운동량 `1e-3 m/s²`, 온도 `1e-4 K/s`, 연속식 `1e-5 s⁻¹`, 두께 `100 m`다.
이는 검증된 최적 계수가 아니다. `pinn_closure_normalized_rms`는 무차원 값이며,
u/v/T 각각의 residual·closure·예측 tendency·관측 tendency RMS도 SI 단위로 별도 기록한다.
무차원 loss 감소와 실제 변화량 개선을 함께 확인한다.

$$
L_{\rm PINN}=L_{\rm mom}+L_T+0.1L_{\rm cont}
 +0.1L_{\rm thick}+0.01L_{\rm closure}
 +0.1L_{\rm upper\ tendency}+0.1L_{\rm surface\ tendency}.
$$

상층 tendency 항은 선택된 u/v/T/Z/omega와 sp의 **관측 6h 변화량**을 감독한다.
이는 PDE residual만 줄이면서 정지장으로 수렴하는 문제를 견제한다.
기존 state/info/static reconstruction, decoded delta/drift, 분포·trajectory 손실은
warmup 후 기존 curriculum에 따라 함께 학습한다. 기존 loss의 `physics`와 새 `pinn_*`는 별도 항이다.

| 단계 | 동작 |
|---|---|
| A warmup, 기본 1 epoch | 관측 endpoint로 residual을 계산해 closure만 학습; 나머지 A와 B/C 모듈 동결 |
| A 본 학습 | 기존 6단계 curriculum 재개 + 복원장 PINN. encoder/decoder/drift/closure 함께 학습 |
| PINN ramp | warmup 뒤 3 epoch 동안 전체 가중치 `0.1 × min(1, joint_epoch/3)` |
| Best A 선택 | warmup·curriculum·ramp 완료 후 expert_validation. 기존 선택 점수에 plateau PINN 가중치를 더함 |
| B | 새 B 학습; A와 PINN 모두 동결. A의 seal/statistics/정보 계약 상속 |
| C | 기존 작은 LR representation 조정 유지. PINN closure/info head는 동결이고 새 PINN loss는 적용하지 않음 |

`--epochs`에는 warmup이 포함된다. `process` profile에서 필요한 최소치는
`warmup_epochs + max(5 × curriculum_interval + 1, ramp_epochs)`다.
trainer 기본 interval 2라면 기본 PINN 설정에서 최소 **12 epoch**,
runner 기본 interval 4라면 최소 **22 epoch**다. 아래 A60은 실행 예시이지 최적값이 아니다.

**이 브랜치의 A는 새로 초기화한다.** 같은 단계 A checkpoint의 `--init` 미세조정은 구현하지 않았다.
첨부 문서의 기존 A checkpoint warm start는 후속 확장 사항이며,
현재는 그 구조와 손실을 보존한 새 A 학습 → 새 B → C를 수행한다.
B/C는 parent checkpoint의 PINN 설정·정규화·weights를 상속하므로 별도 `--pinn`을 주지 않는다.
A가 바뀌었으므로 과거 B/C weights와 섞지 않는다.

## 5. 준비 → A → B/C 실행

새 브랜치를 받고 저장소 최상위에서 가상환경을 활성화한 뒤 실행한다.
데이터 경로는 실제 파일로 바꾼다. 아래 준비 명령의 기본 동작은 다운로드 계획 출력이다.

```bash
git fetch origin feature/a-hybrid-pinn-physics
git switch feature/a-hybrid-pinn-physics
python -m pip install -e '.[test,plots,io]'

export ARCHIVE=/workspace/data/era5-temporal-6h.npz
export INFO_FIELDS=/workspace/data/era5-extra-pinn-aligned.nc
export INFO=/workspace/data/era5-information-pinn.npz
export RUN=/workspace/experiments/a-hybrid-pinn-new

python scripts/prepare_era5_extra.py --archive "$ARCHIVE" --regrid linear \
  --pinn --pinn-levels 500 850
```

CDS 인증과 대상 기간·전송량을 확인한 뒤 전체 생성이 필요하면 다음을 사용한다.
인증/원본 크기/공간 보간 선택은 [ERA5 준비 안내](ERA5_EXTRA_DATA_PREPARATION.md)를 따른다.
그 안내의 기존 5변수 용량은 추가 PINN 변수까지 포함한 용량이 아니다.

```bash
python scripts/prepare_era5_extra.py --archive "$ARCHIVE" --regrid linear \
  --pinn --pinn-levels 500 850 --download --output "$INFO_FIELDS"

python -m climate_diffusion.physical_information --archive "$ARCHIVE" \
  --fields "$INFO_FIELDS" --output "$INFO" --pinn --pinn-levels 500 850

export MODE=enriched PROFILE=process DEVICE=cuda
export PINN=1 PINN_LEVELS='500 850'
export PINN_WEIGHT=0.1 PINN_WARMUP_EPOCHS=1 PINN_RAMP_EPOCHS=3
export A_EPOCHS=60 CURRICULUM_INTERVAL=4 B_EPOCHS=30 C_EPOCHS=10
export M=4 TAU=4 BATCH=2

bash scripts/run_a_information_120h.sh preflight
bash scripts/run_a_information_120h.sh A
bash scripts/run_a_information_120h.sh audit
```

`a.metrics.json`의 `pinn_*`와 `a-audit.json`의 AE delta/drift/persistence 비교를 확인한다.
새 A가 유용한지 판단한 후 같은 실행 환경에서 다음을 수행한다.

```bash
bash scripts/run_a_information_120h.sh B
bash scripts/run_a_information_120h.sh C
bash scripts/run_a_information_120h.sh validation
bash scripts/run_a_information_120h.sh render
# validation으로 설정을 확정한 뒤 최종 test를 별도로 실행한다.
bash scripts/run_a_information_120h.sh test
```

원본 디스크를 제한하는 streaming 경로도 `PINN=1`, `PINN_LEVELS='500 850'`을 사용한다.
`INFO`는 **새 shard 디렉터리**, `RUN`은 아직 없는 새 실행 디렉터리로 지정하고
[streaming 매뉴얼](STREAMING_ERA5_TRAINING.md)의 나머지 설정을 적용한다.

```bash
export INFO=/workspace/data/era5-pinn-shards
export RUN=/workspace/experiments/a-hybrid-pinn-stream-new
export PINN=1 PINN_LEVELS='500 850'
export THROUGH=A
bash scripts/run_streaming_a_information.sh
```

이 경로는 실제 다운로드를 시작한다. `THROUGH=A`라도 요청한 정보 기간의 다운로드를 마칠 때까지
producer가 계속될 수 있다. 기존 필수 변수만 있는 shard store에 PINN 변수를 이어붙이지 않는다.

## 6. 검증과 실험 해석

가벼운 합성 확인은 CPU에서 실행할 수 있다. 출력 경로는 새 이름을 사용한다.

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python scripts/smoke_hybrid_pinn.py \
  --output outputs/a-hybrid-pinn-smoke-new
python -m pytest -q
```

물리 단위·부호·마스크·manufactured solution, encoder/decoder/drift까지의 gradient,
warmup 동결·checkpoint 재로드·B/C 분리 유지가 코드 검증 대상이다.
**합성 테스트 통과는 ERA5 예측 성능 개선의 근거가 아니다.**

2026-09-22 검증: CPU/Python 3.12/PyTorch 2.14에서 전체 회귀 141개를 통과했고,
최종 통합 테스트 10개도 통과했다(뒤에 추가된 가중치 계산 검사 3개 포함, 현재 총 144개).
합성 smoke는 A7/B1/C1 epoch, member 2개, 실제 시간 20 step과 heldout 2개 사례를 완료했다.
B 전후 A 가중치 일치, B/C 전후 PINN 가중치 일치, warmup/ramp/best 선택 조건을 확인했다.
마지막 A epoch에서 PINN loss의 gradient norm은 encoder 0.05446, surface decoder 0.002469,
drift 0.007634, information encoder 0.03903, information decoder 0.1500, closure 0.07408이었다.
closure를 고정한 25회 최적화 검사에서도 복원된 상층·지표 tendency 오차가 감소했다.

실제 비교에서는 같은 데이터/분할/seed와 동일한 총 본 학습 epoch를 사용한 PINN off/on을 비교한다.
residual 감소와 함께 decoded drift 오차, `Delta state` 크기·방향, 바람/온도/Z 전파,
CRPS·spread–skill·평균 편향을 확인한다. `pinn_valid_fraction`이 낮으면 지형 mask 때문에
평가 지역이 제한된 것이므로 지역별 coverage를 함께 보고한다.
closure만 커지거나 정지장에 가까워지며 residual이 줄면 성공으로 해석하지 않는다.
C는 PINN loss를 재적용하지 않으므로 최종 C의 장기 물리 일관성은 별도 검증 대상이다.
