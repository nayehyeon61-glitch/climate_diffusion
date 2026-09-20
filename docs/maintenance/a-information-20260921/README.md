# 분리형 A 코드 점검·정리 — 2026-09-21

대상: `feature/a-manifold-information-process`, 점검 시작 HEAD `f5c9026dd59b84610d73534007abfb2f2b148022`.
기존 A→frozen-A B→small-LR C, surface4변수, 추가7입력, 물리시간 recurrence를 유지했다.
**모델 재설계·loss 계수 변경·ERA5 재학습 작업이 아니다.**

## 발견 및 수정

| 항목 | 이전 동작 | 보수 및 검증 |
|---|---|---|
| 평가 RMS 집계 | case RMSE/spread를 단순 평균 | case MSE/variance 평균 후 sqrt. 이전 값은 `mean_case_*`로 보존. 오차1/3 사례에서2가 아니라√5임을 테스트 |
| 예측 파일 보존 | JSON report와 forecast에 같은 새 `.npz` 경로를 주면 마지막 JSON 쓰기로 예측을 덮어쓸 수 있음 | 경로 충돌을 checkpoint 로드 전에 거부 |
| B/C loss 로그 | 일부 weighted 항만 있어 실제 total을 재구성할 수 없음 | specialization, C marginal Energy/CRPS·PI·anchor 기여 추가. 기존 계산식/순서 유지, weighted 합=loss 검증 |
| A drift-only 진단 | 사용하지 않을 stochastic sampler를 먼저20step 실행 | 진짜 drift-only 경로로 우회; sampler가 호출되면 실패하는 테스트와20step drift backward |
| 입력 schema | dynamic 변수 kind의 임의 문자열, optional 중복 요청을 조기에 거부하지 않음 | static/dynamic 정확한 enum과 optional 중복 검증; sidecar 준비 시6h 요구 |
| window 선택 | cap1 또는 큰 stride로1개가 선택되면 임의 첫2개로 대체 | 설명 있는 fail-fast. metric에 필요한2개 이상을 사용자가 지정 |
| A 품질 감사 split | C 선택용 validation을 A 진행 판단에도 기본 사용 | 기본 expert_validation으로 이동. 과거 재현은 `--split validation`으로 명시, test는 감사 대상으로 거부 |
| 실행 문서 | 복사 블록에 audit 이후 자동 진행 위험, full/pilot 설정 모호 | 새 폴더·로그·명시적 full 설정·A 확인/test 확인을 포함한 단일 블록, CLI/Bash 검사 |

코드: [평가](../../../src/climate_diffusion/information_forecast.py),
[단계별 objective](../../../src/climate_diffusion/train_information_process.py),
[A 모델](../../../src/climate_diffusion/information_process.py),
[물리정보 검증](../../../src/climate_diffusion/physical_information.py),
[테스트](../../../tests/test_information_process.py).

## 확인된 모델 계약

- 출력은 msl(Pa), t2m(K), u10/v10(m/s) 네 변수다. 추가 입력은 Z850/Z500/Z250, U850/V850, terrain height/slope다.
- origin 추가 정보를 history 및 horizon 동안 고정한다. 미래 상층 관측은 감독용이며 rollout/router 입력으로 들어가지 않는다.
- A의 msl 포함 surface state·tendency fair CRPS 및 joint trajectory Energy가 실제 stochastic graph에 연결된다.
  static terrain은 learned head의 area-weighted L2이며 distribution score에서 제외된다. pressure에도 deterministic anchor는 남는다.
- B는 A manifold/information 모듈을 동결하되 decoder 입력의 gradient는 유지한다. A auxiliary sampler를 B experts로 복사하지 않는다.
- C는 기존 marginal scores·PI·anchor 보정이다. A의 새 curriculum 전체를 C에 적용하는 Loss V2가 아니다.
  C에서 계산되는 `static_l2`/AE delta 등은 모두 active loss라는 의미가 아니므로 `weighted_*`를 확인한다.
- FM tau transport와 physical drift q/day가 구분되고6h×20step으로 진행한다. member별 independent persistent noise,
  다음 step 입력=이전 출력,20step loss backward, checkpoint reload 및6h/12h prefix가 유지된다.
- checkpoint format/parameter shapes는 그대로여서 이 브랜치의 기존 checkpoint를 읽는다. 다른 legacy/joint-AB 형식은 명시적으로 거부한다.

## 실제 검증 범위

기존 테스트75개 → 보수 후 **84개 통과**, CPU에서24.15초. 기존 경고2개(torch JIT deprecation, test scalar conversion)는 남는다.
새 회귀검사9개는 RMS 집계·경로 충돌·A/B/C별 loss 합/실제 gradient·frozen B·drift-only 우회·schema·static score 제외·잘못된 cap/audit를 다룬다.
CLI help5개, runner Bash 문법, 매뉴얼 첫 복사 블록 Bash 문법, compileall, synthetic 실제 preflight를 확인했다.
수정한 Mermaid3개를 실제 parser로 검사했다.

수정 전/후 각각 같은 시간의존 synthetic으로 A6/B2/C2를 **처음부터** 실행했다.
조건:320개 관측,4×4×8 field, manifold4/K2/M4/tau4, train4window/epoch, validation2origin/seed83, full20step BPTT.
상층 synthetic은 surface toy wave에서 파생됐으며 실제 추가 ERA5 정보의 효과를 증명하지 않는다.

[동일성 감사 JSON](verification.json):

- A/B/C 각각113개 state tensor가 **수정 전후 bitwise 동일**, epoch별 train/validation loss와 selection도 정확히 동일.
- B에서77개 frozen tensor 유지. A/B/C best epoch6/2/1.
- 저장 forecast `[4,20,128]`도 bitwise 동일. 모델 동작이 바뀐 것이 아니라 집계·검증·로깅을 보수했다.
- 모든4member의6h MP4는20frame,12h GIF는10frame. valid time과 prediction 배열의 exact subset을 확인했다.
- `weighted_*` 합과 total의 최대 차이는9.1e-7 이하(부동소수점 연산 순서).
- CPU PyTorch2.14.0+cpu, CUDA 없음. 보수 후 stage별 A3.72/B9.00/C20.32초,
  학습·평가·렌더 약60초, 프로세스 누적 peak RSS533024KiB(약521MiB). 실제4090 VRAM/시간은 미측정.
  수정 전 실행은 렌더를 생략했으므로 총 wall time의 직접 속도 비교는 하지 않는다.

## 성능 해석: 여전히 미해결인 부분

| 동일한 보수 후 synthetic validation | 값 |
|---|---:|
| pooled normalized RMSE |1.592719|
| state fair CRPS |1.155066|
| transition fair CRPS |0.797325|
| joint trajectory Energy |1.702615|
| pooled spread |0.248128|
| 80% interval coverage |5.87%|

기존 산술평균 RMSE는1.592675였다. **그 차이는 집계 수정이지 학습/성능 변화가 아니다.**
짧은 smoke는 여전히 심하게 underdispersed하며 새 A가 dynamics 문제를 해결했다고 주장할 수 없다.
[expert_validation A 감사](a-audit-expert-validation.json)에서 AE 변화량 진폭비는 약.032~.045,
drift는.034~.038, drift error는4변수 모두 zero-tendency 기준보다 약간 크다.
이는 수정 전과 동일한 짧은 toy 학습 모델이며 실제 ERA5의 원인/한계 증명이 아니다.
과거 validation 감사와 이번 expert_validation 감사의 수치는 서로 다른 관측쌍이므로 전후 개선 비교에 사용하지 않는다.

실제 ERA5 archive/상층 sidecar/연결 GPU는 없었고 실제 장기 재학습은 실행하지 않았다.
모든 member 영상의 quiver는 고정된 Eulerian 격자의 방향/길이 표시다. FPS·화살표 배율이나 한 프레임으로 dynamics 개선을 주장하지 않는다.

## 산출물과 재현

- [실제 A/B/C loss 그래프](training.png), [같은 모델의 평가/집계 그래프](comparison.png)
- [합성 member0의12h 간격 영상](member-000-12h.gif), [모든 member6h tendency](member-tendencies-6h.png)
- [summary](summary.json), [validation](validation-process.json), `process-{A,B,C}.metrics.json`
- [전체 학습 단일 명령 블록](../../../flow-matching_moe/A_INFORMATION_TRAINING_MANUAL.md),
  [Mermaid](../../../struct-picture/16-separate-a-information-process.md)

```bash
python -m pip install -e '.[test,plots,io]'
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest -q
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python scripts/smoke_information_process.py \
  --output outputs/maintenance-reproduce --report outputs/maintenance-reproduce-report --only-process
```

수정 전/후 별도 smoke 폴더가 있을 때만 동일성을 감사한다:

```bash
python scripts/verify_a_information_maintenance.py \
  --before outputs/maintenance-before --after outputs/maintenance-after \
  --after-report outputs/maintenance-after-report --output outputs/maintenance-check.json
```

원래 실험·weights·예측은 덮어쓰지 않았다. 보고서의 synthetic 파일은 새 폴더로만 추가한다.
전체 학습의 모델 확대, loss 재설계, exact optimizer resume, ERA5 성능 개선은 이번 유지보수의 완료 항목이 아니다.
