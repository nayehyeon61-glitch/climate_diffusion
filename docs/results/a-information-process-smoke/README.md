# 분리형 A information/process: 실행 결과와 한계

기반 `feature/physical-recurrent-optimization` **4dd9bf4051aa8c277099c980313bdde91d48f854**에서 구현했다.
기존 `results/physical-recurrent-run-001`의 사용자 GIF는 f897046 실제 ERA5 run이며 아래 수치와 비교할 수 없다.
results runner의 B member MSE .001/C0, temporal ramp B5/C3를 유지했다. joint-ab/main/results는 변경하지 않았다.

## 추가 입력이 실제 존재하는가?

**enriched 모드에서는 존재하고 A의 encoding과 stochastic loss에 연결된다.**

| 역할 | 변수 |
|---|---|
| 기존 surface 입력/출력 | msl, t2m, u10, v10 |
| 별도 필수 동적 conditioning | Z850, Z500, Z250, U850, V850 |
| 별도 정적 conditioning | terrain height, 구면 거리로 계산한 terrain slope |
| 명시적 옵션 | T850/T500, U500/V500, SST, q850 |

msl/MSLP는 중복 채널로 넣지 않는다. 필수 enriched 정보가 없으면 fail-fast하며 surface-only 비교 모드는 구별한다.
현재 실제 ERA5 추가 입력 파일은 제공되지 않았다. 이번 sidecar는 상층 변수를 surface toy wave로부터 만든
**합성 입력**이다. 실제 상층 관측이나 독립된 추가 정보의 효과를 입증한 실험은 아니다.

A 마지막 epoch 첫 batch에서 information encoder의 raw gradient norm은 stateCRPS **.06237**,
transitionCRPS **.02030**, pathEnergy **.07565**였다. 이름만 있는 입력 경로가 아님을 보여준다.
encoder의 static L2 raw norm .32195, coefficient .05 반영 norm 약 .01610;
reconstruction raw/weighted norm .08111. 한 batch 관측이며 일반적인 gradient 지배 결론은 아니다.

## 실행한 학습과 비교

시간의존 toy 320개 관측, 6h, surface4×4×8, manifold4, K2, M4, tau midpoint4,
A6/B2/C2epoch, epoch당4window, history6/stride1, full20 physical step의 BPTT.
split/data seed19, trainingseed7, validationseed83를 고정했다. train 통계는 고유113개 관측/112쌍에서 fit했다.
test tuning은 하지 않았다. 평가2origin은 겹치는 미래 구간을 포함하여 독립 표본2개라는 의미도 아니다.

| 새 wrapper 내부 A ablation → 동일 B/C | normalized RMSE ↓ | state fair CRPS ↓ | transition fair CRPS ↓ | path fair Energy ↓ | 80% coverage |
|---|---:|---:|---:|---:|---:|
| surface baseline | 1.59986 | 1.22890 | .82450 | 1.79985 | 2.65% |
| decoded-dynamics 추가 | 1.59890 | 1.22683 | .82425 | 1.79743 | 2.77% |
| physical-information 추가 | 1.59138 | 1.15425 | .79736 | 1.70232 | 5.98% |
| information + stochastic process/path | 1.59267 | 1.15507 | .79732 | 1.70262 | 5.87% |
| 마지막 모델 drift-only | 1.61943 | 1.30157 | .85644 | 1.90837 | 0% |

Persistence RMSE는1.61181이다. **process/path 추가가 information-only보다 RMSE/CRPS를 개선하지 않았다.**
비교 baseline은 새 wrapper의 controlled ablation이며 원래 legacy ABC를 동일 예산으로 완전히 재현한 결과는 아니다.
공유 core 초기화는 같은 seed지만 info MLP 유무와 후속 random 초기화까지 전부 동일한 architecture는 아니다.
원래 legacy trainer 직접 대조, 다중 seed/계절, 실제 ERA5는 미실행이다.

Process 모델 best epoch는 A6/B2/C1. A는 curriculum phase6 이후에만 best 후보가 된다.
Ensemble spread는 .24807, mean skill error는1.59267, coverage는5.87%로 **심하게 underdispersed**하다.
따라서 기능 연결은 검증했지만 ensemble 보정/기상 dynamics 문제를 해결했다고 보고하지 않는다.
분산만 확대하는 보상은 넣지 않았다. A aux sampler는 B에서 교체되므로 A 개선이 B에 전달되는지도 후속 검증 대상이다.

## A 자체 감사

[a-audit.json](a-audit.json)은 validation20개 고유6h쌍의 관측 양 endpoint AE와 finite-step drift를 검사한다.

| 변수 | AE 변화량 진폭비 | drift 변화량 진폭비 |
|---|---:|---:|
| msl | .04067 | .03523 |
| t2m | .03443 | .03453 |
| u10 | .04001 | .03442 |
| v10 | .03879 | .03343 |

이 짧은 smoke의 A 변화량은 여전히 약하다. drift RMSE는4변수 모두 zero-tendency보다 약간 크다.
Tangent oracle는 decoder Jacobian의 방향 표현 검사지 실제 예측 성공이 아니다. 이 모델을 ERA5 운영용 best A라고 사용하면 안 된다.
학습 길이/데이터·loss 균형과 표현 제약을 분리하는 추가 실험이 필요하다. 진폭비만1에 맞추는 것이 목표는 아니다.

## 산출물

- [비교 그림](comparison.png), [A/B/C loss](training.png), [A gradient](gradient.png), [생성 경로 routing](routing.png)
- [동일 forecast의 모든 member 12h GIF](members-12h/): member000–003, 10개 미래시점/120h
- [6h member 진단](members-6h/): 20개 미래시점/120h, fixed-scale t2m+wind
- [모든 수치 summary](summary.json), `process-{A,B,C}.metrics.json`, `.metadata.json`, `.manifest.json`
- [상세 구조](../../../struct-picture/16-separate-a-information-process.md), [실행 매뉴얼](../../../flow-matching_moe/A_INFORMATION_TRAINING_MANUAL.md)

GIF는 SYNTHETIC truth 표기를 사용한다. 평균 대신 모든 member를 동일 NPZ에서 렌더했고 재추론하지 않았다.
원본 weights/archive는 `outputs/information-smoke-001`에 별도 보존하며 보고서의 manifest는 그 파일의 SHA다.
Git에는 기존 사용자 weights를 대체하지 않고 새 코드/보고서/그림만 추가한다. synthetic dataset/weights는 스크립트로 재현한다.

## 비용 및 재현 범위

최초4profile 학습·평가·렌더 약139.77초, CPU1thread. 마지막 profile A3.59/B8.49/C17.99초;
프로세스 누적 peak RSS491848KiB(약480MiB). 이는 toy4×4×8/r4의 실측이며 4090/ERA5 비용으로 환산할 수 없다.
실제 GPU VRAM/장기 ERA5 재학습/추가 ERA5 변수 수집은 미실행이다.

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python scripts/smoke_information_process.py \
  --output outputs/a-information-reproduce --report outputs/a-information-report
python scripts/audit_information_process.py --checkpoint outputs/a-information-reproduce/process-A.pt \
  --archive outputs/a-information-reproduce/synthetic-states.npz \
  --information outputs/a-information-reproduce/information.npz \
  --output outputs/a-information-report/a-audit.json --max-pairs 20
python -m climate_diffusion.information_forecast --checkpoint outputs/a-information-reproduce/process-C.pt \
  --archive outputs/a-information-reproduce/synthetic-states.npz \
  --information outputs/a-information-reproduce/information.npz \
  --output outputs/a-information-report/generated-routing.json --max-cases 2
python scripts/visualize_information_process.py --report outputs/a-information-report
```

최초 실행 뒤 수정은 단위/schema 검사, gradient 입력 검사, provenance/CLI guard, 생성 routing 진단과 문서 보완이다.
최종 회귀검사 **75개 통과**, Mermaid3개 parse, CLI4개 help, Bash/compile 확인을 완료했다.
최종 코드 A6/B2/C2를 새 폴더에서 다시 실행해 동일한 validation aggregate를 얻었다(렌더 제외30.47초).
B에서 A의77개 tensor가 bitwise 유지되고36개 B trainable tensor가 바뀐 것을 직접 확인했다.
4member의6h MP4는20frame, 12h GIF는10frame이며 배열/valid time이 정확한 subset이었다.
자세한 범위는 [verification.json](verification.json)에 기록했다.
