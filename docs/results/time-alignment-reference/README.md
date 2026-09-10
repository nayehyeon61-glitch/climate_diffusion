# 시간 정합성 보정의 사용자 참고 출력

## 우선 참고할 실제 MP4 영상

후속으로 제공된 [test-case-compare.mp4 수신본](source-comparison.mp4)을 바이트 변경 없이
추가 보존했습니다. **이 MP4를 시간 변화 비교의 우선 참고 자료로 사용합니다.** 아래의 PNG
단일 프레임 제한은 이전 첨부에만 해당합니다.

- 실제 메타데이터: 1260×482, 30 frames, 2.5 fps, 재생시간12초.
- 첫 프레임 표제: origin `2018-11-16T06:00:00`, `+24h (1d)`.
- 마지막 프레임 표제: 같은 origin, `+720h (30d)`.
- 첫/마지막 화면에서 generated ensemble mean의 변화가 실제 ERA5보다 작아 보입니다.
  변화율·풍속 오차의 수치적 결론은 원본 배열과 plotting 코드로 검증해야 합니다.
- SHA-256: `38ede9812b1a4cd6dcf38d6f6aea4483673789ff1fae7d99e54e582f0bd431df`.

후속 작업은 **이 비교 영상 형식을 재현하는 시각화 시스템을 먼저 구현**하고, 실제 예측
시각과 벡터/상태 변화율을 검증·보정하는 순서입니다. 학습된 모델과 ERA5 배열을 읽어 같은
valid time의 패널을 생성하고, 공통 색상·quiver scale·단위·프레임 시간 표기를 사용해야 합니다.
Ensemble mean뿐 아니라 member별 시나리오와 spread도 볼 수 있게 하며, 인접 시점 변화량을
실제 delta-time으로 나눈 진단 및 temporal anomaly/lag 비교를 함께 출력합니다.

재생시간12초를 예보시간30일의 물리 velocity로 해석하거나, 화면을 빨리 재생하는 것을 모델의
시간 정합성 수정으로 취급하지 않습니다. 바람 u/v, 생성 flow velocity, 실제 상태 시간 미분을
구분해야 합니다. 이전 MoE(meta) 영상과 신규 manifold 모델의 결과도 구분합니다.

## 이전 단일 프레임 참고 이미지

2026-09-08 사용자가 제공한 `test-case-compare(1).gif`의 수신본을 바이트 변경 없이 보존합니다.
파일명은 gif였으나 실제 수신 파일은 **PNG, 1260×483, 1 frame**입니다. 따라서 이 파일만으로
시간 변화 속도나 GIF 재생 간격을 측정할 수 없습니다. 원본 이름과 혼동하지 않도록 확장자를
실제 형식에 맞췄습니다. SHA-256:
`22224936907adda20f93fbc7648e626fd2fe86852b814ed5ba791f49b63cbcf7`.

![사용자가 제공한 ERA5 비교 출력](source-comparison.png)

그림 표제: `Flow-Matching MoE (meta) vs actual ERA5 | real held-out test window |
+24h (1d) after 2018-11-16T06:00:00`. 왼쪽 generated ensemble mean, 오른쪽 actual ERA5,
색상 t2m(K), 화살표는 바람 벡터로 보이나 정확한 변수·배율·단위는 원본 plotting 코드로 검증해야
합니다. 표제는 **이전 MoE meta 경로**를 가리키며 신규 manifold smoke 결과가 아닙니다.
그림 표제의 real-data 표기는 사용자 산출물의 표제이며 이 환경에서 실제 원본 배열을 검증한
것은 아닙니다.

사용자 후속 요청은 현재 구조를 유지하며 **시간에 따른 출력 변화와 vector의 시간 비율**을
맞추는 보정입니다. 다음 작업에서는 다음을 분리해서 확인합니다.

1. Origin, history stride, target index, physical lead, valid time, forecast_step_hours 계약.
2. 생성 시간 tau의 FM velocity, 물리 시간의 기상장 변화율, u10/v10 풍속은 서로 다른 양이라는 점.
3. 양쪽 패널과 모든 시점의 같은 valid time, color limits, quiver scale 및 frame duration.
4. 실제 배열의 lead별 RMSE/CRPS, 바람 벡터 오차, 인접 시점 state 변화량, anomaly·phase/lag.
5. Leadwise marginal 학습이 시간 정합성에 주는 한계와 현재 모듈 안에서 가능한 학습/추론 보정.

GIF 한 프레임이나 ensemble mean의 평활화만으로 모델의 시간 scaling 오류를 단정하지 않습니다.
MP4는 확보했지만 checkpoint와 예측/ERA5 배열까지 확보한 것은 아닙니다. 배열이 없으면
부족한 증거를 명시하고 synthetic 시간 계약 검증을 수행합니다. 향후 수정에서도 기존 실험·checkpoint를 보존하고 feature branch를
사용하며 main에 병합하지 않습니다.
