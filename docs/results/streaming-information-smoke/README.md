# Streaming 추가 정보: 실행 검증

범위: mocked CDS와 **실제 CPU 합성 학습**. 실제 ERA5 다운로드/재학습, 기상 예측 개선,
4090 성능 시험이 아니다. 모델/loss 의미는 변경하지 않았다.

## 실행

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest -q
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python scripts/smoke_streaming_information.py \
  --output /workspace/experiments/streaming-synthetic-new
```

Python 3.12, torch 2.14.0+cpu. 합성320시점,4×4×8 surface,4차원 manifold.
3일 단위 요청을 월 경계에서 나눠28개 shard 생성. Native CDS0.25도 크기의 파일을
모의한 것이 아니므로 여기의 파일 용량/속도를 실제 CDS에 외삽하지 않는다.

## 실제 결과

| 검사 | 결과 |
|---|---|
| 새 코드와 기존 regression | 107개 통과 |
| A 시작 | 28개 중15개 준비, producer 실행 중 |
| B 시작 | 28개 중16개 준비, producer 실행 중 |
| C 시작 | 28개 준비, producer 성공 종료 |
| 학습 | A6/B1/C1 epoch, 각 train2 window, M2/tau1, full20 physical steps |
| checkpoint | A→B→C 및 최종 reload; stage별 소비 shard SHA 저장 |
| 영상 | 동일 forecast의 member0/1 각각6h·12h GIF, 합계4개 |
| 최종 원본 정리 | 검증된 converted shard28개 보존, owned raw NetCDF0개 |
| 전체 smoke 시간 | 약33.05초; 모의 요청 지연과 영상 출력 포함 |
| 학습 로그의 최대 RSS | 518,784KiB; 단일 학습 프로세스, GPU VRAM 미측정 |

초기 통합 검사에서 producer 종료 뒤 검증된 원본 일부가 남는 것을 관측하여, 최종 결과를
기록하기 전에 전체 shard를 다시 검증하는 **재실행 가능한 원본 정리 단계**도 연결했다.
삭제는 shard 검증/원본 receipt/원본 SHA/소유 디렉터리 확인 후에만 일어난다.
이 단계는 CDS를 호출하지 않으며 임의의 다른 파일을 지우지 않는다.

최종 validation 2 window: 정규화 RMSE1.5963, state CRPS1.2277, transition CRPS0.8357,
coverage80≈1.16%. 이는 연결 확인용 짧은 합성 실험으로, 확률 보정·ERA5 dynamics 목표를
달성한 결과가 아니다. 기존 학습과 성능 우열을 주장하지 않는다.

실행 기록(전체 summary/validation 및 주요 epoch loss 발췌): [summary.json](summary.json), [training.json](training.json),
[validation.json](validation.json). 영상/예측/weights는 smoke 실행 출력 폴더에 생성하며
기존 사용자 실험 파일을 변경하지 않는다.

## 회귀 검사의 내용

- 중단된 producer 재개, 이미 검증한 shard의 무다운로드 재사용.
- 공개된 NPZ와 receipt 사이의 중단 복구. receipt 없는 파일은 학습기에 보이지 않음.
- checksum 손상 시 원본 보존; 소유하지 않은 디렉터리/심볼릭 링크 삭제 거부; producer lock.
- time/mask/unit/static terrain 오류 거부, 조각 경계를 넘는120h labels.
- train 고유 관측/인접쌍 통계와 직접 계산 일치, held-out shard 추가 전후 통계 불변.
- A/B 준비 이전 누락 자료 fail-fast; checkpoint에 pinned된 조각 변경 거부.
- 실제 shell runner의 A→B→C 순서, producer 병행,6h/12h 렌더 호출, 기존 RUN 덮어쓰기 거부.
- 기존 separate A loss/backward, frozen B, recurrence, checkpoint, 전체 모델 regression.

## 남은 조건

실제 CDS 인증/약관과 ERA5 surface archive가 필요하다. CPU producer는 전체 native 요청
전송량을 줄이지 않는다. train/expert_validation prefix 완료 전에는 GPU 학습을 시작하지
않는다. optimizer/RNG exact resume는 기존 trainer와 동일하게 미지원이다.
실제 CUDA/CuDNN, 장기간 데이터 손상·파일시스템 장애·CDS 서비스 재시도는 이번 작은 검사의
실측 범위가 아니다. 기존 split/통계와 checkpoint provenance를 유지한 pilot부터 실행한다.
