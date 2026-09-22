# A64 / B512 확장 모델 통합 검증

기반 `efdf085`에서 차원 설정을 확장한 `feature/a64-b512-expanded`의 검증 기록이다.
실제 기상 예측력 검증이 아니라 구현·수치 연결 확인이다.

## 실행

```bash
PYTHONPATH=src python scripts/smoke_hybrid_pinn.py --expanded --output /tmp/a64-b512-smoke-new
PYTHONPATH=src python -m pytest tests/test_manifold_moe.py tests/test_information_process.py tests/test_pinn_training.py tests/test_recurrent_flow.py tests/test_expanded_latent.py -q
```

관련 기존 테스트 47개와 새 확장 테스트 5개, 총 52개를 통과했다.
실제 확장 차원으로 A7/B1/C1 합성 학습 후 120시간(6시간 × 20) 앙상블 평가를 완료했다.
실행 환경은 PyTorch 2.14.0 CPU, 1 thread이며 CUDA 학습은 수행하지 않았다.

| 항목 | 값 |
|---|---|
| 표면 격자 | 4변수 × 4위도 × 8경도, `state_dim=128` |
| A manifold / B expert / hidden | 64 / 512 / 512 |
| 전문가 수 / ensemble 크기 | 4 / 2 |
| 전체 모델 parameter 수 | 5,302,116 (이 합성 입력 크기에서만 해당) |
| 합성 snapshot / epoch당 학습 window | 320 / 2 |
| 평가 초기시각 수 | 2 |
| 통합 smoke 소요 시간 | 약 37.4초 |

확인한 사항:

- A의 surface/information encoder, drift, PINN, auxiliary sampler의 차원·gradient 연결.
- B 전문가 내부 code512와 전체 상태 후보 출력, Jacobian `D×64`, metric `64×64`.
- 직접 solve와 cached Cholesky 투영의 수치 일치.
- B/C가 새 A의 차원을 정확히 상속하고 B에서 A가 유지되는지 확인.
- C에서 기존 정책대로 A/B 일부가 갱신되고, B/C에서 PINN과 A 보조 sampler는 동결.
- 새64 및 명시적 기존16 checkpoint의 재로드 후 예측 일치.

원시 요약은 [summary.json](summary.json)에 저장했다. 이 최소 학습의 정규화 RMSE는
1.6573, persistence는 1.6118로 모델이 더 나쁘다. 작은 합성 자료·두 초기시각·두 member의
기능 확인 결과이며, 확대 전후의 예측력 또는 확률 보정 성능 비교로 해석하지 않는다.
실제 ERA5의 동일 분할·학습 조건 비교는 아직 수행하지 않았다.
