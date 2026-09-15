# Joint A+B + Ensemble Loss V2

2026-09-15: [한국어 전체 실행 순서](../../../flow-matching_moe/JOINT_AB_TRAINING_MANUAL.md)와
[CLI/단위 검증 보고](manual-audit-2026-09-15.md)를 추가했습니다. 새 장기 학습 결과가 아닙니다.
현재 AB V2 tendency score의 물리 단위 환산 및 gradient logger 오류가 확인되어
장기 ERA5 재학습 전 별도 보수가 필요합니다.

Implementation branch: `feature/joint-ab-loss-v2`, based on optimization
commit `4dd9bf4051aa8c277099c980313bdde91d48f854`.

This directory documents code validation only until a real ERA5 archive and
GPU run are attached. No synthetic smoke result is evidence of ERA5 skill or
spread improvement. The earlier tiny synthetic comparison
(joint V2 RMSE 1.23425, CRPS .87361, spread .13096, coverage .06592) did not
demonstrate ensemble expansion, and is not promoted as a new result.

The A audit on 640 unique observed 6 h pairs remains the motivation, not a
proof of representational impossibility: AE/drift amplitude ratios were
msl .534/.335, t2m .208/.094, u10 .573/.361 and v10 .497/.334, while drift
tendency error exceeded the zero-tendency baseline for all four variables.
The separate ERA5 B-to-C audit improved RMSE/CRPS/Energy and coverage slightly
while spread decreased; therefore C alone is not identified as the collapse
cause.

Implemented checks live in `tests/test_joint_ab.py`. Expected run artifacts
are checkpoint + metadata/metrics/manifest, validation JSON, all-member 6 h
and 12 h prefix renders, and the loss chart. Runtime, RSS and CUDA peak memory
must be reported from the actual machine; GPU measurements are not inferred
from CPU smoke.
