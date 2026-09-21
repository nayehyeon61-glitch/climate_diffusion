# 추가 ERA5 전처리와 분리형 A/B 학습 병행

```mermaid
flowchart TB
    S["기존 surface archive와 고정 split"] --> P["작은 기간별 CDS 요청 계획"]
    P --> R["현재 raw NetCDF 다운로드"]
    R --> V["공간 변환과 시간·단위·mask 검사"]
    V --> C["작은 shard 공개와 checksum receipt"]
    C --> D["검증된 해당 원본만 삭제"]
    D --> R
    C --> G{"train·expert validation 준비?"}
    G -->|"준비 완료"| A["고정 train 통계로 A 학습·seal"]
    A --> B["A 동결 후 B 학습"]
    C --> F{"전체 기간 준비?"}
    B --> J["두 조건 충족 후 C 보정"]
    F -->|"producer 성공 종료"| J
    J --> E["validation과 단일 forecast 저장"]
    E --> M["같은 member를 6h·12h 영상으로 출력"]
```

작은 shard는 모든 epoch에서 재사용한다. 다운로드 중 train 통계를 다시 fit하지 않는다.
A/B가 사용하는 prefix를 전부 준비한 뒤 나머지 held-out 기간 전처리와 학습을 병행한다.
6시간×20 physical step, origin-fixed conditioning, A/B/C gradient/freeze와 loss는 그대로다.
실제 시간 부족/누락 셀을 보간해 채우지 않으며 시점별 shard 경계가 trajectory를 끊지 않는다.
