# 작은 ERA5 다운로드와 A→B→C 학습 병행

`feature/a-manifold-information-process`용이다. 기존 **surface NPZ + schema**는 준비되어 있어야
한다. 이 기능은 추가 상층 자료의 대형 원본을 순차 다운로드/정리한다. A/B 분리, loss,
6h×20=120h recurrence, member identity는 유지한다.

## 바로 실행

CDS 계정·두 데이터셋 약관·서버의 `~/.cdsapirc`를 먼저 설정한다. 이 명령을 실행하면
실제 CDS 요청과 GPU 학습이 시작된다. 이미 설치한 CUDA PyTorch를 사용하는 환경에서 실행한다.
`ffmpeg`가 없으면 서버 패키지 관리자로 설치해야 MP4 출력까지 가능하다.

```bash
bash <<'BASH'
set -euo pipefail
TAG="$(date -u +%Y%m%dT%H%M%SZ)"
GIT_LFS_SKIP_SMUDGE=1 git clone --single-branch --branch feature/a-manifold-information-process \
  https://github.com/nayehyeon61-glitch/climate_diffusion.git "climate_stream_${TAG}"
cd "climate_stream_${TAG}"
python -m venv --system-site-packages .venv
source .venv/bin/activate
python -m pip install -e '.[test,plots,io]' cdsapi
python -c 'import torch; assert torch.cuda.is_available(), "CUDA PyTorch 환경을 확인하세요"'
command -v ffmpeg >/dev/null || { echo 'ffmpeg 설치 후 다시 실행하세요'; exit 1; }

export ARCHIVE=/workspace/data/era5-temporal-6h.npz
# 최초에는 새 경로. 중단한 다운로드는 같은 INFO를 그대로 재사용한다.
export INFO=/workspace/data/era5-extra-shards-v1
export RUN="/workspace/experiments/a-stream-${TAG}"
export DEVICE=cuda PROFILE=process
export CHUNK_DAYS=3 REGRID=linear
export M=4 TAU=4 BATCH=2 SEED=7 LR=0.001
export HISTORY_STRIDE=4 WINDOW_STRIDE=1 MAX_WINDOWS=0
export A_EPOCHS=60 CURRICULUM_INTERVAL=4 B_EPOCHS=30 C_EPOCHS=10
export B_MEMBER_WEIGHT=0.001 EVAL_CASES=0
export THROUGH=render

# 계획 출력에는 인증·다운로드·삭제가 없다.
python scripts/stream_era5_extra.py --archive "$ARCHIVE" --store "$INFO" \
  --regrid "$REGRID" --days-per-request "$CHUNK_DAYS"
# surface 검사 → CPU 다운로드/변환 → 준비된 자료로 GPU A/B → C → validation/동일 forecast 영상
bash scripts/run_streaming_a_information.sh
echo "결과: $RUN; 다운로드 로그: $RUN/download.log"
# validation으로 설정 확정 후에만 별도로 실행:
# bash scripts/run_a_information_120h.sh test
BASH
```

`linear`는 경도 주기경계를 포함한 공간 선형 보간이며 면적 보존 평균이 아니다.
기존 surface의 block mean을 동일하게 재현하려면 `REGRID=coarsen-match`를 선택한다.
그 결과 좌표가 archive와 다르면 중단한다. 준비 방식이 다른 경우 같은 INFO를 재사용하지 않는다.
격자는 archive에서 읽으므로 사용자 16×32를 정확히 따른다. 지위고도는 geopotential/9.80665(m),
u850/v850는 m/s, terrain_slope는 구면 거리로 계산한 무차원값이다.

## 병행하는 범위

1. 기존 surface의 시간·mask·split·120h 계약을 먼저 검사한다.
2. CPU 프로세스가 오래된 날짜부터 최대 `CHUNK_DAYS`일씩 요청한다. 월 경계에서 분리한다.
3. z850/z500/z250/u850/v850를 변환하고 terrain_height/slope를 붙여 작은 NPZ 조각으로 저장한다.
4. 시간·격자·단위·finite/observed mask·static terrain·체크섬 검증 후 commit receipt를 공개한다.
5. `--delete-raw`일 때 **그 조각에 사용한 원본만** 삭제한다. 작은 NPZ와 요청/체크섬 기록은 남긴다.
6. **train + expert_validation의 필요한 마지막 target까지** 준비되면 A/B 학습을 시작한다.
   이때 남은 calibration/validation/test 기간은 CPU 프로세스가 계속 준비한다.
7. 전체 자료가 준비되고 producer가 성공 종료한 뒤 C/validation/render로 진행한다.

**처음 3일만 받고 전체 archive 학습을 시작하는 방식은 아니다.** train-only state/tendency
통계를 고정하고 매 epoch 동일한 expert_validation으로 best를 고르기 위해 준비 장벽이 있다.
학습 대상 연도·split은 원래 archive에서 결정되며, 도착한 조각마다 통계를 다시 fit하거나
checkpoint/optimizer를 초기화하지 않는다. 별도 원격 컴퓨터를 생성하는 기능도 아니다.
같은 RunPod/서버에서 CPU 데이터 준비와 GPU 학습을 별도 프로세스로 병행한다.

63년 전체 자료라면 train prefix를 준비하는 시간도 길 수 있다. 이 구현은 **동시에 보관하는
원본 용량**과 전체 다운로드 완료까지의 GPU 대기를 줄인다. CDS 큐·전체 전송량 자체는 줄이지
않는다. 몇 년짜리 pilot을 하려면 surface archive도 별도 기간으로 만들며 split이 바뀜을 기록한다.

## 저장·시간·학습 계약

| 파일 | 역할 |
|---|---|
| `INFO/plan.json`, `metadata.json` | 불변 archive/schema/time/grid/단위/변환 계획 |
| `INFO/terrain.npz` | 한 번 변환한 static terrain과 slope |
| `INFO/chunks/000000.npz` + `.json` | 작은 데이터와 commit/checksum receipt |
| `INFO/raw/*.nc` | 현재 처리 중 원본. 검증 후에만 정리 |
| `INFO/raw/*.request.json` | 원본 삭제 후에도 남는 요청·체크섬 근거 |
| `INFO/complete.json` | 전체 조각 공개 완료 기록 |
| `RUN/readiness-AB.json`, `readiness-all.json` | 어느 시점까지 검사 후 학습을 시작했는지 |
| `RUN/raw-cleanup.json` | producer 종료 후 최종 검증·원본 정리 결과 |

Reader는 소수 조각만 메모리에 cache하고 필요한 slice를 연결한다. 120h window와 train
tendency 통계는 조각/월 경계를 그대로 넘으며 실제 6h 인접쌍을 누락하지 않는다. 추가 정보의
state/tendency 통계는 train의 고유 관측/쌍에서 면적 가중 float64 두 번 순회로 계산하고 고정한다.
정규화는 읽을 때 적용한다. 기존 surface NPZ 로딩과 A seal은 여전히 전체 train 배열을 사용하는
부분이 있으므로 **전체 학습 RAM/VRAM이 일정하다는 보장은 아니다.**

Inference conditioning은 origin의 추가 정보로 고정된다. 미래 extra는 감독 label로만 읽힌다.
msl 중복 채널을 추가하지 않으며 surface 네 변수, A auxiliary 확률 loss와 static L2, B/C 정책은
기존과 같다. 한 번 만든 forecast NPZ로 모든 member의 6h/12h 영상을 그린다.

Checkpoint는 기존 sidecar일 때 파일 SHA를 유지하고, shard일 때 불변 plan+metadata identity와
**실제로 소비한 조각 SHA 목록**을 저장한다. 뒤 기간을 새로 추가해도 앞서 쓴 데이터가 바뀌면
다음 stage가 거부한다. 기존 단일 NPZ 입력/기존 checkpoint는 계속 지원한다. 단일 NPZ로 학습한
parent를 값이 비슷하다는 이유로 새 shard 입력에 자동 연결하지 않는다.

## 중단, 재개, 오류

다운로드만 재개하거나 GPU 없이 전처리만 할 수 있다:

```bash
python scripts/stream_era5_extra.py --archive "$ARCHIVE" --store "$INFO" \
  --regrid "$REGRID" --days-per-request "$CHUNK_DAYS" --download --delete-raw
python scripts/stream_era5_extra.py --archive "$ARCHIVE" --store "$INFO" --check-ready all
python scripts/stream_era5_extra.py --archive "$ARCHIVE" --store "$INFO" --prune-verified-raw
```

- 완료·검증된 작은 조각은 재다운로드하지 않는다. 파일 공개 후 receipt 생성 전에 중단돼도
  원본을 삭제하지 않았으므로 작은 파일을 다시 검사해서 이어간다. 완료 store 재개에는 CDS 요청이 없다.
- 미완성 현재 요청은 다시 받는다. `--delete-raw`를 생략하면 원본을 보존한다.
- checksum/mask/units/time/grid 오류는 중단한다. 손상 데이터를 자동으로 덮어쓰거나 외부 경로를 삭제하지
  않는다. 손상 원인을 확인하고 새 store에서 재생성한다. 여러 producer가 같은 store를 쓰면 lock으로 거부한다.
- Ctrl-C/TERM은 runner의 producer도 종료한다. 검증된 조각은 남는다. `download.log`에서 인증·큐·용량 오류를 본다.
- 준비 대기는 기본 24h timeout이다. `WAIT_TIMEOUT_SECONDS`로 조절한다. timeout은 완료 의미가 아니다.
- **학습 optimizer/RNG exact resume는 여전히 지원하지 않는다.** A 중간 중단이면 같은 INFO와 **새 RUN**으로 A를
  다시 시작한다. 완료한 A를 사용해 B부터 진행하려면 동일 환경변수를 복원하고 기존 RUN에
  `START_STAGE=B THROUGH=render bash scripts/run_streaming_a_information.sh`를 실행한다.
  B 완료 후에는 `START_STAGE=C`, C 완료 후에는 `START_STAGE=validation`을 사용한다.
- `THROUGH=A`이면 A/audit까지, `THROUGH=B`이면 B까지 학습한다. producer는 나머지 다운로드를 끝낸 뒤 종료한다.
  audit를 직접 비교한 뒤 다음 stage로 진행하는 데 쓸 수 있다.
- full 명령의 epoch/LR는 기존 후보값이다. 짧은 pilot은 새 RUN에서
  `A_EPOCHS=6 CURRICULUM_INTERVAL=1 B_EPOCHS=1 C_EPOCHS=1 MAX_WINDOWS=4 EVAL_CASES=2`로 한다.
  process A는 여섯 curriculum 단계가 있으므로 1~2 epoch를 완료 profile로 사용하지 않는다.
- OOM은 batch>=2를 유지하면서 모델을 바꾸기 전에 학습/평가 M, tau, window budget을 별도 pilot에서 확인한다.
  다운로드를 줄이려면 CHUNK_DAYS를 줄인 **새 store**를 사용한다. 이는 요청 수/큐 대기를 늘릴 수 있다.
  기존 store의 계획을 중간에 바꾸지 않는다.

## 확인한 범위

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest -q
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python scripts/smoke_streaming_information.py \
  --output /workspace/experiments/streaming-synthetic-new
```

Mock CDS NetCDF를 실제 변환/검증/삭제하고, producer와 CPU 합성 학습을 병행하는 검사다.
실제 CDS 다운로드, 사용자의 ERA5 장기 재학습, 4090 VRAM·속도는 이 변경에서 실측하지 않았다.
결과 기록은 [streaming smoke 보고](../docs/results/streaming-information-smoke/README.md),
구조는 [Mermaid](../struct-picture/17-streaming-information-data.md)를 참고한다.
