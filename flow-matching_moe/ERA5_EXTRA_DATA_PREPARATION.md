# 추가 ERA5 입력 준비: Git 배포, 인증, 용량 확인

`INFO_FIELDS` 파일이 없는 경우 학습 전에 수행하는 단계다. ZIP/sandbox 링크 없이
이 브랜치의 `scripts/prepare_era5_extra.py`를 직접 실행한다. 모델/loss/weights는 변경하지 않는다.

## 지금 바로 실행할 명령

해당 feature 브랜치의 기존 저장소와 가상환경에서 실행한다. 사용자 변경 때문에
fast-forward가 실패하면 reset/force하지 말고 별도 clone을 사용한다.

```bash
git switch feature/a-manifold-information-process
git pull --ff-only origin feature/a-manifold-information-process
python -m pip install 'cdsapi>=0.7.7' numpy pandas xarray scipy netCDF4

export ARCHIVE=/workspace/data/era5-temporal-6h.npz

# 1. 전체 기간의 요청 수와 용량만 확인. 인증/다운로드 없이 동작한다.
python scripts/prepare_era5_extra.py --archive "$ARCHIVE" --regrid linear

# 2. CDS 인증과 약관 동의 후 첫 2일만 다운로드/정렬 시험.
python scripts/prepare_era5_extra.py --archive "$ARCHIVE" --regrid linear \
  --probe-days 2 --download --output /workspace/data/era5-extra-probe.nc
```

위 두 번째 명령은 사용자가 실제로 실행할 때 CDS 요청을 보낸다.
`--download`를 빼면 probe 계획만 확인한다. 시험 파일은 전체 archive의 학습 입력이 아니다.
`coverage=probe_subset_not_for_full_training`이 기록되며 `physical_information`에 전체 archive와
함께 전달하면 시간 불일치로 거부되는 것이 정상이다. 이름만 aligned로 바꿔 우회하지 않는다.

## 인증

같은 서버 사용자로 실행할 때 읽을 수 있는 `~/.cdsapirc`에 본인의 설정을 저장한다.

```yaml
url: https://cds.climate.copernicus.eu/api
key: <PERSONAL-ACCESS-TOKEN>
```

- 계정/토큰: https://cds.climate.copernicus.eu/how-to-api
- pressure levels 약관: https://cds.climate.copernicus.eu/datasets/reanalysis-era5-pressure-levels?tab=download
- single levels 약관: https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels?tab=download

CDS 공식 안내에 따라 데이터셋 약관 동의는 웹에서 사용자가 수행한다. 같은 token으로 두
데이터셋에 접근하며 비밀번호를 key로 넣지 않는다. 토큰을 Git/로그/대화에 올리지 않는다.
설정 파일 유무만으로 서버 인증/약관 통과를 보장하지 않으므로 작은 실제 요청으로 확인한다.

## 63년 archive의 크기

사용자가 보고한 1959-01-01 00UTC부터 2021-12-31 18UTC까지 exact6h이면 92,044시점이다.
5개 동적 필드, float32로 단순 계산하면 다음과 같다.

| 범위 | 계산 | 비압축 크기 |
|---|---|---:|
| 원본 0.25도 | 92044 × 5 × 721 × 1440 × 4 bytes | 약1.91 TB / 1.74 TiB |
| 목표16×32 | 92044 × 5 × 16 × 32 × 4 bytes | 약0.943 GB |

이는 **파일 다운로드 실측값이 아니다.** 압축률/요청 경계의 여분 시각/메타데이터/임시
공간이 반영되지 않았다. 다운로드 시간도 CDS 큐와 네트워크에 따라 달라 확정할 수 없다.
이 도구는 native 자료를 받은 뒤 로컬에서 정렬한다. 작은 최종 파일이 원본 전송량을
줄이는 것은 아니다. raw cache를 보존하므로 디스크 규모도 함께 검토해야 한다.

첫 2일 probe 성공 후, 처음 학습할 기간과 디스크 예산을 정한다. 몇 년짜리 pilot을
선택하면 **surface NPZ와 schema도 별도 기간 archive로 준비**해야 한다. 63년 surface에
짧은 추가 정보만 연결할 수 없으며 새 archive는 새 split/statistics를 만든다.
기존 63년 실험과 같은 split을 유지한 성능 비교라고 보고하지 않는다.

CDS 서버 측 재격자화나 다른 저장소의 coarse 자료로 전송량을 줄이는 방법은 별도 검증 대상이다.
이 도구가 이를 구현했다고 주장하거나 검증되지 않은 `grid` API 옵션을 사용하지 않는다.

## 자료·격자 계약

- Pressure levels: geopotential at 850/500/250 hPa, u/v at850hPa.
- Single levels: 첫 시각의 surface geopotential을 static terrain으로 사용.
- Geopotential은9.80665로 나눠 m로 변환. 출력의 z850 등도 m 단위의 동적 지위고도다.
- terrain_height는 `[lat,lon]`이며 상층 고도와 다르다. terrain_slope는 기존 준비기가 계산한다.
- 추가 선택변수 t850/t500/u500/v500/sst/q850는 이 다운로드 도구에 포함하지 않는다.
- 동적 출력은 `[time,lat,lon]`, 시각은 원래 archive에서 exact selection. 시간 보간 없음.
- `--regrid linear`: 경도 주기경계를 고려한 bilinear 공간 보간. 보존적 면적 평균은 아니다.
  기존 surface가 block mean이면 공간적 평균 범위가 다를 수 있으므로 결과 해석에 기록한다.
- `--regrid coarsen-match`: 기존 `_coarsen_global_fields`와 같은 block mean을 시도하며
  결과 좌표가 목표 archive와 일치할 때만 성공. 불일치를 좌표 덮어쓰기로 숨기지 않는다.
- ascending latitude, exact pole 제외, 균일한 전지구 주기 경도와 격자/단위/mask 계약을 확인한다.
- 누락 시각/셀/알 수 없는 units/여러 expver 또는 ensemble 자료는 명시적으로 중단한다.

기존 `physical_information.py` 자체는 공간 보간하지 않는다. 여기의 **독립 준비 단계**에서
명시적으로 정렬한 뒤 기존 strict validator를 통과시킨다.

## 범위를 결정한 후 전체 생성과 학습 연결

아래는 전체 archive 다운로드 명령이다. 소규모 probe와 구별해 실행한다.

```bash
export INFO_FIELDS=/workspace/data/era5-extra-aligned.nc
export INFO=/workspace/data/era5-information-v1.npz

python scripts/prepare_era5_extra.py --archive "$ARCHIVE" --regrid linear \
  --download --output "$INFO_FIELDS"

python -m climate_diffusion.physical_information \
  --archive "$ARCHIVE" --fields "$INFO_FIELDS" --output "$INFO"
```

출력/provenance는 덮어쓰지 않는다. 성공한 요청의 raw cache/checksum/request를 재사용한다.
기본 한 달 안에서3일 단위로 geopotential/바람을 나눠 요청하며 `--days-per-request 1`로
단일 요청 크기를 줄일 수 있다. 요청 수를 늘리므로 빨라진다는 의미는 아니다.
중단 시 완료한 raw 다운로드는 재사용하지만 정렬 출력은 다시 조립한다.
인증·데이터 준비 완료 후 [전체 A→B→C 매뉴얼](A_INFORMATION_TRAINING_MANUAL.md)의
preflight부터 진행한다. 학습 도중 데이터 파일을 바꾸거나 과거 checkpoint에 다른 sidecar를 넣지 않는다.

## 실제 검증 범위

```bash
python -m pip install pytest
python -m pytest -q tests/test_prepare_era5_extra.py
```

데이터 도구 전용12개 검사: gap-free6h, 월 경계, 날짜 누락, pressure level/height 단위,
경도 주기 보간, missing cells, block-mean 정합, mock CDS 전체 조립/cache/checksum,
92,044시점 계획 계산, probe 표기/기존 archive 보존.
실제 CDS 토큰이나 사용자 ERA5 archive를 사용하지 않았고 다운로드/ERA5 재학습은 미실행이다.
학습 모델 파일과 기존 결과/weights는 변경하지 않았다.
