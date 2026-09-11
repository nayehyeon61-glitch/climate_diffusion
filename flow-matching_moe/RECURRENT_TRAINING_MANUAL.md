# 물리시간 재귀 Flow: 전체 학습 실행 매뉴얼

대상 브랜치: `feature/physical-time-recurrent-flow`. 기존 full-state MoE·PI manifold·ensemble은 유지하고 **물리시간 연결/잔차 FM target**을 변경했습니다. 새로운 대형 backbone이나 expert 수 확대는 없습니다.

현재 실제 구현은 다음과 같습니다.

1. A의 physical drift를 추론에 연결합니다.
2. 동일 member의 이전 q를 다음6h step에 전달합니다.
3. 각 physical step에서 residual FM을 적분한 **샘플**을 drift와 결합합니다.
4. 생성120h trajectory에 joint Energy, mean/member tendency, wind 보조 loss를 적용합니다.

이 변경은 구현·합성 smoke를 통과한 실험 경로이지, 실제 ERA5 dynamics 해결을 보증한 모델이 아닙니다. [실행 결과와 미달 항목](../docs/results/recurrent-smoke-final/README.md)을 먼저 확인하세요.

## 1. 두 시간축·residual 수학 계약

정규화 state를 x, encoder 출력을 z, 표준화 잠재좌표를 q라 두면:

```math
q=(E(x)-\mu_z)/\sigma_z,\quad \bar D(q)=D(\mu_z+\sigma_zq),\quad
b(q)=f_{\mathrm{drift}}(\mu_z+\sigma_z q)/\sigma_z.
```

A는 `z_next ≈ z + step_hours/24 × latent_drift(z)`로 학습하므로 drift의 단위는 **z/day**, b는 **q/day**입니다. 현재 code의 day를 hour로 오해하면24배 오류입니다.

```math
r^*={q_{j+1}^*-q_j^*\over\Delta t_{hours}/24}-b(q_j^*),\qquad
r_\tau=(1-\tau)\sigma_\epsilon\epsilon+\tau r^*,\quad
u_\tau=r^*-\sigma_\epsilon\epsilon.
```

`r*`, teacher-forced physical q, target encoder/drift 경로는 detach합니다. 미래 관측은 이 학습쌍과 loss에서만 사용합니다. history context는 origin의 관측으로 만들고 고정하며, 새 network 없이 현재 q와 physical clock을 추가 condition으로 사용합니다.

각 expert의 full-state FM transport 후보는 현재 **physical qj**의 Jacobian으로 투영한 뒤 gate로 합칩니다. residual r_tau를 물리 state인 것처럼 decode하거나 거기서 physical chart를 찾지 않습니다.

```math
{dr_\tau^{(m)}\over d\tau}=\sum_k\pi_k(q_j^{(m)},h,\tau,t_j)a_k(r_\tau^{(m)},q_j^{(m)},h,\tau,t_j),
\qquad
q_{j+1}^{(m)}=q_j^{(m)}+{\Delta t_{hours}\over24}\left[b(q_j^{(m)})+r_1^{(m)}\right].
```

안쪽은 midpoint **생성 τ ODE**, 바깥은 archive cadence의 **Euler physical integrator**입니다. `dr/dτ`를 b에 바로 더하지 않습니다. `--integration-steps`는 안쪽 τ 해상도이며 물리6h를 바꾸는 flag가 아닙니다. 12h 영상도 내부6h 재귀를 수행한 뒤 subsample합니다.

모든 member는 관측 q0에서 출발하며 독립 base epsilon으로 갈라집니다. 같은 member는20개 step에서 epsilon을 유지하고 step innovation은 없습니다. 공유 noise만으로 올바른 temporal joint law를 보증하지는 않습니다. 긴 Euler rollout의 누적오차/노출 편향은 남는 위험입니다.

출력은 `x0 + Dbar(qj) - Dbar(q0)`입니다. 고정 origin-offset을 더해 AE reconstruction의 초기 jump를 제거합니다. offset은 q의 recurrence를 끊지 않으며 decoder Jacobian도 바꾸지 않습니다. 그러나 이는 origin-anchored translated chart이지 원래 decoder manifold의 물리 보존 보증은 아닙니다. 개선 비교에서 이 origin 보정의 효과와 실제 dynamics 학습을 분리하세요.

## 2. 새 loss와 stage 경계

기존 FM/PI/지역 responsibility/균형/diversity/투영/ensemble score는 유지합니다. 기존 `loss_delta`는 train tendency scale과 실제 dt로 표준화한 ensemble-mean 차분 MSE입니다. 신규 항은 `--delta-member-weight`이며 default0, 아래 profile은 작은 시작값0.001입니다.

```math
L_{member}=\mathbb E_j {1\over M}\sum_m\|d_j^{(m)}-d_j^*\|_W^2
=L_{mean}+\mathbb E_j\operatorname{Var}_m(d_j^{(m)})_W.
```

따라서 member MSE를 크게 주면 ensemble variance를 벌점으로 누릅니다. raw `loss_delta_member`, `loss_delta`, `delta_member_variance_penalty`를 별도로 기록하며 동일 warmup ramp를 적용합니다. 0-weight ablation과 CRPS/coverage/spread를 함께 확인하세요. delta와 tendency를 서로 별개 독립 loss처럼 중복 가중하지 않습니다.

`loss_trajectory`는 모든 endpoint와 increment를 합친 **fair Energy**입니다. `trajectory_edges=0`이면 origin 포함21state/20edge 전체에 실제 loss와 BPTT를 적용합니다. `>0`은 부분 구간 score지만 그 구간까지 origin부터 재귀 적분합니다. **detach/truncated BPTT 구현이 아니므로 sub-block만 줄여도 메모리가 크게 줄어든다고 가정하지 마세요.**

| 단계 | 학습 데이터 | parameters/선택 |
|---|---|---|
| A | train의 인접 관측 | encoder/decoder/drift 새 초기화, expert_validation best seal |
| B | train | A parameters 동결, decoder 입력 미분 유지, experts/gate/history 학습 |
| C | calibration | B 재로드, 작은 manifold LR와 reference anchor, validation best 선택 |

미래 delta를 inference router나 history에 넣지 않습니다. 기존 timestamp/observed pair fail-fast, train-only state/dynamics statistics, 변수 scale floor, area weighting, five-way future-target purge를 재사용합니다. A/C batch1은 여전히 거부합니다. calm 방향 mask는 truth 기반이고 u/v output은 그대로입니다.

## 3. 설치와 합성 사전 검사

```bash
git clone --single-branch --branch feature/physical-time-recurrent-flow \
  https://github.com/nayehyeon61-glitch/climate_diffusion.git climate_diffusion_recurrent
cd climate_diffusion_recurrent
python -m venv --system-site-packages .venv
source .venv/bin/activate
python -m pip install -e '.[test,plots,io]'
python -c 'import torch; print(torch.__version__, torch.cuda.is_available())'
ffmpeg -version
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest -q
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python scripts/smoke_recurrent_flow.py \
  --work-dir outputs/my-recurrent-smoke-001 --report-dir outputs/my-recurrent-report-001
python scripts/visualize_recurrent_flow.py --report-dir outputs/my-recurrent-report-001
```

Smoke는 두 A를 동일 seed로 새로 학습하여 weights가 정확히 같은지 검증하고, baseline/recurrent B/C를 따로 학습합니다. 기존 실험의 trained checkpoint를 from-scratch라고 재사용하지 않습니다. 원본 결과/새 출력 폴더를 덮어쓰지 않으므로 재실행 시 경로를 바꾸세요. ffmpeg 없는 환경에서는 MP4를 생성할 수 없으므로 설치된 이미지에서 실행하세요.

## 4. 실제 ERA5/RunPod: 경로와 profile

이미 확보한 archive 또는 NetCDF/Zarr를 사용합니다. GPU/데이터 유료 자원 생성은 자동화하지 않습니다. 기존 run-001의 파일은 보존하고 **새 폴더**를 지정합니다.

```bash
export TEMPORAL_ARCHIVE=/workspace/data/era5-temporal-6h.npz
# archive가 없을 때만 기존 NetCDF/Zarr에서 준비:
export TEMPORAL_FIELDS=/workspace/data/era5_wb2_6h_full.nc
export TEMPORAL_RUN=/workspace/experiments/physical-recurrent-run-001
export TEMPORAL_DEVICE=cuda
export TEMPORAL_BATCH=2
export TEMPORAL_MEMBERS=4
export TEMPORAL_TAU_STEPS=4
export TEMPORAL_EDGES=0
bash scripts/run_recurrent_120h.sh prepare
```

prepare는 새 run 경로와 code commit/environment를 기록하고 preflight를 수행합니다. 입력·단위·시간·mask 오류를 고친 뒤 다음 단계로 넘어갑니다. grid는 schema가 기준입니다. 기존 run-001 metadata는 실제 **4×16×32=2048**이며 README의18×36은 pooling 요청 크기였습니다.

처음에는 별도 pilot 폴더에서 `TEMPORAL_A_EPOCHS=1`, `TEMPORAL_B_EPOCHS=1`, `TEMPORAL_C_EPOCHS=1`로 아래 전 단계를 한 번 확인하세요. pilot의 weights를 전체 from-scratch 학습으로 이어 부르지 말고 새 full run 폴더에서 기본 epoch로 다시 시작합니다.

## 5. A → 확인 → B → 확인 → C

```bash
bash scripts/run_recurrent_120h.sh A
bash scripts/run_recurrent_120h.sh B
bash scripts/run_recurrent_120h.sh C
```

한 줄씩 실행하고 확인합니다. A: `preflight-a-verified.json`, best epoch, 변수별 reconstruction/latent dynamics를 확인합니다. B: frozen A SHA/weights와 `loss_trajectory`, `loss_delta_member`, 변수별 temporal gradient가 finite/nonzero인지 확인합니다. C: PI/anchor 유지, best selection과 spread를 봅니다. 마지막 epoch와 선택된 best는 다릅니다.

`scripts/run_recurrent_120h.sh`에 실제 CLI가 모두 들어 있습니다. A에는 `--forecast-dynamics recurrent_residual --horizon-steps 20`을 지정하고 B/C는 이전 checkpoint config를 계승합니다. defaults는 A50/B40/C10, B/C baseLR0.001, C joint factor0.1와 manifold factor0.1, weight decay0.0001입니다. **최적 계수·epoch라는 주장은 아닙니다.**

각 단계는 새 optimizer를 만들며 `.pt/.metadata.json/.metrics.json/.manifest.json`에 best/통계/설정/부모 SHA를 기록합니다. optimizer 중단점의 bitwise resume은 제공하지 않습니다. code commit이 실험 중 바뀌면 runner가 중단합니다. full-window 경로가 오래 걸리면 step time·peak VRAM을 측정한 후 새 pilot 설정으로 줄이세요.

## 6. Validation·같은 forecast의 모든 member

```bash
bash scripts/run_recurrent_120h.sh validation
bash scripts/run_recurrent_120h.sh render
```

validation32cases는 전체 validation이 아니라 초기 표본 평가입니다. routing 진단은4origins×4members의 실제20-step 경로입니다. forecast를 한 번 저장하여6h MP4(20frame),12h GIF(10frame)에서 동일 member를 유지합니다. mean은 기본 출력이 아닙니다.

`--diagnostic-views`는 fixed quiver에 더해 adaptive quiver GIF와 wind-magnitude/t2m-tendency GIF를 추가합니다. adaptive는 한 frame의 양쪽에 같은 scale을 쓰지만 시간별 scale이 바뀌므로 실제 magnitude 비교에는 fixed 출력과 JSON을 사용합니다. 화살표 위치는 Eulerian 격자에 고정됩니다.

진단 JSON은 raw/projected **FM transport**와 샘플된 residual의 **physical q/hour**를 분리합니다. full-state physical component는 J(q)×velocity×state_scale로 변수별 단위를 복원한 국소 derivative이며, 유한6h endpoint 차분과 비선형성 때문에 정확히 같지는 않습니다. 실제 tendency 판정은 `member-NNN.json`을 우선합니다. 작은 true tendency의 ratio는 null/valid count를 확인하세요.

## 7. 설정 고정 후 test, 비교 실험

```bash
bash scripts/run_recurrent_120h.sh test
```

test로 loss coefficient를 튜닝하지 않습니다. 비교는 동일 archive/split/seed/member 수/τ steps에서 기존 baseline, drift-only, recurrent residual로 수행합니다. 직접 평가 시 `evaluate-climate-flow --moe-mode drift_only` 또는 `--moe-mode residual_only`를 지정할 수 있습니다. `uniform`, `expert:0`도 residual expert ablation에 사용됩니다.

drift-only는 zero residual입니다. **expert parameter를0으로 해도 FM residual은 source noise가 남아0이 아닙니다.** C drift-only는 C에서 공동 보정된 drift이므로 독립적으로 최적화한 deterministic baseline이라고 부르지 않습니다.

## 8. 메모리·호환성·다음 보정 기준

- 새 checkpoint format은 `climate_diffusion.manifold_recurrent_fm.v1`입니다. 기존 manifold v1은 원래 lead-conditioned 경로로 읽힙니다. config/format 불일치는 reject하며 이전 B/C weights를 recurrent로 묵시 전환하지 않습니다. 이 매뉴얼은 새 A부터 학습합니다.
- 파라미터 수는 같은 dimensions의 기존 모델과 같습니다. 비용은 커집니다: 최대20physical step×2×tau_steps×K개 expert 평가와 decoder Jacobian/BPTT입니다. C marginal score와 trajectory score가 별도 샘플 경로여서 둘 다 비용을 가집니다.
- RTX4090은 처음 full20, batch2(마지막 singleton 병합 시3), M2~4, τ steps2~4로1epoch를 측정하세요. 이는 초기 예산이지 VRAM 보장이 아닙니다. 현재 gradient checkpointing/AMP/truncated BPTT는 추가하지 않았습니다.
- 최종 합성 실행에서 전체 process RSS는 약663MiB였지만 GPU VRAM과 같은 지표가 아닙니다. 실제 D2048/r16, multi-year archive는 CPU RAM·seal 비용도 별도 측정해야 합니다.
- acceptance는 +6h 이후 tendency 개선뿐 아니라 state skill, Energy/CRPS, spread/coverage, 영역별 expert skill을 함께 만족해야 합니다. 작은 candidate cosine/높은 usage만으로 물리 regime 분리 성공을 주장하지 않습니다. 실제 ERA5 개선은 아직 미검증입니다.

수정 위치: `recurrent_flow.py`(FM/physical step/단위/trace), `train_manifold_moe.py`(_pairs/_sample/_epoch/저장), `temporal_supervision.py`(member loss), `recurrent_diagnostics.py`(경로 검사), `trajectory_output.py`(영상), `inference.py`(format 분기).

[학습/gradient 그림](../struct-picture/13-physical-recurrent-training.md) · [두 적분의 확대 그림](../struct-picture/14-residual-fm-and-physical-step.md)
