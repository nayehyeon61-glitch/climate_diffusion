# Physical-Time Recurrent Ensemble Flow Matching 수정 지시서
Project: climate_diffusion
Target context: results/temporal-120h-era5-run-001
Reference code commit: 0e641b270d8884105371e93adffdc756a5929750

## 0. 목적

현재 Physics-informed Manifold MoE Ensemble Flow Matching 구조 자체는 유지한다.

유지할 핵심 구조:
Physical Manifold
→ PI Encoder / Decoder
→ Flow Experts
→ Tangent Projection
→ Local Manifold Gate
→ Expert Fusion
→ Flow ODE
→ Ensemble Members

이번 수정의 목적은 모델을 새로 확장하는 것이 아니라,
현재 서로 분리되어 있는 "physical-time dynamics", "Flow Matching generative time",
"ensemble trajectory"의 연결을 정확히 만드는 것이다.

현재 관찰된 문제:
- +6h 또는 첫 lead에서는 의미 있는 변화가 존재함.
- 이후 +12h~+120h에서 prediction tendency가 급격히 작아짐.
- GIF에서 vector field가 거의 고정된 것처럼 보임.
- 이는 단순 rendering artifact가 아니라 실제 forecast tendency collapse와 관련됨.

핵심 목표:
각 ensemble member가 독립적인 forecast lead를 생성하는 것이 아니라,
자기 자신의 이전 상태를 다음 physical time step으로 넘기는 recurrent trajectory를 생성하도록 수정한다.


======================================================================
1. 가장 중요한 개념 분리: generative time tau vs physical forecast time t
======================================================================

반드시 두 시간축을 구분한다.

(1) Flow Matching generative time:
    tau in [0, 1]

    q_tau = (1 - tau) q_source + tau q_target

tau는 source/noise에서 target sample로 이동하는 artificial/generative path time이다.
tau를 6h, 12h, 18h 같은 physical forecast time으로 해석하지 않는다.

(2) Physical forecast time:
    t = 0h, 6h, 12h, ..., 120h

실제 대기 상태는 physical time을 따라 다음과 같이 진화해야 한다.

    z_t -> z_{t+6h} -> z_{t+12h} -> ... -> z_{t+120h}

최종 구현에서는 tau와 t를 코드 변수/함수 인자 수준에서도 가능한 한 명확히 분리한다.


======================================================================
2. 현재 trajectory sampling 방식 수정
======================================================================

현재 문제:
각 lead가 동일한 initial latent/source로부터 독립적으로 생성되는 구조에 가깝다.

현재 개념:
    q0 -> forecast(+6h)
    q0 -> forecast(+12h)
    q0 -> forecast(+18h)
    ...
    q0 -> forecast(+120h)

수정 목표:
각 ensemble member가 이전 예측 상태를 다음 physical step의 입력으로 사용한다.

수정 후:
    member m:
    z_0^(m)
      -> z_6^(m)
      -> z_12^(m)
      -> z_18^(m)
      -> ...
      -> z_120^(m)

즉 physical autoregressive/recurrent rollout을 구현한다.

권장 수식:

    dz^(m)/dt = a_final^(m)(z_t^(m), h_t, t, epsilon_m)

그리고 각 6h step은

    z_{t+Delta t}^(m)
      = Integrate(
          z_t^(m),
          a_final^(m),
          t -> t + Delta t
        )

Delta t = 6 hours.

Euler 식으로 표현하면:

    z_{t+6}^(m)
      = z_t^(m)
        + Delta t * a_final^(m)

실제 구현은 현재 Flow ODE integrator를 재사용해도 된다.


======================================================================
3. Stage A latent_drift를 실제 forecast dynamics에 연결
======================================================================

현재 Stage A에서는 adjacent physical states로 latent dynamics를 학습한다.

예:
    z_{t+Delta t}
      ~= z_t + Delta t * f_drift(z_t)

하지만 현재 inference의 main forecast field는
Flow Experts -> Tangent Projection -> Gate Fusion -> ODE
쪽을 사용하고,
Stage A에서 학습한 latent_drift가 실제 forecast vector field의 중심축으로 직접 사용되지 않는다.

수정 목표:
latent_drift를 deterministic physical backbone으로 사용하고,
MoE Flow Experts는 residual dynamics를 학습하도록 연결한다.

권장 구조:

    a_base = f_drift(z_t)

    Delta a_k
      = residual velocity predicted by Flow Expert k

    a_final^(m)
      = a_base
        + sum_k pi_k^(m) * Delta a_k^(m)

즉:

    a_final^(m)
      = f_drift(z_t^(m))
        + Σ_k pi_k^(m) Delta a_k^(m)

이 구조의 의미:
- f_drift:
  평균적/기본적인 physical evolution 유지
- expert residual:
  regime-specific correction
- stochastic member:
  ensemble uncertainty
- gate:
  local manifold/regime-dependent expert weighting

중요:
Flow Expert가 전체 대기 dynamics를 처음부터 다시 학습하게 하지 말고,
가능하면 Stage A physical drift에 대한 residual correction을 학습하도록 한다.


======================================================================
4. Ensemble은 반드시 유지한다
======================================================================

Ensemble을 제거하지 않는다.

각 member m = 1,...,M은 독립적인 stochastic trajectory를 가진다.

예:

    member 1:
    z0^1 -> z6^1 -> z12^1 -> ... -> z120^1

    member 2:
    z0^2 -> z6^2 -> z12^2 -> ... -> z120^2

    ...

    member M:
    z0^M -> z6^M -> z12^M -> ... -> z120^M

각 member는 다음 중 적어도 하나를 통해 차이를 유지한다.
- member-specific source noise epsilon_m
- member-specific residual expert response
- member-specific stochastic perturbation
- member-specific gate response, if state divergence induces it

중요:
한 member 안에서는 시간축을 따라 stochastic identity/coupling을 유지한다.
매 lead마다 완전히 새로운 독립 noise를 다시 뽑아 trajectory identity가 끊기지 않도록 한다.

권장:
- member별 base noise/state는 trajectory 전체에서 일관되게 유지
- 필요한 경우 step noise는 별도 도입하되 training/inference 규칙을 동일하게 유지


======================================================================
5. sample_trajectory / forecast 계열 함수 수정
======================================================================

우선 확인할 주요 코드 위치:

- src/climate_diffusion/manifold_moe.py
- src/climate_diffusion/train_manifold_moe.py
- src/climate_diffusion/temporal_supervision.py
- src/climate_diffusion/inference.py
- tests/test_temporal_supervision.py
- tests/test_manifold_moe.py
- src/climate_diffusion/trajectory_output.py

핵심 수정 대상은 sample_trajectory 또는 이에 준하는 trajectory generation 함수이다.

기존 의미:
    동일 initial source + lead conditioning
    -> 각 lead forecast를 개별 생성

새 의미:
    current_state = initial_state

    for physical_step in lead_steps:
        velocity = physical_dynamics(
            current_state,
            context,
            physical_time,
            member_noise
        )

        current_state = integrate_one_physical_step(
            current_state,
            velocity,
            dt=6h
        )

        trajectory.append(current_state)

Pseudo-code:

    def sample_trajectory(...):
        members = initialize_members(...)
        current = members

        outputs = []

        for step in physical_steps:
            t_phys = step * step_hours

            base_drift = latent_drift(current)

            residual = moe_residual_field(
                current,
                context,
                t_phys,
                member_noise,
            )

            final_velocity = base_drift + residual

            current = integrate_physical_step(
                current,
                final_velocity,
                dt_hours=step_hours,
            )

            outputs.append(decode(current))

        return stack(outputs)

주의:
현재 field/integrate 함수가 tau-integrator라면
physical integration과 generative tau integration을 혼합하지 않는다.

필요하면 함수 계층을 분리한다.

예:
    integrate_flow_tau(...)
    integrate_physical_time(...)

또는:
    sample_residual_velocity_via_fm(...)
    physical_step(...)


======================================================================
6. Temporal loss 수정
======================================================================

현재 delta/tendency loss의 기본 아이디어는 유지한다.

실제 physical tendency:

    v_true(t)
      = (s_{t+Delta t} - s_t) / Delta t

예측 tendency:

    v_pred^(m)(t)
      = (s_hat_{t+Delta t}^(m) - s_hat_t^(m)) / Delta t

현재 ensemble mean 기반 tendency supervision만으로는
개별 member가 quasi-static해지는 현상을 충분히 막지 못할 수 있다.

따라서 member-wise tendency loss를 추가한다.

권장:

    L_delta_member
      = (1/M) Σ_m Σ_t
        || v_pred^(m)(t) - v_true(t) ||_W^2

그리고 기존 ensemble-mean tendency loss도 유지 가능:

    L_delta_mean
      = Σ_t
        || mean_m[v_pred^(m)(t)] - v_true(t) ||_W^2

최종:

    L_delta
      = lambda_member * L_delta_member
        + lambda_mean * L_delta_mean

주의:
모든 member를 truth에 지나치게 강하게 붙이면 ensemble spread가 사라질 수 있다.

따라서 권장:
- member-wise loss는 moderate weight
- distribution/energy/CRPS loss와 함께 사용
- spread regularization 유지
- ensemble diversity가 collapse하지 않는지 함께 모니터링


======================================================================
7. Full 120h trajectory supervision
======================================================================

현재 training에서 trajectory_edges=2인 경우,
전체 20개 6h edge 중 짧은 sub-block만 감독하게 된다.

이번 수정 실험에서는 우선:

    trajectory_edges = 0

즉 전체 120h / 20 edge를 training trajectory loss에 사용한다.

목적:
모델이 단순히 짧은 local transition만 맞추는 것이 아니라
0h -> 120h 동안 movement를 유지하도록 직접 감독한다.

Loss:

    L_traj
      = Σ_{j=1}^{20}
        || s_hat_{t+6j} - s_{t+6j} ||_W^2

또는 현재 fair energy / probabilistic trajectory objective를 유지하되
full physical trajectory를 대상으로 한다.


======================================================================
8. Loss 전체 설계 권장안
======================================================================

기존 A/B/C stage는 유지한다.

Stage A: Physical Manifold Pretraining
--------------------------------------
목적:
- reconstruction
- physics/invariant
- manifold metric
- adjacent physical dynamics

Loss:

    L_A =
        L_reconstruction
      + lambda_phys L_physics
      + lambda_inv L_invariant
      + lambda_metric L_metric
      + lambda_drift L_latent_dynamics

중요:
Stage A에서 학습한 f_drift는 이후 Stage B/C forecast에 실제 연결한다.


Stage B: Expert Specialization
------------------------------
목적:
- expert residual field 학습
- local gate 학습
- expert collapse 방지

권장 target:
전체 target velocity가 아니라,
가능하면 base drift를 뺀 residual target을 사용.

    v_residual_target
      = v_target - f_drift(z)

Expert k:
    Delta a_k ~= v_residual_target

Loss:
    L_B =
        L_expert_FM
      + L_fused_FM
      + lambda_gate L_gate
      + lambda_balance L_balance
      + lambda_div L_diversity
      + lambda_proj L_projection


Stage C: Joint Physical Trajectory Calibration
----------------------------------------------
목적:
실제 120h recurrent rollout을 end-to-end로 보정한다.

    L_C =
        L_FM
      + lambda_energy L_energy
      + lambda_crps L_CRPS
      + lambda_traj L_trajectory
      + lambda_delta L_delta_member/mean
      + lambda_wspeed L_wind_speed
      + lambda_wdir L_wind_direction
      + lambda_anchor L_anchor
      + lambda_PI L_PI

중요:
Stage C temporal losses는 반드시 recurrent physical rollout 결과에 대해 계산한다.


======================================================================
9. Manifold dimension 관련 진단
======================================================================

현재 실제 run:
    state_dim = 2048
    manifold_dim = 16

이는 매우 강한 compression이다.

현재 reconstruction RMSE가 충분히 작지 않고,
tangent projection 후 velocity magnitude가 줄어드는 경우
manifold dimension이 dynamics bottleneck일 가능성이 있다.

하지만 이것은 1차 수정 대상이 아니다.

우선:
1. physical recurrent connection 수정
2. latent_drift 연결
3. full trajectory supervision

을 한 뒤에도 tendency collapse가 남는 경우
다음 ablation을 수행한다.

    manifold_dim = 16 / 32 / 64

각 설정에서 측정:
- reconstruction RMSE
- tangent projection norm ratio
- pre-projection velocity norm
- post-projection velocity norm
- physical tendency amplitude ratio
- candidate cosine similarity


======================================================================
10. 반드시 추가할 diagnostics
======================================================================

각 physical lead마다 아래 값을 저장한다.

A. Physical tendency
--------------------
    truth_tendency_rms
    prediction_tendency_rms_per_member
    ensemble_mean_tendency_rms
    amplitude_ratio

목표:
+6h 이후에도 amplitude ratio가 0에 가까이 붕괴하지 않는지 확인.


B. Drift / residual decomposition
---------------------------------
각 step에서:

    ||a_base||
    ||a_residual||
    ||a_final||

그리고 변수별:
    msl
    t2m
    u10
    v10


C. Projection attenuation
-------------------------
각 expert:

    raw_norm = ||v_raw,k||
    projected_norm = ||v_projected,k||

    projection_ratio
      = projected_norm / raw_norm

projection_ratio가 매우 작으면 tangent projection이 dynamics를 죽이고 있는 것이다.


D. Gate / expert diversity
--------------------------
    expert usage
    candidate cosine similarity
    residual velocity variance
    gate entropy

현재 candidate cosine similarity가 높았으므로
expert collapse 여부를 계속 확인한다.


E. Ensemble quality
-------------------
    ensemble spread
    spread/skill ratio
    CRPS
    central 80% coverage

physical dynamics를 강화하면서 ensemble이 deterministic하게 collapse하지 않는지 확인한다.


======================================================================
11. Rendering 관련 수정
======================================================================

현재 rendering은 주원인으로 보지 않는다.

다만 시각적 진단을 위해 다음을 추가한다.

1. fixed quiver scale GIF
2. adaptive/local quiver scale diagnostic GIF
3. vector magnitude heatmap
4. difference field:
       prediction(t+6)-prediction(t)
5. truth difference field:
       truth(t+6)-truth(t)

중요:
adaptive scale GIF는 presentation/diagnostic용일 뿐,
실제 magnitude 비교는 fixed scale 결과를 기준으로 한다.


======================================================================
12. 테스트 수정
======================================================================

기존 "모든 lead가 동일 initial latent source에서 독립적으로 시작해야 한다"
성격의 테스트가 있다면 recurrent physical rollout 의미에 맞게 수정한다.

새 테스트 요구사항:

Test 1. Physical recurrence
---------------------------
두 번째 lead의 initial state가 첫 번째 forecast state와 일치하는지 검증.

    input_to_step_2 == output_of_step_1


Test 2. Member identity
-----------------------
member m의 noise/state coupling이 전체 trajectory에서 유지되는지 검증.


Test 3. No cross-member contamination
-------------------------------------
member 1의 상태가 member 2의 next-state 입력으로 사용되지 않는지 검증.


Test 4. Drift connection
------------------------
expert residual을 zero로 강제했을 때:

    trajectory
      == latent_drift-only trajectory

가 되는지 검증.


Test 5. Residual connection
---------------------------
latent_drift를 zero로 강제했을 때
expert/MoE residual만으로 rollout이 가능한지 검증.


Test 6. Full recurrent gradient
-------------------------------
120h trajectory loss의 gradient가
- experts
- gate
- history encoder
- Stage C에서 허용된 manifold parameters

까지 전달되는지 검증.


Test 7. Tendency scaling
------------------------
dt=6h, 12h 변경 시 tendency가 올바르게 1/dt scaling되는지 검증.


Test 8. Synthetic constant velocity
-----------------------------------
synthetic data:

    x_{t+1} = x_t + c

를 사용하여 20-step rollout을 수행했을 때
예측 tendency가 첫 step 후 0으로 collapse하지 않는지 검증.

이 테스트는 반드시 추가할 것을 권장한다.


======================================================================
13. 구현 우선순위
======================================================================

Priority 1:
- recurrent physical trajectory 연결
- sample_trajectory 의미 수정

Priority 2:
- Stage A latent_drift를 final velocity backbone으로 연결

Priority 3:
- residual expert target으로 Stage B 수정

Priority 4:
- member-wise delta/tendency loss 추가

Priority 5:
- trajectory_edges=0 full 120h supervision

Priority 6:
- diagnostics 추가

Priority 7:
- manifold_dim 16/32/64 ablation

Priority 8:
- expert diversity/gating 세부 튜닝


======================================================================
14. 이번 수정에서 하지 않을 것
======================================================================

이번 수정의 목적은 architecture 확장이 아니다.

하지 않을 것:
- 새로운 대형 backbone 추가
- Transformer/FNO를 새로 추가
- variable 수를 무작정 확장
- WeatherNext2 재도입
- IBTrACS를 FM state target으로 직접 섞기
- rendering만 수정하고 문제 해결로 간주

현재 모델 구성요소를 최대한 유지하면서
physical-time 연결을 교정한다.


======================================================================
15. 최종 목표 구조
======================================================================

최종 권장 수식:

    z_{t+Delta t}^{(m)}
      =
      z_t^{(m)}
      +
      Delta t [
          f_drift(z_t^{(m)})
          +
          Σ_k pi_k(z_t^{(m)}, h_t, t)
              Delta f_k(
                  z_t^{(m)},
                  h_t,
                  t,
                  epsilon_m
              )
      ]

where:

    m = ensemble member
    k = MoE expert
    Delta t = 6h
    f_drift = Stage A physical latent dynamics
    Delta f_k = expert residual vector field
    pi_k = local manifold gate
    epsilon_m = member-specific stochastic source

Trajectory:

    z_0^(m)
      -> z_6^(m)
      -> z_12^(m)
      -> ...
      -> z_120^(m)

for all m = 1,...,M.


======================================================================
16. 성공 기준 / Acceptance Criteria
======================================================================

수정 후 실제 ERA5 120h run에서 아래를 확인한다.

필수:
[ ] GIF에서 vector field가 단순 정지 상태가 아님
[ ] +6h 이후 prediction tendency RMS가 즉시 1~5% 수준으로 붕괴하지 않음
[ ] 20-step physical recurrent rollout이 실제로 사용됨
[ ] ensemble member M개가 모두 유지됨
[ ] member별 trajectory continuity 유지
[ ] Stage A latent_drift가 실제 forecast dynamics에 사용됨
[ ] full 120h temporal loss가 backprop 가능
[ ] rendering 이전 raw NPZ diagnostics에서도 dynamics가 확인됨

권장 정량 기준:
- tendency amplitude ratio가 lead 2 이후 지속적으로 거의 0에 붙지 않을 것
- persistence RMSE보다 forecast RMSE가 계속 우수할 것
- ensemble spread가 지나치게 collapse하지 않을 것
- projection ratio가 비정상적으로 작지 않을 것
- candidate cosine similarity가 1에 과도하게 수렴하지 않을 것


======================================================================
17. 한 문장 요약
======================================================================

"Physics-informed Manifold MoE Ensemble Flow Matching" 구조는 유지하되,
각 lead를 독립적으로 생성하는 방식에서 벗어나
Stage A physical drift + MoE stochastic residual을 사용하여
각 ensemble member가 6시간 단위로 이전 상태를 이어받는
physical-time recurrent ensemble trajectory를 생성하도록 연결을 수정한다.

