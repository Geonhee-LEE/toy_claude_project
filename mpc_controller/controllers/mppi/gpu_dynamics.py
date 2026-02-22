"""JAX JIT 동역학 — lax.scan + vmap 기반 배치 rollout.

최대 병목(~65%) 해결: Python for-loop(N=30) → XLA fused kernel (1회).
dynamics_fn을 static_argnums로 전달 → 모델별 JIT 캐시 자동 분리.
"""

import jax
import jax.numpy as jnp
from jax import lax
from functools import partial


# ─────────────────────────────────────────────────────────────
# 동역학 함수 (pure function, JAX-traceable)
# ─────────────────────────────────────────────────────────────

def diff_drive_dynamics(state, control):
    """Differential drive 연속시간 동역학.

    State: [x, y, θ]  Control: [v, ω]
    ẋ = v·cos(θ), ẏ = v·sin(θ), θ̇ = ω
    """
    theta = state[2]
    v = control[0]
    omega = control[1]
    return jnp.array([v * jnp.cos(theta), v * jnp.sin(theta), omega])


def swerve_dynamics(state, control):
    """Swerve drive 연속시간 동역학.

    State: [x, y, θ]  Control: [vx, vy, ω]
    Body frame → World frame 변환.
    """
    theta = state[2]
    vx_body = control[0]
    vy_body = control[1]
    omega = control[2]
    cos_t = jnp.cos(theta)
    sin_t = jnp.sin(theta)
    x_dot = vx_body * cos_t - vy_body * sin_t
    y_dot = vx_body * sin_t + vy_body * cos_t
    return jnp.array([x_dot, y_dot, omega])


def non_coaxial_swerve_dynamics(state, control):
    """Non-coaxial swerve drive 연속시간 동역학.

    State: [x, y, θ, β]  Control: [vx, vy, ω]
    β: 조향 각도 상태 (nx=4).
    """
    theta = state[2]
    vx_body = control[0]
    vy_body = control[1]
    omega = control[2]
    cos_t = jnp.cos(theta)
    sin_t = jnp.sin(theta)
    x_dot = vx_body * cos_t - vy_body * sin_t
    y_dot = vx_body * sin_t + vy_body * cos_t
    beta_dot = omega  # 조향 각속도
    return jnp.array([x_dot, y_dot, omega, beta_dot])


# ─────────────────────────────────────────────────────────────
# RK4 적분 + rollout
# ─────────────────────────────────────────────────────────────

def _rk4_step(dynamics_fn, state, control, dt):
    """RK4 1스텝 적분 (pure function)."""
    k1 = dynamics_fn(state, control)
    k2 = dynamics_fn(state + dt / 2 * k1, control)
    k3 = dynamics_fn(state + dt / 2 * k2, control)
    k4 = dynamics_fn(state + dt * k3, control)
    next_state = state + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    # 각도 정규화 (θ = state[2])
    next_state = next_state.at[2].set(
        jnp.arctan2(jnp.sin(next_state[2]), jnp.cos(next_state[2]))
    )
    return next_state


def _make_scan_fn(dynamics_fn, dt):
    """lax.scan용 step function 생성."""
    def scan_step(state, control):
        next_state = _rk4_step(dynamics_fn, state, control, dt)
        return next_state, next_state
    return scan_step


def _single_rollout(dynamics_fn, x0, controls, dt):
    """단일 제어 시퀀스 rollout: lax.scan으로 N스텝 순차 적분.

    Args:
        dynamics_fn: 동역학 함수 (state, control) -> state_dot
        x0: (nx,) 초기 상태
        controls: (N, nu) 제어 시퀀스
        dt: 시간 간격

    Returns:
        (N+1, nx) 궤적 (x0 포함)
    """
    scan_fn = _make_scan_fn(dynamics_fn, dt)
    _, trajectory = lax.scan(scan_fn, x0, controls)
    # x0을 앞에 붙임: (N+1, nx)
    return jnp.concatenate([x0[None, :], trajectory], axis=0)


def make_rollout_batch_jit(dynamics_fn, dt):
    """JIT-compiled 배치 rollout 함수 생성.

    vmap으로 K 차원 자동 병렬화 + lax.scan으로 N스텝 fusion.

    Args:
        dynamics_fn: 동역학 함수 (diff_drive/swerve/non_coaxial_swerve)
        dt: 시간 간격

    Returns:
        rollout_fn(x0, control_sequences) -> (K, N+1, nx) 궤적
        x0: (nx,) 초기 상태
        control_sequences: (K, N, nu) 제어 시퀀스
    """
    def _rollout_single(controls):
        """Closure: x0는 외부에서 전달 (partial로)."""
        # 이 함수는 아래 _rollout_batch에서 vmap으로 감싸짐
        raise NotImplementedError  # placeholder

    @jax.jit
    def rollout_batch(x0, control_sequences):
        """배치 rollout.

        Args:
            x0: (nx,) 초기 상태 (모든 K 샘플 동일)
            control_sequences: (K, N, nu) 제어 시퀀스

        Returns:
            (K, N+1, nx) 궤적
        """
        # vmap: K 차원 자동 벡터화
        # in_axes=0 → control_sequences의 첫 축(K)을 병렬화
        batched_rollout = jax.vmap(
            lambda controls: _single_rollout(dynamics_fn, x0, controls, dt)
        )
        return batched_rollout(control_sequences)

    return rollout_batch


# ─────────────────────────────────────────────────────────────
# 제어 클리핑 (JIT)
# ─────────────────────────────────────────────────────────────

@jax.jit
def clip_controls_jit(controls, max_velocity, max_omega):
    """제어 입력 클리핑 (JAX).

    Args:
        controls: (..., nu) 제어 입력 (nu=2: [v, ω] 또는 nu=3: [vx, vy, ω])
        max_velocity: 최대 선속도
        max_omega: 최대 각속도

    Returns:
        클리핑된 제어 입력
    """
    nu = controls.shape[-1]
    if nu == 2:
        # diff_drive: [v, ω]
        clipped_v = jnp.clip(controls[..., 0], -max_velocity, max_velocity)
        clipped_w = jnp.clip(controls[..., 1], -max_omega, max_omega)
        return jnp.stack([clipped_v, clipped_w], axis=-1)
    else:
        # swerve: [vx, vy, ω]
        clipped_vx = jnp.clip(controls[..., 0], -max_velocity, max_velocity)
        clipped_vy = jnp.clip(controls[..., 1], -max_velocity, max_velocity)
        clipped_w = jnp.clip(controls[..., 2], -max_omega, max_omega)
        return jnp.stack([clipped_vx, clipped_vy, clipped_w], axis=-1)


# ─────────────────────────────────────────────────────────────
# 유틸리티
# ─────────────────────────────────────────────────────────────

DYNAMICS_REGISTRY = {
    "diff_drive": diff_drive_dynamics,
    "swerve": swerve_dynamics,
    "non_coaxial_swerve": non_coaxial_swerve_dynamics,
}


def get_dynamics_fn(model_name: str = "diff_drive"):
    """모델 이름으로 동역학 함수 조회.

    Args:
        model_name: "diff_drive", "swerve", "non_coaxial_swerve"

    Returns:
        JAX-traceable 동역학 함수
    """
    if model_name not in DYNAMICS_REGISTRY:
        raise ValueError(
            f"Unknown model: {model_name}. "
            f"Available: {list(DYNAMICS_REGISTRY.keys())}"
        )
    return DYNAMICS_REGISTRY[model_name]
