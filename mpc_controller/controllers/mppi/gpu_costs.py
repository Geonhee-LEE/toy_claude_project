"""JAX JIT 비용 함수 — 단일 kernel fusion.

기존 CompositeMPPICost의 Python for-loop을 단일 @jit 함수로 fusion.
ObstacleCost 벡터화: (K,N+1,1,2) - (1,1,M,2) → (K,N+1,M) 한 번에 계산.
"""

import jax
import jax.numpy as jnp
from functools import partial


@jax.jit
def normalize_angle_jit(angles):
    """각도 정규화 [-π, π] (JAX)."""
    return jnp.arctan2(jnp.sin(angles), jnp.cos(angles))


@jax.jit
def state_tracking_cost_jit(trajectories, reference, q_diag):
    """상태 추적 비용 (JAX).

    cost_k = Σ_{t=0}^{N-1} (x_t - ref_t)^T Q (x_t - ref_t)

    Args:
        trajectories: (K, N+1, nx)
        reference: (N+1, nx)
        q_diag: (nx,) Q 대각 원소

    Returns:
        (K,) 비용
    """
    error = trajectories[:, :-1, :] - reference[None, :-1, :]
    # 각도 오차 정규화 (θ = index 2)
    error = error.at[:, :, 2].set(normalize_angle_jit(error[:, :, 2]))
    weighted = error ** 2 * q_diag[None, None, :]
    return jnp.sum(weighted, axis=(1, 2))


@jax.jit
def terminal_cost_jit(trajectories, reference, qf_diag):
    """터미널 비용 (JAX).

    cost_k = (x_N - ref_N)^T Qf (x_N - ref_N)

    Args:
        trajectories: (K, N+1, nx)
        reference: (N+1, nx)
        qf_diag: (nx,) Qf 대각 원소

    Returns:
        (K,) 비용
    """
    error = trajectories[:, -1, :] - reference[-1, :]
    error = error.at[:, 2].set(normalize_angle_jit(error[:, 2]))
    weighted = error ** 2 * qf_diag[None, :]
    return jnp.sum(weighted, axis=1)


@jax.jit
def control_effort_cost_jit(controls, r_diag):
    """제어 입력 비용 (JAX).

    cost_k = Σ_t u_t^T R u_t

    Args:
        controls: (K, N, nu)
        r_diag: (nu,) R 대각 원소

    Returns:
        (K,) 비용
    """
    weighted = controls ** 2 * r_diag[None, None, :]
    return jnp.sum(weighted, axis=(1, 2))


@jax.jit
def control_rate_cost_jit(controls, r_rate_diag):
    """제어 변화율 비용 (JAX).

    cost_k = Σ_t (u_{t+1} - u_t)^T R_rate (u_{t+1} - u_t)

    Args:
        controls: (K, N, nu)
        r_rate_diag: (nu,) R_rate 대각 원소

    Returns:
        (K,) 비용
    """
    du = controls[:, 1:, :] - controls[:, :-1, :]
    weighted = du ** 2 * r_rate_diag[None, None, :]
    return jnp.sum(weighted, axis=(1, 2))


@jax.jit
def obstacle_cost_jit(trajectories, obstacles, weight, safety_margin):
    """장애물 회피 비용 — 벡터화 (JAX).

    M개 장애물 동시 처리: Python for-loop 완전 제거.
    (K,N+1,1,2) - (1,1,M,2) → (K,N+1,M) 거리 한번에 계산.

    Args:
        trajectories: (K, N+1, nx)
        obstacles: (M, 3) [x, y, radius]
        weight: 가중치 스칼라
        safety_margin: 안전 마진 [m]

    Returns:
        (K,) 비용
    """
    positions = trajectories[:, :, :2]  # (K, N+1, 2)
    obs_xy = obstacles[:, :2]           # (M, 2)
    obs_radius = obstacles[:, 2]        # (M,)

    # (K, N+1, 1, 2) - (1, 1, M, 2) → (K, N+1, M, 2) → (K, N+1, M)
    diff = positions[:, :, None, :] - obs_xy[None, None, :, :]
    dists = jnp.sqrt(jnp.sum(diff ** 2, axis=-1))

    # (M,) safety_dist = radius + margin
    safety_dists = obs_radius[None, None, :] + safety_margin

    penetration = jnp.maximum(0.0, safety_dists - dists)
    cost_per_sample = weight * jnp.sum(penetration ** 2, axis=(1, 2))
    return cost_per_sample


@jax.jit
def jerk_cost_jit(perturbed_du, r_jerk, jerk_weight):
    """Smooth-MPPI jerk 비용: ‖ΔΔu‖² (JAX).

    ΔΔu = Δu[t+1] - Δu[t] (제어 변화율의 변화율)

    Args:
        perturbed_du: (K, N, nu) Δu 시퀀스
        r_jerk: (nu,) jerk 가중치 대각 원소
        jerk_weight: 전체 스케일

    Returns:
        (K,) jerk 비용
    """
    ddu = perturbed_du[:, 1:, :] - perturbed_du[:, :-1, :]
    return jerk_weight * jnp.sum(ddu ** 2 * r_jerk[None, None, :], axis=(1, 2))


def make_compute_all_costs_jit(has_r_rate: bool, has_obstacles: bool):
    """비용 함수 통합 JIT kernel 생성.

    정적 플래그로 불필요한 비용 계산 분기 제거 → XLA 최적화.

    Args:
        has_r_rate: ControlRateCost 활성화 여부
        has_obstacles: ObstacleCost 활성화 여부

    Returns:
        compute_all_costs(trajectories, controls, reference, cost_params) → (K,)
    """

    @jax.jit
    def compute_all_costs(trajectories, controls, reference, cost_params):
        """통합 비용 계산.

        Args:
            trajectories: (K, N+1, nx)
            controls: (K, N, nu)
            reference: (N+1, nx)
            cost_params: dict with q_diag, qf_diag, r_diag, [r_rate_diag, obstacles, obs_weight, safety_margin]

        Returns:
            (K,) 총 비용
        """
        q_diag = cost_params["q_diag"]
        qf_diag = cost_params["qf_diag"]
        r_diag = cost_params["r_diag"]

        total = state_tracking_cost_jit(trajectories, reference, q_diag)
        total = total + terminal_cost_jit(trajectories, reference, qf_diag)
        total = total + control_effort_cost_jit(controls, r_diag)

        if has_r_rate:
            r_rate_diag = cost_params["r_rate_diag"]
            total = total + control_rate_cost_jit(controls, r_rate_diag)

        if has_obstacles:
            obstacles = cost_params["obstacles"]
            obs_weight = cost_params["obs_weight"]
            safety_margin = cost_params["safety_margin"]
            total = total + obstacle_cost_jit(
                trajectories, obstacles, obs_weight, safety_margin
            )

        return total

    return compute_all_costs
