"""통합 MPPI JIT Kernel — 전체 compute_control을 GPU에서 실행.

GPU↔CPU 전송 최소화: 호출당 2회 (입력 전송 + 결과 반환).
중간 결과는 모두 GPU 메모리에 유지.

┌──────────────────────────────────────────────────────┐
│ CPU 측                          GPU 측               │
│  x0, U, ref ──── 1회 전송 ────► mppi_step_jit()    │
│                                  ├ sample noise      │
│                                  ├ perturb + clip    │
│                                  ├ rollout (scan)    │
│                                  ├ cost (fused)      │
│                                  ├ softmax weights   │
│                                  └ weighted update   │
│  u_opt, info ◄── 1회 전송 ────── return             │
└──────────────────────────────────────────────────────┘
"""

import logging

import jax
import jax.numpy as jnp
from jax import random
import numpy as np

from mpc_controller.controllers.mppi.gpu_dynamics import (
    make_rollout_batch_jit,
    clip_controls_jit,
    get_dynamics_fn,
)
from mpc_controller.controllers.mppi.gpu_costs import make_compute_all_costs_jit
from mpc_controller.controllers.mppi.gpu_sampling import (
    gaussian_sample_jit,
    colored_noise_sample_jit,
)
from mpc_controller.controllers.mppi.gpu_backend import to_jax, to_numpy

logger = logging.getLogger(__name__)


class GPUMPPIKernel:
    """GPU에서 실행되는 MPPI 핵심 연산 커널.

    JIT 컴파일된 함수들을 조합하여 전체 MPPI 스텝을 GPU에서 수행.
    CPU측 base_mppi.py의 _compute_control_gpu()에서 호출.
    """

    def __init__(
        self,
        N: int,
        K: int,
        nu: int,
        nx: int,
        dt: float,
        lambda_: float,
        noise_sigma: np.ndarray,
        q_diag: np.ndarray,
        qf_diag: np.ndarray,
        r_diag: np.ndarray,
        max_velocity: float,
        max_omega: float,
        r_rate_diag: np.ndarray | None = None,
        colored_noise: bool = False,
        noise_beta: float = 2.0,
        model_name: str = "diff_drive",
        use_float32: bool = False,
        warmup: bool = True,
    ):
        self.N = N
        self.K = K
        self.nu = nu
        self.nx = nx
        self.dt = dt
        self.lambda_ = lambda_
        self.model_name = model_name

        # dtype 설정
        dtype = jnp.float32 if use_float32 else jnp.float64

        # JAX 배열로 변환
        self._sigma = to_jax(noise_sigma).astype(dtype)
        self._q_diag = to_jax(q_diag).astype(dtype)
        self._qf_diag = to_jax(qf_diag).astype(dtype)
        self._r_diag = to_jax(r_diag).astype(dtype)
        self._max_velocity = float(max_velocity)
        self._max_omega = float(max_omega)
        self._colored_noise = colored_noise
        self._noise_beta = float(noise_beta)
        self._has_r_rate = r_rate_diag is not None
        self._r_rate_diag = (
            to_jax(r_rate_diag).astype(dtype) if r_rate_diag is not None else None
        )
        self._dtype = dtype

        # PRNG key 초기화
        self._rng_key = random.PRNGKey(42)

        # 장애물 (동적 업데이트 가능)
        self._obstacles = None
        self._obs_weight = 1000.0
        self._safety_margin = 0.3
        self._has_obstacles = False

        # JIT 함수 생성
        dynamics_fn = get_dynamics_fn(model_name)
        self._rollout_fn = make_rollout_batch_jit(dynamics_fn, dt)
        self._cost_fn = make_compute_all_costs_jit(
            has_r_rate=self._has_r_rate,
            has_obstacles=False,
        )

        # warmup: 더미 입력으로 JIT 사전 컴파일
        if warmup:
            self._warmup()

    def _warmup(self):
        """JIT 사전 컴파일 (첫 호출 지연 제거)."""
        logger.info("GPU MPPI kernel warmup: JIT compiling...")
        dummy_x0 = jnp.zeros(self.nx, dtype=self._dtype)
        dummy_U = jnp.zeros((self.N, self.nu), dtype=self._dtype)
        dummy_ref = jnp.zeros((self.N + 1, self.nx), dtype=self._dtype)

        # mppi_step 한 번 호출 → 모든 JIT 함수 컴파일
        self.mppi_step(dummy_x0, dummy_U, dummy_ref)
        logger.info("GPU MPPI kernel warmup complete.")

    def set_obstacles(
        self,
        obstacles: np.ndarray,
        weight: float = 1000.0,
        safety_margin: float = 0.3,
    ):
        """장애물 업데이트 + 비용 함수 재생성.

        장애물 수 변경 시 JIT 재컴파일 발생 (최초 1회).
        """
        self._has_obstacles = len(obstacles) > 0
        if self._has_obstacles:
            self._obstacles = to_jax(obstacles).astype(self._dtype)
        else:
            self._obstacles = None
        self._obs_weight = weight
        self._safety_margin = safety_margin

        # 장애물 유무에 따라 비용 함수 재생성
        self._cost_fn = make_compute_all_costs_jit(
            has_r_rate=self._has_r_rate,
            has_obstacles=self._has_obstacles,
        )

    def mppi_step(self, x0, U, reference):
        """전체 MPPI 1스텝 (GPU).

        Args:
            x0: (nx,) JAX 배열 — 현재 상태
            U: (N, nu) JAX 배열 — 현재 제어열
            reference: (N+1, nx) JAX 배열 — 참조 궤적

        Returns:
            (U_new, info_dict):
                U_new: (N, nu) 업데이트된 제어열 (JAX)
                info_dict: trajectories, costs, weights, best_idx 등 (JAX)
        """
        K = self.K
        N = self.N
        nu = self.nu

        # 1. PRNG key 분할
        self._rng_key, subkey = random.split(self._rng_key)

        # 2. 노이즈 샘플링
        if self._colored_noise:
            noise = colored_noise_sample_jit(
                subkey, K, N, nu, self._sigma, self._noise_beta
            )
        else:
            noise = gaussian_sample_jit(subkey, K, N, nu, self._sigma)

        noise = noise.astype(self._dtype)

        # 3. 제어열에 노이즈 추가 + 클리핑
        perturbed = U[None, :, :] + noise  # (K, N, nu)
        perturbed = clip_controls_jit(
            perturbed, self._max_velocity, self._max_omega
        )

        # 4. 배치 rollout
        trajectories = self._rollout_fn(x0, perturbed)  # (K, N+1, nx)

        # 5. 비용 계산
        cost_params = {
            "q_diag": self._q_diag,
            "qf_diag": self._qf_diag,
            "r_diag": self._r_diag,
        }
        if self._has_r_rate:
            cost_params["r_rate_diag"] = self._r_rate_diag
        if self._has_obstacles and self._obstacles is not None:
            cost_params["obstacles"] = self._obstacles
            cost_params["obs_weight"] = self._obs_weight
            cost_params["safety_margin"] = self._safety_margin

        costs = self._cost_fn(trajectories, perturbed, reference, cost_params)

        # 6. Softmax 가중치
        shifted = -costs / self.lambda_
        shifted = shifted - jnp.max(shifted)
        exp_vals = jnp.exp(shifted)
        weights = exp_vals / jnp.sum(exp_vals)

        # 7. 가중 평균으로 제어열 업데이트
        weighted_noise = weights[:, None, None] * noise  # (K, N, nu)
        U_new = U + jnp.sum(weighted_noise, axis=0)  # (N, nu)
        U_new = clip_controls_jit(
            U_new[None, :, :], self._max_velocity, self._max_omega
        )[0]

        # 8. info dict (JAX 배열)
        best_idx = jnp.argmin(costs)
        weighted_traj = jnp.sum(
            weights[:, None, None] * trajectories, axis=0
        )
        ess = 1.0 / jnp.sum(weights ** 2)

        info = {
            "trajectories": trajectories,
            "costs": costs,
            "weights": weights,
            "best_idx": best_idx,
            "best_trajectory": trajectories[best_idx],
            "weighted_trajectory": weighted_traj,
            "ess": ess,
        }

        return U_new, info

    def update_lambda(self, lambda_: float):
        """온도 파라미터 업데이트."""
        self.lambda_ = lambda_
