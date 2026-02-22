"""Smooth MPPI (SMPPI) 컨트롤러 — Δu input-lifting 기반 구조적 부드러움.

Kim et al. (2021) 기반, pytorch_mppi v0.8.0+ 참고.

핵심 아이디어: u space 대신 Δu (제어 변화량) space에서 최적화하여
구조적으로 부드러운 제어 시퀀스를 생성한다.

┌──────────────────────────────────────────────────────────────┐
│ Vanilla MPPI:  optimize u[0..N-1]     → jerky 가능          │
│                                                              │
│ Smooth MPPI:   optimize Δu[0..N-1]   → cumsum → u          │
│                u[t] = u_prev + Σ_{i=0}^{t} Δu[i]            │
│                cost += R_jerk · ‖ΔΔu‖²                      │
│                                                              │
│  알고리즘:                                                    │
│   1. Δu space에서 노이즈 샘플링                              │
│   2. cumsum으로 u 시퀀스 복원                                │
│   3. 비용 계산 (tracking + terminal + effort + jerk)         │
│   4. Δu space에서 가중 평균 업데이트                         │
└──────────────────────────────────────────────────────────────┘
"""

import logging
import time
from typing import Optional, Tuple

import numpy as np

from mpc_controller.controllers.mppi.base_mppi import MPPIController
from mpc_controller.controllers.mppi.utils import softmax_weights, effective_sample_size
from mpc_controller.models.differential_drive import RobotParams
from mpc_controller.controllers.mppi.mppi_params import MPPIParams

logger = logging.getLogger(__name__)


class SmoothMPPIController(MPPIController):
    """Smooth MPPI — Δu input-lifting 기반 구조적 부드러움.

    Δu space에서 최적화: noise → Δu → cumsum → u
    추가 jerk cost로 ΔΔu 페널티 부여.
    """

    def __init__(
        self,
        robot_params: RobotParams | None = None,
        mppi_params: MPPIParams | None = None,
        seed: Optional[int] = None,
        obstacles: Optional[np.ndarray] = None,
    ):
        # GPU 초기화를 지연시키기 위해 use_gpu를 임시 비활성화
        _use_gpu = mppi_params.use_gpu if mppi_params else False
        if mppi_params and mppi_params.use_gpu:
            mppi_params = MPPIParams(**{
                f.name: getattr(mppi_params, f.name) for f in mppi_params.__dataclass_fields__.values()
            })
            mppi_params.use_gpu = False

        super().__init__(robot_params, mppi_params, seed, obstacles)

        # Δu warm-start 시퀀스 (U 대신 DU를 직접 유지)
        self.DU = np.zeros((self.params.N, self.dynamics.nu))

        # Jerk 가중치 기본값
        self._R_jerk = self.params.smooth_R_jerk
        if self._R_jerk is None:
            self._R_jerk = np.array([0.1, 0.1])
        self._jerk_weight = self.params.smooth_action_cost_weight

        # 이전 스텝의 마지막 적용 제어 (cumsum 기준점)
        self._u_prev = np.zeros(self.dynamics.nu)

        # 지연된 GPU 초기화
        if _use_gpu:
            self.params.use_gpu = True
            from mpc_controller.controllers.mppi.gpu_backend import is_jax_available
            if is_jax_available():
                self._use_gpu = True
                self._init_gpu()
            else:
                logger.warning("use_gpu=True but JAX not available. Falling back to CPU.")

    def _init_gpu(self):
        """GPU 커널 초기화 (Smooth-MPPI용 JAX 배열 추가 준비)."""
        super()._init_gpu()
        from mpc_controller.controllers.mppi.gpu_backend import to_jax
        self._R_jerk_jax = to_jax(self._R_jerk)

    def _compute_control_gpu(
        self,
        current_state: np.ndarray,
        reference_trajectory: np.ndarray,
    ) -> Tuple[np.ndarray, dict]:
        """GPU 경로: Smooth-MPPI Δu space 최적화."""
        from mpc_controller.controllers.mppi.gpu_backend import to_numpy

        solve_start = time.perf_counter()

        # 1. DU shift
        self.DU[:-1] = self.DU[1:]
        self.DU[-1] = 0.0

        # 2. CPU → GPU
        x0_jax = self._to_jax(current_state)
        DU_jax = self._to_jax(self.DU)
        u_prev_jax = self._to_jax(self._u_prev)
        ref_jax = self._to_jax(reference_trajectory)

        # Adaptive temperature
        current_lambda = self._get_current_lambda()
        self._gpu_kernel.update_lambda(current_lambda)

        # 3. GPU smooth_mppi_step
        DU_new_jax, gpu_info = self._gpu_kernel.smooth_mppi_step(
            x0_jax, DU_jax, u_prev_jax, ref_jax,
            self._R_jerk_jax, self._jerk_weight,
        )

        # 4. GPU → CPU
        self.DU = to_numpy(DU_new_jax)
        self.U = self._u_prev[np.newaxis, :] + np.cumsum(self.DU, axis=0)
        self.U = self.dynamics.clip_controls(self.U[np.newaxis, :, :])[0]

        u_opt = self.U[0].copy()
        self._u_prev = u_opt.copy()
        solve_time = time.perf_counter() - solve_start

        # info 변환
        costs = to_numpy(gpu_info["costs"])
        weights = to_numpy(gpu_info["weights"])
        trajectories = to_numpy(gpu_info["trajectories"])
        best_idx = int(to_numpy(gpu_info["best_idx"]))
        ess = float(to_numpy(gpu_info["ess"]))
        weighted_traj = to_numpy(gpu_info["weighted_trajectory"])

        if self._adaptive_temp is not None:
            current_lambda = self._adaptive_temp.update(ess, self.params.K)

        du_norm = float(np.mean(np.abs(self.DU)))

        self._iteration_count += 1
        logger.info(
            f"SMPPI-GPU iteration {self._iteration_count}: "
            f"solve_time={solve_time*1000:.2f}ms, "
            f"min_cost={costs[best_idx]:.4f}, "
            f"du_norm={du_norm:.4f}, "
            f"ESS={ess:.1f}/{self.params.K}"
        )

        info = {
            "predicted_trajectory": weighted_traj,
            "predicted_controls": self.U.copy(),
            "cost": float(costs[best_idx]),
            "mean_cost": float(np.mean(costs)),
            "solve_time": solve_time,
            "solver_status": "optimal",
            "sample_trajectories": trajectories,
            "sample_weights": weights,
            "sample_costs": costs,
            "best_trajectory": trajectories[best_idx],
            "best_index": best_idx,
            "ess": ess,
            "temperature": current_lambda,
            "delta_u_sequence": self.DU.copy(),
            "delta_u_norm": du_norm,
            "u_prev": self._u_prev.copy(),
            "backend": "gpu",
        }

        return u_opt, info

    def compute_control(
        self,
        current_state: np.ndarray,
        reference_trajectory: np.ndarray,
    ) -> Tuple[np.ndarray, dict]:
        """Δu space에서 최적화 후 cumsum으로 u 복원."""
        if self._use_gpu:
            return self._compute_control_gpu(current_state, reference_trajectory)

        solve_start = time.perf_counter()

        N = self.params.N
        K = self.params.K
        nu = self.dynamics.nu

        # 1. 이전 DU 시퀀스 shift
        self.DU[:-1] = self.DU[1:]
        self.DU[-1] = 0.0

        # 2. Δu space에서 노이즈 샘플링
        delta_noise = self.sampler.sample(K, N, nu)  # (K, N, nu)

        # 3. Δu 시퀀스에 노이즈 추가
        perturbed_du = self.DU[np.newaxis, :, :] + delta_noise  # (K, N, nu)

        # 4. cumsum으로 u 시퀀스 복원: u[t] = u_prev + Σ_{i=0}^{t} Δu[i]
        u_sequences = self._u_prev[np.newaxis, np.newaxis, :] + np.cumsum(
            perturbed_du, axis=1
        )  # (K, N, nu)

        # 5. 클리핑
        u_sequences = self.dynamics.clip_controls(u_sequences)

        # 6. 배치 rollout
        trajectories = self.dynamics.rollout_batch(
            current_state, u_sequences, self.params.dt
        )  # (K, N+1, nx)

        # 7. 비용 계산 (기본 cost + jerk cost)
        costs = self.cost.compute(
            trajectories, u_sequences, reference_trajectory
        )  # (K,)

        # Jerk cost: ‖ΔΔu‖² = ‖Δu[t+1] - Δu[t]‖²
        if self._jerk_weight > 0 and N > 1:
            ddu = perturbed_du[:, 1:, :] - perturbed_du[:, :-1, :]  # (K, N-1, nu)
            jerk_cost = self._jerk_weight * np.sum(
                ddu ** 2 * self._R_jerk[np.newaxis, np.newaxis, :],
                axis=(1, 2),
            )  # (K,)
            costs += jerk_cost

        # 8. Softmax 가중치
        current_lambda = self._get_current_lambda()
        weights = self._compute_weights(costs)  # (K,)

        # 9. Δu space에서 가중 평균 업데이트
        weighted_delta_noise = weights[:, np.newaxis, np.newaxis] * delta_noise
        self.DU += np.sum(weighted_delta_noise, axis=0)

        # 10. 최적 u 시퀀스 복원
        self.U = self._u_prev[np.newaxis, :] + np.cumsum(self.DU, axis=0)
        self.U = self.dynamics.clip_controls(self.U[np.newaxis, :, :])[0]

        # 11. 최적 제어 추출
        u_opt = self.U[0].copy()

        # u_prev 업데이트 (다음 스텝의 cumsum 기준점)
        self._u_prev = u_opt.copy()

        solve_time = time.perf_counter() - solve_start

        # 가중 평균 궤적 계산
        weighted_traj = np.sum(
            weights[:, np.newaxis, np.newaxis] * trajectories, axis=0
        )

        # 최적 샘플 인덱스
        best_idx = np.argmin(costs)

        # ESS
        ess = effective_sample_size(weights)

        # Adaptive Temperature 업데이트
        if self._adaptive_temp is not None:
            current_lambda = self._adaptive_temp.update(ess, K)

        # Δu 크기 계산 (smooth 효과 측정)
        du_norm = float(np.mean(np.abs(self.DU)))

        # 로깅
        self._iteration_count += 1
        logger.info(
            f"SMPPI iteration {self._iteration_count}: "
            f"solve_time={solve_time*1000:.2f}ms, "
            f"min_cost={costs[best_idx]:.4f}, "
            f"du_norm={du_norm:.4f}, "
            f"ESS={ess:.1f}/{K}"
        )

        info = {
            "predicted_trajectory": weighted_traj,
            "predicted_controls": self.U.copy(),
            "cost": float(costs[best_idx]),
            "mean_cost": float(np.mean(costs)),
            "solve_time": solve_time,
            "solver_status": "optimal",
            "sample_trajectories": trajectories,
            "sample_weights": weights,
            "sample_costs": costs,
            "best_trajectory": trajectories[best_idx],
            "best_index": best_idx,
            "ess": ess,
            "temperature": current_lambda,
            # SMPPI 전용
            "delta_u_sequence": self.DU.copy(),
            "delta_u_norm": du_norm,
            "u_prev": self._u_prev.copy(),
        }

        return u_opt, info

    def reset(self) -> None:
        """제어열 및 Δu 시퀀스 초기화."""
        super().reset()
        self.DU = np.zeros((self.params.N, self.dynamics.nu))
        self._u_prev = np.zeros(self.dynamics.nu)
