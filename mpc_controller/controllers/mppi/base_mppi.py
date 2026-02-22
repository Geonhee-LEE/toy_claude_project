"""Vanilla MPPI 핵심 알고리즘."""

import logging
import time
from typing import Optional, Tuple

import numpy as np

from mpc_controller.models.differential_drive import DifferentialDriveModel, RobotParams
from mpc_controller.controllers.mppi.mppi_params import MPPIParams
from mpc_controller.controllers.mppi.dynamics_wrapper import BatchDynamicsWrapper
from mpc_controller.controllers.mppi.cost_functions import (
    CompositeMPPICost,
    StateTrackingCost,
    TerminalCost,
    ControlEffortCost,
    ControlRateCost,
    ObstacleCost,
)
from mpc_controller.controllers.mppi.sampling import GaussianSampler, ColoredNoiseSampler
from mpc_controller.controllers.mppi.adaptive_temperature import AdaptiveTemperature
from mpc_controller.controllers.mppi.utils import softmax_weights, effective_sample_size

logger = logging.getLogger(__name__)


class MPPIController:
    """Vanilla MPPI 컨트롤러.

    compute_control(state, ref) -> (control, info) 인터페이스로
    기존 MPCController와 교체 가능.

    알고리즘 흐름:
      1. 이전 제어열 shift
      2. 노이즈 샘플링 (K, N, nu)
      3. 제어열에 노이즈 추가 → 클리핑
      4. 배치 rollout (K, N+1, nx)
      5. 비용 계산 (K,)
      6. Softmax 가중치 계산
      7. 가중 평균으로 제어열 업데이트
      8. 첫 번째 제어 반환
    """

    def __init__(
        self,
        robot_params: RobotParams | None = None,
        mppi_params: MPPIParams | None = None,
        seed: Optional[int] = None,
        obstacles: Optional[np.ndarray] = None,
    ):
        """
        Args:
            robot_params: 로봇 물리 파라미터
            mppi_params: MPPI 튜닝 파라미터
            seed: 랜덤 시드
            obstacles: (M, 3) 장애물 배열 [x, y, radius]
        """
        self.robot_params = robot_params or RobotParams()
        self.params = mppi_params or MPPIParams()
        self._iteration_count = 0

        # 구성 요소 초기화
        self.dynamics = BatchDynamicsWrapper(self.robot_params)

        # 샘플러 선택: Colored Noise 또는 Gaussian
        if self.params.colored_noise:
            self.sampler = ColoredNoiseSampler(
                self.params.noise_sigma,
                beta=self.params.noise_beta,
                seed=seed,
            )
        else:
            self.sampler = GaussianSampler(self.params.noise_sigma, seed=seed)

        # Adaptive Temperature (옵트인)
        self._adaptive_temp = None
        if self.params.adaptive_temperature:
            config = self.params.adaptive_temp_config or {}
            self._adaptive_temp = AdaptiveTemperature(
                initial_lambda=self.params.lambda_,
                **config,
            )

        # 비용 함수 구성
        self.cost = CompositeMPPICost()
        self.cost.add(StateTrackingCost(self.params.Q))
        self.cost.add(TerminalCost(self.params.Qf))
        self.cost.add(ControlEffortCost(self.params.R))

        if self.params.R_rate is not None:
            self.cost.add(ControlRateCost(self.params.R_rate))

        if obstacles is not None and len(obstacles) > 0:
            self.cost.add(ObstacleCost(obstacles))

        # 제어열 초기화 (warm start)
        self.U = np.zeros((self.params.N, self.dynamics.nu))

        # GPU 가속 초기화 (use_gpu=False 기본 → 기존 동작 100% 유지)
        self._use_gpu = False
        if self.params.use_gpu:
            from mpc_controller.controllers.mppi.gpu_backend import is_jax_available
            if is_jax_available():
                self._use_gpu = True
                self._init_gpu()
            else:
                logger.warning(
                    "use_gpu=True but JAX not available. Falling back to CPU."
                )

        # 로깅 설정
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)

    def _init_gpu(self):
        """GPU 커널 초기화."""
        from mpc_controller.controllers.mppi.gpu_mppi_kernel import GPUMPPIKernel
        from mpc_controller.controllers.mppi.gpu_backend import to_jax, get_backend_name

        r_rate_diag = None
        if self.params.R_rate is not None:
            r_rate_diag = (
                np.diag(self.params.R_rate)
                if self.params.R_rate.ndim == 2
                else self.params.R_rate
            )

        self._gpu_kernel = GPUMPPIKernel(
            N=self.params.N,
            K=self.params.K,
            nu=self.dynamics.nu,
            nx=self.dynamics.nx,
            dt=self.params.dt,
            lambda_=self.params.lambda_,
            noise_sigma=self.params.noise_sigma,
            q_diag=np.diag(self.params.Q),
            qf_diag=np.diag(self.params.Qf),
            r_diag=np.diag(self.params.R),
            max_velocity=self.robot_params.max_velocity,
            max_omega=self.robot_params.max_omega,
            r_rate_diag=r_rate_diag,
            colored_noise=self.params.colored_noise,
            noise_beta=self.params.noise_beta,
            model_name="diff_drive",
            use_float32=self.params.gpu_float32,
            warmup=self.params.gpu_warmup,
        )
        self._to_jax = to_jax
        logger.info(f"GPU MPPI initialized (backend: {get_backend_name()})")

    def _compute_control_gpu(
        self,
        current_state: np.ndarray,
        reference_trajectory: np.ndarray,
    ) -> Tuple[np.ndarray, dict]:
        """GPU 경로: 전체 MPPI 스텝을 GPU에서 실행.

        CPU↔GPU 전송 2회만 발생 (입력 전송 + 결과 반환).
        """
        from mpc_controller.controllers.mppi.gpu_backend import to_numpy

        solve_start = time.perf_counter()

        # 1. 이전 제어열 shift
        self.U[:-1] = self.U[1:]
        self.U[-1] = 0.0

        # 2. CPU → GPU 전송 (1회)
        x0_jax = self._to_jax(current_state)
        U_jax = self._to_jax(self.U)
        ref_jax = self._to_jax(reference_trajectory)

        # Adaptive temperature 반영
        current_lambda = self._get_current_lambda()
        self._gpu_kernel.update_lambda(current_lambda)

        # 3. GPU에서 MPPI 스텝 실행
        U_new_jax, gpu_info = self._gpu_kernel.mppi_step(x0_jax, U_jax, ref_jax)

        # 4. GPU → CPU 전송 (1회)
        self.U = to_numpy(U_new_jax)
        u_opt = self.U[0].copy()
        solve_time = time.perf_counter() - solve_start

        # info dict 변환 (JAX → NumPy)
        costs = to_numpy(gpu_info["costs"])
        weights = to_numpy(gpu_info["weights"])
        trajectories = to_numpy(gpu_info["trajectories"])
        best_idx = int(to_numpy(gpu_info["best_idx"]))
        ess = float(to_numpy(gpu_info["ess"]))
        weighted_traj = to_numpy(gpu_info["weighted_trajectory"])

        # Adaptive Temperature 업데이트
        if self._adaptive_temp is not None:
            current_lambda = self._adaptive_temp.update(ess, self.params.K)

        # 로깅
        self._iteration_count += 1
        logger.info(
            f"MPPI-GPU iteration {self._iteration_count}: "
            f"solve_time={solve_time*1000:.2f}ms, "
            f"min_cost={costs[best_idx]:.4f}, "
            f"mean_cost={np.mean(costs):.4f}, "
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
            "backend": "gpu",
        }

        return u_opt, info

    def compute_control(
        self,
        current_state: np.ndarray,
        reference_trajectory: np.ndarray,
    ) -> Tuple[np.ndarray, dict]:
        """최적 제어 입력 계산.

        Args:
            current_state: 현재 로봇 상태 [x, y, theta]
            reference_trajectory: 참조 궤적 (N+1, 3)

        Returns:
            (control, info) 튜플:
              - control: [v, omega] 최적 제어
              - info: 디버그/시각화 정보
        """
        # GPU 분기: use_gpu=True이고 JAX 사용 가능 시 GPU 경로 실행
        if self._use_gpu:
            return self._compute_control_gpu(current_state, reference_trajectory)

        solve_start = time.perf_counter()

        N = self.params.N
        K = self.params.K
        nu = self.dynamics.nu

        # 1. 이전 제어열 shift (한 스텝 앞으로)
        self.U[:-1] = self.U[1:]
        self.U[-1] = 0.0

        # 2. 노이즈 샘플링
        noise = self.sampler.sample(K, N, nu)  # (K, N, nu)

        # 3. 제어열에 노이즈 추가
        perturbed_controls = self.U[np.newaxis, :, :] + noise  # (K, N, nu)
        perturbed_controls = self.dynamics.clip_controls(perturbed_controls)

        # 4. 배치 rollout
        trajectories = self.dynamics.rollout_batch(
            current_state, perturbed_controls, self.params.dt
        )  # (K, N+1, nx)

        # 5. 비용 계산
        costs = self.cost.compute(
            trajectories, perturbed_controls, reference_trajectory
        )  # (K,)

        # 6. Softmax 가중치
        current_lambda = self._get_current_lambda()
        weights = self._compute_weights(costs)  # (K,)

        # 7. 가중 평균으로 제어열 업데이트
        # (K, 1, 1) * (K, N, nu) -> sum -> (N, nu)
        weighted_noise = weights[:, np.newaxis, np.newaxis] * noise
        self.U += np.sum(weighted_noise, axis=0)
        self.U = self.dynamics.clip_controls(
            self.U[np.newaxis, :, :]
        )[0]

        # 8. 최적 제어 추출
        u_opt = self.U[0].copy()
        solve_time = time.perf_counter() - solve_start

        # 가중 평균 궤적 계산
        weighted_traj = np.sum(
            weights[:, np.newaxis, np.newaxis] * trajectories, axis=0
        )  # (N+1, nx)

        # 최적 샘플 인덱스
        best_idx = np.argmin(costs)

        # ESS 계산
        ess = effective_sample_size(weights)

        # Adaptive Temperature 업데이트
        if self._adaptive_temp is not None:
            current_lambda = self._adaptive_temp.update(ess, K)

        # 로깅
        self._iteration_count += 1
        logger.info(
            f"MPPI iteration {self._iteration_count}: "
            f"solve_time={solve_time*1000:.2f}ms, "
            f"min_cost={costs[best_idx]:.4f}, "
            f"mean_cost={np.mean(costs):.4f}, "
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
        }

        return u_opt, info

    def set_obstacles(self, obstacles: np.ndarray) -> None:
        """장애물 목록 업데이트.

        Args:
            obstacles: (M, 3) 장애물 배열 [x, y, radius]
        """
        # GPU 경로: GPU 커널 장애물 업데이트
        if self._use_gpu:
            self._gpu_kernel.set_obstacles(obstacles)
            return

        # 비용 함수 재구성
        self.cost = CompositeMPPICost()
        self.cost.add(StateTrackingCost(self.params.Q))
        self.cost.add(TerminalCost(self.params.Qf))
        self.cost.add(ControlEffortCost(self.params.R))
        if self.params.R_rate is not None:
            self.cost.add(ControlRateCost(self.params.R_rate))
        if len(obstacles) > 0:
            self.cost.add(ObstacleCost(obstacles))

    def _get_current_lambda(self) -> float:
        """현재 온도 파라미터 반환 (adaptive 적용 시 동적 값)."""
        if self._adaptive_temp is not None:
            return self._adaptive_temp.lambda_
        return self.params.lambda_

    def _compute_weights(self, costs: np.ndarray) -> np.ndarray:
        """비용에서 가중치 계산. 서브클래스에서 오버라이드 가능."""
        return softmax_weights(costs, self._get_current_lambda())

    def reset(self) -> None:
        """제어열 초기화 및 반복 횟수 리셋."""
        self.U = np.zeros((self.params.N, self.dynamics.nu))
        self._iteration_count = 0
        logger.debug("MPPI controller reset")
