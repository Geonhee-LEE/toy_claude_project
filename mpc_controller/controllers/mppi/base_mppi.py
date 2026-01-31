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
    ObstacleCost,
)
from mpc_controller.controllers.mppi.sampling import GaussianSampler
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
        self.sampler = GaussianSampler(self.params.noise_sigma, seed=seed)

        # 비용 함수 구성
        self.cost = CompositeMPPICost()
        self.cost.add(StateTrackingCost(self.params.Q))
        self.cost.add(TerminalCost(self.params.Qf))
        self.cost.add(ControlEffortCost(self.params.R))

        if obstacles is not None and len(obstacles) > 0:
            self.cost.add(ObstacleCost(obstacles))

        # 제어열 초기화 (warm start)
        self.U = np.zeros((self.params.N, self.dynamics.nu))

        # 로깅 설정
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)

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
        weights = softmax_weights(costs, self.params.lambda_)  # (K,)

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
            "temperature": self.params.lambda_,
        }

        return u_opt, info

    def set_obstacles(self, obstacles: np.ndarray) -> None:
        """장애물 목록 업데이트.

        Args:
            obstacles: (M, 3) 장애물 배열 [x, y, radius]
        """
        # 비용 함수 재구성
        self.cost = CompositeMPPICost()
        self.cost.add(StateTrackingCost(self.params.Q))
        self.cost.add(TerminalCost(self.params.Qf))
        self.cost.add(ControlEffortCost(self.params.R))
        if len(obstacles) > 0:
            self.cost.add(ObstacleCost(obstacles))

    def reset(self) -> None:
        """제어열 초기화 및 반복 횟수 리셋."""
        self.U = np.zeros((self.params.N, self.dynamics.nu))
        self._iteration_count = 0
        logger.debug("MPPI controller reset")
