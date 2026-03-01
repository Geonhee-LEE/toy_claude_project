"""C++ pybind11 컴포넌트를 수동 조합한 MPPI 파이프라인.

compute_control(state, ref) -> (control, info) 인터페이스를 준수하며,
Python 순수 구현과 동일한 알고리즘을 C++ 컴포넌트로 실행한다.
"""

import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from mpc_controller_ros2.mppi import (
    MPPIParams,
    DiffDriveModel,
    SwerveDriveModel,
    NonCoaxialSwerveModel,
    BatchDynamicsWrapper,
    GaussianSampler,
    StateTrackingCost,
    TerminalCost,
    ControlEffortCost,
    ObstacleCost,
    VanillaMPPIWeights,
    LogMPPIWeights,
    TsallisMPPIWeights,
    RiskAwareMPPIWeights,
    AdaptiveTemperature,
    compute_ess,
)


# ─────────────────────────────────────────────────────────────
# 모델별 설정
# ─────────────────────────────────────────────────────────────

MODEL_CONFIG = {
    "diff_drive": {
        "nx": 3, "nu": 2,
        "Q": np.diag([10.0, 10.0, 1.0]),
        "R": np.diag([0.01, 0.01]),
        "Qf": np.diag([100.0, 100.0, 10.0]),
        "noise_sigma": np.array([0.3, 0.3]),
        "model_factory": lambda: DiffDriveModel(0.0, 1.0, -1.5, 1.5),
    },
    "swerve": {
        "nx": 3, "nu": 3,
        "Q": np.diag([10.0, 10.0, 1.0]),
        "R": np.diag([0.01, 0.01, 0.01]),
        "Qf": np.diag([100.0, 100.0, 10.0]),
        "noise_sigma": np.array([0.3, 0.2, 0.3]),
        "model_factory": lambda: SwerveDriveModel(-1.0, 1.0, 0.5, 1.5),
    },
    "non_coaxial_swerve": {
        "nx": 4, "nu": 3,
        "Q": np.diag([10.0, 10.0, 1.0, 0.1]),
        "R": np.diag([0.01, 0.01, 0.01]),
        "Qf": np.diag([100.0, 100.0, 10.0, 1.0]),
        "noise_sigma": np.array([0.3, 0.3, 0.2]),
        "model_factory": lambda: NonCoaxialSwerveModel(0.0, 1.0, 1.5, 2.0),
    },
    "non_coaxial_swerve_60deg": {
        "nx": 4, "nu": 3,
        "Q": np.diag([10.0, 10.0, 5.0, 0.1]),
        "R": np.diag([0.01, 0.01, 0.005]),
        "Qf": np.diag([100.0, 100.0, 50.0, 1.0]),
        "noise_sigma": np.array([0.3, 0.3, 0.4]),
        "model_factory": lambda: NonCoaxialSwerveModel(0.0, 1.0, 1.5, 2.0, np.pi / 3.0),
    },
}


# ─────────────────────────────────────────────────────────────
# 가중치 전략 팩토리
# ─────────────────────────────────────────────────────────────

WEIGHT_FACTORIES = {
    "vanilla":    lambda: VanillaMPPIWeights(),
    "log":        lambda: LogMPPIWeights(),
    "tsallis":    lambda: TsallisMPPIWeights(1.1),
    "risk_aware": lambda: RiskAwareMPPIWeights(0.7),
}


class CppMPPIAssembler:
    """C++ pybind11 컴포넌트를 수동 조합한 MPPI 솔버.

    Python MPPIController와 동일한 7-step 알고리즘:
      1. Shift U (warmstart)
      2. C++ GaussianSampler.sample()
      3. Perturb + C++ clipControls
      4. C++ BatchDynamicsWrapper.rolloutBatch()
      5. 개별 C++ cost 함수 합산
      6. C++ WeightComputation.compute()
      7. Weighted mean update

    Parameters
    ----------
    model_type : str
        "diff_drive", "swerve", "non_coaxial_swerve"
    weight_type : str
        "vanilla", "log", "tsallis", "risk_aware"
    K : int
        샘플 수
    N : int
        호라이즌 스텝 수
    dt : float
        시간 간격
    lambda_ : float
        온도 파라미터
    adaptive_temperature : bool
        ESS 기반 λ 자동 튜닝
    seed : int
        랜덤 시드
    obstacles : Optional[np.ndarray]
        장애물 (M, 3) [x, y, radius]
    max_steering_angle : Optional[float]
        NonCoaxialSwerve 스티어링 제한 (rad). None이면 기본값 사용.
    """

    def __init__(
        self,
        model_type: str = "diff_drive",
        weight_type: str = "vanilla",
        K: int = 512,
        N: int = 20,
        dt: float = 0.05,
        lambda_: float = 10.0,
        adaptive_temperature: bool = True,
        seed: int = 42,
        obstacles: Optional[np.ndarray] = None,
        max_steering_angle: Optional[float] = None,
    ):
        # max_steering_angle 지정 시 자동으로 60deg config 선택
        effective_type = model_type
        if max_steering_angle is not None and model_type == "non_coaxial_swerve":
            if abs(max_steering_angle - np.pi / 3.0) < 0.01:
                effective_type = "non_coaxial_swerve_60deg"
        cfg = MODEL_CONFIG.get(effective_type, MODEL_CONFIG[model_type])
        self.model_type = model_type
        self.weight_type = weight_type
        self.nx = cfg["nx"]
        self.nu = cfg["nu"]
        self.K = K
        self.N = N
        self.dt = dt
        self.lambda_ = lambda_

        # C++ 모션 모델
        if max_steering_angle is not None and model_type == "non_coaxial_swerve" and effective_type == model_type:
            self.model = NonCoaxialSwerveModel(0.0, 1.0, 1.5, 2.0, max_steering_angle)
        else:
            self.model = cfg["model_factory"]()

        # C++ MPPIParams (BatchDynamicsWrapper 생성용)
        self._params = MPPIParams()
        self._params.N = N
        self._params.K = K
        self._params.dt = dt
        self._params.lambda_ = lambda_
        self._params.motion_model = model_type
        self._params.Q = cfg["Q"]
        self._params.R = cfg["R"]
        self._params.Qf = cfg["Qf"]
        self._params.noise_sigma = cfg["noise_sigma"]

        # C++ 컴포넌트 조립
        self.dynamics = BatchDynamicsWrapper(self._params, self.model)
        self.sampler = GaussianSampler(cfg["noise_sigma"], seed=seed)
        self.weight_fn = WEIGHT_FACTORIES[weight_type]()

        # 개별 cost 함수 리스트
        self._costs: List = []
        self._costs.append(("state_tracking", StateTrackingCost(cfg["Q"])))
        self._costs.append(("terminal", TerminalCost(cfg["Qf"])))
        self._costs.append(("control_effort", ControlEffortCost(cfg["R"])))

        # 장애물
        self._obstacle_cost = None
        if obstacles is not None and len(obstacles) > 0:
            self._obstacle_cost = ObstacleCost(10.0, 0.5)
            obs_list = [obs for obs in obstacles]
            self._obstacle_cost.setObstacles(obs_list)
            self._costs.append(("obstacle", self._obstacle_cost))

        # Adaptive temperature
        self._adaptive = adaptive_temperature
        self._adaptive_temp = None
        if adaptive_temperature:
            self._adaptive_temp = AdaptiveTemperature(
                lambda_, 0.5, 1.0, 0.001, 100.0
            )

        # 제어열 초기화
        self.U = np.zeros((N, self.nu))

    @property
    def params(self):
        """Python MPPIController 호환 속성."""
        return self._params

    def reset(self):
        """제어열 + adaptive temperature 초기화."""
        self.U = np.zeros((self.N, self.nu))
        if self._adaptive_temp is not None:
            self._adaptive_temp.reset(self.lambda_)

    def set_obstacles(self, obstacles: np.ndarray):
        """장애물 동적 업데이트."""
        if self._obstacle_cost is None:
            self._obstacle_cost = ObstacleCost(10.0, 0.5)
            self._costs.append(("obstacle", self._obstacle_cost))
        obs_list = [obs for obs in obstacles]
        self._obstacle_cost.setObstacles(obs_list)

    def compute_control(
        self,
        state: np.ndarray,
        reference_trajectory: np.ndarray,
    ) -> Tuple[np.ndarray, dict]:
        """MPPI 최적 제어 계산.

        Parameters
        ----------
        state : (nx,) 현재 상태
        reference_trajectory : (N+1, nx) 참조 궤적

        Returns
        -------
        u_opt : (nu,) 최적 제어
        info : dict 메트릭
        """
        t_start = time.perf_counter()

        # 1. Shift U (warmstart)
        self.U[:-1] = self.U[1:]
        self.U[-1] = 0.0

        # 2. C++ 샘플링 → list[K × (N, nu)]
        noise_list = self.sampler.sample(self.K, self.N, self.nu)

        # list → (K, N, nu) 배열
        noise = np.array(noise_list)  # (K, N, nu)

        # 3. Perturb + clip
        perturbed = self.U[np.newaxis, :, :] + noise  # (K, N, nu)
        perturbed_list = [perturbed[k] for k in range(self.K)]
        clipped_list = [self.model.clipControls(p[np.newaxis]) for p in perturbed.reshape(-1, self.nu)]
        # 보다 효율적: 배치로 clip
        perturbed_flat = perturbed.reshape(-1, self.nu)
        clipped_flat = self.model.clipControls(perturbed_flat)
        perturbed = clipped_flat.reshape(self.K, self.N, self.nu)
        perturbed_list = [perturbed[k] for k in range(self.K)]

        # 4. C++ rollout → list[K × (N+1, nx)]
        trajs_list = self.dynamics.rolloutBatch(state, perturbed_list, self.dt)

        # 5. 비용 계산 (개별 cost 합산)
        costs = np.zeros(self.K)
        for name, cost_fn in self._costs:
            costs += cost_fn.compute(trajs_list, perturbed_list, reference_trajectory)

        # 6. 가중치 계산
        current_lambda = self.lambda_
        if self._adaptive_temp is not None:
            current_lambda = self._adaptive_temp.getLambda()
        weights = self.weight_fn.compute(costs, current_lambda)

        # 7. 제어열 업데이트
        # noise = perturbed - U → 실제 적용된 노이즈
        applied_noise = perturbed - self.U[np.newaxis, :, :]
        weighted_noise = weights[:, np.newaxis, np.newaxis] * applied_noise
        self.U += np.sum(weighted_noise, axis=0)

        # clip 최종 제어열
        U_flat = self.model.clipControls(self.U.reshape(-1, self.nu))
        self.U = U_flat.reshape(self.N, self.nu)

        # 최적 제어 추출
        u_opt = self.U[0].copy()

        # ESS 계산
        ess = float(compute_ess(weights))

        # Adaptive temperature 업데이트
        if self._adaptive_temp is not None:
            current_lambda = self._adaptive_temp.update(ess, self.K)

        # 예측 궤적 (가중 평균)
        trajs_arr = np.array(trajs_list)  # (K, N+1, nx)
        predicted_traj = np.sum(
            weights[:, np.newaxis, np.newaxis] * trajs_arr, axis=0
        )

        best_idx = int(np.argmin(costs))

        solve_time = time.perf_counter() - t_start

        info = {
            "predicted_trajectory": predicted_traj,
            "predicted_controls": self.U.copy(),
            "cost": float(costs[best_idx]),
            "mean_cost": float(np.mean(costs)),
            "solve_time": solve_time,
            "sample_trajectories": trajs_arr,
            "sample_weights": weights,
            "sample_costs": costs,
            "best_trajectory": trajs_arr[best_idx],
            "best_index": best_idx,
            "ess": ess,
            "temperature": current_lambda,
            "backend": "cpp",
        }

        return u_opt, info
