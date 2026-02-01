"""Spline-MPPI 컨트롤러 — B-spline 보간 기반 smooth sampling.

ICRA 2024 논문 기반. P개 제어점(knot)에만 노이즈를 부여하고,
B-spline basis matrix로 N개 시점으로 보간하여 구조적으로 부드러운 제어를 생성.

┌──────────────────────────────────────────────────────────────┐
│ Vanilla:  noise (K, N, nu) → 직접 perturb (N개 독립 노이즈)  │
│                                                              │
│ Spline:   noise (K, P, nu) → basis (N,P) @ knots            │
│           → interpolated (K, N, nu) smooth                   │
│           (P ≈ 8 << N = 20)                                  │
│                                                              │
│ B-spline basis: 순수 NumPy 구현 (scipy 미사용, NFR-1)        │
│                                                              │
│  알고리즘:                                                    │
│   1. P개 knot에 노이즈 샘플링 (K, P, nu)                    │
│   2. B-spline basis (N, P) 행렬로 보간                       │
│   3. 보간된 (K, N, nu) 제어로 rollout/cost                   │
│   4. Knot space에서 가중 평균 업데이트                       │
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


def _bspline_basis(N: int, P: int, degree: int = 3) -> np.ndarray:
    """균일 B-spline basis matrix 계산 (순수 NumPy, scipy 미사용).

    de Boor 재귀 알고리즘으로 B-spline basis를 생성.
    균일(uniform) 노트 벡터 사용.

    Args:
        N: 보간 시점 수 (출력 차원)
        P: 제어점(knot) 수
        degree: B-spline 차수 (기본 3 = cubic)

    Returns:
        (N, P) basis matrix — 각 행의 합 ≈ 1
    """
    k = degree
    # 균일 knot 벡터: clamped (양 끝 고정)
    # clamped B-spline: 처음 k+1개 = 0, 마지막 k+1개 = 1
    # 내부 knot 수 = P - k - 1
    n_knots = P + k + 1
    knots = np.zeros(n_knots)
    n_internal = P - k - 1
    if n_internal > 0:
        internal = np.linspace(0, 1, n_internal + 2)[1:-1]
        knots[k + 1 : k + 1 + n_internal] = internal
    knots[P:] = 1.0

    # 평가 지점
    t = np.linspace(0, 1, N)
    # 마지막 점이 정확히 1이면 마지막 basis에 포함되도록 약간 줄임
    t[-1] = 1.0 - 1e-10

    # de Boor 재귀
    basis = np.zeros((N, P))

    # Degree 0
    B = np.zeros((N, n_knots - 1))
    for i in range(n_knots - 1):
        mask = (t >= knots[i]) & (t < knots[i + 1])
        B[:, i] = mask.astype(float)

    # 재귀적으로 degree를 올림
    for d in range(1, k + 1):
        B_new = np.zeros((N, n_knots - 1 - d))
        for i in range(n_knots - 1 - d):
            denom1 = knots[i + d] - knots[i]
            denom2 = knots[i + d + 1] - knots[i + 1]
            term1 = np.zeros(N)
            term2 = np.zeros(N)
            if denom1 > 1e-12:
                term1 = (t - knots[i]) / denom1 * B[:, i]
            if denom2 > 1e-12:
                term2 = (knots[i + d + 1] - t) / denom2 * B[:, i + 1]
            B_new[:, i] = term1 + term2
        B = B_new

    basis = B[:, :P]

    # 행 정규화 (수치 오차 보정)
    row_sums = basis.sum(axis=1, keepdims=True)
    row_sums = np.maximum(row_sums, 1e-12)
    basis = basis / row_sums

    return basis


class SplineMPPIController(MPPIController):
    """Spline-MPPI — B-spline 보간 기반 smooth sampling.

    P개 knot에만 노이즈 → B-spline basis로 N개 보간.
    P << N이므로 노이즈 차원이 줄어들어 자연스럽게 부드러운 제어 생성.
    """

    def __init__(
        self,
        robot_params: RobotParams | None = None,
        mppi_params: MPPIParams | None = None,
        seed: Optional[int] = None,
        obstacles: Optional[np.ndarray] = None,
    ):
        super().__init__(robot_params, mppi_params, seed, obstacles)

        self._P = self.params.spline_num_knots
        self._degree = self.params.spline_degree

        # Knot 노이즈 표준편차 (기본: noise_sigma 재사용)
        self._knot_sigma = self.params.spline_knot_sigma
        if self._knot_sigma is None:
            self._knot_sigma = self.params.noise_sigma

        # B-spline basis 사전 계산 (N, P) — 변경되지 않으므로 캐싱
        self._basis = _bspline_basis(self.params.N, self._P, self._degree)

        # Knot space warm-start
        self.U_knots = np.zeros((self._P, self.dynamics.nu))

        # Knot RNG (sampler와 독립)
        self._knot_rng = np.random.default_rng(
            seed + 1000 if seed is not None else None
        )

    def compute_control(
        self,
        current_state: np.ndarray,
        reference_trajectory: np.ndarray,
    ) -> Tuple[np.ndarray, dict]:
        """Knot space에서 최적화 후 B-spline 보간."""
        solve_start = time.perf_counter()

        N = self.params.N
        K = self.params.K
        nu = self.dynamics.nu
        P = self._P

        # 1. Knot 시퀀스 shift
        self.U_knots[:-1] = self.U_knots[1:]
        self.U_knots[-1] = 0.0

        # 2. Knot space에서 노이즈 샘플링 (K, P, nu)
        knot_noise = self._knot_rng.standard_normal((K, P, nu))
        knot_noise *= self._knot_sigma[np.newaxis, np.newaxis, :]

        # 3. Knot perturb
        perturbed_knots = self.U_knots[np.newaxis, :, :] + knot_noise  # (K, P, nu)

        # 4. B-spline 보간: (K, P, nu) → (K, N, nu)
        # basis: (N, P), perturbed_knots: (K, P, nu)
        # np.einsum("np,kpd->knd", basis, knots)
        perturbed_controls = np.einsum(
            "np,kpd->knd", self._basis, perturbed_knots
        )  # (K, N, nu)

        # 5. 클리핑
        perturbed_controls = self.dynamics.clip_controls(perturbed_controls)

        # 6. 배치 rollout
        trajectories = self.dynamics.rollout_batch(
            current_state, perturbed_controls, self.params.dt
        )  # (K, N+1, nx)

        # 7. 비용 계산
        costs = self.cost.compute(
            trajectories, perturbed_controls, reference_trajectory
        )  # (K,)

        # 8. Softmax 가중치
        current_lambda = self._get_current_lambda()
        weights = self._compute_weights(costs)  # (K,)

        # 9. Knot space에서 가중 평균 업데이트
        weighted_knot_noise = weights[:, np.newaxis, np.newaxis] * knot_noise
        self.U_knots += np.sum(weighted_knot_noise, axis=0)

        # 10. U 복원 (B-spline 보간)
        self.U = self._basis @ self.U_knots  # (N, nu)
        self.U = self.dynamics.clip_controls(self.U[np.newaxis, :, :])[0]

        # 11. 최적 제어 추출
        u_opt = self.U[0].copy()
        solve_time = time.perf_counter() - solve_start

        # 가중 평균 궤적
        weighted_traj = np.sum(
            weights[:, np.newaxis, np.newaxis] * trajectories, axis=0
        )

        best_idx = np.argmin(costs)
        ess = effective_sample_size(weights)

        if self._adaptive_temp is not None:
            current_lambda = self._adaptive_temp.update(ess, K)

        # 부드러움 측정: 제어 변화율
        du = np.diff(self.U, axis=0)
        control_rate = float(np.mean(np.abs(du))) if len(du) > 0 else 0.0

        self._iteration_count += 1
        logger.info(
            f"Spline-MPPI iteration {self._iteration_count}: "
            f"solve_time={solve_time*1000:.2f}ms, "
            f"min_cost={costs[best_idx]:.4f}, "
            f"knots={P}, control_rate={control_rate:.4f}, "
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
            # Spline-MPPI 전용
            "knot_controls": self.U_knots.copy(),
            "spline_basis": self._basis,
            "num_knots": P,
            "control_rate": control_rate,
        }

        return u_opt, info

    def reset(self) -> None:
        """제어열 및 knot 시퀀스 초기화."""
        super().reset()
        self.U_knots = np.zeros((self._P, self.dynamics.nu))
