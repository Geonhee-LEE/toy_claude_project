"""Stein Variational MPPI (SVMPC) 컨트롤러 — SVGD 기반 샘플 다양성 유도.

Lambert et al. (2020) "Stein Variational Model Predictive Control" 기반 구현.

SVGD (Stein Variational Gradient Descent)로 샘플 간 상호작용을 통해
다중 모달 분포를 효과적으로 탐색한다.

┌──────────────────────────────────────────────────────────────┐
│ Vanilla MPPI:  sample → rollout → cost → weight → update    │
│                                                              │
│ SVMPC:         sample → rollout → cost ─┐                   │
│                                          │  SVGD Loop (L회)  │
│                                          ├→ softmax weights  │
│                                          ├→ RBF kernel (K×K) │
│                                          ├→ attractive force │
│                                          ├→ repulsive force  │
│                                          ├→ samples += ε·f   │
│                                          ├→ re-rollout       │
│                                          └→ re-cost          │
│                                         weight → update      │
└──────────────────────────────────────────────────────────────┘

SVGD update rule (gradient-free 근사):
  particles: (K, D) — flatten된 제어 시퀀스, D = N × nu
  매력력: attract_i = Σ_j w_j · k(x_j, x_i) · (x_j - x_i)
  반발력: repel_i  = (1/K) Σ_j ∇_{x_i} k(x_j, x_i)
  x_i ← x_i + ε · (attract_i + repel_i)

  svgd_num_iterations=0 → Vanilla 동등 (super() 위임)
  svgd_num_iterations=3 → SVGD 3회 반복 (기본 권장)
"""

import logging
import time
from typing import Optional, Tuple

import numpy as np

from mpc_controller.controllers.mppi.base_mppi import MPPIController
from mpc_controller.controllers.mppi.utils import (
    softmax_weights,
    effective_sample_size,
)

logger = logging.getLogger(__name__)


class SteinVariationalMPPIController(MPPIController):
    """Stein Variational MPPI — SVGD 커널 기반 샘플 다양성 유도.

    svgd_num_iterations=0 → Vanilla (super() 위임, 하위 호환)
    svgd_num_iterations>0 → SVGD 루프로 샘플 분포 개선 후 가중 평균
    """

    def compute_control(
        self,
        current_state: np.ndarray,
        reference_trajectory: np.ndarray,
    ) -> Tuple[np.ndarray, dict]:
        """최적 제어 입력 계산 — SVGD 루프 포함.

        svgd_num_iterations=0이면 Vanilla와 동일하게 super() 위임.

        Args:
            current_state: 현재 로봇 상태 [x, y, theta]
            reference_trajectory: 참조 궤적 (N+1, 3)

        Returns:
            (control, info) 튜플
        """
        # svgd_num_iterations=0 → Vanilla 동등
        if self.params.svgd_num_iterations == 0:
            return super().compute_control(current_state, reference_trajectory)

        solve_start = time.perf_counter()

        N = self.params.N
        K = self.params.K
        nu = self.dynamics.nu
        L = self.params.svgd_num_iterations

        # ──── Phase 1: Vanilla 동일 (shift, sample, perturb, rollout, cost) ────

        # 1. 이전 제어열 shift
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

        # 샘플 다양성 측정 (SVGD 전)
        diversity_before = self._compute_diversity(perturbed_controls)

        # ──── Phase 2: SVGD Loop ────
        # 최적화: pairwise diff (K,K,D) 텐서를 iteration당 1회만 계산하여
        # median_bandwidth, rbf_kernel, _svgd_update 간 공유.
        # 기존 3회 → 1회로 메모리 할당 ~67% 절감.

        D = N * nu  # flatten 차원

        for svgd_iter in range(L):
            # flatten: (K, N, nu) → (K, D)
            particles = perturbed_controls.reshape(K, D)

            # softmax 가중치 (현재 비용 기반)
            current_lambda = self._get_current_lambda()
            weights = softmax_weights(costs, current_lambda)  # (K,)

            # ── 핵심 최적화: pairwise diff 1회 계산 후 재사용 ──
            # diff[j, i] = x_j - x_i → (K, K, D)
            diff = particles[:, np.newaxis, :] - particles[np.newaxis, :, :]
            sq_dist = np.sum(diff ** 2, axis=-1)  # (K, K)

            # bandwidth (median heuristic 또는 고정값) — sq_dist 재사용
            if self.params.svgd_bandwidth is not None:
                h = self.params.svgd_bandwidth
            else:
                triu_idx = np.triu_indices(K, k=1)
                med = np.median(sq_dist[triu_idx])
                h = max(np.sqrt(med / (2.0 * np.log(K + 1))), 1e-6)

            # RBF 커널 — sq_dist 재사용
            kernel = np.exp(-sq_dist / (2.0 * h ** 2))  # (K, K)

            # SVGD force — diff, kernel 재사용 (추가 (K,K,D) 할당 없음)
            force = self._svgd_update(diff, weights, kernel, h, K)  # (K, D)

            # particles 업데이트
            particles = particles + self.params.svgd_step_size * force

            # unflatten: (K, D) → (K, N, nu)
            perturbed_controls = particles.reshape(K, N, nu)

            # 클리핑
            perturbed_controls = self.dynamics.clip_controls(perturbed_controls)

            # re-rollout
            trajectories = self.dynamics.rollout_batch(
                current_state, perturbed_controls, self.params.dt
            )

            # re-cost
            costs = self.cost.compute(
                trajectories, perturbed_controls, reference_trajectory
            )

        # 샘플 다양성 측정 (SVGD 후)
        diversity_after = self._compute_diversity(perturbed_controls)

        # ──── Phase 3: final weight, update U, return ────

        current_lambda = self._get_current_lambda()
        weights = self._compute_weights(costs)  # (K,)

        # effective noise 역산: perturbed_controls - U
        effective_noise = perturbed_controls - self.U[np.newaxis, :, :]

        # 가중 평균으로 제어열 업데이트
        weighted_noise = weights[:, np.newaxis, np.newaxis] * effective_noise
        self.U += np.sum(weighted_noise, axis=0)
        self.U = self.dynamics.clip_controls(
            self.U[np.newaxis, :, :]
        )[0]

        # 최적 제어 추출
        u_opt = self.U[0].copy()
        solve_time = time.perf_counter() - solve_start

        # 가중 평균 궤적
        weighted_traj = np.sum(
            weights[:, np.newaxis, np.newaxis] * trajectories, axis=0
        )

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
            f"SVMPC iteration {self._iteration_count}: "
            f"solve_time={solve_time*1000:.2f}ms, "
            f"min_cost={costs[best_idx]:.4f}, "
            f"svgd_iters={L}, "
            f"diversity={diversity_before:.4f}→{diversity_after:.4f}, "
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
            # SVGD 전용 info
            "svgd_iterations": L,
            "sample_diversity_before": diversity_before,
            "sample_diversity_after": diversity_after,
        }

        return u_opt, info

    @staticmethod
    def _svgd_update(
        diff: np.ndarray,
        weights: np.ndarray,
        kernel: np.ndarray,
        bandwidth: float,
        K: int,
    ) -> np.ndarray:
        """Gradient-free SVGD force 계산: attractive + repulsive.

        attractive_i = Σ_j w_j · k(x_j, x_i) · (x_j - x_i)
        repulsive_i  = (1/K) Σ_j k(x_j, x_i) · (x_j - x_i) / h²

        최적화: diff (K,K,D)를 외부에서 받아 재사용.
        repulsive도 einsum으로 중간 (K,K,D) 텐서 할당 제거.

        Args:
            diff: (K, K, D) pairwise 차이 (x_j - x_i), 사전 계산됨
            weights: (K,) softmax 가중치
            kernel: (K, K) RBF 커널 행렬
            bandwidth: h
            K: 샘플 수

        Returns:
            (K, D) SVGD force
        """
        # Attractive force: 저비용 샘플 방향으로 끌어당김
        # attract_i = Σ_j w_j · k(x_j, x_i) · (x_j - x_i)
        weighted_kernel = weights[:, np.newaxis] * kernel  # (K, K)
        attractive = np.einsum("ji,jid->id", weighted_kernel, diff)  # (K, D)

        # Repulsive force: 샘플 간 다양성 유지
        # repel_i = (1/K) Σ_j k(x_j, x_i) · (x_j - x_i) / h²
        # einsum으로 직접 축약 → 중간 (K,K,D) kernel_grad 텐서 제거
        repulsive = np.einsum("ji,jid->id", kernel, diff) / (K * bandwidth ** 2)

        return attractive + repulsive

    @staticmethod
    def _compute_diversity(controls: np.ndarray) -> float:
        """샘플 다양성 측정 (평균 pairwise L2 거리).

        Args:
            controls: (K, N, nu) 제어 시퀀스

        Returns:
            평균 pairwise 거리
        """
        K = controls.shape[0]
        if K <= 1:
            return 0.0
        flat = controls.reshape(K, -1)  # (K, D)
        # 효율성을 위해 무작위 서브샘플링 (K가 클 때)
        max_samples = min(K, 128)
        if K > max_samples:
            idx = np.random.choice(K, max_samples, replace=False)
            flat = flat[idx]
            K_sub = max_samples
        else:
            K_sub = K
        diff = flat[:, np.newaxis, :] - flat[np.newaxis, :, :]  # (K_sub, K_sub, D)
        dists = np.sqrt(np.sum(diff ** 2, axis=-1))  # (K_sub, K_sub)
        # 상삼각 원소만
        triu_idx = np.triu_indices(K_sub, k=1)
        return float(np.mean(dists[triu_idx]))
