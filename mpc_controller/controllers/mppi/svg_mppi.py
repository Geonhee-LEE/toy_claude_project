"""SVG-MPPI (Stein Variational Guided MPPI) 컨트롤러 — Guide particle 다중 모드 탐색.

Kondo et al., ICRA 2024 (proj-svg_mppi) 기반 구현.

핵심 아이디어: G개 guide particle만 SVGD로 최적화한 뒤,
나머지 K-G개 샘플을 guide 주변에서 리샘플링하여 효율적으로 다중 모드 탐색.

┌──────────────────────────────────────────────────────────────┐
│ SVMPC:    all K → SVGD(K,K) → weight       (느림, O(K²D))    │
│                                                              │
│ SVG-MPPI: G guides → SVGD(G,G) → expand    (빠름, G<<K)     │
│           K-G followers → sample near guides → weight        │
│                                                              │
│  알고리즘:                                                    │
│   1. G개 guide particle 초기화 (상위 비용 기반 선택)          │
│   2. SVGD 최적화 (G×G 커널, L회 반복)                        │
│   3. 각 guide 주변 K/G개 follower 리샘플링                   │
│   4. 전체 K개로 rollout → cost → weight → update             │
└──────────────────────────────────────────────────────────────┘
"""

import logging
import time
from typing import Optional, Tuple

import numpy as np

from mpc_controller.controllers.mppi.stein_variational_mppi import (
    SteinVariationalMPPIController,
)
from mpc_controller.controllers.mppi.utils import (
    softmax_weights,
    effective_sample_size,
)
from mpc_controller.models.differential_drive import RobotParams
from mpc_controller.controllers.mppi.mppi_params import MPPIParams

logger = logging.getLogger(__name__)


class SVGMPPIController(SteinVariationalMPPIController):
    """SVG-MPPI — Guide particle 기반 효율적 다중 모드 탐색.

    SteinVariationalMPPIController를 상속하여 _svgd_update() 재사용.
    G << K로 SVGD 계산량을 줄이면서 다중 모드 탐색 능력 유지.
    """

    def compute_control(
        self,
        current_state: np.ndarray,
        reference_trajectory: np.ndarray,
    ) -> Tuple[np.ndarray, dict]:
        """Guide particle SVGD + follower resampling."""
        G = self.params.svg_num_guide_particles
        K = self.params.K
        L = self.params.svg_guide_iterations

        # G=0 또는 L=0이면 Vanilla fallback
        if G <= 0 or L <= 0:
            return MPPIController_compute_control(self, current_state, reference_trajectory)

        solve_start = time.perf_counter()

        N = self.params.N
        nu = self.dynamics.nu
        D = N * nu

        # ──── Phase 1: 초기 샘플링 & 비용 ────

        # shift
        self.U[:-1] = self.U[1:]
        self.U[-1] = 0.0

        # 전체 K개 샘플링 (guide 선택용)
        noise = self.sampler.sample(K, N, nu)  # (K, N, nu)
        perturbed_controls = self.U[np.newaxis, :, :] + noise
        perturbed_controls = self.dynamics.clip_controls(perturbed_controls)

        trajectories = self.dynamics.rollout_batch(
            current_state, perturbed_controls, self.params.dt
        )
        costs = self.cost.compute(
            trajectories, perturbed_controls, reference_trajectory
        )

        # ──── Phase 2: Guide particle 선택 ────

        # 비용 상위(최저) G개를 guide로 선택
        G = min(G, K)
        guide_idx = np.argpartition(costs, G)[:G]
        guide_particles = perturbed_controls[guide_idx].reshape(G, D)  # (G, D)
        guide_costs = costs[guide_idx]

        diversity_before = self._compute_diversity(
            guide_particles.reshape(G, N, nu)
        )

        # ──── Phase 3: SVGD on guides only (G×G) ────

        step_size = self.params.svg_guide_step_size

        for svgd_iter in range(L):
            # Softmax weights on guides
            current_lambda = self._get_current_lambda()
            guide_weights = softmax_weights(guide_costs, current_lambda)

            # Pairwise diff (G, G, D) — G << K이므로 메모리 효율적
            diff = guide_particles[:, np.newaxis, :] - guide_particles[np.newaxis, :, :]
            sq_dist = np.sum(diff ** 2, axis=-1)  # (G, G)

            # Bandwidth (median heuristic)
            if self.params.svgd_bandwidth is not None:
                h = self.params.svgd_bandwidth
            else:
                triu_idx = np.triu_indices(G, k=1)
                if len(triu_idx[0]) > 0:
                    med = np.median(sq_dist[triu_idx])
                    h = max(np.sqrt(med / (2.0 * np.log(G + 1))), 1e-6)
                else:
                    h = 1.0

            # RBF kernel
            kernel = np.exp(-sq_dist / (2.0 * h ** 2))

            # SVGD force (재사용)
            force = self._svgd_update(diff, guide_weights, kernel, h, G)

            # Guide 업데이트
            guide_particles = guide_particles + step_size * force

            # Clip & re-evaluate
            guide_controls = guide_particles.reshape(G, N, nu)
            guide_controls = self.dynamics.clip_controls(guide_controls)
            guide_particles = guide_controls.reshape(G, D)

            guide_trajs = self.dynamics.rollout_batch(
                current_state, guide_controls, self.params.dt
            )
            guide_costs = self.cost.compute(
                guide_trajs, guide_controls, reference_trajectory
            )

        diversity_after = self._compute_diversity(
            guide_particles.reshape(G, N, nu)
        )

        # ──── Phase 4: Follower resampling ────

        # 각 guide 주변에서 (K-G)/G개씩 리샘플링
        guide_controls = guide_particles.reshape(G, N, nu)
        n_followers = K - G
        followers_per_guide = max(1, n_followers // G)
        resample_std = self.params.svg_resample_std

        all_controls = [guide_controls]  # guides 자체 포함
        for g in range(G):
            n_f = followers_per_guide if g < G - 1 else (n_followers - followers_per_guide * (G - 1))
            if n_f <= 0:
                continue
            follower_noise = self.sampler.sample(n_f, N, nu) * resample_std
            follower_controls = guide_controls[g : g + 1, :, :] + follower_noise
            follower_controls = self.dynamics.clip_controls(follower_controls)
            all_controls.append(follower_controls)

        all_controls = np.concatenate(all_controls, axis=0)  # (K_total, N, nu)
        K_total = all_controls.shape[0]

        # ──── Phase 5: 전체 rollout & weight ────

        all_trajectories = self.dynamics.rollout_batch(
            current_state, all_controls, self.params.dt
        )
        all_costs = self.cost.compute(
            all_trajectories, all_controls, reference_trajectory
        )

        current_lambda = self._get_current_lambda()
        weights = self._compute_weights(all_costs)

        # 가중 평균으로 제어열 업데이트
        effective_noise = all_controls - self.U[np.newaxis, :, :]
        weighted_noise = weights[:, np.newaxis, np.newaxis] * effective_noise
        self.U += np.sum(weighted_noise, axis=0)
        self.U = self.dynamics.clip_controls(self.U[np.newaxis, :, :])[0]

        u_opt = self.U[0].copy()
        solve_time = time.perf_counter() - solve_start

        weighted_traj = np.sum(
            weights[:, np.newaxis, np.newaxis] * all_trajectories, axis=0
        )
        best_idx = np.argmin(all_costs)
        ess = effective_sample_size(weights)

        if self._adaptive_temp is not None:
            current_lambda = self._adaptive_temp.update(ess, K_total)

        self._iteration_count += 1
        logger.info(
            f"SVG-MPPI iteration {self._iteration_count}: "
            f"solve_time={solve_time*1000:.2f}ms, "
            f"min_cost={all_costs[best_idx]:.4f}, "
            f"guides={G}, followers={n_followers}, "
            f"diversity={diversity_before:.4f}→{diversity_after:.4f}, "
            f"ESS={ess:.1f}/{K_total}"
        )

        info = {
            "predicted_trajectory": weighted_traj,
            "predicted_controls": self.U.copy(),
            "cost": float(all_costs[best_idx]),
            "mean_cost": float(np.mean(all_costs)),
            "solve_time": solve_time,
            "solver_status": "optimal",
            "sample_trajectories": all_trajectories,
            "sample_weights": weights,
            "sample_costs": all_costs,
            "best_trajectory": all_trajectories[best_idx],
            "best_index": best_idx,
            "ess": ess,
            "temperature": current_lambda,
            # SVG-MPPI 전용
            "num_guides": G,
            "num_followers": n_followers,
            "guide_iterations": L,
            "guide_diversity_before": diversity_before,
            "guide_diversity_after": diversity_after,
            "guide_costs": guide_costs.copy(),
        }

        return u_opt, info


def MPPIController_compute_control(controller, state, ref):
    """Vanilla MPPI fallback (base class)."""
    from mpc_controller.controllers.mppi.base_mppi import MPPIController
    return MPPIController.compute_control(controller, state, ref)
