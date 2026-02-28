"""GPU MPPI 변형 통합 테스트 — 8종 GPU vs CPU.

각 변형의 GPU 경로가 정상 동작하는지 검증:
1. use_gpu=True 생성 + compute_control 정상 반환
2. 연속 5회 호출 수렴 (발산 없음)
3. weights-only 변형: GPU weight == CPU weight
"""

import numpy as np
import pytest

try:
    import jax
    import jax.numpy as jnp

    jax.config.update("jax_enable_x64", True)
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False

pytestmark = pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not installed")

from mpc_controller.models.differential_drive import RobotParams
from mpc_controller.controllers.mppi.mppi_params import MPPIParams


# ─── 공통 Fixtures ───

@pytest.fixture
def robot_params():
    return RobotParams()


@pytest.fixture
def state():
    return np.array([0.0, 0.0, 0.0])


@pytest.fixture
def reference():
    """직선 참조 궤적 (N+1=31 스텝)."""
    N = 30
    ref = np.zeros((N + 1, 3))
    for t in range(N + 1):
        ref[t, 0] = t * 0.05
    return ref


# ═══════════════════════════════════════════════════════════════
# Vanilla GPU (기존 — 회귀 확인)
# ═══════════════════════════════════════════════════════════════

class TestVanillaGPU:
    """Vanilla MPPI GPU 회귀 테스트."""

    def test_gpu_compute_control(self, robot_params, state, reference):
        from mpc_controller.controllers.mppi.base_mppi import MPPIController

        params = MPPIParams(K=128, N=30, use_gpu=True, gpu_warmup=False)
        ctrl = MPPIController(robot_params, params, seed=42)

        u, info = ctrl.compute_control(state, reference)
        assert u.shape == (2,)
        assert info.get("backend") == "gpu"

    def test_convergence(self, robot_params, state, reference):
        from mpc_controller.controllers.mppi.base_mppi import MPPIController

        params = MPPIParams(K=256, N=30, use_gpu=True, gpu_warmup=False)
        ctrl = MPPIController(robot_params, params, seed=42)

        costs = []
        for _ in range(5):
            _, info = ctrl.compute_control(state, reference)
            costs.append(info["cost"])

        # 발산하지 않음
        assert all(np.isfinite(c) for c in costs)


# ═══════════════════════════════════════════════════════════════
# Log-MPPI GPU
# ═══════════════════════════════════════════════════════════════

class TestLogMPPIGPU:
    """Log-MPPI GPU 검증."""

    def test_gpu_compute_control(self, robot_params, state, reference):
        from mpc_controller.controllers.mppi.log_mppi import LogMPPIController

        params = MPPIParams(K=128, N=30, use_gpu=True, gpu_warmup=False)
        ctrl = LogMPPIController(robot_params, params, seed=42)

        u, info = ctrl.compute_control(state, reference)
        assert u.shape == (2,)
        assert info.get("backend") == "gpu"

    def test_convergence(self, robot_params, state, reference):
        from mpc_controller.controllers.mppi.log_mppi import LogMPPIController

        params = MPPIParams(K=256, N=30, use_gpu=True, gpu_warmup=False)
        ctrl = LogMPPIController(robot_params, params, seed=42)

        for _ in range(5):
            _, info = ctrl.compute_control(state, reference)
            assert np.isfinite(info["cost"])


# ═══════════════════════════════════════════════════════════════
# Tsallis-MPPI GPU
# ═══════════════════════════════════════════════════════════════

class TestTsallisMPPIGPU:
    """Tsallis-MPPI GPU 검증."""

    def test_gpu_compute_control(self, robot_params, state, reference):
        from mpc_controller.controllers.mppi.tsallis_mppi import TsallisMPPIController

        params = MPPIParams(K=128, N=30, tsallis_q=1.5, use_gpu=True, gpu_warmup=False)
        ctrl = TsallisMPPIController(robot_params, params, seed=42)

        u, info = ctrl.compute_control(state, reference)
        assert u.shape == (2,)
        assert info.get("backend") == "gpu"

    def test_convergence(self, robot_params, state, reference):
        from mpc_controller.controllers.mppi.tsallis_mppi import TsallisMPPIController

        params = MPPIParams(K=256, N=30, tsallis_q=1.5, use_gpu=True, gpu_warmup=False)
        ctrl = TsallisMPPIController(robot_params, params, seed=42)

        for _ in range(5):
            _, info = ctrl.compute_control(state, reference)
            assert np.isfinite(info["cost"])


# ═══════════════════════════════════════════════════════════════
# Risk-Aware (CVaR) GPU
# ═══════════════════════════════════════════════════════════════

class TestRiskAwareMPPIGPU:
    """Risk-Aware MPPI GPU 검증."""

    def test_gpu_compute_control(self, robot_params, state, reference):
        from mpc_controller.controllers.mppi.risk_aware_mppi import RiskAwareMPPIController

        params = MPPIParams(K=128, N=30, cvar_alpha=0.5, use_gpu=True, gpu_warmup=False)
        ctrl = RiskAwareMPPIController(robot_params, params, seed=42)

        u, info = ctrl.compute_control(state, reference)
        assert u.shape == (2,)
        assert info.get("backend") == "gpu"

    def test_convergence(self, robot_params, state, reference):
        from mpc_controller.controllers.mppi.risk_aware_mppi import RiskAwareMPPIController

        params = MPPIParams(K=256, N=30, cvar_alpha=0.5, use_gpu=True, gpu_warmup=False)
        ctrl = RiskAwareMPPIController(robot_params, params, seed=42)

        for _ in range(5):
            _, info = ctrl.compute_control(state, reference)
            assert np.isfinite(info["cost"])


# ═══════════════════════════════════════════════════════════════
# Tube-MPPI GPU (코드 변경 없음 — 검증만)
# ═══════════════════════════════════════════════════════════════

class TestTubeMPPIGPU:
    """Tube-MPPI GPU 검증 (super().compute_control()이 GPU 경로 사용)."""

    def test_gpu_tube_enabled(self, robot_params, state, reference):
        from mpc_controller.controllers.mppi.tube_mppi import TubeMPPIController

        params = MPPIParams(
            K=128, N=30, use_gpu=True, gpu_warmup=False,
            tube_enabled=True,
        )
        ctrl = TubeMPPIController(robot_params, params, seed=42)

        u, info = ctrl.compute_control(state, reference)
        assert u.shape == (2,)
        assert info.get("tube_enabled") is True

    def test_gpu_tube_disabled(self, robot_params, state, reference):
        """tube_enabled=False → GPU Vanilla 경로."""
        from mpc_controller.controllers.mppi.tube_mppi import TubeMPPIController

        params = MPPIParams(
            K=128, N=30, use_gpu=True, gpu_warmup=False,
            tube_enabled=False,
        )
        ctrl = TubeMPPIController(robot_params, params, seed=42)

        u, info = ctrl.compute_control(state, reference)
        assert u.shape == (2,)
        assert info.get("backend") == "gpu"


# ═══════════════════════════════════════════════════════════════
# Smooth-MPPI GPU
# ═══════════════════════════════════════════════════════════════

class TestSmoothMPPIGPU:
    """Smooth-MPPI GPU 검증."""

    def test_gpu_compute_control(self, robot_params, state, reference):
        from mpc_controller.controllers.mppi.smooth_mppi import SmoothMPPIController

        params = MPPIParams(K=128, N=30, use_gpu=True, gpu_warmup=False)
        ctrl = SmoothMPPIController(robot_params, params, seed=42)

        u, info = ctrl.compute_control(state, reference)
        assert u.shape == (2,)
        assert info.get("backend") == "gpu"

    def test_convergence(self, robot_params, state, reference):
        from mpc_controller.controllers.mppi.smooth_mppi import SmoothMPPIController

        params = MPPIParams(K=256, N=30, use_gpu=True, gpu_warmup=False)
        ctrl = SmoothMPPIController(robot_params, params, seed=42)

        for _ in range(5):
            _, info = ctrl.compute_control(state, reference)
            assert np.isfinite(info["cost"])

    def test_has_smooth_info(self, robot_params, state, reference):
        """Smooth-MPPI 전용 info key 존재."""
        from mpc_controller.controllers.mppi.smooth_mppi import SmoothMPPIController

        params = MPPIParams(K=128, N=30, use_gpu=True, gpu_warmup=False)
        ctrl = SmoothMPPIController(robot_params, params, seed=42)

        _, info = ctrl.compute_control(state, reference)
        assert "delta_u_sequence" in info
        assert "delta_u_norm" in info


# ═══════════════════════════════════════════════════════════════
# Spline-MPPI GPU
# ═══════════════════════════════════════════════════════════════

class TestSplineMPPIGPU:
    """Spline-MPPI GPU 검증."""

    def test_gpu_compute_control(self, robot_params, state, reference):
        from mpc_controller.controllers.mppi.spline_mppi import SplineMPPIController

        params = MPPIParams(K=128, N=30, use_gpu=True, gpu_warmup=False)
        ctrl = SplineMPPIController(robot_params, params, seed=42)

        u, info = ctrl.compute_control(state, reference)
        assert u.shape == (2,)
        assert info.get("backend") == "gpu"

    def test_convergence(self, robot_params, state, reference):
        from mpc_controller.controllers.mppi.spline_mppi import SplineMPPIController

        params = MPPIParams(K=256, N=30, use_gpu=True, gpu_warmup=False)
        ctrl = SplineMPPIController(robot_params, params, seed=42)

        for _ in range(5):
            _, info = ctrl.compute_control(state, reference)
            assert np.isfinite(info["cost"])

    def test_has_spline_info(self, robot_params, state, reference):
        """Spline-MPPI 전용 info key 존재."""
        from mpc_controller.controllers.mppi.spline_mppi import SplineMPPIController

        params = MPPIParams(K=128, N=30, use_gpu=True, gpu_warmup=False)
        ctrl = SplineMPPIController(robot_params, params, seed=42)

        _, info = ctrl.compute_control(state, reference)
        assert "knot_controls" in info
        assert "num_knots" in info


# ═══════════════════════════════════════════════════════════════
# SVMPC (Stein Variational MPPI) GPU
# ═══════════════════════════════════════════════════════════════

class TestSVMPCGPU:
    """SVMPC GPU 검증."""

    def test_gpu_compute_control(self, robot_params, state, reference):
        from mpc_controller.controllers.mppi.stein_variational_mppi import SteinVariationalMPPIController

        params = MPPIParams(
            K=128, N=30, use_gpu=True, gpu_warmup=False,
            svgd_num_iterations=2, svgd_step_size=0.1,
        )
        ctrl = SteinVariationalMPPIController(robot_params, params, seed=42)

        u, info = ctrl.compute_control(state, reference)
        assert u.shape == (2,)
        assert info.get("backend") == "gpu"

    def test_convergence(self, robot_params, state, reference):
        from mpc_controller.controllers.mppi.stein_variational_mppi import SteinVariationalMPPIController

        params = MPPIParams(
            K=128, N=30, use_gpu=True, gpu_warmup=False,
            svgd_num_iterations=2, svgd_step_size=0.1,
        )
        ctrl = SteinVariationalMPPIController(robot_params, params, seed=42)

        for _ in range(5):
            _, info = ctrl.compute_control(state, reference)
            assert np.isfinite(info["cost"])

    def test_diversity_measured(self, robot_params, state, reference):
        """SVGD diversity 측정 존재."""
        from mpc_controller.controllers.mppi.stein_variational_mppi import SteinVariationalMPPIController

        params = MPPIParams(
            K=128, N=30, use_gpu=True, gpu_warmup=False,
            svgd_num_iterations=2,
        )
        ctrl = SteinVariationalMPPIController(robot_params, params, seed=42)

        _, info = ctrl.compute_control(state, reference)
        assert "sample_diversity_before" in info
        assert "sample_diversity_after" in info

    def test_svgd0_fallback_vanilla(self, robot_params, state, reference):
        """svgd_num_iterations=0 → Vanilla GPU 경로."""
        from mpc_controller.controllers.mppi.stein_variational_mppi import SteinVariationalMPPIController

        params = MPPIParams(
            K=128, N=30, use_gpu=True, gpu_warmup=False,
            svgd_num_iterations=0,
        )
        ctrl = SteinVariationalMPPIController(robot_params, params, seed=42)

        u, info = ctrl.compute_control(state, reference)
        assert u.shape == (2,)
        # Vanilla GPU 경로로 실행됨
        assert info.get("backend") == "gpu"


# ═══════════════════════════════════════════════════════════════
# SVG-MPPI GPU
# ═══════════════════════════════════════════════════════════════

class TestSVGMPPIGPU:
    """SVG-MPPI GPU 검증."""

    def test_gpu_compute_control(self, robot_params, state, reference):
        from mpc_controller.controllers.mppi.svg_mppi import SVGMPPIController

        params = MPPIParams(
            K=128, N=30, use_gpu=True, gpu_warmup=False,
            svg_num_guide_particles=8, svg_guide_iterations=2,
        )
        ctrl = SVGMPPIController(robot_params, params, seed=42)

        u, info = ctrl.compute_control(state, reference)
        assert u.shape == (2,)
        assert info.get("backend") == "gpu"

    def test_convergence(self, robot_params, state, reference):
        from mpc_controller.controllers.mppi.svg_mppi import SVGMPPIController

        params = MPPIParams(
            K=128, N=30, use_gpu=True, gpu_warmup=False,
            svg_num_guide_particles=8, svg_guide_iterations=2,
        )
        ctrl = SVGMPPIController(robot_params, params, seed=42)

        for _ in range(5):
            _, info = ctrl.compute_control(state, reference)
            assert np.isfinite(info["cost"])

    def test_guide_diversity(self, robot_params, state, reference):
        """SVG-MPPI guide diversity 측정 존재."""
        from mpc_controller.controllers.mppi.svg_mppi import SVGMPPIController

        params = MPPIParams(
            K=128, N=30, use_gpu=True, gpu_warmup=False,
            svg_num_guide_particles=8, svg_guide_iterations=2,
        )
        ctrl = SVGMPPIController(robot_params, params, seed=42)

        _, info = ctrl.compute_control(state, reference)
        assert "guide_diversity_before" in info
        assert "guide_diversity_after" in info
        assert "num_guides" in info


# ═══════════════════════════════════════════════════════════════
# GPU Weight 정확도 — GPU vs CPU 수치 비교
# ═══════════════════════════════════════════════════════════════

class TestGPUWeightAccuracy:
    """GPU 가중치 == CPU 가중치 (weights-only 변형)."""

    def test_log_gpu_matches_cpu(self):
        """Log-MPPI: GPU weight == CPU log_sum_exp weight."""
        from mpc_controller.controllers.mppi.gpu_weights import log_softmax_weights
        from mpc_controller.controllers.mppi.utils import log_sum_exp

        np.random.seed(42)
        costs = np.random.rand(200) * 20
        lambda_ = 5.0

        # CPU (log_mppi.py 로직)
        log_w = -costs / lambda_
        log_w -= log_sum_exp(log_w)
        cpu_w = np.exp(log_w)

        # GPU
        gpu_w = np.asarray(log_softmax_weights(jnp.array(costs), lambda_))

        np.testing.assert_allclose(gpu_w, cpu_w, rtol=1e-10)

    def test_tsallis_gpu_matches_cpu(self):
        """Tsallis-MPPI: GPU weight == CPU q_exponential weight."""
        from mpc_controller.controllers.mppi.gpu_weights import make_tsallis_weights
        from mpc_controller.controllers.mppi.utils import q_exponential

        np.random.seed(42)
        costs = np.random.rand(200) * 20
        lambda_ = 5.0
        q = 1.5

        # CPU (tsallis_mppi.py 로직)
        centered = costs - np.min(costs)
        raw = q_exponential(-centered / lambda_, q)
        total = np.sum(raw)
        cpu_w = raw / total if total > 0 else np.ones(200) / 200

        # GPU
        tsallis_fn = make_tsallis_weights(q)
        gpu_w = np.asarray(tsallis_fn(jnp.array(costs), lambda_))

        np.testing.assert_allclose(gpu_w, cpu_w, rtol=1e-6)

    def test_cvar_gpu_selects_correct_samples(self):
        """CVaR: GPU 가 최저비용 샘플을 올바르게 선택."""
        from mpc_controller.controllers.mppi.gpu_weights import make_cvar_weights

        costs = jnp.arange(20, dtype=jnp.float64)  # [0, 1, ..., 19]
        alpha = 0.5  # n_keep = 10
        lambda_ = 5.0

        fn = make_cvar_weights(alpha, 20)
        w = np.asarray(fn(costs, lambda_))

        # 최저 10개 (0~9)만 non-zero
        n_nonzero = np.sum(w > 1e-15)
        assert n_nonzero == 10

        # 상위 10개 (10~19)는 0
        assert np.sum(w[10:]) < 1e-15
