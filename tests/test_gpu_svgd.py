"""GPU SVGD 커널 단위 테스트.

SVGD JIT 함수 (svgd_step, median_bandwidth, diversity) 검증.
CPU utils 함수와 수치 일치 확인.
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


@pytest.fixture
def particles():
    """테스트 입자 (K=32, D=10)."""
    np.random.seed(123)
    return jnp.array(np.random.randn(32, 10))


@pytest.fixture
def weights():
    """테스트 가중치 (K=32)."""
    np.random.seed(456)
    w = np.random.rand(32)
    w /= w.sum()
    return jnp.array(w)


class TestSVGDStep:
    """svgd_step_jit 검증."""

    def test_output_shape(self, particles, weights):
        """출력 shape == 입력 shape."""
        from mpc_controller.controllers.mppi.gpu_svgd import svgd_step_jit

        updated = svgd_step_jit(particles, weights, 1.0, 0.1)
        assert updated.shape == particles.shape

    def test_attractive_direction(self):
        """매력력: 저비용(고가중치) 샘플 방향으로 이동."""
        from mpc_controller.controllers.mppi.gpu_svgd import svgd_step_jit

        # 2개 입자: p0=[0,0](고가중치), p1=[1,0](저가중치)
        p = jnp.array([[0.0, 0.0], [1.0, 0.0]])
        w = jnp.array([0.9, 0.1])  # p0가 저비용
        h = 1.0
        step = 0.1

        updated = svgd_step_jit(p, w, h, step)

        # p1은 p0 방향(왼쪽)으로 이동해야 함
        assert float(updated[1, 0]) < 1.0

    def test_repulsive_force(self):
        """반발력: 동일 위치 입자가 서로 밀어냄."""
        from mpc_controller.controllers.mppi.gpu_svgd import svgd_step_jit

        # 3개 입자 동일 위치 + 1개 다른 위치
        p = jnp.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [5.0, 0.0]])
        w = jnp.array([0.25, 0.25, 0.25, 0.25])
        h = 1.0
        step = 0.5

        updated = svgd_step_jit(p, w, h, step)

        # 동일 위치 3개가 분산되어야 함 (완전히 동일하지 않을 수 있음)
        # 적어도 step > 0이면 업데이트가 발생
        total_movement = jnp.sum(jnp.abs(updated - p))
        assert float(total_movement) > 0

    def test_step_size_zero_no_change(self, particles, weights):
        """step_size=0 → 변화 없음."""
        from mpc_controller.controllers.mppi.gpu_svgd import svgd_step_jit

        updated = svgd_step_jit(particles, weights, 1.0, 0.0)
        np.testing.assert_allclose(
            np.asarray(updated), np.asarray(particles), atol=1e-12
        )


class TestMedianBandwidth:
    """median_bandwidth_jit 검증."""

    def test_positive_output(self, particles):
        """bandwidth > 0."""
        from mpc_controller.controllers.mppi.gpu_svgd import median_bandwidth_jit

        h = median_bandwidth_jit(particles)
        assert float(h) > 0

    def test_matches_cpu(self, particles):
        """GPU bandwidth ≈ CPU bandwidth (sort 기반 근사 → rtol=1e-3)."""
        from mpc_controller.controllers.mppi.gpu_svgd import median_bandwidth_jit
        from mpc_controller.controllers.mppi.utils import median_bandwidth

        gpu_h = float(np.asarray(median_bandwidth_jit(particles)))
        cpu_h = median_bandwidth(np.asarray(particles))

        np.testing.assert_allclose(gpu_h, cpu_h, rtol=1e-3)

    def test_identical_particles_min_value(self):
        """동일 입자 → bandwidth = 1e-6 (최소 보장)."""
        from mpc_controller.controllers.mppi.gpu_svgd import median_bandwidth_jit

        p = jnp.ones((10, 5))
        h = median_bandwidth_jit(p)
        assert float(h) >= 1e-6


class TestDiversity:
    """compute_diversity_jit 검증."""

    def test_single_point_zero(self):
        """단일 입자 → diversity = 0."""
        from mpc_controller.controllers.mppi.gpu_svgd import compute_diversity_jit

        p = jnp.array([[1.0, 2.0, 3.0]])
        d = compute_diversity_jit(p)
        assert float(d) == 0.0

    def test_identical_points_zero(self):
        """동일 입자들 → diversity = 0."""
        from mpc_controller.controllers.mppi.gpu_svgd import compute_diversity_jit

        p = jnp.ones((10, 5))
        d = compute_diversity_jit(p)
        np.testing.assert_allclose(float(d), 0.0, atol=1e-10)

    def test_spread_points_positive(self, particles):
        """분산된 입자 → diversity > 0."""
        from mpc_controller.controllers.mppi.gpu_svgd import compute_diversity_jit

        d = compute_diversity_jit(particles)
        assert float(d) > 0


class TestRBFKernel:
    """rbf_kernel_jit 검증."""

    def test_matches_cpu(self, particles):
        """GPU RBF == CPU RBF."""
        from mpc_controller.controllers.mppi.gpu_svgd import rbf_kernel_jit
        from mpc_controller.controllers.mppi.utils import rbf_kernel

        h = 1.0
        gpu_k = np.asarray(rbf_kernel_jit(particles, h))
        cpu_k = rbf_kernel(np.asarray(particles), h)

        np.testing.assert_allclose(gpu_k, cpu_k, rtol=1e-10)

    def test_diagonal_is_one(self, particles):
        """대각선 = 1.0 (자기 자신 커널)."""
        from mpc_controller.controllers.mppi.gpu_svgd import rbf_kernel_jit

        k = rbf_kernel_jit(particles, 1.0)
        np.testing.assert_allclose(np.asarray(jnp.diag(k)), 1.0, atol=1e-12)

    def test_symmetric(self, particles):
        """커널 행렬 대칭."""
        from mpc_controller.controllers.mppi.gpu_svgd import rbf_kernel_jit

        k = rbf_kernel_jit(particles, 1.0)
        np.testing.assert_allclose(np.asarray(k), np.asarray(k.T), atol=1e-12)
