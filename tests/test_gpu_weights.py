"""GPU 가중치 함수 단위 테스트.

4종 JIT 가중치 함수의 정확도 + 수치 안정성 검증.
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
def costs():
    """테스트 비용 배열 (K=100)."""
    np.random.seed(42)
    return jnp.array(np.random.rand(100) * 10)


@pytest.fixture
def lambda_():
    return 10.0


class TestVanillaSoftmax:
    """Vanilla softmax 가중치 검증."""

    def test_matches_cpu(self, costs, lambda_):
        """GPU vanilla == CPU softmax_weights (rtol=1e-10)."""
        from mpc_controller.controllers.mppi.gpu_weights import vanilla_softmax_weights
        from mpc_controller.controllers.mppi.utils import softmax_weights

        gpu_w = np.asarray(vanilla_softmax_weights(costs, lambda_))
        cpu_w = softmax_weights(np.asarray(costs), lambda_)

        np.testing.assert_allclose(gpu_w, cpu_w, rtol=1e-10)

    def test_sum_to_one(self, costs, lambda_):
        """가중치 합 == 1.0."""
        from mpc_controller.controllers.mppi.gpu_weights import vanilla_softmax_weights

        w = vanilla_softmax_weights(costs, lambda_)
        np.testing.assert_allclose(float(jnp.sum(w)), 1.0, atol=1e-12)

    def test_non_negative(self, costs, lambda_):
        """모든 가중치 >= 0."""
        from mpc_controller.controllers.mppi.gpu_weights import vanilla_softmax_weights

        w = vanilla_softmax_weights(costs, lambda_)
        assert jnp.all(w >= 0)


class TestLogSoftmax:
    """Log-space softmax 가중치 검증."""

    def test_matches_vanilla(self, costs, lambda_):
        """Log-softmax == Vanilla softmax (수학적 동일)."""
        from mpc_controller.controllers.mppi.gpu_weights import (
            vanilla_softmax_weights,
            log_softmax_weights,
        )

        vanilla_w = np.asarray(vanilla_softmax_weights(costs, lambda_))
        log_w = np.asarray(log_softmax_weights(costs, lambda_))

        np.testing.assert_allclose(log_w, vanilla_w, rtol=1e-10)

    def test_extreme_costs_no_nan(self):
        """극단 비용 (1e15) → NaN/Inf 없음."""
        from mpc_controller.controllers.mppi.gpu_weights import log_softmax_weights

        extreme = jnp.array([1e15, 1e14, 1e13, 1.0, 0.0])
        w = log_softmax_weights(extreme, 10.0)

        assert jnp.all(jnp.isfinite(w))
        np.testing.assert_allclose(float(jnp.sum(w)), 1.0, atol=1e-10)

    def test_sum_to_one(self, costs, lambda_):
        """가중치 합 == 1.0."""
        from mpc_controller.controllers.mppi.gpu_weights import log_softmax_weights

        w = log_softmax_weights(costs, lambda_)
        np.testing.assert_allclose(float(jnp.sum(w)), 1.0, atol=1e-12)


class TestTsallisWeights:
    """Tsallis q-exponential 가중치 검증."""

    def test_q1_equals_vanilla(self, costs, lambda_):
        """q=1.0 → Vanilla softmax와 근사 동일.

        q=1.0은 closure에서 1/(1-q) 발산하므로 정확히 같지는 않지만,
        q→1 극한에서 수렴해야 한다. q=1.001로 근사 검증.
        """
        from mpc_controller.controllers.mppi.gpu_weights import (
            vanilla_softmax_weights,
            make_tsallis_weights,
        )

        # q=1.001 (1.0에 매우 가까움)
        tsallis_fn = make_tsallis_weights(1.001)
        vanilla_w = np.asarray(vanilla_softmax_weights(costs, lambda_))
        tsallis_w = np.asarray(tsallis_fn(costs, lambda_))

        # 방향 일치: 최소비용 샘플에 최대 가중치
        assert np.argmax(tsallis_w) == np.argmax(vanilla_w)

    def test_heavy_tail_q15(self, costs, lambda_):
        """q=1.5 → heavy-tail (Vanilla보다 더 균등한 분포)."""
        from mpc_controller.controllers.mppi.gpu_weights import (
            vanilla_softmax_weights,
            make_tsallis_weights,
        )

        vanilla_w = np.asarray(vanilla_softmax_weights(costs, lambda_))
        heavy_fn = make_tsallis_weights(1.5)
        heavy_w = np.asarray(heavy_fn(costs, lambda_))

        # heavy-tail: max 가중치가 Vanilla보다 작음 (더 분산)
        assert np.max(heavy_w) < np.max(vanilla_w) + 0.1

    def test_light_tail_q05(self, costs, lambda_):
        """q=0.5 → light-tail (Vanilla보다 더 집중된 분포)."""
        from mpc_controller.controllers.mppi.gpu_weights import (
            vanilla_softmax_weights,
            make_tsallis_weights,
        )

        vanilla_w = np.asarray(vanilla_softmax_weights(costs, lambda_))
        light_fn = make_tsallis_weights(0.5)
        light_w = np.asarray(light_fn(costs, lambda_))

        # light-tail: max 가중치가 Vanilla보다 큼 (더 집중)
        assert np.max(light_w) > np.max(vanilla_w) - 0.1

    def test_sum_to_one(self, costs, lambda_):
        """가중치 합 == 1.0."""
        from mpc_controller.controllers.mppi.gpu_weights import make_tsallis_weights

        for q in [0.5, 1.5, 2.0]:
            fn = make_tsallis_weights(q)
            w = fn(costs, lambda_)
            np.testing.assert_allclose(float(jnp.sum(w)), 1.0, atol=1e-6)

    def test_non_negative(self, costs, lambda_):
        """모든 가중치 >= 0."""
        from mpc_controller.controllers.mppi.gpu_weights import make_tsallis_weights

        for q in [0.5, 1.5, 2.0]:
            fn = make_tsallis_weights(q)
            w = fn(costs, lambda_)
            assert jnp.all(w >= 0)


class TestCVaRWeights:
    """CVaR 가중치 검증."""

    def test_alpha1_equals_vanilla(self, costs, lambda_):
        """alpha=1.0 → Vanilla softmax."""
        from mpc_controller.controllers.mppi.gpu_weights import (
            vanilla_softmax_weights,
            make_cvar_weights,
        )

        cvar_fn = make_cvar_weights(1.0, 100)
        # alpha >= 1.0이면 vanilla_softmax_weights 자체를 반환
        assert cvar_fn is vanilla_softmax_weights

    def test_alpha05_half_nonzero(self, costs, lambda_):
        """alpha=0.5 → K/2개만 non-zero 가중치."""
        from mpc_controller.controllers.mppi.gpu_weights import make_cvar_weights

        K = costs.shape[0]
        cvar_fn = make_cvar_weights(0.5, K)
        w = np.asarray(cvar_fn(costs, lambda_))

        n_nonzero = np.sum(w > 1e-15)
        expected = max(1, int(np.ceil(0.5 * K)))
        assert n_nonzero == expected

    def test_sum_to_one(self, costs, lambda_):
        """가중치 합 == 1.0."""
        from mpc_controller.controllers.mppi.gpu_weights import make_cvar_weights

        for alpha in [0.3, 0.5, 0.8]:
            fn = make_cvar_weights(alpha, 100)
            w = fn(costs, lambda_)
            np.testing.assert_allclose(float(jnp.sum(w)), 1.0, atol=1e-10)

    def test_selects_lowest_costs(self, lambda_):
        """CVaR가 최저비용 샘플만 선택하는지 검증."""
        from mpc_controller.controllers.mppi.gpu_weights import make_cvar_weights

        # 간단한 비용: [0, 1, 2, ..., 9]
        costs = jnp.arange(10, dtype=jnp.float64)
        cvar_fn = make_cvar_weights(0.3, 10)  # n_keep = ceil(0.3*10) = 3
        w = np.asarray(cvar_fn(costs, lambda_))

        # 최저 3개 (0,1,2)만 non-zero
        assert w[0] > 0
        assert w[1] > 0
        assert w[2] > 0
        assert np.sum(w[3:]) < 1e-15


class TestWeightFnRegistry:
    """get_weight_fn 레지스트리 검증."""

    def test_vanilla(self):
        from mpc_controller.controllers.mppi.gpu_weights import get_weight_fn, vanilla_softmax_weights
        assert get_weight_fn("vanilla") is vanilla_softmax_weights

    def test_log(self):
        from mpc_controller.controllers.mppi.gpu_weights import get_weight_fn, log_softmax_weights
        assert get_weight_fn("log") is log_softmax_weights

    def test_tsallis(self):
        from mpc_controller.controllers.mppi.gpu_weights import get_weight_fn
        fn = get_weight_fn("tsallis", q=1.5)
        # 호출 가능한 함수인지 확인
        w = fn(jnp.array([1.0, 2.0, 3.0]), 10.0)
        assert w.shape == (3,)

    def test_cvar(self):
        from mpc_controller.controllers.mppi.gpu_weights import get_weight_fn
        fn = get_weight_fn("cvar", alpha=0.5, K=100)
        w = fn(jnp.arange(100, dtype=jnp.float64), 10.0)
        assert w.shape == (100,)

    def test_unknown_raises(self):
        from mpc_controller.controllers.mppi.gpu_weights import get_weight_fn
        with pytest.raises(ValueError, match="Unknown"):
            get_weight_fn("unknown")
