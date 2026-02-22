"""GPU MPPI 정확도 + 성능 테스트.

JAX CPU fallback에서도 실행 가능.
GPU가 없으면 성능 테스트는 skip.
"""

import time

import numpy as np
import pytest

# ─── JAX 가용성 체크 ───
try:
    import jax
    import jax.numpy as jnp

    jax.config.update("jax_enable_x64", True)
    JAX_AVAILABLE = True
    GPU_AVAILABLE = any(d.platform == "gpu" for d in jax.devices())
except ImportError:
    JAX_AVAILABLE = False
    GPU_AVAILABLE = False

pytestmark = pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not installed")


# ─── Fixtures ───

@pytest.fixture
def dt():
    return 0.05


@pytest.fixture
def simple_state():
    """초기 상태 [x=0, y=0, θ=0]."""
    return np.array([0.0, 0.0, 0.0])


@pytest.fixture
def reference_trajectory():
    """직선 참조 궤적 (N+1=31 스텝, 전진)."""
    N = 30
    ref = np.zeros((N + 1, 3))
    for t in range(N + 1):
        ref[t, 0] = t * 0.05  # x 전진
    return ref


@pytest.fixture
def obstacles():
    """테스트 장애물: 3개."""
    return np.array([
        [2.0, 0.5, 0.3],
        [3.0, -0.3, 0.2],
        [4.0, 0.0, 0.4],
    ])


# ═══════════════════════════════════════════════════════════════
# GPU Dynamics 테스트
# ═══════════════════════════════════════════════════════════════

class TestGPUDynamics:
    """GPU 동역학 (lax.scan + vmap) 정확도 검증."""

    def test_diff_drive_rk4_accuracy(self, dt):
        """RK4 1-step: GPU 동역학이 해석적 결과와 일치하는지 검증."""
        from mpc_controller.controllers.mppi.gpu_dynamics import (
            diff_drive_dynamics,
            _rk4_step,
        )

        state = jnp.array([0.0, 0.0, 0.0])
        control = jnp.array([1.0, 0.0])  # 직진, 회전 없음

        next_state = _rk4_step(diff_drive_dynamics, state, control, dt)

        # 직진: x += v*dt, y=0, θ=0
        np.testing.assert_allclose(float(next_state[0]), dt, atol=1e-10)
        np.testing.assert_allclose(float(next_state[1]), 0.0, atol=1e-10)
        np.testing.assert_allclose(float(next_state[2]), 0.0, atol=1e-10)

    def test_swerve_dynamics(self, dt):
        """Swerve drive: body frame → world frame 변환 검증."""
        from mpc_controller.controllers.mppi.gpu_dynamics import (
            swerve_dynamics,
            _rk4_step,
        )

        state = jnp.array([0.0, 0.0, np.pi / 2])  # 90도 회전
        control = jnp.array([1.0, 0.0, 0.0])       # body vx=1

        next_state = _rk4_step(swerve_dynamics, state, control, dt)

        # body vx=1, θ=90° → world ẋ=0, ẏ=1
        np.testing.assert_allclose(float(next_state[0]), 0.0, atol=1e-10)
        np.testing.assert_allclose(float(next_state[1]), dt, atol=1e-10)

    def test_non_coaxial_dynamics(self, dt):
        """Non-coaxial swerve: nx=4 (β 포함) 검증."""
        from mpc_controller.controllers.mppi.gpu_dynamics import (
            non_coaxial_swerve_dynamics,
            _rk4_step,
        )

        state = jnp.array([0.0, 0.0, 0.0, 0.0])
        control = jnp.array([1.0, 0.0, 0.5])

        next_state = _rk4_step(non_coaxial_swerve_dynamics, state, control, dt)

        assert next_state.shape == (4,)
        np.testing.assert_allclose(float(next_state[0]), dt, atol=1e-3)

    def test_rollout_matches_cpu(self, simple_state, dt):
        """GPU rollout == CPU rollout (rtol=1e-10)."""
        from mpc_controller.controllers.mppi.gpu_dynamics import make_rollout_batch_jit, diff_drive_dynamics
        from mpc_controller.controllers.mppi.dynamics_wrapper import BatchDynamicsWrapper
        from mpc_controller.models.differential_drive import RobotParams

        K, N, nu = 64, 30, 2
        np.random.seed(42)
        controls_np = np.random.randn(K, N, nu) * 0.3
        controls_np[..., 0] = np.clip(controls_np[..., 0], -1.0, 1.0)
        controls_np[..., 1] = np.clip(controls_np[..., 1], -1.5, 1.5)

        # CPU rollout
        cpu_dynamics = BatchDynamicsWrapper(RobotParams())
        traj_cpu = cpu_dynamics.rollout_batch(simple_state, controls_np, dt)

        # GPU rollout
        rollout_fn = make_rollout_batch_jit(diff_drive_dynamics, dt)
        x0_jax = jnp.asarray(simple_state)
        controls_jax = jnp.asarray(controls_np)
        traj_gpu = np.asarray(rollout_fn(x0_jax, controls_jax))

        np.testing.assert_allclose(traj_gpu, traj_cpu, rtol=1e-10, atol=1e-12)

    def test_angle_normalization(self, dt):
        """큰 각도에서 arctan2 정규화 동작 확인."""
        from mpc_controller.controllers.mppi.gpu_dynamics import (
            diff_drive_dynamics,
            _rk4_step,
        )

        state = jnp.array([0.0, 0.0, 3.0])    # 3 rad (≈172°)
        control = jnp.array([0.0, 1.0])          # 순수 회전

        next_state = _rk4_step(diff_drive_dynamics, state, control, dt)

        # 결과 각도가 [-π, π] 범위 내
        assert -np.pi <= float(next_state[2]) <= np.pi


# ═══════════════════════════════════════════════════════════════
# GPU Costs 테스트
# ═══════════════════════════════════════════════════════════════

class TestGPUCosts:
    """GPU 비용 함수 정확도 검증."""

    def _make_test_data(self, K=64, N=30, nx=3, nu=2):
        """테스트 데이터 생성."""
        np.random.seed(123)
        trajectories = np.random.randn(K, N + 1, nx) * 0.5
        controls = np.random.randn(K, N, nu) * 0.3
        reference = np.random.randn(N + 1, nx) * 0.5
        return trajectories, controls, reference

    def test_state_tracking_matches_cpu(self):
        """StateTrackingCost: GPU == CPU."""
        from mpc_controller.controllers.mppi.gpu_costs import state_tracking_cost_jit
        from mpc_controller.controllers.mppi.cost_functions import StateTrackingCost

        Q = np.diag([10.0, 10.0, 1.0])
        traj, ctrl, ref = self._make_test_data()

        cpu_cost = StateTrackingCost(Q).compute(traj, ctrl, ref)
        gpu_cost = np.asarray(state_tracking_cost_jit(
            jnp.asarray(traj), jnp.asarray(ref), jnp.asarray(np.diag(Q))
        ))

        np.testing.assert_allclose(gpu_cost, cpu_cost, rtol=1e-10, atol=1e-12)

    def test_terminal_cost_matches(self):
        """TerminalCost: GPU == CPU."""
        from mpc_controller.controllers.mppi.gpu_costs import terminal_cost_jit
        from mpc_controller.controllers.mppi.cost_functions import TerminalCost

        Qf = np.diag([50.0, 50.0, 5.0])
        traj, ctrl, ref = self._make_test_data()

        cpu_cost = TerminalCost(Qf).compute(traj, ctrl, ref)
        gpu_cost = np.asarray(terminal_cost_jit(
            jnp.asarray(traj), jnp.asarray(ref), jnp.asarray(np.diag(Qf))
        ))

        np.testing.assert_allclose(gpu_cost, cpu_cost, rtol=1e-10, atol=1e-12)

    def test_control_effort_matches(self):
        """ControlEffortCost: GPU == CPU."""
        from mpc_controller.controllers.mppi.gpu_costs import control_effort_cost_jit
        from mpc_controller.controllers.mppi.cost_functions import ControlEffortCost

        R = np.diag([0.01, 0.01])
        traj, ctrl, ref = self._make_test_data()

        cpu_cost = ControlEffortCost(R).compute(traj, ctrl, ref)
        gpu_cost = np.asarray(control_effort_cost_jit(
            jnp.asarray(ctrl), jnp.asarray(np.diag(R))
        ))

        np.testing.assert_allclose(gpu_cost, cpu_cost, rtol=1e-10, atol=1e-12)

    def test_control_rate_matches(self):
        """ControlRateCost: GPU == CPU."""
        from mpc_controller.controllers.mppi.gpu_costs import control_rate_cost_jit
        from mpc_controller.controllers.mppi.cost_functions import ControlRateCost

        R_rate = np.array([0.1, 0.1])
        traj, ctrl, ref = self._make_test_data()

        cpu_cost = ControlRateCost(R_rate).compute(traj, ctrl, ref)
        gpu_cost = np.asarray(control_rate_cost_jit(
            jnp.asarray(ctrl), jnp.asarray(R_rate)
        ))

        np.testing.assert_allclose(gpu_cost, cpu_cost, rtol=1e-10, atol=1e-12)

    def test_obstacle_cost_vectorized(self, obstacles):
        """ObstacleCost 벡터화: GPU == CPU."""
        from mpc_controller.controllers.mppi.gpu_costs import obstacle_cost_jit
        from mpc_controller.controllers.mppi.cost_functions import ObstacleCost

        traj, ctrl, ref = self._make_test_data()
        weight = 1000.0
        safety_margin = 0.3

        cpu_cost = ObstacleCost(obstacles, weight, safety_margin).compute(traj, ctrl, ref)
        gpu_cost = np.asarray(obstacle_cost_jit(
            jnp.asarray(traj), jnp.asarray(obstacles),
            weight, safety_margin,
        ))

        np.testing.assert_allclose(gpu_cost, cpu_cost, rtol=1e-10, atol=1e-12)

    def test_composite_cost_fusion(self, obstacles):
        """통합 비용 (모든 cost fusion): GPU == CPU 합산."""
        from mpc_controller.controllers.mppi.gpu_costs import make_compute_all_costs_jit
        from mpc_controller.controllers.mppi.cost_functions import (
            CompositeMPPICost, StateTrackingCost, TerminalCost,
            ControlEffortCost, ControlRateCost, ObstacleCost,
        )

        Q = np.diag([10.0, 10.0, 1.0])
        Qf = np.diag([50.0, 50.0, 5.0])
        R = np.diag([0.01, 0.01])
        R_rate = np.array([0.1, 0.1])
        weight = 1000.0
        safety_margin = 0.3

        traj, ctrl, ref = self._make_test_data()

        # CPU
        cpu_cost_fn = CompositeMPPICost()
        cpu_cost_fn.add(StateTrackingCost(Q))
        cpu_cost_fn.add(TerminalCost(Qf))
        cpu_cost_fn.add(ControlEffortCost(R))
        cpu_cost_fn.add(ControlRateCost(R_rate))
        cpu_cost_fn.add(ObstacleCost(obstacles, weight, safety_margin))
        cpu_total = cpu_cost_fn.compute(traj, ctrl, ref)

        # GPU
        gpu_fn = make_compute_all_costs_jit(has_r_rate=True, has_obstacles=True)
        cost_params = {
            "q_diag": jnp.asarray(np.diag(Q)),
            "qf_diag": jnp.asarray(np.diag(Qf)),
            "r_diag": jnp.asarray(np.diag(R)),
            "r_rate_diag": jnp.asarray(R_rate),
            "obstacles": jnp.asarray(obstacles),
            "obs_weight": weight,
            "safety_margin": safety_margin,
        }
        gpu_total = np.asarray(gpu_fn(
            jnp.asarray(traj), jnp.asarray(ctrl), jnp.asarray(ref), cost_params
        ))

        np.testing.assert_allclose(gpu_total, cpu_total, rtol=1e-10, atol=1e-12)


# ═══════════════════════════════════════════════════════════════
# GPU Sampling 테스트
# ═══════════════════════════════════════════════════════════════

class TestGPUSampling:
    """GPU 샘플링 정확도 검증."""

    def test_gaussian_shape_and_scale(self):
        """가우시안: (K,N,nu) 형상 + sigma 스케일."""
        from mpc_controller.controllers.mppi.gpu_sampling import gaussian_sample_jit
        from jax import random

        key = random.PRNGKey(0)
        K, N, nu = 1024, 30, 2
        sigma = jnp.array([0.3, 0.5])

        noise = gaussian_sample_jit(key, K, N, nu, sigma)

        assert noise.shape == (K, N, nu)
        # 표준편차 검증 (충분한 샘플 → 5% 허용)
        std_v = float(jnp.std(noise[:, :, 0]))
        std_w = float(jnp.std(noise[:, :, 1]))
        np.testing.assert_allclose(std_v, 0.3, atol=0.03)
        np.testing.assert_allclose(std_w, 0.5, atol=0.05)

    def test_colored_noise_correlation(self):
        """Colored noise: 시간 자기상관 존재 확인."""
        from mpc_controller.controllers.mppi.gpu_sampling import colored_noise_sample_jit
        from jax import random

        key = random.PRNGKey(1)
        K, N, nu = 2048, 30, 2
        sigma = jnp.array([0.3, 0.3])
        beta = 1.0

        noise = colored_noise_sample_jit(key, K, N, nu, sigma, beta)

        assert noise.shape == (K, N, nu)

        # lag-1 자기상관: colored > 가우시안 (≈0)
        noise_np = np.asarray(noise[:, :, 0])
        corr_sum = 0.0
        for k in range(min(K, 500)):
            corr_sum += np.corrcoef(noise_np[k, :-1], noise_np[k, 1:])[0, 1]
        avg_corr = corr_sum / min(K, 500)

        # OU 프로세스: lag-1 상관 = exp(-β) ≈ 0.368 (β=1)
        assert avg_corr > 0.2, f"Expected positive autocorrelation, got {avg_corr:.3f}"


# ═══════════════════════════════════════════════════════════════
# GPU MPPI Kernel 통합 테스트
# ═══════════════════════════════════════════════════════════════

class TestGPUMPPIKernel:
    """GPU MPPI 커널 통합 검증."""

    def test_mppi_step_returns_valid(self, simple_state, reference_trajectory):
        """GPU MPPI 스텝: 유효한 제어 + info dict 반환."""
        from mpc_controller.controllers.mppi.gpu_mppi_kernel import GPUMPPIKernel

        kernel = GPUMPPIKernel(
            N=30, K=256, nu=2, nx=3, dt=0.05,
            lambda_=10.0,
            noise_sigma=np.array([0.3, 0.3]),
            q_diag=np.array([10.0, 10.0, 1.0]),
            qf_diag=np.array([50.0, 50.0, 5.0]),
            r_diag=np.array([0.01, 0.01]),
            max_velocity=1.0, max_omega=1.5,
            warmup=True,
        )

        x0 = jnp.asarray(simple_state)
        U = jnp.zeros((30, 2))
        ref = jnp.asarray(reference_trajectory)

        U_new, info = kernel.mppi_step(x0, U, ref)

        assert U_new.shape == (30, 2)
        assert "costs" in info
        assert "weights" in info
        assert "trajectories" in info
        assert info["trajectories"].shape == (256, 31, 3)

    def test_warmup_no_error(self):
        """JIT warmup 정상 완료."""
        from mpc_controller.controllers.mppi.gpu_mppi_kernel import GPUMPPIKernel

        kernel = GPUMPPIKernel(
            N=10, K=64, nu=2, nx=3, dt=0.05,
            lambda_=10.0,
            noise_sigma=np.array([0.3, 0.3]),
            q_diag=np.array([10.0, 10.0, 1.0]),
            qf_diag=np.array([50.0, 50.0, 5.0]),
            r_diag=np.array([0.01, 0.01]),
            max_velocity=1.0, max_omega=1.5,
            warmup=True,
        )
        # warmup이 에러 없이 완료되면 통과
        assert kernel is not None

    def test_mppi_step_with_obstacles(self, simple_state, reference_trajectory, obstacles):
        """장애물이 있을 때 GPU MPPI 스텝 동작."""
        from mpc_controller.controllers.mppi.gpu_mppi_kernel import GPUMPPIKernel

        kernel = GPUMPPIKernel(
            N=30, K=256, nu=2, nx=3, dt=0.05,
            lambda_=10.0,
            noise_sigma=np.array([0.3, 0.3]),
            q_diag=np.array([10.0, 10.0, 1.0]),
            qf_diag=np.array([50.0, 50.0, 5.0]),
            r_diag=np.array([0.01, 0.01]),
            max_velocity=1.0, max_omega=1.5,
            warmup=False,
        )
        kernel.set_obstacles(obstacles)

        x0 = jnp.asarray(simple_state)
        U = jnp.zeros((30, 2))
        ref = jnp.asarray(reference_trajectory)

        U_new, info = kernel.mppi_step(x0, U, ref)
        assert U_new.shape == (30, 2)
        # 장애물 비용이 포함되어 total cost > 0
        assert float(jnp.min(info["costs"])) >= 0

    def test_fallback_no_gpu(self):
        """GPU 없으면 CPU fallback (base_mppi 경로)."""
        from mpc_controller.controllers.mppi.base_mppi import MPPIController
        from mpc_controller.controllers.mppi.mppi_params import MPPIParams
        from mpc_controller.models.differential_drive import RobotParams

        params = MPPIParams(K=64, N=10, use_gpu=False)
        ctrl = MPPIController(RobotParams(), params)
        assert ctrl._use_gpu is False

    def test_base_mppi_gpu_integration(self, simple_state, reference_trajectory):
        """base_mppi GPU 경로 통합 테스트."""
        from mpc_controller.controllers.mppi.base_mppi import MPPIController
        from mpc_controller.controllers.mppi.mppi_params import MPPIParams
        from mpc_controller.models.differential_drive import RobotParams

        params = MPPIParams(K=256, N=30, use_gpu=True)
        ctrl = MPPIController(RobotParams(), params, seed=42)

        u, info = ctrl.compute_control(simple_state, reference_trajectory)

        assert u.shape == (2,)
        assert "solve_time" in info
        assert info.get("backend") == "gpu"


# ═══════════════════════════════════════════════════════════════
# GPU 성능 테스트
# ═══════════════════════════════════════════════════════════════

class TestGPUPerformance:
    """GPU 성능 검증 (GPU 있을 때만 실행)."""

    @pytest.mark.skipif(not GPU_AVAILABLE, reason="No GPU available")
    def test_k4096_under_5ms(self, simple_state, reference_trajectory):
        """K=4096, N=30 < 5ms 목표."""
        from mpc_controller.controllers.mppi.gpu_mppi_kernel import GPUMPPIKernel

        kernel = GPUMPPIKernel(
            N=30, K=4096, nu=2, nx=3, dt=0.05,
            lambda_=10.0,
            noise_sigma=np.array([0.3, 0.3]),
            q_diag=np.array([10.0, 10.0, 1.0]),
            qf_diag=np.array([50.0, 50.0, 5.0]),
            r_diag=np.array([0.01, 0.01]),
            max_velocity=1.0, max_omega=1.5,
            warmup=True,
        )

        x0 = jnp.asarray(simple_state)
        U = jnp.zeros((30, 2))
        ref = jnp.asarray(reference_trajectory)

        # 워밍업 후 10회 평균
        times = []
        for _ in range(10):
            start = time.perf_counter()
            kernel.mppi_step(x0, U, ref)
            jax.block_until_ready(U)
            elapsed = (time.perf_counter() - start) * 1000
            times.append(elapsed)

        avg_ms = np.mean(times[2:])  # 처음 2회 제외
        assert avg_ms < 5.0, f"K=4096 avg={avg_ms:.2f}ms (target <5ms)"

    @pytest.mark.skipif(not GPU_AVAILABLE, reason="No GPU available")
    def test_gpu_faster_than_cpu(self, simple_state, reference_trajectory):
        """GPU < CPU (K=1024 이상)."""
        from mpc_controller.controllers.mppi.base_mppi import MPPIController
        from mpc_controller.controllers.mppi.mppi_params import MPPIParams
        from mpc_controller.models.differential_drive import RobotParams

        K = 1024

        # CPU 시간 측정
        cpu_params = MPPIParams(K=K, N=30, use_gpu=False)
        cpu_ctrl = MPPIController(RobotParams(), cpu_params, seed=42)
        cpu_times = []
        for _ in range(5):
            start = time.perf_counter()
            cpu_ctrl.compute_control(simple_state, reference_trajectory)
            cpu_times.append((time.perf_counter() - start) * 1000)
        cpu_avg = np.mean(cpu_times[1:])

        # GPU 시간 측정
        gpu_params = MPPIParams(K=K, N=30, use_gpu=True)
        gpu_ctrl = MPPIController(RobotParams(), gpu_params, seed=42)
        gpu_times = []
        for _ in range(5):
            start = time.perf_counter()
            gpu_ctrl.compute_control(simple_state, reference_trajectory)
            gpu_times.append((time.perf_counter() - start) * 1000)
        gpu_avg = np.mean(gpu_times[1:])

        assert gpu_avg < cpu_avg, (
            f"GPU ({gpu_avg:.2f}ms) not faster than CPU ({cpu_avg:.2f}ms)"
        )


# ═══════════════════════════════════════════════════════════════
# GPU Backend 테스트
# ═══════════════════════════════════════════════════════════════

class TestGPUBackend:
    """GPU 백엔드 유틸리티 테스트."""

    def test_is_jax_available(self):
        from mpc_controller.controllers.mppi.gpu_backend import is_jax_available
        assert is_jax_available() is True

    def test_to_jax_roundtrip(self):
        """NumPy → JAX → NumPy 왕복 변환."""
        from mpc_controller.controllers.mppi.gpu_backend import to_jax, to_numpy

        arr = np.array([1.0, 2.0, 3.0])
        jax_arr = to_jax(arr)
        back = to_numpy(jax_arr)
        np.testing.assert_array_equal(arr, back)

    def test_get_backend_name(self):
        from mpc_controller.controllers.mppi.gpu_backend import get_backend_name
        name = get_backend_name()
        assert name in ("jax-gpu", "jax-cpu", "numpy")

    def test_float32_dtype(self):
        from mpc_controller.controllers.mppi.gpu_backend import get_dtype
        dt32 = get_dtype(use_float32=True)
        dt64 = get_dtype(use_float32=False)
        assert dt32 == jnp.float32
        assert dt64 == jnp.float64
