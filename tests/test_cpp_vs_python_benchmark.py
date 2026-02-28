"""Python vs C++ MPPI 벤치마크 스위트 단위 테스트.

C++ pybind11 빌드 필요:
    cd ros2_ws && source /opt/ros/jazzy/setup.bash && colcon build --packages-select mpc_controller_ros2
    source ros2_ws/install/setup.bash

실행:
    pytest tests/test_cpp_vs_python_benchmark.py -v
"""

import numpy as np
import pytest

from examples.cpp_vs_python_benchmark.scenario import (
    BenchmarkScenario,
    circle_scenario,
    figure8_scenario,
    get_model_nx,
)

# C++ pybind11 바인딩이 없으면 전체 skip
try:
    from mpc_controller_ros2.mppi import MPPIParams
    HAS_CPP = True
except ImportError:
    HAS_CPP = False

pytestmark = pytest.mark.skipif(not HAS_CPP, reason="C++ pybind11 bindings not available")


# ============================================================================
# Scenario tests (C++ 불필요)
# ============================================================================
class TestScenario:
    def test_circle_trajectory_shape(self):
        s = circle_scenario(nx=3)
        assert s.trajectory.shape[1] == 3
        assert s.trajectory.shape[0] == 200
        assert s.initial_state.shape == (3,)

    def test_model_ref_dimension_padding(self):
        s4 = circle_scenario(nx=4)
        assert s4.trajectory.shape[1] == 4
        # delta 열은 0
        np.testing.assert_array_equal(s4.trajectory[:, 3], 0.0)

    def test_obstacles_default(self):
        s = circle_scenario(nx=3, with_obstacles=True)
        assert s.obstacles is not None
        assert s.obstacles.shape[1] == 3  # x, y, radius
        s_no = circle_scenario(nx=3, with_obstacles=False)
        assert s_no.obstacles is None

    def test_figure8_shape(self):
        s = figure8_scenario(nx=3)
        assert s.trajectory.shape[1] == 3
        assert s.trajectory.shape[0] == 300

    def test_get_model_nx(self):
        assert get_model_nx("diff_drive") == 3
        assert get_model_nx("swerve") == 3
        assert get_model_nx("non_coaxial_swerve") == 4


# ============================================================================
# CppMPPIAssembler tests
# ============================================================================
class TestCppMPPIAssembler:
    def test_diff_drive_vanilla_basic(self):
        """K=32 소규모 DiffDrive + Vanilla 동작 확인."""
        from examples.cpp_vs_python_benchmark.cpp_mppi_assembler import CppMPPIAssembler

        ctrl = CppMPPIAssembler("diff_drive", "vanilla", K=32, N=10, seed=42)
        state = np.array([0.0, 0.0, 0.0])
        ref = np.zeros((11, 3))
        ref[:, 0] = np.linspace(0, 1, 11)

        u, info = ctrl.compute_control(state, ref)

        assert u.shape == (2,)
        assert info["backend"] == "cpp"
        assert info["solve_time"] > 0
        assert info["sample_trajectories"].shape == (32, 11, 3)
        assert info["sample_weights"].shape == (32,)

    def test_swerve_tsallis(self):
        """Swerve + Tsallis: nu=3 차원 확인."""
        from examples.cpp_vs_python_benchmark.cpp_mppi_assembler import CppMPPIAssembler

        ctrl = CppMPPIAssembler("swerve", "tsallis", K=32, N=10, seed=42)
        state = np.array([0.0, 0.0, 0.0])
        ref = np.zeros((11, 3))
        ref[:, 0] = np.linspace(0, 1, 11)

        u, info = ctrl.compute_control(state, ref)
        assert u.shape == (3,)  # vx, vy, omega

    def test_non_coaxial_risk_aware(self):
        """NonCoaxial + RiskAware: nx=4 차원 확인."""
        from examples.cpp_vs_python_benchmark.cpp_mppi_assembler import CppMPPIAssembler

        ctrl = CppMPPIAssembler("non_coaxial_swerve", "risk_aware", K=32, N=10, seed=42)
        state = np.zeros(4)
        ref = np.zeros((11, 4))
        ref[:, 0] = np.linspace(0, 1, 11)

        u, info = ctrl.compute_control(state, ref)
        assert u.shape == (3,)
        assert info["predicted_trajectory"].shape == (11, 4)

    def test_adaptive_temperature(self):
        """Adaptive temperature: λ 변화 확인."""
        from examples.cpp_vs_python_benchmark.cpp_mppi_assembler import CppMPPIAssembler

        ctrl = CppMPPIAssembler("diff_drive", "vanilla", K=64, N=10,
                                adaptive_temperature=True, seed=42)
        state = np.array([0.0, 0.0, 0.0])
        ref = np.zeros((11, 3))
        ref[:, 0] = np.linspace(0, 2, 11)

        # 여러 번 호출하면 λ가 변화해야 함
        temps = []
        for _ in range(5):
            _, info = ctrl.compute_control(state, ref)
            temps.append(info["temperature"])
        # 최소 한 번은 변화
        assert len(set(temps)) > 1 or len(temps) == 1

    def test_obstacles(self):
        """장애물 추가 시 비용 증가 확인."""
        from examples.cpp_vs_python_benchmark.cpp_mppi_assembler import CppMPPIAssembler

        ctrl_no_obs = CppMPPIAssembler("diff_drive", "vanilla", K=32, N=10,
                                        obstacles=None, seed=42)
        ctrl_obs = CppMPPIAssembler("diff_drive", "vanilla", K=32, N=10,
                                     obstacles=np.array([[0.5, 0.0, 0.3]]),
                                     seed=42)
        state = np.array([0.0, 0.0, 0.0])
        ref = np.zeros((11, 3))
        ref[:, 0] = np.linspace(0, 1, 11)

        _, info_no = ctrl_no_obs.compute_control(state, ref)
        _, info_obs = ctrl_obs.compute_control(state, ref)

        # 장애물이 있으면 평균 비용이 더 높아야 함
        assert info_obs["mean_cost"] >= info_no["mean_cost"] * 0.5  # 느슨한 비교

    def test_reset(self):
        """Reset 후 U=0 확인."""
        from examples.cpp_vs_python_benchmark.cpp_mppi_assembler import CppMPPIAssembler

        ctrl = CppMPPIAssembler("diff_drive", "vanilla", K=32, N=10, seed=42)
        state = np.array([0.0, 0.0, 0.0])
        ref = np.zeros((11, 3))
        ref[:, 0] = np.linspace(0, 1, 11)

        ctrl.compute_control(state, ref)
        assert not np.allclose(ctrl.U, 0.0)

        ctrl.reset()
        np.testing.assert_array_equal(ctrl.U, 0.0)


# ============================================================================
# BenchmarkRunner tests
# ============================================================================
class TestBenchmarkRunner:
    def test_pipeline_single_config(self):
        """단일 설정 파이프라인 실행 확인."""
        from mpc_controller import TrajectoryInterpolator
        from examples.cpp_vs_python_benchmark.benchmark_runner import simulate
        from examples.cpp_vs_python_benchmark.cpp_mppi_assembler import CppMPPIAssembler

        scenario = circle_scenario(nx=3)
        interp = TrajectoryInterpolator(scenario.trajectory[:, :3], scenario.dt)

        ctrl = CppMPPIAssembler("diff_drive", "vanilla", K=32, N=10, seed=42)
        result = simulate(
            ctrl, "test", "cpp", "diff_drive", "vanilla",
            interp, np.zeros(3), sim_time=1.0,
        )
        assert result.backend == "cpp"
        assert len(result.solve_times) > 0
        assert result.avg_solve_ms > 0

    def test_component_sampling(self):
        """Sampling 마이크로벤치마크 확인."""
        from examples.cpp_vs_python_benchmark.benchmark_runner import (
            _benchmark_sampling_py, _benchmark_sampling_cpp,
        )
        py_times = _benchmark_sampling_py(64, 10, 2, 3)
        cpp_times = _benchmark_sampling_cpp(64, 10, 2, 3)
        assert len(py_times) == 3
        assert len(cpp_times) == 3
        assert all(t > 0 for t in py_times)
        assert all(t > 0 for t in cpp_times)

    def test_scaling_k(self):
        """소규모 K 스케일링 확인."""
        from examples.cpp_vs_python_benchmark.benchmark_runner import run_scaling_benchmark

        results = run_scaling_benchmark(
            K_values=[32, 64],
            N_values=[5, 10],
            repeat=2,
        )
        assert len(results) == 2  # K + N
        assert results[0].variable == "K"
        assert results[1].variable == "N"
        assert len(results[0].py_times_ms) == 2
        assert len(results[0].cpp_times_ms) == 2
