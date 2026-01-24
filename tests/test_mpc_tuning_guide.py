#!/usr/bin/env python3
"""
MPC 파라미터 튜닝 가이드 테스트

examples/mpc_tuning_guide.py의 주요 기능을 테스트합니다.
"""

import unittest
import numpy as np
import sys
from pathlib import Path

# 프로젝트 루트 경로 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from examples.mpc_tuning_guide import (
    generate_test_trajectory,
    run_mpc_tuning_test,
    TuningResult,
)
from mpc_controller.controllers.mpc import MPCController, MPCParams
from mpc_controller.models.differential_drive import RobotParams


class TestTrajectoryGeneration(unittest.TestCase):
    """궤적 생성 함수 테스트."""

    def test_sine_trajectory(self):
        """정현파 궤적 생성 테스트."""
        traj = generate_test_trajectory("sine", n_points=100)

        self.assertEqual(traj.shape, (100, 3), "궤적 shape이 올바르지 않음")
        self.assertTrue(np.all(np.isfinite(traj)), "궤적에 무한대나 NaN이 포함됨")

        # x는 증가해야 함
        self.assertTrue(np.all(np.diff(traj[:, 0]) > 0), "x 좌표가 증가하지 않음")

        # y는 정현파 패턴이어야 함 (진동)
        self.assertGreater(np.max(traj[:, 1]), 0, "y 좌표 최대값이 양수가 아님")
        self.assertLess(np.min(traj[:, 1]), 0, "y 좌표 최소값이 음수가 아님")

    def test_circle_trajectory(self):
        """원형 궤적 생성 테스트."""
        traj = generate_test_trajectory("circle", n_points=100)

        self.assertEqual(traj.shape, (100, 3), "궤적 shape이 올바르지 않음")
        self.assertTrue(np.all(np.isfinite(traj)), "궤적에 무한대나 NaN이 포함됨")

        # 중심으로부터의 거리가 일정해야 함 (반지름 2.0)
        distances = np.sqrt(traj[:, 0] ** 2 + traj[:, 1] ** 2)
        self.assertTrue(np.allclose(distances, 2.0, atol=0.1), "원의 반지름이 일정하지 않음")

    def test_straight_trajectory(self):
        """직선 궤적 생성 테스트."""
        traj = generate_test_trajectory("straight", n_points=100)

        self.assertEqual(traj.shape, (100, 3), "궤적 shape이 올바르지 않음")
        self.assertTrue(np.all(np.isfinite(traj)), "궤적에 무한대나 NaN이 포함됨")

        # y 좌표가 모두 0이어야 함
        self.assertTrue(np.allclose(traj[:, 1], 0.0, atol=1e-6), "직선 궤적의 y 좌표가 0이 아님")

        # theta가 모두 0이어야 함
        self.assertTrue(np.allclose(traj[:, 2], 0.0, atol=1e-6), "직선 궤적의 theta가 0이 아님")

    def test_invalid_trajectory_type(self):
        """잘못된 궤적 타입 테스트."""
        with self.assertRaises(ValueError):
            generate_test_trajectory("invalid_type")


class TestMPCTuning(unittest.TestCase):
    """MPC 튜닝 테스트."""

    def setUp(self):
        """기본 테스트 설정."""
        self.robot_params = RobotParams(max_velocity=1.0, max_omega=2.0)
        self.mpc_params = MPCParams(
            N=10,  # 빠른 테스트를 위해 작은 값 사용
            dt=0.1,
            Q=np.diag([10.0, 10.0, 1.0]),
            R=np.diag([0.1, 0.1]),
        )
        self.controller = MPCController(self.robot_params, self.mpc_params, enable_soft_constraints=False)
        self.reference = generate_test_trajectory("straight", n_points=50)
        self.initial_state = np.array([0.0, 0.0, 0.0])

    def test_run_mpc_tuning_test(self):
        """MPC 튜닝 테스트 실행 검증."""
        controller, reference, initial_state = self.controller, self.reference, self.initial_state

        result = run_mpc_tuning_test(
            controller,
            reference,
            initial_state,
            dt=0.1,
            max_steps=30,  # 빠른 테스트를 위해 작은 값
            name="Test MPC",
        )

        # TuningResult 구조 검증
        self.assertIsInstance(result, TuningResult, "결과가 TuningResult 타입이 아님")
        self.assertEqual(result.name, "Test MPC", "이름이 올바르지 않음")

        # 상태 검증
        self.assertEqual(result.states.shape[1], 3, "상태 차원이 올바르지 않음")
        self.assertGreater(len(result.states), 1, "상태가 업데이트되지 않음")
        self.assertTrue(np.all(np.isfinite(result.states)), "상태에 무한대나 NaN이 포함됨")

        # 제어 입력 검증
        self.assertEqual(result.controls.shape[1], 2, "제어 입력 차원이 올바르지 않음")
        self.assertGreater(len(result.controls), 0, "제어 입력이 생성되지 않음")
        self.assertTrue(np.all(np.isfinite(result.controls)), "제어 입력에 무한대나 NaN이 포함됨")

        # 비용 및 솔버 시간 검증
        self.assertEqual(len(result.costs), len(result.controls), "비용 개수가 제어 입력 개수와 다름")
        self.assertEqual(len(result.solve_times), len(result.controls), "솔버 시간 개수가 제어 입력 개수와 다름")
        self.assertTrue(all(t > 0 for t in result.solve_times), "솔버 시간이 양수가 아님")

        # 추적 오차 검증
        self.assertEqual(len(result.tracking_errors), len(result.controls), "추적 오차 개수가 제어 입력 개수와 다름")
        self.assertTrue(all(e >= 0 for e in result.tracking_errors), "추적 오차가 음수임")

        # 제어 부드러움 검증
        self.assertGreaterEqual(result.control_smoothness, 0, "제어 부드러움이 음수임")

        # 전체 시간 검증
        self.assertGreater(result.total_time, 0, "전체 시간이 양수가 아님")

    def test_different_prediction_horizons(self):
        """다양한 예측 구간 테스트."""
        robot_params = RobotParams(max_velocity=1.0, max_omega=2.0)
        reference = generate_test_trajectory("straight", n_points=50)
        initial_state = np.array([0.0, 0.0, 0.0])

        N_values = [5, 10, 15]
        results = []

        for N in N_values:
            mpc_params = MPCParams(
                N=N,
                dt=0.1,
                Q=np.diag([10.0, 10.0, 1.0]),
                R=np.diag([0.1, 0.1]),
            )
            controller = MPCController(robot_params, mpc_params, enable_soft_constraints=False)
            result = run_mpc_tuning_test(
                controller, reference, initial_state, max_steps=20, name=f"N={N}"
            )
            results.append(result)

        # 모든 테스트가 성공적으로 완료되었는지 확인
        self.assertTrue(all(len(r.controls) > 0 for r in results), "일부 테스트가 제어 입력을 생성하지 못함")

        # N이 클수록 일반적으로 솔버 시간이 증가해야 함 (항상은 아님)
        avg_solve_times = [np.mean(r.solve_times) for r in results]
        self.assertTrue(all(t > 0 for t in avg_solve_times), "평균 솔버 시간이 양수가 아님")

    def test_different_state_weights(self):
        """다양한 상태 가중치 테스트."""
        robot_params = RobotParams(max_velocity=1.0, max_omega=2.0)
        reference = generate_test_trajectory("sine", n_points=50)
        initial_state = np.array([0.0, 0.0, 0.0])

        Q_configs = [
            np.diag([1.0, 1.0, 0.1]),
            np.diag([10.0, 10.0, 1.0]),
            np.diag([100.0, 100.0, 10.0]),
        ]
        results = []

        for Q in Q_configs:
            mpc_params = MPCParams(
                N=10,
                dt=0.1,
                Q=Q,
                R=np.diag([0.1, 0.1]),
            )
            controller = MPCController(robot_params, mpc_params, enable_soft_constraints=False)
            result = run_mpc_tuning_test(
                controller, reference, initial_state, max_steps=20, name=f"Q={Q[0,0]}"
            )
            results.append(result)

        # 모든 테스트가 성공적으로 완료되었는지 확인
        self.assertTrue(all(len(r.controls) > 0 for r in results), "일부 테스트가 제어 입력을 생성하지 못함")

        # Q가 클수록 일반적으로 추적 오차가 감소해야 함 (항상은 아님)
        avg_errors = [np.mean(r.tracking_errors) for r in results]
        self.assertTrue(all(e >= 0 for e in avg_errors), "평균 추적 오차가 음수임")


class TestTuningResult(unittest.TestCase):
    """TuningResult 데이터클래스 테스트."""

    def test_tuning_result_creation(self):
        """TuningResult 생성 테스트."""
        result = TuningResult(
            name="Test",
            params=MPCParams(),
            states=np.zeros((10, 3)),
            controls=np.zeros((9, 2)),
            costs=[1.0, 2.0, 3.0],
            solve_times=[0.001, 0.002, 0.003],
            tracking_errors=[0.1, 0.2, 0.3],
            control_smoothness=0.5,
            total_time=1.0,
        )

        self.assertEqual(result.name, "Test", "이름이 올바르지 않음")
        self.assertEqual(result.states.shape, (10, 3), "상태 shape이 올바르지 않음")
        self.assertEqual(result.controls.shape, (9, 2), "제어 입력 shape이 올바르지 않음")
        self.assertEqual(len(result.costs), 3, "비용 개수가 올바르지 않음")
        self.assertEqual(len(result.solve_times), 3, "솔버 시간 개수가 올바르지 않음")
        self.assertEqual(len(result.tracking_errors), 3, "추적 오차 개수가 올바르지 않음")
        self.assertEqual(result.control_smoothness, 0.5, "제어 부드러움이 올바르지 않음")
        self.assertEqual(result.total_time, 1.0, "전체 시간이 올바르지 않음")


if __name__ == "__main__":
    unittest.main(verbosity=2)
