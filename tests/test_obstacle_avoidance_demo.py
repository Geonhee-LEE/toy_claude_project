#!/usr/bin/env python3
"""
Obstacle Avoidance Demo 테스트.

장애물 회피 데모의 기본 기능을 테스트합니다.
"""

import unittest
import numpy as np
from examples.obstacle_avoidance_demo import (
    Obstacle,
    create_obstacle_scenario,
    generate_reference_trajectory,
    run_obstacle_avoidance_simulation,
)


class TestObstacle(unittest.TestCase):
    """Obstacle 클래스 테스트."""

    def test_distance_calculation(self):
        """거리 계산 테스트."""
        obs = Obstacle(x=2.0, y=2.0, radius=0.5)

        # 정확한 거리 계산
        dist = obs.distance_to(5.0, 6.0)
        expected = np.sqrt((5.0 - 2.0) ** 2 + (6.0 - 2.0) ** 2)
        self.assertAlmostEqual(dist, expected, places=5)

        # 중심까지의 거리
        dist_center = obs.distance_to(2.0, 2.0)
        self.assertAlmostEqual(dist_center, 0.0, places=5)

    def test_collision_detection(self):
        """충돌 감지 테스트."""
        obs = Obstacle(x=2.0, y=2.0, radius=0.5)

        # 충돌 (장애물 내부)
        self.assertTrue(obs.is_collision(2.0, 2.0))
        self.assertTrue(obs.is_collision(2.3, 2.0))

        # 충돌 아님 (장애물 외부)
        self.assertFalse(obs.is_collision(5.0, 5.0))
        self.assertFalse(obs.is_collision(3.0, 2.0))

        # 안전 마진 포함 테스트
        self.assertTrue(obs.is_collision(2.6, 2.0, safety_margin=0.2))
        self.assertFalse(obs.is_collision(2.8, 2.0, safety_margin=0.2))


class TestScenarioGeneration(unittest.TestCase):
    """시나리오 생성 테스트."""

    def test_create_obstacle_scenario(self):
        """장애물 시나리오 생성 테스트."""
        initial_state, goal_state, obstacles = create_obstacle_scenario()

        # 초기 상태 확인
        self.assertEqual(len(initial_state), 3)
        self.assertEqual(initial_state[0], 0.0)
        self.assertEqual(initial_state[1], 0.0)

        # 목표 상태 확인
        self.assertEqual(len(goal_state), 3)
        self.assertEqual(goal_state[0], 5.0)
        self.assertEqual(goal_state[1], 5.0)

        # 장애물 개수 확인
        self.assertEqual(len(obstacles), 3)

        # 각 장애물의 속성 확인
        for obs in obstacles:
            self.assertIsInstance(obs, Obstacle)
            self.assertGreater(obs.radius, 0)
            self.assertIsInstance(obs.x, float)
            self.assertIsInstance(obs.y, float)

    def test_generate_reference_trajectory(self):
        """참조 궤적 생성 테스트."""
        start = np.array([0.0, 0.0, 0.0])
        goal = np.array([5.0, 5.0, np.pi / 4])

        # 기본 포인트 수
        ref = generate_reference_trajectory(start, goal, n_points=100)
        self.assertEqual(ref.shape, (100, 3))

        # 시작점과 끝점 확인
        np.testing.assert_array_almost_equal(ref[0, :2], start[:2])
        np.testing.assert_array_almost_equal(ref[-1, :2], goal[:2])

        # 다른 포인트 수
        ref_short = generate_reference_trajectory(start, goal, n_points=10)
        self.assertEqual(ref_short.shape, (10, 3))


class TestSimulation(unittest.TestCase):
    """시뮬레이션 실행 테스트."""

    def test_simulation_runs_without_error(self):
        """시뮬레이션이 에러 없이 실행되는지 테스트."""
        initial_state, goal_state, obstacles = create_obstacle_scenario()

        # 짧은 시뮬레이션 실행 (빠른 테스트)
        result = run_obstacle_avoidance_simulation(
            initial_state=initial_state,
            goal_state=goal_state,
            obstacles=obstacles,
            enable_obstacle_avoidance=True,
            max_steps=10,  # 짧게 실행
        )

        # 결과 검증
        self.assertIsNotNone(result)
        self.assertGreater(len(result.states), 0)
        self.assertGreaterEqual(result.total_time, 0)

    def test_simulation_initial_state_preserved(self):
        """시뮬레이션 초기 상태가 유지되는지 테스트."""
        initial_state, goal_state, obstacles = create_obstacle_scenario()

        result = run_obstacle_avoidance_simulation(
            initial_state=initial_state,
            goal_state=goal_state,
            obstacles=obstacles,
            enable_obstacle_avoidance=False,
            max_steps=5,
        )

        # 첫 상태가 초기 상태와 같은지 확인
        np.testing.assert_array_almost_equal(
            result.states[0], initial_state, decimal=5
        )

    def test_simulation_with_and_without_avoidance(self):
        """회피 활성화/비활성화 시뮬레이션 비교 테스트."""
        initial_state, goal_state, obstacles = create_obstacle_scenario()

        # 회피 활성화
        result_with = run_obstacle_avoidance_simulation(
            initial_state=initial_state,
            goal_state=goal_state,
            obstacles=obstacles,
            enable_obstacle_avoidance=True,
            max_steps=10,
        )

        # 회피 비활성화
        result_without = run_obstacle_avoidance_simulation(
            initial_state=initial_state,
            goal_state=goal_state,
            obstacles=obstacles,
            enable_obstacle_avoidance=False,
            max_steps=10,
        )

        # 둘 다 실행되었는지 확인
        self.assertGreater(len(result_with.states), 0)
        self.assertGreater(len(result_without.states), 0)


if __name__ == "__main__":
    unittest.main()
