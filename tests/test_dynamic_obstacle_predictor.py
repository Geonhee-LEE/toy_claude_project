"""
Dynamic Obstacle Predictor 단위 테스트.

동적 장애물 예측 알고리즘의 정확성을 검증합니다.
"""

import pytest
import numpy as np

from mpc_controller.planners.dynamic_obstacle_predictor import (
    DynamicObstaclePredictor,
    DynamicObstacleState,
    PredictionModel,
)


class TestDynamicObstacleState:
    """DynamicObstacleState 클래스 테스트."""

    def test_speed_calculation(self):
        """속도 크기 계산 테스트."""
        state = DynamicObstacleState(
            x=0.0, y=0.0, vx=3.0, vy=4.0, radius=0.5
        )
        assert state.speed == pytest.approx(5.0, rel=1e-5)

    def test_heading_calculation(self):
        """진행 방향 각도 계산 테스트."""
        # 동쪽 방향 (0도)
        state1 = DynamicObstacleState(
            x=0.0, y=0.0, vx=1.0, vy=0.0, radius=0.5
        )
        assert state1.heading == pytest.approx(0.0, rel=1e-5)

        # 북쪽 방향 (90도)
        state2 = DynamicObstacleState(
            x=0.0, y=0.0, vx=0.0, vy=1.0, radius=0.5
        )
        assert state2.heading == pytest.approx(np.pi / 2, rel=1e-5)

        # 남서쪽 방향 (-135도)
        state3 = DynamicObstacleState(
            x=0.0, y=0.0, vx=-1.0, vy=-1.0, radius=0.5
        )
        assert state3.heading == pytest.approx(-3 * np.pi / 4, rel=1e-5)


class TestConstantVelocityPrediction:
    """일정 속도 모델 예측 테스트."""

    def test_linear_motion_prediction(self):
        """직선 운동 예측 테스트."""
        predictor = DynamicObstaclePredictor(
            model=PredictionModel.CONSTANT_VELOCITY,
            uncertainty_growth_rate=0.1,
        )

        state = DynamicObstacleState(
            x=0.0, y=0.0, vx=1.0, vy=0.5, radius=0.3
        )

        dt = 0.1
        steps = 10
        predictions = predictor.predict_trajectory(state, dt, steps)

        # 10스텝 후 위치 확인
        pred_x, pred_y, pred_radius = predictions[-1]

        # 예상 위치: x = 0 + 1.0 * 1.0 = 1.0, y = 0 + 0.5 * 1.0 = 0.5
        expected_x = state.x + state.vx * (dt * steps)
        expected_y = state.y + state.vy * (dt * steps)

        assert pred_x == pytest.approx(expected_x, rel=1e-5)
        assert pred_y == pytest.approx(expected_y, rel=1e-5)

        # 불확실성 증가 확인
        expected_uncertainty = 0.1 * (dt * steps)
        expected_radius = state.radius + expected_uncertainty
        assert pred_radius == pytest.approx(expected_radius, rel=1e-5)

    def test_stationary_obstacle(self):
        """정지 장애물 예측 테스트."""
        predictor = DynamicObstaclePredictor(
            model=PredictionModel.CONSTANT_VELOCITY
        )

        state = DynamicObstacleState(
            x=5.0, y=3.0, vx=0.0, vy=0.0, radius=0.5
        )

        dt = 0.1
        steps = 10
        predictions = predictor.predict_trajectory(state, dt, steps)

        # 모든 예측 위치가 초기 위치와 같아야 함
        for pred_x, pred_y, pred_radius in predictions:
            assert pred_x == pytest.approx(5.0, rel=1e-5)
            assert pred_y == pytest.approx(3.0, rel=1e-5)


class TestConstantAccelerationPrediction:
    """일정 가속도 모델 예측 테스트."""

    def test_accelerated_motion_prediction(self):
        """가속 운동 예측 테스트."""
        predictor = DynamicObstaclePredictor(
            model=PredictionModel.CONSTANT_ACCELERATION,
            uncertainty_growth_rate=0.1,
        )

        state = DynamicObstacleState(
            x=0.0, y=0.0, vx=0.0, vy=0.0, ax=1.0, ay=0.5, radius=0.3
        )

        dt = 0.1
        steps = 10
        predictions = predictor.predict_trajectory(state, dt, steps)

        # 10스텝 후 위치 확인
        pred_x, pred_y, pred_radius = predictions[-1]
        t = dt * steps

        # 예상 위치: x = 0 + 0*t + 0.5*1.0*t^2
        expected_x = state.x + state.vx * t + 0.5 * state.ax * t**2
        expected_y = state.y + state.vy * t + 0.5 * state.ay * t**2

        assert pred_x == pytest.approx(expected_x, rel=1e-5)
        assert pred_y == pytest.approx(expected_y, rel=1e-5)

    def test_deceleration_prediction(self):
        """감속 운동 예측 테스트."""
        predictor = DynamicObstaclePredictor(
            model=PredictionModel.CONSTANT_ACCELERATION
        )

        # 초기 속도 1.0 m/s, 가속도 -0.5 m/s^2
        state = DynamicObstacleState(
            x=0.0, y=0.0, vx=1.0, vy=0.0, ax=-0.5, ay=0.0, radius=0.3
        )

        dt = 0.1
        steps = 20  # 2초
        predictions = predictor.predict_trajectory(state, dt, steps)

        # 2초 후 속도: v = 1.0 - 0.5*2.0 = 0.0
        # 2초 후 위치: x = 0 + 1.0*2.0 + 0.5*(-0.5)*2.0^2 = 2.0 - 1.0 = 1.0
        pred_x, pred_y, pred_radius = predictions[-1]
        assert pred_x == pytest.approx(1.0, rel=1e-5)


class TestCollisionTimePrediction:
    """충돌 시간 예측 테스트."""

    def test_head_on_collision(self):
        """정면 충돌 시간 예측 테스트."""
        predictor = DynamicObstaclePredictor(
            model=PredictionModel.CONSTANT_VELOCITY
        )

        # 장애물이 로봇을 향해 다가옴
        # 초기 거리 10m, 속도 2 m/s -> 충돌 시간 ~5초
        state = DynamicObstacleState(
            x=10.0, y=0.0, vx=-2.0, vy=0.0, radius=0.3
        )

        robot_x, robot_y = 0.0, 0.0
        robot_radius = 0.3
        safety_margin = 0.4

        collision_time = predictor.predict_collision_time(
            state, robot_x, robot_y, robot_radius, safety_margin
        )

        # 충돌 거리: 0.3 + 0.3 + 0.4 = 1.0m
        # 상대 속도: 2.0 m/s
        # 충돌 시간: (10.0 - 1.0) / 2.0 = 4.5초
        assert collision_time is not None
        assert collision_time == pytest.approx(4.5, rel=1e-2)

    def test_no_collision(self):
        """충돌 없음 테스트."""
        predictor = DynamicObstaclePredictor(
            model=PredictionModel.CONSTANT_VELOCITY
        )

        # 장애물이 로봇에서 멀어짐
        state = DynamicObstacleState(
            x=10.0, y=0.0, vx=2.0, vy=0.0, radius=0.3
        )

        robot_x, robot_y = 0.0, 0.0
        robot_radius = 0.3

        collision_time = predictor.predict_collision_time(
            state, robot_x, robot_y, robot_radius
        )

        assert collision_time is None

    def test_perpendicular_motion(self):
        """수직 운동 (충돌 없음) 테스트."""
        predictor = DynamicObstaclePredictor(
            model=PredictionModel.CONSTANT_VELOCITY
        )

        # 장애물이 로봇 앞을 지나감 (충분히 멀리)
        state = DynamicObstacleState(
            x=5.0, y=5.0, vx=0.0, vy=-1.0, radius=0.3
        )

        robot_x, robot_y = 0.0, 0.0
        robot_radius = 0.3

        collision_time = predictor.predict_collision_time(
            state, robot_x, robot_y, robot_radius
        )

        # 거리가 충분히 멀어 충돌 없음
        assert collision_time is None


class TestMultipleObstaclesPrediction:
    """여러 장애물 예측 테스트."""

    def test_multiple_obstacles_prediction(self):
        """여러 장애물 동시 예측 테스트."""
        predictor = DynamicObstaclePredictor(
            model=PredictionModel.CONSTANT_VELOCITY
        )

        obstacles = [
            DynamicObstacleState(x=0.0, y=0.0, vx=1.0, vy=0.0, radius=0.3),
            DynamicObstacleState(x=5.0, y=5.0, vx=0.0, vy=-1.0, radius=0.4),
            DynamicObstacleState(x=10.0, y=10.0, vx=-0.5, vy=-0.5, radius=0.5),
        ]

        dt = 0.1
        steps = 10
        predictions = predictor.predict_multiple_obstacles(obstacles, dt, steps)

        # 예측 개수 확인
        assert len(predictions) == 3

        # 각 장애물의 예측 스텝 수 확인
        for pred in predictions:
            assert len(pred) == steps


class TestStateHistory:
    """상태 히스토리 관리 테스트."""

    def test_update_state_history(self):
        """상태 히스토리 업데이트 테스트."""
        predictor = DynamicObstaclePredictor(
            model=PredictionModel.CONSTANT_VELOCITY
        )

        obstacle_id = 1
        state1 = DynamicObstacleState(
            x=0.0, y=0.0, vx=1.0, vy=0.0, radius=0.3, timestamp=0.0
        )
        state2 = DynamicObstacleState(
            x=0.1, y=0.0, vx=1.0, vy=0.0, radius=0.3, timestamp=0.1
        )

        predictor.update_state_history(obstacle_id, state1)
        predictor.update_state_history(obstacle_id, state2)

        assert len(predictor.state_history[obstacle_id]) == 2

    def test_max_history_limit(self):
        """최대 히스토리 개수 제한 테스트."""
        predictor = DynamicObstaclePredictor(
            model=PredictionModel.CONSTANT_VELOCITY
        )

        obstacle_id = 1
        max_history = 5

        # 10개의 상태 추가
        for i in range(10):
            state = DynamicObstacleState(
                x=float(i), y=0.0, vx=1.0, vy=0.0, radius=0.3, timestamp=float(i) * 0.1
            )
            predictor.update_state_history(obstacle_id, state, max_history)

        # 최대 5개만 유지되어야 함
        assert len(predictor.state_history[obstacle_id]) == max_history

    def test_estimate_acceleration(self):
        """가속도 추정 테스트."""
        predictor = DynamicObstaclePredictor(
            model=PredictionModel.CONSTANT_VELOCITY
        )

        obstacle_id = 1

        # 가속 운동 시뮬레이션
        # v(t) = v0 + a*t, a = 0.5 m/s^2
        state1 = DynamicObstacleState(
            x=0.0, y=0.0, vx=1.0, vy=0.0, radius=0.3, timestamp=0.0
        )
        state2 = DynamicObstacleState(
            x=0.105, y=0.0, vx=1.05, vy=0.0, radius=0.3, timestamp=0.1
        )

        predictor.update_state_history(obstacle_id, state1)
        predictor.update_state_history(obstacle_id, state2)

        ax, ay = predictor.estimate_acceleration(obstacle_id)

        # 가속도: (1.05 - 1.0) / 0.1 = 0.5 m/s^2
        assert ax == pytest.approx(0.5, rel=1e-5)
        assert ay == pytest.approx(0.0, rel=1e-5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
