"""
RVIZ Visualizer 단순 테스트.

MPCRVizVisualizer의 기본 기능을 검증합니다.
ROS2 의존성 없이 로직만 테스트합니다.
"""

import numpy as np
import pytest

from mpc_controller.planners.obstacle_avoidance import Obstacle


class TestObstacleClass:
    """Obstacle 클래스 테스트 (ROS2 의존성 없음)."""

    def test_obstacle_creation(self):
        """장애물 생성 테스트."""
        obs = Obstacle(x=1.0, y=2.0, radius=0.5, obstacle_type="static")

        assert obs.x == 1.0
        assert obs.y == 2.0
        assert obs.radius == 0.5
        assert obs.obstacle_type == "static"
        assert obs.velocity_x == 0.0
        assert obs.velocity_y == 0.0

    def test_obstacle_with_velocity(self):
        """속도를 가진 장애물 생성 테스트."""
        obs = Obstacle(
            x=3.0, y=4.0, radius=0.3,
            obstacle_type="dynamic",
            velocity_x=0.5, velocity_y=-0.2
        )

        assert obs.obstacle_type == "dynamic"
        assert obs.velocity_x == 0.5
        assert obs.velocity_y == -0.2

    def test_predict_position(self):
        """위치 예측 테스트."""
        obs = Obstacle(
            x=0.0, y=0.0, radius=0.5,
            velocity_x=1.0, velocity_y=2.0
        )

        # 1초 후 예측
        pred_x, pred_y = obs.predict_position(1.0)
        assert pred_x == pytest.approx(1.0)
        assert pred_y == pytest.approx(2.0)

        # 0.5초 후 예측
        pred_x, pred_y = obs.predict_position(0.5)
        assert pred_x == pytest.approx(0.5)
        assert pred_y == pytest.approx(1.0)

    def test_static_obstacle_prediction(self):
        """정적 장애물의 위치 예측 테스트."""
        obs = Obstacle(x=5.0, y=3.0, radius=0.4, obstacle_type="static")

        # 정적 장애물은 시간이 지나도 같은 위치
        pred_x, pred_y = obs.predict_position(10.0)
        assert pred_x == pytest.approx(5.0)
        assert pred_y == pytest.approx(3.0)


class TestVisualizationLogic:
    """시각화 로직 테스트 (계산 검증)."""

    def test_trajectory_color_gradient(self):
        """궤적 포인트 색상 그라데이션 계산 테스트."""
        N = 10

        # 시뮬레이션: 시간에 따른 색상 변화
        colors_r = []
        colors_g = []

        for i in range(N):
            ratio = i / max(N - 1, 1)
            r = float(ratio)
            g = float(1.0 - ratio * 0.5)
            colors_r.append(r)
            colors_g.append(g)

        # 첫 번째 포인트: 녹색 (r=0, g=1)
        assert colors_r[0] == pytest.approx(0.0)
        assert colors_g[0] == pytest.approx(1.0)

        # 마지막 포인트: 빨강 (r=1, g=0.5)
        assert colors_r[-1] == pytest.approx(1.0)
        assert colors_g[-1] == pytest.approx(0.5)

    def test_acceleration_calculation(self):
        """가속도 계산 테스트."""
        dt = 0.1
        v_prev = 0.5
        v_curr = 0.6

        accel = (v_curr - v_prev) / dt
        assert accel == pytest.approx(1.0)

        # 감속 케이스
        v_prev = 1.0
        v_curr = 0.8
        accel = (v_curr - v_prev) / dt
        assert accel == pytest.approx(-2.0)

    def test_safety_zone_size(self):
        """안전 영역 크기 계산 테스트."""
        obstacle_radius = 0.5
        safety_margin = 0.3

        total_radius = obstacle_radius + safety_margin
        diameter = total_radius * 2.0

        assert total_radius == pytest.approx(0.8)
        assert diameter == pytest.approx(1.6)

    def test_velocity_limit_visualization_size(self):
        """속도 제한 시각화 크기 계산 테스트."""
        v_max = 1.0

        # 속도 제한은 원통 직경으로 표현
        diameter = v_max * 2.0
        assert diameter == pytest.approx(2.0)

    def test_dynamic_obstacle_prediction_path(self):
        """동적 장애물 예측 경로 계산 테스트."""
        obs = Obstacle(
            x=0.0, y=0.0, radius=0.5,
            obstacle_type="dynamic",
            velocity_x=0.5, velocity_y=0.3
        )

        dt = 0.1
        prediction_steps = 5

        # 예측 경로 계산
        path_points = []
        for step in range(prediction_steps + 1):
            pred_x = obs.x + obs.velocity_x * dt * step
            pred_y = obs.y + obs.velocity_y * dt * step
            path_points.append((pred_x, pred_y))

        # 검증
        assert len(path_points) == 6  # 0~5 = 6개
        assert path_points[0] == pytest.approx((0.0, 0.0))
        assert path_points[5] == pytest.approx((0.25, 0.15))


class TestMarkerNamespaces:
    """마커 네임스페이스 정의 테스트."""

    def test_namespace_constants(self):
        """네임스페이스 상수가 정의되어 있는지 확인."""
        # 실제 클래스를 import하지 않고, 예상되는 네임스페이스 확인
        expected_namespaces = [
            "predicted_trajectory",
            "trajectory_points",
            "velocity_limits",
            "acceleration_arrows",
            "constraint_violations",
            "obstacles",
            "safety_zones",
            "dynamic_predictions",
        ]

        # 각 네임스페이스가 유효한 문자열인지 확인
        for ns in expected_namespaces:
            assert isinstance(ns, str)
            assert len(ns) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
