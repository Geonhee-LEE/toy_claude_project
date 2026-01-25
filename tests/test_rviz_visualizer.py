"""
RVIZ Visualizer 단위 테스트.

MPCRVizVisualizer의 마커 생성 기능을 테스트합니다.
"""

import sys
import numpy as np
import pytest
from unittest.mock import Mock, MagicMock, patch

# ROS2 모듈 모킹
sys.modules['rclpy'] = MagicMock()
sys.modules['rclpy.node'] = MagicMock()
sys.modules['rclpy.time'] = MagicMock()
sys.modules['geometry_msgs'] = MagicMock()
sys.modules['geometry_msgs.msg'] = MagicMock()
sys.modules['std_msgs'] = MagicMock()
sys.modules['std_msgs.msg'] = MagicMock()
sys.modules['visualization_msgs'] = MagicMock()
sys.modules['visualization_msgs.msg'] = MagicMock()
sys.modules['nav_msgs'] = MagicMock()
sys.modules['nav_msgs.msg'] = MagicMock()

# Marker 상수 정의
class MockMarker:
    LINE_STRIP = 4
    SPHERE = 2
    CYLINDER = 3
    ARROW = 0
    TEXT_VIEW_FACING = 9
    ADD = 0
    DELETEALL = 3

sys.modules['visualization_msgs.msg'].Marker = MockMarker

from mpc_controller.planners.obstacle_avoidance import Obstacle


class TestMPCRVizVisualizer:
    """RVIZ 시각화 테스트."""

    @pytest.fixture
    def mock_node(self):
        """Mock ROS2 노드 생성."""
        node = Mock()
        clock = Mock()
        clock.now.return_value.to_msg.return_value = Time(seconds=0).to_msg()
        node.get_clock.return_value = clock
        return node

    @pytest.fixture
    def visualizer(self, mock_node):
        """MPCRVizVisualizer 인스턴스 생성."""
        return MPCRVizVisualizer(node=mock_node, frame_id="odom")

    @pytest.fixture
    def sample_mpc_info(self):
        """샘플 MPC 정보 생성."""
        N = 10
        predicted_trajectory = np.array([
            [i * 0.1, i * 0.05, i * 0.01] for i in range(N + 1)
        ])
        predicted_controls = np.array([
            [0.5 + i * 0.01, 0.1 + i * 0.005] for i in range(N)
        ])

        return {
            'predicted_trajectory': predicted_trajectory,
            'predicted_controls': predicted_controls,
            'soft_constraints': {
                'has_violations': False,
                'max_velocity_violation': 0.0,
                'max_acceleration_violation': 0.0,
            },
        }

    @pytest.fixture
    def sample_obstacles(self):
        """샘플 장애물 리스트 생성."""
        return [
            Obstacle(x=2.0, y=1.0, radius=0.5, obstacle_type="static"),
            Obstacle(x=3.0, y=2.0, radius=0.3, obstacle_type="dynamic",
                    velocity_x=0.2, velocity_y=0.1),
        ]

    def test_initialization(self, mock_node):
        """초기화 테스트."""
        visualizer = MPCRVizVisualizer(node=mock_node, frame_id="test_frame")
        assert visualizer.node == mock_node
        assert visualizer.frame_id == "test_frame"

    def test_create_trajectory_line_marker(self, visualizer, sample_mpc_info, mock_node):
        """예측 궤적 라인 마커 생성 테스트."""
        timestamp = mock_node.get_clock().now().to_msg()
        predicted_states = sample_mpc_info['predicted_trajectory']

        marker = visualizer._create_trajectory_line_marker(
            predicted_states, timestamp
        )

        assert isinstance(marker, Marker)
        assert marker.type == Marker.LINE_STRIP
        assert marker.ns == MPCRVizVisualizer.NS_PREDICTED_TRAJ
        assert len(marker.points) == len(predicted_states)
        assert marker.color.g == 1.0  # 녹색

    def test_create_trajectory_point_markers(self, visualizer, sample_mpc_info, mock_node):
        """예측 궤적 포인트 마커 생성 테스트."""
        timestamp = mock_node.get_clock().now().to_msg()
        predicted_states = sample_mpc_info['predicted_trajectory']

        markers = visualizer._create_trajectory_point_markers(
            predicted_states, timestamp
        )

        assert isinstance(markers, list)
        assert len(markers) == len(predicted_states)

        # 첫 번째 마커 검증
        first_marker = markers[0]
        assert first_marker.type == Marker.SPHERE
        assert first_marker.ns == MPCRVizVisualizer.NS_TRAJ_POINTS
        # 시간에 따라 색상이 변해야 함 (녹색 -> 빨강)
        assert first_marker.color.g > 0.5  # 초반은 녹색 성분이 많음

        # 마지막 마커 검증
        last_marker = markers[-1]
        assert last_marker.color.r > 0.5  # 후반은 빨강 성분이 많음

    def test_create_violation_marker(self, visualizer, mock_node):
        """제약조건 위반 마커 생성 테스트."""
        timestamp = mock_node.get_clock().now().to_msg()
        current_state = np.array([1.0, 2.0, 0.5])
        soft_info = {
            'has_violations': True,
            'max_velocity_violation': 0.123,
            'max_acceleration_violation': 0.456,
        }

        marker = visualizer._create_violation_marker(
            current_state, soft_info, timestamp
        )

        assert isinstance(marker, Marker)
        assert marker.type == Marker.TEXT_VIEW_FACING
        assert marker.ns == MPCRVizVisualizer.NS_CONSTRAINT_VIOLATIONS
        assert marker.color.r == 1.0  # 빨강
        assert '0.123' in marker.text
        assert '0.456' in marker.text

    def test_create_velocity_constraint_markers(self, visualizer, mock_node):
        """속도 제약 경계 마커 생성 테스트."""
        timestamp = mock_node.get_clock().now().to_msg()
        current_state = np.array([0.0, 0.0, 0.0])
        robot_params = {'v_max': 1.0, 'omega_max': 1.5}

        markers = visualizer._create_velocity_constraint_markers(
            current_state, robot_params, timestamp
        )

        assert isinstance(markers, list)
        assert len(markers) > 0

        marker = markers[0]
        assert marker.type == Marker.CYLINDER
        assert marker.ns == MPCRVizVisualizer.NS_VELOCITY_LIMITS
        # 크기는 v_max * 2.0
        assert marker.scale.x == pytest.approx(2.0)

    def test_create_acceleration_markers(self, visualizer, sample_mpc_info, mock_node):
        """가속도 벡터 마커 생성 테스트."""
        timestamp = mock_node.get_clock().now().to_msg()
        predicted_states = sample_mpc_info['predicted_trajectory']
        predicted_controls = sample_mpc_info['predicted_controls']

        markers = visualizer._create_acceleration_markers(
            predicted_states, predicted_controls, timestamp
        )

        assert isinstance(markers, list)
        # subsample=3이므로 일부만 생성됨
        for marker in markers:
            assert marker.type == Marker.ARROW
            assert marker.ns == MPCRVizVisualizer.NS_ACCELERATION_ARROWS
            assert len(marker.points) == 2  # start, end

    def test_create_obstacle_markers(self, visualizer, sample_obstacles, mock_node):
        """장애물 마커 생성 테스트."""
        timestamp = mock_node.get_clock().now().to_msg()

        markers = visualizer._create_obstacle_markers(
            sample_obstacles, timestamp
        )

        assert isinstance(markers, list)
        assert len(markers) == len(sample_obstacles)

        # 정적 장애물 검증
        static_marker = markers[0]
        assert static_marker.type == Marker.CYLINDER
        assert static_marker.ns == MPCRVizVisualizer.NS_OBSTACLES
        assert static_marker.color.r == 1.0  # 빨강

        # 동적 장애물 검증
        dynamic_marker = markers[1]
        assert dynamic_marker.color.r == 1.0
        assert dynamic_marker.color.g == pytest.approx(0.5)  # 주황

    def test_create_safety_zone_markers(self, visualizer, sample_obstacles, mock_node):
        """안전 영역 마커 생성 테스트."""
        timestamp = mock_node.get_clock().now().to_msg()
        safety_margin = 0.3

        markers = visualizer._create_safety_zone_markers(
            sample_obstacles, timestamp, safety_margin
        )

        assert isinstance(markers, list)
        assert len(markers) == len(sample_obstacles)

        marker = markers[0]
        assert marker.type == Marker.CYLINDER
        assert marker.ns == MPCRVizVisualizer.NS_SAFETY_ZONES
        # 크기는 (radius + safety_margin) * 2
        expected_size = (sample_obstacles[0].radius + safety_margin) * 2.0
        assert marker.scale.x == pytest.approx(expected_size)
        assert marker.color.r == 1.0  # 노랑
        assert marker.color.g == 1.0

    def test_create_dynamic_obstacle_predictions(self, visualizer, sample_obstacles, mock_node):
        """동적 장애물 예측 경로 마커 생성 테스트."""
        timestamp = mock_node.get_clock().now().to_msg()

        markers = visualizer._create_dynamic_obstacle_predictions(
            sample_obstacles, timestamp, prediction_steps=10, dt=0.1
        )

        assert isinstance(markers, list)
        # 동적 장애물 하나만 있으므로 1개 생성
        assert len(markers) == 1

        marker = markers[0]
        assert marker.type == Marker.LINE_STRIP
        assert marker.ns == MPCRVizVisualizer.NS_DYNAMIC_PREDICTIONS
        # prediction_steps + 1개 포인트
        assert len(marker.points) == 11

    def test_create_marker_array(self, visualizer, sample_mpc_info, sample_obstacles, mock_node):
        """전체 마커 배열 생성 테스트."""
        current_state = np.array([0.0, 0.0, 0.0])
        robot_params = {'v_max': 1.0, 'omega_max': 1.5}

        marker_array = visualizer.create_marker_array(
            current_state=current_state,
            mpc_info=sample_mpc_info,
            obstacles=sample_obstacles,
            robot_params=robot_params,
        )

        assert isinstance(marker_array, MarkerArray)
        assert len(marker_array.markers) > 0

        # 각 네임스페이스별 마커가 포함되어 있는지 확인
        namespaces = set(marker.ns for marker in marker_array.markers)
        assert MPCRVizVisualizer.NS_PREDICTED_TRAJ in namespaces
        assert MPCRVizVisualizer.NS_TRAJ_POINTS in namespaces
        assert MPCRVizVisualizer.NS_OBSTACLES in namespaces

    def test_create_marker_array_with_violations(self, visualizer, sample_mpc_info, mock_node):
        """제약조건 위반이 있는 경우 마커 배열 생성 테스트."""
        current_state = np.array([0.0, 0.0, 0.0])
        robot_params = {'v_max': 1.0, 'omega_max': 1.5}

        # 제약조건 위반 추가
        sample_mpc_info['soft_constraints']['has_violations'] = True
        sample_mpc_info['soft_constraints']['max_velocity_violation'] = 0.5

        marker_array = visualizer.create_marker_array(
            current_state=current_state,
            mpc_info=sample_mpc_info,
            robot_params=robot_params,
        )

        # 위반 마커가 포함되어 있는지 확인
        namespaces = set(marker.ns for marker in marker_array.markers)
        assert MPCRVizVisualizer.NS_CONSTRAINT_VIOLATIONS in namespaces

    def test_create_delete_all_markers(self, visualizer):
        """모든 마커 삭제 테스트."""
        marker_array = visualizer.create_delete_all_markers()

        assert isinstance(marker_array, MarkerArray)
        assert len(marker_array.markers) > 0

        # 모든 마커가 DELETEALL 액션인지 확인
        for marker in marker_array.markers:
            assert marker.action == Marker.DELETEALL

    def test_empty_obstacles(self, visualizer, sample_mpc_info, mock_node):
        """장애물이 없는 경우 테스트."""
        current_state = np.array([0.0, 0.0, 0.0])
        robot_params = {'v_max': 1.0, 'omega_max': 1.5}

        marker_array = visualizer.create_marker_array(
            current_state=current_state,
            mpc_info=sample_mpc_info,
            obstacles=[],  # 빈 리스트
            robot_params=robot_params,
        )

        # 장애물 관련 마커가 없어야 함
        namespaces = set(marker.ns for marker in marker_array.markers)
        assert MPCRVizVisualizer.NS_OBSTACLES not in namespaces
        assert MPCRVizVisualizer.NS_SAFETY_ZONES not in namespaces

    def test_no_dynamic_obstacles(self, visualizer, mock_node):
        """동적 장애물이 없는 경우 테스트."""
        timestamp = mock_node.get_clock().now().to_msg()
        static_obstacles = [
            Obstacle(x=1.0, y=1.0, radius=0.5, obstacle_type="static"),
        ]

        markers = visualizer._create_dynamic_obstacle_predictions(
            static_obstacles, timestamp
        )

        # 정적 장애물만 있으므로 동적 예측 마커 없음
        assert len(markers) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
