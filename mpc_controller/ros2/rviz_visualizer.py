"""
RVIZ Visualization Module for MPC Controller.

MPC 컨트롤러의 예측 궤적, 제약조건, 장애물을 RVIZ에서 시각화합니다.
"""

from typing import List, Optional, Dict
import numpy as np
from rclpy.node import Node
from rclpy.time import Time
from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray

from mpc_controller.planners.obstacle_avoidance import Obstacle


class MPCRVizVisualizer:
    """
    MPC 컨트롤러용 RVIZ 시각화 클래스.

    다양한 MPC 정보를 RVIZ 마커로 변환하여 시각화합니다:
    - 예측 궤적 (Predicted Trajectory)
    - 속도/가속도 제약조건 (Velocity/Acceleration Constraints)
    - 장애물 (Obstacles)
    - 안전 영역 (Safety Zones)
    """

    # 마커 네임스페이스
    NS_PREDICTED_TRAJ = "predicted_trajectory"
    NS_TRAJ_POINTS = "trajectory_points"
    NS_VELOCITY_LIMITS = "velocity_limits"
    NS_ACCELERATION_ARROWS = "acceleration_arrows"
    NS_CONSTRAINT_VIOLATIONS = "constraint_violations"
    NS_OBSTACLES = "obstacles"
    NS_SAFETY_ZONES = "safety_zones"
    NS_DYNAMIC_PREDICTIONS = "dynamic_predictions"

    def __init__(
        self,
        node: Node,
        frame_id: str = "odom",
    ):
        """
        RVIZ 시각화 초기화.

        Args:
            node: ROS2 노드
            frame_id: 시각화 프레임 ID
        """
        self.node = node
        self.frame_id = frame_id

    def create_marker_array(
        self,
        current_state: np.ndarray,
        mpc_info: Dict,
        obstacles: Optional[List[Obstacle]] = None,
        robot_params: Optional[Dict] = None,
    ) -> MarkerArray:
        """
        모든 시각화 마커를 생성합니다.

        Args:
            current_state: 현재 로봇 상태 [x, y, theta]
            mpc_info: MPC 계산 정보 딕셔너리
            obstacles: 장애물 리스트
            robot_params: 로봇 파라미터 (v_max, omega_max 등)

        Returns:
            MarkerArray containing all visualization markers
        """
        marker_array = MarkerArray()
        timestamp = self.node.get_clock().now().to_msg()

        # 1. 예측 궤적 라인
        traj_line_marker = self._create_trajectory_line_marker(
            mpc_info['predicted_trajectory'],
            timestamp,
        )
        marker_array.markers.append(traj_line_marker)

        # 2. 예측 궤적 포인트 (그라데이션)
        traj_point_markers = self._create_trajectory_point_markers(
            mpc_info['predicted_trajectory'],
            timestamp,
        )
        marker_array.markers.extend(traj_point_markers)

        # 3. 제약조건 위반 표시
        soft_info = mpc_info.get('soft_constraints', {})
        if soft_info.get('has_violations', False):
            violation_marker = self._create_violation_marker(
                current_state,
                soft_info,
                timestamp,
            )
            marker_array.markers.append(violation_marker)

        # 4. 속도 제약 경계 시각화
        if robot_params is not None:
            velocity_markers = self._create_velocity_constraint_markers(
                current_state,
                robot_params,
                timestamp,
            )
            marker_array.markers.extend(velocity_markers)

        # 5. 가속도 벡터 시각화
        if 'predicted_controls' in mpc_info:
            accel_markers = self._create_acceleration_markers(
                mpc_info['predicted_trajectory'],
                mpc_info['predicted_controls'],
                timestamp,
            )
            marker_array.markers.extend(accel_markers)

        # 6. 장애물 시각화
        if obstacles is not None and len(obstacles) > 0:
            obstacle_markers = self._create_obstacle_markers(
                obstacles,
                timestamp,
            )
            marker_array.markers.extend(obstacle_markers)

            # 7. 안전 영역 시각화
            safety_markers = self._create_safety_zone_markers(
                obstacles,
                timestamp,
            )
            marker_array.markers.extend(safety_markers)

            # 8. 동적 장애물 예측 경로
            dynamic_markers = self._create_dynamic_obstacle_predictions(
                obstacles,
                timestamp,
            )
            marker_array.markers.extend(dynamic_markers)

        return marker_array

    def _create_trajectory_line_marker(
        self,
        predicted_states: np.ndarray,
        timestamp: Time,
    ) -> Marker:
        """
        예측 궤적을 LINE_STRIP 마커로 생성.

        Args:
            predicted_states: 예측된 상태 배열 (N+1, 3)
            timestamp: 타임스탬프

        Returns:
            LINE_STRIP marker
        """
        marker = Marker()
        marker.header.stamp = timestamp
        marker.header.frame_id = self.frame_id
        marker.ns = self.NS_PREDICTED_TRAJ
        marker.id = 0
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD

        marker.scale.x = 0.05  # 선 두께
        marker.color = ColorRGBA(r=0.0, g=1.0, b=0.0, a=0.8)

        for state in predicted_states:
            point = Point()
            point.x = float(state[0])
            point.y = float(state[1])
            point.z = 0.05  # 지면 위로 살짝 올림
            marker.points.append(point)

        return marker

    def _create_trajectory_point_markers(
        self,
        predicted_states: np.ndarray,
        timestamp: Time,
    ) -> List[Marker]:
        """
        예측 궤적의 각 포인트를 SPHERE 마커로 생성 (시간에 따른 색상 그라데이션).

        Args:
            predicted_states: 예측된 상태 배열 (N+1, 3)
            timestamp: 타임스탬프

        Returns:
            List of SPHERE markers
        """
        markers = []
        N = len(predicted_states)

        for i, state in enumerate(predicted_states):
            marker = Marker()
            marker.header.stamp = timestamp
            marker.header.frame_id = self.frame_id
            marker.ns = self.NS_TRAJ_POINTS
            marker.id = i
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD

            marker.pose.position.x = float(state[0])
            marker.pose.position.y = float(state[1])
            marker.pose.position.z = 0.1

            marker.pose.orientation.w = 1.0

            # 시간에 따른 크기 감소 (가까운 미래 = 크게)
            scale = 0.15 * (1.0 - 0.7 * i / N)
            marker.scale.x = scale
            marker.scale.y = scale
            marker.scale.z = scale

            # 색상 그라데이션: 녹색 -> 노랑 -> 빨강
            # i=0 (현재): 녹색, i=N (먼 미래): 빨강
            ratio = i / max(N - 1, 1)
            marker.color = ColorRGBA(
                r=float(ratio),
                g=float(1.0 - ratio * 0.5),
                b=0.0,
                a=0.7,
            )

            markers.append(marker)

        return markers

    def _create_violation_marker(
        self,
        current_state: np.ndarray,
        soft_info: Dict,
        timestamp: Time,
    ) -> Marker:
        """
        제약조건 위반 정보를 TEXT_VIEW_FACING 마커로 생성.

        Args:
            current_state: 현재 로봇 상태 [x, y, theta]
            soft_info: Soft constraint 정보
            timestamp: 타임스탬프

        Returns:
            TEXT_VIEW_FACING marker
        """
        marker = Marker()
        marker.header.stamp = timestamp
        marker.header.frame_id = self.frame_id
        marker.ns = self.NS_CONSTRAINT_VIOLATIONS
        marker.id = 0
        marker.type = Marker.TEXT_VIEW_FACING
        marker.action = Marker.ADD

        marker.pose.position.x = float(current_state[0])
        marker.pose.position.y = float(current_state[1])
        marker.pose.position.z = 1.5  # 로봇 위에 표시

        marker.scale.z = 0.3  # 텍스트 크기
        marker.color = ColorRGBA(r=1.0, g=0.0, b=0.0, a=1.0)

        max_vel_viol = soft_info.get('max_velocity_violation', 0.0)
        max_acc_viol = soft_info.get('max_acceleration_violation', 0.0)
        marker.text = f'⚠️ 제약 위반\nVel: {max_vel_viol:.3f}\nAcc: {max_acc_viol:.3f}'

        return marker

    def _create_velocity_constraint_markers(
        self,
        current_state: np.ndarray,
        robot_params: Dict,
        timestamp: Time,
    ) -> List[Marker]:
        """
        속도 제약 경계를 시각화 (로봇 주변 원통형).

        Args:
            current_state: 현재 로봇 상태 [x, y, theta]
            robot_params: 로봇 파라미터 {'v_max', 'omega_max'}
            timestamp: 타임스탬프

        Returns:
            List of CYLINDER markers
        """
        markers = []

        # 선속도 한계를 반지름으로 표현
        v_max = robot_params.get('v_max', 1.0)

        marker = Marker()
        marker.header.stamp = timestamp
        marker.header.frame_id = self.frame_id
        marker.ns = self.NS_VELOCITY_LIMITS
        marker.id = 0
        marker.type = Marker.CYLINDER
        marker.action = Marker.ADD

        marker.pose.position.x = float(current_state[0])
        marker.pose.position.y = float(current_state[1])
        marker.pose.position.z = 0.01  # 지면 레벨
        marker.pose.orientation.w = 1.0

        # 최대 속도에 비례하는 원통 크기
        marker.scale.x = v_max * 2.0  # 직경
        marker.scale.y = v_max * 2.0
        marker.scale.z = 0.02  # 얇은 원판

        marker.color = ColorRGBA(r=0.0, g=0.5, b=1.0, a=0.2)

        markers.append(marker)

        return markers

    def _create_acceleration_markers(
        self,
        predicted_states: np.ndarray,
        predicted_controls: np.ndarray,
        timestamp: Time,
        subsample: int = 3,
    ) -> List[Marker]:
        """
        가속도 벡터를 ARROW 마커로 시각화.

        Args:
            predicted_states: 예측된 상태 배열 (N+1, 3)
            predicted_controls: 예측된 제어 입력 (N, 2) [v, omega]
            timestamp: 타임스탬프
            subsample: 표시할 화살표 간격 (성능 최적화)

        Returns:
            List of ARROW markers
        """
        markers = []
        N = len(predicted_controls)

        for i in range(0, N, subsample):
            if i == 0:
                continue  # 첫 번째는 가속도 계산 불가

            v_curr = predicted_controls[i, 0]
            v_prev = predicted_controls[i - 1, 0]

            # 가속도 계산 (간단히 속도 차이로 표현)
            accel = v_curr - v_prev

            # 가속도가 너무 작으면 스킵
            if abs(accel) < 0.01:
                continue

            marker = Marker()
            marker.header.stamp = timestamp
            marker.header.frame_id = self.frame_id
            marker.ns = self.NS_ACCELERATION_ARROWS
            marker.id = i
            marker.type = Marker.ARROW
            marker.action = Marker.ADD

            # 화살표 시작점
            state = predicted_states[i]
            theta = state[2]

            start = Point()
            start.x = float(state[0])
            start.y = float(state[1])
            start.z = 0.2

            # 화살표 끝점 (진행 방향으로 가속도 크기만큼)
            arrow_length = abs(accel) * 0.5  # 스케일 조정
            end = Point()
            end.x = start.x + arrow_length * np.cos(theta)
            end.y = start.y + arrow_length * np.sin(theta)
            end.z = 0.2

            marker.points = [start, end]

            # 화살표 크기
            marker.scale.x = 0.05  # 축 두께
            marker.scale.y = 0.08  # 화살촉 폭
            marker.scale.z = 0.08  # 화살촉 높이

            # 색상: 가속(초록), 감속(빨강)
            if accel > 0:
                marker.color = ColorRGBA(r=0.0, g=1.0, b=0.0, a=0.8)
            else:
                marker.color = ColorRGBA(r=1.0, g=0.5, b=0.0, a=0.8)

            markers.append(marker)

        return markers

    def _create_obstacle_markers(
        self,
        obstacles: List[Obstacle],
        timestamp: Time,
    ) -> List[Marker]:
        """
        장애물을 CYLINDER 마커로 시각화.

        Args:
            obstacles: 장애물 리스트
            timestamp: 타임스탬프

        Returns:
            List of CYLINDER markers
        """
        markers = []

        for i, obs in enumerate(obstacles):
            marker = Marker()
            marker.header.stamp = timestamp
            marker.header.frame_id = self.frame_id
            marker.ns = self.NS_OBSTACLES
            marker.id = i
            marker.type = Marker.CYLINDER
            marker.action = Marker.ADD

            marker.pose.position.x = obs.x
            marker.pose.position.y = obs.y
            marker.pose.position.z = 0.5  # 높이 중간
            marker.pose.orientation.w = 1.0

            marker.scale.x = obs.radius * 2.0  # 직경
            marker.scale.y = obs.radius * 2.0
            marker.scale.z = 1.0  # 높이

            # 색상: 정적(빨강), 동적(주황)
            if obs.obstacle_type == "static":
                marker.color = ColorRGBA(r=1.0, g=0.0, b=0.0, a=0.7)
            else:
                marker.color = ColorRGBA(r=1.0, g=0.5, b=0.0, a=0.7)

            markers.append(marker)

        return markers

    def _create_safety_zone_markers(
        self,
        obstacles: List[Obstacle],
        timestamp: Time,
        safety_margin: float = 0.3,
    ) -> List[Marker]:
        """
        장애물 안전 영역을 CYLINDER 마커로 시각화.

        Args:
            obstacles: 장애물 리스트
            timestamp: 타임스탬프
            safety_margin: 안전 마진 거리

        Returns:
            List of CYLINDER markers
        """
        markers = []

        for i, obs in enumerate(obstacles):
            marker = Marker()
            marker.header.stamp = timestamp
            marker.header.frame_id = self.frame_id
            marker.ns = self.NS_SAFETY_ZONES
            marker.id = i
            marker.type = Marker.CYLINDER
            marker.action = Marker.ADD

            marker.pose.position.x = obs.x
            marker.pose.position.y = obs.y
            marker.pose.position.z = 0.01  # 지면 레벨
            marker.pose.orientation.w = 1.0

            # 장애물 반지름 + 안전 마진
            total_radius = obs.radius + safety_margin
            marker.scale.x = total_radius * 2.0
            marker.scale.y = total_radius * 2.0
            marker.scale.z = 0.02  # 얇은 원판

            marker.color = ColorRGBA(r=1.0, g=1.0, b=0.0, a=0.3)

            markers.append(marker)

        return markers

    def _create_dynamic_obstacle_predictions(
        self,
        obstacles: List[Obstacle],
        timestamp: Time,
        prediction_steps: int = 10,
        dt: float = 0.1,
    ) -> List[Marker]:
        """
        동적 장애물의 예측 경로를 LINE_STRIP 마커로 시각화.

        Args:
            obstacles: 장애물 리스트
            timestamp: 타임스탬프
            prediction_steps: 예측 스텝 수
            dt: 시간 간격

        Returns:
            List of LINE_STRIP markers
        """
        markers = []

        dynamic_obs_count = 0
        for obs in obstacles:
            if obs.obstacle_type != "dynamic":
                continue

            # 속도가 없으면 스킵
            if abs(obs.velocity_x) < 0.01 and abs(obs.velocity_y) < 0.01:
                continue

            marker = Marker()
            marker.header.stamp = timestamp
            marker.header.frame_id = self.frame_id
            marker.ns = self.NS_DYNAMIC_PREDICTIONS
            marker.id = dynamic_obs_count
            marker.type = Marker.LINE_STRIP
            marker.action = Marker.ADD

            marker.scale.x = 0.03  # 선 두께
            marker.color = ColorRGBA(r=1.0, g=0.5, b=0.0, a=0.6)

            # 예측 경로 포인트 생성
            current_x, current_y = obs.x, obs.y
            for step in range(prediction_steps + 1):
                point = Point()
                point.x = current_x + obs.velocity_x * dt * step
                point.y = current_y + obs.velocity_y * dt * step
                point.z = 0.5  # 장애물 중간 높이
                marker.points.append(point)

            markers.append(marker)
            dynamic_obs_count += 1

        return markers

    def create_delete_all_markers(self) -> MarkerArray:
        """
        모든 마커를 삭제하는 MarkerArray 생성.

        Returns:
            DELETE ALL action을 가진 MarkerArray
        """
        marker_array = MarkerArray()

        # 각 네임스페이스마다 삭제 마커 생성
        namespaces = [
            self.NS_PREDICTED_TRAJ,
            self.NS_TRAJ_POINTS,
            self.NS_VELOCITY_LIMITS,
            self.NS_ACCELERATION_ARROWS,
            self.NS_CONSTRAINT_VIOLATIONS,
            self.NS_OBSTACLES,
            self.NS_SAFETY_ZONES,
            self.NS_DYNAMIC_PREDICTIONS,
        ]

        for i, ns in enumerate(namespaces):
            marker = Marker()
            marker.header.frame_id = self.frame_id
            marker.ns = ns
            marker.id = 0
            marker.action = Marker.DELETEALL
            marker_array.markers.append(marker)

        return marker_array
