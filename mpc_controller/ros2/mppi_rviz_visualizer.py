"""MPPI RVIZ 시각화 모듈.

MPPI 컨트롤러의 샘플 궤적, 가중 궤적, 비용 히트맵을 RVIZ에서 시각화합니다.
"""

from typing import Dict, Optional

import numpy as np
from rclpy.node import Node
from rclpy.time import Time
from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray


class MPPIRVizVisualizer:
    """MPPI 컨트롤러용 RVIZ 시각화 클래스.

    시각화 요소:
    - 샘플 궤적 (LINE_STRIP, 투명도 = 가중치)
    - 가중 평균 궤적 (굵은 시안 라인)
    - 최적 샘플 (마젠타)
    - 비용 히트맵 (초록 → 빨강 그라데이션)
    - 온도/ESS 텍스트
    """

    NS_SAMPLE_TRAJ = "mppi_sample_trajectories"
    NS_WEIGHTED_TRAJ = "mppi_weighted_trajectory"
    NS_BEST_TRAJ = "mppi_best_trajectory"
    NS_COST_HEATMAP = "mppi_cost_heatmap"
    NS_INFO_TEXT = "mppi_info_text"

    MAX_DISPLAY_SAMPLES = 50  # 성능을 위해 표시할 최대 샘플 수

    def __init__(
        self,
        node: Node,
        frame_id: str = "odom",
    ):
        self.node = node
        self.frame_id = frame_id

    def create_marker_array(
        self,
        current_state: np.ndarray,
        mppi_info: Dict,
    ) -> MarkerArray:
        """모든 MPPI 시각화 마커를 생성합니다.

        Args:
            current_state: 현재 로봇 상태 [x, y, theta]
            mppi_info: MPPI compute_control의 info dict

        Returns:
            MarkerArray
        """
        marker_array = MarkerArray()
        timestamp = self.node.get_clock().now().to_msg()

        # 1. 샘플 궤적 (투명도 = 가중치)
        if "sample_trajectories" in mppi_info and "sample_weights" in mppi_info:
            sample_markers = self._create_sample_trajectory_markers(
                mppi_info["sample_trajectories"],
                mppi_info["sample_weights"],
                mppi_info.get("sample_costs", None),
                timestamp,
            )
            marker_array.markers.extend(sample_markers)

        # 2. 가중 평균 궤적 (시안)
        if "predicted_trajectory" in mppi_info:
            weighted_marker = self._create_weighted_trajectory_marker(
                mppi_info["predicted_trajectory"],
                timestamp,
            )
            marker_array.markers.append(weighted_marker)

        # 3. 최적 샘플 궤적 (마젠타)
        if "best_trajectory" in mppi_info:
            best_marker = self._create_best_trajectory_marker(
                mppi_info["best_trajectory"],
                timestamp,
            )
            marker_array.markers.append(best_marker)

        # 4. 비용 히트맵 (초록→빨강)
        if "sample_trajectories" in mppi_info and "sample_costs" in mppi_info:
            heatmap_markers = self._create_cost_heatmap_markers(
                mppi_info["sample_trajectories"],
                mppi_info["sample_costs"],
                timestamp,
            )
            marker_array.markers.extend(heatmap_markers)

        # 5. 온도/ESS 텍스트
        info_marker = self._create_info_text_marker(
            current_state,
            mppi_info,
            timestamp,
        )
        marker_array.markers.append(info_marker)

        return marker_array

    def _create_sample_trajectory_markers(
        self,
        sample_trajectories: np.ndarray,
        sample_weights: np.ndarray,
        sample_costs: Optional[np.ndarray],
        timestamp: Time,
    ) -> list:
        """샘플 궤적을 LINE_STRIP 마커로 생성 (투명도 = 가중치).

        Args:
            sample_trajectories: (K, N+1, 3)
            sample_weights: (K,)
            sample_costs: (K,) or None
            timestamp: 타임스탬프

        Returns:
            Marker 리스트
        """
        markers = []
        K = sample_trajectories.shape[0]

        # 가중치 상위 샘플만 표시
        top_indices = np.argsort(sample_weights)[-self.MAX_DISPLAY_SAMPLES:]

        max_weight = np.max(sample_weights)

        for rank, idx in enumerate(top_indices):
            marker = Marker()
            marker.header.stamp = timestamp
            marker.header.frame_id = self.frame_id
            marker.ns = self.NS_SAMPLE_TRAJ
            marker.id = rank
            marker.type = Marker.LINE_STRIP
            marker.action = Marker.ADD

            marker.scale.x = 0.02

            # 투명도 = 가중치 비례
            alpha = float(np.clip(sample_weights[idx] / max_weight, 0.05, 0.8))
            marker.color = ColorRGBA(r=0.5, g=0.5, b=1.0, a=alpha)

            for state in sample_trajectories[idx]:
                point = Point()
                point.x = float(state[0])
                point.y = float(state[1])
                point.z = 0.02
                marker.points.append(point)

            markers.append(marker)

        return markers

    def _create_weighted_trajectory_marker(
        self,
        weighted_trajectory: np.ndarray,
        timestamp: Time,
    ) -> Marker:
        """가중 평균 궤적 (시안, 굵은 라인).

        Args:
            weighted_trajectory: (N+1, 3)
            timestamp: 타임스탬프

        Returns:
            LINE_STRIP Marker
        """
        marker = Marker()
        marker.header.stamp = timestamp
        marker.header.frame_id = self.frame_id
        marker.ns = self.NS_WEIGHTED_TRAJ
        marker.id = 0
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD

        marker.scale.x = 0.08  # 굵은 라인
        marker.color = ColorRGBA(r=0.0, g=1.0, b=1.0, a=0.9)  # 시안

        for state in weighted_trajectory:
            point = Point()
            point.x = float(state[0])
            point.y = float(state[1])
            point.z = 0.05
            marker.points.append(point)

        return marker

    def _create_best_trajectory_marker(
        self,
        best_trajectory: np.ndarray,
        timestamp: Time,
    ) -> Marker:
        """최적 샘플 궤적 (마젠타).

        Args:
            best_trajectory: (N+1, 3)
            timestamp: 타임스탬프

        Returns:
            LINE_STRIP Marker
        """
        marker = Marker()
        marker.header.stamp = timestamp
        marker.header.frame_id = self.frame_id
        marker.ns = self.NS_BEST_TRAJ
        marker.id = 0
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD

        marker.scale.x = 0.05
        marker.color = ColorRGBA(r=1.0, g=0.0, b=1.0, a=0.9)  # 마젠타

        for state in best_trajectory:
            point = Point()
            point.x = float(state[0])
            point.y = float(state[1])
            point.z = 0.04
            marker.points.append(point)

        return marker

    def _create_cost_heatmap_markers(
        self,
        sample_trajectories: np.ndarray,
        sample_costs: np.ndarray,
        timestamp: Time,
    ) -> list:
        """비용 히트맵 (초록 → 빨강 그라데이션).

        각 샘플의 끝점을 비용에 따른 색상으로 표시.

        Args:
            sample_trajectories: (K, N+1, 3)
            sample_costs: (K,)
            timestamp: 타임스탬프

        Returns:
            Marker 리스트
        """
        markers = []
        K = sample_trajectories.shape[0]

        # 비용 정규화
        min_cost = np.min(sample_costs)
        max_cost = np.max(sample_costs)
        cost_range = max_cost - min_cost
        if cost_range < 1e-6:
            cost_range = 1.0

        # 상위/하위 일부만 표시
        indices = np.linspace(0, K - 1, min(K, self.MAX_DISPLAY_SAMPLES)).astype(int)

        for rank, idx in enumerate(indices):
            marker = Marker()
            marker.header.stamp = timestamp
            marker.header.frame_id = self.frame_id
            marker.ns = self.NS_COST_HEATMAP
            marker.id = rank
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD

            endpoint = sample_trajectories[idx, -1, :]
            marker.pose.position.x = float(endpoint[0])
            marker.pose.position.y = float(endpoint[1])
            marker.pose.position.z = 0.15
            marker.pose.orientation.w = 1.0

            marker.scale.x = 0.08
            marker.scale.y = 0.08
            marker.scale.z = 0.08

            # 비용 비례 색상: 초록(낮음) → 빨강(높음)
            ratio = (sample_costs[idx] - min_cost) / cost_range
            ratio = float(np.clip(ratio, 0.0, 1.0))
            marker.color = ColorRGBA(
                r=ratio,
                g=1.0 - ratio,
                b=0.0,
                a=0.6,
            )

            markers.append(marker)

        return markers

    def _create_info_text_marker(
        self,
        current_state: np.ndarray,
        mppi_info: Dict,
        timestamp: Time,
    ) -> Marker:
        """온도/ESS 정보 텍스트.

        Args:
            current_state: [x, y, theta]
            mppi_info: MPPI info dict
            timestamp: 타임스탬프

        Returns:
            TEXT_VIEW_FACING Marker
        """
        marker = Marker()
        marker.header.stamp = timestamp
        marker.header.frame_id = self.frame_id
        marker.ns = self.NS_INFO_TEXT
        marker.id = 0
        marker.type = Marker.TEXT_VIEW_FACING
        marker.action = Marker.ADD

        marker.pose.position.x = float(current_state[0])
        marker.pose.position.y = float(current_state[1])
        marker.pose.position.z = 1.5

        marker.scale.z = 0.25
        marker.color = ColorRGBA(r=1.0, g=1.0, b=1.0, a=0.9)

        temp = mppi_info.get("temperature", 0.0)
        ess = mppi_info.get("ess", 0.0)
        min_cost = mppi_info.get("cost", 0.0)
        solve_ms = mppi_info.get("solve_time", 0.0) * 1000

        marker.text = (
            f"MPPI\n"
            f"T={temp:.2f} ESS={ess:.0f}\n"
            f"cost={min_cost:.2f}\n"
            f"{solve_ms:.1f}ms"
        )

        return marker

    def create_delete_all_markers(self) -> MarkerArray:
        """모든 MPPI 마커를 삭제하는 MarkerArray."""
        marker_array = MarkerArray()

        for ns in [
            self.NS_SAMPLE_TRAJ,
            self.NS_WEIGHTED_TRAJ,
            self.NS_BEST_TRAJ,
            self.NS_COST_HEATMAP,
            self.NS_INFO_TEXT,
        ]:
            marker = Marker()
            marker.header.frame_id = self.frame_id
            marker.ns = ns
            marker.id = 0
            marker.action = Marker.DELETEALL
            marker_array.markers.append(marker)

        return marker_array
