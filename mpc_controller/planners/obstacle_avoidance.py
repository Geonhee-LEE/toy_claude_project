"""
Obstacle Avoidance Module.

MPC 기반 장애물 회피 알고리즘을 제공합니다.
"""

from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass
class Obstacle:
    """장애물 데이터 클래스."""

    x: float
    y: float
    radius: float
    velocity_x: float = 0.0
    velocity_y: float = 0.0
    obstacle_type: str = "static"  # static, dynamic

    def predict_position(self, dt: float) -> tuple[float, float]:
        """
        dt 시간 후의 예상 위치를 반환합니다.

        Args:
            dt: Time step

        Returns:
            (predicted_x, predicted_y)
        """
        return (
            self.x + self.velocity_x * dt,
            self.y + self.velocity_y * dt,
        )


class ObstacleAvoidance:
    """
    장애물 회피 클래스.

    MPC 제약조건 생성 및 충돌 검사를 담당합니다.
    """

    def __init__(
        self,
        safety_margin: float = 0.3,
        detection_range: float = 5.0,
        prediction_horizon: float = 2.0,
    ):
        """
        Initialize obstacle avoidance.

        Args:
            safety_margin: Minimum safe distance from obstacles
            detection_range: Maximum detection range
            prediction_horizon: Time horizon for dynamic obstacle prediction
        """
        self.safety_margin = safety_margin
        self.detection_range = detection_range
        self.prediction_horizon = prediction_horizon
        self.obstacles: list[Obstacle] = []

    def update_obstacles(self, obstacles: list[Obstacle]) -> None:
        """
        장애물 목록을 업데이트합니다.

        Args:
            obstacles: List of current obstacles
        """
        self.obstacles = obstacles

    def add_obstacle(self, obstacle: Obstacle) -> None:
        """
        장애물을 추가합니다.

        Args:
            obstacle: Obstacle to add
        """
        self.obstacles.append(obstacle)

    def clear_obstacles(self) -> None:
        """모든 장애물을 제거합니다."""
        self.obstacles.clear()

    def get_nearby_obstacles(
        self,
        x: float,
        y: float,
    ) -> list[Obstacle]:
        """
        현재 위치 근처의 장애물을 반환합니다.

        Args:
            x, y: Current position

        Returns:
            List of nearby obstacles
        """
        nearby = []
        for obs in self.obstacles:
            dist = np.sqrt((obs.x - x) ** 2 + (obs.y - y) ** 2)
            if dist < self.detection_range:
                nearby.append(obs)
        return nearby

    def check_collision(
        self,
        x: float,
        y: float,
        robot_radius: float = 0.3,
    ) -> tuple[bool, Optional[Obstacle]]:
        """
        충돌 여부를 확인합니다.

        Args:
            x, y: Position to check
            robot_radius: Robot collision radius

        Returns:
            (is_collision, colliding_obstacle)
        """
        for obs in self.obstacles:
            dist = np.sqrt((obs.x - x) ** 2 + (obs.y - y) ** 2)
            min_dist = obs.radius + robot_radius + self.safety_margin
            if dist < min_dist:
                return True, obs
        return False, None

    def check_path_collision(
        self,
        path_x: np.ndarray,
        path_y: np.ndarray,
        robot_radius: float = 0.3,
    ) -> tuple[bool, int]:
        """
        경로 전체에 대한 충돌 여부를 확인합니다.

        Args:
            path_x, path_y: Path points
            robot_radius: Robot collision radius

        Returns:
            (has_collision, first_collision_index)
        """
        for i in range(len(path_x)):
            is_collision, _ = self.check_collision(
                path_x[i], path_y[i], robot_radius
            )
            if is_collision:
                return True, i
        return False, -1

    def compute_repulsive_force(
        self,
        x: float,
        y: float,
        influence_distance: float = 2.0,
    ) -> tuple[float, float]:
        """
        Potential Field 기반 반발력을 계산합니다.

        Args:
            x, y: Current position
            influence_distance: Distance of influence

        Returns:
            (force_x, force_y)
        """
        force_x, force_y = 0.0, 0.0

        for obs in self.obstacles:
            dx = x - obs.x
            dy = y - obs.y
            dist = np.sqrt(dx ** 2 + dy ** 2)

            if dist < influence_distance and dist > 0.01:
                # 반발력 크기 (거리에 반비례)
                magnitude = (1.0 / dist - 1.0 / influence_distance) / (dist ** 2)

                # 단위 벡터
                unit_x = dx / dist
                unit_y = dy / dist

                force_x += magnitude * unit_x
                force_y += magnitude * unit_y

        return force_x, force_y

    def get_constraint_distances(
        self,
        x: float,
        y: float,
        theta: float,
        num_rays: int = 8,
        max_range: float = 5.0,
    ) -> np.ndarray:
        """
        여러 방향으로 장애물까지의 거리를 계산합니다.

        레이캐스팅 방식으로 장애물 감지를 수행합니다.

        Args:
            x, y: Current position
            theta: Current heading
            num_rays: Number of rays to cast
            max_range: Maximum ray range

        Returns:
            Array of distances for each ray
        """
        angles = np.linspace(-np.pi, np.pi, num_rays, endpoint=False) + theta
        distances = np.full(num_rays, max_range)

        for i, angle in enumerate(angles):
            ray_end_x = x + max_range * np.cos(angle)
            ray_end_y = y + max_range * np.sin(angle)

            for obs in self.obstacles:
                # 레이-원 교차 검사
                dist = self._ray_circle_intersection(
                    x, y, ray_end_x, ray_end_y,
                    obs.x, obs.y, obs.radius,
                )
                if dist is not None:
                    distances[i] = min(distances[i], dist)

        return distances

    def _ray_circle_intersection(
        self,
        x1: float, y1: float,
        x2: float, y2: float,
        cx: float, cy: float, r: float,
    ) -> Optional[float]:
        """
        레이와 원의 교차점까지 거리를 계산합니다.

        Args:
            x1, y1: Ray start
            x2, y2: Ray end
            cx, cy: Circle center
            r: Circle radius

        Returns:
            Distance to intersection or None
        """
        dx = x2 - x1
        dy = y2 - y1
        fx = x1 - cx
        fy = y1 - cy

        a = dx ** 2 + dy ** 2
        b = 2 * (fx * dx + fy * dy)
        c = fx ** 2 + fy ** 2 - r ** 2

        discriminant = b ** 2 - 4 * a * c

        if discriminant < 0:
            return None

        discriminant = np.sqrt(discriminant)
        t1 = (-b - discriminant) / (2 * a)
        t2 = (-b + discriminant) / (2 * a)

        if 0 <= t1 <= 1:
            return t1 * np.sqrt(a)
        if 0 <= t2 <= 1:
            return t2 * np.sqrt(a)

        return None

    def predict_dynamic_obstacles(
        self,
        dt: float,
        steps: int,
    ) -> list[list[tuple[float, float]]]:
        """
        동적 장애물의 미래 위치를 예측합니다.

        Args:
            dt: Time step
            steps: Number of prediction steps

        Returns:
            List of predicted positions for each dynamic obstacle
        """
        predictions = []

        for obs in self.obstacles:
            if obs.obstacle_type == "dynamic":
                obs_predictions = []
                for i in range(steps):
                    t = (i + 1) * dt
                    pred_x, pred_y = obs.predict_position(t)
                    obs_predictions.append((pred_x, pred_y))
                predictions.append(obs_predictions)

        return predictions

    def get_mpc_constraints(
        self,
        x: float,
        y: float,
        horizon: int,
        dt: float,
    ) -> list[dict]:
        """
        MPC 최적화를 위한 장애물 회피 제약조건을 생성합니다.

        Args:
            x, y: Current position
            horizon: MPC prediction horizon
            dt: Time step

        Returns:
            List of constraint dictionaries
        """
        constraints = []
        nearby = self.get_nearby_obstacles(x, y)

        for obs in nearby:
            min_dist = obs.radius + self.safety_margin

            if obs.obstacle_type == "static":
                # 정적 장애물: 위치 고정
                constraints.append({
                    "type": "circle",
                    "center": (obs.x, obs.y),
                    "min_distance": min_dist,
                    "horizon": list(range(horizon)),
                })
            else:
                # 동적 장애물: 각 스텝별 위치 예측
                for i in range(horizon):
                    t = (i + 1) * dt
                    pred_x, pred_y = obs.predict_position(t)
                    constraints.append({
                        "type": "circle",
                        "center": (pred_x, pred_y),
                        "min_distance": min_dist,
                        "horizon": [i],
                    })

        return constraints
