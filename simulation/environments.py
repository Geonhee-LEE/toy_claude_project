"""Simulation environments with obstacles and boundaries."""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional
from abc import ABC, abstractmethod

import numpy as np


@dataclass
class Obstacle:
    """Base class for obstacles."""
    pass


@dataclass
class CircleObstacle(Obstacle):
    """Circular obstacle."""
    center: np.ndarray
    radius: float

    def contains(self, point: np.ndarray) -> bool:
        """Check if point is inside obstacle."""
        return np.linalg.norm(point[:2] - self.center) < self.radius

    def distance(self, point: np.ndarray) -> float:
        """Get signed distance to obstacle (negative inside)."""
        return np.linalg.norm(point[:2] - self.center) - self.radius


@dataclass
class RectangleObstacle(Obstacle):
    """Rectangular obstacle (axis-aligned)."""
    center: np.ndarray
    width: float
    height: float

    def contains(self, point: np.ndarray) -> bool:
        """Check if point is inside obstacle."""
        dx = abs(point[0] - self.center[0])
        dy = abs(point[1] - self.center[1])
        return dx < self.width / 2 and dy < self.height / 2

    def distance(self, point: np.ndarray) -> float:
        """Get signed distance to obstacle (negative inside)."""
        dx = abs(point[0] - self.center[0]) - self.width / 2
        dy = abs(point[1] - self.center[1]) - self.height / 2

        if dx > 0 and dy > 0:
            return np.sqrt(dx**2 + dy**2)
        return max(dx, dy)


@dataclass
class WallObstacle(Obstacle):
    """Line segment wall obstacle."""
    start: np.ndarray
    end: np.ndarray
    thickness: float = 0.1

    def distance(self, point: np.ndarray) -> float:
        """Get distance to wall."""
        # Vector from start to end
        wall_vec = self.end - self.start
        wall_len = np.linalg.norm(wall_vec)
        wall_dir = wall_vec / wall_len

        # Vector from start to point
        to_point = point[:2] - self.start

        # Project onto wall
        proj_len = np.dot(to_point, wall_dir)
        proj_len = np.clip(proj_len, 0, wall_len)

        # Closest point on wall
        closest = self.start + proj_len * wall_dir

        return np.linalg.norm(point[:2] - closest) - self.thickness / 2

    def contains(self, point: np.ndarray) -> bool:
        """Check if point is inside wall."""
        return self.distance(point) < 0


class Environment(ABC):
    """Base class for simulation environments."""

    def __init__(
        self,
        bounds: Tuple[float, float, float, float],  # (x_min, x_max, y_min, y_max)
        obstacles: List[Obstacle] | None = None,
    ):
        self.bounds = bounds
        self.obstacles = obstacles or []

    @abstractmethod
    def name(self) -> str:
        """Return environment name."""
        pass

    def add_obstacle(self, obstacle: Obstacle) -> None:
        """Add an obstacle to the environment."""
        self.obstacles.append(obstacle)

    def is_collision(self, state: np.ndarray, robot_radius: float = 0.2) -> bool:
        """Check if robot at state collides with any obstacle."""
        for obs in self.obstacles:
            if obs.distance(state) < robot_radius:
                return True
        return False

    def is_in_bounds(self, state: np.ndarray) -> bool:
        """Check if state is within bounds."""
        x, y = state[0], state[1]
        return (
            self.bounds[0] <= x <= self.bounds[1] and
            self.bounds[2] <= y <= self.bounds[3]
        )

    def get_obstacle_distances(self, state: np.ndarray) -> List[float]:
        """Get distances to all obstacles."""
        return [obs.distance(state) for obs in self.obstacles]

    def min_obstacle_distance(self, state: np.ndarray) -> float:
        """Get minimum distance to any obstacle."""
        if not self.obstacles:
            return float('inf')
        return min(self.get_obstacle_distances(state))


class EmptyEnvironment(Environment):
    """Empty environment with no obstacles."""

    def __init__(
        self,
        bounds: Tuple[float, float, float, float] = (-10, 10, -10, 10),
    ):
        super().__init__(bounds, [])

    def name(self) -> str:
        return "Empty"


class ObstacleFieldEnvironment(Environment):
    """Environment with randomly placed circular obstacles."""

    def __init__(
        self,
        bounds: Tuple[float, float, float, float] = (-5, 5, -5, 5),
        num_obstacles: int = 10,
        min_radius: float = 0.2,
        max_radius: float = 0.5,
        seed: int | None = None,
    ):
        super().__init__(bounds, [])
        self._generate_obstacles(num_obstacles, min_radius, max_radius, seed)

    def _generate_obstacles(
        self,
        num_obstacles: int,
        min_radius: float,
        max_radius: float,
        seed: int | None,
    ) -> None:
        """Generate random obstacles."""
        rng = np.random.default_rng(seed)

        x_min, x_max, y_min, y_max = self.bounds
        margin = max_radius * 2

        for _ in range(num_obstacles):
            center = np.array([
                rng.uniform(x_min + margin, x_max - margin),
                rng.uniform(y_min + margin, y_max - margin),
            ])
            radius = rng.uniform(min_radius, max_radius)
            self.add_obstacle(CircleObstacle(center, radius))

    def name(self) -> str:
        return f"ObstacleField ({len(self.obstacles)} obstacles)"


class CorridorEnvironment(Environment):
    """Narrow corridor environment."""

    def __init__(
        self,
        length: float = 10.0,
        width: float = 1.5,
        num_turns: int = 2,
    ):
        bounds = (-2, length + 2, -length, length)
        super().__init__(bounds, [])
        self.length = length
        self.width = width
        self._create_corridor(num_turns)

    def _create_corridor(self, num_turns: int) -> None:
        """Create corridor walls."""
        wall_thickness = 0.1
        segment_length = self.length / (num_turns + 1)

        # Create a winding corridor
        y_offset = 0
        x_pos = 0

        for i in range(num_turns + 1):
            # Horizontal segment
            # Top wall
            self.add_obstacle(WallObstacle(
                start=np.array([x_pos, y_offset + self.width / 2]),
                end=np.array([x_pos + segment_length, y_offset + self.width / 2]),
                thickness=wall_thickness,
            ))
            # Bottom wall
            self.add_obstacle(WallObstacle(
                start=np.array([x_pos, y_offset - self.width / 2]),
                end=np.array([x_pos + segment_length, y_offset - self.width / 2]),
                thickness=wall_thickness,
            ))

            x_pos += segment_length

            # Add turn (except for last segment)
            if i < num_turns:
                turn_dir = 1 if i % 2 == 0 else -1
                turn_length = self.width * 2

                # Vertical segment walls
                self.add_obstacle(WallObstacle(
                    start=np.array([x_pos - self.width / 2, y_offset]),
                    end=np.array([x_pos - self.width / 2, y_offset + turn_dir * turn_length]),
                    thickness=wall_thickness,
                ))
                self.add_obstacle(WallObstacle(
                    start=np.array([x_pos + self.width / 2, y_offset]),
                    end=np.array([x_pos + self.width / 2, y_offset + turn_dir * turn_length]),
                    thickness=wall_thickness,
                ))

                y_offset += turn_dir * turn_length

    def name(self) -> str:
        return "Corridor"


class ParkingLotEnvironment(Environment):
    """Parking lot environment with parking spots."""

    def __init__(
        self,
        num_rows: int = 2,
        spots_per_row: int = 5,
        spot_width: float = 1.0,
        spot_depth: float = 2.0,
        aisle_width: float = 3.0,
    ):
        total_width = spots_per_row * spot_width + 2
        total_height = num_rows * (spot_depth + aisle_width) + aisle_width
        bounds = (-1, total_width, -1, total_height)
        super().__init__(bounds, [])

        self._create_parking_lot(num_rows, spots_per_row, spot_width, spot_depth, aisle_width)

    def _create_parking_lot(
        self,
        num_rows: int,
        spots_per_row: int,
        spot_width: float,
        spot_depth: float,
        aisle_width: float,
    ) -> None:
        """Create parking lot obstacles (parked cars)."""
        # Add some parked cars (random spots filled)
        rng = np.random.default_rng(42)

        for row in range(num_rows):
            y_base = row * (spot_depth + aisle_width) + aisle_width

            for spot in range(spots_per_row):
                # 50% chance of a car being parked
                if rng.random() < 0.5:
                    x_center = spot * spot_width + spot_width / 2
                    y_center = y_base + spot_depth / 2

                    self.add_obstacle(RectangleObstacle(
                        center=np.array([x_center, y_center]),
                        width=spot_width * 0.8,
                        height=spot_depth * 0.9,
                    ))

    def name(self) -> str:
        return "ParkingLot"


class MazeEnvironment(Environment):
    """Simple maze environment."""

    def __init__(
        self,
        size: float = 8.0,
        wall_thickness: float = 0.1,
    ):
        bounds = (-1, size + 1, -1, size + 1)
        super().__init__(bounds, [])
        self.size = size
        self._create_maze(wall_thickness)

    def _create_maze(self, wall_thickness: float) -> None:
        """Create maze walls."""
        s = self.size

        # Outer walls
        walls = [
            # Outer boundary
            (np.array([0, 0]), np.array([s, 0])),
            (np.array([s, 0]), np.array([s, s])),
            (np.array([s, s]), np.array([0, s])),
            (np.array([0, s]), np.array([0, 0])),

            # Internal walls
            (np.array([s*0.25, 0]), np.array([s*0.25, s*0.6])),
            (np.array([s*0.5, s*0.4]), np.array([s*0.5, s])),
            (np.array([s*0.75, 0]), np.array([s*0.75, s*0.6])),
            (np.array([s*0.25, s*0.8]), np.array([s*0.75, s*0.8])),
        ]

        for start, end in walls:
            self.add_obstacle(WallObstacle(start, end, wall_thickness))

    def name(self) -> str:
        return "Maze"


def get_environment(name: str, **kwargs) -> Environment:
    """
    Get environment by name.

    Args:
        name: Environment name ('empty', 'obstacles', 'corridor', 'parking', 'maze')
        **kwargs: Additional arguments for the environment

    Returns:
        Environment instance
    """
    environments = {
        'empty': EmptyEnvironment,
        'obstacles': ObstacleFieldEnvironment,
        'corridor': CorridorEnvironment,
        'parking': ParkingLotEnvironment,
        'maze': MazeEnvironment,
    }

    if name.lower() not in environments:
        raise ValueError(f"Unknown environment: {name}. Available: {list(environments.keys())}")

    return environments[name.lower()](**kwargs)
