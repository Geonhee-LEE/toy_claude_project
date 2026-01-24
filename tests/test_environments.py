"""Tests for simulation environments."""

import numpy as np
import pytest

from simulation.environments import (
    CircleObstacle,
    RectangleObstacle,
    WallObstacle,
    EmptyEnvironment,
    ObstacleFieldEnvironment,
    CorridorEnvironment,
    ParkingLotEnvironment,
    MazeEnvironment,
    get_environment,
)


class TestCircleObstacle:
    """Tests for circular obstacles."""

    def test_contains_inside(self):
        obs = CircleObstacle(center=np.array([0.0, 0.0]), radius=1.0)
        assert obs.contains(np.array([0.0, 0.0]))
        assert obs.contains(np.array([0.5, 0.0]))

    def test_contains_outside(self):
        obs = CircleObstacle(center=np.array([0.0, 0.0]), radius=1.0)
        assert not obs.contains(np.array([2.0, 0.0]))
        assert not obs.contains(np.array([1.5, 1.5]))

    def test_distance_inside(self):
        obs = CircleObstacle(center=np.array([0.0, 0.0]), radius=1.0)
        assert obs.distance(np.array([0.0, 0.0])) < 0

    def test_distance_outside(self):
        obs = CircleObstacle(center=np.array([0.0, 0.0]), radius=1.0)
        dist = obs.distance(np.array([2.0, 0.0]))
        assert abs(dist - 1.0) < 1e-6


class TestRectangleObstacle:
    """Tests for rectangular obstacles."""

    def test_contains_inside(self):
        obs = RectangleObstacle(center=np.array([0.0, 0.0]), width=2.0, height=1.0)
        assert obs.contains(np.array([0.0, 0.0]))
        assert obs.contains(np.array([0.5, 0.2]))

    def test_contains_outside(self):
        obs = RectangleObstacle(center=np.array([0.0, 0.0]), width=2.0, height=1.0)
        assert not obs.contains(np.array([2.0, 0.0]))
        assert not obs.contains(np.array([0.0, 1.0]))

    def test_distance_inside(self):
        obs = RectangleObstacle(center=np.array([0.0, 0.0]), width=2.0, height=1.0)
        assert obs.distance(np.array([0.0, 0.0])) < 0

    def test_distance_outside(self):
        obs = RectangleObstacle(center=np.array([0.0, 0.0]), width=2.0, height=1.0)
        dist = obs.distance(np.array([2.0, 0.0]))
        assert dist > 0


class TestWallObstacle:
    """Tests for wall obstacles."""

    def test_distance_on_wall(self):
        obs = WallObstacle(
            start=np.array([0.0, 0.0]),
            end=np.array([2.0, 0.0]),
            thickness=0.1,
        )
        # Point on the wall line
        dist = obs.distance(np.array([1.0, 0.0]))
        assert dist < 0  # Inside the wall thickness

    def test_distance_away_from_wall(self):
        obs = WallObstacle(
            start=np.array([0.0, 0.0]),
            end=np.array([2.0, 0.0]),
            thickness=0.1,
        )
        # Point away from wall
        dist = obs.distance(np.array([1.0, 1.0]))
        assert dist > 0


class TestEnvironments:
    """Tests for environment classes."""

    def test_empty_environment(self):
        env = EmptyEnvironment()
        assert len(env.obstacles) == 0
        assert env.name() == "Empty"

    def test_empty_environment_custom_bounds(self):
        env = EmptyEnvironment(bounds=(-5, 5, -5, 5))
        assert env.bounds == (-5, 5, -5, 5)

    def test_obstacle_field_environment(self):
        env = ObstacleFieldEnvironment(num_obstacles=5, seed=42)
        assert len(env.obstacles) == 5
        assert "5 obstacles" in env.name()

    def test_obstacle_field_reproducible(self):
        env1 = ObstacleFieldEnvironment(num_obstacles=5, seed=42)
        env2 = ObstacleFieldEnvironment(num_obstacles=5, seed=42)
        # Same seed should produce same obstacles
        for o1, o2 in zip(env1.obstacles, env2.obstacles):
            np.testing.assert_array_almost_equal(o1.center, o2.center)
            assert abs(o1.radius - o2.radius) < 1e-6

    def test_corridor_environment(self):
        env = CorridorEnvironment(length=10.0, width=1.5, num_turns=2)
        assert len(env.obstacles) > 0
        assert env.name() == "Corridor"

    def test_parking_lot_environment(self):
        env = ParkingLotEnvironment(num_rows=2, spots_per_row=5)
        assert env.name() == "ParkingLot"

    def test_maze_environment(self):
        env = MazeEnvironment(size=8.0)
        assert len(env.obstacles) > 0
        assert env.name() == "Maze"


class TestEnvironmentMethods:
    """Tests for environment base class methods."""

    def test_is_collision(self):
        env = EmptyEnvironment()
        env.add_obstacle(CircleObstacle(center=np.array([0.0, 0.0]), radius=1.0))

        # Robot at center should collide
        assert env.is_collision(np.array([0.0, 0.0, 0.0]), robot_radius=0.2)
        # Robot far away should not collide
        assert not env.is_collision(np.array([5.0, 5.0, 0.0]), robot_radius=0.2)

    def test_is_in_bounds(self):
        env = EmptyEnvironment(bounds=(-5, 5, -5, 5))
        assert env.is_in_bounds(np.array([0.0, 0.0, 0.0]))
        assert env.is_in_bounds(np.array([4.9, 4.9, 0.0]))
        assert not env.is_in_bounds(np.array([6.0, 0.0, 0.0]))

    def test_min_obstacle_distance(self):
        env = EmptyEnvironment()
        env.add_obstacle(CircleObstacle(center=np.array([0.0, 0.0]), radius=1.0))
        env.add_obstacle(CircleObstacle(center=np.array([5.0, 0.0]), radius=1.0))

        state = np.array([2.0, 0.0, 0.0])
        min_dist = env.min_obstacle_distance(state)
        # Distance to first obstacle should be 1.0 (2.0 - 1.0)
        assert abs(min_dist - 1.0) < 1e-6


class TestGetEnvironment:
    """Tests for get_environment factory function."""

    def test_get_empty(self):
        env = get_environment("empty")
        assert isinstance(env, EmptyEnvironment)

    def test_get_obstacles(self):
        env = get_environment("obstacles", num_obstacles=3)
        assert isinstance(env, ObstacleFieldEnvironment)
        assert len(env.obstacles) == 3

    def test_get_corridor(self):
        env = get_environment("corridor")
        assert isinstance(env, CorridorEnvironment)

    def test_get_parking(self):
        env = get_environment("parking")
        assert isinstance(env, ParkingLotEnvironment)

    def test_get_maze(self):
        env = get_environment("maze")
        assert isinstance(env, MazeEnvironment)

    def test_get_case_insensitive(self):
        env1 = get_environment("EMPTY")
        env2 = get_environment("Empty")
        assert isinstance(env1, EmptyEnvironment)
        assert isinstance(env2, EmptyEnvironment)

    def test_get_unknown_raises(self):
        with pytest.raises(ValueError):
            get_environment("unknown_environment")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
