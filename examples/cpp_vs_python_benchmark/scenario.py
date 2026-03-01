"""벤치마크 시나리오 정의 — 궤적, 장애물, 초기 조건."""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from mpc_controller.utils.trajectory import LookaheadInterpolator  # noqa: F401 — re-export


@dataclass
class BenchmarkScenario:
    """벤치마크에 사용할 시나리오 정의."""
    name: str
    trajectory: np.ndarray          # (M, nx) 참조 궤적
    initial_state: np.ndarray       # (nx,)
    obstacles: Optional[np.ndarray] = None  # (num_obs, 3) [x, y, radius]
    sim_time: float = 10.0
    dt: float = 0.05


def circle_scenario(
    nx: int = 3,
    radius: float = 3.0,
    num_points: int = 200,
    with_obstacles: bool = True,
) -> BenchmarkScenario:
    """원형 궤적 시나리오."""
    angles = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
    traj = np.zeros((num_points, nx))
    traj[:, 0] = radius * np.cos(angles)       # x
    traj[:, 1] = radius * np.sin(angles)       # y
    # theta = 접선 방향
    traj[:, 2] = np.unwrap(angles + np.pi / 2)

    # nx > 3인 경우 나머지 열은 이미 0으로 초기화됨 (예: NonCoaxial delta=0)

    initial_state = np.zeros(nx)
    initial_state[0] = radius  # 시작점: (radius, 0)

    obstacles = None
    if with_obstacles:
        obstacles = np.array([
            [0.0, 0.0, 0.3],      # 원 중심
            [2.0, 2.0, 0.3],
            [-2.0, -1.5, 0.3],
        ])

    return BenchmarkScenario(
        name="circle",
        trajectory=traj,
        initial_state=initial_state,
        obstacles=obstacles,
        sim_time=10.0,
    )


def figure8_scenario(
    nx: int = 3,
    scale: float = 3.0,
    num_points: int = 300,
    with_obstacles: bool = True,
) -> BenchmarkScenario:
    """8자 궤적 시나리오."""
    t = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
    traj = np.zeros((num_points, nx))
    traj[:, 0] = scale * np.sin(t)
    traj[:, 1] = scale * np.sin(t) * np.cos(t)

    # theta = 접선 방향
    dx = np.gradient(traj[:, 0])
    dy = np.gradient(traj[:, 1])
    traj[:, 2] = np.unwrap(np.arctan2(dy, dx))

    # nx > 3인 경우 나머지 열은 이미 0으로 초기화됨

    initial_state = np.zeros(nx)

    obstacles = None
    if with_obstacles:
        obstacles = np.array([
            [1.5, 1.0, 0.3],
            [-1.5, -1.0, 0.3],
        ])

    return BenchmarkScenario(
        name="figure8",
        trajectory=traj,
        initial_state=initial_state,
        obstacles=obstacles,
        sim_time=15.0,
    )


def tight_turn_scenario(
    nx: int = 4,
    radius: float = 1.5,
    num_points: int = 200,
    with_obstacles: bool = False,
) -> BenchmarkScenario:
    """반경 1.5m 타이트 턴 — 60° 제한에서 추적 오차 증가 예상."""
    angles = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
    traj = np.zeros((num_points, nx))
    traj[:, 0] = radius * np.cos(angles)
    traj[:, 1] = radius * np.sin(angles)
    traj[:, 2] = np.unwrap(angles + np.pi / 2)

    initial_state = np.zeros(nx)
    initial_state[0] = radius

    obstacles = None
    if with_obstacles:
        obstacles = np.array([
            [0.0, 0.0, 0.2],
        ])

    return BenchmarkScenario(
        name="tight_turn",
        trajectory=traj,
        initial_state=initial_state,
        obstacles=obstacles,
        sim_time=12.0,
    )


def slalom_scenario(
    nx: int = 4,
    gate_spacing: float = 2.0,
    amplitude: float = 1.5,
    num_gates: int = 6,
    num_points: int = 300,
) -> BenchmarkScenario:
    """연속 S자 슬라럼 — 스티어링 포화 빈도 비교."""
    total_length = gate_spacing * num_gates
    x = np.linspace(0, total_length, num_points)
    y = amplitude * np.sin(2 * np.pi * x / (2 * gate_spacing))

    traj = np.zeros((num_points, nx))
    traj[:, 0] = x
    traj[:, 1] = y

    dx = np.gradient(x)
    dy = np.gradient(y)
    traj[:, 2] = np.unwrap(np.arctan2(dy, dx))

    initial_state = np.zeros(nx)

    return BenchmarkScenario(
        name="slalom",
        trajectory=traj,
        initial_state=initial_state,
        obstacles=None,
        sim_time=15.0,
    )


def get_model_nx(model_type: str) -> int:
    """모델 타입에 따른 state 차원."""
    return {"diff_drive": 3, "swerve": 3, "non_coaxial_swerve": 4}[model_type]
