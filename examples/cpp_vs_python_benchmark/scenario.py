"""벤치마크 시나리오 정의 — 궤적, 장애물, 초기 조건."""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


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


def get_model_nx(model_type: str) -> int:
    """모델 타입에 따른 state 차원."""
    return {"diff_drive": 3, "swerve": 3, "non_coaxial_swerve": 4}[model_type]


# ─────────────────────────────────────────────────────────────
# Lookahead 기반 인터폴레이터
# ─────────────────────────────────────────────────────────────

class LookaheadInterpolator:
    """로봇 위치 기반 closest-point + arc-length lookahead 참조 궤적 생성.

    C++ nav2 플러그인의 pathToReferenceTrajectory 로직을 Python으로 이식:
      1. closest point 탐색 (프루닝 윈도우)
      2. arc-length 누적
      3. lookahead 거리 내에서 N등분 선형 보간
      4. 경로 접선 theta + 각도 연속성 보정

    Parameters
    ----------
    trajectory : (M, nx) 전체 참조 궤적
    dt : float
        궤적 점 간 시간 간격 (완주 판정용)
    lookahead_dist : float
        Lookahead 거리 [m]. 0이면 자동 (v_max * N * mpc_dt)
    min_lookahead : float
        최소 lookahead [m] (goal 근처 수렴 보장)
    v_max : float
        최대 속도 [m/s] (자동 lookahead 계산용)
    """

    def __init__(
        self,
        trajectory: np.ndarray,
        dt: float = 0.05,
        lookahead_dist: float = 0.0,
        min_lookahead: float = 0.5,
        v_max: float = 1.0,
    ):
        self.trajectory = trajectory
        self.dt = dt
        self.num_points = len(trajectory)
        self.nx = trajectory.shape[1]
        self.lookahead_dist = lookahead_dist
        self.min_lookahead = min_lookahead
        self.v_max = v_max

        # arc-length 사전 계산
        diffs = np.diff(trajectory[:, :2], axis=0)
        seg_lengths = np.linalg.norm(diffs, axis=1)
        self.arc_lengths = np.zeros(self.num_points)
        self.arc_lengths[1:] = np.cumsum(seg_lengths)
        self.total_arc_length = self.arc_lengths[-1]

        # 프루닝 인덱스 (점진 탐색용)
        self._prune_idx = 0

    def reset(self):
        """프루닝 인덱스 초기화."""
        self._prune_idx = 0

    def find_closest_point(self, position: np.ndarray) -> tuple:
        """현재 위치에서 가장 가까운 경로 점 탐색 (윈도우 기반).

        Returns
        -------
        (index, distance)
        """
        search_end = min(self._prune_idx + 50, self.num_points)
        search_range = slice(self._prune_idx, search_end)
        dists = np.linalg.norm(
            self.trajectory[search_range, :2] - position[:2], axis=1
        )
        local_idx = np.argmin(dists)
        global_idx = self._prune_idx + local_idx
        self._prune_idx = global_idx
        return global_idx, float(dists[local_idx])

    def get_reference(
        self,
        current_position: np.ndarray,
        horizon: int,
        mpc_dt: float,
        current_theta: float = None,
    ) -> np.ndarray:
        """로봇 위치 기반 lookahead 참조 궤적 생성.

        Parameters
        ----------
        current_position : (2,) or (nx,) — x, y [, theta, ...]
        horizon : int — N
        mpc_dt : float
        current_theta : float, optional

        Returns
        -------
        reference : (horizon+1, nx)
        """
        # 1. Closest point
        closest_idx, _ = self.find_closest_point(current_position)

        # 2. Lookahead 거리 결정
        if self.lookahead_dist > 0:
            lookahead = self.lookahead_dist
        else:
            lookahead = self.v_max * horizon * mpc_dt

        # 남은 경로 길이
        remaining_arc = self.total_arc_length - self.arc_lengths[closest_idx]
        effective_lookahead = max(
            self.min_lookahead,
            min(lookahead, remaining_arc),
        )
        step_distance = effective_lookahead / horizon

        # 3. Arc-length stepping으로 참조 궤적 생성
        reference = np.zeros((horizon + 1, self.nx))
        base_arc = self.arc_lengths[closest_idx]
        path_idx = closest_idx

        for t in range(horizon + 1):
            target_arc = base_arc + t * step_distance

            # 목표 arc-length에 해당하는 구간 탐색
            while (path_idx < self.num_points - 1 and
                   self.arc_lengths[path_idx + 1] < target_arc):
                path_idx += 1

            if path_idx >= self.num_points - 1:
                # 경로 끝 도달
                reference[t, :] = self.trajectory[-1].copy()
            else:
                # 구간 내 선형 보간
                seg_len = self.arc_lengths[path_idx + 1] - self.arc_lengths[path_idx]
                if seg_len > 1e-6:
                    alpha = (target_arc - self.arc_lengths[path_idx]) / seg_len
                else:
                    alpha = 0.0

                reference[t, :2] = (
                    (1 - alpha) * self.trajectory[path_idx, :2]
                    + alpha * self.trajectory[path_idx + 1, :2]
                )

                # 경로 접선 theta
                dx = (self.trajectory[path_idx + 1, 0]
                      - self.trajectory[path_idx, 0])
                dy = (self.trajectory[path_idx + 1, 1]
                      - self.trajectory[path_idx, 1])
                tangent_len = np.sqrt(dx * dx + dy * dy)
                if tangent_len > 1e-6:
                    reference[t, 2] = np.arctan2(dy, dx)
                else:
                    reference[t, 2] = (reference[t - 1, 2] if t > 0 else 0.0)

                # nx > 3인 경우 나머지 상태는 보간
                if self.nx > 3:
                    reference[t, 3:] = (
                        (1 - alpha) * self.trajectory[path_idx, 3:]
                        + alpha * self.trajectory[path_idx + 1, 3:]
                    )

        # 4. 각도 연속성 보정
        # theta를 unwrap하여 급격한 점프 방지
        reference[:, 2] = np.unwrap(reference[:, 2])

        if current_theta is not None:
            ref_theta_0 = reference[0, 2]
            diff = np.arctan2(
                np.sin(ref_theta_0 - current_theta),
                np.cos(ref_theta_0 - current_theta),
            )
            target = current_theta + diff
            offset = target - ref_theta_0
            reference[:, 2] += offset

        return reference
