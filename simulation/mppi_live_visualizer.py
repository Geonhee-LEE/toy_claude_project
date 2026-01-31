"""MPPI 전용 실시간 시각화.

기존 LiveVisualizer 패턴을 따르되, MPPI 특화 요소를 추가:
- 상위 가중치 샘플 궤적 (투명도 = 가중치)
- 가중 평균 궤적 (시안)
- 최적 샘플 궤적 (마젠타)
- MPPI 상태 정보 (ESS, 온도, 비용)
"""

from typing import Optional

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np


class MPPILiveVisualizer:
    """MPPI 컨트롤러용 실시간 시각화."""

    MAX_DISPLAY_SAMPLES = 30

    def __init__(
        self,
        reference_trajectory: np.ndarray,
        title: str = "MPPI Path Tracking (Live)",
        robot_length: float = 0.3,
        robot_width: float = 0.2,
        update_interval: int = 2,
    ):
        self.reference_trajectory = reference_trajectory
        self.robot_length = robot_length
        self.robot_width = robot_width
        self.update_interval = update_interval
        self.step_count = 0

        plt.ion()

        self.fig, self.axes = plt.subplots(1, 2, figsize=(16, 7))

        # ─── 궤적 플롯 ───
        self.ax_traj = self.axes[0]
        self.ax_traj.set_title(title)
        self.ax_traj.set_xlabel("X [m]")
        self.ax_traj.set_ylabel("Y [m]")
        self.ax_traj.grid(True, alpha=0.3)
        self.ax_traj.axis("equal")

        # 참조 궤적
        self.ax_traj.plot(
            reference_trajectory[:, 0],
            reference_trajectory[:, 1],
            "b--", linewidth=1.5, label="Reference", alpha=0.5,
        )

        # 실제 궤적
        (self.trace_line,) = self.ax_traj.plot(
            [], [], "r-", linewidth=2, label="Actual",
        )

        # 샘플 궤적 라인들 (미리 생성)
        self.sample_lines = []
        for _ in range(self.MAX_DISPLAY_SAMPLES):
            (line,) = self.ax_traj.plot(
                [], [], "-", color="steelblue", alpha=0.1, linewidth=0.5,
            )
            self.sample_lines.append(line)

        # 가중 평균 궤적 (시안)
        (self.weighted_line,) = self.ax_traj.plot(
            [], [], "c-", linewidth=3, label="Weighted Mean", alpha=0.9,
        )

        # 최적 샘플 궤적 (마젠타)
        (self.best_line,) = self.ax_traj.plot(
            [], [], "m-", linewidth=2, label="Best Sample", alpha=0.8,
        )

        # 로봇 패치
        self.robot_patch = patches.Rectangle(
            (0, 0), robot_length, robot_width,
            angle=0, fill=True, facecolor="red",
            edgecolor="black", linewidth=2, alpha=0.8,
        )
        self.ax_traj.add_patch(self.robot_patch)

        # 방향 표시
        (self.direction_line,) = self.ax_traj.plot([], [], "k-", linewidth=2)

        # 축 범위
        x_margin = 1.0
        y_margin = 1.0
        self.ax_traj.set_xlim(
            reference_trajectory[:, 0].min() - x_margin,
            reference_trajectory[:, 0].max() + x_margin,
        )
        self.ax_traj.set_ylim(
            reference_trajectory[:, 1].min() - y_margin,
            reference_trajectory[:, 1].max() + y_margin,
        )
        self.ax_traj.legend(loc="upper right", fontsize=9)

        # ─── 정보 패널 ───
        self.ax_info = self.axes[1]
        self.ax_info.axis("off")
        self.ax_info.set_title("MPPI Status")

        self.info_text = self.ax_info.text(
            0.05, 0.95, "", transform=self.ax_info.transAxes,
            fontsize=11, verticalalignment="top", fontfamily="monospace",
        )

        # 데이터 저장
        self.trace_x = []
        self.trace_y = []

        plt.tight_layout()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def update(
        self,
        state: np.ndarray,
        control: np.ndarray,
        reference: np.ndarray,
        prediction: Optional[np.ndarray] = None,
        info: Optional[dict] = None,
        time: float = 0.0,
    ) -> None:
        """시각화 업데이트.

        Args:
            state: 현재 상태 [x, y, theta]
            control: 제어 입력 [v, omega]
            reference: 현재 참조 [x, y, theta]
            prediction: 예측 궤적 (N+1, 3) - MPPI weighted mean
            info: MPPI info dict (sample_trajectories, sample_weights 등)
            time: 시뮬레이션 시간
        """
        self.step_count += 1
        self.trace_x.append(state[0])
        self.trace_y.append(state[1])

        if self.step_count % self.update_interval != 0:
            return

        # 궤적 트레이스
        self.trace_line.set_data(self.trace_x, self.trace_y)

        # 가중 평균 궤적
        if prediction is not None:
            self.weighted_line.set_data(prediction[:, 0], prediction[:, 1])

        # MPPI 샘플 궤적 + 최적 샘플
        if info is not None:
            sample_traj = info.get("sample_trajectories")
            sample_weights = info.get("sample_weights")
            best_traj = info.get("best_trajectory")

            if sample_traj is not None and sample_weights is not None:
                top_idx = np.argsort(sample_weights)[-self.MAX_DISPLAY_SAMPLES:]
                max_w = np.max(sample_weights)

                for rank, idx in enumerate(top_idx):
                    alpha = float(np.clip(
                        sample_weights[idx] / max_w * 0.6, 0.03, 0.6,
                    ))
                    self.sample_lines[rank].set_data(
                        sample_traj[idx, :, 0],
                        sample_traj[idx, :, 1],
                    )
                    self.sample_lines[rank].set_alpha(alpha)

                # 사용하지 않는 라인 숨기기
                for rank in range(len(top_idx), self.MAX_DISPLAY_SAMPLES):
                    self.sample_lines[rank].set_data([], [])

            if best_traj is not None:
                self.best_line.set_data(best_traj[:, 0], best_traj[:, 1])

        # 로봇 위치
        x, y, theta = state
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        corner_x = x - (self.robot_length / 2 * cos_t - self.robot_width / 2 * sin_t)
        corner_y = y - (self.robot_length / 2 * sin_t + self.robot_width / 2 * cos_t)
        self.robot_patch.set_xy((corner_x, corner_y))
        self.robot_patch.angle = np.degrees(theta)

        dir_len = self.robot_length * 0.8
        self.direction_line.set_data(
            [x, x + dir_len * cos_t],
            [y, y + dir_len * sin_t],
        )

        # 정보 텍스트
        pos_error = np.sqrt(
            (state[0] - reference[0]) ** 2 + (state[1] - reference[1]) ** 2
        )
        heading_error = np.arctan2(
            np.sin(state[2] - reference[2]),
            np.cos(state[2] - reference[2]),
        )

        info_str = (
            f"Time: {time:.2f} s\n"
            f"\n"
            f"State:\n"
            f"  x: {state[0]:+.3f} m\n"
            f"  y: {state[1]:+.3f} m\n"
            f"  theta: {np.degrees(state[2]):+.1f} deg\n"
            f"\n"
            f"Control:\n"
            f"  v: {control[0]:+.3f} m/s\n"
            f"  omega: {control[1]:+.3f} rad/s\n"
            f"\n"
            f"Tracking Error:\n"
            f"  position: {pos_error:.4f} m\n"
            f"  heading: {np.degrees(heading_error):+.1f} deg\n"
        )

        if info is not None:
            ess = info.get("ess", 0)
            temp = info.get("temperature", 0)
            cost = info.get("cost", 0)
            mean_cost = info.get("mean_cost", 0)
            solve_ms = info.get("solve_time", 0) * 1000
            K = len(info.get("sample_weights", []))

            info_str += (
                f"\n"
                f"MPPI Info:\n"
                f"  cost (min): {cost:.2f}\n"
                f"  cost (mean): {mean_cost:.2f}\n"
                f"  solve_time: {solve_ms:.1f} ms\n"
                f"  temperature: {temp:.1f}\n"
                f"  ESS: {ess:.0f} / {K}\n"
                f"  ESS ratio: {ess / max(K, 1) * 100:.1f}%\n"
            )

        self.info_text.set_text(info_str)

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def close(self) -> None:
        """시각화 종료."""
        plt.ioff()

    def wait_for_close(self) -> None:
        """사용자가 창을 닫을 때까지 대기."""
        plt.ioff()
        plt.show()
