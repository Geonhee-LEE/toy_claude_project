"""MPPI 실시간 시각화 및 데모 테스트."""

from unittest.mock import patch, MagicMock

import numpy as np
import pytest

from simulation.mppi_live_visualizer import MPPILiveVisualizer


class TestMPPILiveVisualizer:
    """MPPILiveVisualizer 클래스 테스트."""

    @pytest.fixture
    def trajectory(self):
        theta = np.linspace(0, 2 * np.pi, 50)
        return np.column_stack([
            2.0 * np.cos(theta),
            2.0 * np.sin(theta),
            theta + np.pi / 2,
        ])

    @pytest.fixture
    def mock_plt(self):
        """matplotlib를 완전히 모킹."""
        with patch("simulation.mppi_live_visualizer.plt") as mock_plt, \
             patch("simulation.mppi_live_visualizer.patches") as mock_patches:
            mock_fig = MagicMock()
            mock_ax_traj = MagicMock()
            mock_ax_info = MagicMock()

            # plot 호출 시 (line,) 언패킹 가능하도록
            mock_line = MagicMock()
            mock_ax_traj.plot.return_value = (mock_line,)
            mock_ax_info.text.return_value = MagicMock()

            mock_plt.subplots.return_value = (mock_fig, [mock_ax_traj, mock_ax_info])

            mock_patch = MagicMock()
            mock_patches.Rectangle.return_value = mock_patch

            yield {
                "plt": mock_plt,
                "fig": mock_fig,
                "ax_traj": mock_ax_traj,
                "ax_info": mock_ax_info,
                "line": mock_line,
                "patch": mock_patch,
            }

    def test_init(self, trajectory, mock_plt):
        """생성자가 올바른 속성을 설정하는지 확인."""
        viz = MPPILiveVisualizer(
            reference_trajectory=trajectory,
            title="Test",
            update_interval=3,
        )

        assert viz.update_interval == 3
        assert viz.step_count == 0
        assert viz.MAX_DISPLAY_SAMPLES == 30
        assert len(viz.trace_x) == 0
        assert len(viz.trace_y) == 0
        mock_plt["plt"].ion.assert_called_once()

    def test_update_skip(self, trajectory, mock_plt):
        """update_interval에 따라 스킵하는지 확인."""
        viz = MPPILiveVisualizer(
            reference_trajectory=trajectory,
            update_interval=3,
        )
        # __init__에서 draw 1회 호출되므로 리셋
        mock_plt["fig"].canvas.draw.reset_mock()

        state = np.array([1.0, 0.0, 0.0])
        control = np.array([0.5, 0.1])
        reference = np.array([1.0, 0.0, 0.0])

        # step 1: 스킵
        viz.update(state, control, reference)
        assert viz.step_count == 1
        assert len(viz.trace_x) == 1
        mock_plt["fig"].canvas.draw.assert_not_called()

        # step 2: 스킵
        viz.update(state, control, reference)
        assert viz.step_count == 2
        mock_plt["fig"].canvas.draw.assert_not_called()

        # step 3: 렌더링
        viz.update(state, control, reference)
        assert viz.step_count == 3
        mock_plt["fig"].canvas.draw.assert_called_once()

    def test_update_with_mppi_info(self, trajectory, mock_plt):
        """MPPI info가 전달되면 정상 동작하는지 확인."""
        viz = MPPILiveVisualizer(
            reference_trajectory=trajectory,
            update_interval=1,
        )
        mock_plt["fig"].canvas.draw.reset_mock()

        K, N = 5, 10
        state = np.array([1.0, 0.0, 0.0])
        control = np.array([0.5, 0.1])
        reference = np.array([1.0, 0.0, 0.0])
        prediction = np.zeros((N + 1, 3))
        info = {
            "sample_trajectories": np.zeros((K, N + 1, 3)),
            "sample_weights": np.ones(K) / K,
            "best_trajectory": np.zeros((N + 1, 3)),
            "ess": 5.0,
            "temperature": 10.0,
            "cost": 1.5,
            "mean_cost": 2.0,
            "solve_time": 0.01,
        }

        viz.update(state, control, reference, prediction=prediction, info=info)
        mock_plt["fig"].canvas.draw.assert_called_once()

    def test_close(self, trajectory, mock_plt):
        """close 호출 시 plt.ioff() 호출 확인."""
        viz = MPPILiveVisualizer(reference_trajectory=trajectory)
        viz.close()
        mock_plt["plt"].ioff.assert_called()

    def test_trace_accumulation(self, trajectory, mock_plt):
        """상태 추적 데이터가 누적되는지 확인."""
        viz = MPPILiveVisualizer(
            reference_trajectory=trajectory,
            update_interval=1,
        )

        states = [
            np.array([1.0, 0.0, 0.0]),
            np.array([1.1, 0.1, 0.05]),
            np.array([1.2, 0.2, 0.1]),
        ]
        control = np.array([0.5, 0.1])
        reference = np.array([1.0, 0.0, 0.0])

        for s in states:
            viz.update(s, control, reference)

        assert len(viz.trace_x) == 3
        assert len(viz.trace_y) == 3
        assert viz.trace_x[0] == 1.0
        assert viz.trace_y[2] == 0.2


class TestMPPIDemoArgparse:
    """mppi_basic_demo.py의 argparse 동작 테스트."""

    def test_import_run_mppi_demo(self):
        """run_mppi_demo 함수가 import 가능한지 확인."""
        from examples.mppi_basic_demo import run_mppi_demo
        assert callable(run_mppi_demo)

    def test_run_mppi_demo_signature(self):
        """run_mppi_demo의 live 파라미터 확인."""
        import inspect
        from examples.mppi_basic_demo import run_mppi_demo
        sig = inspect.signature(run_mppi_demo)
        assert "live" in sig.parameters
        assert sig.parameters["live"].default is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
