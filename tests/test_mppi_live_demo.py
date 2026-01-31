#!/usr/bin/env python3
"""Tests for MPPI live demo functionality."""

import unittest
from unittest.mock import MagicMock, patch
import numpy as np
from examples.mppi_basic_demo import MPPILiveVisualizer, run_mppi_demo
from simulation.visualizer import LiveVisualizer


class TestMPPILiveVisualizer(unittest.TestCase):
    """Test MPPI live visualizer functionality."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a simple trajectory for testing
        self.trajectory = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0]
        ])
        
    def test_initialization(self):
        """Test MPPILiveVisualizer initialization."""
        visualizer = MPPILiveVisualizer(
            trajectory=self.trajectory,
            update_interval=0.1
        )
        
        # Should inherit from LiveVisualizer
        self.assertIsInstance(visualizer, LiveVisualizer)
        self.assertEqual(visualizer.update_interval, 0.1)
        self.assertIsNone(visualizer.mppi_info_text)
    
    @patch('matplotlib.pyplot.subplots')
    def test_setup_plot(self):
        """Test MPPI-specific plot setup."""
        with patch.object(MPPILiveVisualizer, '__init__', lambda x, trajectory, update_interval: None):
            visualizer = MPPILiveVisualizer(None, None)
            
            # Mock the parent setup and ax
            visualizer.ax = MagicMock()
            
            # Mock the parent setup_plot method
            with patch.object(LiveVisualizer, 'setup_plot'):
                visualizer.setup_plot()
                
                # Should call ax.text to create info text
                visualizer.ax.text.assert_called_once()
    
    @patch('matplotlib.pyplot.subplots')
    def test_update_visualization_with_mppi_data(self):
        """Test update with MPPI-specific data."""
        with patch.object(MPPILiveVisualizer, '__init__', lambda x, trajectory, update_interval: None):
            visualizer = MPPILiveVisualizer(None, None)
            
            # Mock attributes and methods
            visualizer.ax = MagicMock()
            visualizer.ax.lines = []
            visualizer.mppi_info_text = MagicMock()
            
            # Mock parent method
            with patch.object(LiveVisualizer, 'update_visualization'):
                
                # Test data
                state = np.array([1.0, 0.5, 0.1])
                control = np.array([0.5, 0.1])
                reference = np.array([1.1, 0.5, 0.0])
                prediction = np.array([[1.0, 0.5, 0.1], [1.2, 0.6, 0.0]])
                
                # MPPI-specific info
                sample_trajectories = [
                    np.array([[1.0, 0.5, 0.1], [1.1, 0.6, 0.0]]),
                    np.array([[1.0, 0.5, 0.1], [1.3, 0.4, 0.2]])
                ]
                weights = np.array([0.6, 0.4])
                info = {
                    'sample_trajectories': sample_trajectories,
                    'weights': weights,
                    'weighted_avg_trajectory': np.array([[1.0, 0.5, 0.1], [1.15, 0.55, 0.05]]),
                    'best_sample_idx': 0,
                    'effective_sample_size': 1.8,
                    'temperature': 0.5,
                    'cost': 10.5
                }
                
                visualizer.update_visualization(
                    state, control, reference, prediction, info, 1.0
                )
                
                # Should update info text
                visualizer.mppi_info_text.set_text.assert_called_once()
                
                # Should create plot lines for trajectories
                self.assertGreater(visualizer.ax.plot.call_count, 0)
    
    @patch('matplotlib.pyplot.subplots')
    def test_update_visualization_without_mppi_data(self):
        """Test update without MPPI-specific data."""
        with patch.object(MPPILiveVisualizer, '__init__', lambda x, trajectory, update_interval: None):
            visualizer = MPPILiveVisualizer(None, None)
            
            # Mock attributes and methods
            visualizer.ax = MagicMock()
            visualizer.ax.lines = []
            visualizer.mppi_info_text = MagicMock()
            
            # Mock parent method
            with patch.object(LiveVisualizer, 'update_visualization'):
                
                # Test data without MPPI info
                state = np.array([1.0, 0.5, 0.1])
                control = np.array([0.5, 0.1])
                reference = np.array([1.1, 0.5, 0.0])
                prediction = np.array([[1.0, 0.5, 0.1], [1.2, 0.6, 0.0]])
                info = {'cost': 5.0}  # Minimal info
                
                visualizer.update_visualization(
                    state, control, reference, prediction, info, 1.0
                )
                
                # Should still update info text (with defaults)
                visualizer.mppi_info_text.set_text.assert_called_once()


class TestMPPIDemo(unittest.TestCase):
    """Test MPPI demo function."""
    
    @patch('examples.mppi_basic_demo.MPPILiveVisualizer')
    @patch('examples.mppi_basic_demo.MPPIController')
    @patch('examples.mppi_basic_demo.Simulator')
    @patch('examples.mppi_basic_demo.TrajectoryInterpolator')
    def test_run_mppi_demo_with_live_visualization(self, mock_interp, mock_sim, mock_mppi, mock_viz):
        """Test demo with live visualization enabled."""
        # Mock simulator behavior
        mock_sim_instance = mock_sim.return_value
        mock_sim_instance.get_measurement.return_value = np.array([1.0, 0.5, 0.1])
        mock_sim_instance.step.return_value = np.array([1.1, 0.6, 0.0])
        mock_sim_instance.compute_tracking_error.return_value = np.array([0.1, 0.1, 0.05])
        
        # Mock trajectory interpolator
        mock_interp_instance = mock_interp.return_value
        mock_interp_instance.get_reference.return_value = np.array([
            [1.1, 0.5, 0.0],
            [1.2, 0.5, 0.0]
        ])
        mock_interp_instance.find_closest_point.return_value = (0, 0.05)  # Close to trajectory
        
        # Mock MPPI controller
        mock_mppi_instance = mock_mppi.return_value
        mock_mppi_instance.compute_control.return_value = (
            np.array([0.5, 0.1]),  # control
            {
                'predicted_trajectory': np.array([[1.0, 0.5, 0.1], [1.1, 0.6, 0.0]]),
                'sample_trajectories': [np.array([[1.0, 0.5, 0.1], [1.1, 0.6, 0.0]])],
                'weights': np.array([1.0]),
                'effective_sample_size': 1.0,
                'temperature': 0.5,
                'cost': 5.0,
                'solve_time': 0.01
            }
        )
        
        # Mock visualizer
        mock_viz_instance = mock_viz.return_value
        
        # Test with a very short max_time to speed up test
        with patch('examples.mppi_basic_demo.SimulationConfig') as mock_config:
            mock_config.return_value.max_time = 0.2  # Very short
            mock_config.return_value.dt = 0.1
            
            # This should not raise any exceptions
            run_mppi_demo(live_visualization=True)
            
            # Verify visualizer was created and used
            mock_viz.assert_called_once()
            mock_viz_instance.start.assert_called_once()
            mock_viz_instance.update.assert_called()
    
    @patch('matplotlib.pyplot.show')
    @patch('examples.mppi_basic_demo.MPPIController')
    @patch('examples.mppi_basic_demo.Simulator')
    @patch('examples.mppi_basic_demo.TrajectoryInterpolator')
    def test_run_mppi_demo_without_live_visualization(self, mock_interp, mock_sim, mock_mppi, mock_show):
        """Test demo without live visualization (static plots)."""
        # Mock simulator behavior
        mock_sim_instance = mock_sim.return_value
        mock_sim_instance.get_measurement.return_value = np.array([1.0, 0.5, 0.1])
        mock_sim_instance.step.return_value = np.array([1.1, 0.6, 0.0])
        mock_sim_instance.compute_tracking_error.return_value = np.array([0.1, 0.1, 0.05])
        
        # Mock trajectory interpolator
        mock_interp_instance = mock_interp.return_value
        mock_interp_instance.get_reference.return_value = np.array([
            [1.1, 0.5, 0.0],
            [1.2, 0.5, 0.0]
        ])
        mock_interp_instance.find_closest_point.return_value = (0, 0.05)  # Close to trajectory
        
        # Mock MPPI controller
        mock_mppi_instance = mock_mppi.return_value
        mock_mppi_instance.compute_control.return_value = (
            np.array([0.5, 0.1]),  # control
            {
                'predicted_trajectory': np.array([[1.0, 0.5, 0.1], [1.1, 0.6, 0.0]]),
                'cost': 5.0,
                'solve_time': 0.01
            }
        )
        
        # Test with a very short max_time to speed up test
        with patch('examples.mppi_basic_demo.SimulationConfig') as mock_config:
            mock_config.return_value.max_time = 0.2  # Very short
            mock_config.return_value.dt = 0.1
            
            # This should not raise any exceptions
            run_mppi_demo(live_visualization=False)
            
            # Verify static plot was shown
            mock_show.assert_called_once()


if __name__ == '__main__':
    unittest.main()
