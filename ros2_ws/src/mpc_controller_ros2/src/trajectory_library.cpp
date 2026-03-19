// =============================================================================
// Trajectory Library — 7종 제어 시퀀스 프리미티브 생성
//
// 프리미티브:
//   1. STRAIGHT       — (v_max, 0, ...)
//   2. TURN_LEFT      — (v_max*0.5, +omega_max, ...)
//   3. TURN_RIGHT     — (v_max*0.5, -omega_max, ...)
//   4. S_CURVE_LEFT   — (v_max*0.7, omega_max·sin(2πt/T), ...)
//   5. S_CURVE_RIGHT  — (v_max*0.7, -omega_max·sin(2πt/T), ...)
//   6. STOP           — (0, 0, ...)
//   7. REVERSE        — (v_min, 0, ...) [v_min < 0일 때만]
//   *. PREVIOUS_SOLUTION — control_sequence_ 복제 (동적)
// =============================================================================

#include "mpc_controller_ros2/trajectory_library.hpp"
#include <cmath>

namespace mpc_controller_ros2
{

void TrajectoryLibrary::generate(
  int N, int nu, double dt,
  double v_max, double v_min, double omega_max)
{
  primitives_.clear();
  prev_solution_idx_ = -1;

  // 1. STRAIGHT
  {
    Eigen::MatrixXd seq = Eigen::MatrixXd::Zero(N, nu);
    for (int t = 0; t < N; ++t) {
      seq(t, 0) = v_max;
    }
    primitives_.push_back({"STRAIGHT", seq});
  }

  // 2. TURN_LEFT
  {
    Eigen::MatrixXd seq = Eigen::MatrixXd::Zero(N, nu);
    for (int t = 0; t < N; ++t) {
      seq(t, 0) = v_max * 0.5;
      if (nu >= 2) {
        seq(t, 1) = omega_max;
      }
    }
    primitives_.push_back({"TURN_LEFT", seq});
  }

  // 3. TURN_RIGHT
  {
    Eigen::MatrixXd seq = Eigen::MatrixXd::Zero(N, nu);
    for (int t = 0; t < N; ++t) {
      seq(t, 0) = v_max * 0.5;
      if (nu >= 2) {
        seq(t, 1) = -omega_max;
      }
    }
    primitives_.push_back({"TURN_RIGHT", seq});
  }

  // 4. S_CURVE_LEFT
  {
    double T = N * dt;
    Eigen::MatrixXd seq = Eigen::MatrixXd::Zero(N, nu);
    for (int t = 0; t < N; ++t) {
      seq(t, 0) = v_max * 0.7;
      if (nu >= 2) {
        seq(t, 1) = omega_max * std::sin(2.0 * M_PI * t * dt / T);
      }
    }
    primitives_.push_back({"S_CURVE_LEFT", seq});
  }

  // 5. S_CURVE_RIGHT
  {
    double T = N * dt;
    Eigen::MatrixXd seq = Eigen::MatrixXd::Zero(N, nu);
    for (int t = 0; t < N; ++t) {
      seq(t, 0) = v_max * 0.7;
      if (nu >= 2) {
        seq(t, 1) = -omega_max * std::sin(2.0 * M_PI * t * dt / T);
      }
    }
    primitives_.push_back({"S_CURVE_RIGHT", seq});
  }

  // 6. STOP
  {
    primitives_.push_back({"STOP", Eigen::MatrixXd::Zero(N, nu)});
  }

  // 7. REVERSE (v_min < 0일 때만)
  if (v_min < 0.0) {
    Eigen::MatrixXd seq = Eigen::MatrixXd::Zero(N, nu);
    for (int t = 0; t < N; ++t) {
      seq(t, 0) = v_min;
    }
    primitives_.push_back({"REVERSE", seq});
  }

  // 8. PREVIOUS_SOLUTION (초기: zero)
  prev_solution_idx_ = static_cast<int>(primitives_.size());
  primitives_.push_back({"PREVIOUS_SOLUTION", Eigen::MatrixXd::Zero(N, nu)});
}

void TrajectoryLibrary::updatePreviousSolution(const Eigen::MatrixXd& control_sequence)
{
  if (prev_solution_idx_ >= 0 &&
      prev_solution_idx_ < static_cast<int>(primitives_.size()))
  {
    primitives_[prev_solution_idx_].control_sequence = control_sequence;
  }
}

}  // namespace mpc_controller_ros2
