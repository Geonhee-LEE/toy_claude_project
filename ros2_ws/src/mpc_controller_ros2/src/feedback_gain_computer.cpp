#include "mpc_controller_ros2/feedback_gain_computer.hpp"

namespace mpc_controller_ros2
{

FeedbackGainComputer::FeedbackGainComputer(int nx, int nu, double regularization)
: nx_(nx), nu_(nu), regularization_(regularization)
{
  V_xx_ = Eigen::MatrixXd::Zero(nx, nx);
}

const std::vector<Eigen::MatrixXd>& FeedbackGainComputer::computeGains(
  const Eigen::MatrixXd& nominal_trajectory,
  const Eigen::MatrixXd& control_sequence,
  const MotionModel& model,
  const Eigen::MatrixXd& Q,
  const Eigen::MatrixXd& Qf,
  const Eigen::MatrixXd& R,
  double dt)
{
  int N = control_sequence.rows();

  // Resize K_ if needed
  if (static_cast<int>(K_.size()) != N) {
    K_.resize(N);
    for (int t = 0; t < N; ++t) {
      K_[t] = Eigen::MatrixXd::Zero(nu_, nx_);
    }
  }

  // Terminal: V_xx = Qf
  V_xx_ = Qf;

  // Backward pass
  for (int t = N - 1; t >= 0; --t) {
    Eigen::VectorXd x_t = nominal_trajectory.row(t).transpose();
    Eigen::VectorXd u_t = control_sequence.row(t).transpose();

    // Linearize
    Linearization lin = model.getLinearization(x_t, u_t, dt);
    const Eigen::MatrixXd& A = lin.A;
    const Eigen::MatrixXd& B = lin.B;

    // Q-function matrices
    Eigen::MatrixXd Q_xx = Q + A.transpose() * V_xx_ * A;
    Eigen::MatrixXd Q_ux = B.transpose() * V_xx_ * A;
    Eigen::MatrixXd Q_uu = R + B.transpose() * V_xx_ * B;

    // Regularize
    Q_uu.diagonal().array() += regularization_;

    // Solve K_t = -Q_uu^{-1} Q_ux
    // For nu=2, use direct 2x2 inverse; otherwise LDLT
    if (nu_ == 2) {
      double a = Q_uu(0, 0), b = Q_uu(0, 1);
      double c = Q_uu(1, 0), d = Q_uu(1, 1);
      double det = a * d - b * c;
      Eigen::MatrixXd Q_uu_inv(2, 2);
      Q_uu_inv(0, 0) = d / det;
      Q_uu_inv(0, 1) = -b / det;
      Q_uu_inv(1, 0) = -c / det;
      Q_uu_inv(1, 1) = a / det;
      K_[t] = -Q_uu_inv * Q_ux;
    } else {
      K_[t] = -Q_uu.ldlt().solve(Q_ux);
    }

    // Update V_xx
    V_xx_ = Q_xx + K_[t].transpose() * Q_uu * K_[t]
           + K_[t].transpose() * Q_ux + Q_ux.transpose() * K_[t];
    V_xx_ = 0.5 * (V_xx_ + V_xx_.transpose());  // Symmetrize
  }

  return K_;
}

}  // namespace mpc_controller_ros2
