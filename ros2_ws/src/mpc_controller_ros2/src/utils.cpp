#include "mpc_controller_ros2/utils.hpp"
#include <algorithm>
#include <limits>

namespace mpc_controller_ros2
{

Eigen::VectorXd normalizeAngleBatch(const Eigen::VectorXd& angles)
{
  Eigen::VectorXd normalized(angles.size());
  for (int i = 0; i < angles.size(); ++i) {
    normalized(i) = normalizeAngle(angles(i));
  }
  return normalized;
}

Eigen::VectorXd softmaxWeights(const Eigen::VectorXd& costs, double lambda)
{
  // Temperature가 0에 가까우면 최소 비용만 선택 (greedy)
  if (lambda < 1e-9) {
    Eigen::VectorXd weights = Eigen::VectorXd::Zero(costs.size());
    int min_idx;
    costs.minCoeff(&min_idx);
    weights(min_idx) = 1.0;
    return weights;
  }

  // Shifted costs for numerical stability
  // shifted = -costs / lambda
  Eigen::VectorXd shifted = -costs / lambda;

  // Subtract maximum for numerical stability (prevents overflow)
  shifted.array() -= shifted.maxCoeff();

  // Compute exp(shifted)
  Eigen::VectorXd exp_vals = shifted.array().exp();

  // Normalize
  double sum = exp_vals.sum();
  if (sum < 1e-12) {
    // Fallback: uniform weights
    return Eigen::VectorXd::Constant(costs.size(), 1.0 / costs.size());
  }

  return exp_vals / sum;
}

double logSumExp(const Eigen::VectorXd& values)
{
  double max_val = values.maxCoeff();
  return max_val + std::log((values.array() - max_val).exp().sum());
}

double quaternionToYaw(const geometry_msgs::msg::Quaternion& quat)
{
  // Convert quaternion to yaw using atan2
  // yaw = atan2(2*(w*z + x*y), 1 - 2*(y^2 + z^2))
  double siny_cosp = 2.0 * (quat.w * quat.z + quat.x * quat.y);
  double cosy_cosp = 1.0 - 2.0 * (quat.y * quat.y + quat.z * quat.z);
  return std::atan2(siny_cosp, cosy_cosp);
}

double computeESS(const Eigen::VectorXd& weights)
{
  // ESS = 1 / Σ(weights[k]^2)
  double sum_squared = weights.array().square().sum();
  if (sum_squared < 1e-12) {
    return 1.0;  // Degenerate case
  }
  return 1.0 / sum_squared;
}

Eigen::VectorXd euclideanDistance2D(
  const Eigen::MatrixXd& points1,
  const Eigen::MatrixXd& points2
)
{
  // points1, points2: N x 2
  // return: N-dimensional vector of distances
  Eigen::MatrixXd diff = points1 - points2;  // N x 2
  return diff.rowwise().norm();  // N
}

Eigen::VectorXd rowwiseMin(const Eigen::MatrixXd& matrix)
{
  // Return minimum value in each row
  Eigen::VectorXd min_vals(matrix.rows());
  for (int i = 0; i < matrix.rows(); ++i) {
    min_vals(i) = matrix.row(i).minCoeff();
  }
  return min_vals;
}

Eigen::VectorXd colwiseMin(const Eigen::MatrixXd& matrix)
{
  // Return minimum value in each column
  Eigen::VectorXd min_vals(matrix.cols());
  for (int i = 0; i < matrix.cols(); ++i) {
    min_vals(i) = matrix.col(i).minCoeff();
  }
  return min_vals;
}

Eigen::VectorXd qExponential(const Eigen::VectorXd& x, double q)
{
  // q → 1 극한: 표준 exp
  if (std::abs(q - 1.0) < 1e-8) {
    return x.array().exp();
  }

  // exp_q(x) = [1 + (1-q)*x]_+^{1/(1-q)}
  double one_minus_q = 1.0 - q;
  double exponent = 1.0 / one_minus_q;

  Eigen::VectorXd result(x.size());
  for (int i = 0; i < x.size(); ++i) {
    double base = 1.0 + one_minus_q * x(i);
    if (base <= 0.0) {
      result(i) = 0.0;  // [...]_+ = max(0, ...)
    } else {
      result(i) = std::pow(base, exponent);
    }
  }
  return result;
}

}  // namespace mpc_controller_ros2
