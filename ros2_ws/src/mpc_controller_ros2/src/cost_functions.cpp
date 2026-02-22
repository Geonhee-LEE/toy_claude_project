#include "mpc_controller_ros2/cost_functions.hpp"
#include "mpc_controller_ros2/utils.hpp"

namespace mpc_controller_ros2
{

// StateTrackingCost
StateTrackingCost::StateTrackingCost(const Eigen::MatrixXd& Q) : Q_(Q) {}

Eigen::VectorXd StateTrackingCost::compute(
  const std::vector<Eigen::MatrixXd>& trajectories,
  const std::vector<Eigen::MatrixXd>& controls,
  const Eigen::MatrixXd& reference
) const
{
  int K = trajectories.size();
  int N = reference.rows() - 1;
  Eigen::VectorXd costs = Eigen::VectorXd::Zero(K);

  for (int k = 0; k < K; ++k) {
    int nx = trajectories[k].cols();
    int ref_cols = reference.cols();
    int min_cols = std::min(nx, ref_cols);
    for (int t = 0; t < N; ++t) {
      Eigen::VectorXd error = Eigen::VectorXd::Zero(nx);
      error.head(min_cols) = trajectories[k].row(t).head(min_cols).transpose()
                           - reference.row(t).head(min_cols).transpose();

      // Normalize angle error (index 2 = theta for all models)
      if (nx >= 3) {
        error(2) = normalizeAngle(error(2));
      }

      costs(k) += error.transpose() * Q_ * error;
    }
  }

  return costs;
}

// TerminalCost
TerminalCost::TerminalCost(const Eigen::MatrixXd& Qf) : Qf_(Qf) {}

Eigen::VectorXd TerminalCost::compute(
  const std::vector<Eigen::MatrixXd>& trajectories,
  const std::vector<Eigen::MatrixXd>& controls,
  const Eigen::MatrixXd& reference
) const
{
  int K = trajectories.size();
  int N = reference.rows() - 1;
  Eigen::VectorXd costs = Eigen::VectorXd::Zero(K);

  for (int k = 0; k < K; ++k) {
    int nx = trajectories[k].cols();
    int ref_cols = reference.cols();
    int min_cols = std::min(nx, ref_cols);
    Eigen::VectorXd error = Eigen::VectorXd::Zero(nx);
    error.head(min_cols) = trajectories[k].row(N).head(min_cols).transpose()
                         - reference.row(N).head(min_cols).transpose();

    if (nx >= 3) {
      error(2) = normalizeAngle(error(2));
    }

    costs(k) = error.transpose() * Qf_ * error;
  }

  return costs;
}

// ControlEffortCost
ControlEffortCost::ControlEffortCost(const Eigen::MatrixXd& R) : R_(R) {}

Eigen::VectorXd ControlEffortCost::compute(
  const std::vector<Eigen::MatrixXd>& trajectories,
  const std::vector<Eigen::MatrixXd>& controls,
  const Eigen::MatrixXd& reference
) const
{
  int K = controls.size();
  int N = controls[0].rows();
  Eigen::VectorXd costs = Eigen::VectorXd::Zero(K);

  for (int k = 0; k < K; ++k) {
    for (int t = 0; t < N; ++t) {
      Eigen::VectorXd u = controls[k].row(t).transpose();
      costs(k) += u.transpose() * R_ * u;
    }
  }

  return costs;
}

// ControlRateCost
ControlRateCost::ControlRateCost(const Eigen::MatrixXd& R_rate) : R_rate_(R_rate) {}

Eigen::VectorXd ControlRateCost::compute(
  const std::vector<Eigen::MatrixXd>& trajectories,
  const std::vector<Eigen::MatrixXd>& controls,
  const Eigen::MatrixXd& reference
) const
{
  int K = controls.size();
  int N = controls[0].rows();
  Eigen::VectorXd costs = Eigen::VectorXd::Zero(K);

  for (int k = 0; k < K; ++k) {
    for (int t = 0; t < N - 1; ++t) {
      Eigen::VectorXd u_curr = controls[k].row(t).transpose();
      Eigen::VectorXd u_next = controls[k].row(t + 1).transpose();
      Eigen::VectorXd du = u_next - u_curr;
      costs(k) += du.transpose() * R_rate_ * du;
    }
  }

  return costs;
}

// PreferForwardCost
PreferForwardCost::PreferForwardCost(double weight, double linear_ratio)
: weight_(weight), linear_ratio_(std::clamp(linear_ratio, 0.0, 1.0)) {}

Eigen::VectorXd PreferForwardCost::compute(
  const std::vector<Eigen::MatrixXd>& trajectories,
  const std::vector<Eigen::MatrixXd>& controls,
  const Eigen::MatrixXd& reference
) const
{
  (void)trajectories;
  (void)reference;

  int K = controls.size();
  int N = controls[0].rows();
  Eigen::VectorXd costs = Eigen::VectorXd::Zero(K);

  for (int k = 0; k < K; ++k) {
    for (int t = 0; t < N; ++t) {
      double v = controls[k](t, 0);
      if (v < 0.0) {
        // 선형+이차 혼합 페널티: weight * (ratio*|v| + (1-ratio)*v²)
        double abs_v = std::abs(v);
        costs(k) += weight_ * (linear_ratio_ * abs_v + (1.0 - linear_ratio_) * v * v);
      }
    }
  }

  return costs;
}

// ObstacleCost
ObstacleCost::ObstacleCost(double weight, double safety_distance)
: weight_(weight), safety_distance_(safety_distance)
{
}

void ObstacleCost::setObstacles(const std::vector<Eigen::Vector3d>& obstacles)
{
  obstacles_ = obstacles;
}

Eigen::VectorXd ObstacleCost::compute(
  const std::vector<Eigen::MatrixXd>& trajectories,
  const std::vector<Eigen::MatrixXd>& controls,
  const Eigen::MatrixXd& reference
) const
{
  int K = trajectories.size();
  Eigen::VectorXd costs = Eigen::VectorXd::Zero(K);

  if (obstacles_.empty()) {
    return costs;
  }

  for (int k = 0; k < K; ++k) {
    const auto& traj = trajectories[k];
    int N = traj.rows() - 1;

    for (int t = 0; t <= N; ++t) {
      Eigen::Vector2d pos = traj.row(t).head<2>().transpose();

      for (const auto& obs : obstacles_) {
        Eigen::Vector2d obs_pos = obs.head<2>();
        double dist = (pos - obs_pos).norm();
        double penetration = safety_distance_ - dist;

        if (penetration > 0) {
          costs(k) += weight_ * penetration * penetration;
        }
      }
    }
  }

  return costs;
}

// CostmapObstacleCost
CostmapObstacleCost::CostmapObstacleCost(double weight, double lethal_cost,
                                           double critical_cost)
: weight_(weight), lethal_cost_(lethal_cost), critical_cost_(critical_cost)
{
}

void CostmapObstacleCost::setCostmap(nav2_costmap_2d::Costmap2D* costmap)
{
  costmap_ = costmap;
}

void CostmapObstacleCost::setMapToOdomTransform(double tx, double ty,
                                                 double cos_th, double sin_th,
                                                 bool use_tf)
{
  tx_ = tx;
  ty_ = ty;
  cos_th_ = cos_th;
  sin_th_ = sin_th;
  use_tf_ = use_tf;
}

Eigen::VectorXd CostmapObstacleCost::compute(
  const std::vector<Eigen::MatrixXd>& trajectories,
  const std::vector<Eigen::MatrixXd>& controls,
  const Eigen::MatrixXd& reference
) const
{
  (void)controls;
  (void)reference;

  int K = trajectories.size();
  Eigen::VectorXd costs = Eigen::VectorXd::Zero(K);

  if (!costmap_) {
    return costs;
  }

  for (int k = 0; k < K; ++k) {
    const auto& traj = trajectories[k];
    int num_points = traj.rows();

    for (int t = 0; t < num_points; ++t) {
      double map_x = traj(t, 0);
      double map_y = traj(t, 1);

      // map→odom 좌표 변환 (costmap은 odom 프레임)
      double query_x, query_y;
      if (use_tf_) {
        query_x = cos_th_ * map_x - sin_th_ * map_y + tx_;
        query_y = sin_th_ * map_x + cos_th_ * map_y + ty_;
      } else {
        query_x = map_x;
        query_y = map_y;
      }

      unsigned int mx, my;
      if (!costmap_->worldToMap(query_x, query_y, mx, my)) {
        // 범위 밖 → lethal 비용
        costs(k) += lethal_cost_;
        continue;
      }

      unsigned char cell_cost = costmap_->getCost(mx, my);

      if (cell_cost >= nav2_costmap_2d::LETHAL_OBSTACLE) {
        costs(k) += lethal_cost_;
      } else if (cell_cost >= nav2_costmap_2d::INSCRIBED_INFLATED_OBSTACLE) {
        costs(k) += critical_cost_;
      } else if (cell_cost > nav2_costmap_2d::FREE_SPACE) {
        // inflation gradient: smooth quadratic cost
        double normalized = static_cast<double>(cell_cost) / 252.0;
        costs(k) += weight_ * normalized * normalized;
      }
    }
  }

  return costs;
}

// CompositeMPPICost
void CompositeMPPICost::addCost(std::unique_ptr<MPPICostFunction> cost)
{
  costs_.push_back(std::move(cost));
}

void CompositeMPPICost::clearCosts()
{
  costs_.clear();
}

Eigen::VectorXd CompositeMPPICost::compute(
  const std::vector<Eigen::MatrixXd>& trajectories,
  const std::vector<Eigen::MatrixXd>& controls,
  const Eigen::MatrixXd& reference
) const
{
  if (costs_.empty()) {
    return Eigen::VectorXd::Zero(trajectories.size());
  }

  Eigen::VectorXd total_costs = costs_[0]->compute(trajectories, controls, reference);

  for (size_t i = 1; i < costs_.size(); ++i) {
    total_costs += costs_[i]->compute(trajectories, controls, reference);
  }

  return total_costs;
}

}  // namespace mpc_controller_ros2
