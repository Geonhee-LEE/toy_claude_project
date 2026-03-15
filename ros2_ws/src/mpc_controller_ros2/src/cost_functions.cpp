#include "mpc_controller_ros2/cost_functions.hpp"
#include "mpc_controller_ros2/ensemble_dynamics_model.hpp"
#include "mpc_controller_ros2/utils.hpp"
#include <omp.h>
#include <utility>

namespace mpc_controller_ros2
{

// ============================================================================
// 대각 행렬 검출 헬퍼
// ============================================================================

static std::pair<bool, Eigen::VectorXd> checkDiagonal(const Eigen::MatrixXd& M)
{
  int n = M.rows();
  if (n != M.cols()) {
    return {false, Eigen::VectorXd()};
  }
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      if (i != j && std::abs(M(i, j)) > 1e-12) {
        return {false, Eigen::VectorXd()};
      }
    }
  }
  return {true, M.diagonal()};
}

// ============================================================================
// StateTrackingCost
// ============================================================================

StateTrackingCost::StateTrackingCost(const Eigen::MatrixXd& Q) : Q_(Q)
{
  auto [diag, vec] = checkDiagonal(Q);
  is_diagonal_ = diag;
  if (diag) { Q_diag_ = vec; }
}

Eigen::VectorXd StateTrackingCost::compute(
  const std::vector<Eigen::MatrixXd>& trajectories,
  const std::vector<Eigen::MatrixXd>& controls,
  const Eigen::MatrixXd& reference
) const
{
  (void)controls;
  int K = trajectories.size();
  int N = reference.rows() - 1;
  Eigen::VectorXd costs = Eigen::VectorXd::Zero(K);

  #pragma omp parallel for schedule(static) if(K > 4096)
  for (int k = 0; k < K; ++k) {
    int nx = trajectories[k].cols();
    int ref_cols = reference.cols();
    int min_cols = std::min(nx, ref_cols);
    double cost_k = 0.0;

    if (is_diagonal_) {
      Eigen::VectorXd error(nx);
      for (int t = 0; t < N; ++t) {
        for (int j = 0; j < min_cols; ++j) {
          error(j) = trajectories[k](t, j) - reference(t, j);
        }
        for (int j = min_cols; j < nx; ++j) {
          error(j) = 0.0;
        }
        if (nx >= 3) {
          error(2) = normalizeAngle(error(2));
        }
        cost_k += error.cwiseAbs2().dot(Q_diag_);
      }
    } else {
      Eigen::VectorXd error(nx);
      for (int t = 0; t < N; ++t) {
        for (int j = 0; j < min_cols; ++j) {
          error(j) = trajectories[k](t, j) - reference(t, j);
        }
        for (int j = min_cols; j < nx; ++j) {
          error(j) = 0.0;
        }
        if (nx >= 3) {
          error(2) = normalizeAngle(error(2));
        }
        cost_k += error.transpose() * Q_ * error;
      }
    }
    costs(k) = cost_k;
  }

  return costs;
}

// ============================================================================
// TerminalCost
// ============================================================================

TerminalCost::TerminalCost(const Eigen::MatrixXd& Qf) : Qf_(Qf)
{
  auto [diag, vec] = checkDiagonal(Qf);
  is_diagonal_ = diag;
  if (diag) { Qf_diag_ = vec; }
}

Eigen::VectorXd TerminalCost::compute(
  const std::vector<Eigen::MatrixXd>& trajectories,
  const std::vector<Eigen::MatrixXd>& controls,
  const Eigen::MatrixXd& reference
) const
{
  (void)controls;
  int K = trajectories.size();
  int N = reference.rows() - 1;
  Eigen::VectorXd costs = Eigen::VectorXd::Zero(K);

  #pragma omp parallel for schedule(static) if(K > 4096)
  for (int k = 0; k < K; ++k) {
    int nx = trajectories[k].cols();
    int ref_cols = reference.cols();
    int min_cols = std::min(nx, ref_cols);

    Eigen::VectorXd error(nx);
    for (int j = 0; j < min_cols; ++j) {
      error(j) = trajectories[k](N, j) - reference(N, j);
    }
    for (int j = min_cols; j < nx; ++j) {
      error(j) = 0.0;
    }
    if (nx >= 3) {
      error(2) = normalizeAngle(error(2));
    }

    if (is_diagonal_) {
      costs(k) = error.cwiseAbs2().dot(Qf_diag_);
    } else {
      costs(k) = error.transpose() * Qf_ * error;
    }
  }

  return costs;
}

// ============================================================================
// ControlEffortCost
// ============================================================================

ControlEffortCost::ControlEffortCost(const Eigen::MatrixXd& R) : R_(R)
{
  auto [diag, vec] = checkDiagonal(R);
  is_diagonal_ = diag;
  if (diag) { R_diag_ = vec; }
}

Eigen::VectorXd ControlEffortCost::compute(
  const std::vector<Eigen::MatrixXd>& trajectories,
  const std::vector<Eigen::MatrixXd>& controls,
  const Eigen::MatrixXd& reference
) const
{
  (void)trajectories;
  (void)reference;
  int K = controls.size();
  int N = controls[0].rows();
  int nu = controls[0].cols();
  Eigen::VectorXd costs = Eigen::VectorXd::Zero(K);

  #pragma omp parallel for schedule(static) if(K > 4096)
  for (int k = 0; k < K; ++k) {
    double cost_k = 0.0;
    if (is_diagonal_) {
      for (int t = 0; t < N; ++t) {
        double dot = 0.0;
        for (int j = 0; j < nu; ++j) {
          double u_j = controls[k](t, j);
          dot += u_j * u_j * R_diag_(j);
        }
        cost_k += dot;
      }
    } else {
      Eigen::VectorXd u(nu);
      for (int t = 0; t < N; ++t) {
        for (int j = 0; j < nu; ++j) {
          u(j) = controls[k](t, j);
        }
        cost_k += u.transpose() * R_ * u;
      }
    }
    costs(k) = cost_k;
  }

  return costs;
}

// ============================================================================
// ControlRateCost
// ============================================================================

ControlRateCost::ControlRateCost(const Eigen::MatrixXd& R_rate) : R_rate_(R_rate)
{
  auto [diag, vec] = checkDiagonal(R_rate);
  is_diagonal_ = diag;
  if (diag) { R_rate_diag_ = vec; }
}

Eigen::VectorXd ControlRateCost::compute(
  const std::vector<Eigen::MatrixXd>& trajectories,
  const std::vector<Eigen::MatrixXd>& controls,
  const Eigen::MatrixXd& reference
) const
{
  (void)trajectories;
  (void)reference;
  int K = controls.size();
  int N = controls[0].rows();
  int nu = controls[0].cols();
  Eigen::VectorXd costs = Eigen::VectorXd::Zero(K);

  #pragma omp parallel for schedule(static) if(K > 4096)
  for (int k = 0; k < K; ++k) {
    double cost_k = 0.0;
    if (is_diagonal_) {
      for (int t = 0; t < N - 1; ++t) {
        double dot = 0.0;
        for (int j = 0; j < nu; ++j) {
          double du = controls[k](t + 1, j) - controls[k](t, j);
          dot += du * du * R_rate_diag_(j);
        }
        cost_k += dot;
      }
    } else {
      Eigen::VectorXd du(nu);
      for (int t = 0; t < N - 1; ++t) {
        for (int j = 0; j < nu; ++j) {
          du(j) = controls[k](t + 1, j) - controls[k](t, j);
        }
        cost_k += du.transpose() * R_rate_ * du;
      }
    }
    costs(k) = cost_k;
  }

  return costs;
}

// ============================================================================
// PreferForwardCost
// ============================================================================

PreferForwardCost::PreferForwardCost(double weight, double linear_ratio,
                                     double velocity_incentive)
: weight_(weight), linear_ratio_(std::clamp(linear_ratio, 0.0, 1.0)),
  velocity_incentive_(velocity_incentive) {}

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

  constexpr double kIncentiveThreshold = 0.1;  // m/s

  #pragma omp parallel for schedule(static) if(K > 4096)
  for (int k = 0; k < K; ++k) {
    double cost_k = 0.0;
    for (int t = 0; t < N; ++t) {
      double v = controls[k](t, 0);
      if (v < 0.0) {
        double abs_v = std::abs(v);
        cost_k += weight_ * (linear_ratio_ * abs_v + (1.0 - linear_ratio_) * v * v);
      }
      if (velocity_incentive_ > 0.0 && v >= 0.0 && v < kIncentiveThreshold) {
        double deficit = kIncentiveThreshold - v;
        cost_k += velocity_incentive_ * deficit * deficit;
      }
    }
    costs(k) = cost_k;
  }

  return costs;
}

// ============================================================================
// ObstacleCost
// ============================================================================

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
  (void)controls;
  (void)reference;

  int K = trajectories.size();
  Eigen::VectorXd costs = Eigen::VectorXd::Zero(K);

  if (obstacles_.empty()) {
    return costs;
  }

  #pragma omp parallel for schedule(static) if(K > 4096)
  for (int k = 0; k < K; ++k) {
    const auto& traj = trajectories[k];
    int N = traj.rows() - 1;
    double cost_k = 0.0;

    for (int t = 0; t <= N; ++t) {
      double px = traj(t, 0);
      double py = traj(t, 1);

      for (const auto& obs : obstacles_) {
        double dx = px - obs(0);
        double dy = py - obs(1);
        double dist = std::sqrt(dx * dx + dy * dy);
        double penetration = safety_distance_ - dist;

        if (penetration > 0) {
          cost_k += weight_ * penetration * penetration;
        }
      }
    }
    costs(k) = cost_k;
  }

  return costs;
}

// ============================================================================
// CostmapObstacleCost
// ============================================================================

CostmapObstacleCost::CostmapObstacleCost(double weight, double lethal_cost,
                                           double critical_cost, int stride)
: weight_(weight), lethal_cost_(lethal_cost), critical_cost_(critical_cost),
  stride_(std::max(1, stride))
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

  // Note: costmap_->worldToMap / getCost are not thread-safe (shared state)
  // so we do NOT parallelize the outer loop here. The stride_ optimization
  // provides speedup instead.
  for (int k = 0; k < K; ++k) {
    const auto& traj = trajectories[k];
    int num_points = traj.rows();

    for (int t = 0; t < num_points; t += stride_) {
      double map_x = traj(t, 0);
      double map_y = traj(t, 1);

      // map -> odom 좌표 변환 (costmap은 odom 프레임)
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

// CostmapObstacleCost::computePerPoint
Eigen::MatrixXd CostmapObstacleCost::computePerPoint(
  const std::vector<Eigen::MatrixXd>& trajectories
) const
{
  int K = trajectories.size();
  if (K == 0) {
    return Eigen::MatrixXd();
  }
  int T = trajectories[0].rows();
  Eigen::MatrixXd per_point = Eigen::MatrixXd::Zero(K, T);

  if (!costmap_) {
    return per_point;
  }

  for (int k = 0; k < K; ++k) {
    const auto& traj = trajectories[k];
    for (int t = 0; t < T; ++t) {
      double map_x = traj(t, 0);
      double map_y = traj(t, 1);

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
        per_point(k, t) = lethal_cost_;
        continue;
      }

      unsigned char cell_cost = costmap_->getCost(mx, my);

      if (cell_cost >= nav2_costmap_2d::LETHAL_OBSTACLE) {
        per_point(k, t) = lethal_cost_;
      } else if (cell_cost >= nav2_costmap_2d::INSCRIBED_INFLATED_OBSTACLE) {
        per_point(k, t) = critical_cost_;
      } else if (cell_cost > nav2_costmap_2d::FREE_SPACE) {
        double normalized = static_cast<double>(cell_cost) / 252.0;
        per_point(k, t) = weight_ * normalized * normalized;
      }
    }
  }

  return per_point;
}

// ============================================================================
// CBFCost
// ============================================================================

CBFCost::CBFCost(BarrierFunctionSet* barrier_set, double weight,
                 double gamma, double dt, double horizon_discount)
: barrier_set_(barrier_set), weight_(weight),
  horizon_discount_(std::clamp(horizon_discount, 0.0, 1.0))
{
  decay_ = 1.0 - gamma * dt;
}

Eigen::VectorXd CBFCost::compute(
  const std::vector<Eigen::MatrixXd>& trajectories,
  const std::vector<Eigen::MatrixXd>& controls,
  const Eigen::MatrixXd& reference
) const
{
  (void)controls;
  (void)reference;

  int K = trajectories.size();
  Eigen::VectorXd costs = Eigen::VectorXd::Zero(K);

  if (!barrier_set_ || barrier_set_->empty()) {
    return costs;
  }

  for (const auto& barrier : barrier_set_->barriers()) {
    #pragma omp parallel for schedule(static) if(K > 4096)
    for (int k = 0; k < K; ++k) {
      const auto& traj = trajectories[k];
      int N = traj.rows() - 1;

      // 배치 평가: h(x_0), h(x_1), ..., h(x_N)
      Eigen::VectorXd h_all = barrier.evaluateBatch(traj);

      double cost_k = 0.0;
      double discount = 1.0;
      for (int t = 0; t < N; ++t) {
        double h_t = h_all(t);
        double h_t1 = h_all(t + 1);

        // DCBF 위반: (1-γdt)·h(x_t) - h(x_{t+1}) > 0
        double violation = decay_ * h_t - h_t1;
        if (violation > 0.0) {
          cost_k += weight_ * discount * violation * violation;
        }
        discount *= horizon_discount_;
      }
      costs(k) += cost_k;
    }
  }

  return costs;
}

// ============================================================================
// BarrierRateCost (BR-MPPI)
// ============================================================================

BarrierRateCost::BarrierRateCost(BarrierFunctionSet* barrier_set, double weight, double dt)
: barrier_set_(barrier_set), weight_(weight), dt_(dt)
{
}

Eigen::VectorXd BarrierRateCost::compute(
  const std::vector<Eigen::MatrixXd>& trajectories,
  const std::vector<Eigen::MatrixXd>& controls,
  const Eigen::MatrixXd& reference
) const
{
  (void)controls;
  (void)reference;

  int K = trajectories.size();
  Eigen::VectorXd costs = Eigen::VectorXd::Zero(K);

  if (!barrier_set_ || barrier_set_->empty()) {
    return costs;
  }

  for (const auto& barrier : barrier_set_->barriers()) {
    #pragma omp parallel for schedule(static) if(K > 4096)
    for (int k = 0; k < K; ++k) {
      const auto& traj = trajectories[k];
      int N = traj.rows() - 1;

      // 배치 평가: h(x_0), ..., h(x_N)
      Eigen::VectorXd h_all = barrier.evaluateBatch(traj);

      double cost_k = 0.0;
      for (int t = 0; t < N; ++t) {
        // dh/dt ≈ (h(x_{t+1}) - h(x_t)) / dt
        double dh_dt = (h_all(t + 1) - h_all(t)) / dt_;

        // 음의 dh/dt (접근 중) → 페널티
        if (dh_dt < 0.0) {
          cost_k += weight_ * dh_dt * dh_dt;
        }
      }
      costs(k) += cost_k;
    }
  }

  return costs;
}

// ============================================================================
// VelocityTrackingCost
// ============================================================================

VelocityTrackingCost::VelocityTrackingCost(double weight, double reference_velocity, double dt)
: weight_(weight), reference_velocity_(reference_velocity), dt_(dt)
{
}

Eigen::VectorXd VelocityTrackingCost::compute(
  const std::vector<Eigen::MatrixXd>& trajectories,
  const std::vector<Eigen::MatrixXd>& controls,
  const Eigen::MatrixXd& reference
) const
{
  (void)controls;

  int K = trajectories.size();
  Eigen::VectorXd costs = Eigen::VectorXd::Zero(K);

  if (weight_ <= 0.0 || reference.rows() < 2) {
    return costs;
  }

  int N = reference.rows() - 1;

  // 경로 접선 벡터 사전 계산 (정규화)
  std::vector<double> tx(N), ty(N);
  for (int t = 0; t < N; ++t) {
    double dx = reference(t + 1, 0) - reference(t, 0);
    double dy = reference(t + 1, 1) - reference(t, 1);
    double len = std::sqrt(dx * dx + dy * dy);
    if (len > 1e-6) {
      tx[t] = dx / len;
      ty[t] = dy / len;
    } else {
      tx[t] = (t > 0) ? tx[t - 1] : 1.0;
      ty[t] = (t > 0) ? ty[t - 1] : 0.0;
    }
  }

  #pragma omp parallel for schedule(static) if(K > 4096)
  for (int k = 0; k < K; ++k) {
    const auto& traj = trajectories[k];
    int T = std::min(N, static_cast<int>(traj.rows()) - 1);
    double cost = 0.0;

    for (int t = 0; t < T; ++t) {
      double vel_x = (traj(t + 1, 0) - traj(t, 0)) / dt_;
      double vel_y = (traj(t + 1, 1) - traj(t, 1)) / dt_;
      double v_along = tx[t] * vel_x + ty[t] * vel_y;
      double err = v_along - reference_velocity_;
      cost += err * err;
    }

    costs(k) = weight_ * cost;
  }

  return costs;
}

// ============================================================================
// CompositeMPPICost
// ============================================================================

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

CostBreakdown CompositeMPPICost::computeDetailed(
  const std::vector<Eigen::MatrixXd>& trajectories,
  const std::vector<Eigen::MatrixXd>& controls,
  const Eigen::MatrixXd& reference
) const
{
  CostBreakdown breakdown;
  int K = trajectories.size();
  breakdown.total_costs = Eigen::VectorXd::Zero(K);

  for (const auto& cost_fn : costs_) {
    Eigen::VectorXd component = cost_fn->compute(trajectories, controls, reference);
    breakdown.component_costs[cost_fn->name()] = component;
    breakdown.total_costs += component;
  }

  return breakdown;
}

// ============================================================================
// UncertaintyAwareCost
// ============================================================================

UncertaintyAwareCost::UncertaintyAwareCost(
  EnsembleDynamicsModel* ensemble_model, double weight, double dt)
: ensemble_model_(ensemble_model), weight_(weight), dt_(dt)
{
}

Eigen::VectorXd UncertaintyAwareCost::compute(
  const std::vector<Eigen::MatrixXd>& trajectories,
  const std::vector<Eigen::MatrixXd>& controls,
  const Eigen::MatrixXd& reference
) const
{
  (void)reference;

  int K = trajectories.size();
  Eigen::VectorXd costs = Eigen::VectorXd::Zero(K);

  if (!ensemble_model_ || weight_ <= 0.0) {
    return costs;
  }

  for (int k = 0; k < K; ++k) {
    const auto& traj = trajectories[k];
    const auto& ctrl = controls[k];
    int N = ctrl.rows();
    double cost_k = 0.0;

    for (int t = 0; t < N; ++t) {
      Eigen::MatrixXd state_mat = traj.row(t);
      Eigen::MatrixXd ctrl_mat = ctrl.row(t);

      auto result = ensemble_model_->predictWithUncertainty(state_mat, ctrl_mat);
      cost_k += result.variance.sum();
    }

    costs(k) = weight_ * cost_k;
  }

  return costs;
}

// ============================================================================
// C3BFCost
// ============================================================================

C3BFCost::C3BFCost(double weight, double dt, double alpha_safe)
: weight_(weight), dt_(dt), alpha_safe_(alpha_safe)
{
}

void C3BFCost::setObstacles(const std::vector<Eigen::Vector3d>& obstacles,
                             const std::vector<Eigen::Vector2d>& velocities)
{
  barriers_.clear();
  barriers_.reserve(obstacles.size());
  for (size_t i = 0; i < obstacles.size(); ++i) {
    C3BFBarrier b(obstacles[i](0), obstacles[i](1), obstacles[i](2),
                  robot_radius_, safety_margin_, alpha_safe_);
    if (i < velocities.size()) {
      b.updateObstacleVelocity(velocities[i](0), velocities[i](1));
    }
    barriers_.push_back(std::move(b));
  }
}

void C3BFCost::setObstacles(const std::vector<Eigen::Vector3d>& obstacles)
{
  std::vector<Eigen::Vector2d> zero_vels(obstacles.size(), Eigen::Vector2d::Zero());
  setObstacles(obstacles, zero_vels);
}

Eigen::VectorXd C3BFCost::compute(
  const std::vector<Eigen::MatrixXd>& trajectories,
  const std::vector<Eigen::MatrixXd>& controls,
  const Eigen::MatrixXd& reference
) const
{
  (void)controls;
  (void)reference;

  int K = trajectories.size();
  Eigen::VectorXd costs = Eigen::VectorXd::Zero(K);

  if (barriers_.empty()) {
    return costs;
  }

  #pragma omp parallel for schedule(static) if(K > 4096)
  for (int k = 0; k < K; ++k) {
    const auto& traj = trajectories[k];
    int N = traj.rows() - 1;
    double cost_k = 0.0;

    for (int t = 0; t < N; ++t) {
      // 궤적에서 로봇 속도 추정: v = (x_{t+1} - x_t) / dt
      double robot_vx = (traj(t + 1, 0) - traj(t, 0)) / dt_;
      double robot_vy = (traj(t + 1, 1) - traj(t, 1)) / dt_;

      for (const auto& barrier : barriers_) {
        Eigen::VectorXd state = traj.row(t).transpose();
        double h = barrier.evaluate(state, robot_vx, robot_vy);

        // 음의 h = 위험 → 페널티
        if (h < 0.0) {
          cost_k += weight_ * h * h;
        }
      }
    }

    costs(k) = cost_k;
  }

  return costs;
}

}  // namespace mpc_controller_ros2
