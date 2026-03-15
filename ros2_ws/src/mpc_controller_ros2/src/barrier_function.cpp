#include "mpc_controller_ros2/barrier_function.hpp"
#include <cmath>

namespace mpc_controller_ros2
{

// ============================================================================
// CircleBarrier
// ============================================================================

CircleBarrier::CircleBarrier(double obs_x, double obs_y, double obs_radius,
                             double robot_radius, double safety_margin)
: obs_x_(obs_x), obs_y_(obs_y)
{
  d_safe_ = obs_radius + robot_radius + safety_margin;
  d_safe_sq_ = d_safe_ * d_safe_;
}

double CircleBarrier::evaluate(const Eigen::VectorXd& state) const
{
  double dx = state(0) - obs_x_;
  double dy = state(1) - obs_y_;
  return dx * dx + dy * dy - d_safe_sq_;
}

Eigen::VectorXd CircleBarrier::evaluateBatch(const Eigen::MatrixXd& states) const
{
  int M = states.rows();
  Eigen::VectorXd h(M);
  for (int i = 0; i < M; ++i) {
    double dx = states(i, 0) - obs_x_;
    double dy = states(i, 1) - obs_y_;
    h(i) = dx * dx + dy * dy - d_safe_sq_;
  }
  return h;
}

Eigen::VectorXd CircleBarrier::gradient(const Eigen::VectorXd& state) const
{
  int nx = state.size();
  Eigen::VectorXd grad = Eigen::VectorXd::Zero(nx);
  grad(0) = 2.0 * (state(0) - obs_x_);
  grad(1) = 2.0 * (state(1) - obs_y_);
  return grad;
}

// ============================================================================
// BarrierFunctionSet
// ============================================================================

BarrierFunctionSet::BarrierFunctionSet(double robot_radius,
                                       double safety_margin,
                                       double activation_distance)
: robot_radius_(robot_radius),
  safety_margin_(safety_margin),
  activation_distance_(activation_distance)
{
}

void BarrierFunctionSet::setObstacles(const std::vector<Eigen::Vector3d>& obstacles)
{
  obstacles_raw_ = obstacles;  // 원본 저장 (마진 변경 시 재구축용)
  barriers_.clear();
  barriers_.reserve(obstacles.size());
  for (const auto& obs : obstacles) {
    barriers_.emplace_back(obs(0), obs(1), obs(2), robot_radius_, safety_margin_);
  }
}

void BarrierFunctionSet::updateSafetyMargin(double new_margin)
{
  if (std::abs(new_margin - safety_margin_) < 1e-12) {
    return;  // 변경 없음
  }
  safety_margin_ = new_margin;

  // 저장된 장애물 원본으로 재구축
  barriers_.clear();
  barriers_.reserve(obstacles_raw_.size());
  for (const auto& obs : obstacles_raw_) {
    barriers_.emplace_back(obs(0), obs(1), obs(2), robot_radius_, safety_margin_);
  }
}

std::vector<const CircleBarrier*> BarrierFunctionSet::getActiveBarriers(
  const Eigen::VectorXd& state) const
{
  std::vector<const CircleBarrier*> active;
  for (const auto& barrier : barriers_) {
    double dx = state(0) - barrier.obsX();
    double dy = state(1) - barrier.obsY();
    double dist = std::sqrt(dx * dx + dy * dy);
    if (dist <= activation_distance_) {
      active.push_back(&barrier);
    }
  }
  return active;
}

Eigen::VectorXd BarrierFunctionSet::evaluateAll(const Eigen::VectorXd& state) const
{
  int n = barriers_.size();
  Eigen::VectorXd values(n);
  for (int i = 0; i < n; ++i) {
    values(i) = barriers_[i].evaluate(state);
  }
  return values;
}

void BarrierFunctionSet::setObstaclesWithVelocity(
  const std::vector<Eigen::Vector3d>& obstacles,
  const std::vector<Eigen::Vector2d>& velocities)
{
  // 기존 CircleBarrier 설정 (하위호환)
  setObstacles(obstacles);

  // 속도 저장
  obstacle_velocities_ = velocities;

  // C3BF barriers 구축
  c3bf_barriers_.clear();
  c3bf_barriers_.reserve(obstacles.size());
  for (size_t i = 0; i < obstacles.size(); ++i) {
    C3BFBarrier b(obstacles[i](0), obstacles[i](1), obstacles[i](2),
                  robot_radius_, safety_margin_, alpha_safe_);
    if (i < velocities.size()) {
      b.updateObstacleVelocity(velocities[i](0), velocities[i](1));
    }
    c3bf_barriers_.push_back(std::move(b));
  }
}

std::vector<const C3BFBarrier*> BarrierFunctionSet::getActiveC3BFBarriers(
  const Eigen::VectorXd& state) const
{
  std::vector<const C3BFBarrier*> active;
  for (const auto& barrier : c3bf_barriers_) {
    double dx = state(0) - barrier.obsX();
    double dy = state(1) - barrier.obsY();
    double dist = std::sqrt(dx * dx + dy * dy);
    if (dist <= activation_distance_) {
      active.push_back(&barrier);
    }
  }
  return active;
}

// ============================================================================
// CBF 합성 메서드
// ============================================================================

double BarrierFunctionSet::evaluateComposite(
  const Eigen::VectorXd& state,
  CBFCompositionMethod method,
  double alpha) const
{
  auto active = getActiveBarriers(state);
  if (active.empty()) {
    return std::numeric_limits<double>::infinity();
  }

  switch (method) {
    case CBFCompositionMethod::MIN: {
      double h_min = active[0]->evaluate(state);
      for (size_t i = 1; i < active.size(); ++i) {
        h_min = std::min(h_min, active[i]->evaluate(state));
      }
      return h_min;
    }

    case CBFCompositionMethod::SMOOTH_MIN: {
      // h_c = -(1/α)·log(Σ exp(-α·h_i))
      // 수치 안정성: max_val 빼기
      double max_neg = -alpha * active[0]->evaluate(state);
      for (size_t i = 1; i < active.size(); ++i) {
        max_neg = std::max(max_neg, -alpha * active[i]->evaluate(state));
      }
      double sum_exp = 0.0;
      for (const auto* b : active) {
        sum_exp += std::exp(-alpha * b->evaluate(state) - max_neg);
      }
      return -(1.0 / alpha) * (std::log(sum_exp) + max_neg);
    }

    case CBFCompositionMethod::LOG_SUM_EXP: {
      // α=1 특수 경우: h_c = -log(Σ exp(-h_i))
      double max_neg = -active[0]->evaluate(state);
      for (size_t i = 1; i < active.size(); ++i) {
        max_neg = std::max(max_neg, -active[i]->evaluate(state));
      }
      double sum_exp = 0.0;
      for (const auto* b : active) {
        sum_exp += std::exp(-b->evaluate(state) - max_neg);
      }
      return -(std::log(sum_exp) + max_neg);
    }

    case CBFCompositionMethod::PRODUCT: {
      // h_c = Π max(0, h_i)
      double product = 1.0;
      for (const auto* b : active) {
        double h = b->evaluate(state);
        product *= std::max(0.0, h);
      }
      return product;
    }
  }

  return 0.0;  // 도달 불가
}

Eigen::VectorXd BarrierFunctionSet::compositeGradient(
  const Eigen::VectorXd& state,
  CBFCompositionMethod method,
  double alpha) const
{
  int nx = state.size();
  auto active = getActiveBarriers(state);
  if (active.empty()) {
    return Eigen::VectorXd::Zero(nx);
  }

  switch (method) {
    case CBFCompositionMethod::MIN: {
      // gradient of min = gradient of argmin barrier
      int min_idx = 0;
      double h_min = active[0]->evaluate(state);
      for (size_t i = 1; i < active.size(); ++i) {
        double h = active[i]->evaluate(state);
        if (h < h_min) {
          h_min = h;
          min_idx = i;
        }
      }
      return active[min_idx]->gradient(state);
    }

    case CBFCompositionMethod::SMOOTH_MIN: {
      // ∇h_c = Σ w_i · ∇h_i, where w_i = exp(-α·h_i) / Σ exp(-α·h_j)
      // 수치 안정성: max_neg 빼기
      std::vector<double> h_vals(active.size());
      double max_neg = -std::numeric_limits<double>::infinity();
      for (size_t i = 0; i < active.size(); ++i) {
        h_vals[i] = active[i]->evaluate(state);
        max_neg = std::max(max_neg, -alpha * h_vals[i]);
      }

      std::vector<double> exp_vals(active.size());
      double sum_exp = 0.0;
      for (size_t i = 0; i < active.size(); ++i) {
        exp_vals[i] = std::exp(-alpha * h_vals[i] - max_neg);
        sum_exp += exp_vals[i];
      }

      Eigen::VectorXd grad = Eigen::VectorXd::Zero(nx);
      for (size_t i = 0; i < active.size(); ++i) {
        double w_i = exp_vals[i] / sum_exp;
        grad += w_i * active[i]->gradient(state);
      }
      return grad;
    }

    case CBFCompositionMethod::LOG_SUM_EXP: {
      // α=1 특수 경우
      std::vector<double> h_vals(active.size());
      double max_neg = -std::numeric_limits<double>::infinity();
      for (size_t i = 0; i < active.size(); ++i) {
        h_vals[i] = active[i]->evaluate(state);
        max_neg = std::max(max_neg, -h_vals[i]);
      }

      std::vector<double> exp_vals(active.size());
      double sum_exp = 0.0;
      for (size_t i = 0; i < active.size(); ++i) {
        exp_vals[i] = std::exp(-h_vals[i] - max_neg);
        sum_exp += exp_vals[i];
      }

      Eigen::VectorXd grad = Eigen::VectorXd::Zero(nx);
      for (size_t i = 0; i < active.size(); ++i) {
        double w_i = exp_vals[i] / sum_exp;
        grad += w_i * active[i]->gradient(state);
      }
      return grad;
    }

    case CBFCompositionMethod::PRODUCT: {
      // h_c = Π max(0, h_i)
      // ∇h_c = Σ_i (Π_{j≠i} max(0, h_j)) · ∇h_i
      std::vector<double> h_vals(active.size());
      for (size_t i = 0; i < active.size(); ++i) {
        h_vals[i] = std::max(0.0, active[i]->evaluate(state));
      }

      Eigen::VectorXd grad = Eigen::VectorXd::Zero(nx);
      for (size_t i = 0; i < active.size(); ++i) {
        double prod_others = 1.0;
        for (size_t j = 0; j < active.size(); ++j) {
          if (j != i) {
            prod_others *= h_vals[j];
          }
        }
        // max(0, h_i) = 0 이면 gradient도 0
        if (active[i]->evaluate(state) > 0.0) {
          grad += prod_others * active[i]->gradient(state);
        }
      }
      return grad;
    }
  }

  return Eigen::VectorXd::Zero(nx);
}

}  // namespace mpc_controller_ros2
