#ifndef MPC_CONTROLLER_ROS2__COST_FUNCTIONS_HPP_
#define MPC_CONTROLLER_ROS2__COST_FUNCTIONS_HPP_

#include <Eigen/Dense>
#include <vector>
#include <map>
#include <memory>
#include <string>
#include <nav2_costmap_2d/costmap_2d.hpp>
#include <nav2_costmap_2d/cost_values.hpp>
#include "mpc_controller_ros2/mppi_params.hpp"
#include "mpc_controller_ros2/barrier_function.hpp"

namespace mpc_controller_ros2
{

/**
 * @brief 비용 분해 구조체 (디버그 시각화용)
 */
struct CostBreakdown
{
  Eigen::VectorXd total_costs;                             // (K,)
  std::map<std::string, Eigen::VectorXd> component_costs;  // name → (K,)
};

/**
 * @brief MPPI 비용 함수 베이스 클래스
 */
class MPPICostFunction
{
public:
  virtual ~MPPICostFunction() = default;

  /** @brief 비용 함수 이름 (디버그 식별용) */
  virtual std::string name() const = 0;

  /**
   * @brief 비용 계산
   * @param trajectories 궤적 벡터 [K개, 각각 (N+1) x 3]
   * @param controls 제어 벡터 [K개, 각각 N x 2]
   * @param reference 참조 궤적 (N+1 x 3)
   * @return 비용 벡터 (K,)
   */
  virtual Eigen::VectorXd compute(
    const std::vector<Eigen::MatrixXd>& trajectories,
    const std::vector<Eigen::MatrixXd>& controls,
    const Eigen::MatrixXd& reference
  ) const = 0;
};

// 구체 비용 함수들
class StateTrackingCost : public MPPICostFunction
{
public:
  explicit StateTrackingCost(const Eigen::MatrixXd& Q);
  std::string name() const override { return "state_tracking"; }
  Eigen::VectorXd compute(
    const std::vector<Eigen::MatrixXd>& trajectories,
    const std::vector<Eigen::MatrixXd>& controls,
    const Eigen::MatrixXd& reference
  ) const override;
private:
  Eigen::MatrixXd Q_;
};

class TerminalCost : public MPPICostFunction
{
public:
  explicit TerminalCost(const Eigen::MatrixXd& Qf);
  std::string name() const override { return "terminal"; }
  Eigen::VectorXd compute(
    const std::vector<Eigen::MatrixXd>& trajectories,
    const std::vector<Eigen::MatrixXd>& controls,
    const Eigen::MatrixXd& reference
  ) const override;
private:
  Eigen::MatrixXd Qf_;
};

class ControlEffortCost : public MPPICostFunction
{
public:
  explicit ControlEffortCost(const Eigen::MatrixXd& R);
  std::string name() const override { return "control_effort"; }
  Eigen::VectorXd compute(
    const std::vector<Eigen::MatrixXd>& trajectories,
    const std::vector<Eigen::MatrixXd>& controls,
    const Eigen::MatrixXd& reference
  ) const override;
private:
  Eigen::MatrixXd R_;
};

class ControlRateCost : public MPPICostFunction
{
public:
  explicit ControlRateCost(const Eigen::MatrixXd& R_rate);
  std::string name() const override { return "control_rate"; }
  Eigen::VectorXd compute(
    const std::vector<Eigen::MatrixXd>& trajectories,
    const std::vector<Eigen::MatrixXd>& controls,
    const Eigen::MatrixXd& reference
  ) const override;
private:
  Eigen::MatrixXd R_rate_;
};

/**
 * @brief 전진 선호 비용 (후진 시 페널티)
 * cost = weight * Σ (ratio * |v| + (1-ratio) * v²)  for v < 0
 * linear_ratio=0.0 → 기존 이차 비용, linear_ratio=0.5 → 선형+이차 혼합
 */
class PreferForwardCost : public MPPICostFunction
{
public:
  explicit PreferForwardCost(double weight, double linear_ratio = 0.0,
                             double velocity_incentive = 0.0);
  std::string name() const override { return "prefer_forward"; }
  Eigen::VectorXd compute(
    const std::vector<Eigen::MatrixXd>& trajectories,
    const std::vector<Eigen::MatrixXd>& controls,
    const Eigen::MatrixXd& reference
  ) const override;
private:
  double weight_;
  double linear_ratio_;
  double velocity_incentive_;
};

class ObstacleCost : public MPPICostFunction
{
public:
  ObstacleCost(double weight, double safety_distance);
  std::string name() const override { return "obstacle"; }

  void setObstacles(const std::vector<Eigen::Vector3d>& obstacles);

  Eigen::VectorXd compute(
    const std::vector<Eigen::MatrixXd>& trajectories,
    const std::vector<Eigen::MatrixXd>& controls,
    const Eigen::MatrixXd& reference
  ) const override;
private:
  double weight_;
  double safety_distance_;
  std::vector<Eigen::Vector3d> obstacles_;
};

/**
 * @brief Costmap 기반 장애물 비용 (TF 변환 + inflation gradient)
 *
 * 기존 ObstacleCost와 달리 costmap을 직접 참조하여:
 * - map→odom 좌표 변환 적용
 * - LETHAL/INSCRIBED/inflation gradient 비용 연속 반영
 */
class CostmapObstacleCost : public MPPICostFunction
{
public:
  CostmapObstacleCost(double weight, double lethal_cost = 1000.0,
                       double critical_cost = 100.0);
  std::string name() const override { return "costmap_obstacle"; }

  void setCostmap(nav2_costmap_2d::Costmap2D* costmap);
  void setMapToOdomTransform(double tx, double ty,
                             double cos_th, double sin_th, bool use_tf);
  void setLethalCost(double cost) { lethal_cost_ = cost; }
  void setCriticalCost(double cost) { critical_cost_ = cost; }

  Eigen::VectorXd compute(
    const std::vector<Eigen::MatrixXd>& trajectories,
    const std::vector<Eigen::MatrixXd>& controls,
    const Eigen::MatrixXd& reference
  ) const override;

  /**
   * @brief 점별 costmap 비용 계산 (디버그 시각화용)
   * @param trajectories 궤적 벡터 [K개]
   * @return 비용 행렬 (K × T) — 각 궤적의 각 점에서의 costmap 비용
   */
  Eigen::MatrixXd computePerPoint(
    const std::vector<Eigen::MatrixXd>& trajectories
  ) const;

private:
  nav2_costmap_2d::Costmap2D* costmap_{nullptr};
  double weight_;
  double lethal_cost_;
  double critical_cost_;
  double tx_{0.0}, ty_{0.0}, cos_th_{1.0}, sin_th_{0.0};
  bool use_tf_{false};
};

/**
 * @brief CBF (Control Barrier Function) Soft Cost
 *
 * DCBF 이산 조건 위반 시 제곱 페널티:
 *   violation = max(0, (1-γ·dt)·h(x_t) - h(x_{t+1}))²
 *
 * MPPI 샘플링에서 안전 궤적을 유도하는 soft cost.
 * (Hard constraint는 CBFSafetyFilter가 담당)
 */
class CBFCost : public MPPICostFunction
{
public:
  CBFCost(BarrierFunctionSet* barrier_set, double weight, double gamma, double dt);
  std::string name() const override { return "cbf"; }

  Eigen::VectorXd compute(
    const std::vector<Eigen::MatrixXd>& trajectories,
    const std::vector<Eigen::MatrixXd>& controls,
    const Eigen::MatrixXd& reference
  ) const override;

private:
  BarrierFunctionSet* barrier_set_;
  double weight_;
  double decay_;  // 1 - γ·dt
};

/**
 * @brief 복합 비용 함수 (모든 비용 합산)
 */
class CompositeMPPICost
{
public:
  void addCost(std::unique_ptr<MPPICostFunction> cost);
  void clearCosts();

  Eigen::VectorXd compute(
    const std::vector<Eigen::MatrixXd>& trajectories,
    const std::vector<Eigen::MatrixXd>& controls,
    const Eigen::MatrixXd& reference
  ) const;

  /**
   * @brief 비용 분해 계산 (디버그용)
   * 각 비용 함수를 개별 호출하고 name() 키로 map에 저장
   */
  CostBreakdown computeDetailed(
    const std::vector<Eigen::MatrixXd>& trajectories,
    const std::vector<Eigen::MatrixXd>& controls,
    const Eigen::MatrixXd& reference
  ) const;

private:
  std::vector<std::unique_ptr<MPPICostFunction>> costs_;
};

}  // namespace mpc_controller_ros2

#endif  // MPC_CONTROLLER_ROS2__COST_FUNCTIONS_HPP_
