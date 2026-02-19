#ifndef MPC_CONTROLLER_ROS2__COST_FUNCTIONS_HPP_
#define MPC_CONTROLLER_ROS2__COST_FUNCTIONS_HPP_

#include <Eigen/Dense>
#include <vector>
#include <memory>
#include <nav2_costmap_2d/costmap_2d.hpp>
#include <nav2_costmap_2d/cost_values.hpp>
#include "mpc_controller_ros2/mppi_params.hpp"

namespace mpc_controller_ros2
{

/**
 * @brief MPPI 비용 함수 베이스 클래스
 */
class MPPICostFunction
{
public:
  virtual ~MPPICostFunction() = default;

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
  explicit StateTrackingCost(const Eigen::Matrix3d& Q);
  Eigen::VectorXd compute(
    const std::vector<Eigen::MatrixXd>& trajectories,
    const std::vector<Eigen::MatrixXd>& controls,
    const Eigen::MatrixXd& reference
  ) const override;
private:
  Eigen::Matrix3d Q_;
};

class TerminalCost : public MPPICostFunction
{
public:
  explicit TerminalCost(const Eigen::Matrix3d& Qf);
  Eigen::VectorXd compute(
    const std::vector<Eigen::MatrixXd>& trajectories,
    const std::vector<Eigen::MatrixXd>& controls,
    const Eigen::MatrixXd& reference
  ) const override;
private:
  Eigen::Matrix3d Qf_;
};

class ControlEffortCost : public MPPICostFunction
{
public:
  explicit ControlEffortCost(const Eigen::Matrix2d& R);
  Eigen::VectorXd compute(
    const std::vector<Eigen::MatrixXd>& trajectories,
    const std::vector<Eigen::MatrixXd>& controls,
    const Eigen::MatrixXd& reference
  ) const override;
private:
  Eigen::Matrix2d R_;
};

class ControlRateCost : public MPPICostFunction
{
public:
  explicit ControlRateCost(const Eigen::Matrix2d& R_rate);
  Eigen::VectorXd compute(
    const std::vector<Eigen::MatrixXd>& trajectories,
    const std::vector<Eigen::MatrixXd>& controls,
    const Eigen::MatrixXd& reference
  ) const override;
private:
  Eigen::Matrix2d R_rate_;
};

/**
 * @brief 전진 선호 비용 (후진 시 페널티)
 * cost = weight * Σ (ratio * |v| + (1-ratio) * v²)  for v < 0
 * linear_ratio=0.0 → 기존 이차 비용, linear_ratio=0.5 → 선형+이차 혼합
 */
class PreferForwardCost : public MPPICostFunction
{
public:
  explicit PreferForwardCost(double weight, double linear_ratio = 0.0);
  Eigen::VectorXd compute(
    const std::vector<Eigen::MatrixXd>& trajectories,
    const std::vector<Eigen::MatrixXd>& controls,
    const Eigen::MatrixXd& reference
  ) const override;
private:
  double weight_;
  double linear_ratio_;
};

class ObstacleCost : public MPPICostFunction
{
public:
  ObstacleCost(double weight, double safety_distance);

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

  void setCostmap(nav2_costmap_2d::Costmap2D* costmap);
  void setMapToOdomTransform(double tx, double ty,
                             double cos_th, double sin_th, bool use_tf);

  Eigen::VectorXd compute(
    const std::vector<Eigen::MatrixXd>& trajectories,
    const std::vector<Eigen::MatrixXd>& controls,
    const Eigen::MatrixXd& reference
  ) const override;

private:
  nav2_costmap_2d::Costmap2D* costmap_{nullptr};
  double weight_;
  double lethal_cost_;
  double critical_cost_;
  double tx_{0.0}, ty_{0.0}, cos_th_{1.0}, sin_th_{0.0};
  bool use_tf_{false};
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

private:
  std::vector<std::unique_ptr<MPPICostFunction>> costs_;
};

}  // namespace mpc_controller_ros2

#endif  // MPC_CONTROLLER_ROS2__COST_FUNCTIONS_HPP_
