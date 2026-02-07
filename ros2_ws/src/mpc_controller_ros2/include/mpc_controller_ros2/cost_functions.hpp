#ifndef MPC_CONTROLLER_ROS2__COST_FUNCTIONS_HPP_
#define MPC_CONTROLLER_ROS2__COST_FUNCTIONS_HPP_

#include <Eigen/Dense>
#include <vector>
#include <memory>
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
 * @brief 복합 비용 함수 (모든 비용 합산)
 */
class CompositeMPPICost
{
public:
  void addCost(std::unique_ptr<MPPICostFunction> cost);

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
