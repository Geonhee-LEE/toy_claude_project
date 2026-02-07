#ifndef MPC_CONTROLLER_ROS2__BATCH_DYNAMICS_WRAPPER_HPP_
#define MPC_CONTROLLER_ROS2__BATCH_DYNAMICS_WRAPPER_HPP_

#include <Eigen/Dense>
#include <vector>
#include "mpc_controller_ros2/mppi_params.hpp"

namespace mpc_controller_ros2
{

/**
 * @brief Differential Drive 로봇 동역학 배치 처리 래퍼
 */
class BatchDynamicsWrapper
{
public:
  explicit BatchDynamicsWrapper(const MPPIParams& params);

  /**
   * @brief 연속 동역학 (배치)
   * @param states 상태 행렬 (M x 3) [x, y, theta]
   * @param controls 제어 행렬 (M x 2) [v, omega]
   * @return 상태 미분 (M x 3) [x_dot, y_dot, theta_dot]
   */
  Eigen::MatrixXd dynamicsBatch(
    const Eigen::MatrixXd& states,
    const Eigen::MatrixXd& controls
  ) const;

  /**
   * @brief RK4 적분 (단일 스텝, 배치)
   * @param states 현재 상태 (M x 3)
   * @param controls 제어 입력 (M x 2)
   * @param dt 시간 간격
   * @return 다음 상태 (M x 3)
   */
  Eigen::MatrixXd propagateBatch(
    const Eigen::MatrixXd& states,
    const Eigen::MatrixXd& controls,
    double dt
  ) const;

  /**
   * @brief 제어 시퀀스 배치 Rollout
   * @param x0 초기 상태 (3,) [x, y, theta]
   * @param control_sequences 제어 시퀀스 벡터 [K개, 각각 N x 2]
   * @param dt 시간 간격
   * @return 궤적 벡터 [K개, 각각 (N+1) x 3]
   */
  std::vector<Eigen::MatrixXd> rolloutBatch(
    const Eigen::Vector3d& x0,
    const std::vector<Eigen::MatrixXd>& control_sequences,
    double dt
  ) const;

  /**
   * @brief 제어 입력 클리핑
   * @param controls 제어 행렬 (M x 2) [v, omega]
   * @return 클리핑된 제어 행렬
   */
  Eigen::MatrixXd clipControls(const Eigen::MatrixXd& controls) const;

private:
  MPPIParams params_;
};

}  // namespace mpc_controller_ros2

#endif  // MPC_CONTROLLER_ROS2__BATCH_DYNAMICS_WRAPPER_HPP_
