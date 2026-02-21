#ifndef MPC_CONTROLLER_ROS2__BATCH_DYNAMICS_WRAPPER_HPP_
#define MPC_CONTROLLER_ROS2__BATCH_DYNAMICS_WRAPPER_HPP_

#include <Eigen/Dense>
#include <memory>
#include <vector>
#include "mpc_controller_ros2/mppi_params.hpp"
#include "mpc_controller_ros2/motion_model.hpp"

namespace mpc_controller_ros2
{

/**
 * @brief 로봇 동역학 배치 처리 래퍼 (MotionModel 위임)
 *
 * 내부적으로 MotionModel에 위임하되, 기존 API를 그대로 유지하여
 * 상위 코드 변경을 최소화합니다.
 *
 * 하위호환:
 *   BatchDynamicsWrapper(params)         → DiffDriveModel 자동 생성
 *   BatchDynamicsWrapper(params, model)  → 외부 모델 주입
 */
class BatchDynamicsWrapper
{
public:
  /** @brief 하위호환 생성자 (DiffDriveModel 자동 생성) */
  explicit BatchDynamicsWrapper(const MPPIParams& params);

  /** @brief 모델 주입 생성자 */
  BatchDynamicsWrapper(const MPPIParams& params, std::shared_ptr<MotionModel> model);

  /**
   * @brief 연속 동역학 (배치)
   * @param states 상태 행렬 (M x nx)
   * @param controls 제어 행렬 (M x nu)
   * @return 상태 미분 (M x nx)
   */
  Eigen::MatrixXd dynamicsBatch(
    const Eigen::MatrixXd& states,
    const Eigen::MatrixXd& controls
  ) const;

  /**
   * @brief RK4 적분 (단일 스텝, 배치)
   * @param states 현재 상태 (M x nx)
   * @param controls 제어 입력 (M x nu)
   * @param dt 시간 간격
   * @return 다음 상태 (M x nx)
   */
  Eigen::MatrixXd propagateBatch(
    const Eigen::MatrixXd& states,
    const Eigen::MatrixXd& controls,
    double dt
  ) const;

  /**
   * @brief 제어 시퀀스 배치 Rollout
   * @param x0 초기 상태 (nx,)
   * @param control_sequences 제어 시퀀스 벡터 [K개, 각각 N x nu]
   * @param dt 시간 간격
   * @return 궤적 벡터 [K개, 각각 (N+1) x nx]
   */
  std::vector<Eigen::MatrixXd> rolloutBatch(
    const Eigen::VectorXd& x0,
    const std::vector<Eigen::MatrixXd>& control_sequences,
    double dt
  ) const;

  /**
   * @brief 제어 입력 클리핑
   * @param controls 제어 행렬 (M x nu)
   * @return 클리핑된 제어 행렬
   */
  Eigen::MatrixXd clipControls(const Eigen::MatrixXd& controls) const;

  /** @brief 내부 MotionModel 참조 */
  const MotionModel& model() const { return *model_; }

private:
  MPPIParams params_;
  std::shared_ptr<MotionModel> model_;
};

}  // namespace mpc_controller_ros2

#endif  // MPC_CONTROLLER_ROS2__BATCH_DYNAMICS_WRAPPER_HPP_
