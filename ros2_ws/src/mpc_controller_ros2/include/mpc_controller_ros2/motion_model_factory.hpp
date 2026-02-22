#ifndef MPC_CONTROLLER_ROS2__MOTION_MODEL_FACTORY_HPP_
#define MPC_CONTROLLER_ROS2__MOTION_MODEL_FACTORY_HPP_

#include "mpc_controller_ros2/motion_model.hpp"
#include "mpc_controller_ros2/mppi_params.hpp"
#include <memory>
#include <string>

namespace mpc_controller_ros2
{

/**
 * @brief MotionModel Factory
 *
 * model_type 문자열로 적절한 MotionModel 인스턴스를 생성합니다.
 *
 * 지원 모델:
 *   "diff_drive"            → DiffDriveModel (nx=3, nu=2)
 *   "swerve"                → SwerveDriveModel (nx=3, nu=3)
 *   "non_coaxial_swerve"    → NonCoaxialSwerveModel (nx=4, nu=3)
 */
class MotionModelFactory
{
public:
  /**
   * @brief MotionModel 생성
   * @param model_type 모델 타입 문자열
   * @param params MPPI 파라미터 (제어 한계값 등)
   * @return 생성된 MotionModel unique_ptr
   * @throws std::invalid_argument 지원하지 않는 model_type인 경우
   */
  static std::unique_ptr<MotionModel> create(
    const std::string& model_type,
    const MPPIParams& params);
};

}  // namespace mpc_controller_ros2

#endif  // MPC_CONTROLLER_ROS2__MOTION_MODEL_FACTORY_HPP_
