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
 *   "ackermann"             → AckermannModel (nx=4, nu=2)
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

  /**
   * @brief Residual Dynamics 래핑된 MotionModel 생성
   * @param model_type 공칭 모델 타입 문자열
   * @param params MPPI 파라미터
   * @return ResidualDynamicsModel (내부에 공칭 모델 + MLP)
   * @throws std::runtime_error MLP 로드 실패 시
   */
  static std::unique_ptr<MotionModel> createWithResidual(
    const std::string& model_type,
    const MPPIParams& params);

  /**
   * @brief Ensemble Dynamics 래핑된 MotionModel 생성
   * @param model_type 공칭 모델 타입 문자열
   * @param params MPPI 파라미터 (ensemble_weights_dir, ensemble_size, ensemble_alpha)
   * @return EnsembleDynamicsModel (내부에 공칭 모델 + M개 MLP)
   * @throws std::runtime_error MLP 로드 실패 시
   */
  static std::unique_ptr<MotionModel> createWithEnsemble(
    const std::string& model_type,
    const MPPIParams& params);
};

}  // namespace mpc_controller_ros2

#endif  // MPC_CONTROLLER_ROS2__MOTION_MODEL_FACTORY_HPP_
