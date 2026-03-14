#include "mpc_controller_ros2/motion_model_factory.hpp"
#include "mpc_controller_ros2/diff_drive_model.hpp"
#include "mpc_controller_ros2/swerve_drive_model.hpp"
#include "mpc_controller_ros2/non_coaxial_swerve_model.hpp"
#include "mpc_controller_ros2/ackermann_model.hpp"
#include "mpc_controller_ros2/wheel_level_4d_model.hpp"
#include "mpc_controller_ros2/residual_dynamics_model.hpp"
#include "mpc_controller_ros2/eigen_mlp.hpp"
#include <stdexcept>

namespace mpc_controller_ros2
{

std::unique_ptr<MotionModel> MotionModelFactory::create(
  const std::string& model_type,
  const MPPIParams& params)
{
  if (model_type == "diff_drive") {
    return std::make_unique<DiffDriveModel>(
      params.v_min, params.v_max, params.omega_min, params.omega_max);
  }
  else if (model_type == "swerve") {
    // Swerve: vx_min, vx_max, vy_max, omega_max
    double vx_min = params.v_min;
    double vx_max = params.v_max;
    double vy_max = (params.vy_max > 0.0) ? params.vy_max : params.v_max;
    double omega_max = params.omega_max;
    return std::make_unique<SwerveDriveModel>(vx_min, vx_max, vy_max, omega_max);
  }
  else if (model_type == "non_coaxial_swerve") {
    return std::make_unique<NonCoaxialSwerveModel>(
      params.v_min, params.v_max, params.omega_max,
      params.max_steering_rate, params.max_steering_angle);
  }
  else if (model_type == "ackermann") {
    return std::make_unique<AckermannModel>(
      params.v_min, params.v_max,
      params.max_steering_rate, params.max_steering_angle,
      params.wheelbase);
  }
  else if (model_type == "wheel_level_4d") {
    return std::make_unique<WheelLevel4DModel>(
      params.hybrid_lf, params.hybrid_lr,
      params.hybrid_dl, params.hybrid_dr,
      params.hybrid_v_wheel_max, params.hybrid_delta_max);
  }
  else {
    throw std::invalid_argument(
      "Unknown motion model type: '" + model_type + "'. "
      "Supported: 'diff_drive', 'swerve', 'non_coaxial_swerve', 'ackermann', 'wheel_level_4d'");
  }
}

std::unique_ptr<MotionModel> MotionModelFactory::createWithResidual(
  const std::string& model_type,
  const MPPIParams& params)
{
  // 공칭 모델 생성
  auto nominal = create(model_type, params);

  // MLP 로드
  if (params.residual_weights_path.empty()) {
    throw std::runtime_error(
      "MotionModelFactory::createWithResidual: residual_weights_path is empty");
  }

  auto mlp = EigenMLP::loadFromFile(params.residual_weights_path);

  // ResidualDynamicsModel로 래핑
  return std::make_unique<ResidualDynamicsModel>(
    std::move(nominal), std::move(mlp), params.residual_alpha);
}

}  // namespace mpc_controller_ros2
