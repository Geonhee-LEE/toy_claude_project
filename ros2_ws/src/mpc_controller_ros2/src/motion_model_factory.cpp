#include "mpc_controller_ros2/motion_model_factory.hpp"
#include "mpc_controller_ros2/diff_drive_model.hpp"
#include "mpc_controller_ros2/swerve_drive_model.hpp"
#include "mpc_controller_ros2/non_coaxial_swerve_model.hpp"
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
    // 현재 MPPIParams에 vy_max 없으므로 v_max를 공유
    double vx_min = params.v_min;
    double vx_max = params.v_max;
    double vy_max = params.v_max;  // TODO: params.vy_max 추가 시 교체
    double omega_max = params.omega_max;
    return std::make_unique<SwerveDriveModel>(vx_min, vx_max, vy_max, omega_max);
  }
  else if (model_type == "non_coaxial_swerve") {
    return std::make_unique<NonCoaxialSwerveModel>(
      params.v_min, params.v_max, params.omega_max,
      params.max_steering_rate, params.max_steering_angle);
  }
  else {
    throw std::invalid_argument(
      "Unknown motion model type: '" + model_type + "'. "
      "Supported: 'diff_drive', 'swerve', 'non_coaxial_swerve'");
  }
}

}  // namespace mpc_controller_ros2
