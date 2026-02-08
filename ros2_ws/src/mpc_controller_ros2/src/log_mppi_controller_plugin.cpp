#include "mpc_controller_ros2/log_mppi_controller_plugin.hpp"
#include <pluginlib/class_list_macros.hpp>

PLUGINLIB_EXPORT_CLASS(mpc_controller_ros2::LogMPPIControllerPlugin, nav2_core::Controller)

namespace mpc_controller_ros2
{

LogMPPIControllerPlugin::LogMPPIControllerPlugin()
  : MPPIControllerPlugin()
{
  weight_computation_ = std::make_unique<LogMPPIWeights>();
}

}  // namespace mpc_controller_ros2
