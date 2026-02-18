#include "mpc_controller_ros2/tsallis_mppi_controller_plugin.hpp"
#include <pluginlib/class_list_macros.hpp>

PLUGINLIB_EXPORT_CLASS(mpc_controller_ros2::TsallisMPPIControllerPlugin, nav2_core::Controller)

namespace mpc_controller_ros2
{

void TsallisMPPIControllerPlugin::configure(
  const rclcpp_lifecycle::LifecycleNode::WeakPtr& parent,
  std::string name,
  std::shared_ptr<tf2_ros::Buffer> tf,
  std::shared_ptr<nav2_costmap_2d::Costmap2DROS> costmap_ros)
{
  // 부모 configure 호출 (파라미터 declare/load 포함)
  MPPIControllerPlugin::configure(parent, name, tf, costmap_ros);

  // tsallis_q 파라미터로 TsallisMPPIWeights Strategy 교체
  auto node = parent.lock();
  double q = node->get_parameter(name + ".tsallis_q").as_double();
  weight_computation_ = std::make_unique<TsallisMPPIWeights>(q);

  RCLCPP_INFO(
    node->get_logger(),
    "Tsallis-MPPI plugin configured: q=%.2f (weight strategy: %s)",
    q, weight_computation_->name().c_str());
}

}  // namespace mpc_controller_ros2
