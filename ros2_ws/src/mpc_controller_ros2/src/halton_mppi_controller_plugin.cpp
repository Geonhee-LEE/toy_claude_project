// =============================================================================
// Halton-MPPI Controller Plugin
//
// MDPI Drones 2026 기반: Halton 저불일치 시퀀스로 제어 공간을
// 균일하게 커버하여 적은 샘플(K)로도 빠른 수렴 달성.
//
// configure()에서 sampler_를 HaltonSampler로 교체.
// computeControl은 부모(MPPIControllerPlugin)의 것을 그대로 사용.
// =============================================================================

#include "mpc_controller_ros2/halton_mppi_controller_plugin.hpp"
#include "mpc_controller_ros2/halton_sampler.hpp"
#include <pluginlib/class_list_macros.hpp>

PLUGINLIB_EXPORT_CLASS(mpc_controller_ros2::HaltonMPPIControllerPlugin, nav2_core::Controller)

namespace mpc_controller_ros2
{

void HaltonMPPIControllerPlugin::configure(
  const rclcpp_lifecycle::LifecycleNode::WeakPtr& parent,
  std::string name,
  std::shared_ptr<tf2_ros::Buffer> tf,
  std::shared_ptr<nav2_costmap_2d::Costmap2DROS> costmap_ros)
{
  MPPIControllerPlugin::configure(parent, name, tf, costmap_ros);

  // Replace sampler with HaltonSampler
  if (params_.halton_enabled) {
    sampler_ = std::make_unique<HaltonSampler>(
      params_.noise_sigma,
      params_.halton_beta,
      params_.halton_sequence_offset);

    auto node = parent.lock();
    RCLCPP_INFO(
      node->get_logger(),
      "Halton-MPPI plugin configured: beta=%.1f, offset=%d",
      params_.halton_beta, params_.halton_sequence_offset);
  }
}

}  // namespace mpc_controller_ros2
