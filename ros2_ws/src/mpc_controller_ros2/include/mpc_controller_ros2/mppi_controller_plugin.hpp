#ifndef MPC_CONTROLLER_ROS2__MPPI_CONTROLLER_PLUGIN_HPP_
#define MPC_CONTROLLER_ROS2__MPPI_CONTROLLER_PLUGIN_HPP_

#include <memory>
#include <string>
#include <vector>

#include "rclcpp/rclcpp.hpp"
#include "nav2_core/controller.hpp"
#include "nav2_costmap_2d/costmap_2d_ros.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "geometry_msgs/msg/twist_stamped.hpp"
#include "nav_msgs/msg/path.hpp"
#include "visualization_msgs/msg/marker_array.hpp"

#include "mpc_controller_ros2/mppi_params.hpp"
#include "mpc_controller_ros2/batch_dynamics_wrapper.hpp"
#include "mpc_controller_ros2/cost_functions.hpp"
#include "mpc_controller_ros2/sampling.hpp"

namespace mpc_controller_ros2
{

/**
 * @brief MPPI info 구조체
 */
struct MPPIInfo
{
  std::vector<Eigen::MatrixXd> sample_trajectories;
  Eigen::VectorXd sample_weights;
  Eigen::MatrixXd best_trajectory;
  Eigen::MatrixXd weighted_avg_trajectory;
  double temperature;
  double ess;
  Eigen::VectorXd costs;
};

/**
 * @brief nav2 MPPI Controller Plugin
 */
class MPPIControllerPlugin : public nav2_core::Controller
{
public:
  MPPIControllerPlugin() = default;
  ~MPPIControllerPlugin() override = default;

  // nav2_core::Controller 인터페이스
  void configure(
    const rclcpp_lifecycle::LifecycleNode::WeakPtr& parent,
    std::string name,
    std::shared_ptr<tf2_ros::Buffer> tf,
    std::shared_ptr<nav2_costmap_2d::Costmap2DROS> costmap_ros
  ) override;

  void cleanup() override;
  void activate() override;
  void deactivate() override;

  geometry_msgs::msg::TwistStamped computeVelocityCommands(
    const geometry_msgs::msg::PoseStamped& pose,
    const geometry_msgs::msg::Twist& velocity,
    nav2_core::GoalChecker* goal_checker
  ) override;

  void setPlan(const nav_msgs::msg::Path& path) override;
  void setSpeedLimit(const double& speed_limit, const bool& percentage) override;

private:
  // MPPI 핵심 알고리즘
  std::pair<Eigen::Vector2d, MPPIInfo> computeControl(
    const Eigen::Vector3d& current_state,
    const Eigen::MatrixXd& reference_trajectory
  );

  // 좌표 변환
  Eigen::Vector3d poseToState(const geometry_msgs::msg::PoseStamped& pose);
  Eigen::MatrixXd pathToReferenceTrajectory(const nav_msgs::msg::Path& path);
  std::vector<Eigen::Vector3d> extractObstaclesFromCostmap();

  // 시각화
  void publishVisualization(
    const MPPIInfo& info,
    const Eigen::Vector3d& current_state,
    const Eigen::MatrixXd& reference_trajectory,
    const Eigen::MatrixXd& weighted_avg_trajectory,
    double computation_time_ms
  );

  // 파라미터 관리
  void declareParameters();
  void loadParameters();
  rcl_interfaces::msg::SetParametersResult onSetParametersCallback(
    const std::vector<rclcpp::Parameter>& parameters
  );

  // ROS2
  rclcpp_lifecycle::LifecycleNode::SharedPtr node_;
  std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
  std::shared_ptr<nav2_costmap_2d::Costmap2DROS> costmap_ros_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr marker_pub_;
  rclcpp::node_interfaces::OnSetParametersCallbackHandle::SharedPtr param_callback_handle_;

  // MPPI 컴포넌트
  MPPIParams params_;
  std::unique_ptr<BatchDynamicsWrapper> dynamics_;
  std::unique_ptr<CompositeMPPICost> cost_function_;
  std::unique_ptr<BaseSampler> sampler_;

  // State
  nav_msgs::msg::Path global_plan_;
  Eigen::MatrixXd control_sequence_;  // N x 2
  double speed_limit_{1.0};
  bool speed_limit_valid_{false};

  std::string plugin_name_;
};

}  // namespace mpc_controller_ros2

#endif  // MPC_CONTROLLER_ROS2__MPPI_CONTROLLER_PLUGIN_HPP_
