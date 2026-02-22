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
#include "mpc_controller_ros2/adaptive_temperature.hpp"
#include "mpc_controller_ros2/tube_mppi.hpp"
#include "mpc_controller_ros2/weight_computation.hpp"
#include "mpc_controller_ros2/barrier_function.hpp"
#include "mpc_controller_ros2/cbf_safety_filter.hpp"
#include "mpc_controller_ros2/non_coaxial_swerve_model.hpp"

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

  // M2 확장 정보
  bool colored_noise_used{false};
  bool adaptive_temp_used{false};
  bool tube_mppi_used{false};
  TubeMPPIInfo tube_info;  // Tube-MPPI 정보

  // SVGD 전용 정보
  int svgd_iterations{0};
  double sample_diversity_before{0.0};
  double sample_diversity_after{0.0};

  // SVG-MPPI 전용 정보
  int num_guides{0};
  int num_followers{0};
  int guide_iterations{0};

  // CBF 안전성 정보
  bool cbf_used{false};
  CBFFilterInfo cbf_filter_info;

  // Collision debug (debug_collision_viz=true일 때만 채워짐)
  CostBreakdown cost_breakdown;
};

/**
 * @brief nav2 MPPI Controller Plugin
 */
class MPPIControllerPlugin : public nav2_core::Controller
{
public:
  MPPIControllerPlugin();
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

protected:
  // Weight computation strategy (서브클래스에서 교체 가능)
  std::unique_ptr<WeightComputation> weight_computation_;

  // MPPI 핵심 알고리즘 (서브클래스에서 override 가능)
  virtual std::pair<Eigen::VectorXd, MPPIInfo> computeControl(
    const Eigen::VectorXd& current_state,
    const Eigen::MatrixXd& reference_trajectory
  );

  // 서브클래스 접근 가능 멤버
  MPPIParams params_;
  Eigen::MatrixXd control_sequence_;  // N x nu
  std::unique_ptr<BatchDynamicsWrapper> dynamics_;
  std::unique_ptr<CompositeMPPICost> cost_function_;
  std::unique_ptr<BaseSampler> sampler_;
  std::unique_ptr<AdaptiveTemperature> adaptive_temp_;

  // ROS2 (서브클래스 로깅용)
  rclcpp_lifecycle::LifecycleNode::SharedPtr node_;
  std::string plugin_name_;

private:
  // 좌표 변환
  Eigen::VectorXd poseToState(const geometry_msgs::msg::PoseStamped& pose);
  Eigen::MatrixXd pathToReferenceTrajectory(
    const nav_msgs::msg::Path& path, const Eigen::VectorXd& current_state);

  // 경로 pruning + costmap 장애물 갱신
  void prunePlan(const Eigen::VectorXd& current_state);
  void updateCostmapObstacles();

  // 시각화
  void publishVisualization(
    const MPPIInfo& info,
    const Eigen::VectorXd& current_state,
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

  // ROS2 (플러그인 내부 전용)
  std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
  std::shared_ptr<nav2_costmap_2d::Costmap2DROS> costmap_ros_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr marker_pub_;
  rclcpp::node_interfaces::OnSetParametersCallbackHandle::SharedPtr param_callback_handle_;

  // M2 확장 컴포넌트
  std::unique_ptr<TubeMPPI> tube_mppi_;

  // State
  nav_msgs::msg::Path global_plan_;
  nav_msgs::msg::Path pruned_plan_;
  size_t prune_start_idx_{0};
  Eigen::MatrixXd nominal_trajectory_;  // N+1 x nx (Tube-MPPI용)
  double speed_limit_{1.0};
  bool speed_limit_valid_{false};
  Eigen::VectorXd current_velocity_;  // (nu,) 현재 속도
  double goal_dist_{std::numeric_limits<double>::max()};  // 목표까지 남은 거리

  // EMA 출력 필터
  Eigen::VectorXd prev_cmd_;
  bool prev_cmd_valid_{false};

  // Non-Coaxial Swerve: steering angle 추적 (poseToState/computeVelocityCommands)
  double last_delta_{0.0};

  // CostmapObstacleCost 비소유 포인터 (cost_function_ 내부 소유)
  CostmapObstacleCost* costmap_obstacle_cost_ptr_{nullptr};

  // CBF (Control Barrier Function) 컴포넌트
  BarrierFunctionSet barrier_set_;
  std::unique_ptr<CBFSafetyFilter> cbf_safety_filter_;
  void updateCBFObstacles();

  // Tube 시각화
  void publishTubeVisualization(
    const TubeMPPIInfo& tube_info,
    const Eigen::MatrixXd& nominal_trajectory
  );

  // 충돌 디버그 시각화
  void publishCollisionDebugVisualization(
    const MPPIInfo& info,
    const Eigen::VectorXd& current_state,
    const Eigen::MatrixXd& reference_trajectory,
    const Eigen::MatrixXd& weighted_avg_trajectory
  );
};

}  // namespace mpc_controller_ros2

#endif  // MPC_CONTROLLER_ROS2__MPPI_CONTROLLER_PLUGIN_HPP_
