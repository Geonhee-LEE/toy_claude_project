#ifndef MPC_CONTROLLER_ROS2__CUDA_MPPI_CONTROLLER_PLUGIN_HPP_
#define MPC_CONTROLLER_ROS2__CUDA_MPPI_CONTROLLER_PLUGIN_HPP_

#include "mpc_controller_ros2/mppi_controller_plugin.hpp"
#include "mpc_controller_ros2/cuda_rollout_engine.hpp"

namespace mpc_controller_ros2
{

/**
 * @brief CUDA GPU 가속 MPPI 컨트롤러 플러그인
 *
 * CudaRolloutEngine을 사용하여 K개 샘플 롤아웃 + 비용 계산을 GPU에서 병렬 실행.
 * CUDA 미사용 환경에서는 CPU 폴백으로 동일 결과 보장.
 *
 * isAvailable() = false → CPU 참조 구현
 * isAvailable() = true  → GPU CUDA 커널
 *
 * nav2 파라미터:
 *   cuda_enabled: true              # GPU 롤아웃 활성화
 *   cuda_device_id: 0               # CUDA 디바이스 ID
 *   cuda_download_trajectories: false # 시각화용 궤적 다운로드
 *   cuda_max_obstacles: 128         # 최대 장애물 수
 */
class CudaMPPIControllerPlugin : public MPPIControllerPlugin
{
public:
  CudaMPPIControllerPlugin() = default;
  ~CudaMPPIControllerPlugin() override = default;

  void configure(
    const rclcpp_lifecycle::LifecycleNode::WeakPtr& parent,
    std::string name,
    std::shared_ptr<tf2_ros::Buffer> tf,
    std::shared_ptr<nav2_costmap_2d::Costmap2DROS> costmap_ros
  ) override;

protected:
  std::pair<Eigen::VectorXd, MPPIInfo> computeControl(
    const Eigen::VectorXd& current_state,
    const Eigen::MatrixXd& reference_trajectory
  ) override;

private:
  std::unique_ptr<CudaRolloutEngine> cuda_engine_;
  bool gpu_available_{false};

  // CUDA 전용 파라미터 (MPPIParams에 없으므로 로컬 저장)
  bool cuda_enabled_{false};
  int cuda_device_id_{0};
  bool cuda_download_trajectories_{false};
  int cuda_max_obstacles_{128};
};

}  // namespace mpc_controller_ros2

#endif  // MPC_CONTROLLER_ROS2__CUDA_MPPI_CONTROLLER_PLUGIN_HPP_
