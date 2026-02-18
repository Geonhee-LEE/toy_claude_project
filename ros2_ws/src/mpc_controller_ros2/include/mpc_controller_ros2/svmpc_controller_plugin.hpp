#ifndef MPC_CONTROLLER_ROS2__SVMPC_CONTROLLER_PLUGIN_HPP_
#define MPC_CONTROLLER_ROS2__SVMPC_CONTROLLER_PLUGIN_HPP_

#include "mpc_controller_ros2/mppi_controller_plugin.hpp"

namespace mpc_controller_ros2
{

/**
 * @brief Stein Variational MPC (SVMPC) nav2 Controller Plugin
 *
 * Lambert et al. (2020) "Stein Variational Model Predictive Control" 기반.
 * SVGD 커널로 샘플 간 상호작용을 통해 multi-modal 분포를 효과적으로 탐색.
 *
 * svgd_num_iterations=0 → Vanilla MPPI와 동등 (backward compatible)
 * svgd_num_iterations>0 → SVGD 루프로 샘플 다양성 증진
 *
 * 알고리즘 플로우:
 *   Phase 1: Vanilla 동일 (shift, sample, perturb, rollout, cost)
 *   Phase 2: SVGD Loop (L회)
 *     - flatten particles: (K,N,nu) → (K,D)
 *     - softmax weights, RBF kernel
 *     - attractive + repulsive force
 *     - particles += step_size * force
 *     - re-rollout, re-cost
 *   Phase 3: effective_noise 역산 → 가중 평균 업데이트
 */
class SVMPCControllerPlugin : public MPPIControllerPlugin
{
public:
  SVMPCControllerPlugin() = default;
  ~SVMPCControllerPlugin() override = default;

  void configure(
    const rclcpp_lifecycle::LifecycleNode::WeakPtr& parent,
    std::string name,
    std::shared_ptr<tf2_ros::Buffer> tf,
    std::shared_ptr<nav2_costmap_2d::Costmap2DROS> costmap_ros
  ) override;

protected:
  std::pair<Eigen::Vector2d, MPPIInfo> computeControl(
    const Eigen::Vector3d& current_state,
    const Eigen::MatrixXd& reference_trajectory
  ) override;

private:
  /**
   * @brief SVGD force 계산: attractive + repulsive
   *
   * attractive_i = Σ_j w_j · k(x_j, x_i) · (x_j - x_i)
   * repulsive_i  = (1/K) · Σ_j k(x_j, x_i) · (x_j - x_i) / h²
   *
   * @param diff (K, K, D) pairwise 차이 (x_j - x_i)
   * @param weights (K) softmax 가중치
   * @param kernel (K, K) RBF 커널 행렬
   * @param bandwidth h
   * @param K 샘플 수
   * @param D flatten 차원
   * @return (K, D) SVGD force
   */
  Eigen::MatrixXd computeSVGDForce(
    const std::vector<Eigen::VectorXd>& diff_flat,
    const Eigen::VectorXd& weights,
    const Eigen::MatrixXd& kernel,
    double bandwidth,
    int K,
    int D
  ) const;

  /**
   * @brief 샘플 다양성 측정 (평균 pairwise L2 거리)
   */
  static double computeDiversity(
    const std::vector<Eigen::MatrixXd>& controls,
    int K, int D
  );

  /**
   * @brief Median heuristic bandwidth 계산
   * h = sqrt(median(sq_dist) / (2 * log(K+1)))
   */
  static double medianBandwidth(
    const Eigen::MatrixXd& sq_dist, int K
  );
};

}  // namespace mpc_controller_ros2

#endif  // MPC_CONTROLLER_ROS2__SVMPC_CONTROLLER_PLUGIN_HPP_
