// =============================================================================
// CUDA MPPI Controller Plugin — GPU 가속 롤아웃 + CPU 폴백
//
// CudaRolloutEngine으로 K개 샘플 병렬 롤아웃 + 비용 계산.
// CUDA 미사용 시 CPU 참조 구현으로 동일 API/결과 보장.
// =============================================================================

#include "mpc_controller_ros2/cuda_mppi_controller_plugin.hpp"
#include "mpc_controller_ros2/utils.hpp"
#include <pluginlib/class_list_macros.hpp>

PLUGINLIB_EXPORT_CLASS(mpc_controller_ros2::CudaMPPIControllerPlugin, nav2_core::Controller)

namespace mpc_controller_ros2
{

void CudaMPPIControllerPlugin::configure(
  const rclcpp_lifecycle::LifecycleNode::WeakPtr& parent,
  std::string name,
  std::shared_ptr<tf2_ros::Buffer> tf,
  std::shared_ptr<nav2_costmap_2d::Costmap2DROS> costmap_ros)
{
  // 기본 MPPI 플러그인 설정
  MPPIControllerPlugin::configure(parent, name, tf, costmap_ros);

  auto node = parent.lock();
  if (!node) {
    return;
  }

  // CUDA 전용 파라미터 선언 + 로드
  node->declare_parameter(name + ".cuda_enabled", true);
  node->declare_parameter(name + ".cuda_device_id", 0);
  node->declare_parameter(name + ".cuda_download_trajectories", false);
  node->declare_parameter(name + ".cuda_max_obstacles", 128);

  cuda_enabled_ = node->get_parameter(name + ".cuda_enabled").as_bool();
  cuda_device_id_ = node->get_parameter(name + ".cuda_device_id").as_int();
  cuda_download_trajectories_ = node->get_parameter(
    name + ".cuda_download_trajectories").as_bool();
  cuda_max_obstacles_ = node->get_parameter(name + ".cuda_max_obstacles").as_int();

  if (!cuda_enabled_) {
    RCLCPP_INFO(node->get_logger(), "CUDA MPPI: disabled by parameter");
    return;
  }

  // CudaRolloutEngine 구성
  CudaRolloutEngine::Config config;
  config.K = params_.K;
  config.N = params_.N;
  config.nx = dynamics_->model().stateDim();
  config.nu = dynamics_->model().controlDim();
  config.max_obstacles = cuda_max_obstacles_;
  config.motion_model = params_.motion_model;

  cuda_engine_ = std::make_unique<CudaRolloutEngine>(config);

  // vy_max 결정
  double vy_max = params_.vy_max;

  cuda_engine_->initialize(
    params_.Q, params_.Qf, params_.R, params_.R_rate,
    params_.v_min, params_.v_max, params_.omega_min, params_.omega_max,
    vy_max);

  gpu_available_ = CudaRolloutEngine::isAvailable();

  RCLCPP_INFO(node->get_logger(),
    "CUDA MPPI configured: gpu_available=%s, K=%d, N=%d, model=%s",
    gpu_available_ ? "true" : "false (CPU fallback)",
    config.K, config.N, config.motion_model.c_str());
}

std::pair<Eigen::VectorXd, MPPIInfo> CudaMPPIControllerPlugin::computeControl(
  const Eigen::VectorXd& current_state,
  const Eigen::MatrixXd& reference_trajectory)
{
  if (!cuda_enabled_ || !cuda_engine_) {
    return MPPIControllerPlugin::computeControl(current_state, reference_trajectory);
  }

  int N = params_.N;
  int K = params_.K;
  int nu = dynamics_->model().controlDim();
  int nx = dynamics_->model().stateDim();

  // Step 1: 제어 시퀀스 시프트 (warm start)
  for (int t = 0; t < N - 1; ++t) {
    control_sequence_.row(t) = control_sequence_.row(t + 1);
  }
  control_sequence_.row(N - 1).setZero();

  // Step 2: 노이즈 샘플링
  auto noise = sampler_->sample(K, N, nu);

  // Step 3: 제어 교란
  std::vector<Eigen::MatrixXd> perturbed_controls;
  perturbed_controls.reserve(K);
  for (int k = 0; k < K; ++k) {
    perturbed_controls.push_back(control_sequence_ + noise[k]);
  }

  // Step 4: 참조 궤적 + 장애물 업데이트
  cuda_engine_->updateReference(reference_trajectory);
  // 장애물은 costmap에서 추출 (간소화: barrier_set_ 사용)
  // TODO: costmap lethal 셀 → point obstacle 변환 연동

  // Step 5: GPU(또는 CPU 폴백) 롤아웃 + 비용
  Eigen::VectorXd costs;
  cuda_engine_->execute(current_state, perturbed_controls, params_.dt, costs);

  // Step 6: 가중치 계산
  double current_lambda = params_.lambda;
  if (params_.adaptive_temperature && adaptive_temp_) {
    Eigen::VectorXd temp_weights = weight_computation_->compute(costs, current_lambda);
    double ess = computeESS(temp_weights);
    current_lambda = adaptive_temp_->update(ess, K);
  }
  Eigen::VectorXd weights = weight_computation_->compute(costs, current_lambda);

  // Step 7: 가중 노이즈 업데이트
  Eigen::MatrixXd weighted_noise = Eigen::MatrixXd::Zero(N, nu);
  for (int k = 0; k < K; ++k) {
    weighted_noise += weights(k) * noise[k];
  }
  control_sequence_ += weighted_noise;
  control_sequence_ = dynamics_->clipControls(control_sequence_);

  // Step 8: 최적 제어 추출
  Eigen::VectorXd u_opt = control_sequence_.row(0).transpose();

  // Step 9: MPPIInfo 구성
  MPPIInfo info;
  if (cuda_download_trajectories_) {
    cuda_engine_->downloadAllTrajectories(info.sample_trajectories);
  }
  info.sample_weights = weights;
  info.temperature = (params_.adaptive_temperature && adaptive_temp_)
                       ? adaptive_temp_->getLambda()
                       : params_.lambda;
  info.ess = computeESS(weights);
  info.costs = costs;

  // 가중 평균 궤적 + 최적 궤적
  if (!info.sample_trajectories.empty()) {
    Eigen::MatrixXd weighted_traj = Eigen::MatrixXd::Zero(N + 1, nx);
    for (int k = 0; k < K; ++k) {
      weighted_traj += weights(k) * info.sample_trajectories[k];
    }
    info.weighted_avg_trajectory = weighted_traj;

    int best_idx;
    costs.minCoeff(&best_idx);
    info.best_trajectory = info.sample_trajectories[best_idx];
  } else {
    // 궤적 미다운로드 시: 최적 제어 시퀀스로 롤아웃
    std::vector<Eigen::MatrixXd> u_vec = {control_sequence_};
    auto traj = dynamics_->rolloutBatch(current_state, u_vec, params_.dt);
    info.weighted_avg_trajectory = traj[0];
    info.best_trajectory = traj[0];
  }

  return {u_opt, info};
}

}  // namespace mpc_controller_ros2
