#ifndef MPC_CONTROLLER_ROS2__CUDA_ROLLOUT_ENGINE_HPP_
#define MPC_CONTROLLER_ROS2__CUDA_ROLLOUT_ENGINE_HPP_

#include <Eigen/Dense>
#include <vector>
#include <string>

namespace mpc_controller_ros2
{

/**
 * @brief GPU CUDA 롤아웃 엔진 (CPU 폴백 포함)
 *
 * CUDA 사용 가능 시 GPU에서 K개 샘플 병렬 롤아웃 + 비용 계산,
 * CUDA 미사용 시 CPU 참조 구현으로 동일 결과 보장.
 *
 * Stub 패턴:
 *   cuda_rollout_engine.hpp      — 순수 C++ 헤더 (API)
 *   cuda_rollout_engine_cpu.cpp  — CPU 참조 구현 (항상 컴파일)
 *   cuda_rollout_engine.cu       — CUDA 구현 (CUDA 빌드 시만)
 */
class CudaRolloutEngine
{
public:
  struct Config
  {
    int K{512};                        // 샘플 수
    int N{30};                         // 예측 horizon 스텝
    int nx{3};                         // 상태 차원
    int nu{2};                         // 제어 차원
    int max_obstacles{128};            // 최대 장애물 수
    std::string motion_model{"diff_drive"};  // "diff_drive" or "swerve"
  };

  explicit CudaRolloutEngine(const Config& config);
  ~CudaRolloutEngine();

  /**
   * @brief 비용 행렬 및 제어 제한 초기화
   */
  void initialize(
    const Eigen::MatrixXd& Q, const Eigen::MatrixXd& Qf,
    const Eigen::MatrixXd& R, const Eigen::MatrixXd& R_rate,
    double v_min, double v_max, double omega_min, double omega_max,
    double vy_max = -1.0);

  /**
   * @brief 장애물 업데이트 (Vector3d: x, y, radius)
   */
  void updateObstacles(const std::vector<Eigen::Vector3d>& obstacles,
                       double safety_distance, double obstacle_weight);

  /**
   * @brief 참조 궤적 업데이트 (N+1 x nx 또는 더 긴 궤적)
   */
  void updateReference(const Eigen::MatrixXd& reference);

  /**
   * @brief 롤아웃 + 비용 계산 실행 (GPU 또는 CPU 폴백)
   * @param x0 초기 상태 (nx,)
   * @param perturbed_controls K개 제어 시퀀스 (각 N x nu)
   * @param dt 시간 간격
   * @param costs_out 비용 벡터 (K,)
   */
  void execute(
    const Eigen::VectorXd& x0,
    const std::vector<Eigen::MatrixXd>& perturbed_controls,
    double dt,
    Eigen::VectorXd& costs_out);

  /**
   * @brief 모든 궤적 다운로드 (시각화용)
   * @param trajectories_out K개 궤적 (각 N+1 x nx)
   */
  void downloadAllTrajectories(std::vector<Eigen::MatrixXd>& trajectories_out) const;

  /**
   * @brief GPU (CUDA) 사용 가능 여부
   * CPU 폴백 빌드: false, CUDA 빌드: true (GPU 감지 시)
   */
  static bool isAvailable();

  const Config& config() const { return config_; }

private:
  Config config_;

  // CPU 참조 구현 버퍼
  std::vector<Eigen::MatrixXd> trajectories_;  // K x (N+1, nx)
  Eigen::MatrixXd Q_, Qf_, R_, R_rate_;
  Eigen::MatrixXd reference_;
  std::vector<Eigen::Vector3d> obstacles_;
  double safety_distance_{0.5};
  double obstacle_weight_{100.0};
  double v_min_{0.0}, v_max_{1.0}, omega_min_{-1.0}, omega_max_{1.0};
  double vy_max_{-1.0};

  // CPU 롤아웃 + 비용 계산
  void executeCPU(
    const Eigen::VectorXd& x0,
    const std::vector<Eigen::MatrixXd>& perturbed_controls,
    double dt,
    Eigen::VectorXd& costs_out);

  // 단일 샘플 RK4 롤아웃
  Eigen::MatrixXd rolloutSingle(
    const Eigen::VectorXd& x0,
    const Eigen::MatrixXd& controls,
    double dt) const;

  // 단일 샘플 비용 계산
  double computeSingleCost(
    const Eigen::MatrixXd& trajectory,
    const Eigen::MatrixXd& controls) const;

  // 동역학 함수 (모델별 분기)
  Eigen::VectorXd dynamics(
    const Eigen::VectorXd& state,
    const Eigen::VectorXd& control) const;
};

}  // namespace mpc_controller_ros2

#endif  // MPC_CONTROLLER_ROS2__CUDA_ROLLOUT_ENGINE_HPP_
