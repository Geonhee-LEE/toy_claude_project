#ifndef MPC_CONTROLLER_ROS2__MPPI_PARAMS_HPP_
#define MPC_CONTROLLER_ROS2__MPPI_PARAMS_HPP_

#include <Eigen/Dense>

namespace mpc_controller_ros2
{

/**
 * @brief MPPI 컨트롤러 파라미터 구조체
 */
struct MPPIParams
{
  // Prediction horizon
  int N{30};              // 예측 스텝 수
  double dt{0.1};         // 시간 간격 (초)

  // Sampling parameters
  int K{1024};            // 샘플 궤적 수
  double lambda{10.0};    // Temperature 파라미터

  // Noise parameters
  Eigen::Vector2d noise_sigma{0.5, 0.5};  // [v, omega] 노이즈 표준편차

  // Cost weights
  Eigen::Matrix3d Q;      // State tracking weight
  Eigen::Matrix3d Qf;     // Terminal state weight
  Eigen::Matrix2d R;      // Control effort weight
  Eigen::Matrix2d R_rate; // Control rate weight

  // Control limits
  double v_max{1.0};      // 최대 선속도 (m/s)
  double v_min{-0.5};     // 최소 선속도 (m/s)
  double omega_max{1.0};  // 최대 각속도 (rad/s)
  double omega_min{-1.0}; // 최소 각속도 (rad/s)

  // Obstacle avoidance
  double obstacle_weight{100.0};    // 장애물 회피 가중치
  double safety_distance{0.5};      // 안전 거리 (m)

  // Visualization
  bool visualize_samples{true};           // 샘플 궤적 표시
  bool visualize_best{true};              // 최적 궤적 표시
  bool visualize_weighted_avg{true};      // 가중 평균 궤적 표시
  bool visualize_reference{true};         // 참조 궤적 표시
  bool visualize_text_info{true};         // 텍스트 정보 표시
  bool visualize_control_sequence{true};  // 제어 시퀀스 화살표 표시
  int max_visualized_samples{20};         // 최대 표시 샘플 수

  // Constructor with default weights
  MPPIParams()
  {
    // State tracking weight: [x, y, theta]
    Q = Eigen::Matrix3d::Zero();
    Q(0, 0) = 10.0;  // x
    Q(1, 1) = 10.0;  // y
    Q(2, 2) = 1.0;   // theta

    // Terminal weight
    Qf = 2.0 * Q;

    // Control effort weight: [v, omega]
    R = Eigen::Matrix2d::Zero();
    R(0, 0) = 0.1;   // v
    R(1, 1) = 0.1;   // omega

    // Control rate weight
    R_rate = Eigen::Matrix2d::Zero();
    R_rate(0, 0) = 1.0;  // v rate
    R_rate(1, 1) = 1.0;  // omega rate
  }
};

}  // namespace mpc_controller_ros2

#endif  // MPC_CONTROLLER_ROS2__MPPI_PARAMS_HPP_
