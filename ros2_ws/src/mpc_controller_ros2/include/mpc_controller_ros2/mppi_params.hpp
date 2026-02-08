#ifndef MPC_CONTROLLER_ROS2__MPPI_PARAMS_HPP_
#define MPC_CONTROLLER_ROS2__MPPI_PARAMS_HPP_

#include <Eigen/Dense>

namespace mpc_controller_ros2
{

/**
 * @brief MPPI 컨트롤러 파라미터 구조체
 *
 * M2 고도화 기능 포함:
 * - Colored Noise (OU 프로세스)
 * - Adaptive Temperature (ESS 기반)
 * - Tube-MPPI (Body frame 피드백)
 */
struct MPPIParams
{
  // ============================================================================
  // 기본 MPPI 파라미터
  // ============================================================================

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

  // ============================================================================
  // Phase 1: Colored Noise 파라미터
  // ============================================================================

  bool colored_noise{false};   // Colored noise 활성화 여부
  double noise_beta{2.0};      // OU 프로세스 시간 상관도 (클수록 백색)

  // ============================================================================
  // Phase 2: Adaptive Temperature 파라미터
  // ============================================================================

  bool adaptive_temperature{false};  // ESS 기반 λ 자동 조정 활성화
  double target_ess_ratio{0.5};      // 목표 ESS 비율 (0~1)
  double adaptation_rate{0.1};       // 적응 속도
  double lambda_min{0.1};            // 최소 λ
  double lambda_max{100.0};          // 최대 λ

  // ============================================================================
  // Phase 3: Tube-MPPI 파라미터
  // ============================================================================

  bool tube_enabled{false};      // Tube-MPPI 활성화 여부
  double tube_width{0.5};        // Tube 폭 (m)

  // Ancillary controller 피드백 게인 (2x3 행렬)
  // [dv]   [k_forward   0          0      ] [e_forward]
  // [dω] = [0           k_lateral  k_angle] [e_lateral]
  //                                         [e_angle  ]
  double k_forward{0.8};   // 전진 방향 오차 게인
  double k_lateral{0.5};   // 측면 오차 게인
  double k_angle{1.0};     // 각도 오차 게인

  // ============================================================================
  // 시각화 파라미터
  // ============================================================================

  bool visualize_samples{true};           // 샘플 궤적 표시
  bool visualize_best{true};              // 최적 궤적 표시
  bool visualize_weighted_avg{true};      // 가중 평균 궤적 표시
  bool visualize_reference{true};         // 참조 궤적 표시
  bool visualize_text_info{true};         // 텍스트 정보 표시
  bool visualize_control_sequence{true};  // 제어 시퀀스 화살표 표시
  bool visualize_tube{true};              // Tube 경계 표시
  int max_visualized_samples{20};         // 최대 표시 샘플 수

  // ============================================================================
  // 생성자
  // ============================================================================

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

  /**
   * @brief 피드백 게인 매트릭스 생성 (2x3)
   */
  Eigen::Matrix<double, 2, 3> getFeedbackGainMatrix() const
  {
    Eigen::Matrix<double, 2, 3> K_fb;
    K_fb << k_forward, 0.0,       0.0,
            0.0,       k_lateral, k_angle;
    return K_fb;
  }
};

}  // namespace mpc_controller_ros2

#endif  // MPC_CONTROLLER_ROS2__MPPI_PARAMS_HPP_
