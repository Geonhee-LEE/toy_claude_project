#ifndef MPC_CONTROLLER_ROS2__MPPI_PARAMS_HPP_
#define MPC_CONTROLLER_ROS2__MPPI_PARAMS_HPP_

#include <Eigen/Dense>

namespace mpc_controller_ros2
{

/**
 * @brief MPPI 컨트롤러 파라미터 구조체
 *
 * 동적 차원 지원: MatrixXd/VectorXd로 nx, nu에 따라 크기가 결정됩니다.
 * 기본 생성자는 DiffDrive (nx=3, nu=2) 기본값을 유지합니다.
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

  // Noise parameters (동적 차원)
  Eigen::VectorXd noise_sigma;  // (nu,) 노이즈 표준편차

  // Cost weights (동적 차원)
  Eigen::MatrixXd Q;      // (nx x nx) State tracking weight
  Eigen::MatrixXd Qf;     // (nx x nx) Terminal state weight
  Eigen::MatrixXd R;      // (nu x nu) Control effort weight
  Eigen::MatrixXd R_rate;  // (nu x nu) Control rate weight

  // Control limits
  double v_max{1.0};      // 최대 선속도 (m/s)
  double v_min{0.0};      // 최소 선속도 (m/s), 0.0=후진 차단
  double omega_max{1.0};  // 최대 각속도 (rad/s)
  double omega_min{-1.0}; // 최소 각속도 (rad/s)

  // Obstacle avoidance
  double obstacle_weight{100.0};    // 장애물 회피 가중치
  double safety_distance{0.5};      // 안전 거리 (m)

  // Forward preference
  double prefer_forward_weight{5.0};         // 전진 선호 가중치 (후진 페널티)
  double prefer_forward_linear_ratio{0.5};   // 선형 비용 비율 (0=이차만, 1=선형만, 0.5=혼합)

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

  // Ancillary controller 피드백 게인 (nu x nx 행렬)
  // DiffDrive 기본:
  // [dv]   [k_forward   0          0      ] [e_forward]
  // [dω] = [0           k_lateral  k_angle] [e_lateral]
  //                                         [e_angle  ]
  double k_forward{0.8};   // 전진 방향 오차 게인
  double k_lateral{0.5};   // 측면 오차 게인
  double k_angle{1.0};     // 각도 오차 게인

  // ============================================================================
  // SOTA 변형 파라미터 (Tsallis, Risk-Aware)
  // ============================================================================

  double tsallis_q{1.5};     // Tsallis q 파라미터 (q>1 heavy-tail, q<1 light-tail, q=1 Vanilla)
  double cvar_alpha{0.5};    // CVaR alpha 파라미터 (1.0=risk-neutral, <1=risk-averse)

  // SVMPC (Stein Variational MPC) 파라미터
  int svgd_num_iterations{0};        // SVGD 반복 횟수 (0=Vanilla 동등, 권장: 1~5)
  double svgd_step_size{0.1};        // SVGD update step size
  double svgd_bandwidth{-1.0};       // RBF bandwidth (-1=median heuristic)

  // ============================================================================
  // M3.5 Smooth/Spline/SVG-MPPI 파라미터
  // ============================================================================

  // Smooth-MPPI (Kim et al. 2021)
  double smooth_R_jerk_v{0.1};           // jerk weight (v 방향)
  double smooth_R_jerk_omega{0.1};       // jerk weight (omega 방향)
  double smooth_action_cost_weight{1.0}; // jerk cost 전체 가중치

  // Spline-MPPI (ICRA 2024)
  int spline_num_knots{12};             // B-spline 제어점 수 (P << N)
  int spline_degree{3};                 // B-spline 차수 (cubic)
  bool spline_auto_knot_sigma{true};    // basis 감쇠 자동 보정 (σ × amp_factor)

  // SVG-MPPI (Kondo et al., ICRA 2024)
  int svg_num_guide_particles{10};      // guide particle 수 (G << K)
  int svg_guide_iterations{3};          // guide SVGD 반복 횟수
  double svg_guide_step_size{0.1};      // guide SVGD step size
  double svg_resample_std{0.3};         // follower 리샘플링 표준편차

  // ============================================================================
  // Motion Model 선택
  // ============================================================================
  std::string motion_model{"diff_drive"};  // "diff_drive", "swerve", "non_coaxial_swerve"

  // ============================================================================
  // Costmap 기반 장애물 비용 파라미터
  // ============================================================================
  bool use_costmap_cost{true};          // CostmapObstacleCost 사용
  double costmap_lethal_cost{1000.0};   // LETHAL 셀 비용
  double costmap_critical_cost{100.0};  // INSCRIBED 셀 비용
  double lookahead_dist{0.0};           // 0 = auto (v_max * N * dt)
  double min_lookahead{0.5};            // 동적 lookahead 최소값 (m)
  double goal_slowdown_dist{1.0};       // 목표 접근 감속 시작 거리 (m)

  // Visualization
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

  /** @brief 기본 생성자 — DiffDrive (nx=3, nu=2) 기본값 */
  MPPIParams()
  {
    // Noise sigma: [v, omega]
    noise_sigma = Eigen::Vector2d(0.5, 0.5);

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
   * @brief 피드백 게인 매트릭스 생성 (nu x nx)
   * DiffDrive 기본: 2x3
   */
  Eigen::MatrixXd getFeedbackGainMatrix() const
  {
    int nu = R.rows();
    int nx = Q.rows();
    Eigen::MatrixXd K_fb = Eigen::MatrixXd::Zero(nu, nx);
    // 기본 게인 매핑 (DiffDrive 호환)
    if (nu >= 2 && nx >= 3) {
      K_fb(0, 0) = k_forward;
      K_fb(1, 1) = k_lateral;
      K_fb(1, 2) = k_angle;
    }
    return K_fb;
  }
};

}  // namespace mpc_controller_ros2

#endif  // MPC_CONTROLLER_ROS2__MPPI_PARAMS_HPP_
