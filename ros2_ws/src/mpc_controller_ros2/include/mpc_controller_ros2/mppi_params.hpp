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

  // Exploitation/Exploration 분할
  double exploration_ratio{0.0};  // 0.0=전부exploitation(현재동작), 0.1=10%탐색(권장)

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
  double vy_max{-1.0};    // 최대 횡방향 속도 (swerve), <0이면 v_max 사용 (하위호환)

  // Obstacle avoidance
  double obstacle_weight{100.0};    // 장애물 회피 가중치
  double safety_distance{0.5};      // 안전 거리 (m)

  // Forward preference
  double prefer_forward_weight{5.0};         // 전진 선호 가중치 (후진 페널티)
  double prefer_forward_linear_ratio{0.5};   // 선형 비용 비율 (0=이차만, 1=선형만, 0.5=혼합)
  double prefer_forward_velocity_incentive{0.0};  // 전진 인센티브 (저속 정지 회피)

  // Control smoothing (EMA 출력 필터)
  double control_smoothing_alpha{1.0};  // 0=이전유지, 1=즉시반영(필터OFF)

  // Savitzky-Golay Filter (EMA 대체)
  bool sg_filter_enabled{false};          // SG 필터 활성화 (true면 EMA 무시)
  int sg_half_window{3};                  // 과거/미래 윈도우 크기
  int sg_poly_order{3};                   // 다항식 차수

  // Information-Theoretic 정규화 (KL-divergence 기반 보수적 업데이트)
  double it_alpha{1.0};    // 1.0=비활성화, 0.975=보수적 (권장: swerve 0.975)

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
  double tube_nominal_reset_threshold{1.0};  // Nominal 리셋 편차 임계 (m)

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
  // Biased-MPPI (Trevisan & Alonso-Mora, RA-L 2024) 파라미터
  // ============================================================================
  bool biased_enabled{true};                   // ancillary 주입 활성화
  double bias_ratio{0.1};                      // J/K 비율 (각 ancillary당)
  bool biased_braking{true};                   // Braking controller
  bool biased_goto_goal{true};                 // GoToGoal controller
  bool biased_path_following{true};            // PathFollowing controller
  bool biased_previous_solution{true};         // PreviousSolution controller
  double biased_goto_goal_gain{1.0};           // GoToGoal P-gain
  double biased_path_following_gain{1.0};      // PathFollowing P-gain

  // ============================================================================
  // DIAL-MPPI (Xue et al., ICRA 2025) 파라미터
  // Diffusion-Inspired Annealing: 다중 스텝 어닐링으로 정밀한 최적 제어 탐색
  // ============================================================================
  bool dial_enabled{true};                       // 어닐링 활성화
  int dial_n_diffuse{5};                         // 어닐링 반복 횟수 N
  double dial_beta1{0.8};                        // 반복 감쇠 계수 β₁
  double dial_beta2{0.5};                        // 호라이즌 감쇠 계수 β₂
  double dial_min_noise{0.01};                   // 최소 노이즈 하한 (수치 안정성)

  // Shield-DIAL (CBF 통합)
  bool dial_shield_enabled{false};               // CBF 안전 필터 활성화

  // Adaptive-DIAL (적응형 반복)
  bool dial_adaptive_enabled{false};             // 적응형 N_diffuse 활성화
  double dial_adaptive_cost_tol{0.01};           // 비용 개선 임계값 (상대)
  int dial_adaptive_min_iter{2};                 // 최소 반복 횟수
  int dial_adaptive_max_iter{10};                // 최대 반복 횟수

  // ============================================================================
  // Residual Dynamics (학습 기반 잔차 동역학)
  // ============================================================================
  bool residual_enabled{false};                    // 잔차 MLP 활성화
  std::string residual_weights_path{""};           // MLP 바이너리 파일 경로
  double residual_alpha{1.0};                      // 잔차 블렌딩 계수 (0=공칭, 1=전체)

  // ============================================================================
  // Ensemble Dynamics (앙상블 MLP 불확실성 추정)
  // ============================================================================
  bool ensemble_enabled{false};                    // 앙상블 활성화
  std::string ensemble_weights_dir{""};            // 앙상블 가중치 디렉토리 (model_0.bin ~ model_{M-1}.bin)
  int ensemble_size{5};                            // 앙상블 MLP 개수 (M)
  double ensemble_alpha{1.0};                      // 앙상블 잔차 블렌딩 계수
  double uncertainty_cost_weight{0.0};             // 분산 비용 가중치 (0=비활성화)

  // ============================================================================
  // C3BF (Collision Cone CBF, 속도 인식 안전성)
  // ============================================================================
  bool c3bf_enabled{false};                        // C3BF 활성화
  double c3bf_alpha_safe{0.7854};                  // 안전 콘 반각 (rad, π/4)
  double c3bf_cost_weight{500.0};                  // C3BF 비용 가중치

  // ============================================================================
  // Adaptive Shield (거리/속도 적응형 CBF)
  // ============================================================================
  bool adaptive_shield_enabled{false};             // 적응형 Shield 활성화
  double adaptive_shield_alpha_min{0.1};           // 최소 alpha
  double adaptive_shield_alpha_max{1.0};           // 최대 alpha
  double adaptive_shield_k_d{1.0};                 // 거리 감쇠 계수
  double adaptive_shield_k_v{0.5};                 // 속도 증폭 계수

  // ============================================================================
  // Horizon-Weighted CBF (시간 할인)
  // ============================================================================
  double cbf_horizon_discount{1.0};                // 1.0=비활성화, <1.0=먼 미래 비용 감쇄

  // ============================================================================
  // Online Data Buffer (런타임 데이터 수집)
  // ============================================================================
  bool online_data_enabled{false};                 // 데이터 수집 활성화
  int online_data_capacity{10000};                 // 링 버퍼 크기
  std::string online_data_export_path{"/tmp/mppi_online_data.csv"};  // 내보내기 경로

  // ============================================================================
  // Online Learning (런타임 모델 리로드)
  // ============================================================================
  bool model_reload_enabled{false};                // 모델 리로드 체크 활성화
  double model_reload_interval_sec{30.0};          // 모델 리로드 체크 간격 (초)

  // ============================================================================
  // Safety Enhancement 파라미터
  // ============================================================================

  // BarrierRateCost (BR-MPPI)
  double barrier_rate_cost_weight{0.0};  // 0=비활성화 (하위호환)

  // ConformalPredictor (동적 안전 마진)
  bool conformal_enabled{false};                   // ACP 마진 활성화
  double conformal_coverage{0.95};                 // 목표 커버리지 확률
  int conformal_window_size{100};                  // 슬라이딩 윈도우
  double conformal_initial_margin{0.3};            // 초기 마진 (m)
  double conformal_min_margin{0.05};               // 최소 마진 (m)
  double conformal_max_margin{1.0};                // 최대 마진 (m)
  double conformal_decay_rate{0.99};               // ACP 가중치 감쇄율

  // Shield-MPPI (per-step CBF 투영)
  int shield_cbf_stride{1};                        // CBF 투영 간격 (1=매 스텝, 2=매 2번째)
  int shield_max_iterations{10};                   // 투영 최대 반복

  // ============================================================================
  // CLF-CBF-QP 통합 안전 필터 파라미터
  // Ames et al. (2019): CLF 수렴 + CBF 안전 보장, slack으로 안전 우선
  // ============================================================================
  bool clf_cbf_enabled{false};                     // CLF-CBF-QP 활성화
  double clf_decay_rate{1.0};                      // CLF decay rate c (V̇ + c·V ≤ δ)
  double clf_slack_penalty{100.0};                 // slack 페널티 p (min p·δ²)
  double clf_P_scale{1.0};                         // P = scale · Q (Lyapunov 가중치)

  // ============================================================================
  // CBF 합성 파라미터 (다중 CBF → 단일 합성 CBF)
  // ============================================================================
  bool cbf_composition_enabled{false};       // 합성 CBF 활성화
  int cbf_composition_method{1};             // 0=MIN, 1=SMOOTH_MIN, 2=LOG_SUM_EXP, 3=PRODUCT
  double cbf_composition_alpha{10.0};        // smooth-min 파라미터 (클수록 min에 가까움)

  // ============================================================================
  // Predictive Safety Filter 파라미터 (N-step CBF 투영)
  // ShieldMPPI의 단일 스텝 투영을 전체 horizon으로 확장
  // ============================================================================
  bool predictive_safety_enabled{false};     // 예측 안전 필터 활성화
  int predictive_safety_horizon{0};          // 투영 horizon (0=전체 N)
  double predictive_safety_decay{1.0};       // gamma 시간 감쇠 (1.0=균일)
  int predictive_safety_max_iterations{10};  // 스텝당 최대 투영 반복

  // ============================================================================
  // Covariance Steering MPPI (CS-MPPI) 파라미터
  // CoVO-MPC (CoRL 2023): 동역학 Jacobian B_t 감도 기반 노이즈 공분산 적응
  // ============================================================================
  bool cs_enabled{true};              // CS 공분산 적응 활성화
  double cs_scale_min{0.1};           // 최소 노이즈 스케일 팩터
  double cs_scale_max{3.0};           // 최대 노이즈 스케일 팩터
  bool cs_feedback_enabled{false};    // per-step 피드백 보정 (V2 확장)
  double cs_feedback_gain{0.5};       // 피드백 게인

  // ============================================================================
  // pi-MPPI (Projection MPPI, Andrejev et al. RA-L 2025) 파라미터
  // ADMM QP 투영 필터: 제어 크기/변화율/가속도 hard bounds 보장
  // ============================================================================
  bool pi_enabled{true};                  // 투영 필터 활성화
  int pi_admm_iterations{10};             // ADMM 반복 횟수
  double pi_admm_rho{1.0};               // ADMM 페널티 파라미터
  int pi_derivative_order{2};             // 1=rate, 2=rate+accel
  double pi_rate_max_v{2.0};             // m/s² (선속도 변화율 상한)
  double pi_rate_max_omega{3.0};         // rad/s² (각속도 변화율 상한)
  double pi_rate_max_vy{2.0};            // m/s² (swerve 횡방향 변화율)
  double pi_accel_max_v{5.0};            // m/s³ (jerk 상한)
  double pi_accel_max_omega{8.0};        // rad/s³ (각가속도 jerk 상한)
  double pi_accel_max_vy{5.0};           // m/s³ (swerve 횡방향 jerk)

  // ============================================================================
  // iLQR Warm-Start 파라미터
  // ============================================================================
  bool ilqr_enabled{false};                          // iLQR warm-start 활성화
  int ilqr_max_iterations{2};                        // iLQR 반복 횟수 (1-2회 충분)
  double ilqr_regularization{1e-6};                  // Q_uu 정규화 (rho)
  int ilqr_line_search_steps{4};                     // line search alpha 후보 수
  double ilqr_cost_tolerance{1e-4};                  // 수렴 판정 임계값

  // ============================================================================
  // Hybrid Swerve MPPI (MPPI-H) 파라미터
  // IROS 2024, arXiv:2409.08648: Low-D↔4D 샘플링 공간 실시간 전환
  // ============================================================================
  bool hybrid_enabled{true};                    // 하이브리드 모드 활성화
  double hybrid_cdist_threshold{0.3};           // 추적 거리 임계값 (m)
  double hybrid_cangle_threshold{0.3};          // 추적 각도 임계값 (rad)
  int hybrid_hysteresis_count{3};               // 모드 전환 히스테리시스 (cycles)

  // 4D 바퀴 기하 파라미터
  double hybrid_lf{0.25};                       // 전방 바퀴 종방향 거리 (m)
  double hybrid_lr{0.25};                       // 후방 바퀴 종방향 거리 (m)
  double hybrid_dl{0.22};                       // 좌측 바퀴 횡방향 거리 (m)
  double hybrid_dr{0.22};                       // 우측 바퀴 횡방향 거리 (m)
  double hybrid_v_wheel_max{2.0};               // 최대 바퀴 속도 (m/s)
  double hybrid_delta_max{1.5708};              // 최대 바퀴 조향각 (rad, π/2)

  // 4D 모드 노이즈 σ
  double hybrid_noise_sigma_vfl{0.5};
  double hybrid_noise_sigma_vrr{0.5};
  double hybrid_noise_sigma_dfl{0.3};
  double hybrid_noise_sigma_drr{0.3};

  // 4D 모드 제어 비용 R
  double hybrid_R_vfl{0.1};
  double hybrid_R_vrr{0.1};
  double hybrid_R_dfl{0.3};
  double hybrid_R_drr{0.3};

  // ============================================================================
  // CBF (Control Barrier Function) 안전성 보장 파라미터
  // ============================================================================
  // Non-Coaxial Swerve / Ackermann 공통 파라미터
  double max_steering_rate{2.0};            // 최대 스티어링 각속도 (rad/s)
  double max_steering_angle{M_PI / 2.0};    // 최대 스티어링 각도 (rad)
  double wheelbase{0.5};                    // Ackermann 축간 거리 (m)

  bool cbf_enabled{false};                 // CBF 활성화 여부
  double cbf_gamma{1.0};                   // CBF class-K 함수 계수
  double cbf_safety_margin{0.3};           // 추가 안전 마진 (m)
  double cbf_robot_radius{0.2};            // 로봇 반지름 (m)
  double cbf_activation_distance{3.0};     // 장애물 활성화 거리 (m)
  double cbf_cost_weight{500.0};           // CBFCost soft cost 가중치
  bool cbf_use_safety_filter{true};        // Post-hoc QP safety filter 사용

  // ============================================================================
  // Dynamic Obstacle Tracker 파라미터
  // ============================================================================
  bool dynamic_obstacle_tracking_enabled{false};    // 동적 장애물 추적 활성화
  double obstacle_cluster_distance{0.15};            // 클러스터링 거리 (m)
  int obstacle_min_cluster_size{3};                  // 최소 클러스터 크기
  double obstacle_velocity_ema_alpha{0.3};           // 속도 EMA 계수 (0~1)
  double obstacle_max_association_distance{0.5};     // 최대 매칭 거리 (m)
  double obstacle_track_timeout{2.0};                // 트랙 타임아웃 (초)

  // ============================================================================
  // 성능 최적화 파라미터
  // ============================================================================
  int num_threads{0};              // OpenMP 스레드 수 (0=auto, OMP_NUM_THREADS 사용)
  int costmap_eval_stride{1};      // Costmap 평가 간격 (1=전부, 2=매 2번째, 3=매 3번째)

  // ============================================================================
  // Motion Model 선택
  // ============================================================================
  std::string motion_model{"diff_drive"};  // "diff_drive", "swerve", "non_coaxial_swerve", "ackermann"

  // ============================================================================
  // Costmap 기반 장애물 비용 파라미터
  // ============================================================================
  bool use_costmap_cost{true};          // CostmapObstacleCost 사용
  double costmap_lethal_cost{1000.0};   // LETHAL 셀 비용
  double costmap_critical_cost{100.0};  // INSCRIBED 셀 비용
  double lookahead_dist{0.0};           // 0 = auto (v_max * N * dt)
  double min_lookahead{0.5};            // 최소 lookahead 거리 (goal 근처 수렴 보장)
  double goal_slowdown_dist{1.0};       // 목표 근처 감속 시작 거리 (m)
  int ref_theta_smooth_window{0};       // Reference theta 스무딩 윈도우 (0=OFF, 홀수 권장: 3,5,7)

  // Velocity Tracking Cost (경로 방향 속도 추적)
  double velocity_tracking_weight{0.0};  // 0=비활성화 (하위호환)
  double reference_velocity{1.0};        // 목표 경로 방향 속도 (m/s)

  // ============================================================================
  // Collision Debug Visualization (기본 OFF — 성능 오버헤드 0)
  // ============================================================================
  bool debug_collision_viz{false};          // 마스터 스위치
  bool debug_cost_breakdown{true};          // 비용 분해 텍스트
  bool debug_collision_points{true};        // 충돌 지점 마커
  bool debug_safety_footprint{true};        // 안전 영역 원
  bool debug_cost_heatmap{true};            // 궤적 비용 히트맵
  double debug_footprint_radius{0.3};       // 로봇 반지름 (m)
  int debug_heatmap_stride{3};              // 히트맵 점 간격

  // Visualization
  bool visualize_samples{true};           // 샘플 궤적 표시
  bool visualize_best{true};              // 최적 궤적 표시
  bool visualize_weighted_avg{true};      // 가중 평균 궤적 표시
  bool visualize_reference{true};         // 참조 궤적 표시
  bool visualize_text_info{true};         // 텍스트 정보 표시
  bool visualize_control_sequence{true};  // 제어 시퀀스 화살표 표시
  bool visualize_tube{true};              // Tube 경계 표시
  bool visualize_cbf{false};              // CBF barrier/correction 표시
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
