#include <gtest/gtest.h>
#include <Eigen/Dense>
#include <cmath>
#include <memory>

#include "mpc_controller_ros2/hybrid_swerve_mppi_controller_plugin.hpp"
#include "mpc_controller_ros2/wheel_level_4d_model.hpp"
#include "mpc_controller_ros2/mppi_params.hpp"
#include "mpc_controller_ros2/batch_dynamics_wrapper.hpp"
#include "mpc_controller_ros2/cost_functions.hpp"
#include "mpc_controller_ros2/sampling.hpp"
#include "mpc_controller_ros2/weight_computation.hpp"
#include "mpc_controller_ros2/motion_model_factory.hpp"
#include "mpc_controller_ros2/swerve_drive_model.hpp"
#include "mpc_controller_ros2/non_coaxial_swerve_model.hpp"

using namespace mpc_controller_ros2;

// =============================================================================
// Accessor: HybridSwerveMPPIControllerPlugin의 protected 멤버 노출
// =============================================================================

class HybridMPPIAccessor : public HybridSwerveMPPIControllerPlugin
{
public:
  using HybridSwerveMPPIControllerPlugin::computeControl;
  using HybridSwerveMPPIControllerPlugin::computeControl4D;
  using HybridSwerveMPPIControllerPlugin::determineMode;
  using HybridSwerveMPPIControllerPlugin::convertLowTo4D;
  using HybridSwerveMPPIControllerPlugin::convert4DToLow;
  using HybridSwerveMPPIControllerPlugin::current_mode_;
  using HybridSwerveMPPIControllerPlugin::mode_switch_counter_;
  using HybridSwerveMPPIControllerPlugin::is_non_coaxial_;
  using HybridSwerveMPPIControllerPlugin::tracked_delta_;
  using HybridSwerveMPPIControllerPlugin::last_ctrl_4d_;
  using HybridSwerveMPPIControllerPlugin::dynamics_4d_;
  using HybridSwerveMPPIControllerPlugin::sampler_4d_;
  using HybridSwerveMPPIControllerPlugin::cost_function_4d_;
  using HybridSwerveMPPIControllerPlugin::control_seq_4d_;
  using HybridSwerveMPPIControllerPlugin::noise_buf_4d_;
  using HybridSwerveMPPIControllerPlugin::perturbed_buf_4d_;
  using HybridSwerveMPPIControllerPlugin::traj_buf_4d_;
  using MPPIControllerPlugin::dynamics_;
  using MPPIControllerPlugin::sampler_;
  using MPPIControllerPlugin::params_;
  using MPPIControllerPlugin::control_sequence_;
  using MPPIControllerPlugin::noise_buffer_;
  using MPPIControllerPlugin::perturbed_buffer_;
  using MPPIControllerPlugin::trajectory_buffer_;
  using MPPIControllerPlugin::weight_computation_;
  using MPPIControllerPlugin::cost_function_;
  using MPPIControllerPlugin::adaptive_temp_;
  using MPPIControllerPlugin::node_;

  void initForTest(const MPPIParams& p, const std::string& model_type)
  {
    params_ = p;
    is_non_coaxial_ = (model_type == "non_coaxial_swerve");

    int N = p.N;
    int K = p.K;

    // Low-D model
    auto model = MotionModelFactory::create(model_type, p);
    dynamics_ = std::make_unique<BatchDynamicsWrapper>(p, std::move(model));
    int nu_low = dynamics_->model().controlDim();
    int nx_low = dynamics_->model().stateDim();

    sampler_ = std::make_unique<GaussianSampler>(p.noise_sigma, 42);
    control_sequence_ = Eigen::MatrixXd::Zero(N, nu_low);
    noise_buffer_.resize(K, Eigen::MatrixXd::Zero(N, nu_low));
    perturbed_buffer_.resize(K, Eigen::MatrixXd::Zero(N, nu_low));
    trajectory_buffer_.resize(K, Eigen::MatrixXd::Zero(N + 1, nx_low));
    weight_computation_ = std::make_unique<VanillaMPPIWeights>();
    cost_function_ = std::make_unique<CompositeMPPICost>();
    cost_function_->addCost(std::make_unique<StateTrackingCost>(p.Q));
    cost_function_->addCost(std::make_unique<TerminalCost>(p.Qf));
    cost_function_->addCost(std::make_unique<ControlEffortCost>(p.R));

    // 4D model
    MPPIParams p4d = p;
    p4d.motion_model = "wheel_level_4d";
    auto model_4d = MotionModelFactory::create("wheel_level_4d", p4d);
    dynamics_4d_ = std::make_unique<BatchDynamicsWrapper>(p4d, std::move(model_4d));

    Eigen::VectorXd noise_sigma_4d(4);
    noise_sigma_4d(0) = p.hybrid_noise_sigma_vfl;
    noise_sigma_4d(1) = p.hybrid_noise_sigma_vrr;
    noise_sigma_4d(2) = p.hybrid_noise_sigma_dfl;
    noise_sigma_4d(3) = p.hybrid_noise_sigma_drr;
    sampler_4d_ = std::make_unique<GaussianSampler>(noise_sigma_4d, 42);

    int nx_4d = 3;
    Eigen::MatrixXd Q_4d = p.Q.topLeftCorner(nx_4d, nx_4d);
    Eigen::MatrixXd Qf_4d = p.Qf.topLeftCorner(nx_4d, nx_4d);
    Eigen::MatrixXd R_4d = Eigen::MatrixXd::Zero(4, 4);
    R_4d(0, 0) = p.hybrid_R_vfl;
    R_4d(1, 1) = p.hybrid_R_vrr;
    R_4d(2, 2) = p.hybrid_R_dfl;
    R_4d(3, 3) = p.hybrid_R_drr;

    cost_function_4d_ = std::make_unique<CompositeMPPICost>();
    cost_function_4d_->addCost(std::make_unique<StateTrackingCost>(Q_4d));
    cost_function_4d_->addCost(std::make_unique<TerminalCost>(Qf_4d));
    cost_function_4d_->addCost(std::make_unique<ControlEffortCost>(R_4d));

    control_seq_4d_ = Eigen::MatrixXd::Zero(N, 4);
    noise_buf_4d_.resize(K, Eigen::MatrixXd::Zero(N, 4));
    perturbed_buf_4d_.resize(K, Eigen::MatrixXd::Zero(N, 4));
    traj_buf_4d_.resize(K, Eigen::MatrixXd::Zero(N + 1, nx_4d));

    current_mode_ = Mode::LOW_D;
    mode_switch_counter_ = 0;
    tracked_delta_ = 0.0;
    last_ctrl_4d_ = Eigen::Vector4d::Zero();
  }
};

// =============================================================================
// Fixture
// =============================================================================

struct HybridTestFixture
{
  MPPIParams params;
  HybridMPPIAccessor accessor;

  void init(const std::string& model_type = "swerve", int K = 128, int N = 15)
  {
    params = MPPIParams();
    params.N = N;
    params.dt = 0.1;
    params.K = K;
    params.lambda = 10.0;
    params.motion_model = model_type;
    params.hybrid_enabled = true;
    params.hybrid_cdist_threshold = 0.3;
    params.hybrid_cangle_threshold = 0.3;
    params.hybrid_hysteresis_count = 3;

    bool is_nc = (model_type == "non_coaxial_swerve");
    int nu = is_nc ? 3 : 3;  // swerve=3, non_coaxial=3
    int nx = is_nc ? 4 : 3;

    params.noise_sigma = Eigen::VectorXd::Constant(nu, 0.5);
    params.Q = Eigen::MatrixXd::Identity(nx, nx) * 10.0;
    params.Q(2, 2) = 1.0;
    if (nx >= 4) params.Q(3, 3) = 0.5;
    params.Qf = params.Q * 2.0;
    params.R = Eigen::MatrixXd::Identity(nu, nu) * 0.1;

    accessor.initForTest(params, model_type);
  }
};

// =============================================================================
// [A] WheelLevel4DModel 기본 (6)
// =============================================================================

TEST(HybridMPPI, A1_Dimensions)
{
  WheelLevel4DModel model(0.25, 0.25, 0.22, 0.22, 2.0, M_PI / 2);
  EXPECT_EQ(model.stateDim(), 3);
  EXPECT_EQ(model.controlDim(), 4);
  EXPECT_TRUE(model.isHolonomic());
  EXPECT_EQ(model.name(), "wheel_level_4d");
  EXPECT_EQ(model.angleIndices(), std::vector<int>({2}));
}

TEST(HybridMPPI, A2_FK_StraightForward)
{
  // V_fl=V_rr=1, δ=0 → Vx=1, Vy=0, ω≈0
  Eigen::Vector4d u4d(1.0, 1.0, 0.0, 0.0);
  Eigen::Vector3d body = WheelLevel4DModel::forwardKinematics(u4d, 0.22, 0.22);
  EXPECT_NEAR(body(0), 1.0, 1e-10);  // Vx
  EXPECT_NEAR(body(1), 0.0, 1e-10);  // Vy
  EXPECT_NEAR(body(2), 0.0, 1e-10);  // ω
}

TEST(HybridMPPI, A3_FK_PureLateral)
{
  // V_fl=V_rr=1, δ=π/2 → Vx≈0, Vy=1
  Eigen::Vector4d u4d(1.0, 1.0, M_PI / 2, M_PI / 2);
  Eigen::Vector3d body = WheelLevel4DModel::forwardKinematics(u4d, 0.22, 0.22);
  EXPECT_NEAR(body(0), 0.0, 1e-10);  // Vx ≈ 0
  EXPECT_NEAR(body(1), 1.0, 1e-10);  // Vy = 1
  EXPECT_NEAR(body(2), 0.0, 1e-10);  // ω ≈ 0
}

TEST(HybridMPPI, A4_FK_PureRotation)
{
  // V_fl=-1, V_rr=1, δ=0 → Vx=0, Vy=0, ω>0
  Eigen::Vector4d u4d(-1.0, 1.0, 0.0, 0.0);
  Eigen::Vector3d body = WheelLevel4DModel::forwardKinematics(u4d, 0.22, 0.22);
  EXPECT_NEAR(body(0), 0.0, 1e-10);  // Vx = 0
  EXPECT_NEAR(body(1), 0.0, 1e-10);  // Vy = 0
  EXPECT_GT(body(2), 0.0);           // ω > 0
  // ω = (1·1 - (-1)·1) / (0.22+0.22) = 2 / 0.44 ≈ 4.545
  EXPECT_NEAR(body(2), 2.0 / 0.44, 1e-6);
}

TEST(HybridMPPI, A5_FKIK_Roundtrip)
{
  // IK(FK(u4d)) ≈ u4d (양의 속도에서)
  double lf = 0.25, lr = 0.25, dl = 0.22, dr = 0.22;
  Eigen::Vector4d u4d(0.8, 0.6, 0.3, -0.2);

  Eigen::Vector3d body = WheelLevel4DModel::forwardKinematics(u4d, dl, dr);
  Eigen::Vector4d u4d_recovered = WheelLevel4DModel::inverseKinematics(body, lf, lr, dl, dr);
  Eigen::Vector3d body_check = WheelLevel4DModel::forwardKinematics(u4d_recovered, dl, dr);

  // FK→IK→FK roundtrip: body velocity 일치
  EXPECT_NEAR(body(0), body_check(0), 1e-6);
  EXPECT_NEAR(body(1), body_check(1), 1e-6);
  EXPECT_NEAR(body(2), body_check(2), 1e-6);
}

TEST(HybridMPPI, A6_ClipControls)
{
  WheelLevel4DModel model(0.25, 0.25, 0.22, 0.22, 2.0, 1.0);

  Eigen::MatrixXd ctrl(3, 4);
  ctrl << 3.0, -3.0, 2.0, -2.0,    // 초과
          1.0,  0.5, 0.3, -0.3,     // 범위 내
          0.0,  0.0, 0.0,  0.0;     // 0

  Eigen::MatrixXd clipped = model.clipControls(ctrl);

  // 속도 클램프: [-2.0, 2.0]
  EXPECT_DOUBLE_EQ(clipped(0, 0), 2.0);
  EXPECT_DOUBLE_EQ(clipped(0, 1), -2.0);
  // 조향각 클램프: [-1.0, 1.0]
  EXPECT_DOUBLE_EQ(clipped(0, 2), 1.0);
  EXPECT_DOUBLE_EQ(clipped(0, 3), -1.0);
  // 범위 내: 변경 없음
  EXPECT_DOUBLE_EQ(clipped(1, 0), 1.0);
  EXPECT_DOUBLE_EQ(clipped(1, 2), 0.3);
}

// =============================================================================
// [B] WheelLevel4DModel 동역학 (2)
// =============================================================================

TEST(HybridMPPI, B7_DynamicsBatch_Straight)
{
  WheelLevel4DModel model(0.25, 0.25, 0.22, 0.22, 2.0, M_PI / 2);

  // 직진: V_fl=V_rr=1.0, δ=0, θ=0
  Eigen::MatrixXd states(1, 3);
  states << 0.0, 0.0, 0.0;
  Eigen::MatrixXd controls(1, 4);
  controls << 1.0, 1.0, 0.0, 0.0;

  Eigen::MatrixXd next = model.propagateBatch(states, controls, 0.1);
  EXPECT_GT(next(0, 0), 0.0);   // x 증가
  EXPECT_NEAR(next(0, 1), 0.0, 1e-10);  // y ≈ 0
  EXPECT_NEAR(next(0, 2), 0.0, 1e-10);  // θ ≈ 0
}

TEST(HybridMPPI, B8_Rollout_ConsistentWithSwerve)
{
  // FK(4D controls) = body velocity → Swerve와 동일한 궤적
  double lf = 0.25, lr = 0.25, dl = 0.22, dr = 0.22;
  double dt = 0.05;
  int N = 10;

  // 직진 + 약간의 회전
  Eigen::Vector3d body_vel(0.5, 0.1, 0.2);
  Eigen::Vector4d u4d = WheelLevel4DModel::inverseKinematics(body_vel, lf, lr, dl, dr);

  // 4D 모델 rollout
  WheelLevel4DModel model_4d(lf, lr, dl, dr, 2.0, M_PI / 2);
  std::vector<Eigen::MatrixXd> ctrl_4d_seq(1, Eigen::MatrixXd(N, 4));
  for (int t = 0; t < N; ++t) ctrl_4d_seq[0].row(t) = u4d.transpose();
  Eigen::VectorXd x0 = Eigen::Vector3d::Zero();
  auto traj_4d = model_4d.rolloutBatch(x0, ctrl_4d_seq, dt);

  // Swerve 모델 rollout (동일 body velocity)
  SwerveDriveModel model_swerve(-0.5, 1.0, 0.5, 1.0);
  std::vector<Eigen::MatrixXd> ctrl_swerve_seq(1, Eigen::MatrixXd(N, 3));
  for (int t = 0; t < N; ++t) ctrl_swerve_seq[0].row(t) = body_vel.transpose();
  auto traj_swerve = model_swerve.rolloutBatch(x0, ctrl_swerve_seq, dt);

  // 종점 비교 (약간의 수치 오차 허용)
  EXPECT_NEAR(traj_4d[0](N, 0), traj_swerve[0](N, 0), 0.05);
  EXPECT_NEAR(traj_4d[0](N, 1), traj_swerve[0](N, 1), 0.05);
  EXPECT_NEAR(traj_4d[0](N, 2), traj_swerve[0](N, 2), 0.05);
}

// =============================================================================
// [C] HybridSwerveMPPI 전환 — Coaxial (4)
// =============================================================================

TEST(HybridMPPI, C9_InitialModeLow)
{
  HybridTestFixture fix;
  fix.init("swerve");
  EXPECT_EQ(fix.accessor.current_mode_, HybridSwerveMPPIControllerPlugin::Mode::LOW_D);
}

TEST(HybridMPPI, C10_SwitchTo4D_LargeError)
{
  HybridTestFixture fix;
  fix.init("swerve");

  Eigen::VectorXd state = Eigen::Vector3d(0.0, 0.0, 0.0);

  // 참조: 멀리 떨어진 목표
  Eigen::MatrixXd ref = Eigen::MatrixXd::Zero(fix.params.N + 1, 3);
  for (int t = 0; t <= fix.params.N; ++t) {
    ref(t, 0) = 5.0;  // 5m 떨어진 목표
    ref(t, 1) = 3.0;
    ref(t, 2) = 1.0;  // 큰 각도 오차
  }

  auto mode = fix.accessor.determineMode(state, ref);
  EXPECT_EQ(mode, HybridSwerveMPPIControllerPlugin::Mode::FOUR_D);
}

TEST(HybridMPPI, C11_SwitchToLow_SmallError)
{
  HybridTestFixture fix;
  fix.init("swerve");

  Eigen::VectorXd state = Eigen::Vector3d(0.0, 0.0, 0.0);

  // 참조: 매우 가까운 목표
  Eigen::MatrixXd ref = Eigen::MatrixXd::Zero(fix.params.N + 1, 3);
  for (int t = 0; t <= fix.params.N; ++t) {
    ref(t, 0) = 0.01;
    ref(t, 1) = 0.01;
    ref(t, 2) = 0.01;
  }

  auto mode = fix.accessor.determineMode(state, ref);
  EXPECT_EQ(mode, HybridSwerveMPPIControllerPlugin::Mode::LOW_D);
}

TEST(HybridMPPI, C12_Hysteresis_Chattering)
{
  HybridTestFixture fix;
  fix.init("swerve");

  // 히스테리시스 카운터 테스트
  fix.accessor.current_mode_ = HybridSwerveMPPIControllerPlugin::Mode::LOW_D;
  fix.accessor.mode_switch_counter_ = 0;

  // 카운터를 수동으로 증가 (히스테리시스 미만)
  fix.accessor.mode_switch_counter_ = fix.params.hybrid_hysteresis_count - 1;
  // 아직 전환 안 됨
  EXPECT_EQ(fix.accessor.current_mode_, HybridSwerveMPPIControllerPlugin::Mode::LOW_D);

  // 카운터가 임계값 도달하면 전환 가능
  fix.accessor.mode_switch_counter_ = fix.params.hybrid_hysteresis_count;
  EXPECT_GE(fix.accessor.mode_switch_counter_, fix.params.hybrid_hysteresis_count);
}

// =============================================================================
// [D] Warm-start 변환 (2)
// =============================================================================

TEST(HybridMPPI, D13_WarmStartLowTo4D)
{
  HybridTestFixture fix;
  fix.init("swerve");

  int N = fix.params.N;

  // Low-D 제어열 설정: 직진 [vx=0.5, vy=0.1, omega=0.2]
  for (int t = 0; t < N; ++t) {
    fix.accessor.control_sequence_(t, 0) = 0.5;
    fix.accessor.control_sequence_(t, 1) = 0.1;
    fix.accessor.control_sequence_(t, 2) = 0.2;
  }

  fix.accessor.convertLowTo4D();

  // 4D 제어열이 유한하고 비어있지 않아야 함
  EXPECT_TRUE(fix.accessor.control_seq_4d_.allFinite());

  // FK(4D) → body velocity ≈ 원래 Low-D
  Eigen::Vector4d u4d = fix.accessor.control_seq_4d_.row(0).transpose();
  Eigen::Vector3d body = WheelLevel4DModel::forwardKinematics(
    u4d, fix.params.hybrid_dl, fix.params.hybrid_dr);

  EXPECT_NEAR(body(0), 0.5, 0.1);  // Vx ≈ 0.5
  EXPECT_NEAR(body(1), 0.1, 0.1);  // Vy ≈ 0.1
  EXPECT_NEAR(body(2), 0.2, 0.2);  // ω ≈ 0.2
}

TEST(HybridMPPI, D14_WarmStart4DToLow)
{
  HybridTestFixture fix;
  fix.init("swerve");

  int N = fix.params.N;

  // 4D 제어열 설정: 직진 [V_fl=0.8, V_rr=0.8, δ_fl=0, δ_rr=0]
  for (int t = 0; t < N; ++t) {
    fix.accessor.control_seq_4d_(t, 0) = 0.8;
    fix.accessor.control_seq_4d_(t, 1) = 0.8;
    fix.accessor.control_seq_4d_(t, 2) = 0.0;
    fix.accessor.control_seq_4d_(t, 3) = 0.0;
  }

  fix.accessor.convert4DToLow();

  // Low-D 결과: Vx ≈ 0.8, Vy ≈ 0, ω ≈ 0
  EXPECT_TRUE(fix.accessor.control_sequence_.allFinite());
  EXPECT_NEAR(fix.accessor.control_sequence_(0, 0), 0.8, 0.1);  // vx
  EXPECT_NEAR(fix.accessor.control_sequence_(0, 1), 0.0, 0.1);  // vy
  EXPECT_NEAR(fix.accessor.control_sequence_(0, 2), 0.0, 0.1);  // omega
}

// =============================================================================
// [E] Non-Coaxial 상태 전환 (2)
// =============================================================================

TEST(HybridMPPI, E15_NonCoaxial_DeltaMapping_To4D)
{
  HybridTestFixture fix;
  fix.init("non_coaxial_swerve");

  // NonCoaxial 상태: [x, y, θ, δ=0.3]
  Eigen::VectorXd state_nc(4);
  state_nc << 1.0, 2.0, 0.5, 0.3;

  // is_non_coaxial_ = true → 4D 상태 = head(3) = [1,2,0.5]
  Eigen::VectorXd state_4d = state_nc.head(3);
  EXPECT_EQ(state_4d.size(), 3);
  EXPECT_DOUBLE_EQ(state_4d(0), 1.0);
  EXPECT_DOUBLE_EQ(state_4d(1), 2.0);
  EXPECT_DOUBLE_EQ(state_4d(2), 0.5);

  // δ 초기화: tracked_delta_ = state_nc(3)
  fix.accessor.tracked_delta_ = state_nc(3);
  EXPECT_DOUBLE_EQ(fix.accessor.tracked_delta_, 0.3);
}

TEST(HybridMPPI, E16_NonCoaxial_DeltaMapping_ToLow)
{
  HybridTestFixture fix;
  fix.init("non_coaxial_swerve");

  // 4D 모드에서 마지막 제어: δ_fl=0.2, δ_rr=0.4
  fix.accessor.last_ctrl_4d_ = Eigen::Vector4d(0.5, 0.5, 0.2, 0.4);

  // δ_avg = (0.2 + 0.4) / 2 = 0.3
  double delta_avg = (fix.accessor.last_ctrl_4d_(2) + fix.accessor.last_ctrl_4d_(3)) / 2.0;
  fix.accessor.tracked_delta_ = delta_avg;
  EXPECT_NEAR(fix.accessor.tracked_delta_, 0.3, 1e-10);

  // 4D 상태 [x,y,θ] → NonCoaxial [x,y,θ,δ_avg]
  Eigen::VectorXd state_4d = Eigen::Vector3d(1.0, 2.0, 0.5);
  Eigen::VectorXd state_nc(4);
  state_nc.head(3) = state_4d;
  state_nc(3) = fix.accessor.tracked_delta_;
  EXPECT_DOUBLE_EQ(state_nc(3), 0.3);
}

// =============================================================================
// [F] 통합 (2)
// =============================================================================

TEST(HybridMPPI, F17_ComputeControl4D_CoaxialMode)
{
  HybridTestFixture fix;
  fix.init("swerve", 128, 10);

  Eigen::VectorXd x0 = Eigen::Vector3d(0.0, 0.0, 0.0);
  Eigen::MatrixXd ref = Eigen::MatrixXd::Zero(fix.params.N + 1, 3);
  for (int t = 0; t <= fix.params.N; ++t) {
    ref(t, 0) = 0.3 * t * fix.params.dt;
  }

  // Low-D → 4D warm-start 변환
  for (int t = 0; t < fix.params.N; ++t) {
    fix.accessor.control_sequence_(t, 0) = 0.3;  // vx
  }
  fix.accessor.convertLowTo4D();

  // 4D 모드로 직접 실행
  auto [u_opt_4d, info_4d] = fix.accessor.computeControl4D(x0, ref);

  EXPECT_EQ(u_opt_4d.size(), 3);  // 출력은 Low-D 형식 (swerve nu=3)
  EXPECT_TRUE(u_opt_4d.allFinite());
  EXPECT_GT(info_4d.ess, 0.0);
  EXPECT_EQ(static_cast<int>(info_4d.sample_trajectories.size()), fix.params.K);

  // 4D → Low-D 역변환 후 Low-D 제어열 유효성 확인
  fix.accessor.convert4DToLow();
  EXPECT_TRUE(fix.accessor.control_sequence_.allFinite());
}

TEST(HybridMPPI, F18_ComputeControl4D_NonCoaxialMode)
{
  HybridTestFixture fix;
  fix.init("non_coaxial_swerve", 128, 10);

  Eigen::VectorXd x0 = Eigen::VectorXd::Zero(4);  // [x, y, θ, δ]
  Eigen::MatrixXd ref = Eigen::MatrixXd::Zero(fix.params.N + 1, 4);
  for (int t = 0; t <= fix.params.N; ++t) {
    ref(t, 0) = 0.3 * t * fix.params.dt;
  }

  // Low-D 제어열 설정
  for (int t = 0; t < fix.params.N; ++t) {
    fix.accessor.control_sequence_(t, 0) = 0.3;  // v
  }

  // 4D warm-start
  fix.accessor.tracked_delta_ = 0.0;
  fix.accessor.convertLowTo4D();

  // 4D 모드 직접 실행
  auto [u_opt_4d, info_4d] = fix.accessor.computeControl4D(x0, ref);
  EXPECT_EQ(u_opt_4d.size(), 3);  // 출력은 Non-Coaxial 형식 (nu=3)
  EXPECT_TRUE(u_opt_4d.allFinite());
  EXPECT_GT(info_4d.ess, 0.0);

  // δ 추적 갱신 확인
  EXPECT_TRUE(std::isfinite(fix.accessor.tracked_delta_));
  EXPECT_TRUE(fix.accessor.last_ctrl_4d_.allFinite());
}

int main(int argc, char** argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
