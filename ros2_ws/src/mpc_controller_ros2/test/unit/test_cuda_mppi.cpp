// =============================================================================
// CUDA MPPI Unit Tests — CPU 참조 구현 검증 (15 gtest)
//
// CudaRolloutEngine의 CPU 폴백 구현을 테스트:
//   1. 생성, 2. GPU 미사용, 3. DiffDrive 롤아웃, 4. 비용 양수,
//   5. 궤적 차원, 6. 추적 비용, 7. 터미널 비용, 8. 제어 노력 비용,
//   9. 제어 변화율 비용, 10. 장애물 비용, 11. 무장애물 비용,
//   12. 제어 클리핑, 13. 각도 정규화, 14. Swerve 모델, 15. 대규모 K
// =============================================================================

#include <gtest/gtest.h>
#include <Eigen/Dense>
#include <cmath>
#include <vector>

#include "mpc_controller_ros2/cuda_rollout_engine.hpp"

namespace mpc_controller_ros2
{

// ============================================================================
// 테스트 픽스처: DiffDrive 기본 설정
// ============================================================================
class CudaMPPITest : public ::testing::Test
{
protected:
  void SetUp() override
  {
    config_.K = 64;
    config_.N = 10;
    config_.nx = 3;
    config_.nu = 2;
    config_.max_obstacles = 128;
    config_.motion_model = "diff_drive";

    engine_ = std::make_unique<CudaRolloutEngine>(config_);

    // Q: state tracking weight
    Q_ = Eigen::Matrix3d::Zero();
    Q_(0, 0) = 10.0;  // x
    Q_(1, 1) = 10.0;  // y
    Q_(2, 2) = 1.0;   // theta

    // Qf: terminal weight
    Qf_ = 2.0 * Q_;

    // R: control effort weight
    R_ = Eigen::Matrix2d::Zero();
    R_(0, 0) = 0.1;   // v
    R_(1, 1) = 0.1;   // omega

    // R_rate: control rate weight
    R_rate_ = Eigen::Matrix2d::Zero();
    R_rate_(0, 0) = 1.0;
    R_rate_(1, 1) = 1.0;

    engine_->initialize(Q_, Qf_, R_, R_rate_, 0.0, 1.0, -1.0, 1.0);

    // 기본 참조 궤적: 원점 정지
    reference_ = Eigen::MatrixXd::Zero(config_.N + 1, config_.nx);
    engine_->updateReference(reference_);
  }

  // 제로 제어로 K개 교란 생성
  std::vector<Eigen::MatrixXd> makeZeroControls(int K = -1) const
  {
    if (K < 0) K = config_.K;
    std::vector<Eigen::MatrixXd> controls(K);
    for (int k = 0; k < K; ++k) {
      controls[k] = Eigen::MatrixXd::Zero(config_.N, config_.nu);
    }
    return controls;
  }

  // 상수 제어로 K개 교란 생성
  std::vector<Eigen::MatrixXd> makeConstantControls(
    double v, double omega, int K = -1) const
  {
    if (K < 0) K = config_.K;
    std::vector<Eigen::MatrixXd> controls(K);
    for (int k = 0; k < K; ++k) {
      controls[k] = Eigen::MatrixXd::Zero(config_.N, config_.nu);
      for (int t = 0; t < config_.N; ++t) {
        controls[k](t, 0) = v;
        controls[k](t, 1) = omega;
      }
    }
    return controls;
  }

  CudaRolloutEngine::Config config_;
  std::unique_ptr<CudaRolloutEngine> engine_;
  Eigen::MatrixXd Q_, Qf_, R_, R_rate_, reference_;
};

// ============================================================================
// Test 1: Construction — 엔진 생성 확인
// ============================================================================
TEST_F(CudaMPPITest, Construction)
{
  EXPECT_EQ(engine_->config().K, 64);
  EXPECT_EQ(engine_->config().N, 10);
  EXPECT_EQ(engine_->config().nx, 3);
  EXPECT_EQ(engine_->config().nu, 2);
  EXPECT_EQ(engine_->config().motion_model, "diff_drive");
}

// ============================================================================
// Test 2: IsAvailable — CPU 빌드는 false
// ============================================================================
TEST_F(CudaMPPITest, IsAvailable)
{
  EXPECT_FALSE(CudaRolloutEngine::isAvailable());
}

// ============================================================================
// Test 3: DiffDriveRollout — execute()로 유한 비용 생성
// ============================================================================
TEST_F(CudaMPPITest, DiffDriveRollout)
{
  Eigen::VectorXd x0 = Eigen::Vector3d(0.0, 0.0, 0.0);
  auto controls = makeConstantControls(0.3, 0.1);

  Eigen::VectorXd costs;
  engine_->execute(x0, controls, 0.1, costs);

  ASSERT_EQ(costs.size(), config_.K);
  for (int k = 0; k < config_.K; ++k) {
    EXPECT_TRUE(std::isfinite(costs(k))) << "cost[" << k << "] is not finite";
  }
}

// ============================================================================
// Test 4: CostPositive — 모든 비용 >= 0
// ============================================================================
TEST_F(CudaMPPITest, CostPositive)
{
  Eigen::VectorXd x0 = Eigen::Vector3d(1.0, 1.0, 0.5);
  auto controls = makeConstantControls(0.5, 0.3);

  Eigen::VectorXd costs;
  engine_->execute(x0, controls, 0.1, costs);

  for (int k = 0; k < config_.K; ++k) {
    EXPECT_GE(costs(k), 0.0) << "cost[" << k << "] is negative";
  }
}

// ============================================================================
// Test 5: RolloutDimensions — K개 궤적, 각 (N+1) x nx
// ============================================================================
TEST_F(CudaMPPITest, RolloutDimensions)
{
  Eigen::VectorXd x0 = Eigen::Vector3d(0.0, 0.0, 0.0);
  auto controls = makeZeroControls();

  Eigen::VectorXd costs;
  engine_->execute(x0, controls, 0.1, costs);

  std::vector<Eigen::MatrixXd> trajectories;
  engine_->downloadAllTrajectories(trajectories);

  ASSERT_EQ(static_cast<int>(trajectories.size()), config_.K);
  for (int k = 0; k < config_.K; ++k) {
    EXPECT_EQ(trajectories[k].rows(), config_.N + 1);
    EXPECT_EQ(trajectories[k].cols(), config_.nx);
  }
}

// ============================================================================
// Test 6: TrackingCost — 참조 근처 → 낮은 비용, 멀리 → 높은 비용
// ============================================================================
TEST_F(CudaMPPITest, TrackingCost)
{
  // 참조: 원점
  Eigen::MatrixXd ref = Eigen::MatrixXd::Zero(config_.N + 1, config_.nx);
  engine_->updateReference(ref);

  // Case A: x0 = 원점, 제로 제어 → 낮은 비용
  Eigen::VectorXd x0_near = Eigen::Vector3d(0.0, 0.0, 0.0);
  auto controls_zero = makeZeroControls(1);

  Eigen::VectorXd costs_near;
  engine_->execute(x0_near, controls_zero, 0.1, costs_near);

  // Case B: x0 = (10, 10, 0), 제로 제어 → 높은 비용
  Eigen::VectorXd x0_far = Eigen::Vector3d(10.0, 10.0, 0.0);
  Eigen::VectorXd costs_far;
  engine_->execute(x0_far, controls_zero, 0.1, costs_far);

  EXPECT_LT(costs_near(0), costs_far(0))
    << "Near-reference cost should be lower than far-reference cost";
}

// ============================================================================
// Test 7: TerminalCost — 마지막 상태가 참조에서 멀면 비용 증가
// ============================================================================
TEST_F(CudaMPPITest, TerminalCost)
{
  // 참조: (5, 5, 0)으로 설정 → 원점 출발 시 터미널 비용 큼
  Eigen::MatrixXd ref_far = Eigen::MatrixXd::Zero(config_.N + 1, config_.nx);
  for (int t = 0; t <= config_.N; ++t) {
    ref_far(t, 0) = 5.0;
    ref_far(t, 1) = 5.0;
  }
  engine_->updateReference(ref_far);

  Eigen::VectorXd x0 = Eigen::Vector3d(0.0, 0.0, 0.0);
  auto controls = makeZeroControls(1);
  Eigen::VectorXd costs;
  engine_->execute(x0, controls, 0.1, costs);

  // 원점→(5,5) 오차 → 비용 확실히 양수
  EXPECT_GT(costs(0), 100.0)
    << "Terminal cost should be significant for large tracking error";
}

// ============================================================================
// Test 8: ControlEffortCost — 큰 제어 → 높은 비용
// ============================================================================
TEST_F(CudaMPPITest, ControlEffortCost)
{
  Eigen::VectorXd x0 = Eigen::Vector3d(0.0, 0.0, 0.0);

  // 작은 제어
  auto controls_small = makeConstantControls(0.1, 0.1, 1);
  Eigen::VectorXd costs_small;
  engine_->execute(x0, controls_small, 0.1, costs_small);

  // 큰 제어
  auto controls_large = makeConstantControls(1.0, 1.0, 1);
  Eigen::VectorXd costs_large;
  engine_->execute(x0, controls_large, 0.1, costs_large);

  EXPECT_GT(costs_large(0), costs_small(0))
    << "Large control effort should produce higher cost";
}

// ============================================================================
// Test 9: ControlRateCost — 급격한 제어 변화 → 높은 비용
// ============================================================================
TEST_F(CudaMPPITest, ControlRateCost)
{
  Eigen::VectorXd x0 = Eigen::Vector3d(0.0, 0.0, 0.0);

  // 일정 제어 (변화율 0)
  auto controls_smooth = makeConstantControls(0.5, 0.3, 1);
  Eigen::VectorXd costs_smooth;
  engine_->execute(x0, controls_smooth, 0.1, costs_smooth);

  // 급변 제어 (짝수/홀수 스텝 번갈아)
  std::vector<Eigen::MatrixXd> controls_jerky(1);
  controls_jerky[0] = Eigen::MatrixXd::Zero(config_.N, config_.nu);
  for (int t = 0; t < config_.N; ++t) {
    controls_jerky[0](t, 0) = (t % 2 == 0) ? 0.8 : 0.2;
    controls_jerky[0](t, 1) = (t % 2 == 0) ? 0.5 : -0.5;
  }
  Eigen::VectorXd costs_jerky;
  engine_->execute(x0, controls_jerky, 0.1, costs_jerky);

  EXPECT_GT(costs_jerky(0), costs_smooth(0))
    << "Jerky controls should have higher rate cost";
}

// ============================================================================
// Test 10: ObstacleCost — 궤적 경로에 장애물 → 비용 증가
// ============================================================================
TEST_F(CudaMPPITest, ObstacleCost)
{
  Eigen::VectorXd x0 = Eigen::Vector3d(0.0, 0.0, 0.0);
  auto controls = makeConstantControls(0.5, 0.0, 1);  // 직진

  // 장애물 없이 실행
  Eigen::VectorXd costs_no_obs;
  engine_->execute(x0, controls, 0.1, costs_no_obs);

  // 궤적 경로에 장애물 추가 (x=0.5, y=0, radius=0.1)
  std::vector<Eigen::Vector3d> obstacles;
  obstacles.push_back(Eigen::Vector3d(0.5, 0.0, 0.1));
  engine_->updateObstacles(obstacles, 0.5, 100.0);

  Eigen::VectorXd costs_with_obs;
  engine_->execute(x0, controls, 0.1, costs_with_obs);

  EXPECT_GT(costs_with_obs(0), costs_no_obs(0))
    << "Obstacle on trajectory path should increase cost";

  // 장애물 제거
  engine_->updateObstacles({}, 0.5, 100.0);
}

// ============================================================================
// Test 11: NoObstacleCost — 장애물 없으면 장애물 비용 = 0
// ============================================================================
TEST_F(CudaMPPITest, NoObstacleCost)
{
  Eigen::VectorXd x0 = Eigen::Vector3d(0.0, 0.0, 0.0);
  auto controls_a = makeConstantControls(0.3, 0.0, 1);

  // 장애물 비활성화 상태에서 실행
  engine_->updateObstacles({}, 0.5, 100.0);
  Eigen::VectorXd costs_a;
  engine_->execute(x0, controls_a, 0.1, costs_a);

  // 궤적과 무관한 먼 장애물 추가
  std::vector<Eigen::Vector3d> far_obstacles;
  far_obstacles.push_back(Eigen::Vector3d(100.0, 100.0, 0.1));
  engine_->updateObstacles(far_obstacles, 0.5, 100.0);
  Eigen::VectorXd costs_b;
  engine_->execute(x0, controls_a, 0.1, costs_b);

  // 멀리 있는 장애물은 비용에 영향 없음
  EXPECT_NEAR(costs_a(0), costs_b(0), 1e-6)
    << "Far away obstacle should not affect cost";

  engine_->updateObstacles({}, 0.5, 100.0);
}

// ============================================================================
// Test 12: ControlClipping — 제어 [v_min, v_max] x [omega_min, omega_max]
// ============================================================================
TEST_F(CudaMPPITest, ControlClipping)
{
  // v_min=0, v_max=1, omega_min=-1, omega_max=1
  Eigen::VectorXd x0 = Eigen::Vector3d(0.0, 0.0, 0.0);

  // 제한 초과 제어
  std::vector<Eigen::MatrixXd> controls_over(1);
  controls_over[0] = Eigen::MatrixXd::Zero(config_.N, config_.nu);
  for (int t = 0; t < config_.N; ++t) {
    controls_over[0](t, 0) = 5.0;    // v >> v_max=1
    controls_over[0](t, 1) = 3.0;    // omega >> omega_max=1
  }

  // 제한 내 제어 (클리핑된 값과 동일)
  auto controls_clipped = makeConstantControls(1.0, 1.0, 1);

  Eigen::VectorXd costs_over, costs_clipped;
  engine_->execute(x0, controls_over, 0.1, costs_over);
  engine_->execute(x0, controls_clipped, 0.1, costs_clipped);

  // 궤적은 동일 (클리핑 후 동일 제어), 제어 비용만 다를 수 있음
  // 궤적 비교로 클리핑 확인
  std::vector<Eigen::MatrixXd> traj_over, traj_clipped;
  engine_->execute(x0, controls_over, 0.1, costs_over);
  engine_->downloadAllTrajectories(traj_over);
  engine_->execute(x0, controls_clipped, 0.1, costs_clipped);
  engine_->downloadAllTrajectories(traj_clipped);

  // 클리핑 후 궤적 동일 확인
  for (int t = 0; t <= config_.N; ++t) {
    for (int i = 0; i < config_.nx; ++i) {
      EXPECT_NEAR(traj_over[0](t, i), traj_clipped[0](t, i), 1e-10)
        << "Trajectories should match after clipping at t=" << t << ", i=" << i;
    }
  }
}

// ============================================================================
// Test 13: AngleNormalization — theta ∈ [-π, π]
// ============================================================================
TEST_F(CudaMPPITest, AngleNormalization)
{
  // 초기 theta=3.0, 큰 omega로 각도 급변
  Eigen::VectorXd x0 = Eigen::Vector3d(0.0, 0.0, 3.0);
  auto controls = makeConstantControls(0.3, 1.0, 1);

  Eigen::VectorXd costs;
  engine_->execute(x0, controls, 0.1, costs);

  std::vector<Eigen::MatrixXd> trajectories;
  engine_->downloadAllTrajectories(trajectories);

  for (int t = 0; t <= config_.N; ++t) {
    double theta = trajectories[0](t, 2);
    EXPECT_GE(theta, -M_PI - 1e-10) << "theta at t=" << t << " below -pi";
    EXPECT_LE(theta, M_PI + 1e-10) << "theta at t=" << t << " above pi";
  }
}

// ============================================================================
// Test 14: SwerveModel — nx=3, nu=3 롤아웃 동작 확인
// ============================================================================
TEST_F(CudaMPPITest, SwerveModel)
{
  CudaRolloutEngine::Config swerve_config;
  swerve_config.K = 16;
  swerve_config.N = 10;
  swerve_config.nx = 3;
  swerve_config.nu = 3;
  swerve_config.motion_model = "swerve";

  auto swerve_engine = std::make_unique<CudaRolloutEngine>(swerve_config);

  // 3x3 행렬 초기화
  Eigen::Matrix3d Q_s = Eigen::Matrix3d::Zero();
  Q_s(0, 0) = 10.0; Q_s(1, 1) = 10.0; Q_s(2, 2) = 1.0;
  Eigen::Matrix3d R_s = Eigen::Matrix3d::Zero();
  R_s(0, 0) = 0.1; R_s(1, 1) = 0.1; R_s(2, 2) = 0.1;
  Eigen::Matrix3d R_rate_s = Eigen::Matrix3d::Identity();

  swerve_engine->initialize(Q_s, 2.0 * Q_s, R_s, R_rate_s,
                            0.0, 1.0, -1.0, 1.0, 0.5);

  // Swerve 참조
  Eigen::MatrixXd ref_s = Eigen::MatrixXd::Zero(11, 3);
  swerve_engine->updateReference(ref_s);

  // u = [vx, vy, omega]
  Eigen::VectorXd x0 = Eigen::Vector3d(0.0, 0.0, 0.0);
  std::vector<Eigen::MatrixXd> controls(16);
  for (int k = 0; k < 16; ++k) {
    controls[k] = Eigen::MatrixXd::Zero(10, 3);
    for (int t = 0; t < 10; ++t) {
      controls[k](t, 0) = 0.3;   // vx
      controls[k](t, 1) = 0.1;   // vy
      controls[k](t, 2) = 0.05;  // omega
    }
  }

  Eigen::VectorXd costs;
  swerve_engine->execute(x0, controls, 0.1, costs);

  ASSERT_EQ(costs.size(), 16);
  for (int k = 0; k < 16; ++k) {
    EXPECT_TRUE(std::isfinite(costs(k))) << "Swerve cost[" << k << "] not finite";
    EXPECT_GE(costs(k), 0.0) << "Swerve cost[" << k << "] negative";
  }

  // 궤적 차원 검증
  std::vector<Eigen::MatrixXd> trajectories;
  swerve_engine->downloadAllTrajectories(trajectories);
  ASSERT_EQ(static_cast<int>(trajectories.size()), 16);
  EXPECT_EQ(trajectories[0].rows(), 11);
  EXPECT_EQ(trajectories[0].cols(), 3);
}

// ============================================================================
// Test 15: LargeK — K=512 대규모 샘플 실행 확인
// ============================================================================
TEST_F(CudaMPPITest, LargeK)
{
  CudaRolloutEngine::Config large_config;
  large_config.K = 512;
  large_config.N = 30;
  large_config.nx = 3;
  large_config.nu = 2;
  large_config.motion_model = "diff_drive";

  auto large_engine = std::make_unique<CudaRolloutEngine>(large_config);
  large_engine->initialize(Q_, Qf_, R_, R_rate_, 0.0, 1.0, -1.0, 1.0);

  Eigen::MatrixXd ref = Eigen::MatrixXd::Zero(31, 3);
  large_engine->updateReference(ref);

  Eigen::VectorXd x0 = Eigen::Vector3d(1.0, 2.0, 0.5);
  std::vector<Eigen::MatrixXd> controls(512);
  for (int k = 0; k < 512; ++k) {
    controls[k] = Eigen::MatrixXd::Random(30, 2) * 0.5;
  }

  Eigen::VectorXd costs;
  large_engine->execute(x0, controls, 0.1, costs);

  ASSERT_EQ(costs.size(), 512);
  for (int k = 0; k < 512; ++k) {
    EXPECT_TRUE(std::isfinite(costs(k))) << "LargeK cost[" << k << "] not finite";
  }
}

}  // namespace mpc_controller_ros2
