/**
 * @brief MotionModel 추상화 + 3종 모델 단위 테스트
 *
 * Phase 0: 새 파일만 테스트 — 기존 코드 무변경
 */
#include <gtest/gtest.h>
#include <cmath>
#include "mpc_controller_ros2/motion_model.hpp"
#include "mpc_controller_ros2/diff_drive_model.hpp"
#include "mpc_controller_ros2/swerve_drive_model.hpp"
#include "mpc_controller_ros2/non_coaxial_swerve_model.hpp"
#include "mpc_controller_ros2/motion_model_factory.hpp"
#include "mpc_controller_ros2/mppi_params.hpp"

using namespace mpc_controller_ros2;

// ============================================================================
// DiffDriveModel Tests
// ============================================================================

class DiffDriveModelTest : public ::testing::Test
{
protected:
  void SetUp() override
  {
    model_ = std::make_unique<DiffDriveModel>(0.0, 1.0, -1.5, 1.5);
  }
  std::unique_ptr<DiffDriveModel> model_;
};

TEST_F(DiffDriveModelTest, Dimensions)
{
  EXPECT_EQ(model_->stateDim(), 3);
  EXPECT_EQ(model_->controlDim(), 2);
  EXPECT_FALSE(model_->isHolonomic());
  EXPECT_EQ(model_->name(), "diff_drive");
  EXPECT_EQ(model_->angleIndices(), std::vector<int>({2}));
}

TEST_F(DiffDriveModelTest, DynamicsBatchStationary)
{
  // 정지 상태 → zero dynamics
  Eigen::MatrixXd states = Eigen::MatrixXd::Zero(5, 3);
  Eigen::MatrixXd controls = Eigen::MatrixXd::Zero(5, 2);

  auto state_dot = model_->dynamicsBatch(states, controls);

  EXPECT_EQ(state_dot.rows(), 5);
  EXPECT_EQ(state_dot.cols(), 3);
  EXPECT_NEAR(state_dot.norm(), 0.0, 1e-10);
}

TEST_F(DiffDriveModelTest, DynamicsBatchForward)
{
  // 전진 (v=1, omega=0, theta=0) → x_dot=1, y_dot=0
  Eigen::MatrixXd states(1, 3);
  states << 0.0, 0.0, 0.0;
  Eigen::MatrixXd controls(1, 2);
  controls << 1.0, 0.0;

  auto dot = model_->dynamicsBatch(states, controls);
  EXPECT_NEAR(dot(0, 0), 1.0, 1e-10);  // x_dot
  EXPECT_NEAR(dot(0, 1), 0.0, 1e-10);  // y_dot
  EXPECT_NEAR(dot(0, 2), 0.0, 1e-10);  // theta_dot
}

TEST_F(DiffDriveModelTest, DynamicsBatchDiagonal)
{
  // theta=pi/4, v=1 → x_dot=cos(pi/4), y_dot=sin(pi/4)
  Eigen::MatrixXd states(1, 3);
  states << 0.0, 0.0, M_PI / 4.0;
  Eigen::MatrixXd controls(1, 2);
  controls << 1.0, 0.0;

  auto dot = model_->dynamicsBatch(states, controls);
  EXPECT_NEAR(dot(0, 0), std::cos(M_PI / 4.0), 1e-10);
  EXPECT_NEAR(dot(0, 1), std::sin(M_PI / 4.0), 1e-10);
}

TEST_F(DiffDriveModelTest, PropagateRK4Accuracy)
{
  // 직선 전진: dt=0.1, v=1 → x ≈ 0.1
  Eigen::MatrixXd states(1, 3);
  states << 0.0, 0.0, 0.0;
  Eigen::MatrixXd controls(1, 2);
  controls << 1.0, 0.0;

  auto next = model_->propagateBatch(states, controls, 0.1);
  EXPECT_NEAR(next(0, 0), 0.1, 1e-6);
  EXPECT_NEAR(next(0, 1), 0.0, 1e-6);
  EXPECT_NEAR(next(0, 2), 0.0, 1e-6);
}

TEST_F(DiffDriveModelTest, PropagateCircularMotion)
{
  // 원운동: v=1, omega=1, dt=0.1 → theta ≈ 0.1
  Eigen::MatrixXd states(1, 3);
  states << 0.0, 0.0, 0.0;
  Eigen::MatrixXd controls(1, 2);
  controls << 1.0, 1.0;

  auto next = model_->propagateBatch(states, controls, 0.1);
  EXPECT_NEAR(next(0, 2), 0.1, 1e-4);
  EXPECT_GT(next(0, 0), 0.0);  // x > 0
}

TEST_F(DiffDriveModelTest, RolloutBatch)
{
  Eigen::VectorXd x0 = Eigen::Vector3d(0.0, 0.0, 0.0);
  int K = 3, N = 5;

  std::vector<Eigen::MatrixXd> ctrl_seqs;
  for (int k = 0; k < K; ++k) {
    Eigen::MatrixXd ctrl = Eigen::MatrixXd::Zero(N, 2);
    ctrl.col(0).setConstant(0.5);  // v = 0.5
    ctrl_seqs.push_back(ctrl);
  }

  auto trajs = model_->rolloutBatch(x0, ctrl_seqs, 0.1);
  EXPECT_EQ(static_cast<int>(trajs.size()), K);
  EXPECT_EQ(trajs[0].rows(), N + 1);
  EXPECT_EQ(trajs[0].cols(), 3);
  // 초기 상태 확인
  EXPECT_NEAR(trajs[0](0, 0), 0.0, 1e-10);
  // 전진 확인
  EXPECT_GT(trajs[0](N, 0), 0.0);
}

TEST_F(DiffDriveModelTest, ClipControls)
{
  Eigen::MatrixXd controls(3, 2);
  controls << 2.0, 3.0,
              -1.0, -2.0,
              0.5, 0.0;

  auto clipped = model_->clipControls(controls);
  EXPECT_NEAR(clipped(0, 0), 1.0, 1e-10);   // v clamped to v_max
  EXPECT_NEAR(clipped(0, 1), 1.5, 1e-10);   // omega clamped to omega_max
  EXPECT_NEAR(clipped(1, 0), 0.0, 1e-10);   // v clamped to v_min
  EXPECT_NEAR(clipped(1, 1), -1.5, 1e-10);  // omega clamped to omega_min
  EXPECT_NEAR(clipped(2, 0), 0.5, 1e-10);   // within range
}

TEST_F(DiffDriveModelTest, NormalizeStates)
{
  Eigen::MatrixXd states(2, 3);
  states << 1.0, 2.0, 4.0 * M_PI,
            3.0, 4.0, -3.0 * M_PI;

  model_->normalizeStates(states);
  EXPECT_NEAR(states(0, 2), 0.0, 1e-10);
  EXPECT_NEAR(states(1, 2), -M_PI, 1e-10);
}

TEST_F(DiffDriveModelTest, ControlToTwistRoundTrip)
{
  Eigen::VectorXd control(2);
  control << 0.8, -0.3;

  auto twist = model_->controlToTwist(control);
  auto recovered = model_->twistToControl(twist);

  EXPECT_NEAR(recovered(0), control(0), 1e-10);
  EXPECT_NEAR(recovered(1), control(1), 1e-10);
}

// ============================================================================
// SwerveDriveModel Tests
// ============================================================================

class SwerveDriveModelTest : public ::testing::Test
{
protected:
  void SetUp() override
  {
    model_ = std::make_unique<SwerveDriveModel>(-1.5, 1.5, 1.5, 2.0);
  }
  std::unique_ptr<SwerveDriveModel> model_;
};

TEST_F(SwerveDriveModelTest, Dimensions)
{
  EXPECT_EQ(model_->stateDim(), 3);
  EXPECT_EQ(model_->controlDim(), 3);
  EXPECT_TRUE(model_->isHolonomic());
  EXPECT_EQ(model_->name(), "swerve");
}

TEST_F(SwerveDriveModelTest, DynamicsBatchStationary)
{
  Eigen::MatrixXd states = Eigen::MatrixXd::Zero(3, 3);
  Eigen::MatrixXd controls = Eigen::MatrixXd::Zero(3, 3);

  auto dot = model_->dynamicsBatch(states, controls);
  EXPECT_NEAR(dot.norm(), 0.0, 1e-10);
}

TEST_F(SwerveDriveModelTest, DynamicsBatchLateralMotion)
{
  // theta=0, vx=0, vy=1 → x_dot=0, y_dot=1
  Eigen::MatrixXd states(1, 3);
  states << 0.0, 0.0, 0.0;
  Eigen::MatrixXd controls(1, 3);
  controls << 0.0, 1.0, 0.0;

  auto dot = model_->dynamicsBatch(states, controls);
  EXPECT_NEAR(dot(0, 0), 0.0, 1e-10);   // x_dot
  EXPECT_NEAR(dot(0, 1), 1.0, 1e-10);   // y_dot
}

TEST_F(SwerveDriveModelTest, DynamicsBatchRotatedFrame)
{
  // theta=pi/2, vx=1, vy=0 → x_dot=cos(pi/2)≈0, y_dot=sin(pi/2)=1
  Eigen::MatrixXd states(1, 3);
  states << 0.0, 0.0, M_PI / 2.0;
  Eigen::MatrixXd controls(1, 3);
  controls << 1.0, 0.0, 0.0;

  auto dot = model_->dynamicsBatch(states, controls);
  EXPECT_NEAR(dot(0, 0), 0.0, 1e-10);
  EXPECT_NEAR(dot(0, 1), 1.0, 1e-10);
}

TEST_F(SwerveDriveModelTest, PropagateRK4)
{
  Eigen::MatrixXd states(1, 3);
  states << 0.0, 0.0, 0.0;
  Eigen::MatrixXd controls(1, 3);
  controls << 1.0, 0.5, 0.0;

  auto next = model_->propagateBatch(states, controls, 0.1);
  EXPECT_NEAR(next(0, 0), 0.1, 1e-4);   // x ≈ vx * dt
  EXPECT_NEAR(next(0, 1), 0.05, 1e-4);  // y ≈ vy * dt
}

TEST_F(SwerveDriveModelTest, RolloutBatch)
{
  Eigen::VectorXd x0 = Eigen::Vector3d(0.0, 0.0, 0.0);
  int K = 2, N = 10;

  std::vector<Eigen::MatrixXd> ctrl_seqs;
  for (int k = 0; k < K; ++k) {
    Eigen::MatrixXd ctrl = Eigen::MatrixXd::Zero(N, 3);
    ctrl.col(0).setConstant(0.5);  // vx
    ctrl.col(1).setConstant(0.3);  // vy
    ctrl_seqs.push_back(ctrl);
  }

  auto trajs = model_->rolloutBatch(x0, ctrl_seqs, 0.1);
  EXPECT_EQ(static_cast<int>(trajs.size()), K);
  EXPECT_EQ(trajs[0].rows(), N + 1);
  EXPECT_EQ(trajs[0].cols(), 3);
  EXPECT_GT(trajs[0](N, 0), 0.0);
  EXPECT_GT(trajs[0](N, 1), 0.0);
}

TEST_F(SwerveDriveModelTest, ClipControls)
{
  Eigen::MatrixXd controls(1, 3);
  controls << 3.0, -3.0, 5.0;

  auto clipped = model_->clipControls(controls);
  EXPECT_NEAR(clipped(0, 0), 1.5, 1e-10);   // vx clamped
  EXPECT_NEAR(clipped(0, 1), -1.5, 1e-10);  // vy clamped
  EXPECT_NEAR(clipped(0, 2), 2.0, 1e-10);   // omega clamped
}

TEST_F(SwerveDriveModelTest, ControlToTwistRoundTrip)
{
  Eigen::VectorXd control(3);
  control << 0.5, -0.3, 0.8;

  auto twist = model_->controlToTwist(control);
  auto recovered = model_->twistToControl(twist);

  EXPECT_NEAR(recovered(0), control(0), 1e-10);
  EXPECT_NEAR(recovered(1), control(1), 1e-10);
  EXPECT_NEAR(recovered(2), control(2), 1e-10);
}

TEST(SwerveClipControlsTest, RespectsVmin)
{
  // vx_min=0 → 음수 속도가 0으로 클리핑
  auto model = std::make_unique<SwerveDriveModel>(0.0, 1.5, 1.5, 2.0);

  Eigen::MatrixXd controls(3, 3);
  controls << -0.5, 0.3, 0.0,
               0.5, -0.3, 0.0,
               1.0, 0.0, 0.0;

  auto clipped = model->clipControls(controls);
  EXPECT_NEAR(clipped(0, 0), 0.0, 1e-10);   // 음수 → 0으로 클리핑
  EXPECT_NEAR(clipped(1, 0), 0.5, 1e-10);   // 양수 유지
  EXPECT_NEAR(clipped(2, 0), 1.0, 1e-10);   // 양수 유지
  EXPECT_NEAR(clipped(0, 1), 0.3, 1e-10);   // vy는 vx_min 영향 없음
}

TEST(SwerveClipControlsTest, NegativeVminAllowsBackward)
{
  // vx_min=-0.5 → -0.5까지 후진 허용
  auto model = std::make_unique<SwerveDriveModel>(-0.5, 1.5, 1.5, 2.0);

  Eigen::MatrixXd controls(2, 3);
  controls << -1.0, 0.0, 0.0,
              -0.3, 0.0, 0.0;

  auto clipped = model->clipControls(controls);
  EXPECT_NEAR(clipped(0, 0), -0.5, 1e-10);  // -1.0 → -0.5로 클리핑
  EXPECT_NEAR(clipped(1, 0), -0.3, 1e-10);  // 범위 내 유지
}

// ============================================================================
// NonCoaxialSwerveModel Tests
// ============================================================================

class NonCoaxialSwerveModelTest : public ::testing::Test
{
protected:
  void SetUp() override
  {
    model_ = std::make_unique<NonCoaxialSwerveModel>(-1.5, 1.5, 2.0, 2.0, M_PI / 2.0);
  }
  std::unique_ptr<NonCoaxialSwerveModel> model_;
};

TEST_F(NonCoaxialSwerveModelTest, Dimensions)
{
  EXPECT_EQ(model_->stateDim(), 4);
  EXPECT_EQ(model_->controlDim(), 3);
  EXPECT_FALSE(model_->isHolonomic());
  EXPECT_EQ(model_->name(), "non_coaxial_swerve");
}

TEST_F(NonCoaxialSwerveModelTest, DynamicsBatchStationary)
{
  Eigen::MatrixXd states = Eigen::MatrixXd::Zero(3, 4);
  Eigen::MatrixXd controls = Eigen::MatrixXd::Zero(3, 3);

  auto dot = model_->dynamicsBatch(states, controls);
  EXPECT_EQ(dot.rows(), 3);
  EXPECT_EQ(dot.cols(), 4);
  EXPECT_NEAR(dot.norm(), 0.0, 1e-10);
}

TEST_F(NonCoaxialSwerveModelTest, DynamicsBatchStraight)
{
  // delta=0, v=1 → vx_body=1, vy_body=0 → forward motion
  Eigen::MatrixXd states(1, 4);
  states << 0.0, 0.0, 0.0, 0.0;  // x,y,theta,delta
  Eigen::MatrixXd controls(1, 3);
  controls << 1.0, 0.0, 0.0;  // v,omega,delta_dot

  auto dot = model_->dynamicsBatch(states, controls);
  EXPECT_NEAR(dot(0, 0), 1.0, 1e-10);  // x_dot
  EXPECT_NEAR(dot(0, 1), 0.0, 1e-10);  // y_dot
  EXPECT_NEAR(dot(0, 2), 0.0, 1e-10);  // theta_dot
  EXPECT_NEAR(dot(0, 3), 0.0, 1e-10);  // delta_dot
}

TEST_F(NonCoaxialSwerveModelTest, DynamicsBatchSteered)
{
  // delta=pi/4, v=1 → diagonal body velocity
  Eigen::MatrixXd states(1, 4);
  states << 0.0, 0.0, 0.0, M_PI / 4.0;
  Eigen::MatrixXd controls(1, 3);
  controls << 1.0, 0.0, 0.0;

  auto dot = model_->dynamicsBatch(states, controls);
  EXPECT_NEAR(dot(0, 0), std::cos(M_PI / 4.0), 1e-10);  // x_dot
  EXPECT_NEAR(dot(0, 1), std::sin(M_PI / 4.0), 1e-10);  // y_dot
}

TEST_F(NonCoaxialSwerveModelTest, PropagateDeltaClamp)
{
  // delta가 max를 넘도록 설정
  Eigen::MatrixXd states(1, 4);
  states << 0.0, 0.0, 0.0, M_PI / 2.0 - 0.1;
  Eigen::MatrixXd controls(1, 3);
  controls << 0.0, 0.0, 5.0;  // 큰 delta_dot

  auto next = model_->propagateBatch(states, controls, 0.5);
  EXPECT_LE(next(0, 3), M_PI / 2.0 + 1e-10);  // delta clamped
}

TEST_F(NonCoaxialSwerveModelTest, RolloutBatch)
{
  Eigen::VectorXd x0(4);
  x0 << 0.0, 0.0, 0.0, 0.0;
  int K = 2, N = 5;

  std::vector<Eigen::MatrixXd> ctrl_seqs;
  for (int k = 0; k < K; ++k) {
    Eigen::MatrixXd ctrl = Eigen::MatrixXd::Zero(N, 3);
    ctrl.col(0).setConstant(0.5);
    ctrl_seqs.push_back(ctrl);
  }

  auto trajs = model_->rolloutBatch(x0, ctrl_seqs, 0.1);
  EXPECT_EQ(static_cast<int>(trajs.size()), K);
  EXPECT_EQ(trajs[0].rows(), N + 1);
  EXPECT_EQ(trajs[0].cols(), 4);
}

TEST_F(NonCoaxialSwerveModelTest, ClipControls)
{
  Eigen::MatrixXd controls(1, 3);
  controls << 5.0, -5.0, 10.0;

  auto clipped = model_->clipControls(controls);
  EXPECT_NEAR(clipped(0, 0), 1.5, 1e-10);
  EXPECT_NEAR(clipped(0, 1), -2.0, 1e-10);
  EXPECT_NEAR(clipped(0, 2), 2.0, 1e-10);
}

TEST_F(NonCoaxialSwerveModelTest, NormalizeStates)
{
  Eigen::MatrixXd states(1, 4);
  states << 0.0, 0.0, 4.0 * M_PI, 3.0;  // theta wraps, delta clamps

  model_->normalizeStates(states);
  EXPECT_NEAR(states(0, 2), 0.0, 1e-10);
  EXPECT_LE(states(0, 3), M_PI / 2.0 + 1e-10);
}

TEST_F(NonCoaxialSwerveModelTest, ControlToTwistRoundTrip)
{
  Eigen::VectorXd control(3);
  control << 0.5, 0.3, 0.1;

  auto twist = model_->controlToTwist(control);
  auto recovered = model_->twistToControl(twist);

  EXPECT_NEAR(recovered(0), control(0), 1e-10);
  EXPECT_NEAR(recovered(1), control(1), 1e-10);
  // delta_dot은 Twist에서 보존 안 됨 → 0.0
  EXPECT_NEAR(recovered(2), 0.0, 1e-10);
}

TEST(NonCoaxialClipControlsTest, RespectsVmin)
{
  // v_min=0 → 음수 속도가 0으로 클리핑
  auto model = std::make_unique<NonCoaxialSwerveModel>(0.0, 1.5, 2.0, 2.0, M_PI / 2.0);

  Eigen::MatrixXd controls(2, 3);
  controls << -0.5, 0.0, 0.0,
               0.5, 0.0, 0.0;

  auto clipped = model->clipControls(controls);
  EXPECT_NEAR(clipped(0, 0), 0.0, 1e-10);   // 음수 → 0으로 클리핑
  EXPECT_NEAR(clipped(1, 0), 0.5, 1e-10);   // 양수 유지
}

TEST(NonCoaxialClipControlsTest, NegativeVminAllowsBackward)
{
  // v_min=-0.3 → -0.3까지 후진 허용
  auto model = std::make_unique<NonCoaxialSwerveModel>(-0.3, 1.5, 2.0, 2.0, M_PI / 2.0);

  Eigen::MatrixXd controls(2, 3);
  controls << -1.0, 0.0, 0.0,
              -0.2, 0.0, 0.0;

  auto clipped = model->clipControls(controls);
  EXPECT_NEAR(clipped(0, 0), -0.3, 1e-10);  // -1.0 → -0.3으로 클리핑
  EXPECT_NEAR(clipped(1, 0), -0.2, 1e-10);  // 범위 내 유지
}

// ============================================================================
// MotionModelFactory Tests
// ============================================================================

TEST(MotionModelFactoryTest, CreateDiffDrive)
{
  MPPIParams params;
  auto model = MotionModelFactory::create("diff_drive", params);
  EXPECT_EQ(model->stateDim(), 3);
  EXPECT_EQ(model->controlDim(), 2);
  EXPECT_EQ(model->name(), "diff_drive");
}

TEST(MotionModelFactoryTest, CreateSwerve)
{
  MPPIParams params;
  auto model = MotionModelFactory::create("swerve", params);
  EXPECT_EQ(model->stateDim(), 3);
  EXPECT_EQ(model->controlDim(), 3);
  EXPECT_EQ(model->name(), "swerve");
}

TEST(MotionModelFactoryTest, CreateNonCoaxialSwerve)
{
  MPPIParams params;
  auto model = MotionModelFactory::create("non_coaxial_swerve", params);
  EXPECT_EQ(model->stateDim(), 4);
  EXPECT_EQ(model->controlDim(), 3);
  EXPECT_EQ(model->name(), "non_coaxial_swerve");
}

TEST(MotionModelFactoryTest, InvalidType)
{
  MPPIParams params;
  EXPECT_THROW(MotionModelFactory::create("invalid", params), std::invalid_argument);
}

// ============================================================================
// Cross-Model Comparison: 동일 조건에서 DiffDrive vs Swerve 비교
// ============================================================================

TEST(CrossModelTest, SameForwardMotion)
{
  // DiffDrive: v=1, omega=0
  auto dd = std::make_unique<DiffDriveModel>(0.0, 2.0, -2.0, 2.0);
  // Swerve: vx=1, vy=0, omega=0
  auto sw = std::make_unique<SwerveDriveModel>(-2.0, 2.0, 2.0, 2.0);

  Eigen::MatrixXd dd_states(1, 3), dd_controls(1, 2);
  dd_states << 0.0, 0.0, 0.0;
  dd_controls << 1.0, 0.0;

  Eigen::MatrixXd sw_states(1, 3), sw_controls(1, 3);
  sw_states << 0.0, 0.0, 0.0;
  sw_controls << 1.0, 0.0, 0.0;

  auto dd_next = dd->propagateBatch(dd_states, dd_controls, 0.1);
  auto sw_next = sw->propagateBatch(sw_states, sw_controls, 0.1);

  // 같은 전진 운동 → 같은 결과
  EXPECT_NEAR(dd_next(0, 0), sw_next(0, 0), 1e-10);
  EXPECT_NEAR(dd_next(0, 1), sw_next(0, 1), 1e-10);
  EXPECT_NEAR(dd_next(0, 2), sw_next(0, 2), 1e-10);
}

int main(int argc, char** argv)
{
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
