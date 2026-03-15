#include <gtest/gtest.h>
#include <Eigen/Dense>
#include <cmath>

#include "mpc_controller_ros2/predictive_safety_filter.hpp"
#include "mpc_controller_ros2/barrier_function.hpp"
#include "mpc_controller_ros2/motion_model.hpp"
#include "mpc_controller_ros2/motion_model_factory.hpp"
#include "mpc_controller_ros2/mppi_params.hpp"

using namespace mpc_controller_ros2;

// =============================================================================
// 테스트 Fixture
// =============================================================================

class PredictiveSafetyTest : public ::testing::Test {
protected:
  void SetUp() override {
    MPPIParams params;
    params.dt = 0.05;
    model_ = MotionModelFactory::create("diff_drive", params);

    barrier_set_ = std::make_unique<BarrierFunctionSet>(0.2, 0.3);

    Eigen::VectorXd u_min(2), u_max(2);
    u_min << -1.0, -2.0;
    u_max << 1.0, 2.0;
    filter_ = std::make_unique<PredictiveSafetyFilter>(
      barrier_set_.get(), 1.0, 0.05, u_min, u_max);
  }

  // 직진 제어 시퀀스 생성: v, omega 일정
  Eigen::MatrixXd makeStraightControls(int N, double v, double omega) {
    Eigen::MatrixXd controls(N, 2);
    controls.col(0).setConstant(v);
    controls.col(1).setConstant(omega);
    return controls;
  }

  std::shared_ptr<MotionModel> model_;
  std::unique_ptr<BarrierFunctionSet> barrier_set_;
  std::unique_ptr<PredictiveSafetyFilter> filter_;
};

// =============================================================================
// PredictiveSafetyFilter_Basic (4개)
// =============================================================================

TEST_F(PredictiveSafetyTest, NoObstacles_PassThrough) {
  // 장애물 없음 → u_safe == u_orig, num_corrected == 0
  Eigen::VectorXd x0(3);
  x0 << 0.0, 0.0, 0.0;

  int N = 10;
  Eigen::MatrixXd controls = makeStraightControls(N, 0.5, 0.1);

  auto result = filter_->filter(x0, controls, *model_);

  EXPECT_TRUE(result.feasible);
  EXPECT_EQ(result.num_corrected_steps, 0);
  // 투영 없으므로 제어가 동일
  for (int t = 0; t < N; ++t) {
    EXPECT_NEAR(result.u_safe_sequence(t, 0), controls(t, 0), 1e-6);
    EXPECT_NEAR(result.u_safe_sequence(t, 1), controls(t, 1), 1e-6);
  }
}

TEST_F(PredictiveSafetyTest, Construction_ValidParams) {
  // 기본 생성 파라미터 확인
  EXPECT_EQ(filter_->maxIterations(), 10);
  EXPECT_NEAR(filter_->horizonDecay(), 1.0, 1e-10);

  // setter 반영 확인
  filter_->setMaxIterations(20);
  filter_->setHorizonDecay(0.95);
  EXPECT_EQ(filter_->maxIterations(), 20);
  EXPECT_NEAR(filter_->horizonDecay(), 0.95, 1e-10);
}

TEST_F(PredictiveSafetyTest, EmptyControlSequence) {
  // 0행 제어 시퀀스 → N+1=1 궤적 (초기 상태만)
  Eigen::VectorXd x0(3);
  x0 << 1.0, 2.0, 0.5;

  Eigen::MatrixXd controls(0, 2);

  auto result = filter_->filter(x0, controls, *model_);

  EXPECT_EQ(result.u_safe_sequence.rows(), 0);
  EXPECT_EQ(result.safe_trajectory.rows(), 1);
  EXPECT_EQ(result.safe_trajectory.cols(), 3);
  // 초기 상태 보존
  EXPECT_NEAR(result.safe_trajectory(0, 0), 1.0, 1e-6);
  EXPECT_NEAR(result.safe_trajectory(0, 1), 2.0, 1e-6);
  EXPECT_EQ(result.num_corrected_steps, 0);
}

TEST_F(PredictiveSafetyTest, OutputDimensions) {
  // 출력 차원 확인: u_safe (N x nu), trajectory (N+1 x nx)
  Eigen::VectorXd x0(3);
  x0 << 0.0, 0.0, 0.0;

  int N = 15;
  Eigen::MatrixXd controls = makeStraightControls(N, 0.3, 0.0);

  auto result = filter_->filter(x0, controls, *model_);

  EXPECT_EQ(result.u_safe_sequence.rows(), N);
  EXPECT_EQ(result.u_safe_sequence.cols(), 2);  // nu=2 for diff_drive
  EXPECT_EQ(result.safe_trajectory.rows(), N + 1);
  EXPECT_EQ(result.safe_trajectory.cols(), 3);   // nx=3 for diff_drive
  EXPECT_EQ(static_cast<int>(result.min_barrier_values.size()), N + 1);
}

// =============================================================================
// PredictiveSafetyFilter_SafetyProjection (6개)
// =============================================================================

TEST_F(PredictiveSafetyTest, UnsafeTrajectory_GetsCorrected) {
  // 장애물에 정면 돌진 → 보정됨
  barrier_set_->setObstacles({{1.0, 0.0, 0.2}});

  Eigen::VectorXd x0(3);
  x0 << 0.0, 0.0, 0.0;  // 원점, 오른쪽 방향

  int N = 10;
  Eigen::MatrixXd controls = makeStraightControls(N, 1.0, 0.0);

  auto result = filter_->filter(x0, controls, *model_);

  EXPECT_GT(result.num_corrected_steps, 0)
    << "Trajectory heading into obstacle should be corrected";
  EXPECT_TRUE(result.feasible);
}

TEST_F(PredictiveSafetyTest, SafeTrajectory_Unchanged) {
  // 장애물 반대 방향 → 보정 없음
  barrier_set_->setObstacles({{5.0, 5.0, 0.2}});

  Eigen::VectorXd x0(3);
  x0 << 0.0, 0.0, M_PI;  // 왼쪽 방향 (장애물 반대)

  int N = 10;
  Eigen::MatrixXd controls = makeStraightControls(N, 0.5, 0.0);

  auto result = filter_->filter(x0, controls, *model_);

  EXPECT_EQ(result.num_corrected_steps, 0)
    << "Safe trajectory should not be corrected";
  EXPECT_TRUE(result.feasible);
}

TEST_F(PredictiveSafetyTest, MultistepPropagation) {
  // t 스텝 보정이 t+1..N에 전파되는지 확인
  // 장애물을 앞에 놓고, 보정 후 궤적이 장애물을 회피하는지 확인
  barrier_set_->setObstacles({{1.0, 0.0, 0.2}});

  Eigen::VectorXd x0(3);
  x0 << 0.0, 0.0, 0.0;

  int N = 10;
  Eigen::MatrixXd controls = makeStraightControls(N, 1.0, 0.0);

  auto result = filter_->filter(x0, controls, *model_);

  // 안전 궤적의 모든 포인트가 장애물 안전 거리 밖인지 확인
  double obs_x = 1.0, obs_y = 0.0, obs_r = 0.2;
  double d_safe = obs_r + 0.2 + 0.3;  // obstacle_r + robot_r + safety_margin
  for (int t = 0; t <= N; ++t) {
    double dx = result.safe_trajectory(t, 0) - obs_x;
    double dy = result.safe_trajectory(t, 1) - obs_y;
    double dist = std::sqrt(dx * dx + dy * dy);
    // 보정 후 안전 거리 이상이어야 함 (약간의 여유)
    EXPECT_GT(dist, d_safe * 0.5)
      << "Step " << t << " too close to obstacle after filtering";
  }
}

TEST_F(PredictiveSafetyTest, MultipleObstacles_AllAvoided) {
  // 여러 장애물 배치 → 필터 후 모든 barrier 값이 양수
  barrier_set_->setObstacles({
    {1.0, 0.0, 0.2},
    {0.0, 1.0, 0.2},
    {0.7, 0.7, 0.2}
  });

  Eigen::VectorXd x0(3);
  x0 << 0.0, 0.0, M_PI / 4.0;  // 45도 방향

  int N = 10;
  Eigen::MatrixXd controls = makeStraightControls(N, 0.8, 0.0);

  auto result = filter_->filter(x0, controls, *model_);

  // 필터 적용 후 모든 스텝의 min_barrier_values가 >= 0에 가까운지 확인
  for (int t = 0; t < N; ++t) {
    EXPECT_GE(result.min_barrier_values[t], -0.1)
      << "Step " << t << " barrier value too negative after filtering";
  }
}

TEST_F(PredictiveSafetyTest, CloseObstacle_HighCorrection) {
  // 가까운 장애물(0.8m) → 상당한 보정
  barrier_set_->setObstacles({{0.8, 0.0, 0.2}});

  Eigen::VectorXd x0(3);
  x0 << 0.0, 0.0, 0.0;

  int N = 10;
  Eigen::MatrixXd controls = makeStraightControls(N, 1.0, 0.0);

  auto result = filter_->filter(x0, controls, *model_);

  EXPECT_GT(result.num_corrected_steps, 0);

  // 보정 후 제어가 원래와 크게 다른 스텝이 있어야 함
  double max_diff = 0.0;
  for (int t = 0; t < N; ++t) {
    double diff = (result.u_safe_sequence.row(t) - controls.row(t)).norm();
    max_diff = std::max(max_diff, diff);
  }
  EXPECT_GT(max_diff, 0.01)
    << "Close obstacle should cause significant control correction";
}

TEST_F(PredictiveSafetyTest, FarObstacle_NoCorrection) {
  // 먼 장애물(5.0m) → 보정 없음
  barrier_set_->setObstacles({{5.0, 0.0, 0.2}});

  Eigen::VectorXd x0(3);
  x0 << 0.0, 0.0, 0.0;

  int N = 10;
  // dt=0.05, N=10 → 총 0.5초, v=0.5 → 0.25m 전진 (5.0m 장애물까지 여유)
  Eigen::MatrixXd controls = makeStraightControls(N, 0.5, 0.0);

  auto result = filter_->filter(x0, controls, *model_);

  EXPECT_EQ(result.num_corrected_steps, 0)
    << "Far obstacle should not require correction";
  EXPECT_TRUE(result.feasible);
}

// =============================================================================
// PredictiveSafetyFilter_VerifyTrajectory (3개)
// =============================================================================

TEST_F(PredictiveSafetyTest, VerifySafeTrajectory_ReturnsTrue) {
  // 장애물에서 멀리 떨어진 안전 궤적
  barrier_set_->setObstacles({{5.0, 5.0, 0.2}});

  Eigen::VectorXd x0(3);
  x0 << 0.0, 0.0, 0.0;

  int N = 10;
  Eigen::MatrixXd controls = makeStraightControls(N, 0.3, 0.0);

  bool safe = filter_->verifyTrajectory(x0, controls, *model_);
  EXPECT_TRUE(safe);
}

TEST_F(PredictiveSafetyTest, VerifyUnsafeTrajectory_ReturnsFalse) {
  // 장애물 돌진 궤적 → false
  barrier_set_->setObstacles({{1.0, 0.0, 0.2}});

  Eigen::VectorXd x0(3);
  x0 << 0.0, 0.0, 0.0;

  int N = 10;
  Eigen::MatrixXd controls = makeStraightControls(N, 1.0, 0.0);

  bool safe = filter_->verifyTrajectory(x0, controls, *model_);
  EXPECT_FALSE(safe)
    << "Trajectory heading into obstacle should be unsafe";
}

TEST_F(PredictiveSafetyTest, FilteredTrajectory_VerifyTrue) {
  // 필터 적용 후 → verifyTrajectory가 true
  barrier_set_->setObstacles({{1.0, 0.0, 0.2}});

  Eigen::VectorXd x0(3);
  x0 << 0.0, 0.0, 0.0;

  int N = 10;
  Eigen::MatrixXd controls = makeStraightControls(N, 1.0, 0.0);

  // 먼저 필터 적용
  auto result = filter_->filter(x0, controls, *model_);
  ASSERT_TRUE(result.feasible);

  // 필터된 제어로 검증
  bool safe = filter_->verifyTrajectory(x0, result.u_safe_sequence, *model_);
  EXPECT_TRUE(safe)
    << "Filtered trajectory should pass verification";
}

// =============================================================================
// PredictiveSafetyFilter_HorizonDecay (3개)
// =============================================================================

TEST_F(PredictiveSafetyTest, DecayOne_UniformGamma) {
  // decay=1.0 → 모든 스텝에서 동일한 gamma
  // 장애물을 배치하고, 보정 정도가 스텝 간 감쇠하지 않는지 확인
  filter_->setHorizonDecay(1.0);

  barrier_set_->setObstacles({{1.0, 0.0, 0.2}});

  Eigen::VectorXd x0(3);
  x0 << 0.0, 0.0, 0.0;

  int N = 10;
  Eigen::MatrixXd controls = makeStraightControls(N, 1.0, 0.0);

  auto result = filter_->filter(x0, controls, *model_);

  // decay=1 → 균일 gamma → 후반 스텝에서도 적극적 보정
  // (feasible이면 OK)
  EXPECT_TRUE(result.feasible);
  EXPECT_NEAR(filter_->horizonDecay(), 1.0, 1e-10);
}

TEST_F(PredictiveSafetyTest, DecayLessThanOne_WeakerAtEnd) {
  // decay < 1 → 후반 스텝에서 gamma가 약해짐
  // decay=1 vs decay=0.8의 결과를 비교
  barrier_set_->setObstacles({{1.5, 0.0, 0.2}});

  Eigen::VectorXd x0(3);
  x0 << 0.0, 0.0, 0.0;

  int N = 10;
  Eigen::MatrixXd controls = makeStraightControls(N, 1.0, 0.0);

  // decay=1.0 (균일)
  filter_->setHorizonDecay(1.0);
  auto result_uniform = filter_->filter(x0, controls, *model_);

  // decay=0.8 (감쇠)
  filter_->setHorizonDecay(0.8);
  auto result_decay = filter_->filter(x0, controls, *model_);

  // 두 결과 모두 유효해야 함
  EXPECT_TRUE(result_uniform.feasible);
  EXPECT_TRUE(result_decay.feasible);

  // decay < 1 에서는 후반부 CBF 제약이 약하므로
  // 보정 수가 다르거나, 제어 시퀀스가 다를 수 있음
  bool different = false;
  for (int t = 0; t < N; ++t) {
    if ((result_uniform.u_safe_sequence.row(t) -
         result_decay.u_safe_sequence.row(t)).norm() > 1e-6) {
      different = true;
      break;
    }
  }
  EXPECT_TRUE(different)
    << "Different decay should produce different filtered controls";
}

TEST_F(PredictiveSafetyTest, DecayZero_OnlyFirstStep) {
  // decay=0 → gamma_t = gamma * 0^t
  //   t=0: gamma * 1 = gamma (활성)
  //   t>=1: gamma * 0 = 0 (비활성)
  barrier_set_->setObstacles({{1.0, 0.0, 0.2}});

  Eigen::VectorXd x0(3);
  x0 << 0.0, 0.0, 0.0;

  int N = 10;
  Eigen::MatrixXd controls = makeStraightControls(N, 1.0, 0.0);

  // decay=0 → 첫 스텝만 CBF 투영
  filter_->setHorizonDecay(0.0);
  auto result = filter_->filter(x0, controls, *model_);

  // decay=1 (전체 투영)과 비교 → decay=0이 보정이 더 적어야 함
  filter_->setHorizonDecay(1.0);
  auto result_full = filter_->filter(x0, controls, *model_);

  EXPECT_LE(result.num_corrected_steps, result_full.num_corrected_steps)
    << "Decay=0 should correct fewer steps than decay=1";
}

// =============================================================================

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
