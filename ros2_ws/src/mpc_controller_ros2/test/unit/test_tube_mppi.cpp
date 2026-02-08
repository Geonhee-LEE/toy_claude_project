#include <gtest/gtest.h>
#include "mpc_controller_ros2/tube_mppi.hpp"
#include "mpc_controller_ros2/ancillary_controller.hpp"
#include <cmath>

using namespace mpc_controller_ros2;

class AncillaryControllerTest : public ::testing::Test
{
protected:
  void SetUp() override
  {
    controller_ = std::make_unique<AncillaryController>(0.8, 0.5, 1.0);
  }

  std::unique_ptr<AncillaryController> controller_;
};

TEST_F(AncillaryControllerTest, BodyFrameErrorForward)
{
  // 로봇이 원점에 있고 전방(x+)을 바라볼 때
  // nominal이 (1, 0)에 있으면 e_forward = 1
  Eigen::Vector3d actual(0.0, 0.0, 0.0);
  Eigen::Vector3d nominal(1.0, 0.0, 0.0);

  auto error = controller_->computeBodyFrameError(nominal, actual);

  EXPECT_NEAR(error(0), 1.0, 1e-6);   // e_forward
  EXPECT_NEAR(error(1), 0.0, 1e-6);   // e_lateral
  EXPECT_NEAR(error(2), 0.0, 1e-6);   // e_angle
}

TEST_F(AncillaryControllerTest, BodyFrameErrorLateral)
{
  // 로봇이 원점에 있고 전방을 바라볼 때
  // nominal이 (0, 1)에 있으면 e_lateral = 1 (왼쪽)
  Eigen::Vector3d actual(0.0, 0.0, 0.0);
  Eigen::Vector3d nominal(0.0, 1.0, 0.0);

  auto error = controller_->computeBodyFrameError(nominal, actual);

  EXPECT_NEAR(error(0), 0.0, 1e-6);   // e_forward
  EXPECT_NEAR(error(1), 1.0, 1e-6);   // e_lateral
  EXPECT_NEAR(error(2), 0.0, 1e-6);   // e_angle
}

TEST_F(AncillaryControllerTest, BodyFrameErrorWithRotation)
{
  // 로봇이 90도 회전한 상태 (y+ 방향을 바라봄)
  Eigen::Vector3d actual(0.0, 0.0, M_PI / 2);
  Eigen::Vector3d nominal(0.0, 1.0, M_PI / 2);

  auto error = controller_->computeBodyFrameError(nominal, actual);

  // nominal이 로봇 전방에 있음
  EXPECT_NEAR(error(0), 1.0, 1e-6);   // e_forward
  EXPECT_NEAR(error(1), 0.0, 1e-6);   // e_lateral
  EXPECT_NEAR(error(2), 0.0, 1e-6);   // e_angle
}

TEST_F(AncillaryControllerTest, AngleError)
{
  // 각도 오차만 있는 경우
  Eigen::Vector3d actual(0.0, 0.0, 0.0);
  Eigen::Vector3d nominal(0.0, 0.0, 0.5);

  auto error = controller_->computeBodyFrameError(nominal, actual);

  EXPECT_NEAR(error(0), 0.0, 1e-6);
  EXPECT_NEAR(error(1), 0.0, 1e-6);
  EXPECT_NEAR(error(2), 0.5, 1e-6);  // e_angle = 0.5 rad
}

TEST_F(AncillaryControllerTest, FeedbackCorrection)
{
  // K_fb = [0.8, 0, 0; 0, 0.5, 1.0]
  Eigen::Vector3d body_error(1.0, 0.5, 0.3);

  auto correction = controller_->computeFeedbackCorrection(body_error);

  // dv = 0.8 * 1.0 = 0.8
  // dω = 0.5 * 0.5 + 1.0 * 0.3 = 0.55
  EXPECT_NEAR(correction(0), 0.8, 1e-6);
  EXPECT_NEAR(correction(1), 0.55, 1e-6);
}

TEST_F(AncillaryControllerTest, CorrectedControl)
{
  Eigen::Vector2d nominal_control(0.5, 0.2);
  Eigen::Vector3d nominal_state(1.0, 0.0, 0.0);
  Eigen::Vector3d actual_state(0.0, 0.0, 0.0);

  auto corrected = controller_->computeCorrectedControl(
    nominal_control, nominal_state, actual_state
  );

  // Body error: e_forward=1.0, e_lateral=0, e_angle=0
  // Correction: dv=0.8, dω=0
  EXPECT_NEAR(corrected(0), 0.5 + 0.8, 1e-6);  // v + dv
  EXPECT_NEAR(corrected(1), 0.2, 1e-6);         // ω (no correction)
}

TEST_F(AncillaryControllerTest, AngleNormalization)
{
  // 각도 정규화 테스트
  EXPECT_NEAR(AncillaryController::normalizeAngle(0.0), 0.0, 1e-6);
  EXPECT_NEAR(AncillaryController::normalizeAngle(M_PI), M_PI, 1e-6);
  EXPECT_NEAR(AncillaryController::normalizeAngle(-M_PI), -M_PI, 1e-6);
  EXPECT_NEAR(AncillaryController::normalizeAngle(2 * M_PI), 0.0, 1e-6);
  EXPECT_NEAR(AncillaryController::normalizeAngle(3 * M_PI), M_PI, 1e-6);
  EXPECT_NEAR(AncillaryController::normalizeAngle(-3 * M_PI), -M_PI, 1e-6);
}

// ============================================================================
// TubeMPPI Tests
// ============================================================================

class TubeMPPITest : public ::testing::Test
{
protected:
  void SetUp() override
  {
    MPPIParams params;
    params.tube_enabled = true;
    params.tube_width = 0.5;
    params.k_forward = 0.8;
    params.k_lateral = 0.5;
    params.k_angle = 1.0;

    tube_mppi_ = std::make_unique<TubeMPPI>(params);
  }

  std::unique_ptr<TubeMPPI> tube_mppi_;
};

TEST_F(TubeMPPITest, Initialization)
{
  EXPECT_DOUBLE_EQ(tube_mppi_->getTubeWidth(), 0.5);
}

TEST_F(TubeMPPITest, ComputeCorrectedControl)
{
  Eigen::Vector2d nominal_control(0.5, 0.1);
  Eigen::MatrixXd nominal_trajectory(3, 3);
  nominal_trajectory << 1.0, 0.0, 0.0,
                        1.5, 0.0, 0.0,
                        2.0, 0.0, 0.0;
  Eigen::Vector3d actual_state(0.0, 0.0, 0.0);

  auto [corrected, info] = tube_mppi_->computeCorrectedControl(
    nominal_control, nominal_trajectory, actual_state
  );

  // nominal_state = (1.0, 0.0, 0.0), actual = (0.0, 0.0, 0.0)
  // e_forward = 1.0, correction dv = 0.8
  EXPECT_NEAR(info.body_error(0), 1.0, 1e-6);
  EXPECT_NEAR(info.feedback_correction(0), 0.8, 1e-6);
  EXPECT_NEAR(corrected(0), 0.5 + 0.8, 1e-6);
}

TEST_F(TubeMPPITest, IsInsideTube)
{
  Eigen::Vector3d nominal(0.0, 0.0, 0.0);

  // 정확히 nominal 위치
  EXPECT_TRUE(tube_mppi_->isInsideTube(nominal, Eigen::Vector3d(0.0, 0.0, 0.1)));

  // tube 폭 내부
  EXPECT_TRUE(tube_mppi_->isInsideTube(nominal, Eigen::Vector3d(0.3, 0.3, 0.0)));

  // tube 폭 외부
  EXPECT_FALSE(tube_mppi_->isInsideTube(nominal, Eigen::Vector3d(1.0, 0.0, 0.0)));
}

TEST_F(TubeMPPITest, TubeBoundaryComputation)
{
  Eigen::MatrixXd trajectory(3, 3);
  trajectory << 0.0, 0.0, 0.0,
                1.0, 0.0, 0.0,
                2.0, 0.0, 0.0;

  auto boundaries = tube_mppi_->computeTubeBoundary(trajectory);

  EXPECT_EQ(boundaries.size(), 3);

  // 첫 번째 점 (theta=0, tube_width=0.5)
  // 좌측: (0, 0.5), 우측: (0, -0.5)
  EXPECT_NEAR(boundaries[0].first(0), 0.0, 1e-6);
  EXPECT_NEAR(boundaries[0].first(1), 0.5, 1e-6);
  EXPECT_NEAR(boundaries[0].second(0), 0.0, 1e-6);
  EXPECT_NEAR(boundaries[0].second(1), -0.5, 1e-6);
}

TEST_F(TubeMPPITest, TubeWidthUpdate)
{
  double initial_width = tube_mppi_->getTubeWidth();

  // 큰 추적 오차 → tube 확장
  tube_mppi_->updateTubeWidth(1.0);
  EXPECT_GT(tube_mppi_->getTubeWidth(), initial_width);

  // 작은 추적 오차 → tube 축소
  for (int i = 0; i < 20; ++i) {
    tube_mppi_->updateTubeWidth(0.1);
  }
  EXPECT_LT(tube_mppi_->getTubeWidth(), 0.5);
}

TEST_F(TubeMPPITest, SetFeedbackGains)
{
  tube_mppi_->setFeedbackGains(1.0, 0.8, 1.5);

  auto gains = tube_mppi_->getAncillaryController().getGains();

  EXPECT_NEAR(gains(0, 0), 1.0, 1e-6);  // k_forward
  EXPECT_NEAR(gains(1, 1), 0.8, 1e-6);  // k_lateral
  EXPECT_NEAR(gains(1, 2), 1.5, 1e-6);  // k_angle
}

int main(int argc, char** argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
