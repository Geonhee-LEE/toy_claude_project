#include <gtest/gtest.h>
#include <Eigen/Dense>
#include <cmath>

#include "mpc_controller_ros2/dynamic_obstacle_tracker.hpp"

using namespace mpc_controller_ros2;

// =============================================================================
// ObstacleClusterer 테스트
// =============================================================================

// 1. SingleCluster — 인접 5셀 → 1 cluster
TEST(ObstacleClusterer, SingleCluster)
{
  std::vector<Eigen::Vector2d> cells = {
    {0.0, 0.0}, {0.05, 0.0}, {0.1, 0.0}, {0.0, 0.05}, {0.05, 0.05}
  };

  auto clusters = ObstacleClusterer::cluster(cells, 0.025, 0.15, 3);

  EXPECT_EQ(static_cast<int>(clusters.size()), 1);
  EXPECT_EQ(clusters[0].cell_count, 5);
}

// 2. TwoClusters — 분리된 2그룹 → 2 cluster
TEST(ObstacleClusterer, TwoClusters)
{
  std::vector<Eigen::Vector2d> cells;

  // 그룹 1: (0, 0) 근처
  for (int i = 0; i < 5; ++i) {
    cells.emplace_back(0.0 + i * 0.05, 0.0);
  }
  // 그룹 2: (5, 5) 근처 — 충분히 먼 거리
  for (int i = 0; i < 4; ++i) {
    cells.emplace_back(5.0 + i * 0.05, 5.0);
  }

  auto clusters = ObstacleClusterer::cluster(cells, 0.025, 0.15, 3);

  EXPECT_EQ(static_cast<int>(clusters.size()), 2);
}

// 3. MinSizeFilter — count < min → 필터
TEST(ObstacleClusterer, MinSizeFilter)
{
  std::vector<Eigen::Vector2d> cells = {
    {0.0, 0.0}, {0.05, 0.0}  // 2셀 → min_cluster_size=3보다 작음
  };

  auto clusters = ObstacleClusterer::cluster(cells, 0.025, 0.15, 3);

  EXPECT_EQ(static_cast<int>(clusters.size()), 0);
}

// 4. CentroidAccuracy — centroid = 평균 위치
TEST(ObstacleClusterer, CentroidAccuracy)
{
  std::vector<Eigen::Vector2d> cells = {
    {1.0, 2.0}, {1.1, 2.0}, {1.0, 2.1}
  };

  auto clusters = ObstacleClusterer::cluster(cells, 0.025, 0.15, 3);

  ASSERT_EQ(static_cast<int>(clusters.size()), 1);
  double expected_cx = (1.0 + 1.1 + 1.0) / 3.0;
  double expected_cy = (2.0 + 2.0 + 2.1) / 3.0;
  EXPECT_NEAR(clusters[0].cx, expected_cx, 1e-6);
  EXPECT_NEAR(clusters[0].cy, expected_cy, 1e-6);
}

// 5. RadiusComputation — max_dist + cell_radius
TEST(ObstacleClusterer, RadiusComputation)
{
  std::vector<Eigen::Vector2d> cells = {
    {0.0, 0.0}, {0.1, 0.0}, {0.0, 0.1}
  };
  double cell_radius = 0.025;

  auto clusters = ObstacleClusterer::cluster(cells, cell_radius, 0.15, 3);

  ASSERT_EQ(static_cast<int>(clusters.size()), 1);
  // centroid ≈ (0.033, 0.033), max_dist from centroid ≈ 0.094
  EXPECT_GT(clusters[0].radius, cell_radius);
  EXPECT_LT(clusters[0].radius, 0.2);  // 합리적 범위
}

// 6. EmptyInput — 0셀 → 0 cluster
TEST(ObstacleClusterer, EmptyInput)
{
  std::vector<Eigen::Vector2d> cells;

  auto clusters = ObstacleClusterer::cluster(cells, 0.025, 0.15, 3);

  EXPECT_EQ(static_cast<int>(clusters.size()), 0);
}

// =============================================================================
// ObstacleTracker 테스트
// =============================================================================

// 7. NewTrackCreation — 첫 프레임 track 생성
TEST(ObstacleTracker, NewTrackCreation)
{
  ObstacleTracker tracker(0.3, 0.5, 2.0);

  std::vector<ClusteredObstacle> clusters = {
    {1.0, 2.0, 0.1, 5},
    {3.0, 4.0, 0.2, 8}
  };

  tracker.update(clusters, 0.0);

  EXPECT_EQ(static_cast<int>(tracker.getTracks().size()), 2);
  EXPECT_NEAR(tracker.getTracks()[0].cx, 1.0, 1e-6);
  EXPECT_NEAR(tracker.getTracks()[1].cx, 3.0, 1e-6);
}

// 8. TrackAssociation — 같은 위치 → 같은 ID
TEST(ObstacleTracker, TrackAssociation)
{
  ObstacleTracker tracker(0.3, 0.5, 2.0);

  // Frame 1
  std::vector<ClusteredObstacle> clusters1 = {{1.0, 2.0, 0.1, 5}};
  tracker.update(clusters1, 0.0);
  int first_id = tracker.getTracks()[0].id;

  // Frame 2: 약간 이동
  std::vector<ClusteredObstacle> clusters2 = {{1.05, 2.02, 0.1, 5}};
  tracker.update(clusters2, 0.1);

  EXPECT_EQ(static_cast<int>(tracker.getTracks().size()), 1);
  EXPECT_EQ(tracker.getTracks()[0].id, first_id);
}

// 9. VelocityEstimation — Δx/Δt ≈ vx
TEST(ObstacleTracker, VelocityEstimation)
{
  ObstacleTracker tracker(1.0, 0.5, 2.0);  // ema_alpha=1.0 (raw 그대로)

  // Frame 1: (1.0, 0.0)
  std::vector<ClusteredObstacle> c1 = {{1.0, 0.0, 0.1, 5}};
  tracker.update(c1, 0.0);

  // Frame 2: (1.1, 0.0), dt=0.1 → vx = 0.1/0.1 = 1.0
  std::vector<ClusteredObstacle> c2 = {{1.1, 0.0, 0.1, 5}};
  tracker.update(c2, 0.1);

  EXPECT_NEAR(tracker.getTracks()[0].vx, 1.0, 0.01);
  EXPECT_NEAR(tracker.getTracks()[0].vy, 0.0, 0.01);
}

// 10. VelocityEMA — 3프레임 EMA 스무딩
TEST(ObstacleTracker, VelocityEMA)
{
  double alpha = 0.3;
  ObstacleTracker tracker(alpha, 0.5, 2.0);

  // Frame 1: (0, 0)
  tracker.update({{0.0, 0.0, 0.1, 5}}, 0.0);

  // Frame 2: (0.1, 0), dt=0.1 → raw vx = 1.0 → ema = 0.3*1.0 + 0.7*0 = 0.3
  tracker.update({{0.1, 0.0, 0.1, 5}}, 0.1);
  double v1 = tracker.getTracks()[0].vx;
  EXPECT_NEAR(v1, alpha * 1.0, 0.01);

  // Frame 3: (0.2, 0), dt=0.1 → raw vx = 1.0 → ema = 0.3*1.0 + 0.7*0.3 = 0.51
  tracker.update({{0.2, 0.0, 0.1, 5}}, 0.2);
  double v2 = tracker.getTracks()[0].vx;
  double expected = alpha * 1.0 + (1.0 - alpha) * v1;
  EXPECT_NEAR(v2, expected, 0.01);
}

// 11. TrackTimeout — timeout 후 삭제
TEST(ObstacleTracker, TrackTimeout)
{
  ObstacleTracker tracker(0.3, 0.5, 1.0);  // timeout = 1초

  tracker.update({{1.0, 0.0, 0.1, 5}}, 0.0);
  EXPECT_EQ(static_cast<int>(tracker.getTracks().size()), 1);

  // 2초 후 빈 업데이트 → 타임아웃
  tracker.update({}, 2.5);
  EXPECT_EQ(static_cast<int>(tracker.getTracks().size()), 0);
}

// 12. MaxAssocDistance — 먼 cluster → 새 track
TEST(ObstacleTracker, MaxAssocDistance)
{
  ObstacleTracker tracker(0.3, 0.3, 2.0);  // max_dist = 0.3

  tracker.update({{0.0, 0.0, 0.1, 5}}, 0.0);

  // 1.0m 떨어진 곳 → 매칭 안 됨 → 새 track + 기존 유지
  tracker.update({{0.0, 0.0, 0.1, 5}, {1.0, 1.0, 0.1, 5}}, 0.1);
  EXPECT_EQ(static_cast<int>(tracker.getTracks().size()), 2);
}

// =============================================================================
// DynamicObstacleTracker (통합) 테스트
// =============================================================================

// 13. ProcessEndToEnd — cells → tracked → (obs, vels)
TEST(DynamicObstacleTracker, ProcessEndToEnd)
{
  DynamicObstacleTracker tracker(0.15, 3, 0.3, 0.5, 2.0);

  // 5 인접 셀
  std::vector<Eigen::Vector2d> cells = {
    {1.0, 2.0}, {1.05, 2.0}, {1.1, 2.0},
    {1.0, 2.05}, {1.05, 2.05}
  };

  auto [obs, vels] = tracker.process(cells, 0.025, 0.0);

  EXPECT_EQ(obs.size(), vels.size());
  EXPECT_GE(static_cast<int>(obs.size()), 1);
}

// 14. OutputFormat — Vector3d + Vector2d 형식
TEST(DynamicObstacleTracker, OutputFormat)
{
  DynamicObstacleTracker tracker(0.15, 3, 0.3, 0.5, 2.0);

  std::vector<Eigen::Vector2d> cells = {
    {0.0, 0.0}, {0.05, 0.0}, {0.1, 0.0}
  };

  auto [obs, vels] = tracker.process(cells, 0.025, 0.0);

  ASSERT_GE(static_cast<int>(obs.size()), 1);
  // Vector3d: [x, y, radius]
  EXPECT_TRUE(std::isfinite(obs[0](0)));
  EXPECT_TRUE(std::isfinite(obs[0](1)));
  EXPECT_GT(obs[0](2), 0.0);  // radius > 0

  // Vector2d: [vx, vy]
  EXPECT_TRUE(std::isfinite(vels[0](0)));
  EXPECT_TRUE(std::isfinite(vels[0](1)));
}

// 15. Reset — reset() 후 빈 상태
TEST(DynamicObstacleTracker, Reset)
{
  DynamicObstacleTracker tracker(0.15, 3, 0.3, 0.5, 2.0);

  std::vector<Eigen::Vector2d> cells = {
    {0.0, 0.0}, {0.05, 0.0}, {0.1, 0.0}
  };

  tracker.process(cells, 0.025, 0.0);
  EXPECT_GE(static_cast<int>(tracker.tracker().getTracks().size()), 1);

  tracker.reset();
  EXPECT_EQ(static_cast<int>(tracker.tracker().getTracks().size()), 0);
}

int main(int argc, char** argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
