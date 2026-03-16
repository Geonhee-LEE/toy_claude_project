#ifndef MPC_CONTROLLER_ROS2__DYNAMIC_OBSTACLE_TRACKER_HPP_
#define MPC_CONTROLLER_ROS2__DYNAMIC_OBSTACLE_TRACKER_HPP_

#include <Eigen/Dense>
#include <vector>
#include <unordered_map>

namespace mpc_controller_ros2
{

/**
 * @brief 클러스터링된 장애물 정보
 */
struct ClusteredObstacle
{
  double cx{0.0};       // centroid x
  double cy{0.0};       // centroid y
  double radius{0.0};   // 반경 (max_dist + cell_radius)
  int cell_count{0};    // 셀 개수
};

/**
 * @brief 추적 중인 장애물 정보
 */
struct TrackedObstacle
{
  int id{0};            // 고유 ID
  double cx{0.0};       // centroid x
  double cy{0.0};       // centroid y
  double vx{0.0};       // 속도 x (m/s)
  double vy{0.0};       // 속도 y (m/s)
  double radius{0.0};   // 반경
  int age{0};           // 프레임 카운트
  double last_seen{0.0}; // 마지막 관측 시간
};

/**
 * @brief Grid-based Connected Components 클러스터러
 *
 * Costmap lethal 셀들을 그리드 기반으로 연결 컴포넌트 분석하여
 * 장애물 클러스터로 그룹핑합니다.
 *
 * 알고리즘:
 *   1. Grid bucketing: 셀 → bucket(floor(x/d), floor(y/d))
 *   2. Union-Find: 인접 8방향 bucket 연결
 *   3. 컴포넌트별: centroid, radius, count
 *   4. count < min_size 필터
 */
class ObstacleClusterer
{
public:
  /**
   * @brief 셀 좌표를 클러스터로 그룹핑
   * @param cells 2D 좌표 벡터 [(x, y), ...]
   * @param cell_radius 개별 셀 반경 (costmap resolution / 2)
   * @param cluster_distance 클러스터링 거리 임계값
   * @param min_cluster_size 최소 클러스터 크기
   * @return 클러스터 벡터
   */
  static std::vector<ClusteredObstacle> cluster(
    const std::vector<Eigen::Vector2d>& cells,
    double cell_radius,
    double cluster_distance = 0.15,
    int min_cluster_size = 3);

private:
  // Union-Find 유틸리티
  static int find(std::vector<int>& parent, int x);
  static void unite(std::vector<int>& parent, std::vector<int>& rank, int a, int b);
};

/**
 * @brief Greedy Nearest-Neighbor 트래커
 *
 * 프레임간 클러스터를 매칭하고 EMA로 속도를 추정합니다.
 *
 * 알고리즘:
 *   1. associate(): greedy nearest-neighbor 매칭
 *   2. Matched: 위치 갱신 + EMA velocity
 *   3. Unmatched clusters → 새 track
 *   4. Unmatched tracks: timeout → 삭제
 */
class ObstacleTracker
{
public:
  explicit ObstacleTracker(
    double ema_alpha = 0.3,
    double max_association_dist = 0.5,
    double track_timeout = 2.0);

  /**
   * @brief 클러스터 업데이트 + 속도 추정
   * @param clusters 현재 프레임 클러스터
   * @param timestamp 현재 시간 (초)
   */
  void update(const std::vector<ClusteredObstacle>& clusters, double timestamp);

  /**
   * @brief 추적 장애물을 setObstaclesWithVelocity() 형식으로 변환
   * @return (obstacles [x,y,r], velocities [vx,vy])
   */
  std::pair<std::vector<Eigen::Vector3d>, std::vector<Eigen::Vector2d>>
  toObstaclesWithVelocity() const;

  /**
   * @brief 현재 추적 장애물 목록 반환
   */
  const std::vector<TrackedObstacle>& getTracks() const { return tracks_; }

  /**
   * @brief 모든 트랙 초기화
   */
  void reset();

private:
  double ema_alpha_;
  double max_association_dist_;
  double track_timeout_;
  int next_id_{0};
  double last_timestamp_{-1.0};
  std::vector<TrackedObstacle> tracks_;
};

/**
 * @brief 통합 Dynamic Obstacle Tracker
 *
 * ObstacleClusterer + ObstacleTracker를 결합하여
 * costmap lethal 셀 → 추적 장애물 (위치 + 속도) 변환을 수행합니다.
 *
 * 흐름:
 *   lethal_cells → cluster() → update() → toObstaclesWithVelocity()
 *                                            → setObstaclesWithVelocity()
 */
class DynamicObstacleTracker
{
public:
  DynamicObstacleTracker(
    double cluster_distance = 0.15,
    int min_cluster_size = 3,
    double ema_alpha = 0.3,
    double max_association_dist = 0.5,
    double track_timeout = 2.0);

  /**
   * @brief 전체 파이프라인 실행
   * @param lethal_cells 2D 좌표 벡터
   * @param cell_radius 개별 셀 반경
   * @param timestamp 현재 시간 (초)
   * @return (obstacles [x,y,r], velocities [vx,vy])
   */
  std::pair<std::vector<Eigen::Vector3d>, std::vector<Eigen::Vector2d>>
  process(
    const std::vector<Eigen::Vector2d>& lethal_cells,
    double cell_radius,
    double timestamp);

  /**
   * @brief 상태 초기화
   */
  void reset();

  /**
   * @brief 현재 트래커 참조
   */
  const ObstacleTracker& tracker() const { return tracker_; }

private:
  double cluster_distance_;
  int min_cluster_size_;
  ObstacleTracker tracker_;
};

}  // namespace mpc_controller_ros2

#endif  // MPC_CONTROLLER_ROS2__DYNAMIC_OBSTACLE_TRACKER_HPP_
