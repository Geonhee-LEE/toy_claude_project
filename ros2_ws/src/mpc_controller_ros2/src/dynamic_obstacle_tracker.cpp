#include "mpc_controller_ros2/dynamic_obstacle_tracker.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <unordered_map>

namespace mpc_controller_ros2
{

// =============================================================================
// Union-Find
// =============================================================================

int ObstacleClusterer::find(std::vector<int>& parent, int x)
{
  while (parent[x] != x) {
    parent[x] = parent[parent[x]];  // path compression
    x = parent[x];
  }
  return x;
}

void ObstacleClusterer::unite(
  std::vector<int>& parent, std::vector<int>& rank, int a, int b)
{
  a = find(parent, a);
  b = find(parent, b);
  if (a == b) { return; }
  if (rank[a] < rank[b]) { std::swap(a, b); }
  parent[b] = a;
  if (rank[a] == rank[b]) { ++rank[a]; }
}

// =============================================================================
// ObstacleClusterer::cluster()
// =============================================================================

std::vector<ClusteredObstacle> ObstacleClusterer::cluster(
  const std::vector<Eigen::Vector2d>& cells,
  double cell_radius,
  double cluster_distance,
  int min_cluster_size)
{
  if (cells.empty()) {
    return {};
  }

  int n = static_cast<int>(cells.size());

  // Grid bucketing
  double d = cluster_distance;
  if (d <= 0.0) { d = 0.15; }

  struct GridKey
  {
    int gx, gy;
    bool operator==(const GridKey& o) const { return gx == o.gx && gy == o.gy; }
  };
  struct GridHash
  {
    size_t operator()(const GridKey& k) const
    {
      return std::hash<int>()(k.gx) ^ (std::hash<int>()(k.gy) << 16);
    }
  };

  // cell → bucket 인덱스
  std::unordered_map<GridKey, std::vector<int>, GridHash> buckets;
  std::vector<GridKey> cell_keys(n);

  for (int i = 0; i < n; ++i) {
    int gx = static_cast<int>(std::floor(cells[i](0) / d));
    int gy = static_cast<int>(std::floor(cells[i](1) / d));
    cell_keys[i] = {gx, gy};
    buckets[{gx, gy}].push_back(i);
  }

  // Union-Find 초기화
  std::vector<int> parent(n), rank_vec(n, 0);
  for (int i = 0; i < n; ++i) { parent[i] = i; }

  // 인접 8방향 연결
  for (auto& [key, indices] : buckets) {
    // 같은 bucket 내 연결
    for (size_t i = 1; i < indices.size(); ++i) {
      unite(parent, rank_vec, indices[0], indices[i]);
    }

    // 인접 bucket 연결
    for (int dx = -1; dx <= 1; ++dx) {
      for (int dy = -1; dy <= 1; ++dy) {
        if (dx == 0 && dy == 0) { continue; }
        GridKey neighbor_key{key.gx + dx, key.gy + dy};
        auto it = buckets.find(neighbor_key);
        if (it != buckets.end() && !it->second.empty()) {
          unite(parent, rank_vec, indices[0], it->second[0]);
        }
      }
    }
  }

  // 컴포넌트별 집계
  std::unordered_map<int, std::vector<int>> components;
  for (int i = 0; i < n; ++i) {
    components[find(parent, i)].push_back(i);
  }

  // 클러스터 생성
  std::vector<ClusteredObstacle> result;
  for (auto& [root, indices] : components) {
    int count = static_cast<int>(indices.size());
    if (count < min_cluster_size) { continue; }

    // Centroid
    double sum_x = 0.0, sum_y = 0.0;
    for (int idx : indices) {
      sum_x += cells[idx](0);
      sum_y += cells[idx](1);
    }
    double cx = sum_x / count;
    double cy = sum_y / count;

    // Radius = max distance from centroid + cell_radius
    double max_dist = 0.0;
    for (int idx : indices) {
      double dist = std::sqrt(
        (cells[idx](0) - cx) * (cells[idx](0) - cx) +
        (cells[idx](1) - cy) * (cells[idx](1) - cy));
      max_dist = std::max(max_dist, dist);
    }

    ClusteredObstacle obs;
    obs.cx = cx;
    obs.cy = cy;
    obs.radius = max_dist + cell_radius;
    obs.cell_count = count;
    result.push_back(obs);
  }

  return result;
}

// =============================================================================
// ObstacleTracker
// =============================================================================

ObstacleTracker::ObstacleTracker(
  double ema_alpha,
  double max_association_dist,
  double track_timeout)
: ema_alpha_(ema_alpha),
  max_association_dist_(max_association_dist),
  track_timeout_(track_timeout)
{
}

void ObstacleTracker::update(
  const std::vector<ClusteredObstacle>& clusters,
  double timestamp)
{
  double dt = (last_timestamp_ >= 0.0) ? (timestamp - last_timestamp_) : 0.0;
  last_timestamp_ = timestamp;

  int n_tracks = static_cast<int>(tracks_.size());
  int n_clusters = static_cast<int>(clusters.size());

  // Greedy nearest-neighbor association
  std::vector<int> track_to_cluster(n_tracks, -1);
  std::vector<int> cluster_to_track(n_clusters, -1);
  std::vector<bool> cluster_matched(n_clusters, false);
  std::vector<bool> track_matched(n_tracks, false);

  // 모든 (track, cluster) 거리 계산 → 정렬 → greedy 매칭
  struct DistPair { double dist; int track_idx; int cluster_idx; };
  std::vector<DistPair> pairs;
  pairs.reserve(n_tracks * n_clusters);

  for (int t = 0; t < n_tracks; ++t) {
    for (int c = 0; c < n_clusters; ++c) {
      double dx = tracks_[t].cx - clusters[c].cx;
      double dy = tracks_[t].cy - clusters[c].cy;
      double dist = std::sqrt(dx * dx + dy * dy);
      if (dist <= max_association_dist_) {
        pairs.push_back({dist, t, c});
      }
    }
  }

  std::sort(pairs.begin(), pairs.end(),
    [](const DistPair& a, const DistPair& b) { return a.dist < b.dist; });

  for (const auto& p : pairs) {
    if (track_matched[p.track_idx] || cluster_matched[p.cluster_idx]) {
      continue;
    }
    track_matched[p.track_idx] = true;
    cluster_matched[p.cluster_idx] = true;
    track_to_cluster[p.track_idx] = p.cluster_idx;
    cluster_to_track[p.cluster_idx] = p.track_idx;
  }

  // Matched tracks: 위치 갱신 + EMA velocity
  for (int t = 0; t < n_tracks; ++t) {
    if (!track_matched[t]) { continue; }
    int c = track_to_cluster[t];
    const auto& cluster = clusters[c];

    if (dt > 1e-6) {
      double raw_vx = (cluster.cx - tracks_[t].cx) / dt;
      double raw_vy = (cluster.cy - tracks_[t].cy) / dt;
      tracks_[t].vx = ema_alpha_ * raw_vx + (1.0 - ema_alpha_) * tracks_[t].vx;
      tracks_[t].vy = ema_alpha_ * raw_vy + (1.0 - ema_alpha_) * tracks_[t].vy;
    }

    tracks_[t].cx = cluster.cx;
    tracks_[t].cy = cluster.cy;
    tracks_[t].radius = cluster.radius;
    tracks_[t].age++;
    tracks_[t].last_seen = timestamp;
  }

  // Unmatched clusters → 새 track
  for (int c = 0; c < n_clusters; ++c) {
    if (cluster_matched[c]) { continue; }
    TrackedObstacle track;
    track.id = next_id_++;
    track.cx = clusters[c].cx;
    track.cy = clusters[c].cy;
    track.vx = 0.0;
    track.vy = 0.0;
    track.radius = clusters[c].radius;
    track.age = 1;
    track.last_seen = timestamp;
    tracks_.push_back(track);
  }

  // Unmatched tracks: timeout → 삭제
  tracks_.erase(
    std::remove_if(tracks_.begin(), tracks_.end(),
      [timestamp, this](const TrackedObstacle& t) {
        return (timestamp - t.last_seen) > track_timeout_;
      }),
    tracks_.end());
}

std::pair<std::vector<Eigen::Vector3d>, std::vector<Eigen::Vector2d>>
ObstacleTracker::toObstaclesWithVelocity() const
{
  std::vector<Eigen::Vector3d> obstacles;
  std::vector<Eigen::Vector2d> velocities;
  obstacles.reserve(tracks_.size());
  velocities.reserve(tracks_.size());

  for (const auto& track : tracks_) {
    obstacles.emplace_back(track.cx, track.cy, track.radius);
    velocities.emplace_back(track.vx, track.vy);
  }

  return {obstacles, velocities};
}

void ObstacleTracker::reset()
{
  tracks_.clear();
  next_id_ = 0;
  last_timestamp_ = -1.0;
}

// =============================================================================
// DynamicObstacleTracker
// =============================================================================

DynamicObstacleTracker::DynamicObstacleTracker(
  double cluster_distance,
  int min_cluster_size,
  double ema_alpha,
  double max_association_dist,
  double track_timeout)
: cluster_distance_(cluster_distance),
  min_cluster_size_(min_cluster_size),
  tracker_(ema_alpha, max_association_dist, track_timeout)
{
}

std::pair<std::vector<Eigen::Vector3d>, std::vector<Eigen::Vector2d>>
DynamicObstacleTracker::process(
  const std::vector<Eigen::Vector2d>& lethal_cells,
  double cell_radius,
  double timestamp)
{
  auto clusters = ObstacleClusterer::cluster(
    lethal_cells, cell_radius, cluster_distance_, min_cluster_size_);

  tracker_.update(clusters, timestamp);

  return tracker_.toObstaclesWithVelocity();
}

void DynamicObstacleTracker::reset()
{
  tracker_.reset();
}

}  // namespace mpc_controller_ros2
