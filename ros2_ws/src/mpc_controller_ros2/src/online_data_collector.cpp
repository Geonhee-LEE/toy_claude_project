#include "mpc_controller_ros2/online_data_collector.hpp"

namespace mpc_controller_ros2
{

OnlineDataCollector::OnlineDataCollector(
  OnlineDataBuffer* buffer,
  const MotionModel* model,
  int state_dim, int control_dim)
: buffer_(buffer),
  model_(model),
  nx_(state_dim),
  nu_(control_dim)
{
}

void OnlineDataCollector::collect(
  const Eigen::VectorXd& prev_state,
  const Eigen::VectorXd& prev_control,
  const Eigen::VectorXd& curr_state,
  double dt)
{
  if (!enabled_) {
    return;
  }

  // 차원 검증 (불일치 시 silent skip)
  if (prev_state.size() != nx_ ||
      prev_control.size() != nu_ ||
      curr_state.size() != nx_) {
    return;
  }

  buffer_->add(prev_state, prev_control, curr_state, dt);
  total_collected_++;
}

OnlineDataCollector::Stats OnlineDataCollector::getStats() const
{
  Stats stats;
  stats.total_collected = total_collected_;
  stats.buffer_size = buffer_->size();
  stats.buffer_full = buffer_->full();
  return stats;
}

}  // namespace mpc_controller_ros2
