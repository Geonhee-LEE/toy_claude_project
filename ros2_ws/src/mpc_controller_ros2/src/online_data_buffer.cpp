#include "mpc_controller_ros2/online_data_buffer.hpp"
#include <fstream>
#include <stdexcept>

namespace mpc_controller_ros2
{

OnlineDataBuffer::OnlineDataBuffer(size_t capacity)
: capacity_(std::max(capacity, size_t(1)))
{
  buffer_.resize(capacity_);
}

void OnlineDataBuffer::add(
  const Eigen::VectorXd& state,
  const Eigen::VectorXd& control,
  const Eigen::VectorXd& next_state,
  double dt)
{
  std::lock_guard<std::mutex> lock(mutex_);
  buffer_[write_idx_] = {state, control, next_state, dt};
  write_idx_ = (write_idx_ + 1) % capacity_;
  if (count_ < capacity_) {
    ++count_;
  }
}

size_t OnlineDataBuffer::exportCSV(const std::string& path) const
{
  std::lock_guard<std::mutex> lock(mutex_);

  std::ofstream ofs(path);
  if (!ofs.is_open()) {
    throw std::runtime_error("OnlineDataBuffer::exportCSV: cannot open " + path);
  }

  // 헤더 생성 (첫 번째 데이터 포인트 기준)
  if (count_ == 0) {
    return 0;
  }

  // 첫 유효 데이터로 차원 결정
  size_t start = (count_ < capacity_) ? 0 : write_idx_;
  const auto& first = buffer_[start];
  int nx = first.state.size();
  int nu = first.control.size();

  // 헤더
  for (int j = 0; j < nx; ++j) ofs << "s" << j << ",";
  for (int j = 0; j < nu; ++j) ofs << "u" << j << ",";
  for (int j = 0; j < nx; ++j) ofs << "ns" << j << ",";
  ofs << "dt\n";

  // 데이터 (링 버퍼 순서대로)
  for (size_t i = 0; i < count_; ++i) {
    size_t idx = (start + i) % capacity_;
    const auto& dp = buffer_[idx];

    for (int j = 0; j < dp.state.size(); ++j) ofs << dp.state(j) << ",";
    for (int j = 0; j < dp.control.size(); ++j) ofs << dp.control(j) << ",";
    for (int j = 0; j < dp.next_state.size(); ++j) ofs << dp.next_state(j) << ",";
    ofs << dp.dt << "\n";
  }

  return count_;
}

void OnlineDataBuffer::clear()
{
  std::lock_guard<std::mutex> lock(mutex_);
  write_idx_ = 0;
  count_ = 0;
}

size_t OnlineDataBuffer::size() const
{
  std::lock_guard<std::mutex> lock(mutex_);
  return count_;
}

bool OnlineDataBuffer::full() const
{
  std::lock_guard<std::mutex> lock(mutex_);
  return count_ >= capacity_;
}

}  // namespace mpc_controller_ros2
