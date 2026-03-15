#include "mpc_controller_ros2/model_reloader.hpp"
#include <stdexcept>

namespace mpc_controller_ros2
{

ModelReloader::ModelReloader(const std::string& weights_dir, int ensemble_size)
: weights_dir_(weights_dir),
  ensemble_size_(ensemble_size),
  last_timestamps_(ensemble_size, std::filesystem::file_time_type::min()),
  last_reload_time_(std::chrono::steady_clock::time_point::min())
{
  if (ensemble_size_ <= 0) {
    throw std::invalid_argument("ModelReloader: ensemble_size must be > 0");
  }
}

std::string ModelReloader::modelPath(int i) const
{
  return weights_dir_ + "/model_" + std::to_string(i) + ".bin";
}

bool ModelReloader::modelsExist() const
{
  for (int i = 0; i < ensemble_size_; ++i) {
    if (!std::filesystem::exists(modelPath(i))) {
      return false;
    }
  }
  return true;
}

bool ModelReloader::modelsUpdated() const
{
  if (!modelsExist()) {
    return false;
  }

  for (int i = 0; i < ensemble_size_; ++i) {
    auto ts = std::filesystem::last_write_time(modelPath(i));
    if (ts != last_timestamps_[i]) {
      return true;
    }
  }
  return false;
}

bool ModelReloader::tryReload()
{
  if (!modelsExist()) {
    return false;
  }

  // 타임스탬프 수집
  std::vector<std::filesystem::file_time_type> new_timestamps(ensemble_size_);
  for (int i = 0; i < ensemble_size_; ++i) {
    new_timestamps[i] = std::filesystem::last_write_time(modelPath(i));
  }

  // 변경 감지 (최초 로드 포함)
  bool changed = false;
  for (int i = 0; i < ensemble_size_; ++i) {
    if (new_timestamps[i] != last_timestamps_[i]) {
      changed = true;
      break;
    }
  }

  if (!changed) {
    return false;
  }

  // 모델 로드 시도
  std::vector<std::unique_ptr<EigenMLP>> new_models;
  new_models.reserve(ensemble_size_);

  try {
    for (int i = 0; i < ensemble_size_; ++i) {
      new_models.push_back(EigenMLP::loadFromFile(modelPath(i)));
    }
  } catch (const std::exception&) {
    // 로드 실패 — 이전 상태 유지
    return false;
  }

  // 성공: 타임스탬프 갱신 + 모델 저장
  last_timestamps_ = new_timestamps;
  loaded_models_ = std::move(new_models);
  last_reload_time_ = std::chrono::steady_clock::now();
  return true;
}

std::vector<std::unique_ptr<EigenMLP>> ModelReloader::takeModels()
{
  return std::move(loaded_models_);
}

}  // namespace mpc_controller_ros2
