#ifndef MPC_CONTROLLER_ROS2__MODEL_RELOADER_HPP_
#define MPC_CONTROLLER_ROS2__MODEL_RELOADER_HPP_

#include "mpc_controller_ros2/eigen_mlp.hpp"
#include <chrono>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace mpc_controller_ros2
{

/**
 * @brief 디스크에서 앙상블 MLP 모델 핫 리로드
 *
 * weights_dir 디렉토리에서 model_0.bin ~ model_{M-1}.bin 파일을
 * 모니터링하고, 타임스탬프 변경 시 새 모델을 로드합니다.
 *
 * 사용 패턴:
 *   1. 주기적으로 modelsUpdated() 확인
 *   2. 변경 감지 시 tryReload() 호출
 *   3. 성공하면 takeModels()로 새 앙상블 획득
 */
class ModelReloader
{
public:
  /**
   * @param weights_dir 모델 파일 디렉토리
   * @param ensemble_size 앙상블 MLP 수 (M)
   */
  ModelReloader(const std::string& weights_dir, int ensemble_size);

  /**
   * @brief 디스크에서 최신 모델 로드 시도
   * @return true if new model loaded successfully
   */
  bool tryReload();

  /**
   * @brief 현재 로드된 앙상블 MLP 반환 (소유권 이전)
   *
   * tryReload() 성공 후 호출. 호출 후 내부 벡터는 비워짐.
   */
  std::vector<std::unique_ptr<EigenMLP>> takeModels();

  /** @brief 마지막 로드 시간 */
  std::chrono::steady_clock::time_point lastReloadTime() const { return last_reload_time_; }

  /** @brief 모델 파일 전부 존재 확인 */
  bool modelsExist() const;

  /**
   * @brief 모델 파일 타임스탬프 변경 확인
   *
   * 최초 호출 시 또는 타임스탬프 변경 감지 시 true 반환.
   */
  bool modelsUpdated() const;

private:
  std::string weights_dir_;
  int ensemble_size_;
  mutable std::vector<std::filesystem::file_time_type> last_timestamps_;
  std::chrono::steady_clock::time_point last_reload_time_;
  std::vector<std::unique_ptr<EigenMLP>> loaded_models_;

  /** @brief i번째 모델 파일 경로 */
  std::string modelPath(int i) const;
};

}  // namespace mpc_controller_ros2

#endif  // MPC_CONTROLLER_ROS2__MODEL_RELOADER_HPP_
