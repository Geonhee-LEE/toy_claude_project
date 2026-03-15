#ifndef MPC_CONTROLLER_ROS2__ONLINE_DATA_COLLECTOR_HPP_
#define MPC_CONTROLLER_ROS2__ONLINE_DATA_COLLECTOR_HPP_

#include "mpc_controller_ros2/online_data_buffer.hpp"
#include "mpc_controller_ros2/motion_model.hpp"
#include <Eigen/Dense>
#include <atomic>

namespace mpc_controller_ros2
{

/**
 * @brief 온라인 데이터 수집기
 *
 * 매 제어 주기마다 (prev_state, prev_control, curr_state, dt) 데이터를
 * OnlineDataBuffer에 수집합니다. MotionModel의 차원 정보를 사용하여
 * 입력 검증을 수행합니다.
 *
 * Thread-safe: enabled_ 플래그는 atomic, buffer는 자체 mutex 보호.
 */
class OnlineDataCollector
{
public:
  struct Stats {
    size_t total_collected{0};
    size_t buffer_size{0};
    bool buffer_full{false};
  };

  /**
   * @param buffer 데이터 저장 버퍼 (비소유, 수명 보장 필요)
   * @param model 동역학 모델 (비소유, 차원 검증용)
   * @param state_dim 상태 차원
   * @param control_dim 제어 차원
   */
  OnlineDataCollector(OnlineDataBuffer* buffer,
                      const MotionModel* model,
                      int state_dim, int control_dim);

  /**
   * @brief 매 제어 주기마다 호출 — 데이터 포인트 수집
   *
   * enabled_가 false이면 무시.
   * 차원 불일치 시 무시 (silent skip).
   */
  void collect(const Eigen::VectorXd& prev_state,
               const Eigen::VectorXd& prev_control,
               const Eigen::VectorXd& curr_state,
               double dt);

  /** @brief 수집 통계 */
  Stats getStats() const;

  /** @brief 수집 활성화/비활성화 */
  void setEnabled(bool enabled) { enabled_ = enabled; }
  bool isEnabled() const { return enabled_; }

private:
  OnlineDataBuffer* buffer_;  // 비소유
  const MotionModel* model_;  // 비소유
  int nx_;
  int nu_;
  std::atomic<bool> enabled_{true};
  std::atomic<size_t> total_collected_{0};
};

}  // namespace mpc_controller_ros2

#endif  // MPC_CONTROLLER_ROS2__ONLINE_DATA_COLLECTOR_HPP_
