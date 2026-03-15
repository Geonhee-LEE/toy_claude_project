#ifndef MPC_CONTROLLER_ROS2__ONLINE_DATA_BUFFER_HPP_
#define MPC_CONTROLLER_ROS2__ONLINE_DATA_BUFFER_HPP_

#include <Eigen/Dense>
#include <vector>
#include <mutex>
#include <string>

namespace mpc_controller_ros2
{

/**
 * @brief 온라인 데이터 수집 링 버퍼
 *
 * computeVelocityCommands() 매 호출 시 (state, control, next_state, dt)를
 * 수집하여 오프라인 학습 데이터를 축적합니다.
 *
 * Thread-safe: mutex 보호.
 * 고정 크기 링 버퍼로 메모리 사용량 제한.
 */
class OnlineDataBuffer
{
public:
  struct DataPoint {
    Eigen::VectorXd state;
    Eigen::VectorXd control;
    Eigen::VectorXd next_state;
    double dt;
  };

  /**
   * @param capacity 최대 데이터 포인트 수
   */
  explicit OnlineDataBuffer(size_t capacity = 10000);

  /**
   * @brief 데이터 포인트 추가 (링 버퍼)
   */
  void add(const Eigen::VectorXd& state,
           const Eigen::VectorXd& control,
           const Eigen::VectorXd& next_state,
           double dt);

  /**
   * @brief CSV 파일로 내보내기
   * @param path 출력 파일 경로
   * @return 내보낸 데이터 포인트 수
   */
  size_t exportCSV(const std::string& path) const;

  /** @brief 버퍼 초기화 */
  void clear();

  /** @brief 현재 저장된 데이터 수 */
  size_t size() const;

  /** @brief 버퍼 용량 */
  size_t capacity() const { return capacity_; }

  /** @brief 버퍼가 가득 찼는지 */
  bool full() const;

private:
  size_t capacity_;
  size_t write_idx_{0};
  size_t count_{0};
  std::vector<DataPoint> buffer_;
  mutable std::mutex mutex_;
};

}  // namespace mpc_controller_ros2

#endif  // MPC_CONTROLLER_ROS2__ONLINE_DATA_BUFFER_HPP_
