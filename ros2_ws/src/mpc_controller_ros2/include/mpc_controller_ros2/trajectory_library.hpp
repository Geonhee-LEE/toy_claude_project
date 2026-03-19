#ifndef MPC_CONTROLLER_ROS2__TRAJECTORY_LIBRARY_HPP_
#define MPC_CONTROLLER_ROS2__TRAJECTORY_LIBRARY_HPP_

#include <Eigen/Dense>
#include <string>
#include <vector>
#include <utility>

namespace mpc_controller_ros2
{

/**
 * @brief 제어 시퀀스 프리미티브 라이브러리
 *
 * 7종 사전 계산된 프리미티브 + 이전 최적 시퀀스를 관리.
 * MPPI 샘플 풀에 결정적 시퀀스로 주입하여 warm-start 다양성 향상.
 */
class TrajectoryLibrary
{
public:
  struct Primitive
  {
    std::string name;
    Eigen::MatrixXd control_sequence;  // (N, nu)
  };

  /**
   * @brief 프리미티브 라이브러리 생성
   * @param N 예측 horizon 스텝 수
   * @param nu 제어 차원
   * @param dt 시간 간격
   * @param v_max 최대 선속도
   * @param v_min 최소 선속도
   * @param omega_max 최대 각속도
   */
  void generate(int N, int nu, double dt,
                double v_max, double v_min, double omega_max);

  /** @brief 이전 최적 시퀀스 업데이트 (동적 프리미티브) */
  void updatePreviousSolution(const Eigen::MatrixXd& control_sequence);

  /** @brief 모든 프리미티브 반환 (PREVIOUS_SOLUTION 포함) */
  const std::vector<Primitive>& getPrimitives() const { return primitives_; }

  /** @brief 프리미티브 수 반환 */
  int numPrimitives() const { return static_cast<int>(primitives_.size()); }

private:
  std::vector<Primitive> primitives_;
  int prev_solution_idx_{-1};  // PREVIOUS_SOLUTION 인덱스
};

}  // namespace mpc_controller_ros2

#endif  // MPC_CONTROLLER_ROS2__TRAJECTORY_LIBRARY_HPP_
