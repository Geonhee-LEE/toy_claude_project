#ifndef MPC_CONTROLLER_ROS2__SAVITZKY_GOLAY_FILTER_HPP_
#define MPC_CONTROLLER_ROS2__SAVITZKY_GOLAY_FILTER_HPP_

#include <Eigen/Dense>
#include <deque>

namespace mpc_controller_ros2
{

/**
 * @brief Savitzky-Golay 필터 — EMA 대체 출력 스무딩
 *
 * 과거 제어 이력 + 미래 예측 시퀀스를 조합하여 다항식 피팅.
 * 지연 없이 부드러운 제어 출력 보장.
 *
 * 참고: mppi_swerve_drive_ros (IROS 2024)에서 검증된 기법.
 */
class SavitzkyGolayFilter
{
public:
  /**
   * @brief 생성자
   * @param half_window 과거/미래 윈도우 크기 (양쪽 각각)
   * @param poly_order 다항식 차수 (< window_size = 2*half_window+1)
   * @param nu 제어 입력 차원
   */
  SavitzkyGolayFilter(int half_window, int poly_order, int nu);

  /**
   * @brief 필터 적용: 과거 이력 + 미래 시퀀스 → smoothed u[current_step]
   * @param control_sequence (N, nu) 미래 예측 제어 시퀀스
   * @param current_step 현재 스텝 인덱스 (보통 0)
   * @return smoothed 제어 벡터 (nu,)
   */
  Eigen::VectorXd apply(
    const Eigen::MatrixXd& control_sequence,
    int current_step = 0
  );

  /**
   * @brief 과거 이력에 제어 입력 추가
   * @param control (nu,) 제어 벡터
   */
  void pushHistory(const Eigen::VectorXd& control);

  /**
   * @brief 필터 상태 초기화
   */
  void reset();

  /** @brief SG 계수 벡터 반환 (테스트용) */
  const Eigen::VectorXd& coefficients() const { return sg_coeffs_; }

  /** @brief 윈도우 크기 반환 (2*half_window+1) */
  int windowSize() const { return 2 * half_window_ + 1; }

  /** @brief half_window 반환 */
  int halfWindow() const { return half_window_; }

private:
  int half_window_;
  int poly_order_;
  int nu_;
  Eigen::VectorXd sg_coeffs_;            // (window_size,) Vandermonde 기반 계수
  std::deque<Eigen::VectorXd> history_;  // 과거 제어 이력

  /**
   * @brief Vandermonde 행렬 기반 SG 계수 계산
   * V = (window_size × poly_order+1), coeffs = (V^T V)^{-1} V^T 의 center 행
   */
  void computeCoefficients();
};

}  // namespace mpc_controller_ros2

#endif  // MPC_CONTROLLER_ROS2__SAVITZKY_GOLAY_FILTER_HPP_
