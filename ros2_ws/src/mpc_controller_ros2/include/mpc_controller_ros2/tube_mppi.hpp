#ifndef MPC_CONTROLLER_ROS2__TUBE_MPPI_HPP_
#define MPC_CONTROLLER_ROS2__TUBE_MPPI_HPP_

#include <Eigen/Dense>
#include <memory>
#include <vector>

#include "mpc_controller_ros2/ancillary_controller.hpp"
#include "mpc_controller_ros2/mppi_params.hpp"

namespace mpc_controller_ros2
{

/**
 * @brief Tube-MPPI 확장 정보 구조체
 */
struct TubeMPPIInfo
{
  Eigen::VectorXd nominal_state;        // 외란 없는 이상 상태
  Eigen::VectorXd nominal_control;      // MPPI가 계산한 nominal 제어
  Eigen::VectorXd body_error;           // Body frame 오차
  Eigen::VectorXd feedback_correction;  // 피드백 보정량
  Eigen::VectorXd applied_control;      // 최종 적용된 제어
  double tube_width{0.0};               // Tube 폭
  std::vector<Eigen::VectorXd> tube_boundary;  // 시각화용 tube 경계점들
};

/**
 * @brief Tube-MPPI 래퍼 클래스
 *
 * MPPI의 nominal 궤적을 따라가면서 실제 상태와의 오차를 body frame에서
 * 보정하는 강건 제어 구조입니다.
 *
 * 구조:
 * ┌─────────────────────────────────────────────────────────┐
 * │                      TubeMPPI                           │
 * │  ┌────────────────┐       ┌────────────────────────┐   │
 * │  │ Nominal MPPI   │  ───→ │ AncillaryController    │   │
 * │  │ (외란 없는     │       │ (Body frame 피드백)    │   │
 * │  │  이상 궤적)    │       │                        │   │
 * │  └────────────────┘       └────────────────────────┘   │
 * │                                                         │
 * │   u_applied = u_nominal + K_fb · (x_nominal - x_actual) │
 * └─────────────────────────────────────────────────────────┘
 *
 * 참조:
 *   - Tube MPC: "Robust Model Predictive Control with Tubes"
 *   - Python M2 구현: mpc_controller/controllers/mppi/tube_mppi.py
 */
class TubeMPPI
{
public:
  /**
   * @brief 생성자
   * @param params MPPI 파라미터 (tube 관련 설정 포함)
   */
  explicit TubeMPPI(const MPPIParams& params);

  /**
   * @brief 피드백 보정된 제어 입력 계산
   * @param nominal_control MPPI가 계산한 nominal 제어 [v, omega]
   * @param nominal_trajectory Nominal 궤적 (N+1 x 3)
   * @param actual_state 실제 로봇 상태 [x, y, theta]
   * @return 보정된 제어 입력과 정보
   */
  std::pair<Eigen::VectorXd, TubeMPPIInfo> computeCorrectedControl(
    const Eigen::VectorXd& nominal_control,
    const Eigen::MatrixXd& nominal_trajectory,
    const Eigen::VectorXd& actual_state
  );

  /**
   * @brief Tube 너비 업데이트 (적응형)
   * @param tracking_error 현재 추적 오차 norm
   */
  void updateTubeWidth(double tracking_error);

  /**
   * @brief Tube 경계 계산 (시각화용)
   * @param nominal_trajectory Nominal 궤적
   * @return 좌/우 경계점 쌍 벡터
   */
  std::vector<std::pair<Eigen::VectorXd, Eigen::VectorXd>> computeTubeBoundary(
    const Eigen::MatrixXd& nominal_trajectory
  ) const;

  /**
   * @brief 상태가 Tube 내부에 있는지 확인
   * @param nominal_state Nominal 상태
   * @param actual_state 실제 상태
   * @return true if 실제 상태가 tube 내부에 있음
   */
  bool isInsideTube(
    const Eigen::VectorXd& nominal_state,
    const Eigen::VectorXd& actual_state
  ) const;

  /**
   * @brief 피드백 게인 업데이트
   */
  void setFeedbackGains(double k_forward, double k_lateral, double k_angle);

  /**
   * @brief Tube 폭 설정
   */
  void setTubeWidth(double width);

  /**
   * @brief 현재 tube 폭 반환
   */
  double getTubeWidth() const { return tube_width_; }

  /**
   * @brief Ancillary controller 참조 반환
   */
  const AncillaryController& getAncillaryController() const { return ancillary_; }

  /**
   * @brief 파라미터 업데이트
   */
  void updateParams(const MPPIParams& params);

private:
  AncillaryController ancillary_;  // Body frame 피드백 컨트롤러
  double tube_width_;              // Tube 폭 (m)
  double max_tube_width_;          // 최대 tube 폭
  double min_tube_width_;          // 최소 tube 폭
  double tube_adaptation_rate_;    // Tube 폭 적응률
};

}  // namespace mpc_controller_ros2

#endif  // MPC_CONTROLLER_ROS2__TUBE_MPPI_HPP_
