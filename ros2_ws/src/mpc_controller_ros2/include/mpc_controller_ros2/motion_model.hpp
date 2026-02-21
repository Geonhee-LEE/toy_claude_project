#ifndef MPC_CONTROLLER_ROS2__MOTION_MODEL_HPP_
#define MPC_CONTROLLER_ROS2__MOTION_MODEL_HPP_

#include <Eigen/Dense>
#include <geometry_msgs/msg/twist.hpp>
#include <string>
#include <vector>

namespace mpc_controller_ros2
{

/**
 * @brief MotionModel 추상 인터페이스
 *
 * DiffDrive / Swerve / NonCoaxialSwerve 등 다양한 로봇 동역학 모델을
 * 통일된 인터페이스로 제공합니다. MPPI 파이프라인의 모든 구성 요소
 * (BatchDynamicsWrapper, CostFunctions, ControllerPlugin 등)는
 * 이 인터페이스를 통해 모델에 접근합니다.
 *
 * 계층 구조:
 * ┌──────────────────────────────────────────┐
 * │ MotionModel (abstract)                    │
 * │  + stateDim() / controlDim()             │
 * │  + dynamicsBatch()  (순수 가상)          │
 * │  + propagateBatch() (기본 RK4)           │
 * │  + rolloutBatch()   (기본 루프)          │
 * │  + clipControls()   (순수 가상)          │
 * │  + normalizeStates()(순수 가상)          │
 * │  + controlToTwist() / twistToControl()   │
 * ├──────────────────────────────────────────┤
 * │ DiffDriveModel     (nx=3, nu=2)          │
 * │ SwerveDriveModel   (nx=3, nu=3)          │
 * │ NonCoaxialSwerveModel (nx=4, nu=3)       │
 * └──────────────────────────────────────────┘
 */
class MotionModel
{
public:
  virtual ~MotionModel() = default;

  /** @brief 상태 차원 (DiffDrive=3, Swerve=3, NonCoaxial=4) */
  virtual int stateDim() const = 0;

  /** @brief 제어 차원 (DiffDrive=2, Swerve=3, NonCoaxial=3) */
  virtual int controlDim() const = 0;

  /** @brief Holonomic 여부 (Swerve=true, DiffDrive/NonCoaxial=false) */
  virtual bool isHolonomic() const = 0;

  /** @brief 모델 이름 ("diff_drive", "swerve", "non_coaxial_swerve") */
  virtual std::string name() const = 0;

  /**
   * @brief 연속 동역학 (배치): x_dot = f(x, u)
   * @param states  (M x nx)
   * @param controls (M x nu)
   * @return state_dot (M x nx)
   */
  virtual Eigen::MatrixXd dynamicsBatch(
    const Eigen::MatrixXd& states,
    const Eigen::MatrixXd& controls) const = 0;

  /**
   * @brief 제어 클리핑 (배치)
   * @param controls (M x nu)
   * @return clipped (M x nu)
   */
  virtual Eigen::MatrixXd clipControls(
    const Eigen::MatrixXd& controls) const = 0;

  /**
   * @brief 상태 정규화 (각도 wrapping 등) — in-place
   * @param states (M x nx)
   */
  virtual void normalizeStates(Eigen::MatrixXd& states) const = 0;

  /**
   * @brief 제어 벡터 → ROS2 Twist 메시지
   * @param control (nu,)
   * @return Twist 메시지
   */
  virtual geometry_msgs::msg::Twist controlToTwist(
    const Eigen::VectorXd& control) const = 0;

  /**
   * @brief ROS2 Twist 메시지 → 제어 벡터
   * @param twist Twist 메시지
   * @return control (nu,)
   */
  virtual Eigen::VectorXd twistToControl(
    const geometry_msgs::msg::Twist& twist) const = 0;

  /**
   * @brief Angle 인덱스 목록 (비용 함수에서 angle error wrapping에 사용)
   * @return DiffDrive/Swerve={2}, NonCoaxial={2}
   */
  virtual std::vector<int> angleIndices() const = 0;

  /**
   * @brief RK4 적분 (단일 스텝, 배치) — 기본 구현 제공
   * @param states  (M x nx)
   * @param controls (M x nu)
   * @param dt 시간 간격
   * @return 다음 상태 (M x nx)
   */
  virtual Eigen::MatrixXd propagateBatch(
    const Eigen::MatrixXd& states,
    const Eigen::MatrixXd& controls,
    double dt) const;

  /**
   * @brief 제어 시퀀스 배치 Rollout — 기본 구현 제공
   * @param x0 초기 상태 (nx,)
   * @param control_sequences 제어 시퀀스 벡터 [K개, 각각 N x nu]
   * @param dt 시간 간격
   * @return 궤적 벡터 [K개, 각각 (N+1) x nx]
   */
  virtual std::vector<Eigen::MatrixXd> rolloutBatch(
    const Eigen::VectorXd& x0,
    const std::vector<Eigen::MatrixXd>& control_sequences,
    double dt) const;
};

}  // namespace mpc_controller_ros2

#endif  // MPC_CONTROLLER_ROS2__MOTION_MODEL_HPP_
