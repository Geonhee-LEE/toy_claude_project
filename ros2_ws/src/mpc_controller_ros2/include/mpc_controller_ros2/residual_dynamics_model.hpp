#ifndef MPC_CONTROLLER_ROS2__RESIDUAL_DYNAMICS_MODEL_HPP_
#define MPC_CONTROLLER_ROS2__RESIDUAL_DYNAMICS_MODEL_HPP_

#include "mpc_controller_ros2/motion_model.hpp"
#include "mpc_controller_ros2/eigen_mlp.hpp"
#include <memory>

namespace mpc_controller_ros2
{

/**
 * @brief 잔차 동역학 모델 (Decorator Pattern)
 *
 * f_total(x, u) = f_nominal(x, u) + alpha * f_residual(MLP([x, u]))
 *
 * - alpha=0: 순수 공칭 모델 (학습 잔차 무시)
 * - alpha=1: 전체 잔차 보정
 * - 0<alpha<1: 블렌딩 (Sim-to-Real 점진 적용)
 *
 * dynamicsBatch()만 override하고, 나머지 메서드는 nominal_에 위임합니다.
 */
class ResidualDynamicsModel : public MotionModel
{
public:
  /**
   * @param nominal 공칭 동역학 모델 (소유권 이전)
   * @param residual_mlp MLP 잔차 모델 (소유권 이전)
   * @param alpha 잔차 블렌딩 계수 [0, 1]
   */
  ResidualDynamicsModel(
    std::unique_ptr<MotionModel> nominal,
    std::unique_ptr<EigenMLP> residual_mlp,
    double alpha = 1.0);

  // MotionModel 인터페이스 구현
  int stateDim() const override { return nominal_->stateDim(); }
  int controlDim() const override { return nominal_->controlDim(); }
  bool isHolonomic() const override { return nominal_->isHolonomic(); }
  std::string name() const override { return "residual_" + nominal_->name(); }

  Eigen::MatrixXd dynamicsBatch(
    const Eigen::MatrixXd& states,
    const Eigen::MatrixXd& controls) const override;

  Eigen::MatrixXd clipControls(
    const Eigen::MatrixXd& controls) const override;

  void normalizeStates(Eigen::MatrixXd& states) const override;

  geometry_msgs::msg::Twist controlToTwist(
    const Eigen::VectorXd& control) const override;

  Eigen::VectorXd twistToControl(
    const geometry_msgs::msg::Twist& twist) const override;

  std::vector<int> angleIndices() const override;

  /** @brief 블렌딩 계수 getter/setter */
  double alpha() const { return alpha_; }
  void setAlpha(double alpha) { alpha_ = std::clamp(alpha, 0.0, 1.0); }

  /** @brief 내부 공칭 모델 접근 */
  const MotionModel& nominal() const { return *nominal_; }

private:
  std::unique_ptr<MotionModel> nominal_;
  std::unique_ptr<EigenMLP> residual_mlp_;
  double alpha_;
};

}  // namespace mpc_controller_ros2

#endif  // MPC_CONTROLLER_ROS2__RESIDUAL_DYNAMICS_MODEL_HPP_
