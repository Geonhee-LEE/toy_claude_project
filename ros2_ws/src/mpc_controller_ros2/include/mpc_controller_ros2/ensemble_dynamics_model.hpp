#ifndef MPC_CONTROLLER_ROS2__ENSEMBLE_DYNAMICS_MODEL_HPP_
#define MPC_CONTROLLER_ROS2__ENSEMBLE_DYNAMICS_MODEL_HPP_

#include "mpc_controller_ros2/motion_model.hpp"
#include "mpc_controller_ros2/eigen_mlp.hpp"
#include <memory>
#include <mutex>
#include <vector>

namespace mpc_controller_ros2
{

/**
 * @brief 앙상블 MLP 동역학 모델 (Decorator Pattern)
 *
 * M개 EigenMLP 앙상블을 사용하여 동역학 잔차를 예측하고
 * 불확실성(분산)을 추정합니다.
 *
 * f_total(x, u) = f_nominal(x, u) + alpha * mean(f_ensemble_i([x, u]))
 *
 * PredictionResult는 평균과 분산을 동시에 반환하여
 * UncertaintyAwareCost 등에서 활용됩니다.
 */
class EnsembleDynamicsModel : public MotionModel
{
public:
  struct PredictionResult {
    Eigen::MatrixXd mean;      // (M, nx) 앙상블 평균 잔차
    Eigen::MatrixXd variance;  // (M, nx) 앙상블 분산
  };

  /**
   * @param nominal 공칭 동역학 모델 (소유권 이전)
   * @param ensemble M개 MLP 앙상블 (소유권 이전)
   * @param alpha 잔차 블렌딩 계수 [0, 1]
   */
  EnsembleDynamicsModel(
    std::unique_ptr<MotionModel> nominal,
    std::vector<std::unique_ptr<EigenMLP>> ensemble,
    double alpha = 1.0);

  // MotionModel 인터페이스 구현
  int stateDim() const override { return nominal_->stateDim(); }
  int controlDim() const override { return nominal_->controlDim(); }
  bool isHolonomic() const override { return nominal_->isHolonomic(); }
  std::string name() const override { return "ensemble_" + nominal_->name(); }

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

  /**
   * @brief 앙상블 예측 + 불확실성 추정
   * @param states (M, nx)
   * @param controls (M, nu)
   * @return {mean, variance} 각각 (M, nx)
   */
  PredictionResult predictWithUncertainty(
    const Eigen::MatrixXd& states,
    const Eigen::MatrixXd& controls) const;

  /** @brief 앙상블 크기 */
  int ensembleSize() const { return static_cast<int>(ensemble_.size()); }

  /** @brief 블렌딩 계수 */
  double alpha() const { return alpha_; }
  void setAlpha(double alpha) { alpha_ = std::clamp(alpha, 0.0, 1.0); }

  /** @brief 내부 공칭 모델 접근 */
  const MotionModel& nominal() const { return *nominal_; }

  /**
   * @brief 앙상블 MLP 핫스왑 (thread-safe)
   *
   * dynamicsBatch/predictWithUncertainty 실행 중에도 안전하게
   * 앙상블을 교체합니다.
   */
  void updateEnsemble(std::vector<std::unique_ptr<EigenMLP>> new_ensemble);

  /** @brief 모델 버전 (핫스왑 카운터) */
  int modelVersion() const { return model_version_; }

private:
  std::unique_ptr<MotionModel> nominal_;
  std::vector<std::unique_ptr<EigenMLP>> ensemble_;
  double alpha_;
  mutable std::mutex ensemble_mutex_;  // 핫스왑 보호
  int model_version_{0};

  /** @brief MLP 입력 특성 구성: [states | controls] */
  Eigen::MatrixXd buildFeatures(
    const Eigen::MatrixXd& states,
    const Eigen::MatrixXd& controls) const;
};

}  // namespace mpc_controller_ros2

#endif  // MPC_CONTROLLER_ROS2__ENSEMBLE_DYNAMICS_MODEL_HPP_
