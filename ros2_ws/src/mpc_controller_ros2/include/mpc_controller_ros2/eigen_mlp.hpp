#ifndef MPC_CONTROLLER_ROS2__EIGEN_MLP_HPP_
#define MPC_CONTROLLER_ROS2__EIGEN_MLP_HPP_

#include <Eigen/Dense>
#include <memory>
#include <string>
#include <vector>

namespace mpc_controller_ros2
{

/**
 * @brief 순수 Eigen MLP 추론 엔진
 *
 * PyTorch 학습 → 바이너리 내보내기 → C++ 추론 파이프라인:
 *   input → Z-score 정규화 → [Linear+ReLU]×(L-1) → Linear → 역정규화 → output
 *
 * 바이너리 포맷:
 *   magic(4B, "EMLP") + version(4B) + n_layers(4B)
 *   + norm_params(in_mean, in_std, out_mean, out_std)
 *   + layer[i](rows, cols, weight_data, bias_data)
 *
 * 성능: K=512 배치 기준 ~0.005ms (BLAS dgemm 활용)
 */
class EigenMLP
{
public:
  struct LayerParams {
    Eigen::MatrixXd weight;  // (out_dim, in_dim)
    Eigen::VectorXd bias;    // (out_dim,)
  };

  struct NormParams {
    Eigen::VectorXd in_mean;   // 입력 평균
    Eigen::VectorXd in_std;    // 입력 표준편차
    Eigen::VectorXd out_mean;  // 출력 평균
    Eigen::VectorXd out_std;   // 출력 표준편차
  };

  /**
   * @brief 바이너리 파일에서 MLP 로드
   * @param path 파일 경로
   * @return EigenMLP 인스턴스
   * @throws std::runtime_error 파일 오류 시
   */
  static std::unique_ptr<EigenMLP> loadFromFile(const std::string& path);

  /**
   * @brief 레이어 파라미터와 정규화로 직접 생성
   */
  EigenMLP(std::vector<LayerParams> layers, NormParams norm);

  /** @brief 단일 입력 순전파 */
  Eigen::VectorXd forward(const Eigen::VectorXd& input) const;

  /** @brief 배치 순전파 — inputs: (K × input_dim) → (K × output_dim) */
  Eigen::MatrixXd forwardBatch(const Eigen::MatrixXd& inputs) const;

  /** @brief 입력 차원 */
  int inputDim() const;

  /** @brief 출력 차원 */
  int outputDim() const;

  /** @brief 레이어 수 */
  int numLayers() const { return static_cast<int>(layers_.size()); }

  /** @brief 정규화 사용 여부 */
  bool hasNormalization() const { return has_norm_; }

private:
  std::vector<LayerParams> layers_;
  NormParams norm_;
  bool has_norm_{false};

  static constexpr uint32_t kMagic = 0x454D4C50;  // "EMLP"
  static constexpr uint32_t kVersion = 1;
};

}  // namespace mpc_controller_ros2

#endif  // MPC_CONTROLLER_ROS2__EIGEN_MLP_HPP_
