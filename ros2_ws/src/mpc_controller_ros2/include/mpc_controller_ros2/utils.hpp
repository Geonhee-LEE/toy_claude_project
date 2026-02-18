#ifndef MPC_CONTROLLER_ROS2__UTILS_HPP_
#define MPC_CONTROLLER_ROS2__UTILS_HPP_

#include <Eigen/Dense>
#include <cmath>
#include <geometry_msgs/msg/quaternion.hpp>

namespace mpc_controller_ros2
{

/**
 * @brief 각도를 [-π, π] 범위로 정규화
 * @param angle 입력 각도 (라디안)
 * @return 정규화된 각도
 */
inline double normalizeAngle(double angle)
{
  // atan2를 사용하여 [-π, π] 범위로 정규화
  return std::atan2(std::sin(angle), std::cos(angle));
}

/**
 * @brief 각도 벡터를 [-π, π] 범위로 정규화 (배치 처리)
 * @param angles 입력 각도 벡터 (라디안)
 * @return 정규화된 각도 벡터
 */
Eigen::VectorXd normalizeAngleBatch(const Eigen::VectorXd& angles);

/**
 * @brief Softmax 가중치 계산 (MPPI용)
 * @param costs 비용 벡터 (K개 샘플)
 * @param lambda Temperature 파라미터
 * @return 정규화된 가중치 벡터 (합 = 1)
 *
 * weights[k] = exp(-costs[k]/lambda) / Σ exp(-costs[i]/lambda)
 * 수치 안정성을 위해 shifted costs 사용
 */
Eigen::VectorXd softmaxWeights(const Eigen::VectorXd& costs, double lambda);

/**
 * @brief Quaternion을 Yaw 각도로 변환
 * @param quat Quaternion 메시지
 * @return Yaw 각도 (라디안, [-π, π])
 */
double quaternionToYaw(const geometry_msgs::msg::Quaternion& quat);

/**
 * @brief Log-Sum-Exp 계산 (수치 안정성 보장)
 * @param values 입력 벡터
 * @return log(sum(exp(values)))
 *
 * max-shift trick: log(Σ exp(v_i)) = max + log(Σ exp(v_i - max))
 */
double logSumExp(const Eigen::VectorXd& values);

/**
 * @brief Effective Sample Size (ESS) 계산
 * @param weights 가중치 벡터 (정규화됨, 합 = 1)
 * @return ESS 값 [1, K]
 *
 * ESS = 1 / Σ(weights[k]^2)
 * ESS가 낮으면 대부분의 가중치가 소수의 샘플에 집중됨 (퇴화)
 */
double computeESS(const Eigen::VectorXd& weights);

/**
 * @brief 2D 유클리드 거리 계산 (배치 처리)
 * @param points1 첫 번째 점들 (N x 2)
 * @param points2 두 번째 점들 (N x 2)
 * @return 거리 벡터 (N)
 */
Eigen::VectorXd euclideanDistance2D(
  const Eigen::MatrixXd& points1,
  const Eigen::MatrixXd& points2
);

/**
 * @brief 행렬의 각 행에서 최소값 찾기
 * @param matrix 입력 행렬 (N x M)
 * @return 각 행의 최소값 벡터 (N)
 */
Eigen::VectorXd rowwiseMin(const Eigen::MatrixXd& matrix);

/**
 * @brief 행렬의 각 열에서 최소값 찾기
 * @param matrix 입력 행렬 (N x M)
 * @return 각 열의 최소값 벡터 (M)
 */
Eigen::VectorXd colwiseMin(const Eigen::MatrixXd& matrix);

/**
 * @brief q-Exponential 함수 (Tsallis 통계역학)
 * @param x 입력 벡터
 * @param q Tsallis 파라미터 (q→1이면 표준 exp)
 * @return exp_q(x) = [1 + (1-q)*x]_+^{1/(1-q)}
 *
 * q > 1: heavy-tail (탐색 증가)
 * q < 1: light-tail (집중 증가)
 * q = 1: 표준 exp (Vanilla MPPI와 동일)
 */
Eigen::VectorXd qExponential(const Eigen::VectorXd& x, double q);

}  // namespace mpc_controller_ros2

#endif  // MPC_CONTROLLER_ROS2__UTILS_HPP_
