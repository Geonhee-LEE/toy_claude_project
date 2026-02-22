#include "mpc_controller_ros2/savitzky_golay_filter.hpp"
#include <cassert>
#include <cmath>
#include <algorithm>

namespace mpc_controller_ros2
{

SavitzkyGolayFilter::SavitzkyGolayFilter(int half_window, int poly_order, int nu)
  : half_window_(std::max(1, half_window)),
    poly_order_(std::max(0, std::min(poly_order, 2 * half_window))),
    nu_(nu)
{
  // poly_order must be < window_size
  int window_size = 2 * half_window_ + 1;
  if (poly_order_ >= window_size) {
    poly_order_ = window_size - 1;
  }

  computeCoefficients();
}

void SavitzkyGolayFilter::computeCoefficients()
{
  int window_size = 2 * half_window_ + 1;
  int num_coeffs = poly_order_ + 1;

  // Vandermonde 행렬: V(i, j) = x_i^j, x_i = i - half_window
  Eigen::MatrixXd V(window_size, num_coeffs);
  for (int i = 0; i < window_size; ++i) {
    double x = static_cast<double>(i - half_window_);
    V(i, 0) = 1.0;
    for (int j = 1; j < num_coeffs; ++j) {
      V(i, j) = V(i, j - 1) * x;
    }
  }

  // SG 계수: (V^T V)^{-1} V^T 의 center 행 (= half_window_ 번째 행)
  // center 행 = 0차 다항식 값 평가 (x=0) → 이는 (V^T V)^{-1} V^T의 첫째 행과 동일
  Eigen::MatrixXd VtV = V.transpose() * V;
  Eigen::MatrixXd VtV_inv_Vt = VtV.ldlt().solve(V.transpose());

  // 첫째 행: 0차(상수항) 계수 → center에서의 smoothed 값
  sg_coeffs_ = VtV_inv_Vt.row(0).transpose();
}

Eigen::VectorXd SavitzkyGolayFilter::apply(
  const Eigen::MatrixXd& control_sequence,
  int current_step
)
{
  int window_size = 2 * half_window_ + 1;
  int history_available = static_cast<int>(history_.size());
  int future_available = static_cast<int>(control_sequence.rows()) - current_step;

  // 윈도우 구성: [과거 half_window개] + [현재] + [미래 half_window개]
  // 이력 부족 시 available만 사용하고 나머지는 현재값으로 패딩
  Eigen::VectorXd current_u = Eigen::VectorXd::Zero(nu_);
  if (future_available > 0) {
    current_u = control_sequence.row(current_step).transpose();
  }

  // 윈도우 데이터 수집
  Eigen::MatrixXd window_data(window_size, nu_);

  for (int i = 0; i < window_size; ++i) {
    int offset = i - half_window_;  // -half_window ~ +half_window

    if (offset < 0) {
      // 과거 이력
      int hist_idx = history_available + offset;  // 뒤에서부터
      if (hist_idx >= 0 && hist_idx < history_available) {
        window_data.row(i) = history_[hist_idx].transpose();
      } else {
        // 이력 부족: 현재값으로 패딩
        window_data.row(i) = current_u.transpose();
      }
    } else if (offset == 0) {
      // 현재
      window_data.row(i) = current_u.transpose();
    } else {
      // 미래 예측
      int future_idx = current_step + offset;
      if (future_idx < control_sequence.rows()) {
        window_data.row(i) = control_sequence.row(future_idx);
      } else {
        // 미래 데이터 부족: 마지막 값으로 패딩
        if (control_sequence.rows() > 0) {
          window_data.row(i) = control_sequence.row(control_sequence.rows() - 1);
        } else {
          window_data.row(i) = current_u.transpose();
        }
      }
    }
  }

  // 각 nu 차원마다 SG 계수와 dot product
  Eigen::VectorXd smoothed(nu_);
  for (int d = 0; d < nu_; ++d) {
    smoothed(d) = sg_coeffs_.dot(window_data.col(d));
  }

  return smoothed;
}

void SavitzkyGolayFilter::pushHistory(const Eigen::VectorXd& control)
{
  history_.push_back(control);
  while (static_cast<int>(history_.size()) > half_window_) {
    history_.pop_front();
  }
}

void SavitzkyGolayFilter::reset()
{
  history_.clear();
}

}  // namespace mpc_controller_ros2
