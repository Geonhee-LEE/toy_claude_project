// =============================================================================
// CUDA Rollout Engine — CPU 참조 구현 (항상 컴파일)
//
// CUDA 빌드 불가 환경에서도 동일한 API로 CPU 폴백 실행.
// RK4 적분 + DiffDrive/Swerve 동역학 + 대각 비용 계산.
// =============================================================================

#include "mpc_controller_ros2/cuda_rollout_engine.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace mpc_controller_ros2
{

// ============================================================================
// 생성자 / 소멸자
// ============================================================================

CudaRolloutEngine::CudaRolloutEngine(const Config& config)
  : config_(config)
{
  trajectories_.resize(config_.K);
}

CudaRolloutEngine::~CudaRolloutEngine() = default;

// ============================================================================
// 초기화
// ============================================================================

void CudaRolloutEngine::initialize(
  const Eigen::MatrixXd& Q, const Eigen::MatrixXd& Qf,
  const Eigen::MatrixXd& R, const Eigen::MatrixXd& R_rate,
  double v_min, double v_max, double omega_min, double omega_max,
  double vy_max)
{
  Q_ = Q;
  Qf_ = Qf;
  R_ = R;
  R_rate_ = R_rate;
  v_min_ = v_min;
  v_max_ = v_max;
  omega_min_ = omega_min;
  omega_max_ = omega_max;
  vy_max_ = vy_max;
}

// ============================================================================
// 장애물 업데이트
// ============================================================================

void CudaRolloutEngine::updateObstacles(
  const std::vector<Eigen::Vector3d>& obstacles,
  double safety_distance, double obstacle_weight)
{
  obstacles_ = obstacles;
  safety_distance_ = safety_distance;
  obstacle_weight_ = obstacle_weight;
}

// ============================================================================
// 참조 궤적 업데이트
// ============================================================================

void CudaRolloutEngine::updateReference(const Eigen::MatrixXd& reference)
{
  reference_ = reference;
}

// ============================================================================
// 실행 (CPU 폴백)
// ============================================================================

void CudaRolloutEngine::execute(
  const Eigen::VectorXd& x0,
  const std::vector<Eigen::MatrixXd>& perturbed_controls,
  double dt,
  Eigen::VectorXd& costs_out)
{
  executeCPU(x0, perturbed_controls, dt, costs_out);
}

// ============================================================================
// GPU 사용 가능 여부 — CPU 빌드는 항상 false
// ============================================================================

bool CudaRolloutEngine::isAvailable()
{
  return false;
}

// ============================================================================
// 궤적 다운로드
// ============================================================================

void CudaRolloutEngine::downloadAllTrajectories(
  std::vector<Eigen::MatrixXd>& trajectories_out) const
{
  trajectories_out = trajectories_;
}

// ============================================================================
// CPU 롤아웃 + 비용 계산 (참조 구현)
// ============================================================================

void CudaRolloutEngine::executeCPU(
  const Eigen::VectorXd& x0,
  const std::vector<Eigen::MatrixXd>& perturbed_controls,
  double dt,
  Eigen::VectorXd& costs_out)
{
  int K = static_cast<int>(perturbed_controls.size());
  int N = config_.N;
  int nu = config_.nu;

  costs_out.resize(K);
  trajectories_.resize(K);

  for (int k = 0; k < K; ++k) {
    // 제어 클리핑
    Eigen::MatrixXd u_clipped = perturbed_controls[k];

    for (int t = 0; t < N; ++t) {
      // vx 클리핑 (첫 번째 제어 입력)
      u_clipped(t, 0) = std::clamp(u_clipped(t, 0), v_min_, v_max_);

      if (config_.motion_model == "swerve" && nu >= 3) {
        // Swerve: u = [vx, vy, omega]
        // vy 클리핑
        double vy_limit = (vy_max_ > 0.0) ? vy_max_ : v_max_;
        u_clipped(t, 1) = std::clamp(u_clipped(t, 1), -vy_limit, vy_limit);
        // omega 클리핑
        u_clipped(t, 2) = std::clamp(u_clipped(t, 2), omega_min_, omega_max_);
      } else {
        // DiffDrive: u = [v, omega]
        u_clipped(t, 1) = std::clamp(u_clipped(t, 1), omega_min_, omega_max_);
      }
    }

    trajectories_[k] = rolloutSingle(x0, u_clipped, dt);
    costs_out(k) = computeSingleCost(trajectories_[k], u_clipped);
  }
}

// ============================================================================
// 동역학 함수 (모델별 분기)
// ============================================================================

Eigen::VectorXd CudaRolloutEngine::dynamics(
  const Eigen::VectorXd& state,
  const Eigen::VectorXd& control) const
{
  int nx = config_.nx;
  Eigen::VectorXd dx = Eigen::VectorXd::Zero(nx);
  double theta = state(2);

  if (config_.motion_model == "swerve" && config_.nu >= 3) {
    // Swerve: dx = vx*cos(θ) - vy*sin(θ)
    //         dy = vx*sin(θ) + vy*cos(θ)
    //         dθ = omega
    double vx = control(0);
    double vy = control(1);
    double omega = control(2);
    dx(0) = vx * std::cos(theta) - vy * std::sin(theta);
    dx(1) = vx * std::sin(theta) + vy * std::cos(theta);
    dx(2) = omega;
  } else {
    // DiffDrive: dx = v*cos(θ), dy = v*sin(θ), dθ = omega
    double v = control(0);
    double omega = control(1);
    dx(0) = v * std::cos(theta);
    dx(1) = v * std::sin(theta);
    dx(2) = omega;
  }

  return dx;
}

// ============================================================================
// 단일 샘플 RK4 롤아웃
// ============================================================================

Eigen::MatrixXd CudaRolloutEngine::rolloutSingle(
  const Eigen::VectorXd& x0,
  const Eigen::MatrixXd& controls,
  double dt) const
{
  int N = controls.rows();
  int nx = config_.nx;

  Eigen::MatrixXd traj(N + 1, nx);
  traj.row(0) = x0.transpose();

  for (int t = 0; t < N; ++t) {
    Eigen::VectorXd s = traj.row(t).transpose();
    Eigen::VectorXd u = controls.row(t).transpose();

    // RK4 적분
    Eigen::VectorXd k1 = dynamics(s, u);
    Eigen::VectorXd k2 = dynamics(s + 0.5 * dt * k1, u);
    Eigen::VectorXd k3 = dynamics(s + 0.5 * dt * k2, u);
    Eigen::VectorXd k4 = dynamics(s + dt * k3, u);

    Eigen::VectorXd s_next = s + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4);

    // 각도 정규화 [-π, π]
    s_next(2) = std::atan2(std::sin(s_next(2)), std::cos(s_next(2)));

    traj.row(t + 1) = s_next.transpose();
  }

  return traj;
}

// ============================================================================
// 단일 샘플 비용 계산
// ============================================================================

double CudaRolloutEngine::computeSingleCost(
  const Eigen::MatrixXd& trajectory,
  const Eigen::MatrixXd& controls) const
{
  double cost = 0.0;
  int N = controls.rows();
  int nx = config_.nx;
  int nu = config_.nu;
  int ref_rows = reference_.rows();

  if (ref_rows == 0) {
    return 0.0;
  }

  // ================================================================
  // 상태 추적 비용 (대각 Q, 속도를 위해 원소별 계산)
  // ================================================================
  for (int t = 0; t <= N; ++t) {
    int ref_idx = std::min(t, ref_rows - 1);
    Eigen::VectorXd err = trajectory.row(t).transpose()
                          - reference_.row(ref_idx).transpose();

    // 각도 오차 정규화
    if (nx >= 3) {
      err(2) = std::atan2(std::sin(err(2)), std::cos(err(2)));
    }

    for (int i = 0; i < nx; ++i) {
      cost += Q_(i, i) * err(i) * err(i);
    }
  }

  // ================================================================
  // 터미널 비용 (대각 Qf)
  // ================================================================
  {
    int ref_idx = std::min(N, ref_rows - 1);
    Eigen::VectorXd err_f = trajectory.row(N).transpose()
                            - reference_.row(ref_idx).transpose();
    if (nx >= 3) {
      err_f(2) = std::atan2(std::sin(err_f(2)), std::cos(err_f(2)));
    }
    for (int i = 0; i < nx; ++i) {
      cost += Qf_(i, i) * err_f(i) * err_f(i);
    }
  }

  // ================================================================
  // 제어 노력 비용 (대각 R)
  // ================================================================
  for (int t = 0; t < N; ++t) {
    Eigen::VectorXd u = controls.row(t).transpose();
    for (int j = 0; j < nu; ++j) {
      cost += R_(j, j) * u(j) * u(j);
    }
  }

  // ================================================================
  // 제어 변화율 비용 (대각 R_rate)
  // ================================================================
  for (int t = 1; t < N; ++t) {
    Eigen::VectorXd du = controls.row(t).transpose()
                         - controls.row(t - 1).transpose();
    for (int j = 0; j < nu; ++j) {
      cost += R_rate_(j, j) * du(j) * du(j);
    }
  }

  // ================================================================
  // 장애물 비용 (안전 거리 기반 이차 페널티)
  // ================================================================
  for (int t = 0; t <= N; ++t) {
    double sx = trajectory(t, 0);
    double sy = trajectory(t, 1);

    for (const auto& obs : obstacles_) {
      double dx = sx - obs(0);
      double dy = sy - obs(1);
      double dist = std::sqrt(dx * dx + dy * dy);
      double d_safe = obs(2) + safety_distance_;

      if (dist < d_safe) {
        double penetration = d_safe - dist;
        cost += obstacle_weight_ * penetration * penetration;
      }
    }
  }

  return cost;
}

}  // namespace mpc_controller_ros2
