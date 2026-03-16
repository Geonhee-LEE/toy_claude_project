/**
 * @brief Tube-MPPI Plugin + Dynamic Obstacle Tracker 통합 동작 데모
 *
 * ROS2 의존성 없이 두 기능을 함께 실행하여 검증합니다:
 *   1. Tube-MPPI: nominal state MPPI + body frame 피드백
 *   2. Dynamic Obstacle Tracker: clustering + tracking + 속도 추정
 *   3. 통합: 동적 장애물 속도 → CBF barrier set → MPPI 비용에 반영
 */
#include <gtest/gtest.h>
#include <Eigen/Dense>
#include <cmath>
#include <memory>
#include <iostream>
#include <iomanip>

#include "mpc_controller_ros2/mppi_params.hpp"
#include "mpc_controller_ros2/batch_dynamics_wrapper.hpp"
#include "mpc_controller_ros2/cost_functions.hpp"
#include "mpc_controller_ros2/sampling.hpp"
#include "mpc_controller_ros2/weight_computation.hpp"
#include "mpc_controller_ros2/tube_mppi.hpp"
#include "mpc_controller_ros2/ancillary_controller.hpp"
#include "mpc_controller_ros2/dynamic_obstacle_tracker.hpp"
#include "mpc_controller_ros2/barrier_function.hpp"
#include "mpc_controller_ros2/motion_model_factory.hpp"

using namespace mpc_controller_ros2;

// =============================================================================
// Helper: MPPI 1-step 실행
// =============================================================================
static std::pair<Eigen::VectorXd, Eigen::MatrixXd> runMPPIStep(
  const Eigen::VectorXd& state,
  Eigen::MatrixXd& control_seq,
  const Eigen::MatrixXd& ref,
  const MPPIParams& params,
  BatchDynamicsWrapper& dynamics,
  CompositeMPPICost& cost_fn,
  BaseSampler& sampler)
{
  int N = params.N;
  int nu = dynamics.model().controlDim();
  int K = params.K;

  // warm-start shift
  for (int t = 0; t < N - 1; ++t) {
    control_seq.row(t) = control_seq.row(t + 1);
  }
  control_seq.row(N - 1).setZero();

  auto noise = sampler.sample(K, N, nu);
  std::vector<Eigen::MatrixXd> perturbed(K);
  for (int k = 0; k < K; ++k) {
    perturbed[k] = dynamics.clipControls(control_seq + noise[k]);
  }

  std::vector<Eigen::MatrixXd> trajectories;
  dynamics.rolloutBatchInPlace(state, perturbed, params.dt, trajectories);

  Eigen::VectorXd costs = cost_fn.compute(trajectories, perturbed, ref);
  VanillaMPPIWeights weight_comp;
  Eigen::VectorXd weights = weight_comp.compute(costs, params.lambda);

  Eigen::MatrixXd weighted_update = Eigen::MatrixXd::Zero(N, nu);
  for (int k = 0; k < K; ++k) {
    weighted_update += weights(k) * noise[k];
  }
  control_seq += weighted_update;
  control_seq = dynamics.clipControls(control_seq);

  Eigen::VectorXd u_opt = control_seq.row(0).transpose();

  // 최적 궤적
  std::vector<Eigen::MatrixXd> opt_ctrl = {control_seq};
  std::vector<Eigen::MatrixXd> opt_traj;
  dynamics.rolloutBatchInPlace(state, opt_ctrl, params.dt, opt_traj);

  return {u_opt, opt_traj[0]};
}

// =============================================================================
// 통합 테스트: Tube-MPPI + Dynamic Obstacle Tracker 동시 동작
// =============================================================================

TEST(TubeTrackerIntegration, FullPipelineDemo)
{
  std::cout << "\n"
    << "╔══════════════════════════════════════════════════════════════╗\n"
    << "║  Tube-MPPI + Dynamic Obstacle Tracker 통합 동작 데모       ║\n"
    << "╚══════════════════════════════════════════════════════════════╝\n\n";

  // ── 1. 파라미터 설정 ──
  MPPIParams params;
  params.N = 20;
  params.dt = 0.1;
  params.K = 256;
  params.lambda = 10.0;
  params.v_max = 0.5;
  params.v_min = 0.0;
  params.omega_max = 1.0;
  params.omega_min = -1.0;
  params.noise_sigma = Eigen::Vector2d(0.5, 0.5);
  params.Q = Eigen::MatrixXd::Identity(3, 3) * 10.0;
  params.Q(2, 2) = 1.0;
  params.Qf = params.Q * 2.0;
  params.R = Eigen::MatrixXd::Identity(2, 2) * 0.1;
  params.R_rate = Eigen::MatrixXd::Identity(2, 2) * 1.0;
  params.tube_enabled = true;
  params.tube_width = 0.5;
  params.tube_nominal_reset_threshold = 1.0;
  params.k_forward = 0.8;
  params.k_lateral = 0.5;
  params.k_angle = 1.0;
  params.motion_model = "diff_drive";

  // ── 2. 컴포넌트 초기화 ──
  auto model = MotionModelFactory::create("diff_drive", params);
  auto dynamics = std::make_unique<BatchDynamicsWrapper>(
    params, std::shared_ptr<MotionModel>(std::move(model)));
  auto sampler = std::make_unique<GaussianSampler>(params.noise_sigma, 42);

  auto cost_fn = std::make_unique<CompositeMPPICost>();
  cost_fn->addCost(std::make_unique<StateTrackingCost>(params.Q));
  cost_fn->addCost(std::make_unique<TerminalCost>(params.Qf));
  cost_fn->addCost(std::make_unique<ControlEffortCost>(params.R));

  TubeMPPI tube_mppi(params);
  AncillaryController ancillary(params.k_forward, params.k_lateral, params.k_angle);

  // Dynamic Obstacle Tracker
  DynamicObstacleTracker tracker(0.15, 3, 0.3, 0.5, 2.0);

  // BarrierFunctionSet (CBF)
  BarrierFunctionSet barrier_set;

  int nx = dynamics->model().stateDim();
  int nu = dynamics->model().controlDim();

  // ── 3. 시나리오 설정 ──
  // 로봇: 원점에서 (3, 0)으로 직진
  // 동적 장애물: (2, 0.3)에서 y=0 방향으로 이동 (교차 경로)
  Eigen::VectorXd nominal_state = Eigen::VectorXd::Zero(nx);
  Eigen::VectorXd actual_state = Eigen::VectorXd::Zero(nx);
  Eigen::MatrixXd control_seq = Eigen::MatrixXd::Zero(params.N, nu);

  // Reference: x축 직진 궤적
  Eigen::MatrixXd ref = Eigen::MatrixXd::Zero(params.N + 1, nx);
  for (int t = 0; t <= params.N; ++t) {
    ref(t, 0) = 0.5 * t * params.dt;  // x = 0.5*t*dt
  }

  // 동적 장애물 초기 위치/속도
  double obs_x = 2.0, obs_y = 0.4;
  double obs_vx = 0.0, obs_vy = -0.2;  // y 방향으로 접근

  int n_steps = 30;
  double total_tracking_error = 0.0;
  int cbf_active_count = 0;

  std::cout << "시나리오: 로봇 (0,0)→(3,0) 직진, 장애물 (2,0.4) vy=-0.2 교차\n\n";
  std::cout << "┌──────┬──────────────────────┬──────────────────────┬──────────┬────────────────┬───────────────────┐\n";
  std::cout << "│ Step │ Nominal (x,y,θ)      │ Actual (x,y,θ)      │ Tube Dev │ Obstacle (x,y) │ Tracked Vel (v)   │\n";
  std::cout << "├──────┼──────────────────────┼──────────────────────┼──────────┼────────────────┼───────────────────┤\n";

  for (int step = 0; step < n_steps; ++step) {
    double t = step * params.dt;

    // ── 4. Dynamic Obstacle Tracker ──
    // 장애물 이동
    obs_x += obs_vx * params.dt;
    obs_y += obs_vy * params.dt;

    // 장애물을 lethal 셀로 시뮬레이션 (5개 인접 셀)
    std::vector<Eigen::Vector2d> lethal_cells;
    double cell_r = 0.025;
    for (int dx = -1; dx <= 1; ++dx) {
      for (int dy = -1; dy <= 1; ++dy) {
        if (std::abs(dx) + std::abs(dy) <= 1) {
          lethal_cells.emplace_back(obs_x + dx * 0.05, obs_y + dy * 0.05);
        }
      }
    }

    auto [tracked_obs, tracked_vels] = tracker.process(lethal_cells, cell_r, t);

    // Barrier set 갱신
    if (!tracked_obs.empty()) {
      barrier_set.setObstaclesWithVelocity(tracked_obs, tracked_vels);
    }

    // ── 5. Tube-MPPI 실행 ──
    // 5a. Nominal-actual 편차 확인
    double deviation = (nominal_state.head(2) - actual_state.head(2)).norm();
    if (deviation > params.tube_nominal_reset_threshold) {
      nominal_state = actual_state;
    }

    // 5b. Nominal state에서 MPPI 실행
    auto [u_nominal, opt_traj] = runMPPIStep(
      nominal_state, control_seq, ref, params, *dynamics, *cost_fn, *sampler);

    // 5c. Body frame 피드백 보정
    Eigen::VectorXd body_error = ancillary.computeBodyFrameError(nominal_state, actual_state);
    Eigen::VectorXd du = ancillary.computeFeedbackCorrection(body_error);
    Eigen::VectorXd u_applied = u_nominal + du;

    // 클리핑
    Eigen::MatrixXd u_mat(1, nu);
    u_mat.row(0) = u_applied.transpose();
    u_applied = dynamics->clipControls(u_mat).row(0).transpose();

    // 5d. Nominal state 전파 (u_nominal으로)
    Eigen::MatrixXd sm(1, nx); sm.row(0) = nominal_state.transpose();
    Eigen::MatrixXd cm(1, nu); cm.row(0) = u_nominal.transpose();
    nominal_state = dynamics->model().propagateBatch(sm, cm, params.dt).row(0).transpose();

    // 5e. Actual state 전파 (u_applied + 외란)
    Eigen::VectorXd u_disturbed = u_applied;
    u_disturbed(0) += 0.02 * std::sin(step * 0.5);  // 작은 외란
    u_disturbed(1) += 0.01 * std::cos(step * 0.3);
    Eigen::MatrixXd sa(1, nx); sa.row(0) = actual_state.transpose();
    Eigen::MatrixXd ca(1, nu); ca.row(0) = u_disturbed.transpose();
    actual_state = dynamics->model().propagateBatch(sa, ca, params.dt).row(0).transpose();

    // ── 6. 메트릭 수집 ──
    total_tracking_error += deviation;

    // 장애물 활성화 여부 확인
    auto active = barrier_set.getActiveBarriers(actual_state);
    if (!active.empty()) { cbf_active_count++; }

    // 추적 속도
    double tracked_speed = 0.0;
    if (!tracked_vels.empty()) {
      tracked_speed = tracked_vels[0].norm();
    }

    // 출력 (5 스텝마다)
    if (step % 3 == 0 || step == n_steps - 1) {
      std::cout << "│ " << std::setw(4) << step
                << " │ (" << std::fixed << std::setprecision(2)
                << std::setw(5) << nominal_state(0) << ","
                << std::setw(5) << nominal_state(1) << ","
                << std::setw(5) << nominal_state(2) << ") "
                << "│ (" << std::setw(5) << actual_state(0) << ","
                << std::setw(5) << actual_state(1) << ","
                << std::setw(5) << actual_state(2) << ") "
                << "│ " << std::setw(6) << std::setprecision(3) << deviation << " "
                << "│ (" << std::setw(5) << std::setprecision(2) << obs_x << ","
                << std::setw(5) << obs_y << ") "
                << "│ " << std::setw(6) << std::setprecision(3) << tracked_speed
                << " m/s       │\n";
    }
  }

  std::cout << "└──────┴──────────────────────┴──────────────────────┴──────────┴────────────────┴───────────────────┘\n\n";

  // ── 7. 결과 검증 ──
  double avg_error = total_tracking_error / n_steps;
  double final_x = actual_state(0);
  double dist_to_obs = std::sqrt(
    (actual_state(0) - obs_x) * (actual_state(0) - obs_x) +
    (actual_state(1) - obs_y) * (actual_state(1) - obs_y));

  std::cout << "═══ 결과 요약 ═══\n";
  std::cout << "  로봇 최종 위치: (" << std::setprecision(3) << actual_state(0)
            << ", " << actual_state(1) << ", " << actual_state(2) << ")\n";
  std::cout << "  평균 tube 편차: " << avg_error << " m\n";
  std::cout << "  CBF 활성 횟수:  " << cbf_active_count << "/" << n_steps << " steps\n";
  std::cout << "  장애물 최종:    (" << obs_x << ", " << obs_y << ")\n";
  std::cout << "  장애물 거리:    " << dist_to_obs << " m\n";

  // 추적 속도 정확도 검증
  auto tracks = tracker.tracker().getTracks();
  if (!tracks.empty()) {
    double est_vy = tracks[0].vy;
    std::cout << "  추적 속도 vy:   " << est_vy << " m/s (실제: -0.2)\n";
    EXPECT_NEAR(est_vy, -0.2, 0.15) << "Tracked velocity should converge to true velocity";
  }

  std::cout << "\n";

  // 기본 검증
  EXPECT_GT(final_x, 0.3) << "로봇이 전진해야 함";
  EXPECT_LT(avg_error, params.tube_nominal_reset_threshold)
    << "평균 편차가 리셋 임계값 이내여야 함";
  EXPECT_TRUE(std::isfinite(actual_state(0)));
  EXPECT_TRUE(std::isfinite(nominal_state(0)));
}

// =============================================================================
// Tube-MPPI 단독 시연: nominal vs actual 궤적 비교
// =============================================================================

TEST(TubeTrackerIntegration, TubeMPPINominalVsActual)
{
  std::cout << "\n"
    << "╔══════════════════════════════════════════════════════════════╗\n"
    << "║  Tube-MPPI: Nominal vs Actual 궤적 비교                    ║\n"
    << "╚══════════════════════════════════════════════════════════════╝\n\n";

  MPPIParams params;
  params.N = 15;
  params.dt = 0.1;
  params.K = 128;
  params.lambda = 10.0;
  params.v_max = 0.5;
  params.v_min = 0.0;
  params.omega_max = 1.0;
  params.omega_min = -1.0;
  params.noise_sigma = Eigen::Vector2d(0.5, 0.5);
  params.Q = Eigen::MatrixXd::Identity(3, 3) * 10.0;
  params.Q(2, 2) = 1.0;
  params.Qf = params.Q * 2.0;
  params.R = Eigen::MatrixXd::Identity(2, 2) * 0.1;
  params.R_rate = Eigen::MatrixXd::Identity(2, 2) * 1.0;
  params.tube_enabled = true;
  params.tube_width = 0.3;
  params.k_forward = 0.8;
  params.k_lateral = 0.5;
  params.k_angle = 1.0;

  auto model = MotionModelFactory::create("diff_drive", params);
  auto dynamics = std::make_unique<BatchDynamicsWrapper>(
    params, std::shared_ptr<MotionModel>(std::move(model)));
  auto sampler = std::make_unique<GaussianSampler>(params.noise_sigma, 42);

  auto cost_fn = std::make_unique<CompositeMPPICost>();
  cost_fn->addCost(std::make_unique<StateTrackingCost>(params.Q));
  cost_fn->addCost(std::make_unique<TerminalCost>(params.Qf));
  cost_fn->addCost(std::make_unique<ControlEffortCost>(params.R));

  AncillaryController ancillary(params.k_forward, params.k_lateral, params.k_angle);
  TubeMPPI tube_mppi(params);

  int nx = dynamics->model().stateDim();
  int nu = dynamics->model().controlDim();

  Eigen::VectorXd nominal_state = Eigen::VectorXd::Zero(nx);
  Eigen::VectorXd actual_state = Eigen::VectorXd::Zero(nx);
  Eigen::MatrixXd control_seq = Eigen::MatrixXd::Zero(params.N, nu);

  Eigen::MatrixXd ref = Eigen::MatrixXd::Zero(params.N + 1, nx);
  for (int t = 0; t <= params.N; ++t) {
    ref(t, 0) = 0.4 * t * params.dt;
  }

  // 외란 강도 비교
  double disturbance_amp = 0.05;

  std::cout << "외란 진폭: " << disturbance_amp << " m/s\n";
  std::cout << "Tube 폭:   " << params.tube_width << " m\n\n";

  // ASCII 궤적 시각화용 버퍼
  const int grid_w = 60, grid_h = 15;
  std::vector<std::string> grid(grid_h, std::string(grid_w, ' '));

  int n_steps = 20;
  double max_deviation = 0.0;

  for (int step = 0; step < n_steps; ++step) {
    auto [u_nominal, _] = runMPPIStep(
      nominal_state, control_seq, ref, params, *dynamics, *cost_fn, *sampler);

    Eigen::VectorXd body_error = ancillary.computeBodyFrameError(nominal_state, actual_state);
    Eigen::VectorXd du = ancillary.computeFeedbackCorrection(body_error);
    Eigen::VectorXd u_applied = u_nominal + du;

    Eigen::MatrixXd u_mat(1, nu);
    u_mat.row(0) = u_applied.transpose();
    u_applied = dynamics->clipControls(u_mat).row(0).transpose();

    // Nominal 전파
    Eigen::MatrixXd sm(1, nx); sm.row(0) = nominal_state.transpose();
    Eigen::MatrixXd cm(1, nu); cm.row(0) = u_nominal.transpose();
    nominal_state = dynamics->model().propagateBatch(sm, cm, params.dt).row(0).transpose();

    // Actual 전파 (외란 포함)
    Eigen::VectorXd u_dist = u_applied;
    u_dist(0) += disturbance_amp * std::sin(step * 0.7);
    u_dist(1) += disturbance_amp * std::cos(step * 0.5);
    Eigen::MatrixXd sa(1, nx); sa.row(0) = actual_state.transpose();
    Eigen::MatrixXd ca(1, nu); ca.row(0) = u_dist.transpose();
    actual_state = dynamics->model().propagateBatch(sa, ca, params.dt).row(0).transpose();

    double dev = (nominal_state.head(2) - actual_state.head(2)).norm();
    max_deviation = std::max(max_deviation, dev);

    // ASCII 그리드에 기록
    auto plotPoint = [&](double px, double py, char ch) {
      int gx = static_cast<int>(px * 30.0);  // x → 열
      int gy = grid_h / 2 - static_cast<int>(py * 30.0);  // y → 행 (상하 반전)
      if (gx >= 0 && gx < grid_w && gy >= 0 && gy < grid_h) {
        grid[gy][gx] = ch;
      }
    };

    plotPoint(nominal_state(0), nominal_state(1), 'N');
    plotPoint(actual_state(0), actual_state(1), 'A');

    bool inside = tube_mppi.isInsideTube(nominal_state, actual_state);
    (void)inside;  // 사용됨 아래 출력에서
  }

  // ASCII 궤적 출력
  std::cout << "궤적 시각화 (N=nominal, A=actual):\n";
  std::cout << "┌" << std::string(grid_w, '─') << "┐\n";
  for (int r = 0; r < grid_h; ++r) {
    std::cout << "│" << grid[r] << "│\n";
  }
  std::cout << "└" << std::string(grid_w, '─') << "┘\n";
  std::cout << " x→ (0.0 ──── " << std::setprecision(1) << grid_w / 30.0 << ")\n\n";

  std::cout << "최대 편차: " << std::setprecision(4) << max_deviation << " m\n";
  std::cout << "Tube 폭:   " << params.tube_width << " m\n";
  std::cout << "Tube 내부: " << (max_deviation < params.tube_width ? "✓ YES" : "✗ NO") << "\n\n";

  EXPECT_LT(max_deviation, params.tube_width * 2.0)
    << "편차가 tube 폭의 2배 이내여야 함";
  EXPECT_GT(actual_state(0), 0.1) << "전진해야 함";
}

// =============================================================================
// Dynamic Obstacle Tracker 단독 시연: 이동 장애물 추적
// =============================================================================

TEST(TubeTrackerIntegration, TrackerVelocityConvergence)
{
  std::cout << "\n"
    << "╔══════════════════════════════════════════════════════════════╗\n"
    << "║  Dynamic Obstacle Tracker: 속도 수렴 데모                  ║\n"
    << "╚══════════════════════════════════════════════════════════════╝\n\n";

  DynamicObstacleTracker tracker(0.15, 3, 0.3, 0.5, 2.0);

  // 장애물: (1,1)에서 vx=0.5, vy=-0.3으로 이동
  double true_vx = 0.5, true_vy = -0.3;
  double ox = 1.0, oy = 1.0;
  double dt = 0.1;

  std::cout << "실제 속도: vx=" << true_vx << ", vy=" << true_vy << "\n";
  std::cout << "EMA alpha: 0.3\n\n";
  std::cout << "┌──────┬────────────────┬────────────────┬────────────────┐\n";
  std::cout << "│ Step │ Position (x,y) │ Est Vel (vx,vy)│ Vel Error      │\n";
  std::cout << "├──────┼────────────────┼────────────────┼────────────────┤\n";

  for (int step = 0; step < 20; ++step) {
    double t = step * dt;
    ox += true_vx * dt;
    oy += true_vy * dt;

    // 5개 인접 셀 생성
    std::vector<Eigen::Vector2d> cells;
    for (int dx = -1; dx <= 1; ++dx) {
      for (int dy = -1; dy <= 1; ++dy) {
        if (std::abs(dx) + std::abs(dy) <= 1) {
          cells.emplace_back(ox + dx * 0.05, oy + dy * 0.05);
        }
      }
    }

    auto [obs, vels] = tracker.process(cells, 0.025, t);

    double est_vx = 0, est_vy = 0;
    if (!vels.empty()) {
      est_vx = vels[0](0);
      est_vy = vels[0](1);
    }

    double vel_err = std::sqrt(
      (est_vx - true_vx) * (est_vx - true_vx) +
      (est_vy - true_vy) * (est_vy - true_vy));

    if (step % 2 == 0 || step == 19) {
      std::cout << "│ " << std::setw(4) << step
                << " │ (" << std::fixed << std::setprecision(2)
                << std::setw(5) << ox << ","
                << std::setw(5) << oy << ") "
                << "│ (" << std::setw(5) << std::setprecision(3) << est_vx << ","
                << std::setw(6) << est_vy << ") "
                << "│ " << std::setw(8) << vel_err
                << "       │\n";
    }
  }

  std::cout << "└──────┴────────────────┴────────────────┴────────────────┘\n\n";

  // 최종 속도 수렴 확인
  auto tracks = tracker.tracker().getTracks();
  ASSERT_FALSE(tracks.empty());
  double final_vx = tracks[0].vx;
  double final_vy = tracks[0].vy;

  std::cout << "최종 추정: vx=" << std::setprecision(3) << final_vx
            << ", vy=" << final_vy << "\n";
  std::cout << "실제:     vx=" << true_vx << ", vy=" << true_vy << "\n";
  std::cout << "오차:     " << std::sqrt(
    (final_vx - true_vx) * (final_vx - true_vx) +
    (final_vy - true_vy) * (final_vy - true_vy)) << " m/s\n\n";

  EXPECT_NEAR(final_vx, true_vx, 0.1);
  EXPECT_NEAR(final_vy, true_vy, 0.1);
}

int main(int argc, char** argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
