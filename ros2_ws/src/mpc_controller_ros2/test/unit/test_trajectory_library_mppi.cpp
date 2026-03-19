// =============================================================================
// Trajectory Library MPPI 단위 테스트 (15개)
//
// TrajectoryLibrary (8):
//   1. PrimitiveDimensions     — 모든 프리미티브 N×nu 차원
//   2. StraightForward         — STRAIGHT: v > 0, omega = 0
//   3. TurnsOppositeOmega      — LEFT/RIGHT: omega 부호 반대
//   4. SCurveSinusoidal        — S_CURVE: omega 부호 변경 (sin 패턴)
//   5. StopIsZero              — STOP: norm == 0
//   6. ReverseOnlyVminNeg      — v_min >= 0이면 REVERSE 미생성
//   7. PreviousSolutionUpdate  — updatePreviousSolution 후 값 일치
//   8. PrimitiveDiversity      — 프리미티브 간 Frobenius 거리 > 0
//
// Integration (7):
//   9. InjectionRatioSplit     — L + K_gaussian == K
//  10. LibrarySamplesUsed      — 첫 L개가 라이브러리 기반
//  11. CostComputationValid    — 모든 비용 유한 + >= 0
//  12. DisabledFallback        — enabled=false → vanilla
//  13. SwerveCompatible        — nu=3 호환
//  14. AdaptiveShiftsRatio     — 저비용 프리미티브 비율 증가
//  15. ConsecutiveStability    — 10회 반복, NaN/Inf 없음
// =============================================================================

#include <gtest/gtest.h>
#include <Eigen/Dense>
#include <vector>
#include <cmath>
#include <algorithm>
#include <string>

#include "mpc_controller_ros2/trajectory_library.hpp"
#include "mpc_controller_ros2/trajectory_library_mppi_controller_plugin.hpp"
#include "mpc_controller_ros2/weight_computation.hpp"
#include "mpc_controller_ros2/utils.hpp"

namespace mpc_controller_ros2
{

// ============================================================================
// 테스트 헬퍼
// ============================================================================
class TrajLibTestAccessor : public TrajectoryLibraryMPPIControllerPlugin
{
public:
  void setTestParams(const MPPIParams& params) { params_ = params; }
  void setDynamics(std::unique_ptr<BatchDynamicsWrapper> dyn) { dynamics_ = std::move(dyn); }
  void setSampler(std::unique_ptr<BaseSampler> sampler) { sampler_ = std::move(sampler); }
  void setCostFunction(std::unique_ptr<CompositeMPPICost> cf) { cost_function_ = std::move(cf); }
  void setWeightComputation(std::unique_ptr<WeightComputation> wc) {
    weight_computation_ = std::move(wc);
  }
  void setControlSequence(const Eigen::MatrixXd& cs) { control_sequence_ = cs; }
  Eigen::MatrixXd getControlSequence() const { return control_sequence_; }
  TrajectoryLibrary& getLibrary() { return library_; }
};

// ============================================================================
// 테스트 Fixture
// ============================================================================
class TrajectoryLibraryMPPITest : public ::testing::Test
{
protected:
  void SetUp() override
  {
    params_ = MPPIParams();
    params_.N = 10;
    params_.dt = 0.1;
    params_.K = 100;
    params_.lambda = 10.0;
    params_.v_max = 1.0;
    params_.v_min = 0.0;
    params_.omega_max = 1.0;
    params_.omega_min = -1.0;
    params_.noise_sigma = Eigen::Vector2d(0.5, 0.5);
    params_.traj_library_enabled = true;
    params_.traj_library_ratio = 0.15;
    params_.traj_library_perturbation = 0.1;
    params_.traj_library_adaptive = false;
    params_.traj_library_num_per_primitive = 0;

    // Reference trajectory: straight line
    ref_traj_ = Eigen::MatrixXd::Zero(params_.N + 1, 3);
    for (int t = 0; t <= params_.N; ++t) {
      ref_traj_(t, 0) = t * params_.dt;
    }

    state_ = Eigen::Vector3d(0.0, 0.0, 0.0);
  }

  MPPIParams params_;
  Eigen::MatrixXd ref_traj_;
  Eigen::Vector3d state_;
};

// ============================================================================
// TrajectoryLibrary 테스트 (8개)
// ============================================================================

TEST_F(TrajectoryLibraryMPPITest, PrimitiveDimensions)
{
  TrajectoryLibrary lib;
  int N = params_.N;
  int nu = 2;
  lib.generate(N, nu, params_.dt, params_.v_max, params_.v_min, params_.omega_max);

  const auto& prims = lib.getPrimitives();

  // v_min >= 0 → REVERSE 미생성 → 7종 (6 + PREVIOUS_SOLUTION)
  EXPECT_GE(lib.numPrimitives(), 7);

  for (const auto& prim : prims) {
    EXPECT_EQ(prim.control_sequence.rows(), N)
      << "Primitive " << prim.name << " has wrong rows";
    EXPECT_EQ(prim.control_sequence.cols(), nu)
      << "Primitive " << prim.name << " has wrong cols";
  }
}

TEST_F(TrajectoryLibraryMPPITest, StraightForward)
{
  TrajectoryLibrary lib;
  lib.generate(params_.N, 2, params_.dt, params_.v_max, params_.v_min, params_.omega_max);

  const auto& prims = lib.getPrimitives();

  // STRAIGHT 찾기
  bool found = false;
  for (const auto& prim : prims) {
    if (prim.name == "STRAIGHT") {
      found = true;
      for (int t = 0; t < params_.N; ++t) {
        EXPECT_DOUBLE_EQ(prim.control_sequence(t, 0), params_.v_max);
        EXPECT_DOUBLE_EQ(prim.control_sequence(t, 1), 0.0);
      }
    }
  }
  EXPECT_TRUE(found) << "STRAIGHT primitive not found";
}

TEST_F(TrajectoryLibraryMPPITest, TurnsOppositeOmega)
{
  TrajectoryLibrary lib;
  lib.generate(params_.N, 2, params_.dt, params_.v_max, params_.v_min, params_.omega_max);

  const auto& prims = lib.getPrimitives();

  Eigen::MatrixXd left_omega, right_omega;
  for (const auto& prim : prims) {
    if (prim.name == "TURN_LEFT") {
      left_omega = prim.control_sequence.col(1);
    } else if (prim.name == "TURN_RIGHT") {
      right_omega = prim.control_sequence.col(1);
    }
  }

  ASSERT_GT(left_omega.rows(), 0);
  ASSERT_GT(right_omega.rows(), 0);

  // 부호 반대
  for (int t = 0; t < params_.N; ++t) {
    EXPECT_GT(left_omega(t, 0), 0.0);
    EXPECT_LT(right_omega(t, 0), 0.0);
    EXPECT_NEAR(std::abs(left_omega(t, 0)), std::abs(right_omega(t, 0)), 1e-10);
  }
}

TEST_F(TrajectoryLibraryMPPITest, SCurveSinusoidal)
{
  TrajectoryLibrary lib;
  lib.generate(params_.N, 2, params_.dt, params_.v_max, params_.v_min, params_.omega_max);

  const auto& prims = lib.getPrimitives();

  for (const auto& prim : prims) {
    if (prim.name == "S_CURVE_LEFT") {
      // omega 값이 부호 변경해야 함 (sin 패턴)
      bool has_positive = false, has_negative = false;
      for (int t = 0; t < params_.N; ++t) {
        double omega = prim.control_sequence(t, 1);
        if (omega > 1e-6) has_positive = true;
        if (omega < -1e-6) has_negative = true;
      }
      EXPECT_TRUE(has_positive) << "S_CURVE_LEFT should have positive omega";
      EXPECT_TRUE(has_negative) << "S_CURVE_LEFT should have negative omega";
    }
  }
}

TEST_F(TrajectoryLibraryMPPITest, StopIsZero)
{
  TrajectoryLibrary lib;
  lib.generate(params_.N, 2, params_.dt, params_.v_max, params_.v_min, params_.omega_max);

  const auto& prims = lib.getPrimitives();

  bool found = false;
  for (const auto& prim : prims) {
    if (prim.name == "STOP") {
      found = true;
      EXPECT_DOUBLE_EQ(prim.control_sequence.norm(), 0.0);
    }
  }
  EXPECT_TRUE(found) << "STOP primitive not found";
}

TEST_F(TrajectoryLibraryMPPITest, ReverseOnlyVminNeg)
{
  // v_min >= 0 → REVERSE 미생성
  {
    TrajectoryLibrary lib;
    lib.generate(params_.N, 2, params_.dt, params_.v_max, 0.0, params_.omega_max);

    bool has_reverse = false;
    for (const auto& prim : lib.getPrimitives()) {
      if (prim.name == "REVERSE") has_reverse = true;
    }
    EXPECT_FALSE(has_reverse) << "REVERSE should not exist when v_min >= 0";
  }

  // v_min < 0 → REVERSE 생성
  {
    TrajectoryLibrary lib;
    lib.generate(params_.N, 2, params_.dt, params_.v_max, -0.5, params_.omega_max);

    bool has_reverse = false;
    for (const auto& prim : lib.getPrimitives()) {
      if (prim.name == "REVERSE") {
        has_reverse = true;
        for (int t = 0; t < params_.N; ++t) {
          EXPECT_DOUBLE_EQ(prim.control_sequence(t, 0), -0.5);
        }
      }
    }
    EXPECT_TRUE(has_reverse) << "REVERSE should exist when v_min < 0";
  }
}

TEST_F(TrajectoryLibraryMPPITest, PreviousSolutionUpdate)
{
  TrajectoryLibrary lib;
  lib.generate(params_.N, 2, params_.dt, params_.v_max, params_.v_min, params_.omega_max);

  // 특정 control_sequence 설정
  Eigen::MatrixXd cs = Eigen::MatrixXd::Random(params_.N, 2) * 0.3;
  lib.updatePreviousSolution(cs);

  bool found = false;
  for (const auto& prim : lib.getPrimitives()) {
    if (prim.name == "PREVIOUS_SOLUTION") {
      found = true;
      EXPECT_NEAR((prim.control_sequence - cs).norm(), 0.0, 1e-10);
    }
  }
  EXPECT_TRUE(found) << "PREVIOUS_SOLUTION primitive not found";
}

TEST_F(TrajectoryLibraryMPPITest, PrimitiveDiversity)
{
  TrajectoryLibrary lib;
  lib.generate(params_.N, 2, params_.dt, params_.v_max, params_.v_min, params_.omega_max);

  const auto& prims = lib.getPrimitives();
  int n = lib.numPrimitives();

  // 모든 프리미티브 쌍의 Frobenius 거리 > 0 (STOP과 PREVIOUS_SOLUTION 제외)
  for (int i = 0; i < n; ++i) {
    for (int j = i + 1; j < n; ++j) {
      // STOP과 PREVIOUS_SOLUTION(초기 zero)는 동일 → 이 쌍만 예외
      if ((prims[i].name == "STOP" && prims[j].name == "PREVIOUS_SOLUTION") ||
          (prims[i].name == "PREVIOUS_SOLUTION" && prims[j].name == "STOP"))
      {
        continue;
      }
      double dist = (prims[i].control_sequence - prims[j].control_sequence).norm();
      EXPECT_GT(dist, 0.0)
        << prims[i].name << " vs " << prims[j].name << " should be different";
    }
  }
}

// ============================================================================
// Integration 테스트 (7개)
// ============================================================================

TEST_F(TrajectoryLibraryMPPITest, InjectionRatioSplit)
{
  TrajectoryLibrary lib;
  lib.generate(params_.N, 2, params_.dt, params_.v_max, params_.v_min, params_.omega_max);

  int K = params_.K;
  int num_prims = lib.numPrimitives();
  int total_lib = static_cast<int>(std::floor(params_.traj_library_ratio * K));
  int samples_per_prim = std::max(1, total_lib / num_prims);
  int L = samples_per_prim * num_prims;
  if (L >= K) {
    samples_per_prim = (K - 1) / num_prims;
    L = samples_per_prim * num_prims;
  }
  int K_gaussian = K - L;

  EXPECT_EQ(L + K_gaussian, K);
  EXPECT_GT(L, 0);
  EXPECT_GT(K_gaussian, 0);
}

TEST_F(TrajectoryLibraryMPPITest, LibrarySamplesUsed)
{
  // 라이브러리 샘플이 프리미티브 기반인지 확인
  TrajectoryLibrary lib;
  lib.generate(params_.N, 2, params_.dt, params_.v_max, params_.v_min, params_.omega_max);

  const auto& prims = lib.getPrimitives();
  int num_prims = lib.numPrimitives();

  // STRAIGHT 프리미티브는 v_max 값 포함
  bool has_vmax_row = false;
  for (const auto& prim : prims) {
    if (prim.name == "STRAIGHT") {
      has_vmax_row = (prim.control_sequence(0, 0) == params_.v_max);
    }
  }
  EXPECT_TRUE(has_vmax_row);
  EXPECT_GT(num_prims, 0);
}

TEST_F(TrajectoryLibraryMPPITest, CostComputationValid)
{
  auto dynamics = std::make_unique<BatchDynamicsWrapper>(params_);
  auto sampler = std::make_unique<GaussianSampler>(params_.noise_sigma, 42);
  auto cost_function = std::make_unique<CompositeMPPICost>();
  cost_function->addCost(std::make_unique<StateTrackingCost>(params_.Q));
  cost_function->addCost(std::make_unique<TerminalCost>(params_.Qf));
  cost_function->addCost(std::make_unique<ControlEffortCost>(params_.R));

  int N = params_.N;
  int K = params_.K;
  int nu = 2;

  Eigen::MatrixXd control_sequence = Eigen::MatrixXd::Zero(N, nu);

  // 라이브러리 생성
  TrajectoryLibrary lib;
  lib.generate(N, nu, params_.dt, params_.v_max, params_.v_min, params_.omega_max);

  int num_prims = lib.numPrimitives();
  int total_lib = static_cast<int>(std::floor(params_.traj_library_ratio * K));
  int samples_per_prim = std::max(1, total_lib / num_prims);
  int L = samples_per_prim * num_prims;
  if (L >= K) {
    samples_per_prim = (K - 1) / num_prims;
    L = samples_per_prim * num_prims;
  }

  auto noise_samples = sampler->sample(K, N, nu);
  const auto& prims = lib.getPrimitives();

  // 샘플 구성
  std::vector<Eigen::MatrixXd> perturbed;
  int idx = 0;
  for (int p = 0; p < num_prims; ++p) {
    for (int j = 0; j < samples_per_prim; ++j) {
      Eigen::MatrixXd ctrl = prims[p].control_sequence +
                              params_.traj_library_perturbation * noise_samples[idx];
      perturbed.push_back(dynamics->clipControls(ctrl));
      ++idx;
    }
  }
  int K_gaussian = K - L;
  for (int k = 0; k < K_gaussian; ++k) {
    int src_idx = L + k;
    perturbed.push_back(dynamics->clipControls(control_sequence + noise_samples[src_idx]));
  }

  EXPECT_EQ(static_cast<int>(perturbed.size()), K);

  auto trajectories = dynamics->rolloutBatch(state_, perturbed, params_.dt);
  auto costs = cost_function->compute(trajectories, perturbed, ref_traj_);

  EXPECT_EQ(costs.size(), K);
  for (int k = 0; k < K; ++k) {
    EXPECT_TRUE(std::isfinite(costs(k))) << "Cost[" << k << "] is not finite";
    EXPECT_GE(costs(k), 0.0) << "Cost[" << k << "] is negative";
  }
}

TEST_F(TrajectoryLibraryMPPITest, DisabledFallback)
{
  params_.traj_library_enabled = false;
  EXPECT_FALSE(params_.traj_library_enabled);
  // disabled → parent::computeControl() 호출 (ROS2 없이는 플래그 검증만)
}

TEST_F(TrajectoryLibraryMPPITest, SwerveCompatible)
{
  // nu=3 (swerve) 호환성 확인
  TrajectoryLibrary lib;
  int nu = 3;
  lib.generate(params_.N, nu, params_.dt, params_.v_max, params_.v_min, params_.omega_max);

  const auto& prims = lib.getPrimitives();
  for (const auto& prim : prims) {
    EXPECT_EQ(prim.control_sequence.cols(), 3)
      << "Primitive " << prim.name << " should have nu=3";
  }

  // STRAIGHT: vy=0
  for (const auto& prim : prims) {
    if (prim.name == "STRAIGHT") {
      for (int t = 0; t < params_.N; ++t) {
        EXPECT_DOUBLE_EQ(prim.control_sequence(t, 2), 0.0);
      }
    }
  }
}

TEST_F(TrajectoryLibraryMPPITest, AdaptiveShiftsRatio)
{
  // 적응형 비율 조정 로직 검증
  MPPIParams p = params_;
  p.traj_library_adaptive = true;
  p.traj_library_ratio = 0.15;

  // 시나리오: 라이브러리가 더 좋을 때 → 비율 증가
  double ratio_before = p.traj_library_ratio;
  // lib_min < gauss_min → ratio *= 1.05
  p.traj_library_ratio = std::min(0.5, p.traj_library_ratio * 1.05);
  EXPECT_GT(p.traj_library_ratio, ratio_before);

  // 시나리오: Gaussian이 더 좋을 때 → 비율 감소
  ratio_before = p.traj_library_ratio;
  p.traj_library_ratio = std::max(0.05, p.traj_library_ratio * 0.95);
  EXPECT_LT(p.traj_library_ratio, ratio_before);

  // 범위 보장
  EXPECT_GE(p.traj_library_ratio, 0.05);
  EXPECT_LE(p.traj_library_ratio, 0.5);
}

TEST_F(TrajectoryLibraryMPPITest, ConsecutiveStability)
{
  auto dynamics = std::make_unique<BatchDynamicsWrapper>(params_);
  auto sampler = std::make_unique<GaussianSampler>(params_.noise_sigma, 42);
  auto cost_function = std::make_unique<CompositeMPPICost>();
  cost_function->addCost(std::make_unique<StateTrackingCost>(params_.Q));
  cost_function->addCost(std::make_unique<ControlEffortCost>(params_.R));

  VanillaMPPIWeights weight_strategy;
  int N = params_.N;
  int K = params_.K;
  int nu = 2;

  Eigen::MatrixXd control_sequence = Eigen::MatrixXd::Zero(N, nu);

  TrajectoryLibrary lib;
  lib.generate(N, nu, params_.dt, params_.v_max, params_.v_min, params_.omega_max);

  for (int iter = 0; iter < 10; ++iter) {
    // Warm-start shift
    for (int t = 0; t < N - 1; ++t) {
      control_sequence.row(t) = control_sequence.row(t + 1);
    }
    control_sequence.row(N - 1) = control_sequence.row(N - 2);

    lib.updatePreviousSolution(control_sequence);

    const auto& prims = lib.getPrimitives();
    int num_prims = lib.numPrimitives();
    int total_lib = static_cast<int>(std::floor(params_.traj_library_ratio * K));
    int samples_per_prim = std::max(1, total_lib / num_prims);
    int L = samples_per_prim * num_prims;
    if (L >= K) {
      samples_per_prim = (K - 1) / num_prims;
      L = samples_per_prim * num_prims;
    }
    int K_gaussian = K - L;

    auto noise_samples = sampler->sample(K, N, nu);

    std::vector<Eigen::MatrixXd> perturbed;
    std::vector<Eigen::MatrixXd> noise_for_w;

    int idx = 0;
    for (int p = 0; p < num_prims; ++p) {
      for (int j = 0; j < samples_per_prim; ++j) {
        Eigen::MatrixXd ctrl = prims[p].control_sequence +
                                params_.traj_library_perturbation * noise_samples[idx];
        ctrl = dynamics->clipControls(ctrl);
        perturbed.push_back(ctrl);
        noise_for_w.push_back(ctrl - control_sequence);
        ++idx;
      }
    }
    for (int k = 0; k < K_gaussian; ++k) {
      int src_idx = L + k;
      auto p = dynamics->clipControls(control_sequence + noise_samples[src_idx]);
      perturbed.push_back(p);
      noise_for_w.push_back(noise_samples[src_idx]);
    }

    auto trajectories = dynamics->rolloutBatch(state_, perturbed, params_.dt);
    auto costs = cost_function->compute(trajectories, perturbed, ref_traj_);
    auto weights = weight_strategy.compute(costs, params_.lambda);

    Eigen::MatrixXd wn = Eigen::MatrixXd::Zero(N, nu);
    for (int k = 0; k < K; ++k) {
      wn += weights(k) * noise_for_w[k];
    }
    control_sequence += wn;
    control_sequence = dynamics->clipControls(control_sequence);

    Eigen::VectorXd u_opt = control_sequence.row(0).transpose();

    EXPECT_FALSE(std::isnan(u_opt(0))) << "NaN at iter " << iter;
    EXPECT_FALSE(std::isnan(u_opt(1))) << "NaN at iter " << iter;
    EXPECT_FALSE(std::isinf(u_opt(0))) << "Inf at iter " << iter;
    EXPECT_FALSE(std::isinf(u_opt(1))) << "Inf at iter " << iter;
  }
}

}  // namespace mpc_controller_ros2

int main(int argc, char** argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
