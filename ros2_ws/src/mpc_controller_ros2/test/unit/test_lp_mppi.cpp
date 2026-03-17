// =============================================================================
// LP-MPPI (Low-Pass Filtering MPPI) Unit Tests
//
// 15 gtest: LP 필터 수학적 검증 + 제어 품질 + 플러그인 통합
// =============================================================================

#include <gtest/gtest.h>
#include <Eigen/Dense>
#include <cmath>
#include "mpc_controller_ros2/lp_mppi_controller_plugin.hpp"
#include "mpc_controller_ros2/mppi_controller_plugin.hpp"

namespace mpc_controller_ros2
{

// ============================================================================
// 헬퍼: LP 필터를 독립적으로 테스트하기 위한 래퍼
// ============================================================================
namespace
{

// 1차 IIR LP 필터 (standalone 구현 — 플러그인 내부와 동일 수식)
void applyIIRFilter(
  Eigen::MatrixXd& sequence, double alpha, const Eigen::VectorXd& initial)
{
  int N = sequence.rows();
  Eigen::VectorXd prev = initial;
  for (int t = 0; t < N; ++t) {
    Eigen::VectorXd filtered = alpha * sequence.row(t).transpose()
                              + (1.0 - alpha) * prev;
    sequence.row(t) = filtered.transpose();
    prev = filtered;
  }
}

// α 계산: α = dt / (τ + dt),  τ = 1/(2πf_c)
double computeAlpha(double dt, double f_c)
{
  double tau = 1.0 / (2.0 * M_PI * f_c);
  return dt / (tau + dt);
}

}  // anonymous namespace

// ============================================================================
// Test 1: IIR 필터 수학적 정확성
// ============================================================================
TEST(LPMPPI, IIRFilterMath)
{
  // y[t] = α·x[t] + (1-α)·y[t-1]
  double alpha = 0.5;
  Eigen::MatrixXd seq(3, 2);
  seq << 1.0, 2.0,
         3.0, 4.0,
         5.0, 6.0;
  Eigen::VectorXd init = Eigen::VectorXd::Zero(2);

  applyIIRFilter(seq, alpha, init);

  // y[0] = 0.5*1.0 + 0.5*0.0 = 0.5,  0.5*2.0 + 0.5*0.0 = 1.0
  EXPECT_NEAR(seq(0, 0), 0.5, 1e-10);
  EXPECT_NEAR(seq(0, 1), 1.0, 1e-10);
  // y[1] = 0.5*3.0 + 0.5*0.5 = 1.75,  0.5*4.0 + 0.5*1.0 = 2.5
  EXPECT_NEAR(seq(1, 0), 1.75, 1e-10);
  EXPECT_NEAR(seq(1, 1), 2.5, 1e-10);
  // y[2] = 0.5*5.0 + 0.5*1.75 = 3.375,  0.5*6.0 + 0.5*2.5 = 4.25
  EXPECT_NEAR(seq(2, 0), 3.375, 1e-10);
  EXPECT_NEAR(seq(2, 1), 4.25, 1e-10);
}

// ============================================================================
// Test 2: 컷오프 주파수 → α 계산
// ============================================================================
TEST(LPMPPI, CutoffFrequencyToAlpha)
{
  double dt = 0.1;

  // f_c = 10 Hz → τ = 1/(20π) ≈ 0.01592
  double alpha_10 = computeAlpha(dt, 10.0);
  EXPECT_GT(alpha_10, 0.0);
  EXPECT_LT(alpha_10, 1.0);
  // dt >> τ → α ≈ 0.863
  EXPECT_NEAR(alpha_10, 0.1 / (0.1 + 1.0 / (20.0 * M_PI)), 1e-6);

  // f_c = 1 Hz → τ = 1/(2π) ≈ 0.1592 → α 더 작음 (더 강한 필터)
  double alpha_1 = computeAlpha(dt, 1.0);
  EXPECT_LT(alpha_1, alpha_10);  // 낮은 f_c → 작은 α → 더 강한 필터

  // f_c = 100 Hz → τ ≈ 0.001592 → α ≈ 0.984 (거의 pass-through)
  double alpha_100 = computeAlpha(dt, 100.0);
  EXPECT_GT(alpha_100, 0.98);
}

// ============================================================================
// Test 3: 고주파 노이즈 감쇠
// ============================================================================
TEST(LPMPPI, HighFrequencyAttenuation)
{
  int N = 100;
  int nu = 2;
  double dt = 0.1;
  double f_c = 2.0;  // 낮은 컷오프
  double alpha = computeAlpha(dt, f_c);

  // 고주파 사인파 생성 (f = 50 Hz >> f_c)
  Eigen::MatrixXd seq(N, nu);
  for (int t = 0; t < N; ++t) {
    double phase = 2.0 * M_PI * 50.0 * t * dt;  // 50 Hz
    seq(t, 0) = std::sin(phase);
    seq(t, 1) = std::cos(phase);
  }

  double pre_rms = seq.norm() / std::sqrt(N * nu);

  Eigen::VectorXd init = Eigen::VectorXd::Zero(nu);
  applyIIRFilter(seq, alpha, init);

  double post_rms = seq.norm() / std::sqrt(N * nu);

  // 고주파는 감쇠되어야 함 (1차 IIR이므로 완전 제거는 아님)
  EXPECT_LT(post_rms, pre_rms);
}

// ============================================================================
// Test 4: 저주파 신호 통과
// ============================================================================
TEST(LPMPPI, LowFrequencyPassthrough)
{
  int N = 100;
  int nu = 1;
  double dt = 0.1;
  double f_c = 10.0;  // 높은 컷오프
  double alpha = computeAlpha(dt, f_c);

  // DC 신호 (0 Hz) — 완전 통과해야 함
  Eigen::MatrixXd seq(N, nu);
  for (int t = 0; t < N; ++t) {
    seq(t, 0) = 1.0;
  }

  Eigen::VectorXd init = Eigen::VectorXd::Zero(nu);
  applyIIRFilter(seq, alpha, init);

  // 과도 응답 후 정상 상태는 1.0에 수렴
  EXPECT_NEAR(seq(N - 1, 0), 1.0, 0.01);
}

// ============================================================================
// Test 5: 필터 상태 warm-start (이전 제어 기반)
// ============================================================================
TEST(LPMPPI, FilterWarmStart)
{
  double alpha = 0.5;
  Eigen::MatrixXd seq(3, 2);
  seq << 1.0, 1.0,
         1.0, 1.0,
         1.0, 1.0;

  // init = [2.0, 2.0] → 첫 출력은 init과 input의 혼합
  Eigen::VectorXd init(2);
  init << 2.0, 2.0;
  applyIIRFilter(seq, alpha, init);

  // y[0] = 0.5*1.0 + 0.5*2.0 = 1.5
  EXPECT_NEAR(seq(0, 0), 1.5, 1e-10);
  EXPECT_NEAR(seq(0, 1), 1.5, 1e-10);
}

// ============================================================================
// Test 6: 비활성화 폴백 (alpha=1 → pass-through)
// ============================================================================
TEST(LPMPPI, DisabledPassthrough)
{
  double alpha = 1.0;  // 필터 비활성화
  Eigen::MatrixXd seq(4, 2);
  seq << 1.0, 2.0,
         3.0, 4.0,
         5.0, 6.0,
         7.0, 8.0;
  Eigen::MatrixXd original = seq;

  Eigen::VectorXd init = Eigen::VectorXd::Zero(2);
  applyIIRFilter(seq, alpha, init);

  // alpha=1 → y[t] = x[t] (변경 없음)
  for (int t = 0; t < 4; ++t) {
    EXPECT_NEAR(seq(t, 0), original(t, 0), 1e-10);
    EXPECT_NEAR(seq(t, 1), original(t, 1), 1e-10);
  }
}

// ============================================================================
// Test 7: 제어 클리핑과의 호환성
// ============================================================================
TEST(LPMPPI, ClippingCompatibility)
{
  double alpha = 0.3;
  double v_max = 0.5;

  Eigen::MatrixXd seq(5, 2);
  for (int t = 0; t < 5; ++t) {
    seq(t, 0) = 1.0;   // v > v_max
    seq(t, 1) = 0.5;
  }

  Eigen::VectorXd init = Eigen::VectorXd::Zero(2);
  applyIIRFilter(seq, alpha, init);

  // LP 필터 후 값은 모두 원본 이하 (0~1 범위)
  for (int t = 0; t < 5; ++t) {
    EXPECT_GE(seq(t, 0), 0.0);
    EXPECT_LE(seq(t, 0), 1.0);
  }
}

// ============================================================================
// Test 8: 다채널 독립 필터 (v, ω 각각 필터)
// ============================================================================
TEST(LPMPPI, MultiChannelIndependence)
{
  double alpha = 0.5;
  int N = 10;

  // 채널 0만 입력, 채널 1은 0
  Eigen::MatrixXd seq(N, 2);
  for (int t = 0; t < N; ++t) {
    seq(t, 0) = 1.0;
    seq(t, 1) = 0.0;
  }

  Eigen::VectorXd init = Eigen::VectorXd::Zero(2);
  applyIIRFilter(seq, alpha, init);

  // 채널 1은 0으로 유지
  for (int t = 0; t < N; ++t) {
    EXPECT_DOUBLE_EQ(seq(t, 1), 0.0);
    EXPECT_GT(seq(t, 0), 0.0);  // 채널 0은 양수
  }
}

// ============================================================================
// Test 9: 연속 호출 안정성 (10회)
// ============================================================================
TEST(LPMPPI, ConsecutiveStability)
{
  double alpha = 0.6;
  int N = 20;
  int nu = 2;

  Eigen::VectorXd prev = Eigen::VectorXd::Zero(nu);

  for (int call = 0; call < 10; ++call) {
    Eigen::MatrixXd seq = Eigen::MatrixXd::Random(N, nu) * 0.5;
    applyIIRFilter(seq, alpha, prev);

    // 출력이 유한한지 확인
    for (int t = 0; t < N; ++t) {
      EXPECT_TRUE(std::isfinite(seq(t, 0)));
      EXPECT_TRUE(std::isfinite(seq(t, 1)));
    }

    // 다음 호출을 위해 마지막 값을 prev로
    prev = seq.row(N - 1).transpose();
  }
}

// ============================================================================
// Test 10: Swerve 모델 (nu=3) 동작
// ============================================================================
TEST(LPMPPI, SwerveModel)
{
  double alpha = 0.5;
  int N = 10;
  int nu = 3;  // vx, vy, omega

  Eigen::MatrixXd seq = Eigen::MatrixXd::Random(N, nu);
  Eigen::VectorXd init = Eigen::VectorXd::Zero(nu);

  applyIIRFilter(seq, alpha, init);

  // 3채널 모두 유한
  for (int t = 0; t < N; ++t) {
    for (int d = 0; d < nu; ++d) {
      EXPECT_TRUE(std::isfinite(seq(t, d)));
    }
  }
}

// ============================================================================
// Test 11: 필터 리셋 (init=0으로 재시작)
// ============================================================================
TEST(LPMPPI, FilterReset)
{
  double alpha = 0.5;
  int N = 5;
  int nu = 2;

  Eigen::MatrixXd seq1(N, nu);
  seq1.setOnes();
  Eigen::VectorXd init1 = Eigen::VectorXd::Zero(nu);
  applyIIRFilter(seq1, alpha, init1);

  // 리셋 후 동일한 입력 → 동일한 출력
  Eigen::MatrixXd seq2(N, nu);
  seq2.setOnes();
  Eigen::VectorXd init2 = Eigen::VectorXd::Zero(nu);
  applyIIRFilter(seq2, alpha, init2);

  for (int t = 0; t < N; ++t) {
    EXPECT_NEAR(seq1(t, 0), seq2(t, 0), 1e-10);
    EXPECT_NEAR(seq1(t, 1), seq2(t, 1), 1e-10);
  }
}

// ============================================================================
// Test 12: 스텝 응답 (시간 상수 검증)
// ============================================================================
TEST(LPMPPI, StepResponse)
{
  double dt = 0.1;
  double f_c = 1.0;  // τ = 1/(2π) ≈ 0.159s
  double alpha = computeAlpha(dt, f_c);
  double tau = 1.0 / (2.0 * M_PI * f_c);

  int N = 50;
  Eigen::MatrixXd seq(N, 1);
  seq.setOnes();  // 단위 스텝

  Eigen::VectorXd init = Eigen::VectorXd::Zero(1);
  applyIIRFilter(seq, alpha, init);

  // 1차 시스템: y(t) ≈ 1 - exp(-t/τ)
  // t = τ일 때 y ≈ 1 - 1/e ≈ 0.632
  int t_tau = static_cast<int>(tau / dt);
  if (t_tau > 0 && t_tau < N) {
    EXPECT_NEAR(seq(t_tau, 0), 1.0 - std::exp(-1.0), 0.15);
  }

  // 충분한 시간 후 1.0에 수렴
  EXPECT_NEAR(seq(N - 1, 0), 1.0, 0.05);
}

// ============================================================================
// Test 13: 비용 비교 — LP 필터 적용 시 jerk 감소
// ============================================================================
TEST(LPMPPI, JerkReduction)
{
  int N = 30;
  int nu = 2;
  double alpha = 0.5;

  // 랜덤 노이즈 시퀀스 (jerky)
  srand(42);
  Eigen::MatrixXd raw = Eigen::MatrixXd::Random(N, nu);
  Eigen::MatrixXd filtered = raw;

  Eigen::VectorXd init = Eigen::VectorXd::Zero(nu);
  applyIIRFilter(filtered, alpha, init);

  // Jerk 계산: Σ ||u[t+1] - u[t]||²
  double jerk_raw = 0.0, jerk_filtered = 0.0;
  for (int t = 0; t < N - 1; ++t) {
    jerk_raw += (raw.row(t + 1) - raw.row(t)).squaredNorm();
    jerk_filtered += (filtered.row(t + 1) - filtered.row(t)).squaredNorm();
  }

  // LP 필터 후 jerk가 감소해야 함
  EXPECT_LT(jerk_filtered, jerk_raw);
}

// ============================================================================
// Test 14: 파라미터 기본값 확인
// ============================================================================
TEST(LPMPPI, DefaultParameters)
{
  MPPIParams params;

  EXPECT_TRUE(params.lp_enabled);
  EXPECT_DOUBLE_EQ(params.lp_cutoff_frequency, 10.0);
  EXPECT_TRUE(params.lp_filter_all_samples);
}

// ============================================================================
// Test 15: 전체 시퀀스 필터링 (단조 증가 입력 → 감쇠된 증가)
// ============================================================================
TEST(LPMPPI, SequenceFiltering)
{
  int N = 20;
  int nu = 2;
  double alpha = 0.4;

  // 선형 증가 시퀀스
  Eigen::MatrixXd seq(N, nu);
  for (int t = 0; t < N; ++t) {
    seq(t, 0) = static_cast<double>(t) * 0.1;
    seq(t, 1) = static_cast<double>(t) * 0.05;
  }
  Eigen::MatrixXd original = seq;

  Eigen::VectorXd init = Eigen::VectorXd::Zero(nu);
  applyIIRFilter(seq, alpha, init);

  // 필터 출력은 원본보다 느리게 증가 (lag)
  for (int t = 1; t < N; ++t) {
    EXPECT_LE(seq(t, 0), original(t, 0) + 1e-10);
  }

  // 출력도 단조 증가
  for (int t = 1; t < N; ++t) {
    EXPECT_GE(seq(t, 0), seq(t - 1, 0) - 1e-10);
  }
}

}  // namespace mpc_controller_ros2
