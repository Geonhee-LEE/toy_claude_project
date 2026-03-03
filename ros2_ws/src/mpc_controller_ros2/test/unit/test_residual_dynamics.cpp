#include <gtest/gtest.h>
#include <Eigen/Dense>
#include <fstream>
#include <cmath>
#include <chrono>

#include "mpc_controller_ros2/eigen_mlp.hpp"
#include "mpc_controller_ros2/residual_dynamics_model.hpp"
#include "mpc_controller_ros2/motion_model_factory.hpp"
#include "mpc_controller_ros2/diff_drive_model.hpp"
#include "mpc_controller_ros2/swerve_drive_model.hpp"
#include "mpc_controller_ros2/ackermann_model.hpp"
#include "mpc_controller_ros2/mppi_params.hpp"

using namespace mpc_controller_ros2;

// =============================================================================
// Helper: 테스트용 EigenMLP 직접 생성
// =============================================================================

static std::unique_ptr<EigenMLP> createTestMLP(int in_dim, int out_dim,
                                                int hidden = 32) {
  // 2-layer MLP: in → hidden → out
  EigenMLP::LayerParams layer1;
  layer1.weight = Eigen::MatrixXd::Random(hidden, in_dim) * 0.1;
  layer1.bias = Eigen::VectorXd::Zero(hidden);

  EigenMLP::LayerParams layer2;
  layer2.weight = Eigen::MatrixXd::Random(out_dim, hidden) * 0.01;
  layer2.bias = Eigen::VectorXd::Zero(out_dim);

  EigenMLP::NormParams norm;
  norm.in_mean = Eigen::VectorXd::Zero(in_dim);
  norm.in_std = Eigen::VectorXd::Ones(in_dim);
  norm.out_mean = Eigen::VectorXd::Zero(out_dim);
  norm.out_std = Eigen::VectorXd::Ones(out_dim);

  return std::make_unique<EigenMLP>(
    std::vector<EigenMLP::LayerParams>{layer1, layer2}, norm);
}

// =============================================================================
// Helper: 바이너리 파일로 MLP 저장
// =============================================================================

static void saveTestMLPFile(const std::string& path, int in_dim, int out_dim,
                            int hidden = 32) {
  std::ofstream file(path, std::ios::binary);

  uint32_t magic = 0x454D4C50;  // "EMLP"
  uint32_t version = 1;
  uint32_t n_layers = 2;
  file.write(reinterpret_cast<const char*>(&magic), sizeof(magic));
  file.write(reinterpret_cast<const char*>(&version), sizeof(version));
  file.write(reinterpret_cast<const char*>(&n_layers), sizeof(n_layers));

  // Norm params
  uint32_t in_d = in_dim, out_d = out_dim;
  file.write(reinterpret_cast<const char*>(&in_d), sizeof(in_d));
  file.write(reinterpret_cast<const char*>(&out_d), sizeof(out_d));

  Eigen::VectorXd in_mean = Eigen::VectorXd::Zero(in_dim);
  Eigen::VectorXd in_std = Eigen::VectorXd::Ones(in_dim);
  Eigen::VectorXd out_mean = Eigen::VectorXd::Zero(out_dim);
  Eigen::VectorXd out_std = Eigen::VectorXd::Ones(out_dim);

  file.write(reinterpret_cast<const char*>(in_mean.data()), in_dim * sizeof(double));
  file.write(reinterpret_cast<const char*>(in_std.data()), in_dim * sizeof(double));
  file.write(reinterpret_cast<const char*>(out_mean.data()), out_dim * sizeof(double));
  file.write(reinterpret_cast<const char*>(out_std.data()), out_dim * sizeof(double));

  // Layer 1: hidden x in_dim
  Eigen::MatrixXd w1 = Eigen::MatrixXd::Random(hidden, in_dim) * 0.1;
  Eigen::VectorXd b1 = Eigen::VectorXd::Zero(hidden);
  uint32_t rows1 = hidden, cols1 = in_dim;
  file.write(reinterpret_cast<const char*>(&rows1), sizeof(rows1));
  file.write(reinterpret_cast<const char*>(&cols1), sizeof(cols1));
  file.write(reinterpret_cast<const char*>(w1.data()), hidden * in_dim * sizeof(double));
  file.write(reinterpret_cast<const char*>(b1.data()), hidden * sizeof(double));

  // Layer 2: out_dim x hidden
  Eigen::MatrixXd w2 = Eigen::MatrixXd::Random(out_dim, hidden) * 0.01;
  Eigen::VectorXd b2 = Eigen::VectorXd::Zero(out_dim);
  uint32_t rows2 = out_dim, cols2 = hidden;
  file.write(reinterpret_cast<const char*>(&rows2), sizeof(rows2));
  file.write(reinterpret_cast<const char*>(&cols2), sizeof(cols2));
  file.write(reinterpret_cast<const char*>(w2.data()), out_dim * hidden * sizeof(double));
  file.write(reinterpret_cast<const char*>(b2.data()), out_dim * sizeof(double));
}

// =============================================================================
// EigenMLP 테스트 (6개)
// =============================================================================

TEST(EigenMLP, ForwardSingle) {
  auto mlp = createTestMLP(5, 3);
  Eigen::VectorXd input = Eigen::VectorXd::Random(5);
  Eigen::VectorXd output = mlp->forward(input);

  EXPECT_EQ(output.size(), 3);
  // 출력이 유한한 값인지 확인
  for (int i = 0; i < output.size(); ++i) {
    EXPECT_TRUE(std::isfinite(output(i)));
  }
}

TEST(EigenMLP, ForwardBatch) {
  auto mlp = createTestMLP(5, 3);
  int K = 128;
  Eigen::MatrixXd inputs = Eigen::MatrixXd::Random(K, 5);
  Eigen::MatrixXd outputs = mlp->forwardBatch(inputs);

  EXPECT_EQ(outputs.rows(), K);
  EXPECT_EQ(outputs.cols(), 3);

  // 배치와 단일 결과가 동일한지 확인
  for (int k = 0; k < std::min(5, K); ++k) {
    Eigen::VectorXd single_out = mlp->forward(inputs.row(k).transpose());
    EXPECT_NEAR((outputs.row(k).transpose() - single_out).norm(), 0.0, 1e-10)
      << "Batch output mismatch at k=" << k;
  }
}

TEST(EigenMLP, ZScoreNorm) {
  // 비균일 정규화 테스트
  EigenMLP::LayerParams layer;
  layer.weight = Eigen::MatrixXd::Identity(3, 3);  // 항등 변환
  layer.bias = Eigen::VectorXd::Zero(3);

  EigenMLP::NormParams norm;
  norm.in_mean = Eigen::Vector3d(1.0, 2.0, 3.0);
  norm.in_std = Eigen::Vector3d(2.0, 0.5, 1.0);
  norm.out_mean = Eigen::Vector3d(10.0, 20.0, 30.0);
  norm.out_std = Eigen::Vector3d(5.0, 10.0, 1.0);

  EigenMLP mlp({layer}, norm);

  Eigen::Vector3d input(1.0, 2.0, 3.0);  // 정규화 후 (0, 0, 0)
  Eigen::Vector3d output = mlp.forward(input);
  // 항등 변환 → 정규화 입력 = (0,0,0) → 역정규화 = out_mean
  EXPECT_NEAR(output(0), 10.0, 1e-10);
  EXPECT_NEAR(output(1), 20.0, 1e-10);
  EXPECT_NEAR(output(2), 30.0, 1e-10);
}

TEST(EigenMLP, ReLU) {
  // ReLU 활성화 확인 (중간 레이어)
  EigenMLP::LayerParams layer1;
  layer1.weight = -Eigen::MatrixXd::Identity(3, 3);  // 부호 반전
  layer1.bias = Eigen::VectorXd::Zero(3);

  EigenMLP::LayerParams layer2;
  layer2.weight = Eigen::MatrixXd::Identity(3, 3);
  layer2.bias = Eigen::VectorXd::Zero(3);

  EigenMLP::NormParams norm;
  norm.in_mean = Eigen::VectorXd::Zero(3);
  norm.in_std = Eigen::VectorXd::Ones(3);
  norm.out_mean = Eigen::VectorXd::Zero(3);
  norm.out_std = Eigen::VectorXd::Ones(3);

  EigenMLP mlp({layer1, layer2}, norm);

  Eigen::Vector3d input(1.0, -2.0, 3.0);
  Eigen::Vector3d output = mlp.forward(input);

  // layer1: (-1, 2, -3) → ReLU → (0, 2, 0) → layer2: (0, 2, 0)
  EXPECT_NEAR(output(0), 0.0, 1e-10);
  EXPECT_NEAR(output(1), 2.0, 1e-10);
  EXPECT_NEAR(output(2), 0.0, 1e-10);
}

TEST(EigenMLP, LoadFromFile) {
  std::string path = "/tmp/test_eigen_mlp.bin";
  int in_dim = 5, out_dim = 3, hidden = 16;
  saveTestMLPFile(path, in_dim, out_dim, hidden);

  auto mlp = EigenMLP::loadFromFile(path);
  EXPECT_EQ(mlp->inputDim(), in_dim);
  EXPECT_EQ(mlp->outputDim(), out_dim);
  EXPECT_EQ(mlp->numLayers(), 2);

  // 순전파 확인
  Eigen::VectorXd input = Eigen::VectorXd::Random(in_dim);
  Eigen::VectorXd output = mlp->forward(input);
  EXPECT_EQ(output.size(), out_dim);

  std::remove(path.c_str());
}

TEST(EigenMLP, InvalidFile) {
  // 존재하지 않는 파일
  EXPECT_THROW(EigenMLP::loadFromFile("/tmp/nonexistent_mlp.bin"),
               std::runtime_error);

  // 잘못된 매직 넘버
  std::string path = "/tmp/bad_magic_mlp.bin";
  {
    std::ofstream f(path, std::ios::binary);
    uint32_t bad_magic = 0xDEADBEEF;
    f.write(reinterpret_cast<const char*>(&bad_magic), sizeof(bad_magic));
  }
  EXPECT_THROW(EigenMLP::loadFromFile(path), std::runtime_error);
  std::remove(path.c_str());
}

// =============================================================================
// ResidualDynamicsModel 테스트 (6개)
// =============================================================================

TEST(ResidualModel, DelegateDims) {
  auto nominal = std::make_unique<DiffDriveModel>(0.0, 1.0, -1.0, 1.0);
  auto mlp = createTestMLP(5, 3);  // nx+nu=3+2=5, out=nx=3

  ResidualDynamicsModel model(std::move(nominal), std::move(mlp), 0.5);

  EXPECT_EQ(model.stateDim(), 3);
  EXPECT_EQ(model.controlDim(), 2);
  EXPECT_FALSE(model.isHolonomic());
  EXPECT_EQ(model.name(), "residual_diff_drive");
}

TEST(ResidualModel, AlphaZero) {
  // alpha=0 → 공칭 모델과 동일
  auto nominal = std::make_unique<DiffDriveModel>(0.0, 1.0, -1.0, 1.0);
  auto mlp = createTestMLP(5, 3);

  ResidualDynamicsModel model(std::move(nominal), std::move(mlp), 0.0);

  // DiffDrive 공칭 동역학
  auto nominal_only = std::make_unique<DiffDriveModel>(0.0, 1.0, -1.0, 1.0);

  int M = 10;
  Eigen::MatrixXd states = Eigen::MatrixXd::Random(M, 3);
  Eigen::MatrixXd controls = Eigen::MatrixXd::Random(M, 2);

  Eigen::MatrixXd residual_out = model.dynamicsBatch(states, controls);
  Eigen::MatrixXd nominal_out = nominal_only->dynamicsBatch(states, controls);

  EXPECT_NEAR((residual_out - nominal_out).norm(), 0.0, 1e-10);
}

TEST(ResidualModel, ResidualAdds) {
  // alpha=1 → 공칭 + 잔차
  auto nominal = std::make_unique<DiffDriveModel>(0.0, 1.0, -1.0, 1.0);
  auto mlp = createTestMLP(5, 3);
  // MLP의 사본 생성 (같은 가중치)
  auto mlp_copy = createTestMLP(5, 3);

  ResidualDynamicsModel model(std::move(nominal), std::move(mlp), 1.0);

  auto nominal_only = std::make_unique<DiffDriveModel>(0.0, 1.0, -1.0, 1.0);

  int M = 10;
  Eigen::MatrixXd states = Eigen::MatrixXd::Random(M, 3);
  Eigen::MatrixXd controls = Eigen::MatrixXd::Random(M, 2);

  Eigen::MatrixXd residual_out = model.dynamicsBatch(states, controls);
  Eigen::MatrixXd nominal_out = nominal_only->dynamicsBatch(states, controls);

  // 잔차 부분이 0이 아니어야 함 (MLP 출력이 0이 아닌 한)
  Eigen::MatrixXd diff = residual_out - nominal_out;
  // 랜덤 가중치이므로 차이가 있을 것
  // (특이한 경우 정확히 0일 수 있지만 확률적으로 매우 낮음)
  EXPECT_GT(diff.norm(), 0.0);
}

TEST(ResidualModel, DelegateClip) {
  auto nominal = std::make_unique<DiffDriveModel>(0.0, 1.0, -1.0, 1.0);
  auto mlp = createTestMLP(5, 3);

  ResidualDynamicsModel model(std::move(nominal), std::move(mlp));

  // 클리핑이 nominal에 위임되는지 확인
  Eigen::MatrixXd controls(2, 2);
  controls << 5.0, 5.0,  // v=5 (max=1), omega=5 (max=1)
             -5.0, -5.0; // v=-5 (min=0), omega=-5 (min=-1)
  Eigen::MatrixXd clipped = model.clipControls(controls);
  EXPECT_LE(clipped.col(0).maxCoeff(), 1.0);
  EXPECT_GE(clipped.col(0).minCoeff(), 0.0);
}

TEST(ResidualModel, DelegateNormalize) {
  auto nominal = std::make_unique<DiffDriveModel>(0.0, 1.0, -1.0, 1.0);
  auto mlp = createTestMLP(5, 3);

  ResidualDynamicsModel model(std::move(nominal), std::move(mlp));

  Eigen::MatrixXd states(1, 3);
  states << 0.0, 0.0, 4.0;  // theta=4.0 (> π)
  model.normalizeStates(states);
  EXPECT_LT(std::abs(states(0, 2)), M_PI + 0.01);
}

TEST(ResidualModel, AllModels) {
  // DiffDrive, Swerve, Ackermann 모두 테스트
  MPPIParams params;

  // DiffDrive (nx=3, nu=2)
  {
    auto nominal = std::make_unique<DiffDriveModel>(0.0, 1.0, -1.0, 1.0);
    auto mlp = createTestMLP(5, 3);  // 3+2=5
    EXPECT_NO_THROW(
      ResidualDynamicsModel(std::move(nominal), std::move(mlp)));
  }

  // Swerve (nx=3, nu=3)
  {
    auto nominal = std::make_unique<SwerveDriveModel>(0.0, 1.0, 0.5, 1.0);
    auto mlp = createTestMLP(6, 3);  // 3+3=6
    EXPECT_NO_THROW(
      ResidualDynamicsModel(std::move(nominal), std::move(mlp)));
  }

  // Ackermann (nx=4, nu=2)
  {
    auto nominal = std::make_unique<AckermannModel>(0.0, 1.0, 2.0, M_PI/3, 0.5);
    auto mlp = createTestMLP(6, 4);  // 4+2=6
    EXPECT_NO_THROW(
      ResidualDynamicsModel(std::move(nominal), std::move(mlp)));
  }

  // 차원 불일치 → 예외
  {
    auto nominal = std::make_unique<DiffDriveModel>(0.0, 1.0, -1.0, 1.0);
    auto mlp = createTestMLP(10, 3);  // 10 != 5
    EXPECT_THROW(
      ResidualDynamicsModel(std::move(nominal), std::move(mlp)),
      std::invalid_argument);
  }
}

// =============================================================================
// Integration 테스트 (3개)
// =============================================================================

TEST(ResidualIntegration, FactoryCreate) {
  // 바이너리 파일 저장 → createWithResidual
  std::string path = "/tmp/test_factory_residual.bin";
  saveTestMLPFile(path, 5, 3, 32);  // DiffDrive: nx+nu=5, out=3

  MPPIParams params;
  params.motion_model = "diff_drive";
  params.residual_enabled = true;
  params.residual_weights_path = path;
  params.residual_alpha = 0.8;

  auto model = MotionModelFactory::createWithResidual("diff_drive", params);
  EXPECT_EQ(model->stateDim(), 3);
  EXPECT_EQ(model->controlDim(), 2);
  EXPECT_EQ(model->name(), "residual_diff_drive");

  std::remove(path.c_str());
}

TEST(ResidualIntegration, RolloutValid) {
  auto nominal = std::make_unique<DiffDriveModel>(0.0, 1.0, -1.0, 1.0);
  auto mlp = createTestMLP(5, 3);

  auto model = std::make_unique<ResidualDynamicsModel>(
    std::move(nominal), std::move(mlp), 0.5);

  // Rollout 테스트
  Eigen::VectorXd x0(3);
  x0 << 0.0, 0.0, 0.0;

  int K = 16, N = 10;
  std::vector<Eigen::MatrixXd> control_seqs(K);
  for (int k = 0; k < K; ++k) {
    control_seqs[k] = Eigen::MatrixXd::Random(N, 2) * 0.5;
  }

  auto trajectories = model->rolloutBatch(x0, control_seqs, 0.1);
  EXPECT_EQ(static_cast<int>(trajectories.size()), K);

  for (int k = 0; k < K; ++k) {
    EXPECT_EQ(trajectories[k].rows(), N + 1);
    EXPECT_EQ(trajectories[k].cols(), 3);
    // 초기 상태 확인
    EXPECT_NEAR(trajectories[k](0, 0), 0.0, 1e-10);
    EXPECT_NEAR(trajectories[k](0, 1), 0.0, 1e-10);
    // 모든 값이 유한한지 확인
    for (int t = 0; t <= N; ++t) {
      for (int j = 0; j < 3; ++j) {
        EXPECT_TRUE(std::isfinite(trajectories[k](t, j)));
      }
    }
  }
}

TEST(ResidualIntegration, PerfBudget) {
  // K=512 배치 MLP 추론이 1ms 이내인지 확인
  int in_dim = 7, out_dim = 4, hidden = 64;  // Ackermann 기준
  auto mlp = createTestMLP(in_dim, out_dim, hidden);

  int K = 512;
  Eigen::MatrixXd inputs = Eigen::MatrixXd::Random(K, in_dim);

  // Warm-up
  mlp->forwardBatch(inputs);

  auto start = std::chrono::high_resolution_clock::now();
  int N_iter = 100;
  for (int i = 0; i < N_iter; ++i) {
    mlp->forwardBatch(inputs);
  }
  auto end = std::chrono::high_resolution_clock::now();
  double ms_per_call = std::chrono::duration<double, std::milli>(end - start).count() / N_iter;

  // MLP 추론이 1ms 이내
  EXPECT_LT(ms_per_call, 1.0)
    << "MLP batch inference too slow: " << ms_per_call << " ms";
}

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
