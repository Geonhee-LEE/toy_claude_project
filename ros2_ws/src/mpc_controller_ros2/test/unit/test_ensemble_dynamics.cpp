#include <gtest/gtest.h>
#include <Eigen/Dense>
#include <fstream>
#include <cmath>
#include <chrono>
#include <filesystem>

#include "mpc_controller_ros2/eigen_mlp.hpp"
#include "mpc_controller_ros2/ensemble_dynamics_model.hpp"
#include "mpc_controller_ros2/online_data_buffer.hpp"
#include "mpc_controller_ros2/cost_functions.hpp"
#include "mpc_controller_ros2/motion_model_factory.hpp"
#include "mpc_controller_ros2/diff_drive_model.hpp"
#include "mpc_controller_ros2/mppi_params.hpp"

using namespace mpc_controller_ros2;

// =============================================================================
// Helper: 테스트용 EigenMLP 직접 생성
// =============================================================================

static std::unique_ptr<EigenMLP> createTestMLP(int in_dim, int out_dim,
                                                int hidden = 32) {
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

static void saveTestMLPFile(const std::string& path, int in_dim, int out_dim,
                            int hidden = 32) {
  std::ofstream file(path, std::ios::binary);

  uint32_t magic = 0x454D4C50;
  uint32_t version = 1;
  uint32_t n_layers = 2;
  file.write(reinterpret_cast<const char*>(&magic), sizeof(magic));
  file.write(reinterpret_cast<const char*>(&version), sizeof(version));
  file.write(reinterpret_cast<const char*>(&n_layers), sizeof(n_layers));

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

  Eigen::MatrixXd w1 = Eigen::MatrixXd::Random(hidden, in_dim) * 0.1;
  Eigen::VectorXd b1 = Eigen::VectorXd::Zero(hidden);
  uint32_t rows1 = hidden, cols1 = in_dim;
  file.write(reinterpret_cast<const char*>(&rows1), sizeof(rows1));
  file.write(reinterpret_cast<const char*>(&cols1), sizeof(cols1));
  file.write(reinterpret_cast<const char*>(w1.data()), hidden * in_dim * sizeof(double));
  file.write(reinterpret_cast<const char*>(b1.data()), hidden * sizeof(double));

  Eigen::MatrixXd w2 = Eigen::MatrixXd::Random(out_dim, hidden) * 0.01;
  Eigen::VectorXd b2 = Eigen::VectorXd::Zero(out_dim);
  uint32_t rows2 = out_dim, cols2 = hidden;
  file.write(reinterpret_cast<const char*>(&rows2), sizeof(rows2));
  file.write(reinterpret_cast<const char*>(&cols2), sizeof(cols2));
  file.write(reinterpret_cast<const char*>(w2.data()), out_dim * hidden * sizeof(double));
  file.write(reinterpret_cast<const char*>(b2.data()), out_dim * sizeof(double));
}

// =============================================================================
// EnsembleDynamicsModel Tests
// =============================================================================

TEST(EnsembleDynamicsModel, Construction_ValidDimensions) {
  int nx = 3, nu = 2;
  auto nominal = std::make_unique<DiffDriveModel>(0.0, 1.0, -1.0, 1.0);

  std::vector<std::unique_ptr<EigenMLP>> ensemble;
  for (int i = 0; i < 5; ++i) {
    ensemble.push_back(createTestMLP(nx + nu, nx));
  }

  EnsembleDynamicsModel model(std::move(nominal), std::move(ensemble), 1.0);
  EXPECT_EQ(model.stateDim(), 3);
  EXPECT_EQ(model.controlDim(), 2);
  EXPECT_EQ(model.ensembleSize(), 5);
  EXPECT_EQ(model.name(), "ensemble_diff_drive");
}

TEST(EnsembleDynamicsModel, DynamicsBatch_MeanPrediction) {
  int nx = 3, nu = 2, K = 64;
  auto nominal = std::make_unique<DiffDriveModel>(0.0, 1.0, -1.0, 1.0);

  std::vector<std::unique_ptr<EigenMLP>> ensemble;
  for (int i = 0; i < 3; ++i) {
    ensemble.push_back(createTestMLP(nx + nu, nx));
  }

  EnsembleDynamicsModel model(std::move(nominal), std::move(ensemble), 1.0);

  Eigen::MatrixXd states = Eigen::MatrixXd::Random(K, nx);
  Eigen::MatrixXd controls = Eigen::MatrixXd::Random(K, nu);

  Eigen::MatrixXd x_dot = model.dynamicsBatch(states, controls);
  EXPECT_EQ(x_dot.rows(), K);
  EXPECT_EQ(x_dot.cols(), nx);

  // 유한 값 확인
  EXPECT_TRUE(x_dot.allFinite());
}

TEST(EnsembleDynamicsModel, PredictWithUncertainty_VariancePositive) {
  int nx = 3, nu = 2, K = 32;
  auto nominal = std::make_unique<DiffDriveModel>(0.0, 1.0, -1.0, 1.0);

  std::vector<std::unique_ptr<EigenMLP>> ensemble;
  for (int i = 0; i < 5; ++i) {
    ensemble.push_back(createTestMLP(nx + nu, nx));
  }

  EnsembleDynamicsModel model(std::move(nominal), std::move(ensemble), 1.0);

  Eigen::MatrixXd states = Eigen::MatrixXd::Random(K, nx);
  Eigen::MatrixXd controls = Eigen::MatrixXd::Random(K, nu);

  auto result = model.predictWithUncertainty(states, controls);

  EXPECT_EQ(result.mean.rows(), K);
  EXPECT_EQ(result.mean.cols(), nx);
  EXPECT_EQ(result.variance.rows(), K);
  EXPECT_EQ(result.variance.cols(), nx);

  // 분산은 음이 아님
  EXPECT_TRUE((result.variance.array() >= 0.0).all());
}

TEST(EnsembleDynamicsModel, IdenticalMLPs_ZeroVariance) {
  int nx = 3, nu = 2, K = 16;
  auto nominal = std::make_unique<DiffDriveModel>(0.0, 1.0, -1.0, 1.0);

  // 동일한 MLP 복제
  auto base_mlp = createTestMLP(nx + nu, nx);
  auto input = Eigen::VectorXd::Ones(nx + nu);
  auto expected = base_mlp->forward(input);

  // 같은 파라미터로 MLP 3개 생성 (정확히 같은 MLP)
  // 동일 시드 보장을 위해 같은 weights 재사용
  EigenMLP::LayerParams l1, l2;
  l1.weight = Eigen::MatrixXd::Ones(32, nx + nu) * 0.01;
  l1.bias = Eigen::VectorXd::Zero(32);
  l2.weight = Eigen::MatrixXd::Ones(nx, 32) * 0.01;
  l2.bias = Eigen::VectorXd::Zero(nx);
  EigenMLP::NormParams norm;
  norm.in_mean = Eigen::VectorXd::Zero(nx + nu);
  norm.in_std = Eigen::VectorXd::Ones(nx + nu);
  norm.out_mean = Eigen::VectorXd::Zero(nx);
  norm.out_std = Eigen::VectorXd::Ones(nx);

  std::vector<std::unique_ptr<EigenMLP>> ensemble;
  for (int i = 0; i < 3; ++i) {
    ensemble.push_back(std::make_unique<EigenMLP>(
      std::vector<EigenMLP::LayerParams>{l1, l2}, norm));
  }

  EnsembleDynamicsModel model(std::move(nominal), std::move(ensemble), 1.0);

  Eigen::MatrixXd states = Eigen::MatrixXd::Ones(K, nx);
  Eigen::MatrixXd controls = Eigen::MatrixXd::Ones(K, nu);

  auto result = model.predictWithUncertainty(states, controls);

  // 동일 MLP → 분산 = 0
  EXPECT_LT(result.variance.maxCoeff(), 1e-10);
}

TEST(EnsembleDynamicsModel, AlphaZero_NominalOnly) {
  int nx = 3, nu = 2, K = 16;
  auto nominal_raw = std::make_unique<DiffDriveModel>(0.0, 1.0, -1.0, 1.0);
  DiffDriveModel nominal_ref(0.0, 1.0, -1.0, 1.0);

  std::vector<std::unique_ptr<EigenMLP>> ensemble;
  for (int i = 0; i < 3; ++i) {
    ensemble.push_back(createTestMLP(nx + nu, nx));
  }

  EnsembleDynamicsModel model(std::move(nominal_raw), std::move(ensemble), 0.0);

  Eigen::MatrixXd states = Eigen::MatrixXd::Random(K, nx);
  Eigen::MatrixXd controls = Eigen::MatrixXd::Random(K, nu);

  Eigen::MatrixXd result = model.dynamicsBatch(states, controls);
  Eigen::MatrixXd expected = nominal_ref.dynamicsBatch(states, controls);

  EXPECT_LT((result - expected).norm(), 1e-10);
}

TEST(EnsembleDynamicsModel, DelegatesClipControls) {
  int nx = 3, nu = 2;
  auto nominal = std::make_unique<DiffDriveModel>(0.0, 1.0, -1.0, 1.0);

  std::vector<std::unique_ptr<EigenMLP>> ensemble;
  ensemble.push_back(createTestMLP(nx + nu, nx));

  EnsembleDynamicsModel model(std::move(nominal), std::move(ensemble), 1.0);

  Eigen::MatrixXd controls(2, nu);
  controls << 5.0, 5.0,
             -5.0, -5.0;

  auto clipped = model.clipControls(controls);
  EXPECT_LE(clipped.maxCoeff(), 1.0);
  EXPECT_GE(clipped.minCoeff(), -1.0);
}

TEST(EnsembleDynamicsModel, DelegatesNormalize) {
  int nx = 3, nu = 2;
  auto nominal = std::make_unique<DiffDriveModel>(0.0, 1.0, -1.0, 1.0);

  std::vector<std::unique_ptr<EigenMLP>> ensemble;
  ensemble.push_back(createTestMLP(nx + nu, nx));

  EnsembleDynamicsModel model(std::move(nominal), std::move(ensemble), 1.0);

  Eigen::MatrixXd states(1, nx);
  states << 0.0, 0.0, 4.0 * M_PI;

  model.normalizeStates(states);
  EXPECT_LT(std::abs(states(0, 2)), M_PI + 0.01);
}

TEST(EnsembleDynamicsModel, Factory_CreateWithEnsemble) {
  int nx = 3, nu = 2;
  std::string dir = "/tmp/test_ensemble_factory";
  std::filesystem::create_directories(dir);

  for (int i = 0; i < 3; ++i) {
    saveTestMLPFile(dir + "/model_" + std::to_string(i) + ".bin", nx + nu, nx);
  }

  MPPIParams params;
  params.ensemble_enabled = true;
  params.ensemble_weights_dir = dir;
  params.ensemble_size = 3;
  params.ensemble_alpha = 0.5;

  auto model = MotionModelFactory::createWithEnsemble("diff_drive", params);
  EXPECT_EQ(model->stateDim(), 3);
  EXPECT_EQ(model->controlDim(), 2);
  EXPECT_TRUE(model->name().find("ensemble") != std::string::npos);

  // 정리
  std::filesystem::remove_all(dir);
}

// =============================================================================
// UncertaintyAwareCost Tests
// =============================================================================

TEST(UncertaintyAwareCost, HighVariance_HighCost) {
  int nx = 3, nu = 2, K = 4, N = 10;
  auto nominal = std::make_unique<DiffDriveModel>(0.0, 1.0, -1.0, 1.0);

  std::vector<std::unique_ptr<EigenMLP>> ensemble;
  for (int i = 0; i < 5; ++i) {
    ensemble.push_back(createTestMLP(nx + nu, nx));
  }

  auto model = std::make_unique<EnsembleDynamicsModel>(
    std::move(nominal), std::move(ensemble), 1.0);
  auto* model_ptr = model.get();

  UncertaintyAwareCost cost(model_ptr, 10.0, 0.1);

  std::vector<Eigen::MatrixXd> trajectories(K);
  std::vector<Eigen::MatrixXd> controls(K);
  for (int k = 0; k < K; ++k) {
    trajectories[k] = Eigen::MatrixXd::Random(N + 1, nx);
    controls[k] = Eigen::MatrixXd::Random(N, nu);
  }
  Eigen::MatrixXd ref = Eigen::MatrixXd::Zero(N + 1, nx);

  Eigen::VectorXd costs = cost.compute(trajectories, controls, ref);
  EXPECT_EQ(costs.size(), K);
  // 랜덤 앙상블 → 분산 > 0 → 비용 > 0
  EXPECT_GT(costs.sum(), 0.0);
}

TEST(UncertaintyAwareCost, WeightZero_ZeroCost) {
  int nx = 3, nu = 2, K = 4, N = 5;
  auto nominal = std::make_unique<DiffDriveModel>(0.0, 1.0, -1.0, 1.0);

  std::vector<std::unique_ptr<EigenMLP>> ensemble;
  for (int i = 0; i < 3; ++i) {
    ensemble.push_back(createTestMLP(nx + nu, nx));
  }

  auto model = std::make_unique<EnsembleDynamicsModel>(
    std::move(nominal), std::move(ensemble), 1.0);

  UncertaintyAwareCost cost(model.get(), 0.0, 0.1);

  std::vector<Eigen::MatrixXd> trajectories(K);
  std::vector<Eigen::MatrixXd> controls(K);
  for (int k = 0; k < K; ++k) {
    trajectories[k] = Eigen::MatrixXd::Random(N + 1, nx);
    controls[k] = Eigen::MatrixXd::Random(N, nu);
  }
  Eigen::MatrixXd ref = Eigen::MatrixXd::Zero(N + 1, nx);

  Eigen::VectorXd costs = cost.compute(trajectories, controls, ref);
  EXPECT_DOUBLE_EQ(costs.sum(), 0.0);
}

TEST(UncertaintyAwareCost, IntegrationWithCompositeCost) {
  int nx = 3, nu = 2, K = 4, N = 5;
  auto nominal = std::make_unique<DiffDriveModel>(0.0, 1.0, -1.0, 1.0);

  std::vector<std::unique_ptr<EigenMLP>> ensemble;
  for (int i = 0; i < 3; ++i) {
    ensemble.push_back(createTestMLP(nx + nu, nx));
  }
  auto model = std::make_unique<EnsembleDynamicsModel>(
    std::move(nominal), std::move(ensemble), 1.0);

  CompositeMPPICost composite;
  composite.addCost(std::make_unique<UncertaintyAwareCost>(model.get(), 5.0, 0.1));

  std::vector<Eigen::MatrixXd> trajectories(K);
  std::vector<Eigen::MatrixXd> controls(K);
  for (int k = 0; k < K; ++k) {
    trajectories[k] = Eigen::MatrixXd::Random(N + 1, nx);
    controls[k] = Eigen::MatrixXd::Random(N, nu);
  }
  Eigen::MatrixXd ref = Eigen::MatrixXd::Zero(N + 1, nx);

  auto breakdown = composite.computeDetailed(trajectories, controls, ref);
  EXPECT_TRUE(breakdown.component_costs.count("uncertainty") > 0);
}

TEST(UncertaintyAwareCost, BatchCorrectShape) {
  int nx = 3, nu = 2, K = 8, N = 10;
  auto nominal = std::make_unique<DiffDriveModel>(0.0, 1.0, -1.0, 1.0);

  std::vector<std::unique_ptr<EigenMLP>> ensemble;
  for (int i = 0; i < 3; ++i) {
    ensemble.push_back(createTestMLP(nx + nu, nx));
  }
  auto model = std::make_unique<EnsembleDynamicsModel>(
    std::move(nominal), std::move(ensemble), 1.0);

  UncertaintyAwareCost cost(model.get(), 1.0, 0.1);

  std::vector<Eigen::MatrixXd> trajectories(K);
  std::vector<Eigen::MatrixXd> controls(K);
  for (int k = 0; k < K; ++k) {
    trajectories[k] = Eigen::MatrixXd::Random(N + 1, nx);
    controls[k] = Eigen::MatrixXd::Random(N, nu);
  }
  Eigen::MatrixXd ref = Eigen::MatrixXd::Zero(N + 1, nx);

  Eigen::VectorXd costs = cost.compute(trajectories, controls, ref);
  EXPECT_EQ(costs.size(), K);
  EXPECT_TRUE(costs.allFinite());
}

// =============================================================================
// OnlineDataBuffer Tests
// =============================================================================

TEST(OnlineDataBuffer, AddAndRetrieve_RingBuffer) {
  OnlineDataBuffer buffer(5);

  Eigen::VectorXd state(3), control(2), next_state(3);
  state << 1, 2, 3;
  control << 0.5, 0.1;
  next_state << 1.1, 2.1, 3.1;

  // 7개 추가 (capacity=5, 2개 덮어쓰기)
  for (int i = 0; i < 7; ++i) {
    buffer.add(state * (i + 1), control, next_state * (i + 1), 0.1);
  }

  EXPECT_EQ(buffer.size(), 5);
  EXPECT_TRUE(buffer.full());
}

TEST(OnlineDataBuffer, ExportCSV_Format) {
  OnlineDataBuffer buffer(100);

  Eigen::VectorXd state(3), control(2), next_state(3);
  state << 1.0, 2.0, 0.5;
  control << 0.3, 0.1;
  next_state << 1.03, 2.01, 0.51;

  buffer.add(state, control, next_state, 0.1);
  buffer.add(state * 2, control * 2, next_state * 2, 0.1);

  std::string path = "/tmp/test_online_data.csv";
  size_t count = buffer.exportCSV(path);
  EXPECT_EQ(count, 2);

  // CSV 파일 내용 확인
  std::ifstream ifs(path);
  std::string header;
  std::getline(ifs, header);
  EXPECT_TRUE(header.find("s0") != std::string::npos);
  EXPECT_TRUE(header.find("u0") != std::string::npos);
  EXPECT_TRUE(header.find("ns0") != std::string::npos);
  EXPECT_TRUE(header.find("dt") != std::string::npos);

  std::string line1;
  std::getline(ifs, line1);
  EXPECT_FALSE(line1.empty());

  std::filesystem::remove(path);
}

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
