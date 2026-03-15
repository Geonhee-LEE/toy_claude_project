#include <gtest/gtest.h>
#include <Eigen/Dense>
#include <filesystem>
#include <fstream>
#include <cmath>

#include "mpc_controller_ros2/online_data_collector.hpp"
#include "mpc_controller_ros2/model_reloader.hpp"
#include "mpc_controller_ros2/online_data_buffer.hpp"
#include "mpc_controller_ros2/ensemble_dynamics_model.hpp"
#include "mpc_controller_ros2/eigen_mlp.hpp"
#include "mpc_controller_ros2/motion_model.hpp"
#include "mpc_controller_ros2/motion_model_factory.hpp"
#include "mpc_controller_ros2/mppi_params.hpp"

using namespace mpc_controller_ros2;
namespace fs = std::filesystem;

// =============================================================================
// Helper: 테스트용 EigenMLP 바이너리 파일 저장
// =============================================================================

static void saveTestMLPFile(const std::string& path, int in_dim, int out_dim,
                            int hidden = 16) {
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
// OnlineDataCollector 테스트 Fixture
// =============================================================================

class OnlineDataCollectorTest : public ::testing::Test {
protected:
  void SetUp() override {
    MPPIParams params;
    model_ = MotionModelFactory::create("diff_drive", params);
    buffer_ = std::make_unique<OnlineDataBuffer>(100);
    collector_ = std::make_unique<OnlineDataCollector>(
      buffer_.get(), model_.get(), 3, 2);
  }

  std::unique_ptr<MotionModel> model_;
  std::unique_ptr<OnlineDataBuffer> buffer_;
  std::unique_ptr<OnlineDataCollector> collector_;

  // DiffDrive: nx=3 [x, y, theta], nu=2 [v, omega]
  static constexpr int kNx = 3;
  static constexpr int kNu = 2;
};

// -----------------------------------------------------------------------------
// 1. CollectValid: 데이터 추가 후 stats 검증
// -----------------------------------------------------------------------------
TEST_F(OnlineDataCollectorTest, CollectValid) {
  Eigen::VectorXd prev_state = Eigen::VectorXd::Zero(kNx);
  Eigen::VectorXd prev_control = Eigen::VectorXd::Zero(kNu);
  Eigen::VectorXd curr_state = Eigen::VectorXd::Ones(kNx) * 0.1;
  double dt = 0.02;

  collector_->collect(prev_state, prev_control, curr_state, dt);

  auto stats = collector_->getStats();
  EXPECT_EQ(stats.total_collected, 1u);
  EXPECT_EQ(stats.buffer_size, 1u);
  EXPECT_FALSE(stats.buffer_full);

  // 추가 수집
  collector_->collect(prev_state, prev_control, curr_state, dt);
  collector_->collect(prev_state, prev_control, curr_state, dt);

  stats = collector_->getStats();
  EXPECT_EQ(stats.total_collected, 3u);
  EXPECT_EQ(stats.buffer_size, 3u);
}

// -----------------------------------------------------------------------------
// 2. DimensionMismatch_Skipped: 잘못된 차원 -> 무시
// -----------------------------------------------------------------------------
TEST_F(OnlineDataCollectorTest, DimensionMismatch_Skipped) {
  Eigen::VectorXd wrong_state = Eigen::VectorXd::Zero(5);  // nx=3인데 5 전달
  Eigen::VectorXd prev_control = Eigen::VectorXd::Zero(kNu);
  Eigen::VectorXd curr_state = Eigen::VectorXd::Zero(kNx);
  double dt = 0.02;

  // prev_state 차원 불일치
  collector_->collect(wrong_state, prev_control, curr_state, dt);
  EXPECT_EQ(collector_->getStats().total_collected, 0u);
  EXPECT_EQ(buffer_->size(), 0u);

  // prev_control 차원 불일치
  Eigen::VectorXd prev_state = Eigen::VectorXd::Zero(kNx);
  Eigen::VectorXd wrong_control = Eigen::VectorXd::Zero(4);  // nu=2인데 4 전달
  collector_->collect(prev_state, wrong_control, curr_state, dt);
  EXPECT_EQ(collector_->getStats().total_collected, 0u);
  EXPECT_EQ(buffer_->size(), 0u);

  // curr_state 차원 불일치
  Eigen::VectorXd wrong_curr = Eigen::VectorXd::Zero(7);
  collector_->collect(prev_state, prev_control, wrong_curr, dt);
  EXPECT_EQ(collector_->getStats().total_collected, 0u);
  EXPECT_EQ(buffer_->size(), 0u);
}

// -----------------------------------------------------------------------------
// 3. Disabled_NoCollection: enabled=false -> 수집 안 함
// -----------------------------------------------------------------------------
TEST_F(OnlineDataCollectorTest, Disabled_NoCollection) {
  collector_->setEnabled(false);
  EXPECT_FALSE(collector_->isEnabled());

  Eigen::VectorXd state = Eigen::VectorXd::Zero(kNx);
  Eigen::VectorXd control = Eigen::VectorXd::Zero(kNu);

  collector_->collect(state, control, state, 0.02);
  collector_->collect(state, control, state, 0.02);

  EXPECT_EQ(collector_->getStats().total_collected, 0u);
  EXPECT_EQ(buffer_->size(), 0u);
}

// -----------------------------------------------------------------------------
// 4. EnableDisableToggle: 토글 후 동작 확인
// -----------------------------------------------------------------------------
TEST_F(OnlineDataCollectorTest, EnableDisableToggle) {
  Eigen::VectorXd state = Eigen::VectorXd::Zero(kNx);
  Eigen::VectorXd control = Eigen::VectorXd::Zero(kNu);

  // 기본: enabled
  EXPECT_TRUE(collector_->isEnabled());
  collector_->collect(state, control, state, 0.02);
  EXPECT_EQ(collector_->getStats().total_collected, 1u);

  // 비활성화
  collector_->setEnabled(false);
  collector_->collect(state, control, state, 0.02);
  EXPECT_EQ(collector_->getStats().total_collected, 1u);  // 변화 없음

  // 다시 활성화
  collector_->setEnabled(true);
  collector_->collect(state, control, state, 0.02);
  EXPECT_EQ(collector_->getStats().total_collected, 2u);
}

// -----------------------------------------------------------------------------
// 5. RingBufferOverflow: 용량 초과 시 링 버퍼 덮어쓰기
// -----------------------------------------------------------------------------
TEST_F(OnlineDataCollectorTest, RingBufferOverflow) {
  // 작은 버퍼로 재생성
  buffer_ = std::make_unique<OnlineDataBuffer>(5);
  collector_ = std::make_unique<OnlineDataCollector>(
    buffer_.get(), model_.get(), kNx, kNu);

  Eigen::VectorXd state = Eigen::VectorXd::Zero(kNx);
  Eigen::VectorXd control = Eigen::VectorXd::Zero(kNu);

  // 5개 채우기
  for (int i = 0; i < 5; ++i) {
    collector_->collect(state, control, state, 0.02);
  }
  EXPECT_EQ(buffer_->size(), 5u);
  EXPECT_TRUE(buffer_->full());
  EXPECT_TRUE(collector_->getStats().buffer_full);

  // 추가 3개 -> 링 버퍼이므로 size는 5 유지, total은 8
  for (int i = 0; i < 3; ++i) {
    collector_->collect(state, control, state, 0.02);
  }
  EXPECT_EQ(buffer_->size(), 5u);  // 용량 제한
  EXPECT_EQ(collector_->getStats().total_collected, 8u);
  EXPECT_TRUE(buffer_->full());
}

// -----------------------------------------------------------------------------
// 6. ThreadSafety_Atomic: enabled_ atomic 플래그 set/get
// -----------------------------------------------------------------------------
TEST_F(OnlineDataCollectorTest, ThreadSafety_Atomic) {
  // atomic 플래그의 기본 동작 검증 (단일 스레드에서 일관성 확인)
  collector_->setEnabled(true);
  EXPECT_TRUE(collector_->isEnabled());

  collector_->setEnabled(false);
  EXPECT_FALSE(collector_->isEnabled());

  collector_->setEnabled(true);
  EXPECT_TRUE(collector_->isEnabled());

  // 빠른 토글 반복 — atomic이 아닌 경우 데이터 레이스 가능
  for (int i = 0; i < 1000; ++i) {
    collector_->setEnabled(i % 2 == 0);
    bool val = collector_->isEnabled();
    EXPECT_EQ(val, (i % 2 == 0));
  }
}

// =============================================================================
// ModelReloader 테스트 Fixture
// =============================================================================

class ModelReloaderTest : public ::testing::Test {
protected:
  void SetUp() override {
    // mkdtemp 패턴으로 임시 디렉토리 생성
    std::string tmpl = (fs::temp_directory_path() / "test_model_reload_XXXXXX").string();
    char* dir = mkdtemp(tmpl.data());
    ASSERT_NE(dir, nullptr) << "Failed to create temp directory";
    temp_dir_ = std::string(dir);
  }

  void TearDown() override {
    if (!temp_dir_.empty() && fs::exists(temp_dir_)) {
      fs::remove_all(temp_dir_);
    }
  }

  // 앙상블 MLP 파일 생성 (model_0.bin ~ model_{M-1}.bin)
  void createEnsembleFiles(int ensemble_size, int in_dim = 5,
                           int out_dim = 3, int hidden = 16) {
    for (int i = 0; i < ensemble_size; ++i) {
      std::string path = temp_dir_ + "/model_" + std::to_string(i) + ".bin";
      saveTestMLPFile(path, in_dim, out_dim, hidden);
    }
  }

  std::string temp_dir_;

  // DiffDrive 기준: nx=3, nu=2 -> MLP input=5, output=3
  static constexpr int kInDim = 5;
  static constexpr int kOutDim = 3;
  static constexpr int kDefaultEnsemble = 3;
};

// -----------------------------------------------------------------------------
// 7. Construction_ValidEnsembleSize: 정상 생성
// -----------------------------------------------------------------------------
TEST_F(ModelReloaderTest, Construction_ValidEnsembleSize) {
  EXPECT_NO_THROW(ModelReloader reloader(temp_dir_, 3));
  EXPECT_NO_THROW(ModelReloader reloader(temp_dir_, 1));
  EXPECT_NO_THROW(ModelReloader reloader(temp_dir_, 10));
}

// -----------------------------------------------------------------------------
// 8. InvalidEnsembleSize_Throws: ensemble_size=0 -> 예외
// -----------------------------------------------------------------------------
TEST_F(ModelReloaderTest, InvalidEnsembleSize_Throws) {
  EXPECT_THROW(ModelReloader reloader(temp_dir_, 0), std::invalid_argument);
}

// -----------------------------------------------------------------------------
// 9. ModelsNotExist: 파일 없음 -> modelsExist()=false
// -----------------------------------------------------------------------------
TEST_F(ModelReloaderTest, ModelsNotExist) {
  ModelReloader reloader(temp_dir_, kDefaultEnsemble);
  EXPECT_FALSE(reloader.modelsExist());
}

// -----------------------------------------------------------------------------
// 10. ModelsExist_AfterCreation: 파일 생성 후 -> modelsExist()=true
// -----------------------------------------------------------------------------
TEST_F(ModelReloaderTest, ModelsExist_AfterCreation) {
  ModelReloader reloader(temp_dir_, kDefaultEnsemble);
  EXPECT_FALSE(reloader.modelsExist());

  // 앙상블 파일 생성
  createEnsembleFiles(kDefaultEnsemble, kInDim, kOutDim);

  EXPECT_TRUE(reloader.modelsExist());

  // 일부만 존재하면 false
  fs::remove(temp_dir_ + "/model_2.bin");
  EXPECT_FALSE(reloader.modelsExist());
}

// -----------------------------------------------------------------------------
// 11. TryReload_Success: 유효한 .bin 파일 -> tryReload()=true
// -----------------------------------------------------------------------------
TEST_F(ModelReloaderTest, TryReload_Success) {
  createEnsembleFiles(kDefaultEnsemble, kInDim, kOutDim);

  ModelReloader reloader(temp_dir_, kDefaultEnsemble);
  EXPECT_TRUE(reloader.modelsExist());

  bool loaded = reloader.tryReload();
  EXPECT_TRUE(loaded);

  // 로드 시간이 기록되었는지 확인
  auto reload_time = reloader.lastReloadTime();
  EXPECT_NE(reload_time.time_since_epoch().count(), 0);
}

// -----------------------------------------------------------------------------
// 12. TakeModels_TransfersOwnership: 로드 후 takeModels -> M개 모델 반환
// -----------------------------------------------------------------------------
TEST_F(ModelReloaderTest, TakeModels_TransfersOwnership) {
  createEnsembleFiles(kDefaultEnsemble, kInDim, kOutDim);

  ModelReloader reloader(temp_dir_, kDefaultEnsemble);
  ASSERT_TRUE(reloader.tryReload());

  auto models = reloader.takeModels();
  EXPECT_EQ(static_cast<int>(models.size()), kDefaultEnsemble);

  // 각 모델이 유효한지 확인
  for (int i = 0; i < kDefaultEnsemble; ++i) {
    ASSERT_NE(models[i], nullptr);
    EXPECT_EQ(models[i]->inputDim(), kInDim);
    EXPECT_EQ(models[i]->outputDim(), kOutDim);

    // 순전파 검증
    Eigen::VectorXd input = Eigen::VectorXd::Random(kInDim);
    Eigen::VectorXd output = models[i]->forward(input);
    EXPECT_EQ(output.size(), kOutDim);
    for (int j = 0; j < kOutDim; ++j) {
      EXPECT_TRUE(std::isfinite(output(j)));
    }
  }

  // takeModels 후 내부 벡터는 비워져야 함
  auto empty_models = reloader.takeModels();
  EXPECT_TRUE(empty_models.empty());
}

// =============================================================================

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
