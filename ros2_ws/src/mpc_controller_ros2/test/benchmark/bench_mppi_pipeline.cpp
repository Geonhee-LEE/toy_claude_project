// =============================================================================
// MPPI Pipeline Microbenchmark
//
// chrono 기반으로 MPPI 파이프라인 각 단계의 소요 시간을 측정합니다.
// K/N 스케일링 및 3개 모션 모델(DiffDrive, Swerve, NonCoaxialSwerve)을 지원합니다.
//
// 사용법:
//   ./bench_mppi_pipeline [--model diff_drive|swerve|non_coaxial_swerve]
//                         [--K 256] [--N 30] [--warmup 5] [--repeat 20]
// =============================================================================

#include <chrono>
#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include <algorithm>
#include <numeric>
#include <cmath>

#include <Eigen/Dense>

#include "mpc_controller_ros2/mppi_params.hpp"
#include "mpc_controller_ros2/motion_model.hpp"
#include "mpc_controller_ros2/motion_model_factory.hpp"
#include "mpc_controller_ros2/batch_dynamics_wrapper.hpp"
#include "mpc_controller_ros2/cost_functions.hpp"
#include "mpc_controller_ros2/sampling.hpp"
#include "mpc_controller_ros2/weight_computation.hpp"
#include "mpc_controller_ros2/utils.hpp"

using Clock = std::chrono::high_resolution_clock;
using Duration = std::chrono::duration<double, std::micro>;

namespace
{

struct BenchResult
{
  std::string stage;
  double mean_us{0.0};
  double std_us{0.0};
  double min_us{0.0};
  double max_us{0.0};
};

BenchResult summarize(const std::string& name, const std::vector<double>& times_us)
{
  BenchResult r;
  r.stage = name;
  if (times_us.empty()) return r;

  double sum = std::accumulate(times_us.begin(), times_us.end(), 0.0);
  r.mean_us = sum / times_us.size();
  r.min_us = *std::min_element(times_us.begin(), times_us.end());
  r.max_us = *std::max_element(times_us.begin(), times_us.end());

  double sq_sum = 0.0;
  for (double v : times_us) {
    sq_sum += (v - r.mean_us) * (v - r.mean_us);
  }
  r.std_us = std::sqrt(sq_sum / times_us.size());
  return r;
}

void printResult(const BenchResult& r)
{
  std::cout << std::setw(20) << std::left << r.stage
            << std::setw(12) << std::right << std::fixed << std::setprecision(1) << r.mean_us
            << std::setw(12) << r.std_us
            << std::setw(12) << r.min_us
            << std::setw(12) << r.max_us
            << std::endl;
}

void printHeader()
{
  std::cout << std::setw(20) << std::left << "Stage"
            << std::setw(12) << std::right << "Mean(us)"
            << std::setw(12) << "Std(us)"
            << std::setw(12) << "Min(us)"
            << std::setw(12) << "Max(us)"
            << std::endl;
  std::cout << std::string(68, '-') << std::endl;
}

mpc_controller_ros2::MPPIParams makeParams(const std::string& model_type, int K, int N)
{
  mpc_controller_ros2::MPPIParams params;
  params.K = K;
  params.N = N;
  params.dt = 0.05;
  params.lambda = 10.0;
  params.motion_model = model_type;

  bool is_nc = (model_type == "non_coaxial_swerve");
  int nu = (model_type == "diff_drive") ? 2 : 3;
  int nx = is_nc ? 4 : 3;

  params.noise_sigma = Eigen::VectorXd::Constant(nu, 0.5);
  params.Q = Eigen::MatrixXd::Identity(nx, nx) * 10.0;
  params.Qf = Eigen::MatrixXd::Identity(nx, nx) * 20.0;
  params.R = Eigen::MatrixXd::Identity(nu, nu) * 0.1;
  params.R_rate = Eigen::MatrixXd::Identity(nu, nu) * 1.0;

  params.v_max = 1.0;
  params.v_min = -0.5;
  params.omega_max = 1.0;
  params.omega_min = -1.0;
  params.vy_max = 0.5;
  params.max_steering_rate = 2.0;
  params.max_steering_angle = M_PI / 3.0;

  return params;
}

void runBenchmark(const std::string& model_type, int K, int N, int warmup, int repeat)
{
  std::cout << "\n========================================" << std::endl;
  std::cout << "Model: " << model_type << "  K=" << K << "  N=" << N << std::endl;
  std::cout << "========================================" << std::endl;

  auto params = makeParams(model_type, K, N);
  int nu = (model_type == "diff_drive") ? 2 : 3;
  int nx = (model_type == "non_coaxial_swerve") ? 4 : 3;

  // Create components
  auto model = mpc_controller_ros2::MotionModelFactory::create(model_type, params);
  auto dynamics = std::make_unique<mpc_controller_ros2::BatchDynamicsWrapper>(
    params, std::shared_ptr<mpc_controller_ros2::MotionModel>(std::move(model)));

  auto sampler = std::make_unique<mpc_controller_ros2::GaussianSampler>(params.noise_sigma, 42);

  auto cost_function = std::make_unique<mpc_controller_ros2::CompositeMPPICost>();
  cost_function->addCost(std::make_unique<mpc_controller_ros2::StateTrackingCost>(params.Q));
  cost_function->addCost(std::make_unique<mpc_controller_ros2::TerminalCost>(params.Qf));
  cost_function->addCost(std::make_unique<mpc_controller_ros2::ControlEffortCost>(params.R));
  cost_function->addCost(std::make_unique<mpc_controller_ros2::ControlRateCost>(params.R_rate));

  auto weight_comp = std::make_unique<mpc_controller_ros2::VanillaMPPIWeights>();

  // Setup
  Eigen::VectorXd x0 = Eigen::VectorXd::Zero(nx);
  Eigen::MatrixXd control_seq = Eigen::MatrixXd::Zero(N, nu);
  Eigen::MatrixXd ref = Eigen::MatrixXd::Zero(N + 1, nx);
  for (int t = 0; t <= N; ++t) {
    ref(t, 0) = 0.03 * t;  // forward motion
  }

  // Pre-allocate buffers
  std::vector<Eigen::MatrixXd> noise_buf(K);
  std::vector<Eigen::MatrixXd> perturbed_buf(K);
  std::vector<Eigen::MatrixXd> traj_buf(K);
  for (int k = 0; k < K; ++k) {
    noise_buf[k] = Eigen::MatrixXd::Zero(N, nu);
    perturbed_buf[k] = Eigen::MatrixXd::Zero(N, nu);
    traj_buf[k] = Eigen::MatrixXd::Zero(N + 1, nx);
  }

  // Timing storage
  std::vector<double> t_sample, t_perturb, t_rollout, t_cost, t_weight, t_update, t_pipeline;

  int total_runs = warmup + repeat;

  for (int run = 0; run < total_runs; ++run) {
    bool record = (run >= warmup);
    auto t_pipe_start = Clock::now();

    // 1. Sampling
    auto t0 = Clock::now();
    sampler->sampleInPlace(noise_buf, K, N, nu);
    auto t1 = Clock::now();
    if (record) t_sample.push_back(Duration(t1 - t0).count());

    // 2. Perturbation
    auto t2 = Clock::now();
    for (int k = 0; k < K; ++k) {
      perturbed_buf[k].noalias() = control_seq + noise_buf[k];
      perturbed_buf[k] = dynamics->clipControls(perturbed_buf[k]);
    }
    auto t3 = Clock::now();
    if (record) t_perturb.push_back(Duration(t3 - t2).count());

    // 3. Rollout
    auto t4 = Clock::now();
    dynamics->rolloutBatchInPlace(x0, perturbed_buf, params.dt, traj_buf);
    auto t5 = Clock::now();
    if (record) t_rollout.push_back(Duration(t5 - t4).count());

    // 4. Cost
    auto t6 = Clock::now();
    Eigen::VectorXd costs = cost_function->compute(traj_buf, perturbed_buf, ref);
    auto t7 = Clock::now();
    if (record) t_cost.push_back(Duration(t7 - t6).count());

    // 5. Weight
    auto t8 = Clock::now();
    Eigen::VectorXd weights = weight_comp->compute(costs, params.lambda);
    auto t9 = Clock::now();
    if (record) t_weight.push_back(Duration(t9 - t8).count());

    // 6. Update
    auto t10 = Clock::now();
    Eigen::MatrixXd weighted_noise = Eigen::MatrixXd::Zero(N, nu);
    for (int k = 0; k < K; ++k) {
      weighted_noise.noalias() += weights(k) * noise_buf[k];
    }
    control_seq += weighted_noise;
    control_seq = dynamics->clipControls(control_seq);
    auto t11 = Clock::now();
    if (record) t_update.push_back(Duration(t11 - t10).count());

    auto t_pipe_end = Clock::now();
    if (record) t_pipeline.push_back(Duration(t_pipe_end - t_pipe_start).count());
  }

  // Print results
  printHeader();
  printResult(summarize("Sampling", t_sample));
  printResult(summarize("Perturbation", t_perturb));
  printResult(summarize("Rollout", t_rollout));
  printResult(summarize("Cost", t_cost));
  printResult(summarize("Weight", t_weight));
  printResult(summarize("Update", t_update));
  std::cout << std::string(68, '=') << std::endl;
  auto pipeline_r = summarize("Pipeline", t_pipeline);
  printResult(pipeline_r);

  double freq_hz = 1e6 / pipeline_r.mean_us;
  std::cout << "\nPipeline frequency: " << std::fixed << std::setprecision(1)
            << freq_hz << " Hz" << std::endl;
  std::cout << "Rollout fraction: " << std::fixed << std::setprecision(1)
            << (summarize("Rollout", t_rollout).mean_us / pipeline_r.mean_us * 100.0)
            << "%" << std::endl;
}

}  // anonymous namespace

int main(int argc, char** argv)
{
  // Parse arguments
  std::string model = "diff_drive";
  int K = 256;
  int N = 30;
  int warmup = 5;
  int repeat = 20;
  bool scaling = false;

  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--model" && i + 1 < argc) { model = argv[++i]; }
    else if (arg == "--K" && i + 1 < argc) { K = std::stoi(argv[++i]); }
    else if (arg == "--N" && i + 1 < argc) { N = std::stoi(argv[++i]); }
    else if (arg == "--warmup" && i + 1 < argc) { warmup = std::stoi(argv[++i]); }
    else if (arg == "--repeat" && i + 1 < argc) { repeat = std::stoi(argv[++i]); }
    else if (arg == "--scaling") { scaling = true; }
    else if (arg == "--help") {
      std::cout << "Usage: bench_mppi_pipeline [options]\n"
                << "  --model <name>     diff_drive|swerve|non_coaxial_swerve\n"
                << "  --K <int>          Sample count (default 256)\n"
                << "  --N <int>          Horizon (default 30)\n"
                << "  --warmup <int>     Warmup iterations (default 5)\n"
                << "  --repeat <int>     Measurement iterations (default 20)\n"
                << "  --scaling          Run K/N scaling sweep\n"
                << std::endl;
      return 0;
    }
  }

  Eigen::setNbThreads(1);

  if (scaling) {
    // K scaling
    std::vector<std::string> models = {"diff_drive", "swerve", "non_coaxial_swerve"};
    std::vector<int> K_values = {64, 128, 256, 512, 1024};
    std::vector<int> N_values = {15, 30, 50};

    for (const auto& m : models) {
      for (int k_val : K_values) {
        runBenchmark(m, k_val, 30, warmup, repeat);
      }
      for (int n_val : N_values) {
        runBenchmark(m, 256, n_val, warmup, repeat);
      }
    }
  } else {
    runBenchmark(model, K, N, warmup, repeat);
  }

  return 0;
}
