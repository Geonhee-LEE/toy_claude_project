#include "mpc_controller_ros2/eigen_mlp.hpp"
#include <fstream>
#include <stdexcept>
#include <cmath>

namespace mpc_controller_ros2
{

EigenMLP::EigenMLP(std::vector<LayerParams> layers, NormParams norm)
: layers_(std::move(layers)), norm_(std::move(norm))
{
  // 정규화 유효성 검사: std > 0이면 활성화
  has_norm_ = (norm_.in_std.size() > 0 && norm_.in_std.minCoeff() > 1e-12);
}

std::unique_ptr<EigenMLP> EigenMLP::loadFromFile(const std::string& path)
{
  std::ifstream file(path, std::ios::binary);
  if (!file.is_open()) {
    throw std::runtime_error("EigenMLP: Cannot open file: " + path);
  }

  // Magic number
  uint32_t magic;
  file.read(reinterpret_cast<char*>(&magic), sizeof(magic));
  if (magic != kMagic) {
    throw std::runtime_error("EigenMLP: Invalid magic number in: " + path);
  }

  // Version
  uint32_t version;
  file.read(reinterpret_cast<char*>(&version), sizeof(version));
  if (version != kVersion) {
    throw std::runtime_error("EigenMLP: Unsupported version " + std::to_string(version));
  }

  // Number of layers
  uint32_t n_layers;
  file.read(reinterpret_cast<char*>(&n_layers), sizeof(n_layers));
  if (n_layers == 0 || n_layers > 100) {
    throw std::runtime_error("EigenMLP: Invalid layer count: " + std::to_string(n_layers));
  }

  // Normalization parameters
  NormParams norm;
  auto read_vector = [&file](int dim) -> Eigen::VectorXd {
    Eigen::VectorXd v(dim);
    file.read(reinterpret_cast<char*>(v.data()), dim * sizeof(double));
    return v;
  };

  uint32_t in_dim, out_dim;
  file.read(reinterpret_cast<char*>(&in_dim), sizeof(in_dim));
  file.read(reinterpret_cast<char*>(&out_dim), sizeof(out_dim));
  norm.in_mean = read_vector(in_dim);
  norm.in_std = read_vector(in_dim);
  norm.out_mean = read_vector(out_dim);
  norm.out_std = read_vector(out_dim);

  // Layers
  std::vector<LayerParams> layers(n_layers);
  for (uint32_t i = 0; i < n_layers; ++i) {
    uint32_t rows, cols;
    file.read(reinterpret_cast<char*>(&rows), sizeof(rows));
    file.read(reinterpret_cast<char*>(&cols), sizeof(cols));

    layers[i].weight.resize(rows, cols);
    file.read(reinterpret_cast<char*>(layers[i].weight.data()),
              rows * cols * sizeof(double));

    layers[i].bias.resize(rows);
    file.read(reinterpret_cast<char*>(layers[i].bias.data()),
              rows * sizeof(double));
  }

  if (!file.good()) {
    throw std::runtime_error("EigenMLP: Error reading file: " + path);
  }

  return std::make_unique<EigenMLP>(std::move(layers), std::move(norm));
}

Eigen::VectorXd EigenMLP::forward(const Eigen::VectorXd& input) const
{
  // Z-score 정규화
  Eigen::VectorXd x = input;
  if (has_norm_) {
    x = (input - norm_.in_mean).cwiseQuotient(norm_.in_std);
  }

  // 순전파: [Linear + ReLU] × (L-1) + Linear
  for (int i = 0; i < static_cast<int>(layers_.size()); ++i) {
    x = layers_[i].weight * x + layers_[i].bias;
    // ReLU (마지막 레이어 제외)
    if (i < static_cast<int>(layers_.size()) - 1) {
      x = x.cwiseMax(0.0);
    }
  }

  // 역정규화
  if (has_norm_) {
    x = x.cwiseProduct(norm_.out_std) + norm_.out_mean;
  }

  return x;
}

Eigen::MatrixXd EigenMLP::forwardBatch(const Eigen::MatrixXd& inputs) const
{
  int K = inputs.rows();

  // Z-score 정규화 (행별)
  Eigen::MatrixXd X = inputs;
  if (has_norm_) {
    X = (inputs.rowwise() - norm_.in_mean.transpose()).array().rowwise()
        / norm_.in_std.transpose().array();
  }

  // 순전파 (BLAS dgemm 활용): X = X * W^T + bias (브로드캐스트)
  for (int i = 0; i < static_cast<int>(layers_.size()); ++i) {
    // X: (K, in) → (K, out) = X * W^T + 1*bias^T
    X = (X * layers_[i].weight.transpose()).rowwise() + layers_[i].bias.transpose();
    // ReLU (마지막 레이어 제외)
    if (i < static_cast<int>(layers_.size()) - 1) {
      X = X.cwiseMax(0.0);
    }
  }

  // 역정규화 (행별)
  if (has_norm_) {
    X = X.array().rowwise() * norm_.out_std.transpose().array();
    X = X.rowwise() + norm_.out_mean.transpose();
  }

  return X;
}

int EigenMLP::inputDim() const
{
  if (layers_.empty()) { return 0; }
  return static_cast<int>(layers_[0].weight.cols());
}

int EigenMLP::outputDim() const
{
  if (layers_.empty()) { return 0; }
  return static_cast<int>(layers_.back().weight.rows());
}

}  // namespace mpc_controller_ros2
