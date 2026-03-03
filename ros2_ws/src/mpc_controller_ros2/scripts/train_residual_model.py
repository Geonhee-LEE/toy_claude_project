#!/usr/bin/env python3
"""
Residual Dynamics MLP 학습 파이프라인

1. Gazebo 데이터 수집 → (state, control, next_state, dt)
2. 잔차 계산: residual = (next - curr)/dt - nominal(curr, ctrl)
3. PyTorch MLP 학습 + Z-score 정규화
4. 바이너리 포맷 내보내기 (EigenMLP::loadFromFile 호환)

Usage:
    python3 train_residual_model.py --data data.csv --output model.bin
    python3 train_residual_model.py --generate-dummy --output model.bin  # 테스트용 더미 모델
"""

import argparse
import struct
import sys
from pathlib import Path

import numpy as np

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


# =============================================================================
# 바이너리 포맷 (EigenMLP::loadFromFile 호환)
# =============================================================================

MAGIC = 0x454D4C50  # "EMLP"
VERSION = 1


def save_eigen_mlp_binary(path: str, layers: list, norm: dict):
    """
    바이너리 파일로 MLP 저장.

    layers: [(weight_np, bias_np), ...]  — weight shape: (out_dim, in_dim)
    norm: {in_mean, in_std, out_mean, out_std}
    """
    with open(path, 'wb') as f:
        # Header
        f.write(struct.pack('<I', MAGIC))
        f.write(struct.pack('<I', VERSION))
        f.write(struct.pack('<I', len(layers)))

        # Normalization
        in_dim = norm['in_mean'].shape[0]
        out_dim = norm['out_mean'].shape[0]
        f.write(struct.pack('<I', in_dim))
        f.write(struct.pack('<I', out_dim))
        f.write(norm['in_mean'].astype(np.float64).tobytes())
        f.write(norm['in_std'].astype(np.float64).tobytes())
        f.write(norm['out_mean'].astype(np.float64).tobytes())
        f.write(norm['out_std'].astype(np.float64).tobytes())

        # Layers
        for weight, bias in layers:
            rows, cols = weight.shape  # (out_dim, in_dim)
            f.write(struct.pack('<I', rows))
            f.write(struct.pack('<I', cols))
            # Eigen은 column-major이지만 loadFromFile에서 row-major로 읽으므로 그대로 저장
            f.write(weight.astype(np.float64).tobytes())
            f.write(bias.astype(np.float64).tobytes())

    print(f"Saved model to {path} ({len(layers)} layers, in={in_dim}, out={out_dim})")


def generate_dummy_model(output_path: str, in_dim: int = 5, out_dim: int = 3,
                         hidden: int = 64, n_hidden: int = 2):
    """테스트용 더미 모델 생성 (랜덤 가중치)"""
    np.random.seed(42)

    layers = []
    prev_dim = in_dim
    for i in range(n_hidden):
        w = np.random.randn(hidden, prev_dim) * 0.1
        b = np.zeros(hidden)
        layers.append((w, b))
        prev_dim = hidden

    # Output layer
    w = np.random.randn(out_dim, prev_dim) * 0.01
    b = np.zeros(out_dim)
    layers.append((w, b))

    norm = {
        'in_mean': np.zeros(in_dim),
        'in_std': np.ones(in_dim),
        'out_mean': np.zeros(out_dim),
        'out_std': np.ones(out_dim),
    }

    save_eigen_mlp_binary(output_path, layers, norm)


# =============================================================================
# PyTorch MLP 정의
# =============================================================================

if HAS_TORCH:
    class ResidualMLP(nn.Module):
        def __init__(self, in_dim, out_dim, hidden=64, n_hidden=2):
            super().__init__()
            layers = []
            prev = in_dim
            for _ in range(n_hidden):
                layers.append(nn.Linear(prev, hidden))
                layers.append(nn.ReLU())
                prev = hidden
            layers.append(nn.Linear(prev, out_dim))
            self.net = nn.Sequential(*layers)

        def forward(self, x):
            return self.net(x)


# =============================================================================
# 공칭 동역학 (DiffDrive)
# =============================================================================

def nominal_diff_drive(state, control):
    """DiffDrive 공칭 동역학: x_dot = f(x, u)"""
    x, y, theta = state[..., 0], state[..., 1], state[..., 2]
    v, omega = control[..., 0], control[..., 1]
    x_dot = np.stack([
        v * np.cos(theta),
        v * np.sin(theta),
        omega
    ], axis=-1)
    return x_dot


# =============================================================================
# 학습 파이프라인
# =============================================================================

def load_data(csv_path: str):
    """
    CSV 파일 로드.
    포맷: x, y, theta, v, omega, x_next, y_next, theta_next, dt
    """
    data = np.loadtxt(csv_path, delimiter=',', skiprows=1)
    states = data[:, :3]       # x, y, theta
    controls = data[:, 3:5]    # v, omega
    next_states = data[:, 5:8] # x_next, y_next, theta_next
    dt = data[:, 8]            # dt
    return states, controls, next_states, dt


def compute_residuals(states, controls, next_states, dt):
    """잔차 = 실제 변화율 - 공칭 변화율"""
    actual_rate = (next_states - states) / dt[:, None]
    nominal_rate = nominal_diff_drive(states, controls)
    residuals = actual_rate - nominal_rate
    return residuals


def train_model(states, controls, residuals, hidden=64, n_hidden=2,
                epochs=200, lr=1e-3, batch_size=256):
    """PyTorch로 MLP 학습"""
    if not HAS_TORCH:
        print("PyTorch가 설치되지 않았습니다. pip install torch")
        sys.exit(1)

    # 특성: [state, control]
    features = np.hstack([states, controls])
    targets = residuals

    # Z-score 정규화
    feat_mean = features.mean(axis=0)
    feat_std = features.std(axis=0) + 1e-8
    target_mean = targets.mean(axis=0)
    target_std = targets.std(axis=0) + 1e-8

    features_norm = (features - feat_mean) / feat_std
    targets_norm = (targets - target_mean) / target_std

    # DataLoader
    dataset = TensorDataset(
        torch.FloatTensor(features_norm),
        torch.FloatTensor(targets_norm)
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 모델
    in_dim = features.shape[1]
    out_dim = targets.shape[1]
    model = ResidualMLP(in_dim, out_dim, hidden, n_hidden)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    # 학습
    for epoch in range(epochs):
        total_loss = 0.0
        for feat_batch, target_batch in loader:
            pred = model(feat_batch)
            loss = criterion(pred, target_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * feat_batch.size(0)
        avg_loss = total_loss / len(dataset)
        if (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")

    # 가중치 추출
    layers = []
    for module in model.net:
        if isinstance(module, nn.Linear):
            w = module.weight.detach().numpy()
            b = module.bias.detach().numpy()
            layers.append((w, b))

    norm = {
        'in_mean': feat_mean,
        'in_std': feat_std,
        'out_mean': target_mean,
        'out_std': target_std,
    }

    return layers, norm


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Residual Dynamics MLP 학습')
    parser.add_argument('--data', type=str, help='CSV 데이터 파일 경로')
    parser.add_argument('--output', type=str, default='residual_model.bin',
                        help='출력 바이너리 파일 경로')
    parser.add_argument('--hidden', type=int, default=64, help='은닉층 크기')
    parser.add_argument('--n-hidden', type=int, default=2, help='은닉층 수')
    parser.add_argument('--epochs', type=int, default=200, help='학습 에폭')
    parser.add_argument('--lr', type=float, default=1e-3, help='학습률')
    parser.add_argument('--batch-size', type=int, default=256, help='배치 크기')
    parser.add_argument('--generate-dummy', action='store_true',
                        help='테스트용 더미 모델 생성')
    parser.add_argument('--in-dim', type=int, default=5,
                        help='더미 모델 입력 차원')
    parser.add_argument('--out-dim', type=int, default=3,
                        help='더미 모델 출력 차원')
    args = parser.parse_args()

    if args.generate_dummy:
        generate_dummy_model(args.output, args.in_dim, args.out_dim,
                             args.hidden, args.n_hidden)
        return

    if not args.data:
        print("--data 또는 --generate-dummy를 지정하세요")
        sys.exit(1)

    print(f"Loading data from {args.data}...")
    states, controls, next_states, dt = load_data(args.data)
    print(f"  Samples: {len(states)}")

    print("Computing residuals...")
    residuals = compute_residuals(states, controls, next_states, dt)
    print(f"  Residual mean: {residuals.mean(axis=0)}")
    print(f"  Residual std:  {residuals.std(axis=0)}")

    print("Training MLP...")
    layers, norm = train_model(
        states, controls, residuals,
        hidden=args.hidden, n_hidden=args.n_hidden,
        epochs=args.epochs, lr=args.lr, batch_size=args.batch_size
    )

    print(f"Saving to {args.output}...")
    save_eigen_mlp_binary(args.output, layers, norm)
    print("Done!")


if __name__ == '__main__':
    main()
