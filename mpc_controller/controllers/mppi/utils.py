"""MPPI 유틸리티 함수."""

import numpy as np


def normalize_angle_batch(angles: np.ndarray) -> np.ndarray:
    """각도 배열을 [-pi, pi] 범위로 정규화 (벡터화).

    Args:
        angles: 임의 shape의 각도 배열 [rad]

    Returns:
        정규화된 각도 배열
    """
    return np.arctan2(np.sin(angles), np.cos(angles))


def log_sum_exp(values: np.ndarray) -> float:
    """수치 안정적 log-sum-exp 계산.

    log(sum(exp(values))) = max + log(sum(exp(values - max)))

    Args:
        values: 1D 배열

    Returns:
        log-sum-exp 값
    """
    max_val = np.max(values)
    return max_val + np.log(np.sum(np.exp(values - max_val)))


def softmax_weights(costs: np.ndarray, lambda_: float) -> np.ndarray:
    """비용에서 softmax 가중치 계산.

    w_k = exp(-S_k / lambda) / sum(exp(-S_j / lambda))

    Args:
        costs: (K,) 비용 배열
        lambda_: 온도 파라미터

    Returns:
        (K,) 정규화된 가중치 배열
    """
    shifted = -costs / lambda_
    shifted -= np.max(shifted)  # 수치 안정성
    exp_vals = np.exp(shifted)
    return exp_vals / np.sum(exp_vals)


def q_exponential(x: np.ndarray, q: float) -> np.ndarray:
    """Tsallis q-exponential: exp_q(x) = [1 + (1-q)*x]_+^{1/(1-q)}.

    q → 1 극한에서 표준 exp(x)와 동일.
    base ≤ 0인 경우 결과는 0 (q>1일 때 1/음수 지수 → 발산 방지).

    Args:
        x: 입력 배열
        q: Tsallis 파라미터 (q=1 → 표준 exp)

    Returns:
        q-exponential 값 배열
    """
    if abs(q - 1.0) < 1e-8:
        return np.exp(x)
    exponent = 1.0 / (1.0 - q)
    base = 1.0 + (1.0 - q) * x
    positive = base > 0
    result = np.zeros_like(x, dtype=float)
    result[positive] = np.power(base[positive], exponent)
    return result


def q_logarithm(x: np.ndarray, q: float) -> np.ndarray:
    """Tsallis q-logarithm: ln_q(x) = (x^{1-q} - 1) / (1-q).

    q → 1 극한에서 표준 ln(x)와 동일.

    Args:
        x: 입력 배열 (양수)
        q: Tsallis 파라미터 (q=1 → 표준 ln)

    Returns:
        q-logarithm 값 배열
    """
    if abs(q - 1.0) < 1e-8:
        return np.log(x)
    return (np.power(x, 1.0 - q) - 1.0) / (1.0 - q)


def effective_sample_size(weights: np.ndarray) -> float:
    """Effective Sample Size (ESS) 계산.

    ESS = 1 / sum(w_k^2), 범위 [1, K]

    Args:
        weights: (K,) 정규화된 가중치

    Returns:
        ESS 값
    """
    return 1.0 / np.sum(weights ** 2)


# ─────────────────────────────────────────────────────────────
# M3d Stein Variational MPPI — RBF 커널 유틸리티
# ─────────────────────────────────────────────────────────────

def rbf_kernel(particles: np.ndarray, bandwidth: float) -> np.ndarray:
    """RBF (Gaussian) 커널 행렬.

    k(x_i, x_j) = exp(-‖x_i - x_j‖² / (2h²))

    Args:
        particles: (K, D) 입자 배열
        bandwidth: h (커널 폭)

    Returns:
        (K, K) 커널 행렬
    """
    # pairwise squared distance: ‖x_i - x_j‖²
    diff = particles[:, np.newaxis, :] - particles[np.newaxis, :, :]  # (K, K, D)
    sq_dist = np.sum(diff ** 2, axis=-1)  # (K, K)
    return np.exp(-sq_dist / (2.0 * bandwidth ** 2))


def rbf_kernel_grad(diff: np.ndarray, kernel: np.ndarray,
                    bandwidth: float) -> np.ndarray:
    """RBF 커널 gradient (각 particle에 대한).

    ∇_{x_i} k(x_j, x_i) = k(x_j, x_i) · (x_j - x_i) / h²

    Args:
        diff: (K, K, D) pairwise 차이 (x_j - x_i)
        kernel: (K, K) 커널 행렬
        bandwidth: h

    Returns:
        (K, K, D) 커널 gradient
    """
    return kernel[:, :, np.newaxis] * diff / (bandwidth ** 2)


def median_bandwidth(particles: np.ndarray) -> float:
    """Median heuristic bandwidth.

    h = sqrt(median(‖x_i - x_j‖²) / (2 · log(K + 1)))

    Args:
        particles: (K, D) 입자 배열

    Returns:
        bandwidth 값 (최소 1e-6 보장)
    """
    diff = particles[:, np.newaxis, :] - particles[np.newaxis, :, :]  # (K, K, D)
    sq_dist = np.sum(diff ** 2, axis=-1)  # (K, K)
    # 상삼각 원소만 사용 (대각선 제외)
    K = particles.shape[0]
    triu_idx = np.triu_indices(K, k=1)
    pairwise_sq = sq_dist[triu_idx]
    med = np.median(pairwise_sq)
    h = np.sqrt(med / (2.0 * np.log(K + 1)))
    return max(h, 1e-6)
