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


def effective_sample_size(weights: np.ndarray) -> float:
    """Effective Sample Size (ESS) 계산.

    ESS = 1 / sum(w_k^2), 범위 [1, K]

    Args:
        weights: (K,) 정규화된 가중치

    Returns:
        ESS 값
    """
    return 1.0 / np.sum(weights ** 2)
