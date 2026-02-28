"""JAX JIT SVGD 커널 — SVMPC/SVG-MPPI 공유.

Stein Variational Gradient Descent (gradient-free 근사) GPU 구현.
K×K pairwise 커널 → attractive + repulsive force.

메모리 제한: K=4096 → (K,K,D) ≈ 8GB. K≤2048 권장.
"""

import jax
import jax.numpy as jnp


@jax.jit
def svgd_step_jit(particles, weights, bandwidth, step_size):
    """SVGD 1 iteration (GPU).

    Args:
        particles: (K, D) 입자 배열
        weights: (K,) softmax 가중치
        bandwidth: h (커널 폭)
        step_size: SVGD 업데이트 step size

    Returns:
        (K, D) 업데이트된 입자
    """
    K = particles.shape[0]

    # pairwise diff: diff[j,i] = x_j - x_i → (K, K, D)
    diff = particles[:, None, :] - particles[None, :, :]
    sq_dist = jnp.sum(diff ** 2, axis=-1)  # (K, K)

    # RBF 커널
    kernel = jnp.exp(-sq_dist / (2.0 * bandwidth ** 2))  # (K, K)

    # Attractive force: 저비용 샘플 방향으로 끌어당김
    # attract_i = Σ_j w_j · k(x_j, x_i) · (x_j - x_i)
    weighted_kernel = weights[:, None] * kernel  # (K, K)
    attractive = jnp.einsum("ji,jid->id", weighted_kernel, diff)  # (K, D)

    # Repulsive force: 샘플 간 다양성 유지
    # repel_i = (1/K) Σ_j k(x_j, x_i) · (x_j - x_i) / h²
    repulsive = jnp.einsum("ji,jid->id", kernel, diff) / (K * bandwidth ** 2)

    return particles + step_size * (attractive + repulsive)


@jax.jit
def median_bandwidth_jit(particles):
    """Median heuristic bandwidth (GPU).

    h = sqrt(median(‖x_i - x_j‖²) / (2·log(K+1)))

    Args:
        particles: (K, D)

    Returns:
        bandwidth 스칼라 (최소 1e-6)
    """
    diff = particles[:, None, :] - particles[None, :, :]
    sq_dist = jnp.sum(diff ** 2, axis=-1)  # (K, K)
    K = particles.shape[0]

    # 상삼각 mask (JIT 호환: boolean index 대신 where + sort)
    mask = jnp.triu(jnp.ones((K, K)), k=1)
    # 하삼각/대각 → 큰 값으로 치환 후 sort로 median 추출
    masked_sq = jnp.where(mask > 0, sq_dist, jnp.finfo(sq_dist.dtype).max)
    flat = masked_sq.ravel()
    sorted_flat = jnp.sort(flat)
    n_pairs = K * (K - 1) // 2
    med = sorted_flat[n_pairs // 2]

    h = jnp.sqrt(med / (2.0 * jnp.log(K + 1.0)))
    return jnp.maximum(h, 1e-6)


@jax.jit
def compute_diversity_jit(particles):
    """샘플 다양성 측정 — 평균 pairwise L2 거리 (GPU).

    Args:
        particles: (K, D)

    Returns:
        평균 pairwise L2 거리 스칼라
    """
    K = particles.shape[0]
    diff = particles[:, None, :] - particles[None, :, :]
    dists = jnp.sqrt(jnp.sum(diff ** 2, axis=-1))  # (K, K)
    mask = jnp.triu(jnp.ones((K, K), dtype=bool), k=1)
    n_pairs = K * (K - 1) / 2.0
    return jnp.sum(dists * mask) / jnp.maximum(n_pairs, 1.0)


@jax.jit
def rbf_kernel_jit(particles, bandwidth):
    """RBF 커널 행렬 (GPU).

    Args:
        particles: (K, D)
        bandwidth: h

    Returns:
        (K, K) 커널 행렬
    """
    diff = particles[:, None, :] - particles[None, :, :]
    sq_dist = jnp.sum(diff ** 2, axis=-1)
    return jnp.exp(-sq_dist / (2.0 * bandwidth ** 2))
