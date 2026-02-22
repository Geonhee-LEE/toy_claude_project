"""JAX 난수 생성 — GPU 가속 샘플링.

JAX의 함수형 PRNG (jax.random) 사용.
Gaussian: 단순 정규 분포, Colored: lax.scan 기반 OU 프로세스.
K, N, nu는 shape에 사용되므로 static_argnums로 지정.
"""

import jax
import jax.numpy as jnp
from jax import random, lax
from functools import partial


@partial(jax.jit, static_argnums=(1, 2, 3))
def gaussian_sample_jit(key, K, N, nu, sigma):
    """가우시안 노이즈 샘플 생성 (JAX).

    noise ~ N(0, diag(σ²))

    Args:
        key: JAX PRNG key
        K: 샘플 수 (static)
        N: 호라이즌 길이 (static)
        nu: 제어 차원 (static)
        sigma: (nu,) 표준편차

    Returns:
        (K, N, nu) 노이즈
    """
    noise = random.normal(key, shape=(K, N, nu))
    return noise * sigma[None, None, :]


@partial(jax.jit, static_argnums=(1, 2, 3))
def colored_noise_sample_jit(key, K, N, nu, sigma, beta):
    """Colored noise (OU 프로세스) 샘플 생성 (JAX).

    lax.scan으로 시간축 순차 처리, K축은 자동 병렬화.

    decay = exp(-β)
    diffusion = σ · √(1 - decay²)
    ε[t] = decay · ε[t-1] + diffusion · w[t]

    Args:
        key: JAX PRNG key
        K: 샘플 수 (static)
        N: 호라이즌 길이 (static)
        nu: 제어 차원 (static)
        sigma: (nu,) 표준편차
        beta: 역상관 속도

    Returns:
        (K, N, nu) 노이즈 (시간 자기상관 존재)
    """
    decay = jnp.exp(-beta)
    diffusion = sigma * jnp.sqrt(1.0 - decay ** 2)

    # N개 키 생성 (초기 + N-1 스텝)
    keys = random.split(key, N)

    # 초기 샘플: 정상 분포 σ
    init = random.normal(keys[0], shape=(K, nu)) * sigma[None, :]

    # lax.scan: ε[t] = decay * ε[t-1] + diffusion * w[t]
    def scan_step(carry, key_t):
        prev = carry
        w = random.normal(key_t, shape=(K, nu))
        next_val = decay * prev + diffusion[None, :] * w
        return next_val, next_val

    _, rest = lax.scan(scan_step, init, keys[1:])  # (N-1, K, nu)

    # init + rest → (N, K, nu) → transpose → (K, N, nu)
    noise = jnp.concatenate([init[None, :, :], rest], axis=0)
    return jnp.transpose(noise, (1, 0, 2))
