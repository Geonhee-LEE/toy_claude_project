"""JAX/NumPy 백엔드 추상화 — GPU 가속 기반 인프라.

JAX GPU가 사용 가능하면 GPU 경로, 아니면 CPU fallback.
jax_enable_x64=True로 float64 정밀도 유지.
"""

import numpy as np

_JAX_AVAILABLE = False
_GPU_AVAILABLE = False

try:
    import jax
    import jax.numpy as jnp
    from jax import random as jax_random

    jax.config.update("jax_enable_x64", True)
    _JAX_AVAILABLE = True
    _GPU_AVAILABLE = any(d.platform == "gpu" for d in jax.devices())
except (ImportError, RuntimeError):
    jnp = None
    jax_random = None


def is_jax_available() -> bool:
    """JAX 설치 여부 (CPU/GPU 무관)."""
    return _JAX_AVAILABLE


def is_gpu_available() -> bool:
    """JAX GPU 디바이스 사용 가능 여부."""
    return _GPU_AVAILABLE


def get_backend_name() -> str:
    """현재 백엔드 이름 반환."""
    if _GPU_AVAILABLE:
        return "jax-gpu"
    if _JAX_AVAILABLE:
        return "jax-cpu"
    return "numpy"


def to_jax(arr: np.ndarray, dtype=None):
    """NumPy 배열 → JAX 배열 변환."""
    if not _JAX_AVAILABLE:
        raise RuntimeError("JAX is not installed")
    if dtype is not None:
        return jnp.asarray(arr, dtype=dtype)
    return jnp.asarray(arr)


def to_numpy(arr) -> np.ndarray:
    """JAX 배열 → NumPy 배열 변환."""
    return np.asarray(arr)


def get_dtype(use_float32: bool = False):
    """GPU 연산용 dtype 결정."""
    if not _JAX_AVAILABLE:
        return np.float64
    return jnp.float32 if use_float32 else jnp.float64
