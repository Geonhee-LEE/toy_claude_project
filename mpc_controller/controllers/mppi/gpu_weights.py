"""JAX JIT 가중치 함수 — 변형별 GPU Strategy.

순수 함수 Strategy 패턴: JAX JIT 호환 (클래스 인스턴스는 tracer 추적 불가).
closure로 파라미터 캡처 → JIT 재컴파일 방지.

┌──────────────────┬───────────────────────────────────────┐
│ 변형             │ 가중치 함수                           │
├──────────────────┼───────────────────────────────────────┤
│ Vanilla          │ vanilla_softmax_weights               │
│ Log-MPPI         │ log_softmax_weights                   │
│ Tsallis-MPPI     │ make_tsallis_weights(q) → closure     │
│ Risk-Aware(CVaR) │ make_cvar_weights(alpha, K) → closure │
└──────────────────┴───────────────────────────────────────┘
"""

import jax
import jax.numpy as jnp


@jax.jit
def vanilla_softmax_weights(costs, lambda_):
    """Vanilla softmax 가중치 (JAX).

    w_k = exp(-S_k / λ) / Σ exp(-S_j / λ)

    Args:
        costs: (K,) 비용 배열
        lambda_: 온도 파라미터

    Returns:
        (K,) 정규화된 가중치
    """
    shifted = -costs / lambda_
    shifted = shifted - jnp.max(shifted)
    exp_vals = jnp.exp(shifted)
    return exp_vals / jnp.sum(exp_vals)


@jax.jit
def log_softmax_weights(costs, lambda_):
    """Log-space softmax 가중치 (JAX).

    log-space에서 계산하여 극단 cost(1e15)에서 NaN/Inf 방지.
    logsumexp로 수치 안정성 보장.

    Args:
        costs: (K,) 비용 배열
        lambda_: 온도 파라미터

    Returns:
        (K,) 정규화된 가중치
    """
    log_w = -costs / lambda_
    log_w = log_w - jax.scipy.special.logsumexp(log_w)
    return jnp.exp(log_w)


def make_tsallis_weights(q):
    """Tsallis q-exponential 가중치 함수 팩토리.

    q를 closure로 캡처 → JIT 재컴파일 방지.
    q=1.0 → Shannon (Vanilla), q>1 → heavy-tail, q<1 → light-tail.

    Args:
        q: Tsallis 파라미터

    Returns:
        JIT 컴파일된 가중치 함수 (costs, lambda_) → (K,)
    """
    @jax.jit
    def tsallis_weights(costs, lambda_):
        centered = costs - jnp.min(costs)
        x = -centered / lambda_
        base = 1.0 + (1.0 - q) * x
        exponent = 1.0 / (1.0 - q)
        raw = jnp.where(base > 0, jnp.power(base, exponent), 0.0)
        total = jnp.sum(raw)
        return jnp.where(total > 0, raw / total, jnp.ones_like(costs) / costs.shape[0])

    return tsallis_weights


def make_cvar_weights(alpha, K):
    """CVaR 가중치 함수 팩토리.

    alpha, K를 closure로 캡처 → n_keep 정적 결정.
    최저 비용 ceil(α·K)개만 선택하여 softmax.

    Args:
        alpha: CVaR 절단 비율 (1.0=전체, <1=risk-averse)
        K: 샘플 수

    Returns:
        JIT 컴파일된 가중치 함수 (costs, lambda_) → (K,)
    """
    n_keep = max(1, int(jnp.ceil(alpha * K)))
    # alpha >= 1.0이면 전체 사용
    if alpha >= 1.0 or n_keep >= K:
        return vanilla_softmax_weights

    @jax.jit
    def cvar_weights(costs, lambda_):
        # 최저비용 n_keep개 선택: top_k(-costs)
        neg_costs = -costs
        _, keep_idx = jax.lax.top_k(neg_costs, n_keep)
        mask = jnp.zeros(K).at[keep_idx].set(1.0)
        shifted = -costs / lambda_
        shifted = shifted - jnp.max(shifted)
        exp_vals = jnp.exp(shifted) * mask
        return exp_vals / jnp.sum(exp_vals)

    return cvar_weights


# ─── Registry ───

WEIGHT_FN_REGISTRY = {
    "vanilla": vanilla_softmax_weights,
    "log": log_softmax_weights,
}


def get_weight_fn(name, **params):
    """이름으로 가중치 함수 조회.

    Args:
        name: "vanilla", "log", "tsallis", "cvar"
        **params: make_tsallis_weights(q=...), make_cvar_weights(alpha=..., K=...)

    Returns:
        JIT 컴파일된 가중치 함수
    """
    if name in WEIGHT_FN_REGISTRY:
        return WEIGHT_FN_REGISTRY[name]
    if name == "tsallis":
        return make_tsallis_weights(params.get("q", 1.0))
    if name == "cvar":
        return make_cvar_weights(params.get("alpha", 1.0), params.get("K", 1024))
    raise ValueError(f"Unknown weight function: {name}")
