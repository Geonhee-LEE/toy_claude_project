"""MPPI 파라미터 데이터클래스."""

from dataclasses import dataclass

import numpy as np


@dataclass
class MPPIParams:
    """MPPI 튜닝 파라미터.

    Attributes:
        N: 예측 호라이즌 스텝 수
        dt: 시간 간격 [s]
        K: 샘플 수
        lambda_: 온도 파라미터 (softmax 스케일링)
        noise_sigma: 제어 입력 노이즈 표준편차 [v_sigma, omega_sigma]
        Q: 상태 가중 행렬 [x, y, theta]
        R: 제어 가중 행렬 [v, omega]
        Qf: 터미널 상태 가중 행렬
    """

    # Horizon
    N: int = 30
    dt: float = 0.05

    # Sampling
    K: int = 1024
    lambda_: float = 10.0

    # Noise standard deviation [v, omega]
    noise_sigma: np.ndarray | None = None

    # State weights [x, y, theta]
    Q: np.ndarray | None = None

    # Control weights [v, omega]
    R: np.ndarray | None = None

    # Terminal state weights
    Qf: np.ndarray | None = None

    # Control rate weights (u_t - u_{t-1}) penalty — None = 비활성
    R_rate: np.ndarray | None = None

    # Adaptive temperature (ESS 기반 λ 자동 조정)
    adaptive_temperature: bool = False
    adaptive_temp_config: dict | None = None

    # Colored noise sampling (Ornstein-Uhlenbeck 프로세스)
    colored_noise: bool = False
    noise_beta: float = 2.0

    # Tsallis-MPPI (Yin et al., 2021 — Variational Inference MPC using Tsallis Divergence)
    tsallis_q: float = 1.0  # 1.0=Vanilla(Shannon), >1=heavy-tail, <1=light-tail

    # M3c Risk-Aware MPPI (CVaR weight truncation)
    # 1.0=risk-neutral(Vanilla), <1=risk-averse, 실용 범위 [0.1, 1.0]
    cvar_alpha: float = 1.0

    # M3d Stein Variational MPPI (Lambert et al., 2020 — SVMPC)
    # SVGD로 샘플 간 상호작용(매력+반발)을 통해 분포 개선
    svgd_num_iterations: int = 0       # 0=Vanilla 동등(비활성), 권장: 1~5
    svgd_step_size: float = 0.1        # SVGD update step size
    svgd_bandwidth: float | None = None  # RBF bandwidth. None=median heuristic

    # M3.5a Smooth MPPI (Kim et al., 2021 — input-lifting Δu space)
    smooth_mppi_enabled: bool = False
    smooth_R_jerk: np.ndarray | None = None         # (nu,) jerk 가중치. None=기본값 [0.1, 0.1]
    smooth_action_cost_weight: float = 1.0           # jerk cost 전체 스케일

    # M3.5b Spline-MPPI (ICRA 2024 — B-spline basis 보간)
    spline_num_knots: int = 12                       # 제어점 수 (P << N)
    spline_degree: int = 3                           # B-spline 차수 (cubic)
    spline_knot_sigma: np.ndarray | None = None      # (nu,) knot 노이즈 σ. None=auto or noise_sigma
    spline_auto_knot_sigma: bool = True              # basis 감쇠 자동 보정 (σ × amp_factor)

    # M3.5c SVG-MPPI (Kondo et al., ICRA 2024 — Guide particle SVGD)
    svg_num_guide_particles: int = 16                # Guide 수 (G << K)
    svg_guide_step_size: float = 0.2                 # SVGD step size for guides
    svg_guide_iterations: int = 3                    # Guide SVGD 반복 횟수
    svg_resample_std: float = 0.1                    # Follower 리샘플링 σ 스케일

    # Tube-MPPI (Williams et al., 2018 — Robust Sampling Based MPPI)
    tube_enabled: bool = False
    tube_K_fb: np.ndarray | None = None           # (nu, nx) 피드백 게인. None=기본값
    tube_max_correction: np.ndarray | None = None  # [dv_max, domega_max]. None=기본값
    tube_disturbance_bound: float = 0.1            # 예상 외란 크기 (튜브 폭 추정용)
    tube_nominal_reset_threshold: float = 1.0      # 편차 > threshold → 명목 상태 리셋

    def __post_init__(self):
        if self.noise_sigma is None:
            self.noise_sigma = np.array([0.3, 0.3])
        if self.Q is None:
            self.Q = np.diag([10.0, 10.0, 1.0])
        if self.R is None:
            self.R = np.diag([0.01, 0.01])
        if self.Qf is None:
            self.Qf = np.diag([50.0, 50.0, 5.0])


@dataclass
class CBFParams:
    """CBF (Control Barrier Function) 파라미터.

    Attributes:
        enabled: CBF 기능 활성화 여부 (False이면 Vanilla MPPI 동일)
        gamma: class-K 함수 계수 — CBF 감쇠율
        safety_margin: 추가 안전 마진 [m]
        robot_radius: 로봇 반경 [m]
        activation_distance: CBF 활성화 거리 [m]
        cost_weight: Soft CBF cost 가중치
        use_safety_filter: Post-hoc QP safety filter 활성화
    """

    enabled: bool = False
    gamma: float = 1.0
    safety_margin: float = 0.3
    robot_radius: float = 0.2
    activation_distance: float = 3.0
    cost_weight: float = 500.0
    use_safety_filter: bool = True
