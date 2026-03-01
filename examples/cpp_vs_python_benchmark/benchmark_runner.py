"""파이프라인 + 컴포넌트 + 스케일링 3종 벤치마크 오케스트레이션.

3가지 벤치마크:
  1. Pipeline: 시뮬레이션 루프를 통한 Python/C++ 파이프라인 비교
  2. Component: 개별 컴포넌트(sampling, rollout, cost, weight) 마이크로벤치마크
  3. Scaling: K/N 변화에 따른 성능 스케일링 분석
"""

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from mpc_controller import (
    MPPIParams as PyMPPIParams,
    RobotParams,
    TrajectoryInterpolator,
)
from mpc_controller.controllers.mppi.base_mppi import MPPIController
from mpc_controller.controllers.mppi.log_mppi import LogMPPIController
from mpc_controller.controllers.mppi.tsallis_mppi import TsallisMPPIController
from mpc_controller.controllers.mppi.risk_aware_mppi import RiskAwareMPPIController
from simulation.simulator import Simulator, SimulationConfig

from .cpp_mppi_assembler import CppMPPIAssembler, MODEL_CONFIG
from .scenario import BenchmarkScenario, LookaheadInterpolator, circle_scenario, get_model_nx


# ─────────────────────────────────────────────────────────────
# 데이터 클래스
# ─────────────────────────────────────────────────────────────

@dataclass
class RunResult:
    """단일 설정의 시뮬레이션 결과."""
    name: str
    backend: str                 # "python" or "cpp"
    model_type: str
    weight_type: str
    ref_mode: str                # "time" or "lookahead"
    states: np.ndarray           # (n_steps+1, nx)
    controls: np.ndarray         # (n_steps, nu)
    tracking_errors: np.ndarray  # (n_steps, nx)
    solve_times: np.ndarray      # (n_steps,)
    costs: np.ndarray            # (n_steps,)
    ess_history: np.ndarray      # (n_steps,)

    @property
    def position_rmse(self) -> float:
        pos_err = self.tracking_errors[:, :2]
        return float(np.sqrt(np.mean(np.sum(pos_err ** 2, axis=1))))

    @property
    def avg_solve_ms(self) -> float:
        return float(np.mean(self.solve_times)) * 1000


@dataclass
class ComponentResult:
    """컴포넌트 마이크로벤치마크 결과."""
    component: str       # "sampling", "rollout", "cost", "weight"
    backend: str         # "python" or "cpp"
    K: int
    N: int
    times_ms: List[float]

    @property
    def median_ms(self) -> float:
        return float(np.median(self.times_ms))

    @property
    def std_ms(self) -> float:
        return float(np.std(self.times_ms))


@dataclass
class ScalingResult:
    """스케일링 분석 결과."""
    variable: str        # "K" or "N"
    values: List[int]
    py_times_ms: List[float]
    cpp_times_ms: List[float]


@dataclass
class BenchmarkResults:
    """전체 벤치마크 결과 컨테이너."""
    pipeline: List[RunResult] = field(default_factory=list)
    component: List[ComponentResult] = field(default_factory=list)
    scaling: List[ScalingResult] = field(default_factory=list)

    def to_dict(self) -> dict:
        """JSON 직렬화용 dict 변환."""
        return {
            "pipeline": [
                {
                    "name": r.name, "backend": r.backend,
                    "model": r.model_type, "weight": r.weight_type,
                    "ref_mode": r.ref_mode,
                    "position_rmse": r.position_rmse,
                    "avg_solve_ms": r.avg_solve_ms,
                    "num_steps": len(r.solve_times),
                }
                for r in self.pipeline
            ],
            "component": [
                {
                    "component": c.component, "backend": c.backend,
                    "K": c.K, "N": c.N,
                    "median_ms": c.median_ms, "std_ms": c.std_ms,
                }
                for c in self.component
            ],
            "scaling": [
                {
                    "variable": s.variable,
                    "values": s.values,
                    "py_times_ms": s.py_times_ms,
                    "cpp_times_ms": s.cpp_times_ms,
                }
                for s in self.scaling
            ],
        }


# ─────────────────────────────────────────────────────────────
# Python 컨트롤러 팩토리 (DiffDrive 전용)
# ─────────────────────────────────────────────────────────────

def _py_base_params(K: int = 512, N: int = 20) -> dict:
    return dict(
        N=N, K=K, dt=0.05, lambda_=10.0,
        noise_sigma=np.array([0.3, 0.3]),
        Q=np.diag([10.0, 10.0, 1.0]),
        R=np.diag([0.01, 0.01]),
        Qf=np.diag([100.0, 100.0, 10.0]),
        adaptive_temperature=True,
        adaptive_temp_config={
            "target_ess_ratio": 0.5,
            "adaptation_rate": 1.0,
            "lambda_min": 0.001,
            "lambda_max": 100.0,
        },
    )


def create_python_controller(
    weight_type: str,
    K: int = 512,
    N: int = 20,
    seed: int = 42,
    obstacles: Optional[np.ndarray] = None,
) -> MPPIController:
    """Python MPPI 컨트롤러 생성 (DiffDrive 전용)."""
    bp = _py_base_params(K, N)
    robot_params = RobotParams()

    if weight_type == "vanilla":
        return MPPIController(robot_params, PyMPPIParams(**bp), seed=seed, obstacles=obstacles)
    elif weight_type == "log":
        return LogMPPIController(robot_params, PyMPPIParams(**bp), seed=seed, obstacles=obstacles)
    elif weight_type == "tsallis":
        bp["tsallis_q"] = 1.1
        return TsallisMPPIController(robot_params, PyMPPIParams(**bp), seed=seed, obstacles=obstacles)
    elif weight_type == "risk_aware":
        bp["cvar_alpha"] = 0.7
        return RiskAwareMPPIController(robot_params, PyMPPIParams(**bp), seed=seed, obstacles=obstacles)
    else:
        raise ValueError(f"Unknown weight_type: {weight_type}")


# ─────────────────────────────────────────────────────────────
# 시뮬레이션 루프
# ─────────────────────────────────────────────────────────────

def simulate(
    controller,
    name: str,
    backend: str,
    model_type: str,
    weight_type: str,
    interpolator,
    initial_state: np.ndarray,
    sim_time: float = 10.0,
    dt: float = 0.05,
) -> RunResult:
    """시뮬레이션 루프 실행.

    interpolator는 TrajectoryInterpolator (시간 기반) 또는
    LookaheadInterpolator (위치 기반) 중 하나를 받는다.
    """
    is_lookahead = isinstance(interpolator, LookaheadInterpolator)
    ref_mode = "lookahead" if is_lookahead else "time"

    robot_params = RobotParams()
    sim_config = SimulationConfig(dt=dt, max_time=sim_time)
    sim = Simulator(robot_params, sim_config)
    sim.reset(initial_state[:3])  # Simulator는 (x,y,theta)만 사용

    # controller 파라미터에서 N, dt 추출
    if hasattr(controller, 'params'):
        ctrl_N = controller.params.N if hasattr(controller.params, 'N') else 20
        ctrl_dt = controller.params.dt if hasattr(controller.params, 'dt') else dt
    else:
        ctrl_N = 20
        ctrl_dt = dt

    controller.reset()
    if is_lookahead:
        interpolator.reset()
    num_steps = int(sim_time / dt)
    nx = len(initial_state)

    states = [initial_state.copy()]
    controls_list, errors_list = [], []
    solve_times_list, costs_list, ess_list = [], [], []

    for step in range(num_steps):
        t = step * dt
        state = sim.get_measurement()

        # 상태 확장 (NonCoaxial: nx=4)
        if nx > 3:
            state_ext = np.zeros(nx)
            state_ext[:3] = state
            state = state_ext

        # 참조 궤적 생성: time-based vs lookahead
        if is_lookahead:
            ref = interpolator.get_reference(
                state, ctrl_N, ctrl_dt, current_theta=state[2],
            )
        else:
            ref = interpolator.get_reference(
                t, ctrl_N, ctrl_dt, current_theta=state[2],
            )

        # 참조 궤적 차원 확장
        if nx > 3 and ref.shape[1] < nx:
            ref_ext = np.zeros((ref.shape[0], nx))
            ref_ext[:, :ref.shape[1]] = ref
            ref = ref_ext

        control, info = controller.compute_control(state, ref)

        # Simulator는 (v, omega)만 사용
        sim_control = control[:2] if len(control) > 2 else control
        next_state = sim.step(sim_control)

        # 상태 확장
        if nx > 3:
            next_ext = np.zeros(nx)
            next_ext[:3] = next_state
            next_state = next_ext

        error = np.zeros(nx)
        error[:3] = sim.compute_tracking_error(state[:3], ref[0, :3])

        states.append(next_state.copy())
        controls_list.append(control.copy())
        errors_list.append(error)
        solve_times_list.append(info["solve_time"])
        costs_list.append(info["cost"])
        ess_list.append(info.get("ess", 0))

        # 궤적 완료 검사
        idx, dist = interpolator.find_closest_point(state[:2])
        if idx >= interpolator.num_points - 1 and dist < 0.1:
            break

    return RunResult(
        name=name,
        backend=backend,
        model_type=model_type,
        weight_type=weight_type,
        ref_mode=ref_mode,
        states=np.array(states),
        controls=np.array(controls_list) if controls_list else np.empty((0, 2)),
        tracking_errors=np.array(errors_list) if errors_list else np.empty((0, nx)),
        solve_times=np.array(solve_times_list),
        costs=np.array(costs_list),
        ess_history=np.array(ess_list),
    )


# ─────────────────────────────────────────────────────────────
# 1. Pipeline Benchmark
# ─────────────────────────────────────────────────────────────

WEIGHT_TYPES = ["vanilla", "log", "tsallis", "risk_aware"]


def _make_interpolators(
    trajectory: np.ndarray,
    dt: float,
    ref_mode: str,
) -> List:
    """ref_mode에 따라 interpolator 리스트 반환.

    Returns
    -------
    list of (label_suffix, interpolator) tuples
        label_suffix: ""(단일 모드) 또는 "/time", "/look" (both 모드)
    """
    traj_xy = trajectory[:, :3]
    if ref_mode == "time":
        return [("", TrajectoryInterpolator(traj_xy, dt))]
    elif ref_mode == "lookahead":
        return [("", LookaheadInterpolator(trajectory, dt))]
    else:  # "both"
        return [
            ("/time", TrajectoryInterpolator(traj_xy, dt)),
            ("/look", LookaheadInterpolator(trajectory, dt)),
        ]


def run_pipeline_benchmark(
    scenario: BenchmarkScenario,
    K: int = 512,
    N: int = 20,
    seed: int = 42,
    ref_mode: str = "time",
) -> List[RunResult]:
    """파이프라인 벤치마크: DiffDrive에서 Py/C++ 비교 + Swerve/NonCoaxial C++ only.

    Parameters
    ----------
    ref_mode : str
        "time" — 기존 시간 기반 인터폴레이션
        "lookahead" — 위치 기반 lookahead 인터폴레이션
        "both" — 양쪽 모두 실행하여 비교
    """
    results = []
    interps_dd = _make_interpolators(scenario.trajectory, scenario.dt, ref_mode)

    for wt in WEIGHT_TYPES:
        for suffix, interp in interps_dd:
            # Python DiffDrive
            py_ctrl = create_python_controller(wt, K=K, N=N, seed=seed,
                                               obstacles=scenario.obstacles)
            result = simulate(
                py_ctrl, f"Py/{wt}{suffix}", "python", "diff_drive", wt,
                interp, np.zeros(3), scenario.sim_time, scenario.dt,
            )
            results.append(result)

            # C++ DiffDrive
            cpp_ctrl = CppMPPIAssembler(
                "diff_drive", wt, K=K, N=N, seed=seed,
                obstacles=scenario.obstacles,
            )
            result = simulate(
                cpp_ctrl, f"C++/{wt}{suffix}", "cpp", "diff_drive", wt,
                interp, np.zeros(3), scenario.sim_time, scenario.dt,
            )
            results.append(result)

    # Swerve / NonCoaxial — C++ only
    for model_type in ["swerve", "non_coaxial_swerve"]:
        nx = get_model_nx(model_type)
        traj = scenario.trajectory.copy()
        if traj.shape[1] < nx:
            traj_ext = np.zeros((traj.shape[0], nx))
            traj_ext[:, :traj.shape[1]] = traj
            traj = traj_ext

        interps = _make_interpolators(traj, scenario.dt, ref_mode)

        for wt in WEIGHT_TYPES:
            for suffix, interp in interps:
                cpp_ctrl = CppMPPIAssembler(
                    model_type, wt, K=K, N=N, seed=seed,
                    obstacles=scenario.obstacles,
                )
                result = simulate(
                    cpp_ctrl, f"C++/{model_type[:3]}/{wt}{suffix}",
                    "cpp", model_type, wt,
                    interp, np.zeros(nx), scenario.sim_time, scenario.dt,
                )
                results.append(result)

    return results


# ─────────────────────────────────────────────────────────────
# 2. Component Benchmark
# ─────────────────────────────────────────────────────────────

def _benchmark_sampling_py(K: int, N: int, nu: int, repeat: int) -> List[float]:
    from mpc_controller.controllers.mppi.sampling import GaussianSampler
    sampler = GaussianSampler(np.ones(nu) * 0.3, seed=42)
    times = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        sampler.sample(K, N, nu)
        times.append((time.perf_counter() - t0) * 1000)
    return times


def _benchmark_sampling_cpp(K: int, N: int, nu: int, repeat: int) -> List[float]:
    from mpc_controller_ros2.mppi import GaussianSampler
    sampler = GaussianSampler(np.ones(nu) * 0.3, seed=42)
    times = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        sampler.sample(K, N, nu)
        times.append((time.perf_counter() - t0) * 1000)
    return times


def _benchmark_rollout_py(K: int, N: int, repeat: int) -> List[float]:
    from mpc_controller.controllers.mppi.dynamics_wrapper import BatchDynamicsWrapper
    dyn = BatchDynamicsWrapper()
    x0 = np.zeros(3)
    controls = np.random.randn(K, N, 2) * 0.3
    times = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        dyn.rollout_batch(x0, controls, 0.05)
        times.append((time.perf_counter() - t0) * 1000)
    return times


def _benchmark_rollout_cpp(K: int, N: int, repeat: int) -> List[float]:
    from mpc_controller_ros2.mppi import MPPIParams, BatchDynamicsWrapper, DiffDriveModel
    model = DiffDriveModel(0.0, 1.0, -1.5, 1.5)
    p = MPPIParams()
    p.N = N
    p.K = K
    dyn = BatchDynamicsWrapper(p, model)
    x0 = np.zeros(3)
    controls_list = [np.random.randn(N, 2) * 0.3 for _ in range(K)]
    times = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        dyn.rolloutBatch(x0, controls_list, 0.05)
        times.append((time.perf_counter() - t0) * 1000)
    return times


def _benchmark_cost_py(K: int, N: int, repeat: int) -> List[float]:
    from mpc_controller.controllers.mppi.cost_functions import (
        CompositeMPPICost, StateTrackingCost, TerminalCost, ControlEffortCost,
    )
    cost = CompositeMPPICost()
    cost.add(StateTrackingCost(np.diag([10.0, 10.0, 1.0])))
    cost.add(TerminalCost(np.diag([100.0, 100.0, 10.0])))
    cost.add(ControlEffortCost(np.diag([0.01, 0.01])))

    trajs = np.random.randn(K, N + 1, 3) * 0.5
    controls = np.random.randn(K, N, 2) * 0.3
    ref = np.zeros((N + 1, 3))
    ref[:, 0] = np.linspace(0, 1, N + 1)
    times = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        cost.compute(trajs, controls, ref)
        times.append((time.perf_counter() - t0) * 1000)
    return times


def _benchmark_cost_cpp(K: int, N: int, repeat: int) -> List[float]:
    from mpc_controller_ros2.mppi import (
        StateTrackingCost, TerminalCost, ControlEffortCost,
    )
    costs_fns = [
        StateTrackingCost(np.diag([10.0, 10.0, 1.0])),
        TerminalCost(np.diag([100.0, 100.0, 10.0])),
        ControlEffortCost(np.diag([0.01, 0.01])),
    ]

    trajs = [np.random.randn(N + 1, 3) * 0.5 for _ in range(K)]
    controls = [np.random.randn(N, 2) * 0.3 for _ in range(K)]
    ref = np.zeros((N + 1, 3))
    ref[:, 0] = np.linspace(0, 1, N + 1)
    times = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        total = np.zeros(K)
        for cf in costs_fns:
            total += cf.compute(trajs, controls, ref)
        times.append((time.perf_counter() - t0) * 1000)
    return times


def _benchmark_weight_py(K: int, repeat: int) -> List[float]:
    costs = np.random.rand(K) * 100
    times = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        # Python softmax 가중치
        shifted = costs - np.min(costs)
        w = np.exp(-shifted / 10.0)
        w /= np.sum(w)
        times.append((time.perf_counter() - t0) * 1000)
    return times


def _benchmark_weight_cpp(K: int, repeat: int) -> List[float]:
    from mpc_controller_ros2.mppi import VanillaMPPIWeights
    wf = VanillaMPPIWeights()
    costs = np.random.rand(K) * 100
    times = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        wf.compute(costs, 10.0)
        times.append((time.perf_counter() - t0) * 1000)
    return times


def run_component_benchmark(
    K: int = 512,
    N: int = 20,
    repeat: int = 20,
) -> List[ComponentResult]:
    """컴포넌트별 마이크로벤치마크."""
    results = []

    # warmup
    _benchmark_sampling_py(32, 5, 2, 1)
    _benchmark_sampling_cpp(32, 5, 2, 1)

    for comp, py_fn, cpp_fn in [
        ("sampling",
         lambda: _benchmark_sampling_py(K, N, 2, repeat),
         lambda: _benchmark_sampling_cpp(K, N, 2, repeat)),
        ("rollout",
         lambda: _benchmark_rollout_py(K, N, repeat),
         lambda: _benchmark_rollout_cpp(K, N, repeat)),
        ("cost",
         lambda: _benchmark_cost_py(K, N, repeat),
         lambda: _benchmark_cost_cpp(K, N, repeat)),
        ("weight",
         lambda: _benchmark_weight_py(K, repeat),
         lambda: _benchmark_weight_cpp(K, repeat)),
    ]:
        py_times = py_fn()
        cpp_times = cpp_fn()
        results.append(ComponentResult(comp, "python", K, N, py_times))
        results.append(ComponentResult(comp, "cpp", K, N, cpp_times))

    return results


# ─────────────────────────────────────────────────────────────
# 3. Scaling Benchmark
# ─────────────────────────────────────────────────────────────

def run_scaling_benchmark(
    K_values: Optional[List[int]] = None,
    N_values: Optional[List[int]] = None,
    repeat: int = 20,
) -> List[ScalingResult]:
    """K/N 스케일링 분석."""
    if K_values is None:
        K_values = [256, 512, 1024, 2048, 4096]
    if N_values is None:
        N_values = [10, 20, 30, 50]

    results = []

    # K 스케일링 (N=20 고정)
    N_fixed = 20
    py_k, cpp_k = [], []
    for K in K_values:
        # warmup
        _benchmark_rollout_py(32, 5, 1)
        _benchmark_rollout_cpp(32, 5, 1)

        py_times = _benchmark_rollout_py(K, N_fixed, repeat)
        cpp_times = _benchmark_rollout_cpp(K, N_fixed, repeat)
        py_k.append(float(np.median(py_times)))
        cpp_k.append(float(np.median(cpp_times)))

    results.append(ScalingResult("K", K_values, py_k, cpp_k))

    # N 스케일링 (K=512 고정)
    K_fixed = 512
    py_n, cpp_n = [], []
    for N in N_values:
        py_times = _benchmark_rollout_py(K_fixed, N, repeat)
        cpp_times = _benchmark_rollout_cpp(K_fixed, N, repeat)
        py_n.append(float(np.median(py_times)))
        cpp_n.append(float(np.median(cpp_times)))

    results.append(ScalingResult("N", N_values, py_n, cpp_n))

    return results


# ─────────────────────────────────────────────────────────────
# 통합 오케스트레이터
# ─────────────────────────────────────────────────────────────

def run_all_benchmarks(
    K: int = 512,
    N: int = 20,
    repeat: int = 20,
    scenario: Optional[BenchmarkScenario] = None,
    run_pipeline: bool = True,
    run_component: bool = True,
    run_scaling: bool = True,
    K_values: Optional[List[int]] = None,
    N_values: Optional[List[int]] = None,
    ref_mode: str = "time",
) -> BenchmarkResults:
    """전체 벤치마크 실행."""
    if scenario is None:
        scenario = circle_scenario(nx=3)

    results = BenchmarkResults()

    if run_pipeline:
        print("=" * 60)
        print(f"  Pipeline Benchmark  (ref_mode={ref_mode})")
        print("=" * 60)
        results.pipeline = run_pipeline_benchmark(
            scenario, K=K, N=N, ref_mode=ref_mode,
        )
        print(f"  → {len(results.pipeline)} configs 완료")

    if run_component:
        print("=" * 60)
        print("  Component Benchmark")
        print("=" * 60)
        results.component = run_component_benchmark(K=K, N=N, repeat=repeat)
        print(f"  → {len(results.component)} measurements 완료")

    if run_scaling:
        print("=" * 60)
        print("  Scaling Benchmark")
        print("=" * 60)
        results.scaling = run_scaling_benchmark(K_values, N_values, repeat=repeat)
        print(f"  → {len(results.scaling)} scaling curves 완료")

    return results
