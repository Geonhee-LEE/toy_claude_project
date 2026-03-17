#!/usr/bin/env python3
"""
Paper-Ready 벤치마크 오케스트레이터

기존 controller_benchmark.py를 확장하여:
  - 다중 시행 (N회 반복 + 워밍업)
  - 메타데이터 수집 (git hash, CPU, ROS distro, 파라미터)
  - 시나리오 YAML 기반 자동 실행
  - per-run JSON 저장 → paper_benchmark_analysis.py로 분석

사용법:
    # 전체 논문용 벤치마크 (paper_core × maze_nav, 3회 반복)
    python3 scripts/paper_benchmark.py

    # 특정 시나리오 + 그룹
    python3 scripts/paper_benchmark.py --scenario maze_nav --group paper_core

    # 빠른 검증 (3 컨트롤러, 1회)
    python3 scripts/paper_benchmark.py --group quick --trials 1

    # 설정 파일 지정
    python3 scripts/paper_benchmark.py --config config/benchmark_scenarios.yaml

    # 기존 JSON에 추가 시행
    python3 scripts/paper_benchmark.py --append-to ~/paper_benchmark_results/bench_20260317_120000
"""

import argparse
import json
import os
import platform
import re
import signal
import subprocess
import sys
import time
import yaml
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ============================================================
# 메타데이터 수집
# ============================================================

def collect_system_metadata() -> Dict[str, Any]:
    """시스템 정보 수집 (재현성용)"""
    meta = {
        'timestamp': datetime.now().isoformat(),
        'platform': platform.platform(),
        'cpu': platform.processor() or 'unknown',
        'cpu_count': os.cpu_count(),
        'python_version': platform.python_version(),
    }

    # CPU 모델 (Linux)
    try:
        with open('/proc/cpuinfo', 'r') as f:
            for line in f:
                if line.startswith('model name'):
                    meta['cpu_model'] = line.split(':')[1].strip()
                    break
    except (IOError, OSError):
        pass

    # 메모리
    try:
        with open('/proc/meminfo', 'r') as f:
            for line in f:
                if line.startswith('MemTotal'):
                    mem_kb = int(line.split()[1])
                    meta['memory_gb'] = round(mem_kb / 1024 / 1024, 1)
                    break
    except (IOError, OSError):
        pass

    # ROS distro
    meta['ros_distro'] = os.environ.get('ROS_DISTRO', 'unknown')

    # Git commit hash
    try:
        result = subprocess.run(
            ['git', 'rev-parse', '--short', 'HEAD'],
            capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            meta['git_hash'] = result.stdout.strip()
        result2 = subprocess.run(
            ['git', 'diff', '--stat', '--quiet'],
            capture_output=True, timeout=5)
        meta['git_dirty'] = (result2.returncode != 0)
    except Exception:
        pass

    return meta


def collect_controller_params(controller: str) -> Dict[str, Any]:
    """컨트롤러 YAML에서 핵심 파라미터 추출"""
    config_dir = Path(__file__).parent.parent / 'config'

    # 컨트롤러명 → YAML 파일 매핑
    yaml_map = {
        'custom': 'nav2_params_custom_mppi.yaml',
        'nav2': 'nav2_params_nav2_mppi.yaml',
        'tube_mppi': 'nav2_params_tube_mppi.yaml',
    }
    # 기본 패턴: nav2_params_{controller}_mppi.yaml
    yaml_name = yaml_map.get(controller,
                             f'nav2_params_{controller}_mppi.yaml')
    yaml_path = config_dir / yaml_name

    params = {'config_file': yaml_name}
    if not yaml_path.exists():
        return params

    try:
        with open(yaml_path) as f:
            data = yaml.safe_load(f)
        fp = data.get('controller_server', {}).get(
            'ros__parameters', {}).get('FollowPath', {})
        # 핵심 파라미터 추출
        for key in ['N', 'dt', 'K', 'lambda', 'v_max', 'omega_max',
                     'noise_sigma_v', 'noise_sigma_omega',
                     'Q_x', 'Q_y', 'Q_theta', 'R_v', 'R_omega',
                     'obstacle_weight', 'safety_distance',
                     'motion_model', 'plugin']:
            if key in fp:
                params[key] = fp[key]
    except Exception:
        pass

    return params


# ============================================================
# E2E 테스트 실행
# ============================================================

LAUNCH_PACKAGE = 'mpc_controller_ros2'
LAUNCH_FILE = 'mppi_ros2_control_nav2.launch.py'


def cleanup_processes():
    """이전 프로세스 정리"""
    patterns = [
        'gz sim', 'ruby.*gz', 'rviz2',
        'controller_server', 'planner_server', 'bt_navigator',
        'behavior_server', 'velocity_smoother',
        'amcl', 'map_server', 'lifecycle_manager',
        'ros_gz_bridge', 'ros_gz_sim',
        'robot_state_publisher', 'nav2_lifecycle',
        'spawner', 'swerve_e2e_test', 'nav2_e2e_test',
    ]
    for pat in patterns:
        subprocess.run(['pkill', '-f', pat],
                       capture_output=True, timeout=5)
    time.sleep(3)


def launch_simulation(controller: str, world: str, mapfile: str,
                      headless: bool = True) -> Optional[subprocess.Popen]:
    """Gazebo + nav2 launch"""
    cmd = [
        'ros2', 'launch', LAUNCH_PACKAGE, LAUNCH_FILE,
        f'controller:={controller}',
        f'world:={world}',
        f'map:={mapfile}',
        f'headless:={"true" if headless else "false"}',
    ]
    try:
        proc = subprocess.Popen(
            cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            preexec_fn=os.setsid)
        return proc
    except Exception as e:
        print(f'  [ERROR] Launch failed: {e}')
        return None


def wait_nav2_active(timeout: float = 60.0) -> bool:
    """nav2 lifecycle 노드 활성화 대기"""
    nodes = ['controller_server', 'planner_server', 'bt_navigator']
    start = time.time()
    stable = 0

    while time.time() - start < timeout:
        all_active = True
        for node in nodes:
            try:
                result = subprocess.run(
                    ['ros2', 'lifecycle', 'get', f'/{node}'],
                    capture_output=True, text=True, timeout=5)
                if 'active [3]' not in result.stdout:
                    all_active = False
                    break
            except Exception:
                all_active = False
                break

        if all_active:
            stable += 1
            if stable >= 3:
                return True
        else:
            stable = 0
        time.sleep(2)

    return False


def run_e2e_test(goals: List[List[float]], timeout: float) -> Dict[str, Any]:
    """swerve_e2e_test.py 실행 + 결과 파싱"""
    goal_x, goal_y = goals[0][0], goals[0][1]

    cmd = [
        'ros2', 'run', LAUNCH_PACKAGE, 'swerve_e2e_test.py',
        '--x', str(goal_x),
        '--y', str(goal_y),
        '--yaw', '0.0',
        '--timeout', str(timeout),
    ]

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True,
            timeout=timeout + 30)
        return parse_e2e_output(result.stdout + result.stderr)
    except subprocess.TimeoutExpired:
        return {'status': 'timeout', 'goal_reached': False}
    except Exception as e:
        return {'status': f'error: {e}', 'goal_reached': False}


def parse_e2e_output(output: str) -> Dict[str, Any]:
    """E2E 테스트 출력 파싱 → 메트릭 dict"""
    metrics: Dict[str, Any] = {}

    # 각 메트릭 정규식 파싱
    patterns = {
        'goal_reached': (r'goal_reached\s*[:=]\s*(True|False)', lambda m: m == 'True'),
        'travel_time': (r'travel_time\s*[:=]\s*([\d.]+)', float),
        'travel_distance': (r'travel_distance\s*[:=]\s*([\d.]+)', float),
        'position_error': (r'(?:final_)?position_error\s*[:=]\s*([\d.]+)', float),
        'yaw_error': (r'(?:final_)?yaw_error\s*[:=]\s*([\d.]+)', float),
        'mean_speed': (r'mean_speed\s*[:=]\s*([\d.]+)', float),
        'max_speed': (r'max_speed\s*[:=]\s*([\d.]+)', float),
        'mean_jerk_vx': (r'mean_jerk_vx\s*[:=]\s*([\d.]+)', float),
        'rms_jerk': (r'rms_jerk\s*[:=]\s*([\d.]+)', float),
        'stall_ratio': (r'stall_ratio\s*[:=]\s*([\d.]+)', float),
        'min_obstacle_dist': (r'min_obstacle_dist\s*[:=]\s*([\d.]+)', float),
        'mean_obstacle_dist': (r'mean_obstacle_dist\s*[:=]\s*([\d.]+)', float),
        'near_misses': (r'(?:num_)?near_misses?\s*[:=]\s*(\d+)', int),
        'collisions': (r'(?:num_)?collisions?\s*[:=]\s*(\d+)', int),
    }

    for key, (pattern, converter) in patterns.items():
        match = re.search(pattern, output, re.IGNORECASE)
        if match:
            try:
                metrics[key] = converter(match.group(1))
            except (ValueError, TypeError):
                pass

    # status 결정
    if 'goal_reached' in metrics:
        metrics['status'] = 'success' if metrics['goal_reached'] else 'goal_not_reached'
    elif 'RESULT' in output and 'PASS' in output:
        metrics['status'] = 'success'
        metrics['goal_reached'] = True
    else:
        metrics['status'] = 'parse_error'
        metrics['goal_reached'] = False

    return metrics


# ============================================================
# 벤치마크 오케스트레이터
# ============================================================

@dataclass
class TrialResult:
    """단일 시행 결과"""
    controller: str
    scenario: str
    trial_idx: int
    is_warmup: bool
    metrics: Dict[str, Any]
    duration_sec: float
    timestamp: str = ''

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()


@dataclass
class BenchmarkRun:
    """전체 벤치마크 실행 결과"""
    metadata: Dict[str, Any]
    config: Dict[str, Any]
    trials: List[Dict[str, Any]] = field(default_factory=list)


def run_benchmark(controllers: List[str], scenario_name: str,
                  scenario_cfg: Dict[str, Any], global_cfg: Dict[str, Any],
                  output_dir: Path, headless: bool = True) -> List[TrialResult]:
    """시나리오 × 컨트롤러 × N회 실행"""
    num_trials = global_cfg.get('num_trials', 3)
    warmup = global_cfg.get('warmup_trials', 1)
    timeout = global_cfg.get('timeout_per_goal', 90)
    launch_wait = global_cfg.get('launch_stabilize_sec', 35)
    nav2_wait = global_cfg.get('nav2_activate_sec', 30)

    world = scenario_cfg.get('world', 'mppi_test_simple.world')
    mapfile = scenario_cfg.get('map', 'test_arena_map.yaml')
    goals = scenario_cfg.get('goals', [[2.0, 0.0]])

    total_runs = len(controllers) * (warmup + num_trials)
    current_run = 0

    results: List[TrialResult] = []

    print(f'\n{"=" * 70}')
    print(f'  Scenario: {scenario_name} — {scenario_cfg.get("description", "")}')
    print(f'  World: {world}, Map: {mapfile}')
    print(f'  Goals: {goals}')
    print(f'  Controllers: {len(controllers)}, '
          f'Trials: {warmup}(warmup) + {num_trials}(measure)')
    print(f'  Total runs: {total_runs}')
    print(f'{"=" * 70}\n')

    for ctrl in controllers:
        ctrl_params = collect_controller_params(ctrl)
        print(f'\n--- Controller: {ctrl} ---')
        print(f'    Params: K={ctrl_params.get("K", "?")}, '
              f'N={ctrl_params.get("N", "?")}, '
              f'lambda={ctrl_params.get("lambda", "?")}')

        for trial in range(warmup + num_trials):
            is_warmup = trial < warmup
            trial_idx = trial - warmup if not is_warmup else trial
            current_run += 1

            label = f'warmup-{trial}' if is_warmup else f'trial-{trial_idx}'
            print(f'  [{current_run}/{total_runs}] {ctrl} / {label}', end=' ... ')
            sys.stdout.flush()

            # 1. Cleanup
            cleanup_processes()

            # 2. Launch
            proc = launch_simulation(ctrl, world, mapfile, headless)
            if proc is None:
                print('LAUNCH FAILED')
                results.append(TrialResult(
                    controller=ctrl, scenario=scenario_name,
                    trial_idx=trial_idx, is_warmup=is_warmup,
                    metrics={'status': 'launch_failed', 'goal_reached': False},
                    duration_sec=0))
                continue

            # 3. Wait for stabilization
            time.sleep(launch_wait)

            # 4. Wait for nav2
            nav2_ok = wait_nav2_active(nav2_wait)
            if not nav2_ok:
                print('NAV2 TIMEOUT')
                cleanup_processes()
                results.append(TrialResult(
                    controller=ctrl, scenario=scenario_name,
                    trial_idx=trial_idx, is_warmup=is_warmup,
                    metrics={'status': 'nav2_timeout', 'goal_reached': False},
                    duration_sec=0))
                continue

            # 5. Run E2E
            t0 = time.time()
            metrics = run_e2e_test(goals, timeout)
            duration = time.time() - t0

            # 6. 파라미터 메타 추가
            metrics['controller_params'] = ctrl_params

            result = TrialResult(
                controller=ctrl, scenario=scenario_name,
                trial_idx=trial_idx, is_warmup=is_warmup,
                metrics=metrics, duration_sec=duration)
            results.append(result)

            status = metrics.get('status', 'unknown')
            goal_ok = metrics.get('goal_reached', False)
            t_time = metrics.get('travel_time', 0)
            print(f'{"PASS" if goal_ok else "FAIL"} ({status}, '
                  f'{t_time:.1f}s, {duration:.0f}s total)')

            # 7. Shutdown
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            except (ProcessLookupError, OSError):
                pass
            cleanup_processes()

            # per-run JSON 저장
            run_file = output_dir / 'raw_runs' / \
                f'{ctrl}_{scenario_name}_{label}.json'
            run_file.parent.mkdir(parents=True, exist_ok=True)
            with open(run_file, 'w') as f:
                json.dump(asdict(result), f, indent=2, default=str)

    return results


# ============================================================
# 집계 + 저장
# ============================================================

def aggregate_results(results: List[TrialResult]) -> Dict[str, Any]:
    """Trial 결과 → 컨트롤러별 통계"""
    from collections import defaultdict
    import math

    # warmup 제외
    measured = [r for r in results if not r.is_warmup]

    groups: Dict[str, List[Dict]] = defaultdict(list)
    for r in measured:
        groups[r.controller].append(r.metrics)

    stats = {}
    for ctrl, metrics_list in groups.items():
        ctrl_stats: Dict[str, Any] = {
            'n_trials': len(metrics_list),
            'n_success': sum(1 for m in metrics_list if m.get('goal_reached')),
            'success_rate': sum(1 for m in metrics_list
                                if m.get('goal_reached')) / max(len(metrics_list), 1),
        }

        # 수치 메트릭별 mean/std/ci95
        numeric_keys = [
            'travel_time', 'travel_distance', 'position_error', 'yaw_error',
            'mean_speed', 'max_speed', 'rms_jerk', 'mean_jerk_vx',
            'stall_ratio', 'min_obstacle_dist', 'mean_obstacle_dist',
            'near_misses', 'collisions',
        ]
        for key in numeric_keys:
            values = [m[key] for m in metrics_list if key in m
                      and isinstance(m[key], (int, float))]
            if values:
                n = len(values)
                mean = sum(values) / n
                if n > 1:
                    variance = sum((v - mean) ** 2 for v in values) / (n - 1)
                    std = math.sqrt(variance)
                    # 95% CI (t-distribution approximation for small n)
                    t_val = {2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
                             6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262,
                             10: 2.228}.get(n, 1.96)
                    ci95 = t_val * std / math.sqrt(n)
                else:
                    std = 0.0
                    ci95 = 0.0

                ctrl_stats[key] = {
                    'mean': round(mean, 4),
                    'std': round(std, 4),
                    'ci95': round(ci95, 4),
                    'min': round(min(values), 4),
                    'max': round(max(values), 4),
                    'values': [round(v, 4) for v in values],
                }
            else:
                ctrl_stats[key] = None

        stats[ctrl] = ctrl_stats

    return stats


def save_benchmark(run: BenchmarkRun, output_dir: Path):
    """전체 벤치마크 결과 저장"""
    output_dir.mkdir(parents=True, exist_ok=True)

    # 메인 결과 파일
    main_file = output_dir / 'benchmark_results.json'
    with open(main_file, 'w') as f:
        json.dump(asdict(run), f, indent=2, default=str)

    print(f'\n  Results saved: {main_file}')

    # 집계 통계
    results = [TrialResult(**t) for t in run.trials]
    stats = aggregate_results(results)
    stats_file = output_dir / 'aggregated_stats.json'
    with open(stats_file, 'w') as f:
        json.dump({
            'metadata': run.metadata,
            'config': run.config,
            'stats': stats,
        }, f, indent=2, default=str)

    print(f'  Stats saved:   {stats_file}')

    # 요약 테이블 출력
    print_summary_table(stats)


def print_summary_table(stats: Dict[str, Any]):
    """ASCII 요약 테이블"""
    if not stats:
        return

    print(f'\n{"=" * 90}')
    print(f'  Paper Benchmark Summary')
    print(f'{"=" * 90}')
    header = (f'  {"Controller":<20s} {"Succ":>5s} {"Time":>10s} '
              f'{"PosErr":>12s} {"Jerk":>12s} {"MinObs":>12s} {"Coll":>6s}')
    print(header)
    print(f'  {"-" * 85}')

    for ctrl, s in sorted(stats.items()):
        succ = f'{s["n_success"]}/{s["n_trials"]}'

        def fmt_stat(key: str, fmt_str: str = '{:.2f}') -> str:
            v = s.get(key)
            if v is None:
                return '—'
            mean_str = fmt_str.format(v['mean'])
            ci_str = fmt_str.format(v['ci95'])
            return f'{mean_str} +/- {ci_str}'

        t_str = fmt_stat('travel_time', '{:.1f}')
        pe_str = fmt_stat('position_error', '{:.3f}')
        jk_str = fmt_stat('rms_jerk', '{:.2f}')
        mo_str = fmt_stat('min_obstacle_dist', '{:.2f}')
        co = s.get('collisions')
        co_str = str(int(co['mean'])) if co else '—'

        print(f'  {ctrl:<20s} {succ:>5s} {t_str:>10s} '
              f'{pe_str:>12s} {jk_str:>12s} {mo_str:>12s} {co_str:>6s}')

    print(f'{"=" * 90}\n')


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description='Paper-Ready MPPI Benchmark Orchestrator')
    parser.add_argument(
        '--config', type=str,
        default='config/benchmark_scenarios.yaml',
        help='벤치마크 시나리오 설정 파일')
    parser.add_argument(
        '--scenario', type=str, default='maze_nav',
        help='실행할 시나리오 (기본: maze_nav)')
    parser.add_argument(
        '--group', type=str, default='paper_core',
        help='컨트롤러 그룹 (기본: paper_core)')
    parser.add_argument(
        '--controllers', type=str, default='',
        help='직접 지정: "custom,smooth,dial"')
    parser.add_argument(
        '--trials', type=int, default=0,
        help='시행 횟수 (0=config 기본값)')
    parser.add_argument(
        '--warmup', type=int, default=-1,
        help='워밍업 횟수 (-1=config 기본값)')
    parser.add_argument(
        '--headless', action='store_true', default=True,
        help='Headless 모드 (기본: True)')
    parser.add_argument(
        '--gui', action='store_true',
        help='GUI 모드 (Gazebo 창 표시)')
    parser.add_argument(
        '--output-dir', type=str, default='',
        help='출력 디렉토리 (기본: config에서 읽음)')

    args = parser.parse_args()

    # 설정 파일 로드
    config_path = Path(__file__).parent.parent / args.config
    if not config_path.exists():
        # 절대 경로 시도
        config_path = Path(args.config)
    if not config_path.exists():
        print(f'[ERROR] Config not found: {args.config}')
        sys.exit(1)

    with open(config_path) as f:
        config = yaml.safe_load(f)

    global_cfg = config.get('global', {})
    scenarios = config.get('scenarios', {})
    groups = config.get('controller_groups', {})

    # 시나리오 선택
    if args.scenario not in scenarios:
        print(f'[ERROR] Unknown scenario: {args.scenario}')
        print(f'  Available: {list(scenarios.keys())}')
        sys.exit(1)
    scenario_cfg = scenarios[args.scenario]

    # 컨트롤러 선택
    if args.controllers:
        controllers = [c.strip() for c in args.controllers.split(',')]
    elif args.group in groups:
        controllers = groups[args.group]
    else:
        print(f'[ERROR] Unknown group: {args.group}')
        print(f'  Available: {list(groups.keys())}')
        sys.exit(1)

    # CLI 오버라이드
    if args.trials > 0:
        global_cfg['num_trials'] = args.trials
    if args.warmup >= 0:
        global_cfg['warmup_trials'] = args.warmup
    if args.gui:
        args.headless = False

    # 출력 디렉토리
    if args.output_dir:
        output_dir = Path(args.output_dir).expanduser()
    else:
        base = Path(global_cfg.get('output_dir',
                                   '~/paper_benchmark_results')).expanduser()
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = base / f'bench_{args.scenario}_{timestamp}'

    # 메타데이터
    metadata = collect_system_metadata()
    metadata['scenario'] = args.scenario
    metadata['controller_group'] = args.group

    print(f'\n  Paper Benchmark')
    print(f'  Git: {metadata.get("git_hash", "?")} '
          f'({"dirty" if metadata.get("git_dirty") else "clean"})')
    print(f'  CPU: {metadata.get("cpu_model", "?")}')
    print(f'  ROS: {metadata.get("ros_distro", "?")}')
    print(f'  Output: {output_dir}')

    # 실행
    results = run_benchmark(
        controllers=controllers,
        scenario_name=args.scenario,
        scenario_cfg=scenario_cfg,
        global_cfg=global_cfg,
        output_dir=output_dir,
        headless=args.headless,
    )

    # 저장
    run = BenchmarkRun(
        metadata=metadata,
        config={
            'scenario': args.scenario,
            'scenario_config': scenario_cfg,
            'controllers': controllers,
            'global': global_cfg,
        },
        trials=[asdict(r) for r in results],
    )
    save_benchmark(run, output_dir)

    # 분석 안내
    print(f'\n  분석 실행:')
    print(f'    python3 scripts/paper_benchmark_analysis.py '
          f'--input {output_dir / "aggregated_stats.json"}')


if __name__ == '__main__':
    main()
