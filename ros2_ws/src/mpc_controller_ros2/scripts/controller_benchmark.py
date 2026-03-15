#!/usr/bin/env python3
"""
controller_benchmark.py
=======================
20종 MPPI 컨트롤러 자동 벤치마크 오케스트레이터

diff_drive / swerve / ackermann 모든 모션 모델을 지원하며,
각 컨트롤러별로 launch → E2E 테스트 → 메트릭 수집을 순차 실행합니다.

사용법:
    # diff_drive 컨트롤러 전체 벤치마크
    python3 controller_benchmark.py --group diff_drive

    # swerve 컨트롤러 전체 벤치마크
    python3 controller_benchmark.py --group swerve

    # 특정 컨트롤러만 벤치마크
    python3 controller_benchmark.py --controllers custom log shield

    # 커스텀 goal + world
    python3 controller_benchmark.py --group swerve \
        --world narrow_passage_world.world --map narrow_passage_map.yaml \
        --goal-x 5.0 --goal-y -3.5

    # 이전 결과에서 리포트만 생성
    python3 controller_benchmark.py --report-only ~/benchmark_results/bench_20260316_*.json
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

# ─── 컨트롤러 그룹 정의 ─────────────────────────────────────────────

DIFF_DRIVE_CONTROLLERS = [
    'custom', 'log', 'tsallis', 'risk_aware', 'svmpc',
    'smooth', 'spline', 'svg', 'biased', 'dial',
    'shield', 'adaptive_shield', 'clf_cbf', 'predictive_safety',
    'ilqr_mppi', 'cs_mppi', 'pi_mppi',
]

SWERVE_CONTROLLERS = [
    'swerve', 'log_swerve', 'tsallis_swerve', 'smooth_swerve',
    'biased_swerve', 'shield_swerve', 'cs_swerve', 'pi_swerve',
    'svg_swerve', 'ilqr_swerve', 'dial_swerve',
    'hybrid_swerve', 'non_coaxial',
]

ACKERMANN_CONTROLLERS = [
    'ackermann',
]

ALL_CONTROLLERS = DIFF_DRIVE_CONTROLLERS + SWERVE_CONTROLLERS + ACKERMANN_CONTROLLERS

GROUP_MAP = {
    'diff_drive': DIFF_DRIVE_CONTROLLERS,
    'swerve': SWERVE_CONTROLLERS,
    'ackermann': ACKERMANN_CONTROLLERS,
    'all': ALL_CONTROLLERS,
    'safety': ['shield', 'adaptive_shield', 'clf_cbf', 'predictive_safety', 'shield_swerve'],
    'quick': ['custom', 'log', 'smooth', 'shield'],
}

# ─── 기본 설정 ────────────────────────────────────────────────────

DEFAULT_WORLD = 'maze_world.world'
DEFAULT_MAP = 'maze_map.yaml'
DEFAULT_GOAL_X = 3.0
DEFAULT_GOAL_Y = 0.0
DEFAULT_GOAL_YAW = 0.0
DEFAULT_TIMEOUT = 90
LAUNCH_STABILIZE_SEC = 35
NAV2_ACTIVATE_SEC = 30
NAV2_CHECK_RETRIES = 10
NAV2_CHECK_INTERVAL = 5


@dataclass
class BenchmarkResult:
    """단일 컨트롤러 벤치마크 결과"""
    controller: str
    status: str  # 'SUCCESS', 'LAUNCH_FAILED', 'NAV2_TIMEOUT', 'TEST_FAILED'
    metrics: Optional[Dict] = None
    error_message: str = ''
    duration_sec: float = 0.0
    timestamp: str = ''


@dataclass
class BenchmarkSuite:
    """벤치마크 스위트 전체 결과"""
    suite_name: str
    world: str
    map_name: str
    goal: Dict
    start_time: str
    end_time: str = ''
    results: List[Dict] = field(default_factory=list)
    total_controllers: int = 0
    successful: int = 0
    failed: int = 0


def cleanup_processes():
    """이전 launch 프로세스 정리"""
    cleanup_script = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        'cleanup_launch.sh'
    )
    if os.path.exists(cleanup_script):
        subprocess.run(['bash', cleanup_script], capture_output=True, timeout=15)
    else:
        # 직접 정리
        patterns = [
            'gz sim', 'parameter_bridge', 'twist_stamper',
            'swerve_kinematics_node', 'odom_to_tf',
            'nav2_lifecycle_bringup', 'robot_state_publisher',
            'map_server --ros', 'amcl --ros',
            'controller_server --ros', 'planner_server --ros',
            'behavior_server --ros', 'bt_navigator --ros',
            'rviz2',
        ]
        for pat in patterns:
            subprocess.run(
                ['pkill', '-9', '-f', pat],
                capture_output=True
            )
    time.sleep(3)


def wait_for_nav2_active(timeout_sec: int = NAV2_ACTIVATE_SEC,
                         retries: int = NAV2_CHECK_RETRIES) -> bool:
    """nav2 lifecycle 활성화 대기"""
    time.sleep(timeout_sec)

    for i in range(retries):
        try:
            bt_result = subprocess.run(
                ['ros2', 'lifecycle', 'get', '/bt_navigator'],
                capture_output=True, text=True, timeout=10
            )
            ctrl_result = subprocess.run(
                ['ros2', 'lifecycle', 'get', '/controller_server'],
                capture_output=True, text=True, timeout=10
            )
            bt_active = 'active' in bt_result.stdout.lower()
            ctrl_active = 'active' in ctrl_result.stdout.lower()

            if bt_active and ctrl_active:
                return True
        except (subprocess.TimeoutExpired, Exception):
            pass

        time.sleep(NAV2_CHECK_INTERVAL)

    return False


def run_e2e_test(goal_x: float, goal_y: float, goal_yaw: float,
                 timeout: float) -> Optional[Dict]:
    """E2E 테스트 실행 및 결과 파싱"""
    cmd = [
        'ros2', 'run', 'mpc_controller_ros2', 'swerve_e2e_test.py',
        '--x', str(goal_x),
        '--y', str(goal_y),
        '--yaw', str(goal_yaw),
        '--timeout', str(timeout),
        '--no-save',
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True, text=True,
            timeout=int(timeout) + 30
        )
        output = result.stdout + result.stderr

        # 메트릭 파싱
        metrics = parse_e2e_output(output)
        return metrics

    except subprocess.TimeoutExpired:
        return None
    except Exception as e:
        print(f'  [ERROR] E2E test exception: {e}')
        return None


def parse_e2e_output(output: str) -> Dict:
    """E2E 테스트 출력에서 메트릭 추출"""
    metrics = {}
    lines = output.split('\n')

    parse_map = {
        'Goal reached': ('goal_reached', lambda v: v.strip().upper() == 'YES'),
        'Travel time': ('travel_time', lambda v: float(v.split()[0])),
        'Travel distance': ('travel_distance', lambda v: float(v.split()[0])),
        'Position error': ('position_error', lambda v: float(v.split()[0])),
        'Yaw error': ('yaw_error', lambda v: float(v.split()[0])),
        'Mean speed': ('mean_speed', lambda v: float(v.split()[0])),
        'Max speed': ('max_speed', lambda v: float(v.split()[0])),
        'Mean jerk vx': ('mean_jerk_vx', lambda v: float(v.split()[0])),
        'Mean jerk vy': ('mean_jerk_vy', lambda v: float(v.split()[0])),
        'Mean jerk omega': ('mean_jerk_omega', lambda v: float(v.split()[0])),
        'RMS jerk': ('rms_jerk', lambda v: float(v.split()[0])),
        'Stall ratio': ('stall_ratio', lambda v: float(v.strip().rstrip('%')) / 100.0),
        'Stall samples': ('stall_samples', lambda v: int(v.split('/')[0].strip())),
        'Min distance': ('min_obstacle_dist', lambda v: float(v.split()[0])),
        'Mean min dist': ('mean_obstacle_dist', lambda v: float(v.split()[0])),
        'Near misses': ('near_misses', lambda v: int(v.split()[0])),
        'Collisions': ('collisions', lambda v: int(v.split()[0])),
    }

    for line in lines:
        for key, (metric_name, parser) in parse_map.items():
            if key in line and ':' in line:
                try:
                    value_str = line.split(':', 1)[1].strip()
                    metrics[metric_name] = parser(value_str)
                except (ValueError, IndexError):
                    pass

    return metrics


def run_single_benchmark(controller: str, args, idx: int, total: int) -> BenchmarkResult:
    """단일 컨트롤러 벤치마크 실행"""
    result = BenchmarkResult(
        controller=controller,
        status='UNKNOWN',
        timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    )

    start = time.time()

    print()
    print('━' * 64)
    print(f'  [{idx}/{total}] {controller}')
    print('━' * 64)

    # 1. 정리
    print('  [1/5] Cleaning up previous processes...')
    cleanup_processes()

    # 2. Launch 시작
    print(f'  [2/5] Launching controller={controller} ...')
    launch_cmd = [
        'ros2', 'launch', 'mpc_controller_ros2', 'mppi_ros2_control_nav2.launch.py',
        f'controller:={controller}',
        f'world:={args.world}',
        f'map:={args.map}',
        'headless:=true',
    ]

    launch_log = f'/tmp/bench_launch_{controller}.log'
    with open(launch_log, 'w') as log_f:
        launch_proc = subprocess.Popen(
            launch_cmd,
            stdout=log_f,
            stderr=subprocess.STDOUT,
            preexec_fn=os.setsid,
        )

    print(f'  [2/5] PID={launch_proc.pid}, stabilizing ({LAUNCH_STABILIZE_SEC}s)...')
    time.sleep(LAUNCH_STABILIZE_SEC)

    # launch 생존 확인
    if launch_proc.poll() is not None:
        result.status = 'LAUNCH_FAILED'
        result.error_message = f'Launch crashed (exit code {launch_proc.returncode})'
        result.duration_sec = time.time() - start
        print(f'  [ERROR] {result.error_message}')
        cleanup_processes()
        return result

    # 3. nav2 활성화 대기
    print(f'  [3/5] Waiting for nav2 activation...')
    if not wait_for_nav2_active():
        result.status = 'NAV2_TIMEOUT'
        result.error_message = 'nav2 nodes did not reach active state'
        result.duration_sec = time.time() - start
        print(f'  [ERROR] {result.error_message}')
        try:
            os.killpg(os.getpgid(launch_proc.pid), signal.SIGTERM)
        except ProcessLookupError:
            pass
        time.sleep(3)
        cleanup_processes()
        return result

    print('  [3/5] nav2 active ✓')
    # 추가 안정화
    time.sleep(10)

    # 4. E2E 테스트
    print(f'  [4/5] Running E2E test (goal: {args.goal_x}, {args.goal_y})...')
    metrics = run_e2e_test(args.goal_x, args.goal_y, args.goal_yaw, args.timeout)

    if metrics:
        result.status = 'SUCCESS'
        result.metrics = metrics
        goal_ok = metrics.get('goal_reached', False)
        print(f'  [4/5] Test complete (goal_reached={goal_ok})')
    else:
        result.status = 'TEST_FAILED'
        result.error_message = 'E2E test returned no metrics'
        print(f'  [ERROR] {result.error_message}')

    # 5. 정리
    print('  [5/5] Shutting down...')
    try:
        os.killpg(os.getpgid(launch_proc.pid), signal.SIGTERM)
    except ProcessLookupError:
        pass
    time.sleep(3)
    cleanup_processes()

    result.duration_sec = time.time() - start
    print(f'  [{controller}] Done ({result.duration_sec:.0f}s) → {result.status}')

    return result


def print_summary_table(results: List[BenchmarkResult]):
    """벤치마크 결과 요약 테이블 출력"""
    print()
    print('=' * 100)
    print('  BENCHMARK SUMMARY')
    print('=' * 100)

    # 헤더
    header = (
        f'{"Controller":<22} {"Status":<12} '
        f'{"Goal":>4} {"Time":>7} {"Dist":>7} '
        f'{"PosErr":>7} {"Jerk":>8} '
        f'{"MinObs":>7} {"Coll":>4} '
        f'{"Speed":>6}'
    )
    print(header)
    print('-' * 100)

    success_results = []

    for r in results:
        if r.status == 'SUCCESS' and r.metrics:
            m = r.metrics
            goal = '✓' if m.get('goal_reached', False) else '✗'
            line = (
                f'{r.controller:<22} {"OK":<12} '
                f'{goal:>4} '
                f'{m.get("travel_time", 0):.1f}s'.rjust(7) + ' '
                f'{m.get("travel_distance", 0):.2f}m'.rjust(7) + ' '
                f'{m.get("position_error", 0):.3f}'.rjust(7) + ' '
                f'{m.get("rms_jerk", 0):.2f}'.rjust(8) + ' '
                f'{m.get("min_obstacle_dist", float("inf")):.2f}'.rjust(7) + ' '
                f'{m.get("collisions", 0)}'.rjust(4) + ' '
                f'{m.get("mean_speed", 0):.2f}'.rjust(6)
            )
            print(line)
            success_results.append(r)
        else:
            print(f'{r.controller:<22} {r.status:<12} {"—":>4} {"—":>7} {"—":>7} '
                  f'{"—":>7} {"—":>8} {"—":>7} {"—":>4} {"—":>6}')

    print('-' * 100)

    # 순위 (성공한 결과만)
    if len(success_results) >= 2:
        print()
        print('  RANKINGS (성공한 컨트롤러만)')
        print('-' * 60)

        # 최단 시간
        by_time = sorted(
            [r for r in success_results if r.metrics.get('goal_reached')],
            key=lambda r: r.metrics.get('travel_time', float('inf'))
        )
        if by_time:
            print(f'  ⏱  Fastest:     {by_time[0].controller} '
                  f'({by_time[0].metrics["travel_time"]:.1f}s)')

        # 최소 jerk
        by_jerk = sorted(
            success_results,
            key=lambda r: r.metrics.get('rms_jerk', float('inf'))
        )
        print(f'  🎯 Smoothest:   {by_jerk[0].controller} '
              f'(jerk={by_jerk[0].metrics.get("rms_jerk", 0):.2f})')

        # 최소 위치 오차
        by_err = sorted(
            [r for r in success_results if r.metrics.get('goal_reached')],
            key=lambda r: r.metrics.get('position_error', float('inf'))
        )
        if by_err:
            print(f'  📐 Most accurate: {by_err[0].controller} '
                  f'(err={by_err[0].metrics.get("position_error", 0):.3f}m)')

        # 안전성 (최대 min_obstacle_dist)
        by_safety = sorted(
            success_results,
            key=lambda r: r.metrics.get('min_obstacle_dist', 0),
            reverse=True
        )
        dist = by_safety[0].metrics.get('min_obstacle_dist', 0)
        if dist < float('inf'):
            print(f'  🛡  Safest:      {by_safety[0].controller} '
                  f'(min_obs={dist:.2f}m)')

    # 통계
    total = len(results)
    ok = sum(1 for r in results if r.status == 'SUCCESS')
    goals = sum(1 for r in success_results
                if r.metrics and r.metrics.get('goal_reached'))
    print()
    print(f'  Total: {total} | Success: {ok} | Goals reached: {goals} | '
          f'Failed: {total - ok}')
    print('=' * 100)


def save_results(suite: BenchmarkSuite, output_dir: str) -> str:
    """결과를 JSON + CSV로 저장"""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # JSON
    json_path = os.path.join(output_dir, f'bench_{timestamp}.json')
    with open(json_path, 'w') as f:
        json.dump(asdict(suite), f, indent=2, default=str)

    # CSV
    csv_path = os.path.join(output_dir, f'bench_{timestamp}.csv')
    metric_keys = [
        'goal_reached', 'travel_time', 'travel_distance',
        'position_error', 'yaw_error', 'mean_speed', 'max_speed',
        'mean_jerk_vx', 'mean_jerk_vy', 'mean_jerk_omega', 'rms_jerk',
        'stall_ratio', 'min_obstacle_dist', 'mean_obstacle_dist',
        'near_misses', 'collisions',
    ]

    with open(csv_path, 'w') as f:
        header = 'controller,status,' + ','.join(metric_keys)
        f.write(header + '\n')

        for r in suite.results:
            row = [r.get('controller', ''), r.get('status', '')]
            metrics = r.get('metrics') or {}
            for k in metric_keys:
                val = metrics.get(k, '')
                row.append(str(val))
            f.write(','.join(row) + '\n')

    print(f'\n  Results saved:')
    print(f'    JSON: {json_path}')
    print(f'    CSV:  {csv_path}')

    return json_path


def main():
    parser = argparse.ArgumentParser(
        description='MPPI 컨트롤러 자동 벤치마크 대시보드',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
컨트롤러 그룹:
  diff_drive  17종 (custom, log, tsallis, ...)
  swerve      13종 (swerve, log_swerve, ...)
  ackermann    1종 (ackermann)
  safety       5종 (shield, adaptive_shield, clf_cbf, predictive_safety, shield_swerve)
  quick        4종 (custom, log, smooth, shield)
  all         31종 (전체)
''')

    parser.add_argument('--group', type=str, default=None,
                        choices=list(GROUP_MAP.keys()),
                        help='컨트롤러 그룹 (기본: quick)')
    parser.add_argument('--controllers', nargs='+', default=None,
                        help='특정 컨트롤러 목록')
    parser.add_argument('--world', type=str, default=DEFAULT_WORLD,
                        help=f'Gazebo world 파일 (기본: {DEFAULT_WORLD})')
    parser.add_argument('--map', type=str, default=DEFAULT_MAP,
                        help=f'nav2 맵 파일 (기본: {DEFAULT_MAP})')
    parser.add_argument('--goal-x', type=float, default=DEFAULT_GOAL_X,
                        dest='goal_x', help='목표 x (m)')
    parser.add_argument('--goal-y', type=float, default=DEFAULT_GOAL_Y,
                        dest='goal_y', help='목표 y (m)')
    parser.add_argument('--goal-yaw', type=float, default=DEFAULT_GOAL_YAW,
                        dest='goal_yaw', help='목표 yaw (rad)')
    parser.add_argument('--timeout', type=float, default=DEFAULT_TIMEOUT,
                        help=f'E2E 테스트 타임아웃 (s, 기본: {DEFAULT_TIMEOUT})')
    parser.add_argument('--output-dir', type=str,
                        default=os.path.expanduser('~/benchmark_results'),
                        help='결과 저장 디렉토리')
    parser.add_argument('--report-only', nargs='+', default=None,
                        help='기존 JSON 파일에서 리포트만 생성')

    args = parser.parse_args()

    # 리포트 전용 모드
    if args.report_only:
        from benchmark_report import generate_report_from_files
        generate_report_from_files(args.report_only)
        return

    # 컨트롤러 목록 결정
    if args.controllers:
        controllers = args.controllers
    elif args.group:
        controllers = GROUP_MAP[args.group]
    else:
        controllers = GROUP_MAP['quick']

    # ROS2 환경 확인
    ros_distro = os.environ.get('ROS_DISTRO', '')
    if not ros_distro:
        print('[ERROR] ROS2 환경이 설정되지 않았습니다.')
        print('  source /opt/ros/jazzy/setup.bash')
        print('  source ~/toy_claude_project/ros2_ws/install/setup.bash')
        sys.exit(1)

    # 벤치마크 시작
    suite = BenchmarkSuite(
        suite_name=f'MPPI Benchmark ({len(controllers)} controllers)',
        world=args.world,
        map_name=args.map,
        goal={'x': args.goal_x, 'y': args.goal_y, 'yaw': args.goal_yaw},
        start_time=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        total_controllers=len(controllers),
    )

    print()
    print('╔' + '═' * 62 + '╗')
    print('║  MPPI Controller Benchmark Dashboard' + ' ' * 24 + '║')
    print('╠' + '═' * 62 + '╣')
    print(f'║  Controllers: {len(controllers):<47}║')
    print(f'║  World:       {args.world:<47}║')
    print(f'║  Map:         {args.map:<47}║')
    print(f'║  Goal:        ({args.goal_x}, {args.goal_y}, yaw={args.goal_yaw})'
          .ljust(63) + '║')
    print(f'║  Timeout:     {args.timeout}s'
          .ljust(63) + '║')
    print('╚' + '═' * 62 + '╝')
    print()
    print(f'  Controllers: {", ".join(controllers)}')
    print()

    results = []
    for idx, ctrl in enumerate(controllers, 1):
        try:
            result = run_single_benchmark(ctrl, args, idx, len(controllers))
            results.append(result)
        except KeyboardInterrupt:
            print('\n\n  [INTERRUPTED] Cleaning up...')
            cleanup_processes()
            break
        except Exception as e:
            print(f'\n  [ERROR] Unexpected error for {ctrl}: {e}')
            results.append(BenchmarkResult(
                controller=ctrl,
                status='ERROR',
                error_message=str(e),
                timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            ))
            cleanup_processes()

    suite.end_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    suite.results = [asdict(r) for r in results]
    suite.successful = sum(1 for r in results if r.status == 'SUCCESS')
    suite.failed = len(results) - suite.successful

    # 요약 출력
    print_summary_table(results)

    # 저장
    save_results(suite, args.output_dir)

    print()
    print('  Benchmark complete!')


if __name__ == '__main__':
    main()
