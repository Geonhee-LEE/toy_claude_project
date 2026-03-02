#!/usr/bin/env python3
"""
MPPI 컨트롤러 스트레스 테스트 (동적 장애물 + 고속 + 고주파)

nav2 순차 goal 전송 + 동적 장애물 왕복 제어 + 메트릭 수집 + pass/fail 판정.

사용법:
    # 기본 스트레스 테스트 (고속, 20Hz)
    ros2 run mpc_controller_ros2 stress_test.py --speed high --frequency 20

    # 중간 속도 테스트
    ros2 run mpc_controller_ros2 stress_test.py --speed medium --frequency 20

    # 동적 장애물 없이 정적 환경 테스트
    ros2 run mpc_controller_ros2 stress_test.py --speed high --no-dynamic-obstacles

    # DIAL-MPPI 비교
    ros2 run mpc_controller_ros2 stress_test.py --speed high --controller dial

    # 커스텀 goal 지정
    ros2 run mpc_controller_ros2 stress_test.py --goals "(7,0);(0,5);(-5,-3)"

전제 조건:
    ros2 launch mpc_controller_ros2 mppi_ros2_control_nav2.launch.py \\
        controller:=stress_test \\
        world:=dynamic_obstacles_world.world \\
        map:=stress_test_map.yaml
"""

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from nav2_msgs.action import NavigateToPose
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
import argparse
import math
import time
import json
import os
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Tuple, Optional


# ============================================================
# 데이터 구조
# ============================================================

@dataclass
class SamplePoint:
    """시간 동기화된 센서 스냅샷"""
    t: float       # 시간 (s)
    vx: float      # cmd_vel 선속도 x (m/s)
    vy: float      # cmd_vel 선속도 y (m/s)
    omega: float   # cmd_vel 각속도 (rad/s)
    x: float       # odom x (m)
    y: float       # odom y (m)
    theta: float   # odom yaw (rad)
    min_scan: float # 최소 lidar 거리 (m)


@dataclass
class StressTestMetrics:
    """스트레스 테스트 종합 메트릭"""
    # 목표 도달
    goals_total: int = 0
    goals_reached: int = 0
    goal_results: List[bool] = field(default_factory=list)

    # 이동 통계
    travel_time: float = 0.0
    travel_distance: float = 0.0

    # 충돌/안전
    collision_count: int = 0
    min_obstacle_distance: float = float('inf')
    mean_obstacle_distance: float = 0.0

    # 속도
    mean_speed: float = 0.0
    max_speed: float = 0.0

    # 제어 부드러움
    rms_jerk: float = 0.0
    mean_jerk_vx: float = 0.0

    # 경로 추종
    path_tracking_error_mean: float = 0.0

    # 계산 시간 (controller_server 간접 측정)
    computation_time_mean_ms: float = 0.0
    computation_time_max_ms: float = 0.0

    # pass/fail 판정
    pass_result: Dict[str, bool] = field(default_factory=dict)
    overall_pass: bool = False

    num_samples: int = 0


# ============================================================
# 동적 장애물 왕복 컨트롤러
# ============================================================

@dataclass
class ObstacleConfig:
    """동적 장애물 설정"""
    name: str          # Gazebo 모델 이름
    axis: str          # 'x' 또는 'y'
    speed: float       # 이동 속도 (m/s)
    amplitude: float   # 왕복 거리 (m, 편도)
    period: float      # 왕복 주기 (s)


DEFAULT_OBSTACLE_CONFIGS = [
    ObstacleConfig('dynamic_obstacle_slow', 'y', 0.3, 2.0, 13.3),
    ObstacleConfig('dynamic_obstacle_medium', 'x', 0.6, 2.0, 6.7),
    ObstacleConfig('dynamic_obstacle_fast', 'y', 1.0, 3.0, 6.0),
    ObstacleConfig('dynamic_obstacle_cross', 'x', 0.8, 2.5, 6.25),
]


class DynamicObstacleController:
    """사각파 왕복 속도 publish → ros_gz_bridge 경유 Gazebo VelocityControl"""

    def __init__(self, node: Node, configs: List[ObstacleConfig]):
        self.node = node
        self.configs = configs
        self.publishers: Dict[str, rclpy.publisher.Publisher] = {}
        self.start_time = time.time()

        for cfg in configs:
            topic = f'/model/{cfg.name}/cmd_vel'
            self.publishers[cfg.name] = node.create_publisher(Twist, topic, 10)
            node.get_logger().info(
                f'  동적 장애물: {cfg.name} ({cfg.axis}축 {cfg.speed}m/s)')

    def update(self):
        """사각파 왕복: 반주기마다 속도 방향 반전"""
        elapsed = time.time() - self.start_time

        for cfg in self.configs:
            half_period = cfg.period / 2.0
            # 사각파: 양/음 반복
            direction = 1.0 if (int(elapsed / half_period) % 2 == 0) else -1.0

            twist = Twist()
            if cfg.axis == 'x':
                twist.linear.x = cfg.speed * direction
            else:
                twist.linear.y = cfg.speed * direction

            self.publishers[cfg.name].publish(twist)

    def stop_all(self):
        """모든 장애물 정지"""
        for cfg in self.configs:
            self.publishers[cfg.name].publish(Twist())


# ============================================================
# 스트레스 테스트 노드
# ============================================================

SPEED_PRESETS = {
    'low': {'v_max': 0.5, 'label': 'low (v_max=0.5)'},
    'medium': {'v_max': 1.0, 'label': 'medium (v_max=1.0)'},
    'high': {'v_max': 1.5, 'label': 'high (v_max=1.5)'},
}

# Pass/Fail 기준
PASS_CRITERIA = {
    'goal_reached': 'all goals reached',
    'no_collision': 'collision_count == 0',
    'jerk_acceptable': 'rms_jerk < 50.0 m/s^3',
    'min_distance_safe': 'min_obstacle_distance > 0.2m',
    'computation_budget': 'max_interval < 3x period (no stalls)',
    'path_tracking': 'mean_goal_dist < 8.0m',
}

COLLISION_THRESHOLD = 0.15  # 충돌 판정 거리 (m)


class StressTestNode(Node):
    def __init__(self, args):
        super().__init__('stress_test')
        self.args = args

        # 데이터 수집
        self.samples: List[SamplePoint] = []
        self.start_time: Optional[float] = None
        self.last_odom_x = 0.0
        self.last_odom_y = 0.0
        self.last_odom_theta = 0.0
        self.odom_received = False
        self.last_min_scan = float('inf')
        self.cmd_vel_timestamps: List[float] = []

        # cmd_vel 구독 (nav2 → controller)
        self.cmd_vel_sub = self.create_subscription(
            Twist, '/cmd_vel_nav', self._cmd_vel_callback, 10)

        # odom 구독
        self.odom_sub = self.create_subscription(
            Odometry, '/odom', self._odom_callback, 10)

        # scan 구독 (최소 장애물 거리)
        self.scan_sub = self.create_subscription(
            LaserScan, '/scan', self._scan_callback, 10)

        # nav2 action client
        self._action_client = ActionClient(
            self, NavigateToPose, 'navigate_to_pose')

        # 동적 장애물 컨트롤러
        self.obstacle_controller: Optional[DynamicObstacleController] = None
        if not args.no_dynamic_obstacles:
            self.obstacle_controller = DynamicObstacleController(
                self, DEFAULT_OBSTACLE_CONFIGS)
            self.obstacle_timer = self.create_timer(0.1, self._obstacle_timer_cb)

        # goal 파싱
        self.goals = self._parse_goals(args.goals)

        speed_info = SPEED_PRESETS.get(args.speed, SPEED_PRESETS['high'])
        self.get_logger().info(
            f'스트레스 테스트 초기화\n'
            f'  Speed     : {speed_info["label"]}\n'
            f'  Frequency : {args.frequency} Hz\n'
            f'  Goals     : {len(self.goals)}개\n'
            f'  Dynamic   : {"ON" if not args.no_dynamic_obstacles else "OFF"}\n'
            f'  Controller: {args.controller}\n'
            f'  Timeout   : {args.timeout}s')

    # ---- 콜백 ----

    def _odom_callback(self, msg):
        self.last_odom_x = msg.pose.pose.position.x
        self.last_odom_y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        self.last_odom_theta = math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        self.odom_received = True

    def _cmd_vel_callback(self, msg):
        now = time.time()
        if self.start_time is None:
            self.start_time = now

        t = now - self.start_time
        self.cmd_vel_timestamps.append(now)

        self.samples.append(SamplePoint(
            t=t,
            vx=msg.linear.x,
            vy=msg.linear.y,
            omega=msg.angular.z,
            x=self.last_odom_x,
            y=self.last_odom_y,
            theta=self.last_odom_theta,
            min_scan=self.last_min_scan,
        ))

    def _scan_callback(self, msg):
        valid_ranges = [r for r in msg.ranges
                        if msg.range_min < r < msg.range_max]
        if valid_ranges:
            self.last_min_scan = min(valid_ranges)

    def _obstacle_timer_cb(self):
        if self.obstacle_controller:
            self.obstacle_controller.update()

    # ---- Goal 관리 ----

    @staticmethod
    def _parse_goals(goals_str: str) -> List[Tuple[float, float]]:
        """'(x1,y1);(x2,y2);...' → [(x1,y1), (x2,y2), ...]"""
        goals = []
        for part in goals_str.split(';'):
            part = part.strip().strip('()')
            if not part:
                continue
            xy = part.split(',')
            if len(xy) == 2:
                goals.append((float(xy[0].strip()), float(xy[1].strip())))
        return goals

    def _wait_for_nav2_stable(self, timeout_sec: float = 60.0):
        """nav2 lifecycle 안정화 대기 (TF 시간 점프 복구 후)"""
        import subprocess
        nodes = ['controller_server', 'planner_server', 'bt_navigator']
        start = time.time()
        stable_count = 0
        required_stable = 3  # 연속 3회 active 확인

        self.get_logger().info('nav2 lifecycle 안정화 대기...')
        while time.time() - start < timeout_sec:
            all_active = True
            for node in nodes:
                try:
                    result = subprocess.run(
                        ['ros2', 'lifecycle', 'get', f'/{node}'],
                        capture_output=True, text=True, timeout=5.0)
                    if 'active [3]' not in result.stdout:
                        all_active = False
                        break
                except Exception:
                    all_active = False
                    break

            if all_active:
                stable_count += 1
                if stable_count >= required_stable:
                    self.get_logger().info(
                        f'nav2 안정화 완료 ({time.time() - start:.0f}s)')
                    return True
            else:
                stable_count = 0

            time.sleep(2.0)

        self.get_logger().warn('nav2 안정화 타임아웃!')
        return False

    def send_goals_sequentially(self) -> List[bool]:
        """순차 goal 전송, 각 goal 성공/실패 반환"""
        self.get_logger().info('nav2 action server 대기 중...')
        if not self._action_client.wait_for_server(timeout_sec=30.0):
            self.get_logger().error('nav2 action server를 찾을 수 없습니다!')
            return [False] * len(self.goals)

        # nav2 lifecycle 안정화 대기 (TF 시간 점프 복구)
        self._wait_for_nav2_stable()

        results = []
        for i, (gx, gy) in enumerate(self.goals):
            self.get_logger().info(
                f'Goal {i + 1}/{len(self.goals)}: ({gx:.1f}, {gy:.1f})')

            goal_msg = NavigateToPose.Goal()
            goal_msg.pose = PoseStamped()
            goal_msg.pose.header.frame_id = 'map'
            goal_msg.pose.header.stamp = self.get_clock().now().to_msg()
            goal_msg.pose.pose.position.x = gx
            goal_msg.pose.pose.position.y = gy
            goal_msg.pose.pose.orientation.w = 1.0

            send_future = self._action_client.send_goal_async(
                goal_msg, feedback_callback=self._feedback_cb)
            rclpy.spin_until_future_complete(self, send_future, timeout_sec=10.0)

            goal_handle = send_future.result()
            if goal_handle is None or not goal_handle.accepted:
                # planner inactive 등으로 거부 → 안정화 대기 후 1회 재시도
                self.get_logger().warn(
                    f'Goal {i + 1} 거부됨 — lifecycle 복구 대기 후 재시도')
                self._wait_for_nav2_stable(timeout_sec=30.0)
                goal_msg.pose.header.stamp = self.get_clock().now().to_msg()
                send_future = self._action_client.send_goal_async(
                    goal_msg, feedback_callback=self._feedback_cb)
                rclpy.spin_until_future_complete(
                    self, send_future, timeout_sec=10.0)
                goal_handle = send_future.result()
                if goal_handle is None or not goal_handle.accepted:
                    self.get_logger().error(f'Goal {i + 1} 재시도 실패')
                    results.append(False)
                    continue

            result_future = goal_handle.get_result_async()
            # 각 goal에 충분한 시간 할당 (최소 90초)
            per_goal_timeout = max(self.args.timeout / max(len(self.goals), 1), 90.0)
            rclpy.spin_until_future_complete(
                self, result_future, timeout_sec=per_goal_timeout)

            if result_future.done():
                status = result_future.result().status
                success = (status == 4)  # SUCCEEDED
                results.append(success)
                self.get_logger().info(
                    f'Goal {i + 1}: {"REACHED" if success else "FAILED"}')
            else:
                self.get_logger().warn(
                    f'Goal {i + 1}: TIMEOUT ({per_goal_timeout:.0f}s)')
                results.append(False)
                # 타임아웃 시 cancel 후 다음 goal
                goal_handle.cancel_goal_async()
                time.sleep(1.0)

        return results

    def _feedback_cb(self, feedback_msg):
        fb = feedback_msg.feedback
        self.get_logger().info(
            f'  남은 거리: {fb.distance_remaining:.2f}m',
            throttle_duration_sec=5.0)

    # ---- 메트릭 계산 ----

    def compute_metrics(self, goal_results: List[bool]) -> StressTestMetrics:
        m = StressTestMetrics()
        n = len(self.samples)
        m.num_samples = n
        m.goals_total = len(self.goals)
        m.goals_reached = sum(goal_results)
        m.goal_results = goal_results

        if n < 2:
            return m

        # 이동 시간/거리
        m.travel_time = self.samples[-1].t - self.samples[0].t

        dist = 0.0
        for i in range(1, n):
            dx = self.samples[i].x - self.samples[i - 1].x
            dy = self.samples[i].y - self.samples[i - 1].y
            dist += math.sqrt(dx * dx + dy * dy)
        m.travel_distance = dist

        # 속도 통계
        speeds = [math.sqrt(s.vx ** 2 + s.vy ** 2) for s in self.samples]
        m.mean_speed = sum(speeds) / n
        m.max_speed = max(speeds)

        # 충돌/안전 거리
        scan_dists = [s.min_scan for s in self.samples
                      if s.min_scan < float('inf')]
        if scan_dists:
            m.min_obstacle_distance = min(scan_dists)
            m.mean_obstacle_distance = sum(scan_dists) / len(scan_dists)
            m.collision_count = sum(
                1 for d in scan_dists if d < COLLISION_THRESHOLD)

        # Jerk 계산 (3점 유한 차분, dt 이상치 필터링)
        # 예상 제어 주기의 0.2~5.0배 범위만 허용 (불규칙 간격 제외)
        expected_dt = 1.0 / max(self.args.frequency, 1)
        dt_min = expected_dt * 0.2
        dt_max = expected_dt * 5.0

        jerks_vx = []
        for i in range(2, n):
            dt1 = self.samples[i].t - self.samples[i - 1].t
            dt2 = self.samples[i - 1].t - self.samples[i - 2].t
            if dt1 < dt_min or dt1 > dt_max or dt2 < dt_min or dt2 > dt_max:
                continue
            acc1 = (self.samples[i].vx - self.samples[i - 1].vx) / dt1
            acc0 = (self.samples[i - 1].vx - self.samples[i - 2].vx) / dt2
            dt_avg = (dt1 + dt2) / 2.0
            jerks_vx.append(abs((acc1 - acc0) / dt_avg))

        if jerks_vx:
            m.mean_jerk_vx = sum(jerks_vx) / len(jerks_vx)
            m.rms_jerk = math.sqrt(sum(j ** 2 for j in jerks_vx) / len(jerks_vx))

        # 경로 추종 오차 (goal까지 직선 대비 횡방향 편차 근사)
        if self.goals:
            # 각 샘플에서 가장 가까운 goal까지 거리
            tracking_errors = []
            for s in self.samples:
                min_err = min(
                    math.sqrt((s.x - gx) ** 2 + (s.y - gy) ** 2)
                    for gx, gy in self.goals)
                tracking_errors.append(min_err)
            m.path_tracking_error_mean = sum(tracking_errors) / len(tracking_errors)

        # 계산 시간 추정 (cmd_vel 간격으로 간접 측정)
        # 리플래닝/취소 gap 제외: 예상 주기의 3배 이내만 유효
        if len(self.cmd_vel_timestamps) > 1:
            expected_ms = 1000.0 / max(self.args.frequency, 1)
            max_valid_ms = expected_ms * 3.0
            intervals_ms = []
            for i in range(1, len(self.cmd_vel_timestamps)):
                dt_ms = (self.cmd_vel_timestamps[i]
                         - self.cmd_vel_timestamps[i - 1]) * 1000.0
                if 0 < dt_ms < max_valid_ms:
                    intervals_ms.append(dt_ms)
            if intervals_ms:
                m.computation_time_mean_ms = sum(intervals_ms) / len(intervals_ms)
                m.computation_time_max_ms = max(intervals_ms)

        return m

    def evaluate_pass_fail(self, m: StressTestMetrics) -> StressTestMetrics:
        """Pass/Fail 기준 판정"""
        freq = self.args.frequency
        period_ms = 1000.0 / freq  # full period

        m.pass_result = {
            'goal_reached': m.goals_reached == m.goals_total,
            'no_collision': m.collision_count == 0,
            'jerk_acceptable': m.rms_jerk < 50.0,
            'min_distance_safe': m.min_obstacle_distance > 0.2,
            'computation_budget': m.computation_time_max_ms < period_ms * 3,
            'path_tracking': m.path_tracking_error_mean < 8.0
                             if m.path_tracking_error_mean > 0 else True,
        }
        m.overall_pass = all(m.pass_result.values())
        return m

    def print_report(self, m: StressTestMetrics):
        """ASCII 리포트 출력"""
        speed_info = SPEED_PRESETS.get(self.args.speed, SPEED_PRESETS['high'])
        freq = self.args.frequency
        period_ms = 1000.0 / freq

        print()
        print('=' * 64)
        print('  MPPI Stress Test Report')
        print('=' * 64)
        print(f'  Controller    : {self.args.controller}')
        print(f'  Speed         : {speed_info["label"]}')
        print(f'  Frequency     : {freq} Hz')
        dynamic_str = f'{len(DEFAULT_OBSTACLE_CONFIGS)} (slow/medium/fast/cross)'
        print(f'  Dynamic Obs.  : '
              f'{"OFF" if self.args.no_dynamic_obstacles else dynamic_str}')
        print('-' * 64)

        # 목표 도달
        print(f'  Goals         : {m.goals_reached}/{m.goals_total} reached'
              f'    Travel: {m.travel_distance:.1f}m / {m.travel_time:.1f}s')

        # 충돌/안전
        print(f'  Collisions    : {m.collision_count}'
              f'              Min dist: {m.min_obstacle_distance:.2f}m')

        # 속도
        print(f'  Speed         : mean={m.mean_speed:.2f}, '
              f'max={m.max_speed:.2f} m/s')

        # 부드러움
        print(f'  Jerk RMS      : {m.rms_jerk:.2f} m/s^3'
              f'     Path err: {m.path_tracking_error_mean:.2f}m (mean)')

        # 계산 시간
        budget_pct = (m.computation_time_mean_ms / (1000.0 / freq) * 100.0
                      if freq > 0 else 0)
        print(f'  Compute       : mean={m.computation_time_mean_ms:.1f}ms, '
              f'max={m.computation_time_max_ms:.1f}ms '
              f'({budget_pct:.1f}% budget)')

        print('-' * 64)
        print('  Pass/Fail Criteria:')
        for criterion, passed in m.pass_result.items():
            status = 'PASS' if passed else 'FAIL'
            desc = PASS_CRITERIA.get(criterion, '')
            print(f'    [{status}] {criterion:24s} {desc}')

        print('-' * 64)
        overall = 'PASS' if m.overall_pass else 'FAIL'
        print(f'  OVERALL       : {overall}')
        print('=' * 64)
        print()

    def save_data(self, m: StressTestMetrics):
        """JSON 결과 저장"""
        output_dir = os.path.expanduser('~/stress_test_results')
        os.makedirs(output_dir, exist_ok=True)

        timestamp = time.strftime('%Y%m%d_%H%M%S')
        filepath = os.path.join(
            output_dir,
            f'stress_{self.args.controller}_{self.args.speed}_{timestamp}.json')

        data = {
            'metrics': asdict(m),
            'config': {
                'controller': self.args.controller,
                'speed': self.args.speed,
                'frequency': self.args.frequency,
                'goals': self.goals,
                'dynamic_obstacles': not self.args.no_dynamic_obstacles,
                'timeout': self.args.timeout,
            },
            'samples': [asdict(s) for s in self.samples],
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

        self.get_logger().info(f'결과 저장: {filepath}')
        return filepath


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description='MPPI 컨트롤러 스트레스 테스트 (동적 장애물 + 고속 + 고주파)')

    parser.add_argument(
        '--speed', choices=['low', 'medium', 'high'], default='high',
        help='속도 프리셋: low(0.5), medium(1.0), high(1.5) m/s')
    parser.add_argument(
        '--frequency', type=int, choices=[10, 20, 50], default=20,
        help='컨트롤러 주파수 (Hz)')
    parser.add_argument(
        '--controller', type=str, default='custom',
        help='컨트롤러 이름 (리포트 표시용)')
    parser.add_argument(
        '--timeout', type=float, default=300.0,
        help='전체 타임아웃 (s)')
    parser.add_argument(
        '--goals', type=str, default='(-7,0);(0,-7);(-7,6)',
        help='순차 목표 좌표 "(x1,y1);(x2,y2);..."')
    parser.add_argument(
        '--no-dynamic-obstacles', action='store_true',
        help='동적 장애물 비활성화 (정적 환경만 테스트)')
    parser.add_argument(
        '--no-save', action='store_true',
        help='JSON 저장 건너뛰기')

    args = parser.parse_args()

    rclpy.init()
    node = StressTestNode(args)

    try:
        # 순차 goal 전송
        goal_results = node.send_goals_sequentially()

        # 메트릭 계산 + 판정
        metrics = node.compute_metrics(goal_results)
        metrics = node.evaluate_pass_fail(metrics)

        # 리포트 출력
        node.print_report(metrics)

        # 저장
        if not args.no_save:
            node.save_data(metrics)

    except KeyboardInterrupt:
        node.get_logger().info('사용자 중단 — 중간 결과 출력')
        metrics = node.compute_metrics([])
        metrics = node.evaluate_pass_fail(metrics)
        node.print_report(metrics)

    finally:
        # 동적 장애물 정지
        if node.obstacle_controller:
            node.obstacle_controller.stop_all()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
