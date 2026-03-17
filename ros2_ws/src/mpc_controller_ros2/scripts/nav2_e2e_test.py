#!/usr/bin/env python3
"""
nav2 E2E 네비게이션 테스트 (DiffDrive, headless 지원)

Gazebo + nav2 환경에서 순차 goal을 전송하고, 도달 여부/충돌/시간을 검증.
CI headless 환경에서 사용 가능.

사용법:
    # 기본 (custom MPPI, mppi_test_simple.world)
    ros2 run mpc_controller_ros2 nav2_e2e_test.py

    # Tube-MPPI + maze 환경
    ros2 run mpc_controller_ros2 nav2_e2e_test.py --controller tube_mppi --world maze

    # 타임아웃/goals 지정
    ros2 run mpc_controller_ros2 nav2_e2e_test.py --goals "(2,0);(-2,2)" --timeout 120

    # CI exit code (0=pass, 1=fail)
    ros2 run mpc_controller_ros2 nav2_e2e_test.py --ci

전제 조건:
    ros2 launch mpc_controller_ros2 mppi_ros2_control_nav2.launch.py \\
        controller:=<controller> headless:=true
"""

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from nav2_msgs.action import NavigateToPose
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
import argparse
import math
import time
import json
import os
import subprocess
import sys
from dataclasses import dataclass, field, asdict
from typing import List, Tuple, Optional


# ============================================================
# 데이터 구조
# ============================================================

COLLISION_THRESHOLD = 0.15  # 충돌 판정 거리 (m)

WORLD_GOALS = {
    'simple': [(2.0, 0.0), (-2.0, 2.0), (0.0, -2.0)],
    'maze': [(2.0, 0.0), (-5.0, 5.0), (7.0, -7.0)],
    'corridor': [(5.0, 0.0), (-5.0, 5.0)],
    'narrow_passage': [(5.0, 0.0), (0.0, -4.0), (7.0, -7.0)],
    'random_forest': [(3.0, 3.0), (-3.0, -3.0), (7.0, -7.0)],
    'dynamic_obstacles': [(-7.0, 0.0), (0.0, -7.0), (-7.0, 6.0)],
}

PASS_CRITERIA = {
    'goals_reached': 'all goals reached',
    'no_collision': 'collision_count == 0',
    'min_distance_safe': 'min_obstacle_distance >= 0.15m',
    'within_timeout': 'completed within timeout',
}


@dataclass
class E2EMetrics:
    """E2E 테스트 종합 메트릭"""
    controller: str = ''
    world: str = ''

    goals_total: int = 0
    goals_reached: int = 0
    goal_results: List[bool] = field(default_factory=list)
    goal_times: List[float] = field(default_factory=list)

    total_time: float = 0.0
    travel_distance: float = 0.0

    collision_count: int = 0
    min_obstacle_distance: float = float('inf')

    timed_out: bool = False

    pass_result: dict = field(default_factory=dict)
    overall_pass: bool = False


# ============================================================
# E2E 테스트 노드
# ============================================================

class Nav2E2ETestNode(Node):
    def __init__(self, args):
        super().__init__('nav2_e2e_test')
        self.args = args

        # Odom 상태
        self.odom_x = 0.0
        self.odom_y = 0.0
        self.odom_received = False
        self.positions: List[Tuple[float, float]] = []

        # Scan 상태
        self.min_scan = float('inf')
        self.min_scan_ever = float('inf')
        self.collision_count = 0

        # 구독
        odom_topic = '/diff_drive_controller/odom'
        self.odom_sub = self.create_subscription(
            Odometry, odom_topic, self._odom_cb, 10)
        self.scan_sub = self.create_subscription(
            LaserScan, '/scan', self._scan_cb, 10)

        # nav2 action client
        self._action_client = ActionClient(
            self, NavigateToPose, 'navigate_to_pose')

        # Goal 설정
        if args.goals:
            self.goals = self._parse_goals(args.goals)
        else:
            self.goals = WORLD_GOALS.get(args.world, WORLD_GOALS['simple'])

        self.get_logger().info(
            f'E2E Test: controller={args.controller}, world={args.world}, '
            f'goals={len(self.goals)}, timeout={args.timeout}s')

    def _odom_cb(self, msg):
        self.odom_x = msg.pose.pose.position.x
        self.odom_y = msg.pose.pose.position.y
        self.odom_received = True
        self.positions.append((self.odom_x, self.odom_y))

    def _scan_cb(self, msg):
        valid = [r for r in msg.ranges if msg.range_min < r < msg.range_max]
        if valid:
            self.min_scan = min(valid)
            if self.min_scan < self.min_scan_ever:
                self.min_scan_ever = self.min_scan
            if self.min_scan < COLLISION_THRESHOLD:
                self.collision_count += 1

    @staticmethod
    def _parse_goals(goals_str: str) -> List[Tuple[float, float]]:
        goals = []
        for part in goals_str.split(';'):
            part = part.strip().strip('()')
            if not part:
                continue
            xy = part.split(',')
            if len(xy) == 2:
                goals.append((float(xy[0].strip()), float(xy[1].strip())))
        return goals

    def _wait_nav2_stable(self, timeout_sec: float = 90.0) -> bool:
        """nav2 lifecycle 안정화 대기"""
        nodes = ['controller_server', 'planner_server', 'bt_navigator']
        start = time.time()
        stable_count = 0

        self.get_logger().info('nav2 lifecycle 안정화 대기...')
        while time.time() - start < timeout_sec:
            all_active = True
            for node_name in nodes:
                try:
                    result = subprocess.run(
                        ['ros2', 'lifecycle', 'get', f'/{node_name}'],
                        capture_output=True, text=True, timeout=5.0)
                    if 'active [3]' not in result.stdout:
                        all_active = False
                        break
                except Exception:
                    all_active = False
                    break

            if all_active:
                stable_count += 1
                if stable_count >= 3:
                    self.get_logger().info(
                        f'nav2 안정화 완료 ({time.time() - start:.0f}s)')
                    return True
            else:
                stable_count = 0

            time.sleep(2.0)

        self.get_logger().warn('nav2 안정화 타임아웃!')
        return False

    def _send_goal(self, gx: float, gy: float, timeout: float) -> bool:
        """단일 goal 전송 → SUCCEEDED 여부 반환"""
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
            self.get_logger().warn(f'Goal ({gx:.1f}, {gy:.1f}) rejected')
            return False

        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(
            self, result_future, timeout_sec=timeout)

        if result_future.done():
            status = result_future.result().status
            return (status == 4)  # SUCCEEDED
        else:
            goal_handle.cancel_goal_async()
            time.sleep(1.0)
            return False

    def _feedback_cb(self, feedback_msg):
        fb = feedback_msg.feedback
        self.get_logger().info(
            f'  remaining: {fb.distance_remaining:.2f}m',
            throttle_duration_sec=5.0)

    def run_test(self) -> E2EMetrics:
        """전체 E2E 테스트 실행"""
        metrics = E2EMetrics(
            controller=self.args.controller,
            world=self.args.world,
            goals_total=len(self.goals),
        )

        # nav2 action server 대기
        self.get_logger().info('nav2 action server 대기...')
        if not self._action_client.wait_for_server(timeout_sec=30.0):
            self.get_logger().error('nav2 action server 없음!')
            metrics.timed_out = True
            return metrics

        # lifecycle 안정화
        if not self._wait_nav2_stable():
            self.get_logger().error('nav2 안정화 실패')
            metrics.timed_out = True
            return metrics

        test_start = time.time()
        per_goal_timeout = max(self.args.timeout / max(len(self.goals), 1), 60.0)

        for i, (gx, gy) in enumerate(self.goals):
            self.get_logger().info(
                f'Goal {i+1}/{len(self.goals)}: ({gx:.1f}, {gy:.1f})')

            goal_start = time.time()
            success = self._send_goal(gx, gy, per_goal_timeout)

            if not success:
                self.get_logger().warn(f'Goal {i+1} 실패 — 재시도')
                self._wait_nav2_stable(timeout_sec=30.0)
                time.sleep(2.0)
                goal_start = time.time()
                success = self._send_goal(gx, gy, per_goal_timeout)

            goal_time = time.time() - goal_start
            metrics.goal_results.append(success)
            metrics.goal_times.append(goal_time)

            status = 'REACHED' if success else 'FAILED'
            self.get_logger().info(
                f'Goal {i+1}: {status} ({goal_time:.1f}s)')

            # 전체 타임아웃 체크
            if time.time() - test_start > self.args.timeout:
                self.get_logger().warn('전체 타임아웃!')
                metrics.timed_out = True
                break

        metrics.total_time = time.time() - test_start
        metrics.goals_reached = sum(metrics.goal_results)
        metrics.collision_count = self.collision_count
        metrics.min_obstacle_distance = self.min_scan_ever

        # 이동 거리
        dist = 0.0
        for i in range(1, len(self.positions)):
            dx = self.positions[i][0] - self.positions[i-1][0]
            dy = self.positions[i][1] - self.positions[i-1][1]
            dist += math.sqrt(dx*dx + dy*dy)
        metrics.travel_distance = dist

        return metrics

    @staticmethod
    def evaluate(m: E2EMetrics) -> E2EMetrics:
        """Pass/Fail 판정"""
        m.pass_result = {
            'goals_reached': m.goals_reached == m.goals_total,
            'no_collision': m.collision_count == 0,
            'min_distance_safe': m.min_obstacle_distance >= COLLISION_THRESHOLD,
            'within_timeout': not m.timed_out,
        }
        m.overall_pass = all(m.pass_result.values())
        return m

    @staticmethod
    def print_report(m: E2EMetrics):
        """ASCII 리포트"""
        print()
        print('=' * 60)
        print('  nav2 E2E Navigation Test Report')
        print('=' * 60)
        print(f'  Controller  : {m.controller}')
        print(f'  World       : {m.world}')
        print(f'  Goals       : {m.goals_reached}/{m.goals_total} reached')
        for i, (ok, t) in enumerate(zip(m.goal_results, m.goal_times)):
            status = 'OK' if ok else 'FAIL'
            print(f'    Goal {i+1}: [{status}] {t:.1f}s')
        print(f'  Total Time  : {m.total_time:.1f}s')
        print(f'  Distance    : {m.travel_distance:.1f}m')
        print(f'  Collisions  : {m.collision_count}')
        print(f'  Min Dist    : {m.min_obstacle_distance:.2f}m')
        print('-' * 60)
        print('  Pass/Fail:')
        for criterion, passed in m.pass_result.items():
            status = 'PASS' if passed else 'FAIL'
            desc = PASS_CRITERIA.get(criterion, '')
            print(f'    [{status}] {criterion:24s} {desc}')
        print('-' * 60)
        overall = 'PASS' if m.overall_pass else 'FAIL'
        print(f'  OVERALL     : {overall}')
        print('=' * 60)
        print()

    @staticmethod
    def save_json(m: E2EMetrics):
        """JSON 결과 저장"""
        output_dir = os.path.expanduser('~/e2e_test_results')
        os.makedirs(output_dir, exist_ok=True)
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        filepath = os.path.join(
            output_dir,
            f'e2e_{m.controller}_{m.world}_{timestamp}.json')
        with open(filepath, 'w') as f:
            json.dump(asdict(m), f, indent=2)
        return filepath


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description='nav2 E2E 네비게이션 테스트 (headless 지원)')
    parser.add_argument(
        '--controller', type=str, default='custom',
        help='컨트롤러 이름 (리포트용)')
    parser.add_argument(
        '--world', type=str, default='simple',
        choices=list(WORLD_GOALS.keys()),
        help='월드 프리셋 (goal 자동 설정)')
    parser.add_argument(
        '--goals', type=str, default='',
        help='커스텀 goal: "(x1,y1);(x2,y2);..."')
    parser.add_argument(
        '--timeout', type=float, default=180.0,
        help='전체 타임아웃 (s)')
    parser.add_argument(
        '--ci', action='store_true',
        help='CI 모드: exit code 0=pass, 1=fail')
    parser.add_argument(
        '--no-save', action='store_true',
        help='JSON 저장 건너뛰기')

    args = parser.parse_args()

    rclpy.init()
    node = Nav2E2ETestNode(args)

    try:
        metrics = node.run_test()
        metrics = Nav2E2ETestNode.evaluate(metrics)
        Nav2E2ETestNode.print_report(metrics)

        if not args.no_save:
            filepath = Nav2E2ETestNode.save_json(metrics)
            node.get_logger().info(f'결과 저장: {filepath}')

        if args.ci:
            sys.exit(0 if metrics.overall_pass else 1)

    except KeyboardInterrupt:
        node.get_logger().info('사용자 중단')

    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
