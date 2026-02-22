#!/usr/bin/env python3
"""
Swerve E2E 시뮬레이션 검증 스크립트

nav2 goal 전송 후 cmd_vel / odom 데이터를 수집하여
궤적 안정화(SG Filter/IT/Exploration) 및 CBF 효과를 정량 분석.

사용법:
    # 기본 테스트 (5m 전진)
    ros2 run mpc_controller_ros2 swerve_e2e_test.py

    # 커스텀 목표
    ros2 run mpc_controller_ros2 swerve_e2e_test.py --x 3.0 --y 2.0 --timeout 60

    # 데이터만 수집 (goal 전송 없이)
    ros2 run mpc_controller_ros2 swerve_e2e_test.py --record-only --timeout 30
"""

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from nav2_msgs.action import NavigateToPose
from geometry_msgs.msg import PoseStamped, Twist, TwistStamped
from nav_msgs.msg import Odometry
import argparse
import math
import time
import json
import os
from dataclasses import dataclass, field, asdict
from typing import List


@dataclass
class SamplePoint:
    t: float       # 시간 (s)
    vx: float      # 선속도 x (m/s)
    vy: float      # 선속도 y (m/s)
    omega: float   # 각속도 (rad/s)
    x: float       # odom x (m)
    y: float       # odom y (m)
    theta: float   # odom yaw (rad)


@dataclass
class E2EMetrics:
    """E2E 검증 메트릭"""
    goal_reached: bool = False
    travel_time: float = 0.0           # 도달 시간 (s)
    travel_distance: float = 0.0       # 주행 거리 (m)
    final_position_error: float = 0.0  # 최종 위치 오차 (m)
    final_yaw_error: float = 0.0       # 최종 yaw 오차 (rad)

    # 속도 통계
    mean_speed: float = 0.0            # 평균 속도 (m/s)
    max_speed: float = 0.0             # 최대 속도 (m/s)

    # 제어 부드러움 (핵심 비교 지표)
    mean_jerk_vx: float = 0.0          # vx jerk 평균 (m/s³)
    mean_jerk_vy: float = 0.0          # vy jerk 평균 (m/s³)
    mean_jerk_omega: float = 0.0       # omega jerk 평균 (rad/s³)
    rms_jerk: float = 0.0             # 전체 jerk RMS

    # 정체/멈칫거림 감지
    num_stalls: int = 0                # 속도 < 0.01m/s 구간 수
    stall_ratio: float = 0.0           # 정체 비율 (0~1)

    num_samples: int = 0


class SwerveE2ETest(Node):
    def __init__(self, args):
        super().__init__('swerve_e2e_test')
        self.args = args

        # 데이터 수집
        self.samples: List[SamplePoint] = []
        self.start_time = None
        self.last_odom_x = 0.0
        self.last_odom_y = 0.0
        self.last_odom_theta = 0.0
        self.odom_received = False
        self.goal_x = args.x
        self.goal_y = args.y
        self.goal_yaw = args.yaw

        # cmd_vel 구독 (nav2 → controller)
        self.cmd_vel_sub = self.create_subscription(
            TwistStamped, '/cmd_vel', self.cmd_vel_callback, 10)

        # Twist (unstamped) 폴백
        self.cmd_vel_unstamped_sub = self.create_subscription(
            Twist, '/cmd_vel_nav', self.cmd_vel_unstamped_callback, 10)

        # odom 구독
        self.odom_sub = self.create_subscription(
            Odometry, '/odom', self.odom_callback, 10)

        # Gazebo odom 폴백
        self.gz_odom_sub = self.create_subscription(
            Odometry, '/model/swerve_robot/odometry', self.odom_callback, 10)

        # nav2 action client
        if not args.record_only:
            self._action_client = ActionClient(
                self, NavigateToPose, 'navigate_to_pose')

        self.get_logger().info(
            f'Swerve E2E Test 초기화 (goal: x={args.x}, y={args.y}, timeout={args.timeout}s)')

    def odom_callback(self, msg):
        self.last_odom_x = msg.pose.pose.position.x
        self.last_odom_y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        self.last_odom_theta = math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        self.odom_received = True

    def cmd_vel_callback(self, msg):
        self._record_sample(msg.twist)

    def cmd_vel_unstamped_callback(self, msg):
        self._record_sample(msg)

    def _record_sample(self, twist):
        if self.start_time is None:
            self.start_time = time.time()

        t = time.time() - self.start_time
        self.samples.append(SamplePoint(
            t=t,
            vx=twist.linear.x,
            vy=twist.linear.y,
            omega=twist.angular.z,
            x=self.last_odom_x,
            y=self.last_odom_y,
            theta=self.last_odom_theta
        ))

    def send_goal_and_wait(self):
        """nav2 goal 전송 후 완료 대기"""
        self.get_logger().info('nav2 action server 대기 중...')
        if not self._action_client.wait_for_server(timeout_sec=30.0):
            self.get_logger().error('nav2 action server를 찾을 수 없습니다!')
            return False

        goal_msg = NavigateToPose.Goal()
        goal_msg.pose = PoseStamped()
        goal_msg.pose.header.frame_id = 'map'
        goal_msg.pose.header.stamp = self.get_clock().now().to_msg()
        goal_msg.pose.pose.position.x = self.goal_x
        goal_msg.pose.pose.position.y = self.goal_y
        goal_msg.pose.pose.position.z = 0.0
        goal_msg.pose.pose.orientation.z = math.sin(self.goal_yaw / 2.0)
        goal_msg.pose.pose.orientation.w = math.cos(self.goal_yaw / 2.0)

        self.get_logger().info(
            f'Goal 전송: x={self.goal_x:.2f}, y={self.goal_y:.2f}, yaw={self.goal_yaw:.2f}')

        send_future = self._action_client.send_goal_async(
            goal_msg, feedback_callback=self._feedback_cb)
        rclpy.spin_until_future_complete(self, send_future, timeout_sec=10.0)

        goal_handle = send_future.result()
        if goal_handle is None or not goal_handle.accepted:
            self.get_logger().error('Goal 거부됨!')
            return False

        self.get_logger().info('Goal 수락됨. 주행 중...')
        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(
            self, result_future, timeout_sec=self.args.timeout)

        if result_future.done():
            status = result_future.result().status
            return status == 4  # SUCCEEDED
        else:
            self.get_logger().warn(f'타임아웃 ({self.args.timeout}s)')
            return False

    def _feedback_cb(self, feedback_msg):
        fb = feedback_msg.feedback
        self.get_logger().info(
            f'남은 거리: {fb.distance_remaining:.2f}m',
            throttle_duration_sec=3.0)

    def record_only(self):
        """goal 없이 데이터만 수집"""
        self.get_logger().info(f'{self.args.timeout}초간 데이터 수집...')
        end_time = time.time() + self.args.timeout
        while time.time() < end_time:
            rclpy.spin_once(self, timeout_sec=0.1)

    def compute_metrics(self) -> E2EMetrics:
        """수집된 데이터로 메트릭 계산"""
        m = E2EMetrics()
        n = len(self.samples)
        m.num_samples = n

        if n < 2:
            return m

        # 도달 시간
        m.travel_time = self.samples[-1].t - self.samples[0].t

        # 주행 거리 (누적)
        dist = 0.0
        for i in range(1, n):
            dx = self.samples[i].x - self.samples[i - 1].x
            dy = self.samples[i].y - self.samples[i - 1].y
            dist += math.sqrt(dx * dx + dy * dy)
        m.travel_distance = dist

        # 최종 위치/yaw 오차
        m.final_position_error = math.sqrt(
            (self.last_odom_x - self.goal_x) ** 2 +
            (self.last_odom_y - self.goal_y) ** 2)
        yaw_diff = self.last_odom_theta - self.goal_yaw
        m.final_yaw_error = abs(math.atan2(math.sin(yaw_diff), math.cos(yaw_diff)))

        # 속도 통계
        speeds = [math.sqrt(s.vx**2 + s.vy**2) for s in self.samples]
        m.mean_speed = sum(speeds) / n
        m.max_speed = max(speeds)

        # Jerk 계산 (유한 차분)
        jerks_vx, jerks_vy, jerks_omega = [], [], []
        for i in range(2, n):
            dt1 = self.samples[i].t - self.samples[i - 1].t
            dt2 = self.samples[i - 1].t - self.samples[i - 2].t
            if dt1 < 1e-6 or dt2 < 1e-6:
                continue

            acc_vx_1 = (self.samples[i].vx - self.samples[i - 1].vx) / dt1
            acc_vx_0 = (self.samples[i - 1].vx - self.samples[i - 2].vx) / dt2
            dt_avg = (dt1 + dt2) / 2.0
            jerk_vx = (acc_vx_1 - acc_vx_0) / dt_avg

            acc_vy_1 = (self.samples[i].vy - self.samples[i - 1].vy) / dt1
            acc_vy_0 = (self.samples[i - 1].vy - self.samples[i - 2].vy) / dt2
            jerk_vy = (acc_vy_1 - acc_vy_0) / dt_avg

            acc_w_1 = (self.samples[i].omega - self.samples[i - 1].omega) / dt1
            acc_w_0 = (self.samples[i - 1].omega - self.samples[i - 2].omega) / dt2
            jerk_w = (acc_w_1 - acc_w_0) / dt_avg

            jerks_vx.append(abs(jerk_vx))
            jerks_vy.append(abs(jerk_vy))
            jerks_omega.append(abs(jerk_w))

        if jerks_vx:
            m.mean_jerk_vx = sum(jerks_vx) / len(jerks_vx)
            m.mean_jerk_vy = sum(jerks_vy) / len(jerks_vy)
            m.mean_jerk_omega = sum(jerks_omega) / len(jerks_omega)
            rms_sq = sum(j**2 for j in jerks_vx) + sum(j**2 for j in jerks_vy)
            m.rms_jerk = math.sqrt(rms_sq / (len(jerks_vx) + len(jerks_vy)))

        # 정체(멈칫거림) 감지: speed < 0.01 구간
        stall_count = sum(1 for s in speeds if s < 0.01)
        m.num_stalls = stall_count
        m.stall_ratio = stall_count / n if n > 0 else 0.0

        return m

    def print_report(self, metrics: E2EMetrics):
        """ASCII 리포트 출력"""
        print()
        print('=' * 60)
        print('  Swerve E2E Test Report')
        print('=' * 60)
        print(f'  Goal reached     : {"YES" if metrics.goal_reached else "NO"}')
        print(f'  Samples          : {metrics.num_samples}')
        print(f'  Travel time      : {metrics.travel_time:.1f} s')
        print(f'  Travel distance  : {metrics.travel_distance:.2f} m')
        print(f'  Position error   : {metrics.final_position_error:.3f} m')
        print(f'  Yaw error        : {math.degrees(metrics.final_yaw_error):.1f} deg')
        print('-' * 60)
        print('  Speed')
        print(f'    Mean           : {metrics.mean_speed:.3f} m/s')
        print(f'    Max            : {metrics.max_speed:.3f} m/s')
        print('-' * 60)
        print('  Smoothness (lower = better)')
        print(f'    Jerk vx (mean) : {metrics.mean_jerk_vx:.3f} m/s^3')
        print(f'    Jerk vy (mean) : {metrics.mean_jerk_vy:.3f} m/s^3')
        print(f'    Jerk omega     : {metrics.mean_jerk_omega:.3f} rad/s^3')
        print(f'    Jerk RMS       : {metrics.rms_jerk:.3f}')
        print('-' * 60)
        print('  Stall detection')
        print(f'    Stall samples  : {metrics.num_stalls} / {metrics.num_samples}')
        print(f'    Stall ratio    : {metrics.stall_ratio:.1%}')
        print('=' * 60)
        print()

    def save_data(self, metrics: E2EMetrics):
        """데이터를 JSON으로 저장"""
        output_dir = os.path.expanduser('~/swerve_e2e_results')
        os.makedirs(output_dir, exist_ok=True)

        timestamp = time.strftime('%Y%m%d_%H%M%S')
        filepath = os.path.join(output_dir, f'e2e_{timestamp}.json')

        data = {
            'metrics': asdict(metrics),
            'config': {
                'goal_x': self.goal_x,
                'goal_y': self.goal_y,
                'goal_yaw': self.goal_yaw,
                'timeout': self.args.timeout,
            },
            'samples': [asdict(s) for s in self.samples]
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

        self.get_logger().info(f'데이터 저장: {filepath}')
        return filepath


def main():
    parser = argparse.ArgumentParser(description='Swerve E2E 검증')
    parser.add_argument('--x', type=float, default=3.0, help='목표 x (m)')
    parser.add_argument('--y', type=float, default=0.0, help='목표 y (m)')
    parser.add_argument('--yaw', type=float, default=0.0, help='목표 yaw (rad)')
    parser.add_argument('--timeout', type=float, default=60.0, help='타임아웃 (s)')
    parser.add_argument('--record-only', action='store_true',
                        help='Goal 없이 데이터만 수집')
    parser.add_argument('--no-save', action='store_true',
                        help='JSON 저장 건너뛰기')
    args = parser.parse_args()

    rclpy.init()
    node = SwerveE2ETest(args)

    try:
        if args.record_only:
            node.record_only()
            goal_reached = False
        else:
            goal_reached = node.send_goal_and_wait()

        metrics = node.compute_metrics()
        metrics.goal_reached = goal_reached
        node.print_report(metrics)

        if not args.no_save:
            node.save_data(metrics)

    except KeyboardInterrupt:
        node.get_logger().info('사용자 중단')
        metrics = node.compute_metrics()
        node.print_report(metrics)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
