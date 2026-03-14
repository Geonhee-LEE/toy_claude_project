#!/usr/bin/env python3
"""
π-MPPI E2E 시뮬레이션 검증 스크립트

nav2 goal 전송 후 cmd_vel / odom 데이터를 수집하여
π-MPPI의 ADMM 투영 필터 효과를 정량 분석.

핵심 검증 항목:
  1. Goal 도달 여부 + 위치/yaw 오차
  2. 제어 부드러움 (rate/accel/jerk)
  3. ★ rate/accel 제약 만족도 (π-MPPI 핵심)
  4. 속도 프로파일 + 정체 감지
  5. 시간 응답 분석 (오버슈트, 정착 시간)

사용법:
    # 기본 테스트 (5m 전진)
    ros2 run mpc_controller_ros2 pi_mppi_e2e_test.py

    # 커스텀 목표
    ros2 run mpc_controller_ros2 pi_mppi_e2e_test.py --x 5.0 --y 3.0 --timeout 90

    # 다중 Goal 순차 주행
    ros2 run mpc_controller_ros2 pi_mppi_e2e_test.py --goals "(5,0);(5,3);(0,3);(0,0)"

    # rate/accel bounds 커스텀 (분석용, 제어기 파라미터와 맞춤)
    ros2 run mpc_controller_ros2 pi_mppi_e2e_test.py --rate-max-v 2.0 --accel-max-v 5.0

    # 데이터만 수집 (goal 없이)
    ros2 run mpc_controller_ros2 pi_mppi_e2e_test.py --record-only --timeout 30
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
from typing import List, Tuple, Optional


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
class ConstraintMetrics:
    """π-MPPI 제약 만족도 메트릭"""
    # Rate (1차 미분) 위반 분석
    rate_max_vx: float = 0.0           # 실측 최대 |dvx/dt| (m/s²)
    rate_max_omega: float = 0.0        # 실측 최대 |domega/dt| (rad/s²)
    rate_mean_vx: float = 0.0          # 평균 |dvx/dt|
    rate_mean_omega: float = 0.0       # 평균 |domega/dt|
    rate_violation_count_vx: int = 0   # 위반 횟수 (> rate_bound)
    rate_violation_count_omega: int = 0
    rate_violation_ratio_vx: float = 0.0   # 위반 비율 (0~1)
    rate_violation_ratio_omega: float = 0.0
    rate_total_samples: int = 0

    # Accel (2차 미분 = jerk) 위반 분석
    accel_max_vx: float = 0.0         # 실측 최대 |d²vx/dt²| (m/s³)
    accel_max_omega: float = 0.0      # 실측 최대 |d²omega/dt²| (rad/s³)
    accel_mean_vx: float = 0.0
    accel_mean_omega: float = 0.0
    accel_violation_count_vx: int = 0
    accel_violation_count_omega: int = 0
    accel_violation_ratio_vx: float = 0.0
    accel_violation_ratio_omega: float = 0.0
    accel_total_samples: int = 0


@dataclass
class E2EMetrics:
    """E2E 검증 메트릭"""
    # Goal 결과
    goals_reached: int = 0
    goals_total: int = 0
    all_goals_reached: bool = False

    # 궤적 통계
    travel_time: float = 0.0           # 도달 시간 (s)
    travel_distance: float = 0.0       # 주행 거리 (m)
    final_position_error: float = 0.0  # 최종 위치 오차 (m)
    final_yaw_error: float = 0.0       # 최종 yaw 오차 (rad)

    # 속도 통계
    mean_speed: float = 0.0            # 평균 속도 (m/s)
    max_speed: float = 0.0             # 최대 속도 (m/s)

    # 제어 부드러움 (전체)
    mean_jerk_vx: float = 0.0          # vx jerk 평균 (m/s³)
    mean_jerk_vy: float = 0.0          # vy jerk 평균 (m/s³)
    mean_jerk_omega: float = 0.0       # omega jerk 평균 (rad/s³)
    rms_jerk: float = 0.0             # 전체 jerk RMS

    # ★ π-MPPI 제약 만족도
    constraints: Optional[ConstraintMetrics] = None

    # 정체 감지
    num_stalls: int = 0
    stall_ratio: float = 0.0

    # 시간 응답
    rise_time: float = 0.0            # 10%→90% 도달 시간 (s)
    settling_time: float = 0.0         # ±5% 범위 정착 시간 (s)
    overshoot_vx: float = 0.0         # vx 오버슈트 비율

    num_samples: int = 0


class PiMPPIE2ETest(Node):
    def __init__(self, args):
        super().__init__('pi_mppi_e2e_test')
        self.args = args

        # 데이터 수집
        self.samples: List[SamplePoint] = []
        self.start_time = None
        self.last_odom_x = 0.0
        self.last_odom_y = 0.0
        self.last_odom_theta = 0.0
        self.odom_received = False

        # 다중 goal 파싱
        self.goals = self._parse_goals(args)
        self.current_goal_idx = 0

        # 제약 bounds (분석용 — 제어기 파라미터와 일치시켜야 함)
        self.rate_max_v = args.rate_max_v
        self.rate_max_omega = args.rate_max_omega
        self.accel_max_v = args.accel_max_v
        self.accel_max_omega = args.accel_max_omega

        # cmd_vel 구독
        self.cmd_vel_sub = self.create_subscription(
            TwistStamped, '/cmd_vel', self.cmd_vel_stamped_cb, 10)
        self.cmd_vel_unstamped_sub = self.create_subscription(
            Twist, '/cmd_vel_nav', self.cmd_vel_unstamped_cb, 10)

        # odom 구독
        self.odom_sub = self.create_subscription(
            Odometry, '/odom', self.odom_callback, 10)

        # nav2 action client
        if not args.record_only:
            self._action_client = ActionClient(
                self, NavigateToPose, 'navigate_to_pose')

        self.get_logger().info(
            f'π-MPPI E2E Test: {len(self.goals)} goal(s), '
            f'rate_max=[{self.rate_max_v}, {self.rate_max_omega}], '
            f'accel_max=[{self.accel_max_v}, {self.accel_max_omega}]')

    def _parse_goals(self, args) -> List[Tuple[float, float, float]]:
        """--goals "(x1,y1);(x2,y2)" 또는 --x/--y/--yaw 파싱"""
        if args.goals:
            goals = []
            for part in args.goals.split(';'):
                part = part.strip().strip('()')
                coords = part.split(',')
                x = float(coords[0])
                y = float(coords[1]) if len(coords) > 1 else 0.0
                yaw = float(coords[2]) if len(coords) > 2 else 0.0
                goals.append((x, y, yaw))
            return goals
        return [(args.x, args.y, args.yaw)]

    def odom_callback(self, msg):
        self.last_odom_x = msg.pose.pose.position.x
        self.last_odom_y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        self.last_odom_theta = math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        self.odom_received = True

    def cmd_vel_stamped_cb(self, msg):
        self._record_sample(msg.twist)

    def cmd_vel_unstamped_cb(self, msg):
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

    # ─────────────────────────────────────────
    # Goal 전송
    # ─────────────────────────────────────────

    def send_goal(self, x, y, yaw) -> bool:
        """단일 goal 전송 + 완료 대기"""
        self.get_logger().info(f'Goal 전송: ({x:.2f}, {y:.2f}, yaw={yaw:.2f})')

        goal_msg = NavigateToPose.Goal()
        goal_msg.pose = PoseStamped()
        goal_msg.pose.header.frame_id = 'map'
        goal_msg.pose.header.stamp = self.get_clock().now().to_msg()
        goal_msg.pose.pose.position.x = x
        goal_msg.pose.pose.position.y = y
        goal_msg.pose.pose.orientation.z = math.sin(yaw / 2.0)
        goal_msg.pose.pose.orientation.w = math.cos(yaw / 2.0)

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
            reached = (status == 4)
            self.get_logger().info(
                f'Goal {"도달" if reached else "실패"} (status={status})')
            return reached
        else:
            self.get_logger().warn(f'타임아웃 ({self.args.timeout}s)')
            return False

    def run_all_goals(self) -> Tuple[int, int]:
        """모든 goal 순차 실행"""
        self.get_logger().info('nav2 action server 대기 중...')
        if not self._action_client.wait_for_server(timeout_sec=30.0):
            self.get_logger().error('nav2 action server를 찾을 수 없습니다!')
            return 0, len(self.goals)

        reached = 0
        for i, (gx, gy, gyaw) in enumerate(self.goals):
            self.get_logger().info(
                f'━━━ Goal {i+1}/{len(self.goals)}: ({gx}, {gy}) ━━━')
            if self.send_goal(gx, gy, gyaw):
                reached += 1
            time.sleep(0.5)  # 다음 goal 전 잠시 대기
        return reached, len(self.goals)

    def _feedback_cb(self, feedback_msg):
        fb = feedback_msg.feedback
        self.get_logger().info(
            f'  남은 거리: {fb.distance_remaining:.2f}m',
            throttle_duration_sec=3.0)

    def record_only(self):
        """goal 없이 데이터만 수집"""
        self.get_logger().info(f'{self.args.timeout}초간 데이터 수집...')
        end_time = time.time() + self.args.timeout
        while time.time() < end_time:
            rclpy.spin_once(self, timeout_sec=0.1)

    # ─────────────────────────────────────────
    # 메트릭 계산
    # ─────────────────────────────────────────

    def compute_metrics(self, goals_reached=0, goals_total=0) -> E2EMetrics:
        m = E2EMetrics()
        m.goals_reached = goals_reached
        m.goals_total = goals_total
        m.all_goals_reached = (goals_reached == goals_total and goals_total > 0)
        n = len(self.samples)
        m.num_samples = n

        if n < 3:
            return m

        # ─── 궤적 통계 ───
        m.travel_time = self.samples[-1].t - self.samples[0].t

        dist = 0.0
        for i in range(1, n):
            dx = self.samples[i].x - self.samples[i - 1].x
            dy = self.samples[i].y - self.samples[i - 1].y
            dist += math.sqrt(dx * dx + dy * dy)
        m.travel_distance = dist

        last_goal = self.goals[-1] if self.goals else (0, 0, 0)
        m.final_position_error = math.sqrt(
            (self.last_odom_x - last_goal[0]) ** 2 +
            (self.last_odom_y - last_goal[1]) ** 2)
        yaw_diff = self.last_odom_theta - last_goal[2]
        m.final_yaw_error = abs(math.atan2(math.sin(yaw_diff), math.cos(yaw_diff)))

        # ─── 속도 통계 ───
        speeds = [math.sqrt(s.vx**2 + s.vy**2) for s in self.samples]
        m.mean_speed = sum(speeds) / n
        m.max_speed = max(speeds)

        # ─── Rate (1차 미분) 계산 ───
        rates_vx, rates_omega = [], []
        for i in range(1, n):
            dt = self.samples[i].t - self.samples[i - 1].t
            if dt < 1e-6 or dt > 1.0:  # 이상치 필터
                continue
            rate_vx = abs(self.samples[i].vx - self.samples[i - 1].vx) / dt
            rate_omega = abs(self.samples[i].omega - self.samples[i - 1].omega) / dt
            rates_vx.append(rate_vx)
            rates_omega.append(rate_omega)

        # ─── Accel/Jerk (2차 미분) 계산 ───
        jerks_vx, jerks_vy, jerks_omega = [], [], []
        for i in range(2, n):
            dt1 = self.samples[i].t - self.samples[i - 1].t
            dt2 = self.samples[i - 1].t - self.samples[i - 2].t
            if dt1 < 1e-6 or dt2 < 1e-6 or dt1 > 1.0 or dt2 > 1.0:
                continue

            acc1_vx = (self.samples[i].vx - self.samples[i - 1].vx) / dt1
            acc0_vx = (self.samples[i - 1].vx - self.samples[i - 2].vx) / dt2
            dt_avg = (dt1 + dt2) / 2.0
            j_vx = abs(acc1_vx - acc0_vx) / dt_avg

            acc1_vy = (self.samples[i].vy - self.samples[i - 1].vy) / dt1
            acc0_vy = (self.samples[i - 1].vy - self.samples[i - 2].vy) / dt2
            j_vy = abs(acc1_vy - acc0_vy) / dt_avg

            acc1_w = (self.samples[i].omega - self.samples[i - 1].omega) / dt1
            acc0_w = (self.samples[i - 1].omega - self.samples[i - 2].omega) / dt2
            j_w = abs(acc1_w - acc0_w) / dt_avg

            jerks_vx.append(j_vx)
            jerks_vy.append(j_vy)
            jerks_omega.append(j_w)

        # Jerk 통계
        if jerks_vx:
            m.mean_jerk_vx = sum(jerks_vx) / len(jerks_vx)
            m.mean_jerk_vy = sum(jerks_vy) / len(jerks_vy)
            m.mean_jerk_omega = sum(jerks_omega) / len(jerks_omega)
            rms_sq = sum(j**2 for j in jerks_vx) + sum(j**2 for j in jerks_vy)
            m.rms_jerk = math.sqrt(rms_sq / (len(jerks_vx) + len(jerks_vy)))

        # ─── ★ π-MPPI 제약 만족도 ───
        cm = ConstraintMetrics()

        if rates_vx:
            cm.rate_max_vx = max(rates_vx)
            cm.rate_max_omega = max(rates_omega)
            cm.rate_mean_vx = sum(rates_vx) / len(rates_vx)
            cm.rate_mean_omega = sum(rates_omega) / len(rates_omega)
            cm.rate_violation_count_vx = sum(
                1 for r in rates_vx if r > self.rate_max_v)
            cm.rate_violation_count_omega = sum(
                1 for r in rates_omega if r > self.rate_max_omega)
            cm.rate_total_samples = len(rates_vx)
            cm.rate_violation_ratio_vx = (
                cm.rate_violation_count_vx / cm.rate_total_samples)
            cm.rate_violation_ratio_omega = (
                cm.rate_violation_count_omega / cm.rate_total_samples)

        if jerks_vx:
            cm.accel_max_vx = max(jerks_vx)
            cm.accel_max_omega = max(jerks_omega)
            cm.accel_mean_vx = sum(jerks_vx) / len(jerks_vx)
            cm.accel_mean_omega = sum(jerks_omega) / len(jerks_omega)
            cm.accel_violation_count_vx = sum(
                1 for j in jerks_vx if j > self.accel_max_v)
            cm.accel_violation_count_omega = sum(
                1 for j in jerks_omega if j > self.accel_max_omega)
            cm.accel_total_samples = len(jerks_vx)
            cm.accel_violation_ratio_vx = (
                cm.accel_violation_count_vx / cm.accel_total_samples)
            cm.accel_violation_ratio_omega = (
                cm.accel_violation_count_omega / cm.accel_total_samples)

        m.constraints = cm

        # ─── 정체 감지 ───
        stall_count = sum(1 for s in speeds if s < 0.01)
        m.num_stalls = stall_count
        m.stall_ratio = stall_count / n if n > 0 else 0.0

        # ─── 시간 응답 분석 (vx 기준) ───
        m.rise_time, m.settling_time, m.overshoot_vx = self._time_response(
            [s.vx for s in self.samples],
            [s.t for s in self.samples])

        return m

    def _time_response(self, values, times):
        """시간 응답 분석: rise time, settling time, overshoot"""
        if len(values) < 10:
            return 0.0, 0.0, 0.0

        # 정상 상태 추정: 후반 20%의 평균
        n = len(values)
        tail = values[int(n * 0.8):]
        if not tail:
            return 0.0, 0.0, 0.0
        steady = sum(tail) / len(tail)
        if abs(steady) < 0.01:
            return 0.0, 0.0, 0.0

        # Rise time: 10% → 90%
        t10 = t90 = None
        for i, v in enumerate(values):
            if t10 is None and v >= 0.1 * steady:
                t10 = times[i]
            if t90 is None and v >= 0.9 * steady:
                t90 = times[i]
                break
        rise = (t90 - t10) if (t10 is not None and t90 is not None) else 0.0

        # Settling time: 마지막으로 ±5% 벗어난 시점
        settle = 0.0
        for i in range(n - 1, -1, -1):
            if abs(values[i] - steady) > 0.05 * abs(steady):
                settle = times[i] - times[0]
                break

        # Overshoot
        peak = max(values)
        overshoot = (peak - steady) / abs(steady) if abs(steady) > 0.01 else 0.0
        overshoot = max(0.0, overshoot)

        return rise, settle, overshoot

    # ─────────────────────────────────────────
    # 리포트 출력
    # ─────────────────────────────────────────

    def print_report(self, m: E2EMetrics):
        cm = m.constraints or ConstraintMetrics()
        W = 64

        def bar(ratio, width=20):
            """제약 만족도 바 (0%=전부 만족, 100%=전부 위반)"""
            filled = int(ratio * width)
            return '█' * filled + '░' * (width - filled)

        def status(ratio, label=""):
            if ratio < 0.01:
                return f'PASS ({label})'
            elif ratio < 0.1:
                return f'WARN ({label})'
            else:
                return f'FAIL ({label})'

        print()
        print('┌' + '─' * W + '┐')
        print('│' + ' π-MPPI E2E Analysis Report'.center(W) + '│')
        print('├' + '─' * W + '┤')

        # Goal
        goal_str = 'YES' if m.all_goals_reached else 'NO'
        print(f'│  Goals reached    : {m.goals_reached}/{m.goals_total} ({goal_str})'
              .ljust(W + 1) + '│')
        print(f'│  Samples          : {m.num_samples}'
              .ljust(W + 1) + '│')
        print(f'│  Travel time      : {m.travel_time:.1f} s'
              .ljust(W + 1) + '│')
        print(f'│  Travel distance  : {m.travel_distance:.2f} m'
              .ljust(W + 1) + '│')
        print(f'│  Position error   : {m.final_position_error:.3f} m'
              .ljust(W + 1) + '│')
        print(f'│  Yaw error        : {math.degrees(m.final_yaw_error):.1f} deg'
              .ljust(W + 1) + '│')

        print('├' + '─' * W + '┤')
        print('│' + ' Speed Profile'.center(W) + '│')
        print('├' + '─' * W + '┤')
        print(f'│  Mean speed       : {m.mean_speed:.3f} m/s'
              .ljust(W + 1) + '│')
        print(f'│  Max speed        : {m.max_speed:.3f} m/s'
              .ljust(W + 1) + '│')
        print(f'│  Stalls           : {m.num_stalls} ({m.stall_ratio:.1%})'
              .ljust(W + 1) + '│')

        print('├' + '─' * W + '┤')
        print('│' + ' Time Response'.center(W) + '│')
        print('├' + '─' * W + '┤')
        print(f'│  Rise time (10→90%): {m.rise_time:.2f} s'
              .ljust(W + 1) + '│')
        print(f'│  Settling time     : {m.settling_time:.2f} s'
              .ljust(W + 1) + '│')
        print(f'│  Overshoot vx      : {m.overshoot_vx:.1%}'
              .ljust(W + 1) + '│')

        print('├' + '─' * W + '┤')
        print('│' + ' Smoothness (lower = better)'.center(W) + '│')
        print('├' + '─' * W + '┤')
        print(f'│  Jerk vx   (mean) : {m.mean_jerk_vx:.3f} m/s³'
              .ljust(W + 1) + '│')
        print(f'│  Jerk vy   (mean) : {m.mean_jerk_vy:.3f} m/s³'
              .ljust(W + 1) + '│')
        print(f'│  Jerk omega(mean) : {m.mean_jerk_omega:.3f} rad/s³'
              .ljust(W + 1) + '│')
        print(f'│  Jerk RMS         : {m.rms_jerk:.3f}'
              .ljust(W + 1) + '│')

        # ★ 핵심: 제약 만족도
        print('├' + '─' * W + '┤')
        print('│' + ' ★ Constraint Satisfaction (π-MPPI)'.center(W) + '│')
        print('├' + '─' * W + '┤')

        print(f'│  Rate bound vx    : {self.rate_max_v:.1f} m/s²'
              .ljust(W + 1) + '│')
        print(f'│    measured max   : {cm.rate_max_vx:.3f} m/s²'
              .ljust(W + 1) + '│')
        print(f'│    measured mean  : {cm.rate_mean_vx:.3f} m/s²'
              .ljust(W + 1) + '│')
        s = status(cm.rate_violation_ratio_vx, f'{cm.rate_violation_count_vx}/{cm.rate_total_samples}')
        print(f'│    violations     : {bar(cm.rate_violation_ratio_vx)} {s}'
              .ljust(W + 1) + '│')

        print(f'│  Rate bound omega : {self.rate_max_omega:.1f} rad/s²'
              .ljust(W + 1) + '│')
        print(f'│    measured max   : {cm.rate_max_omega:.3f} rad/s²'
              .ljust(W + 1) + '│')
        s = status(cm.rate_violation_ratio_omega, f'{cm.rate_violation_count_omega}/{cm.rate_total_samples}')
        print(f'│    violations     : {bar(cm.rate_violation_ratio_omega)} {s}'
              .ljust(W + 1) + '│')

        print(f'│  Accel bound vx   : {self.accel_max_v:.1f} m/s³'
              .ljust(W + 1) + '│')
        print(f'│    measured max   : {cm.accel_max_vx:.3f} m/s³'
              .ljust(W + 1) + '│')
        s = status(cm.accel_violation_ratio_vx, f'{cm.accel_violation_count_vx}/{cm.accel_total_samples}')
        print(f'│    violations     : {bar(cm.accel_violation_ratio_vx)} {s}'
              .ljust(W + 1) + '│')

        print(f'│  Accel bound omega: {self.accel_max_omega:.1f} rad/s³'
              .ljust(W + 1) + '│')
        print(f'│    measured max   : {cm.accel_max_omega:.3f} rad/s³'
              .ljust(W + 1) + '│')
        s = status(cm.accel_violation_ratio_omega, f'{cm.accel_violation_count_omega}/{cm.accel_total_samples}')
        print(f'│    violations     : {bar(cm.accel_violation_ratio_omega)} {s}'
              .ljust(W + 1) + '│')

        # ─── 종합 판정 ───
        print('├' + '─' * W + '┤')
        print('│' + ' Summary'.center(W) + '│')
        print('├' + '─' * W + '┤')

        checks = [
            ('goal_reached', m.all_goals_reached or m.goals_total == 0),
            ('rate_vx_ok', cm.rate_violation_ratio_vx < 0.1),
            ('rate_omega_ok', cm.rate_violation_ratio_omega < 0.1),
            ('accel_vx_ok', cm.accel_violation_ratio_vx < 0.1),
            ('accel_omega_ok', cm.accel_violation_ratio_omega < 0.1),
            ('jerk_ok', m.rms_jerk < 50.0),
            ('no_stall', m.stall_ratio < 0.3),
        ]
        for name, ok in checks:
            mark = 'PASS' if ok else 'FAIL'
            print(f'│  [{mark}] {name}'.ljust(W + 1) + '│')

        all_pass = all(ok for _, ok in checks)
        result = 'ALL PASS' if all_pass else 'SOME FAILED'
        print('├' + '─' * W + '┤')
        print(f'│  Result: {result}'.ljust(W + 1) + '│')
        print('└' + '─' * W + '┘')
        print()

    # ─────────────────────────────────────────
    # 데이터 저장
    # ─────────────────────────────────────────

    def save_data(self, metrics: E2EMetrics):
        output_dir = os.path.expanduser('~/pi_mppi_e2e_results')
        os.makedirs(output_dir, exist_ok=True)

        timestamp = time.strftime('%Y%m%d_%H%M%S')
        filepath = os.path.join(output_dir, f'pi_mppi_e2e_{timestamp}.json')

        data = {
            'metrics': asdict(metrics),
            'config': {
                'goals': self.goals,
                'timeout': self.args.timeout,
                'rate_max_v': self.rate_max_v,
                'rate_max_omega': self.rate_max_omega,
                'accel_max_v': self.accel_max_v,
                'accel_max_omega': self.accel_max_omega,
            },
            'samples': [asdict(s) for s in self.samples]
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

        self.get_logger().info(f'데이터 저장: {filepath}')
        return filepath


def main():
    parser = argparse.ArgumentParser(
        description='π-MPPI E2E 검증 — rate/accel 제약 만족도 분석')
    parser.add_argument('--x', type=float, default=5.0, help='목표 x (m)')
    parser.add_argument('--y', type=float, default=0.0, help='목표 y (m)')
    parser.add_argument('--yaw', type=float, default=0.0, help='목표 yaw (rad)')
    parser.add_argument('--goals', type=str, default=None,
                        help='다중 goal: "(x1,y1);(x2,y2);..."')
    parser.add_argument('--timeout', type=float, default=90.0,
                        help='goal별 타임아웃 (s)')
    parser.add_argument('--record-only', action='store_true',
                        help='Goal 없이 데이터만 수집')
    parser.add_argument('--no-save', action='store_true',
                        help='JSON 저장 건너뛰기')

    # π-MPPI 제약 bounds (제어기 파라미터와 일치시켜야 분석 정확)
    parser.add_argument('--rate-max-v', type=float, default=2.0,
                        help='rate bound vx (m/s²)')
    parser.add_argument('--rate-max-omega', type=float, default=3.0,
                        help='rate bound omega (rad/s²)')
    parser.add_argument('--accel-max-v', type=float, default=5.0,
                        help='accel bound vx (m/s³)')
    parser.add_argument('--accel-max-omega', type=float, default=8.0,
                        help='accel bound omega (rad/s³)')
    args = parser.parse_args()

    rclpy.init()
    node = PiMPPIE2ETest(args)

    try:
        if args.record_only:
            node.record_only()
            reached, total = 0, 0
        else:
            reached, total = node.run_all_goals()

        metrics = node.compute_metrics(reached, total)
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
