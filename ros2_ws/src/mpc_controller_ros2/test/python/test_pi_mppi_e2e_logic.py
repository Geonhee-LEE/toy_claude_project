#!/usr/bin/env python3
"""
π-MPPI E2E 분석 로직 단위 테스트

ROS2/Gazebo 없이 합성 데이터로 메트릭 계산 로직을 검증.

테스트 항목:
  [A] SamplePoint / 데이터 구조 (2)
  [B] 궤적 통계 (3)
  [C] Rate 제약 분석 (3)
  [D] Accel/Jerk 제약 분석 (3)
  [E] 시간 응답 분석 (3)
  [F] 리포트 출력 (1)
  [G] 엣지 케이스 (2)
  [H] Goal 파싱 (2)
  [I] JSON 직렬화 (1)

총 20 tests
"""

import sys
import os
import math
import json
import unittest
from dataclasses import asdict
from unittest.mock import MagicMock
from typing import List

# pi_mppi_e2e_test.py의 순수 로직만 import하기 위해
# ROS2 의존성을 mock 처리
sys.modules['rclpy'] = MagicMock()
sys.modules['rclpy.node'] = MagicMock()
sys.modules['rclpy.action'] = MagicMock()
sys.modules['nav2_msgs'] = MagicMock()
sys.modules['nav2_msgs.action'] = MagicMock()
sys.modules['geometry_msgs'] = MagicMock()
sys.modules['geometry_msgs.msg'] = MagicMock()
sys.modules['nav_msgs'] = MagicMock()
sys.modules['nav_msgs.msg'] = MagicMock()

# 스크립트 경로 추가
scripts_dir = os.path.join(
    os.path.dirname(__file__), '..', '..', 'scripts')
sys.path.insert(0, os.path.abspath(scripts_dir))

from pi_mppi_e2e_test import (
    SamplePoint, E2EMetrics, ConstraintMetrics,
    parse_goals_string, compute_time_response,
    compute_e2e_metrics, print_e2e_report,
)

# 기본 분석 파라미터
DEFAULT_RATE_MAX_V = 2.0
DEFAULT_RATE_MAX_OMEGA = 3.0
DEFAULT_ACCEL_MAX_V = 5.0
DEFAULT_ACCEL_MAX_OMEGA = 8.0
DEFAULT_GOALS = [(5.0, 0.0, 0.0)]


# =============================================================================
# Helper: 합성 데이터 생성
# =============================================================================

def make_constant_samples(n=100, dt=0.1, vx=0.3, omega=0.0) -> List[SamplePoint]:
    """일정 속도 직진 데이터"""
    samples = []
    x, y, theta = 0.0, 0.0, 0.0
    for i in range(n):
        t = i * dt
        samples.append(SamplePoint(
            t=t, vx=vx, vy=0.0, omega=omega,
            x=x, y=y, theta=theta
        ))
        x += vx * dt * math.cos(theta)
        y += vx * dt * math.sin(theta)
        theta += omega * dt
    return samples


def make_step_samples(n=100, dt=0.1, v_start=0.0, v_end=0.5,
                      step_at=50) -> List[SamplePoint]:
    """스텝 입력 (step_at 시점에 속도 급변)"""
    samples = []
    x = 0.0
    for i in range(n):
        t = i * dt
        vx = v_start if i < step_at else v_end
        samples.append(SamplePoint(
            t=t, vx=vx, vy=0.0, omega=0.0,
            x=x, y=0.0, theta=0.0
        ))
        x += vx * dt
    return samples


def make_ramp_samples(n=100, dt=0.1, v_max=0.5) -> List[SamplePoint]:
    """선형 ramp-up → 정상 상태"""
    samples = []
    x = 0.0
    ramp_end = n // 3
    for i in range(n):
        t = i * dt
        if i < ramp_end:
            vx = v_max * (i / ramp_end)
        else:
            vx = v_max
        samples.append(SamplePoint(
            t=t, vx=vx, vy=0.0, omega=0.0,
            x=x, y=0.0, theta=0.0
        ))
        x += vx * dt
    return samples


def make_sinusoidal_samples(n=200, dt=0.1, freq=0.5, amp=0.3,
                            bias=0.3) -> List[SamplePoint]:
    """사인파 속도 (고주파 → 높은 rate/accel)"""
    samples = []
    x = 0.0
    for i in range(n):
        t = i * dt
        vx = bias + amp * math.sin(2 * math.pi * freq * t)
        omega = 0.2 * math.sin(2 * math.pi * freq * 0.7 * t)
        samples.append(SamplePoint(
            t=t, vx=vx, vy=0.0, omega=omega,
            x=x, y=0.0, theta=0.0
        ))
        x += vx * dt
    return samples


def make_noisy_samples(n=100, dt=0.1, vx_base=0.3,
                       noise_amp=0.05) -> List[SamplePoint]:
    """약한 노이즈가 있는 데이터 (제약 만족 기대)"""
    import random
    random.seed(42)
    samples = []
    x = 0.0
    for i in range(n):
        t = i * dt
        vx = vx_base + noise_amp * (random.random() - 0.5)
        samples.append(SamplePoint(
            t=t, vx=vx, vy=0.0, omega=0.0,
            x=x, y=0.0, theta=0.0
        ))
        x += vx * dt
    return samples


def _metrics(samples, goals=None, rate_max_v=DEFAULT_RATE_MAX_V,
             rate_max_omega=DEFAULT_RATE_MAX_OMEGA,
             accel_max_v=DEFAULT_ACCEL_MAX_V,
             accel_max_omega=DEFAULT_ACCEL_MAX_OMEGA,
             goals_reached=1, goals_total=1):
    """헬퍼: compute_e2e_metrics 래퍼"""
    goals = goals or DEFAULT_GOALS
    last = samples[-1] if samples else SamplePoint(0, 0, 0, 0, 0, 0, 0)
    return compute_e2e_metrics(
        samples=samples,
        goals=goals,
        last_odom_x=last.x,
        last_odom_y=last.y,
        last_odom_theta=last.theta,
        rate_max_v=rate_max_v,
        rate_max_omega=rate_max_omega,
        accel_max_v=accel_max_v,
        accel_max_omega=accel_max_omega,
        goals_reached=goals_reached,
        goals_total=goals_total,
    )


# =============================================================================
# [A] 데이터 구조 테스트 (2)
# =============================================================================

class TestDataStructures(unittest.TestCase):

    def test_sample_point_fields(self):
        """SamplePoint 필드가 올바르게 생성되는지"""
        sp = SamplePoint(t=1.0, vx=0.3, vy=0.1, omega=0.5,
                         x=1.0, y=2.0, theta=0.3)
        self.assertAlmostEqual(sp.t, 1.0)
        self.assertAlmostEqual(sp.vx, 0.3)
        self.assertAlmostEqual(sp.omega, 0.5)

    def test_constraint_metrics_defaults(self):
        """ConstraintMetrics 기본값이 0"""
        cm = ConstraintMetrics()
        self.assertEqual(cm.rate_violation_count_vx, 0)
        self.assertAlmostEqual(cm.rate_violation_ratio_vx, 0.0)
        self.assertAlmostEqual(cm.accel_max_vx, 0.0)


# =============================================================================
# [B] 궤적 통계 테스트 (3)
# =============================================================================

class TestTrajectoryStats(unittest.TestCase):

    def test_travel_time(self):
        """travel_time이 올바르게 계산되는지"""
        samples = make_constant_samples(n=50, dt=0.1, vx=0.5)
        m = _metrics(samples)
        expected_time = 49 * 0.1  # (n-1) * dt
        self.assertAlmostEqual(m.travel_time, expected_time, places=1)

    def test_travel_distance(self):
        """travel_distance가 직진 시 vx * time과 유사"""
        samples = make_constant_samples(n=100, dt=0.1, vx=0.3)
        m = _metrics(samples)
        self.assertGreater(m.travel_distance, 2.5)
        self.assertLess(m.travel_distance, 3.5)

    def test_speed_stats(self):
        """mean/max speed 계산"""
        samples = make_constant_samples(n=50, dt=0.1, vx=0.4)
        m = _metrics(samples)
        self.assertAlmostEqual(m.mean_speed, 0.4, places=2)
        self.assertAlmostEqual(m.max_speed, 0.4, places=2)


# =============================================================================
# [C] Rate 제약 분석 테스트 (3)
# =============================================================================

class TestRateConstraints(unittest.TestCase):

    def test_constant_velocity_no_rate_violation(self):
        """일정 속도 → rate=0 → 위반 없음"""
        samples = make_constant_samples(n=100, dt=0.1, vx=0.3)
        m = _metrics(samples, rate_max_v=2.0)
        cm = m.constraints
        self.assertEqual(cm.rate_violation_count_vx, 0)
        self.assertAlmostEqual(cm.rate_violation_ratio_vx, 0.0)
        self.assertLess(cm.rate_max_vx, 0.01)

    def test_step_input_rate_violation(self):
        """스텝 입력 (0→0.5 즉시) → rate = 0.5/0.1 = 5.0 > 2.0 → 위반"""
        samples = make_step_samples(n=100, dt=0.1, v_start=0.0,
                                    v_end=0.5, step_at=50)
        m = _metrics(samples, rate_max_v=2.0)
        cm = m.constraints
        self.assertGreater(cm.rate_violation_count_vx, 0)
        self.assertGreater(cm.rate_max_vx, 2.0)

    def test_ramp_input_no_rate_violation(self):
        """선형 ramp (rate = v_max / ramp_time) → 충분히 느리면 위반 없음"""
        samples = make_ramp_samples(n=100, dt=0.1, v_max=0.5)
        m = _metrics(samples, rate_max_v=2.0)
        cm = m.constraints
        self.assertEqual(cm.rate_violation_count_vx, 0)
        self.assertLess(cm.rate_max_vx, 2.0)


# =============================================================================
# [D] Accel/Jerk 제약 분석 테스트 (3)
# =============================================================================

class TestAccelConstraints(unittest.TestCase):

    def test_constant_velocity_no_accel(self):
        """일정 속도 → jerk=0 → 위반 없음"""
        samples = make_constant_samples(n=100, dt=0.1, vx=0.3)
        m = _metrics(samples, accel_max_v=5.0)
        self.assertLess(m.mean_jerk_vx, 0.01)
        self.assertEqual(m.constraints.accel_violation_count_vx, 0)

    def test_high_freq_sinusoidal_accel_violation(self):
        """고주파 사인파 → 높은 가속도 → 위반 가능"""
        samples = make_sinusoidal_samples(
            n=200, dt=0.1, freq=2.0, amp=0.5, bias=0.5)
        m = _metrics(samples, accel_max_v=5.0)
        self.assertGreater(m.constraints.accel_max_vx, 5.0)
        self.assertGreater(m.constraints.accel_violation_count_vx, 0)

    def test_smooth_signal_no_accel_violation(self):
        """매끄러운 신호 → 작은 accel → 위반 없음"""
        samples = make_noisy_samples(n=100, dt=0.1, vx_base=0.3,
                                     noise_amp=0.01)
        m = _metrics(samples, accel_max_v=50.0)
        self.assertEqual(m.constraints.accel_violation_count_vx, 0)


# =============================================================================
# [E] 시간 응답 분석 테스트 (3)
# =============================================================================

class TestTimeResponse(unittest.TestCase):

    def test_rise_time_ramp(self):
        """ramp-up에서 rise time 계산"""
        samples = make_ramp_samples(n=100, dt=0.1, v_max=0.5)
        m = _metrics(samples)
        self.assertGreater(m.rise_time, 0.5)
        self.assertLess(m.rise_time, 5.0)

    def test_step_has_zero_rise_time(self):
        """스텝 입력 → rise time ≈ 0 (즉시 도달)"""
        samples = make_step_samples(n=100, dt=0.1, v_start=0.0,
                                    v_end=0.5, step_at=50)
        m = _metrics(samples)
        self.assertLess(m.rise_time, 0.2)

    def test_constant_no_overshoot(self):
        """일정 속도 → 오버슈트 0"""
        samples = make_constant_samples(n=100, dt=0.1, vx=0.3)
        m = _metrics(samples)
        self.assertAlmostEqual(m.overshoot_vx, 0.0, places=2)

    def test_compute_time_response_direct(self):
        """compute_time_response 순수 함수 직접 호출"""
        values = [0.0] * 10 + [1.0] * 90
        times = [i * 0.1 for i in range(100)]
        rise, settle, overshoot = compute_time_response(values, times)
        self.assertLess(rise, 0.2)  # 즉시 전환
        self.assertAlmostEqual(overshoot, 0.0, places=2)


# =============================================================================
# [F] 리포트 출력 테스트 (1)
# =============================================================================

class TestReport(unittest.TestCase):

    def test_print_report_no_crash(self):
        """print_e2e_report가 예외 없이 동작하는지"""
        samples = make_ramp_samples(n=50, dt=0.1, v_max=0.5)
        m = _metrics(samples)

        import io
        from contextlib import redirect_stdout
        buf = io.StringIO()
        with redirect_stdout(buf):
            print_e2e_report(m, DEFAULT_RATE_MAX_V, DEFAULT_RATE_MAX_OMEGA,
                             DEFAULT_ACCEL_MAX_V, DEFAULT_ACCEL_MAX_OMEGA)

        output = buf.getvalue()
        self.assertIn('π-MPPI', output)
        self.assertIn('Constraint Satisfaction', output)
        self.assertIn('PASS', output)


# =============================================================================
# [G] 엣지 케이스 테스트 (2)
# =============================================================================

class TestEdgeCases(unittest.TestCase):

    def test_too_few_samples(self):
        """샘플 < 3개 → 기본 메트릭 반환 (crash 없음)"""
        samples = [
            SamplePoint(t=0.0, vx=0.0, vy=0.0, omega=0.0,
                        x=0.0, y=0.0, theta=0.0)
        ]
        m = _metrics(samples, goals_reached=0)
        self.assertEqual(m.num_samples, 1)
        self.assertAlmostEqual(m.travel_time, 0.0)
        self.assertIsNone(m.constraints)

    def test_zero_dt_samples_filtered(self):
        """dt=0인 중복 샘플 → 이상치 필터링, crash 없음"""
        samples = [
            SamplePoint(t=0.0, vx=0.0, vy=0.0, omega=0.0,
                        x=0.0, y=0.0, theta=0.0),
            SamplePoint(t=0.0, vx=0.5, vy=0.0, omega=0.0,
                        x=0.0, y=0.0, theta=0.0),  # dt=0!
            SamplePoint(t=0.0, vx=0.5, vy=0.0, omega=0.0,
                        x=0.0, y=0.0, theta=0.0),  # dt=0!
            SamplePoint(t=0.1, vx=0.5, vy=0.0, omega=0.0,
                        x=0.05, y=0.0, theta=0.0),
            SamplePoint(t=0.2, vx=0.5, vy=0.0, omega=0.0,
                        x=0.1, y=0.0, theta=0.0),
        ]
        m = _metrics(samples, goals_reached=0, goals_total=0)
        self.assertEqual(m.num_samples, 5)
        self.assertTrue(math.isfinite(m.mean_speed))


# =============================================================================
# [H] Goal 파싱 테스트 (2)
# =============================================================================

class TestGoalParsing(unittest.TestCase):

    def test_single_goal(self):
        """단일 goal 파싱"""
        goals = parse_goals_string("(3.0,2.0)")
        self.assertEqual(len(goals), 1)
        self.assertAlmostEqual(goals[0][0], 3.0)
        self.assertAlmostEqual(goals[0][1], 2.0)
        self.assertAlmostEqual(goals[0][2], 0.0)  # yaw 기본값

    def test_parse_multi_goals(self):
        """다중 goal 파싱"""
        goals = parse_goals_string("(5,0);(5,3);(0,3);(0,0)")
        self.assertEqual(len(goals), 4)
        self.assertAlmostEqual(goals[0][0], 5.0)
        self.assertAlmostEqual(goals[0][1], 0.0)
        self.assertAlmostEqual(goals[1][0], 5.0)
        self.assertAlmostEqual(goals[1][1], 3.0)
        self.assertAlmostEqual(goals[2][0], 0.0)
        self.assertAlmostEqual(goals[2][1], 3.0)
        self.assertAlmostEqual(goals[3][0], 0.0)
        self.assertAlmostEqual(goals[3][1], 0.0)


# =============================================================================
# [I] JSON 직렬화 테스트 (1)
# =============================================================================

class TestJsonSerialization(unittest.TestCase):

    def test_metrics_serializable(self):
        """E2EMetrics가 JSON 직렬화 가능"""
        samples = make_ramp_samples(n=50, dt=0.1, v_max=0.5)
        m = _metrics(samples)

        data = {
            'metrics': asdict(m),
            'config': {
                'goals': DEFAULT_GOALS,
                'rate_max_v': DEFAULT_RATE_MAX_V,
            },
            'samples': [asdict(s) for s in samples[:5]],
        }

        json_str = json.dumps(data, indent=2)
        self.assertIn('"travel_time"', json_str)
        self.assertIn('"rate_violation_count_vx"', json_str)

        loaded = json.loads(json_str)
        self.assertAlmostEqual(
            loaded['metrics']['travel_time'], m.travel_time, places=3)


# =============================================================================
# [추가] 통합 시나리오 테스트 (1)
# =============================================================================

class TestIntegrationScenarios(unittest.TestCase):

    def test_smooth_vs_rough_comparison(self):
        """매끄러운 신호 vs 거친 신호 → jerk/rate 차이 검증"""
        samples_smooth = make_ramp_samples(n=100, dt=0.1, v_max=0.5)
        m_smooth = _metrics(samples_smooth, rate_max_v=2.0, accel_max_v=5.0)

        samples_rough = make_sinusoidal_samples(
            n=100, dt=0.1, freq=2.0, amp=0.3, bias=0.3)
        m_rough = _metrics(samples_rough, rate_max_v=2.0, accel_max_v=5.0)

        # 매끄러운 신호가 jerk RMS 더 낮아야 함
        self.assertLess(m_smooth.rms_jerk, m_rough.rms_jerk)

        # 거친 신호가 rate 위반 더 많아야 함
        self.assertGreaterEqual(
            m_rough.constraints.rate_violation_count_vx,
            m_smooth.constraints.rate_violation_count_vx)

        print(f'\n[Smooth vs Rough]')
        print(f'  Smooth: jerk_rms={m_smooth.rms_jerk:.3f}, '
              f'rate_violations={m_smooth.constraints.rate_violation_count_vx}')
        print(f'  Rough:  jerk_rms={m_rough.rms_jerk:.3f}, '
              f'rate_violations={m_rough.constraints.rate_violation_count_vx}')


if __name__ == '__main__':
    unittest.main()
