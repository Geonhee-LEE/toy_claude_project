#!/usr/bin/env python3
"""
동적 장애물 예측기 검증 스크립트.

pytest 없이 직접 테스트를 실행합니다.
"""

import sys
import numpy as np

from mpc_controller.planners.dynamic_obstacle_predictor import (
    DynamicObstaclePredictor,
    DynamicObstacleState,
    PredictionModel,
)


def test_constant_velocity_prediction():
    """일정 속도 모델 예측 테스트."""
    print("\n[TEST] Constant Velocity Prediction")

    predictor = DynamicObstaclePredictor(
        model=PredictionModel.CONSTANT_VELOCITY,
        uncertainty_growth_rate=0.1,
    )

    state = DynamicObstacleState(
        x=0.0, y=0.0, vx=1.0, vy=0.5, radius=0.3
    )

    dt = 0.1
    steps = 10
    predictions = predictor.predict_trajectory(state, dt, steps)

    # 10스텝 후 위치 확인
    pred_x, pred_y, pred_radius = predictions[-1]
    expected_x = state.x + state.vx * (dt * steps)
    expected_y = state.y + state.vy * (dt * steps)

    assert abs(pred_x - expected_x) < 1e-5, f"X 예측 실패: {pred_x} != {expected_x}"
    assert abs(pred_y - expected_y) < 1e-5, f"Y 예측 실패: {pred_y} != {expected_y}"

    print(f"  ✓ 예측 위치: ({pred_x:.3f}, {pred_y:.3f})")
    print(f"  ✓ 예측 반경 (불확실성 포함): {pred_radius:.3f}m")
    print("  PASS")


def test_collision_time_prediction():
    """충돌 시간 예측 테스트."""
    print("\n[TEST] Collision Time Prediction")

    predictor = DynamicObstaclePredictor(
        model=PredictionModel.CONSTANT_VELOCITY,
        max_prediction_time=10.0,  # 충분한 예측 시간 설정
    )

    # 장애물이 로봇을 향해 다가옴
    state = DynamicObstacleState(
        x=10.0, y=0.0, vx=-2.0, vy=0.0, radius=0.3
    )

    robot_x, robot_y = 0.0, 0.0
    robot_radius = 0.3
    safety_margin = 0.4

    collision_time = predictor.predict_collision_time(
        state, robot_x, robot_y, robot_radius, safety_margin
    )

    # 충돌 거리: 0.3 + 0.3 + 0.4 = 1.0m
    # 상대 속도: 2.0 m/s
    # 충돌 시간: (10.0 - 1.0) / 2.0 = 4.5초
    expected_time = 4.5

    assert collision_time is not None, "충돌 시간이 None"
    assert abs(collision_time - expected_time) < 0.1, \
        f"충돌 시간 예측 실패: {collision_time} != {expected_time}"

    print(f"  ✓ 예측 충돌 시간: {collision_time:.2f}초")
    print("  PASS")


def test_constant_acceleration_prediction():
    """일정 가속도 모델 예측 테스트."""
    print("\n[TEST] Constant Acceleration Prediction")

    predictor = DynamicObstaclePredictor(
        model=PredictionModel.CONSTANT_ACCELERATION,
        uncertainty_growth_rate=0.1,
    )

    state = DynamicObstacleState(
        x=0.0, y=0.0, vx=0.0, vy=0.0, ax=1.0, ay=0.5, radius=0.3
    )

    dt = 0.1
    steps = 10
    predictions = predictor.predict_trajectory(state, dt, steps)

    # 10스텝 후 위치 확인
    pred_x, pred_y, pred_radius = predictions[-1]
    t = dt * steps

    expected_x = state.x + state.vx * t + 0.5 * state.ax * t**2
    expected_y = state.y + state.vy * t + 0.5 * state.ay * t**2

    assert abs(pred_x - expected_x) < 1e-5, f"X 예측 실패: {pred_x} != {expected_x}"
    assert abs(pred_y - expected_y) < 1e-5, f"Y 예측 실패: {pred_y} != {expected_y}"

    print(f"  ✓ 예측 위치 (가속 운동): ({pred_x:.3f}, {pred_y:.3f})")
    print("  PASS")


def test_acceleration_estimation():
    """가속도 추정 테스트."""
    print("\n[TEST] Acceleration Estimation")

    predictor = DynamicObstaclePredictor(
        model=PredictionModel.CONSTANT_VELOCITY
    )

    obstacle_id = 1

    # 가속 운동 시뮬레이션
    state1 = DynamicObstacleState(
        x=0.0, y=0.0, vx=1.0, vy=0.0, radius=0.3, timestamp=0.0
    )
    state2 = DynamicObstacleState(
        x=0.105, y=0.0, vx=1.05, vy=0.0, radius=0.3, timestamp=0.1
    )

    predictor.update_state_history(obstacle_id, state1)
    predictor.update_state_history(obstacle_id, state2)

    ax, ay = predictor.estimate_acceleration(obstacle_id)

    expected_ax = 0.5
    expected_ay = 0.0

    assert abs(ax - expected_ax) < 1e-5, f"가속도 X 추정 실패: {ax} != {expected_ax}"
    assert abs(ay - expected_ay) < 1e-5, f"가속도 Y 추정 실패: {ay} != {expected_ay}"

    print(f"  ✓ 추정 가속도: ({ax:.3f}, {ay:.3f}) m/s²")
    print("  PASS")


def test_no_collision():
    """충돌 없음 테스트."""
    print("\n[TEST] No Collision Scenario")

    predictor = DynamicObstaclePredictor(
        model=PredictionModel.CONSTANT_VELOCITY
    )

    # 장애물이 로봇에서 멀어짐
    state = DynamicObstacleState(
        x=10.0, y=0.0, vx=2.0, vy=0.0, radius=0.3
    )

    robot_x, robot_y = 0.0, 0.0
    robot_radius = 0.3

    collision_time = predictor.predict_collision_time(
        state, robot_x, robot_y, robot_radius
    )

    assert collision_time is None, "충돌 없어야 하는데 충돌 시간이 계산됨"

    print("  ✓ 충돌 없음 (예상대로)")
    print("  PASS")


def main():
    """모든 테스트 실행."""
    print("=" * 70)
    print("  동적 장애물 예측기 검증 테스트")
    print("=" * 70)

    tests = [
        test_constant_velocity_prediction,
        test_collision_time_prediction,
        test_constant_acceleration_prediction,
        test_acceleration_estimation,
        test_no_collision,
    ]

    passed = 0
    failed = 0

    for test_func in tests:
        try:
            test_func()
            passed += 1
        except AssertionError as e:
            print(f"  FAILED: {e}")
            failed += 1
        except Exception as e:
            print(f"  ERROR: {e}")
            failed += 1

    print("\n" + "=" * 70)
    print(f"  테스트 결과: {passed} 통과, {failed} 실패")
    print("=" * 70)

    if failed > 0:
        sys.exit(1)

    print("\n모든 테스트 통과! ✓")


if __name__ == "__main__":
    main()
