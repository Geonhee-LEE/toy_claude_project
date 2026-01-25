#!/usr/bin/env python3
"""충돌 시간 예측 디버그 스크립트."""

from mpc_controller.planners.dynamic_obstacle_predictor import (
    DynamicObstaclePredictor,
    DynamicObstacleState,
    PredictionModel,
)

predictor = DynamicObstaclePredictor(
    model=PredictionModel.CONSTANT_VELOCITY,
    max_prediction_time=10.0,  # 10초로 증가
)

# 장애물이 로봇을 향해 다가옴
state = DynamicObstacleState(
    x=10.0, y=0.0, vx=-2.0, vy=0.0, radius=0.3
)

robot_x, robot_y = 0.0, 0.0
robot_radius = 0.3
safety_margin = 0.4

print("장애물 정보:")
print(f"  위치: ({state.x}, {state.y})")
print(f"  속도: ({state.vx}, {state.vy})")
print(f"  반경: {state.radius}")

print("\n로봇 정보:")
print(f"  위치: ({robot_x}, {robot_y})")
print(f"  반경: {robot_radius}")
print(f"  안전 마진: {safety_margin}")

# 디버깅
import numpy as np

rel_x = state.x - robot_x
rel_y = state.y - robot_y
rel_vx = state.vx
rel_vy = state.vy

current_dist = np.sqrt(rel_x**2 + rel_y**2)
collision_dist = state.radius + robot_radius + safety_margin

print(f"\n계산 정보:")
print(f"  현재 거리: {current_dist:.3f}m")
print(f"  충돌 거리: {collision_dist:.3f}m")

a = rel_vx**2 + rel_vy**2
b = 2 * (rel_x * rel_vx + rel_y * rel_vy)
c = rel_x**2 + rel_y**2 - collision_dist**2

print(f"\n2차 방정식 계수:")
print(f"  a = {a:.3f}")
print(f"  b = {b:.3f}")
print(f"  c = {c:.3f}")

discriminant = b**2 - 4*a*c
print(f"  discriminant = {discriminant:.3f}")

if discriminant >= 0:
    sqrt_disc = np.sqrt(discriminant)
    t1 = (-b - sqrt_disc) / (2*a)
    t2 = (-b + sqrt_disc) / (2*a)
    print(f"  t1 = {t1:.3f}초")
    print(f"  t2 = {t2:.3f}초")

collision_time = predictor.predict_collision_time(
    state, robot_x, robot_y, robot_radius, safety_margin
)

print(f"\n충돌 시간 예측: {collision_time}")
