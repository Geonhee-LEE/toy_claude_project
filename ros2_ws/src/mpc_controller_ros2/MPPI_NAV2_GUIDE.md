# MPPI Controller + nav2 실행 가이드

## 시스템 구성도

```
┌─────────────────────────────────────────────────┐
│          Gazebo Harmonic Simulation             │
│  - DiffDrive / Swerve / NonCoaxial robot       │
│  - World with obstacles                         │
│  - Lidar sensor                                 │
└─────────────────────────────────────────────────┘
                    ↓ (ros_gz_bridge)
┌─────────────────────────────────────────────────┐
│              ROS2 nav2 Stack                    │
│  ┌───────────────────────────────────────────┐  │
│  │ bt_navigator (Behavior Tree)              │  │
│  └───────────────────────────────────────────┘  │
│         ↓ (goal)          ↓ (path)              │
│  ┌──────────────┐   ┌──────────────────────┐   │
│  │planner_server│   │ controller_server    │   │
│  │  (NavFn)     │   │  (MPPI 8종 플러그인) │   │
│  └──────────────┘   └──────────────────────┘   │
│         ↓                    ↓                   │
│  ┌─────────────────────────────────────────┐   │
│  │      Costmaps (local/global)            │   │
│  │  - Obstacle detection (Lidar)           │   │
│  │  - Inflation layer                      │   │
│  └─────────────────────────────────────────┘   │
└─────────────────────────────────────────────────┘
                    ↓ (cmd_vel)
┌─────────────────────────────────────────────────┐
│         Robot Hardware (Gazebo)                 │
│  - DiffDrive / Swerve controller               │
│  - Odometry feedback                            │
└─────────────────────────────────────────────────┘
```

## 빌드

```bash
cd /home/geonhee/toy_claude_project/ros2_ws
colcon build --packages-select mpc_controller_ros2 --symlink-install
source install/setup.bash
```

## 실행 방법

### 방법 1: 통합 Launch 파일 (권장)

하나의 명령으로 Gazebo + nav2 + MPPI 모두 실행:

```bash
source install/setup.bash

# DiffDrive (기본)
ros2 launch mpc_controller_ros2 mppi_ros2_control_nav2.launch.py

# Swerve Drive (홀로노믹, vx/vy/omega)
ros2 launch mpc_controller_ros2 mppi_ros2_control_nav2.launch.py controller:=swerve

# Non-Coaxial Swerve (비홀로노믹, v/omega/delta_dot)
ros2 launch mpc_controller_ros2 mppi_ros2_control_nav2.launch.py controller:=non_coaxial

# 플러그인 변형 사용
ros2 launch mpc_controller_ros2 mppi_ros2_control_nav2.launch.py controller:=spline  # Spline-MPPI
ros2 launch mpc_controller_ros2 mppi_ros2_control_nav2.launch.py controller:=log     # Log-MPPI
ros2 launch mpc_controller_ros2 mppi_ros2_control_nav2.launch.py controller:=svmpc   # SVMPC
```

실행되는 노드들:
- ✅ Gazebo Harmonic (시뮬레이터)
- ✅ ros_gz_bridge (Gazebo ↔ ROS2)
- ✅ map_server (맵 서버)
- ✅ controller_server (MPPI 컨트롤러)
- ✅ planner_server (경로 계획)
- ✅ behavior_server (행동 서버)
- ✅ bt_navigator (행동 트리)
- ✅ velocity_smoother (속도 평활화)
- ✅ lifecycle_manager (노드 관리)
- ✅ RVIZ2 (시각화)

### 방법 2: 개별 실행

**터미널 1: Gazebo + Bridge**
```bash
ros2 launch mpc_controller_ros2 gazebo_harmonic_test.launch.py
```

**터미널 2: nav2 Stack**
```bash
ros2 launch nav2_bringup navigation_launch.py \
    params_file:=src/mpc_controller_ros2/config/nav2_params.yaml \
    use_sim_time:=true
```

## Goal 전송

### 방법 1: Python 스크립트 (권장)

```bash
# 기본 목표 (5m 전진)
ros2 run mpc_controller_ros2 send_nav_goal.py

# 커스텀 목표
ros2 run mpc_controller_ros2 send_nav_goal.py --x 8.0 --y 2.0 --yaw 1.57

# 여러 목표 예시
ros2 run mpc_controller_ros2 send_nav_goal.py --x 3.0 --y 0.0 --yaw 0.0    # 전진
ros2 run mpc_controller_ros2 send_nav_goal.py --x 5.0 --y 2.0 --yaw 0.0    # 대각선
ros2 run mpc_controller_ros2 send_nav_goal.py --x 0.0 --y 0.0 --yaw 3.14   # 원점 복귀
```

### 방법 2: RVIZ2에서 수동 설정

1. RVIZ2 상단 도구바에서 "2D Goal Pose" 버튼 클릭
2. 맵에서 목표 위치 클릭 후 드래그하여 방향 설정
3. 로봇이 자동으로 경로를 계획하고 추종

### 방법 3: Topic으로 직접 전송

```bash
ros2 topic pub /goal_pose geometry_msgs/msg/PoseStamped "{
  header: {frame_id: 'map'},
  pose: {
    position: {x: 5.0, y: 0.0, z: 0.0},
    orientation: {w: 1.0}
  }
}"
```

## 모니터링

### RVIZ2 확인 사항

1. **로봇 모델**: base_link, 바퀴, lidar
2. **Lidar 스캔**: 장애물 감지
3. **Costmap**:
   - 빨간색: 장애물
   - 노란색: Inflation zone
   - 파란색: 자유 공간
4. **계획된 경로**: 녹색 선
5. **MPPI 샘플 궤적**: 회색 반투명 선들
6. **최적 궤적**: 빨간색 선
7. **Reference 경로**: 주황색 선

### 터미널 모니터링

```bash
# MPPI 제어 출력 모니터링
ros2 topic echo /cmd_vel

# Odometry 확인
ros2 topic echo /odom

# Costmap 확인
ros2 topic echo /local_costmap/costmap

# TF tree 확인
ros2 run tf2_tools view_frames
```

### rqt_graph로 노드 연결 확인

```bash
rqt_graph
```

## MPPI 파라미터 튜닝

실시간 파라미터 변경:

```bash
# Temperature 파라미터 (탐색 vs 최적화 균형)
ros2 param set /controller_server FollowPath.lambda 15.0

# 장애물 회피 강도
ros2 param set /controller_server FollowPath.obstacle_weight 300.0
ros2 param set /controller_server FollowPath.costmap_lethal_cost 5000.0
ros2 param set /controller_server FollowPath.costmap_critical_cost 500.0

# Goal approach 튜닝
ros2 param set /controller_server FollowPath.min_lookahead 0.5
ros2 param set /controller_server FollowPath.goal_slowdown_dist 0.5

# Heading 유지 강화 (swerve용)
ros2 param set /controller_server FollowPath.R_omega 0.8

# Collision debug 실시간 활성화
ros2 param set /controller_server FollowPath.debug_collision_viz true

# 현재 파라미터 확인
ros2 param list /controller_server | grep FollowPath
```

### 모델별 YAML 설정 파일

| 모델 | MPPI YAML | 공통 YAML |
|------|-----------|-----------|
| DiffDrive | `config/nav2_params.yaml` | (내장) |
| Swerve | `config/nav2_params_swerve_mppi.yaml` | `config/nav2_params_swerve.yaml` |
| NonCoaxial | `config/nav2_params_non_coaxial_mppi.yaml` | `config/nav2_params_swerve.yaml` |

### Swerve MPPI 튜닝 가이드

Swerve(홀로노믹) 모델은 vx/vy/omega 3축 제어로 DiffDrive보다 복잡한 튜닝이 필요합니다.

**핵심 파라미터:**

| 파라미터 | 기본값 | 설명 |
|----------|--------|------|
| `v_min` | -0.5 | **0.0으로 설정 금지** — Spline-MPPI 비대칭 clipping bias 발생 |
| `lookahead_dist` | 1.5 | **0(auto)은 비권장** — auto=v_max*N*dt가 도달불가 거리 생성 |
| `target_ess_ratio` | 0.2 | 낮을수록 상위 샘플에 가중치 집중 (0.5는 너무 균등) |
| `noise_sigma_vy` | 0.2 | vy 노이즈 과다 시 측면 오실레이션 유발 |
| `R_vy` | 1.0 | vy 제어 비용 — 낮으면 불필요한 측면 이동 발생 |
| `control_smoothing_alpha` | 0.5 | EMA 필터 강도 (0=이전유지, 1=필터OFF) |

**주의사항:**

```
┌─ Swerve MPPI에서 vx≡0이 발생하는 경우 ─────────────────────┐
│                                                             │
│  ① v_min=0.0 → 비대칭 clipping → MPPI 업데이트 bias       │
│  ② lookahead=auto(4.5m) → 모든 샘플 도달 불가 → 비용 포화  │
│  ③ target_ess_ratio=0.5 → 가중치 균등 → 업데이트 ≈ 0       │
│                                                             │
│  해결: v_min=-0.5, lookahead=1.5, target_ess=0.2            │
└─────────────────────────────────────────────────────────────┘
```

### cmd_vel 오실레이션 분석

```bash
# 1. cmd_vel 녹화 (45초)
python3 /tmp/record_cmd_vel.py

# 2. goal 전송 (별도 터미널)
ros2 run mpc_controller_ros2 send_nav_goal.py --x 3.0 --y 0.0 --yaw 0.0

# 3. 분석 (녹화 완료 후)
python3 /tmp/analyze_cmd_vel.py
```

주요 지표:
- **부호 전환율**: < 15% 양호, > 25% 오실레이션 심각
- **제어 변화율**: dvx/dvy mean < 0.05 양호
- **ESS**: < 20%가 최적 (RCLCPP_DEBUG로 확인)

### Spline-MPPI 디버그

```bash
# 컨트롤러 로그 레벨 DEBUG로 변경 (u_opt, min_cost, ESS, knots_vx, ref0 출력)
ros2 service call /controller_server/set_logger_level \
  rcl_interfaces/srv/SetLoggerLevel "{logger_name: 'controller_server', level: 10}"
```

## 테스트 시나리오

### 1. 직선 주행 테스트

```bash
ros2 run mpc_controller_ros2 send_nav_goal.py --x 5.0 --y 0.0 --yaw 0.0
```

예상 결과:
- 로봇이 직선으로 5m 전진
- MPPI 샘플들이 reference 경로 주변에 분포
- 장애물 없으면 부드러운 가속/감속

### 2. 장애물 회피 테스트

```bash
# World에 장애물이 있는 위치로 이동
ros2 run mpc_controller_ros2 send_nav_goal.py --x 3.0 --y 2.0 --yaw 0.0
```

예상 결과:
- Costmap에 장애물 표시
- MPPI가 장애물을 우회하는 궤적 생성
- 샘플 궤적들이 장애물 주변에서 분산

### 3. 회전 + 이동 테스트

```bash
ros2 run mpc_controller_ros2 send_nav_goal.py --x 5.0 --y 5.0 --yaw 1.57
```

예상 결과:
- 목표 방향으로 회전 후 이동
- 경로 추종 정확도 확인

### 4. 좁은 통로 주행 테스트

```bash
# World의 wall 사이로 통과
ros2 run mpc_controller_ros2 send_nav_goal.py --x 10.0 --y 0.0 --yaw 0.0
```

예상 결과:
- 좁은 공간에서 속도 감소
- 안전 거리 유지하며 통과

## 문제 해결

### 1. "nav2 action server를 찾을 수 없습니다"

확인 사항:
```bash
# bt_navigator 노드 확인
ros2 node list | grep bt_navigator

# lifecycle 상태 확인
ros2 lifecycle list /bt_navigator
ros2 lifecycle get /bt_navigator

# 필요시 활성화
ros2 lifecycle set /bt_navigator configure
ros2 lifecycle set /bt_navigator activate
```

### 2. 로봇이 움직이지 않음

확인 사항:
```bash
# cmd_vel 토픽 확인
ros2 topic echo /cmd_vel

# MPPI 컨트롤러 로그 확인
ros2 node info /controller_server

# TF 확인
ros2 run tf2_ros tf2_echo map base_link
```

### 3. Costmap이 비어있음

확인 사항:
```bash
# Lidar 데이터 확인
ros2 topic echo /scan

# Costmap 업데이트 확인
ros2 topic hz /local_costmap/costmap
```

### 4. MPPI 샘플이 보이지 않음

파라미터 확인:
```bash
ros2 param get /controller_server FollowPath.visualize_samples
# true로 설정되어 있어야 함

# RVIZ에서 MarkerArray 토픽 추가
# Topic: /mpc_markers
```

## 성능 벤치마크

예상 성능 (K=512, N=30):
- **제어 주파수**: 20 Hz
- **계산 시간**: < 50ms/iteration
- **경로 추종 오차**: < 0.3m (RMSE)
- **장애물 회피**: 안전 거리 > 0.6m 유지

## 파일 위치

- Launch 파일: `launch/mppi_ros2_control_nav2.launch.py`
- nav2 파라미터 (DiffDrive): `config/nav2_params.yaml`
- nav2 파라미터 (Swerve): `config/nav2_params_swerve.yaml` + `config/nav2_params_swerve_mppi.yaml`
- nav2 파라미터 (NonCoaxial): `config/nav2_params_swerve.yaml` + `config/nav2_params_non_coaxial_mppi.yaml`
- 플러그인 XML: `plugins/mppi_controller_plugin.xml`
- Goal 전송 스크립트: `scripts/send_nav_goal.py`
- 로봇 URDF: `urdf/swerve_robot.urdf`
- World 파일: `worlds/mppi_test_simple.world`

## 다음 단계

1. ✅ Gazebo + nav2 + MPPI 통합 완료
2. ✅ 고급 MPPI 8종 플러그인 (M3/M3.5/M5 완료)
3. ✅ MotionModel 추상화 (DiffDrive/Swerve/NonCoaxial)
4. ✅ Goal 수렴 + 장애물 회피 튜닝
5. ✅ Swerve 오실레이션 진단 + MPPI 옵티마이저 수렴 수정
6. 🔄 실제 로봇 테스트
7. 📊 GPU 가속 (M2 잔여)
