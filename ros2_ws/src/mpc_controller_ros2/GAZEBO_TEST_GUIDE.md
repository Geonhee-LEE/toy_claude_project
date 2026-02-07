# Gazebo MPPI 주행 테스트 가이드

## 시스템 구성

```
┌─────────────┐      ┌──────────────┐      ┌─────────────┐
│   Gazebo    │─────▶│ Controller   │─────▶│   Robot     │
│  (World +   │ scan │   Server     │cmd_vel│ (diff_drive)│
│   Robot)    │◀─────│   (MPPI)     │◀─────│             │
└─────────────┘ odom └──────────────┘ pose └─────────────┘
                         ▲
                         │ /plan
                    ┌────┴─────┐
                    │  Goal    │
                    │Publisher │
                    └──────────┘
```

## 1. 전체 시스템 실행 (권장)

### 터미널 1: Nav2 + Gazebo + MPPI 실행
```bash
cd ~/toy_claude_project/ros2_ws
source install/setup.bash
ros2 launch mpc_controller_ros2 mppi_navigation.launch.py
```

**실행 내용:**
- Gazebo 시뮬레이션 (world + 로봇)
- Controller Server (MPPI 플러그인)
- Planner Server
- Local Costmap
- RVIZ2
- TF: map → odom → base_link

**확인 사항:**
```bash
# 별도 터미널에서 확인
ros2 topic list | grep -E "(cmd_vel|scan|odom|plan)"
# 출력 예상:
# /cmd_vel
# /scan
# /odom
# /plan
```

### 터미널 2: 경로 전송 및 주행

#### 테스트 1: 직선 주행 (8m 전진)
```bash
cd ~/toy_claude_project/ros2_ws
source install/setup.bash
ros2 run mpc_controller_ros2 send_goal.py straight 8.0 0.0
```

**예상 동작:**
- 로봇이 (0,0)에서 (8,0)까지 직선 주행
- 장애물 회피 (obstacle_1, obstacle_2 근처)
- RVIZ에 궤적 시각화

#### 테스트 2: 곡선 주행 (90도 회전)
```bash
ros2 run mpc_controller_ros2 send_goal.py curve 3.0 90.0
```

**예상 동작:**
- 반지름 3m 원호 궤적
- 90도 회전

#### 테스트 3: 장애물 회피 경로
```bash
ros2 run mpc_controller_ros2 send_goal.py obstacle
```

**예상 동작:**
- S자 경로로 장애물 회피
- obstacle_1 (3,2), obstacle_2 (5,-1) 우회

### 터미널 3: 실시간 모니터링 (선택)

#### MPPI 정보 확인
```bash
# 제어 명령 확인
ros2 topic echo /cmd_vel

# Lidar 스캔 확인
ros2 topic echo /scan --once

# Odometry 확인
ros2 topic echo /odom

# MPPI 시각화 마커
ros2 topic echo /FollowPath/mppi_markers --once
```

#### 파라미터 실시간 변경
```bash
# Temperature 조정
ros2 param set /controller_server FollowPath.lambda 15.0

# 샘플 개수 (재시작 필요)
# ros2 param set /controller_server FollowPath.K 1024

# 노이즈 레벨 조정
ros2 param set /controller_server FollowPath.noise_sigma_v 0.5

# 속도 제한 조정
ros2 param set /controller_server FollowPath.v_max 0.8
```

## 2. 단계별 실행 (디버깅용)

### Step 1: Gazebo만 실행
```bash
ros2 launch mpc_controller_ros2 gazebo_mppi_test.launch.py
```

### Step 2: 수동 제어 테스트
```bash
# 직진
ros2 topic pub /cmd_vel geometry_msgs/msg/Twist \
  "{linear: {x: 0.3}, angular: {z: 0.0}}" --once

# 회전
ros2 topic pub /cmd_vel geometry_msgs/msg/Twist \
  "{linear: {x: 0.0}, angular: {z: 0.5}}" --once

# 정지
ros2 topic pub /cmd_vel geometry_msgs/msg/Twist \
  "{linear: {x: 0.0}, angular: {z: 0.0}}" --once
```

### Step 3: Controller Server만 실행
```bash
ros2 run nav2_controller controller_server \
  --ros-args --params-file install/mpc_controller_ros2/share/mpc_controller_ros2/config/nav2_params.yaml
```

## 3. RVIZ 시각화 설정

RVIZ가 자동으로 실행되지만, 수동으로 토픽 추가 가능:

### 필수 Display 추가
1. **RobotModel**: robot_description 토픽
2. **LaserScan**: /scan 토픽
3. **Odometry**: /odom 토픽
4. **Path**: /plan 토픽 (참조 경로)
5. **MarkerArray**: /FollowPath/mppi_markers 토픽 (MPPI 시각화)
   - mppi_reference (노랑)
   - mppi_weighted_avg (파랑)
   - mppi_best_trajectory (빨강)
   - mppi_samples (회색)
   - mppi_control_arrows (청록색)
   - mppi_text_info (흰색)

### Fixed Frame 설정
- Fixed Frame: `map`

## 4. 성능 평가

### 계산 시간 확인
```bash
# MPPI 텍스트 정보에 표시됨 (RVIZ)
# 또는 로그 확인
ros2 topic echo /rosout | grep "MPPI:"
```

### 경로 추적 오차 측정
```bash
# 실제 위치 vs 참조 경로 비교
ros2 topic echo /odom &
ros2 topic echo /plan
```

## 5. 트러블슈팅

### 문제 1: 로봇이 움직이지 않음
**확인:**
```bash
# Controller server 실행 상태
ros2 node list | grep controller

# /plan 토픽 수신 확인
ros2 topic info /plan

# cmd_vel 퍼블리시 확인
ros2 topic hz /cmd_vel
```

**해결:**
- Controller server 재시작
- 경로 재전송 (`send_goal.py` 다시 실행)

### 문제 2: Gazebo가 느림
**해결:**
```bash
# Real-time factor 확인 (Gazebo 하단)
# 1.0 이하면 정상, 0.5 이하면 느림

# 샘플 개수 줄이기
ros2 param set /controller_server FollowPath.K 256

# 시각화 끄기
ros2 param set /controller_server FollowPath.visualize_samples false
```

### 문제 3: RVIZ에 마커가 안 보임
**확인:**
```bash
# 마커 토픽 확인
ros2 topic echo /FollowPath/mppi_markers --once

# RVIZ Fixed Frame 확인 (map이어야 함)
```

## 6. 주요 파라미터 설명

### MPPI 파라미터 (`nav2_params.yaml`)
| 파라미터 | 기본값 | 설명 |
|---------|--------|------|
| N | 30 | 예측 시간 스텝 |
| dt | 0.1 | 시간 간격 (초) |
| K | 512 | 샘플 궤적 수 |
| lambda | 10.0 | Temperature (낮을수록 exploitation) |
| v_max | 0.5 | 최대 선속도 (m/s) |
| omega_max | 1.0 | 최대 각속도 (rad/s) |
| Q_x, Q_y | 15.0 | 위치 추적 가중치 |
| R_v, R_omega | 0.2 | 제어 입력 가중치 |
| obstacle_weight | 150.0 | 장애물 회피 가중치 |
| safety_distance | 0.6 | 안전 거리 (m) |

### Costmap 파라미터
| 파라미터 | 기본값 | 설명 |
|---------|--------|------|
| update_frequency | 5.0 | Costmap 업데이트 주파수 (Hz) |
| width, height | 10 | Costmap 크기 (m) |
| resolution | 0.05 | 해상도 (m/cell) |
| robot_radius | 0.3 | 로봇 반경 (m) |
| inflation_radius | 0.8 | Inflation 반경 (m) |

## 7. 예상 결과

### 성공 기준
- ✓ 로봇이 목표 지점까지 주행
- ✓ 장애물과 충돌하지 않음 (안전 거리 유지)
- ✓ 경로 추적 오차 < 0.3m
- ✓ 계산 시간 < 50ms (K=512 기준)
- ✓ RVIZ에 궤적 시각화

### 실행 예시 (비디오)
```bash
# 전체 시스템 실행
ros2 launch mpc_controller_ros2 mppi_navigation.launch.py

# 별도 터미널에서 직선 주행 명령
ros2 run mpc_controller_ros2 send_goal.py straight 8.0 0.0

# RVIZ에서 확인:
# - 노랑 선: 참조 경로
# - 파랑 선: 가중 평균 궤적
# - 빨강 선: 최적 샘플 궤적
# - 회색 선: 샘플 궤적들
# - 흰색 텍스트: ESS, λ, Cost, Time
```

## 8. 다음 단계

실행 결과를 바탕으로:
1. 경로 추적 정확도 측정
2. Python vs C++ 성능 비교
3. 다양한 시나리오 테스트
4. 파라미터 튜닝
