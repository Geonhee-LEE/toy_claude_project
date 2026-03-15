#!/bin/bash
# =============================================================
# swerve_controller_benchmark.sh
# Swerve 모델 기반 다양한 MPPI 컨트롤러 성능 벤치마크
#
# 사용법:
#   bash ros2_ws/src/mpc_controller_ros2/scripts/swerve_controller_benchmark.sh
#
# 각 컨트롤러별로:
#   1. launch 시작 (swerve URDF)
#   2. nav2 활성화 대기
#   3. E2E goal 전송 + 메트릭 수집
#   4. 정리 후 다음 컨트롤러
# =============================================================

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PKG_DIR="$(dirname "$SCRIPT_DIR")"
CLEANUP_SCRIPT="$SCRIPT_DIR/cleanup_launch.sh"
RESULTS_DIR="$HOME/swerve_benchmark_results"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_FILE="$RESULTS_DIR/benchmark_${TIMESTAMP}.txt"

# Swerve 호환 컨트롤러 목록 (장애물 회피 벤치마크)
# 이전 벤치마크에서 상위 성능 + 특성 다양한 8종 선정
CONTROLLERS=(
    "swerve"
    "non_coaxial"
    "dial_swerve"
    "log_swerve"
    "smooth_swerve"
    "biased_swerve"
    "cs_swerve"
    "pi_swerve"
)

# 목표 지점: barrier 2 (y=-2, gap at x=5) 통과 필수 경로
# S(0,0) → barrier2 gap(x=5, y=-2) → G(5,-3.5)
# 최소 1개 narrow passage (0.8m 폭) 통과 + pillar_2 (x=5, y=-3.5) 근처
GOAL_X=5.0
GOAL_Y=-3.5
GOAL_YAW=0.0
TIMEOUT=120
LAUNCH_WAIT=35    # launch 후 안정화 대기 (초)
NAV2_WAIT=30      # nav2 lifecycle 활성화 대기 (초)

# World/Map 설정
WORLD="narrow_passage_world.world"
MAP="narrow_passage_map.yaml"

mkdir -p "$RESULTS_DIR"

echo "============================================================"
echo "  Swerve Controller Benchmark"
echo "  $(date)"
echo "  World: $WORLD"
echo "  Goal: ($GOAL_X, $GOAL_Y, yaw=$GOAL_YAW)"
echo "  Controllers: ${CONTROLLERS[*]}"
echo "============================================================"
echo ""

# 결과 헤더
cat > "$RESULTS_FILE" << 'HEADER'
============================================================
  Swerve Controller Benchmark Results
============================================================
HEADER
echo "Date: $(date)" >> "$RESULTS_FILE"
echo "World: $WORLD" >> "$RESULTS_FILE"
echo "Goal: ($GOAL_X, $GOAL_Y, yaw=$GOAL_YAW)" >> "$RESULTS_FILE"
echo "" >> "$RESULTS_FILE"

# Source ROS2
source /opt/ros/jazzy/setup.bash
source /home/geonhee/toy_claude_project/ros2_ws/install/setup.bash

run_single_test() {
    local ctrl="$1"
    local idx="$2"
    local total="$3"

    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  [$idx/$total] Testing: $ctrl"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    # 1. 기존 프로세스 정리
    echo "[cleanup] Cleaning up previous processes..."
    bash "$CLEANUP_SCRIPT" 2>/dev/null || true
    sleep 5

    # 2. Launch 시작
    echo "[launch] Starting controller=$ctrl world=$WORLD ..."
    ros2 launch mpc_controller_ros2 mppi_ros2_control_nav2.launch.py \
        controller:="$ctrl" \
        world:="$WORLD" \
        map:="$MAP" \
        headless:=true \
        nav2_stress:="${NAV2_STRESS:-false}" \
        2>&1 | tee "/tmp/launch_${ctrl}.log" &
    LAUNCH_PID=$!

    echo "[launch] PID=$LAUNCH_PID, waiting ${LAUNCH_WAIT}s for stabilization..."
    sleep "$LAUNCH_WAIT"

    # launch가 살아있는지 확인
    if ! kill -0 "$LAUNCH_PID" 2>/dev/null; then
        echo "[ERROR] Launch crashed for $ctrl!"
        echo "[$ctrl] LAUNCH_FAILED" >> "$RESULTS_FILE"
        return 1
    fi

    # 3. nav2 활성화 대기
    echo "[nav2] Waiting ${NAV2_WAIT}s for nav2 lifecycle activation..."
    sleep "$NAV2_WAIT"

    # bt_navigator 활성 확인 (goal 수락 가능 상태)
    local nav2_ready=false
    for i in $(seq 1 10); do
        local bt_state=$(ros2 lifecycle get /bt_navigator 2>/dev/null)
        local ctrl_state=$(ros2 lifecycle get /controller_server 2>/dev/null)
        echo "[nav2] Check $i/10: bt_navigator=$bt_state, controller_server=$ctrl_state"
        if echo "$bt_state" | grep -q "active" && echo "$ctrl_state" | grep -q "active"; then
            nav2_ready=true
            break
        fi
        sleep 5
    done

    if [ "$nav2_ready" = false ]; then
        echo "[ERROR] nav2 not fully active for $ctrl!"
        echo "[$ctrl] NAV2_NOT_ACTIVE" >> "$RESULTS_FILE"
        kill "$LAUNCH_PID" 2>/dev/null || true
        sleep 3
        bash "$CLEANUP_SCRIPT" 2>/dev/null || true
        return 1
    fi

    echo "[nav2] bt_navigator + controller_server active ✓"
    # 추가 안정화 대기 (amcl map 수신 등)
    sleep 10

    # 4. E2E 테스트 실행
    echo "[test] Running E2E test (goal: $GOAL_X, $GOAL_Y, timeout: ${TIMEOUT}s)..."
    local test_output
    test_output=$(ros2 run mpc_controller_ros2 swerve_e2e_test.py \
        --x "$GOAL_X" --y "$GOAL_Y" --yaw "$GOAL_YAW" \
        --timeout "$TIMEOUT" --no-save 2>&1) || true

    echo "$test_output"

    # 5. 결과 파싱 및 저장
    echo "" >> "$RESULTS_FILE"
    echo "━━━ $ctrl ━━━" >> "$RESULTS_FILE"
    echo "$test_output" | grep -E "(Goal reached|Travel time|Travel distance|Position error|Yaw error|Mean.*:|Max.*:|Jerk.*:|Stall.*:|Min distance|Mean min|Near misses|Collisions)" >> "$RESULTS_FILE" 2>/dev/null || true

    # 6. 정리
    echo "[cleanup] Shutting down $ctrl..."
    kill "$LAUNCH_PID" 2>/dev/null || true
    sleep 3
    bash "$CLEANUP_SCRIPT" 2>/dev/null || true
    sleep 5

    echo "[done] $ctrl complete"
}

# 메인 루프
TOTAL=${#CONTROLLERS[@]}
IDX=0
for ctrl in "${CONTROLLERS[@]}"; do
    IDX=$((IDX + 1))
    run_single_test "$ctrl" "$IDX" "$TOTAL" || true
done

# 최종 요약
echo ""
echo "============================================================"
echo "  Benchmark Complete!"
echo "  Results: $RESULTS_FILE"
echo "============================================================"
echo ""
cat "$RESULTS_FILE"
