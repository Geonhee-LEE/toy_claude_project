#!/usr/bin/env python3
"""Nav2 lifecycle bringup — bond-free lifecycle manager.

nav2_lifecycle_manager의 service call이 비결정적으로 실패하는 문제를 우회:
  1. 각 노드에 bond_timeout=0.0 설정 → bond 비활성화 (self-deactivation 방지)
  2. 일부 노드는 autostart() 내장 메커니즘으로 자동 활성화
  3. 자동 활성화 실패 시 `ros2 lifecycle set` subprocess로 수동 전환 (재시도)
  4. 주기적으로 노드 상태 모니터링 → crash 시 재활성화

사용법:
    ros2 run mpc_controller_ros2 nav2_lifecycle_bringup.py \
        --ros-args -p node_names:="map_server,amcl,controller_server,..."
"""

import subprocess
import threading
import time

import rclpy
from rclpy.node import Node


class Nav2LifecycleBringup(Node):
    """Nav2 lifecycle bringup — bond-free lifecycle manager."""

    def __init__(self):
        super().__init__('nav2_lifecycle_bringup')

        # Parameters
        self.declare_parameter('node_names', '')
        self.declare_parameter('check_interval', 5.0)
        self.declare_parameter('max_retries', 10)

        names_str = self.get_parameter('node_names').value
        self.node_names = [n.strip() for n in names_str.split(',') if n.strip()]
        self.check_interval = self.get_parameter('check_interval').value
        self.max_retries = self.get_parameter('max_retries').value

        if not self.node_names:
            self.get_logger().error('No node_names provided!')
            return

        self.get_logger().info(f'Managing {len(self.node_names)} nodes: {self.node_names}')
        self.get_logger().info('Bond-free mode: bond_timeout=0.0 on all managed nodes')

        # Lifecycle monitor thread
        self.active_nodes = set()
        self.retry_count = {}  # node_name -> count
        self.monitor_thread = threading.Thread(
            target=self._monitor_loop, daemon=True
        )
        self.monitor_thread.start()

    def _get_node_state(self, node_name):
        """노드의 현재 lifecycle 상태를 조회."""
        try:
            result = subprocess.run(
                ['ros2', 'lifecycle', 'get', f'/{node_name}'],
                capture_output=True, text=True, timeout=5,
            )
            output = result.stdout.strip().lower()
            if 'active' in output:
                return 'active'
            elif 'inactive' in output:
                return 'inactive'
            elif 'unconfigured' in output:
                return 'unconfigured'
            elif 'finalized' in output:
                return 'finalized'
            return 'unknown'
        except Exception:
            return 'unknown'

    def _lifecycle_set(self, node_name, transition):
        """단일 lifecycle 전환 시도."""
        try:
            result = subprocess.run(
                ['ros2', 'lifecycle', 'set', f'/{node_name}', transition],
                capture_output=True, text=True, timeout=30,
            )
            if 'Transitioning successful' in result.stdout:
                return True
        except Exception:
            pass
        return False

    def _activate_node(self, name):
        """노드 상태에 따라 configure + activate 수행."""
        state = self._get_node_state(name)

        if state == 'active':
            if name not in self.active_nodes:
                self.get_logger().info(f'{name}: active')
                self.active_nodes.add(name)
            return True

        retries = self.retry_count.get(name, 0)
        if retries >= self.max_retries:
            return False

        if state == 'unconfigured':
            self.get_logger().info(f'{name}: unconfigured → configuring...')
            if self._lifecycle_set(name, 'configure'):
                self.get_logger().info(f'{name}: configure OK')
                time.sleep(0.5)
                if self._lifecycle_set(name, 'activate'):
                    self.get_logger().info(f'{name}: activate OK')
                    self.active_nodes.add(name)
                    return True
                else:
                    self.retry_count[name] = retries + 1
            else:
                # autostart race: configure 실패 시 이미 active일 수 있음
                state2 = self._get_node_state(name)
                if state2 == 'active':
                    self.get_logger().info(f'{name}: active (autostarted)')
                    self.active_nodes.add(name)
                    return True
                else:
                    self.retry_count[name] = retries + 1

        elif state == 'inactive':
            self.get_logger().info(f'{name}: inactive → activating...')
            if self._lifecycle_set(name, 'activate'):
                self.get_logger().info(f'{name}: activate OK')
                self.active_nodes.add(name)
                return True
            else:
                self.retry_count[name] = retries + 1

        elif state == 'unknown':
            pass  # 노드가 아직 시작되지 않음

        return False

    def _monitor_loop(self):
        """주기적으로 노드 상태를 확인하고 필요 시 lifecycle 전환."""
        # 초기 대기: 노드들이 autostart할 시간
        time.sleep(8.0)

        all_active_logged = False

        while rclpy.ok():
            # 1. 미활성 노드 활성화 시도
            for name in self.node_names:
                if name not in self.active_nodes:
                    self._activate_node(name)

            # 2. 전체 활성 상태 확인
            if len(self.active_nodes) == len(self.node_names):
                if not all_active_logged:
                    self.get_logger().info(
                        f'All {len(self.active_nodes)}/{len(self.node_names)} nodes active!'
                    )
                    all_active_logged = True

            time.sleep(self.check_interval)

            # 3. active 노드의 상태 확인 (crash 감지)
            for name in list(self.active_nodes):
                state = self._get_node_state(name)
                if state != 'active' and state != 'unknown':
                    self.get_logger().warn(
                        f'{name}: was active, now {state} — will re-activate next cycle'
                    )
                    self.active_nodes.discard(name)
                    self.retry_count[name] = 0  # 재시도 카운터 리셋
                    all_active_logged = False


def main():
    rclpy.init()
    node = Nav2LifecycleBringup()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
