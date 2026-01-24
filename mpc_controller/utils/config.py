"""
Configuration Management Module.

YAML 설정 파일을 로드하고 관리합니다.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional
import yaml


@dataclass
class RobotConfig:
    """로봇 물리 파라미터 설정."""

    wheel_base: float = 0.5
    wheel_radius: float = 0.1
    track_width: float = 0.4
    max_linear_velocity: float = 2.0
    max_angular_velocity: float = 2.0
    max_linear_acceleration: float = 1.0
    max_angular_acceleration: float = 2.0


@dataclass
class MPCConfig:
    """MPC 컨트롤러 설정."""

    prediction_horizon: int = 20
    control_horizon: int = 10
    dt: float = 0.1

    # 비용 함수 가중치
    position_weight: float = 10.0
    orientation_weight: float = 5.0
    velocity_weight: float = 1.0
    control_effort_weight: float = 0.1
    terminal_weight: float = 100.0

    # 솔버 설정
    solver_type: str = "ipopt"
    max_iterations: int = 100
    tolerance: float = 1e-6


@dataclass
class SimulationConfig:
    """시뮬레이션 설정."""

    dt: float = 0.01
    max_time: float = 60.0

    # 초기 상태
    initial_x: float = 0.0
    initial_y: float = 0.0
    initial_theta: float = 0.0

    # 노이즈 설정
    noise_enabled: bool = False
    position_noise_std: float = 0.01
    orientation_noise_std: float = 0.01


@dataclass
class ObstacleConfig:
    """장애물 회피 설정."""

    enabled: bool = True
    safety_margin: float = 0.3
    detection_range: float = 5.0


@dataclass
class ProjectConfig:
    """전체 프로젝트 설정."""

    robot: RobotConfig = field(default_factory=RobotConfig)
    mpc: MPCConfig = field(default_factory=MPCConfig)
    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    obstacle: ObstacleConfig = field(default_factory=ObstacleConfig)


class ConfigManager:
    """
    설정 관리 클래스.

    YAML 파일에서 설정을 로드하고 관리합니다.
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration manager.

        Args:
            config_path: Path to configuration YAML file
        """
        self._config: ProjectConfig = ProjectConfig()

        if config_path:
            self.load(config_path)

    @property
    def config(self) -> ProjectConfig:
        """현재 설정을 반환합니다."""
        return self._config

    @property
    def robot(self) -> RobotConfig:
        """로봇 설정을 반환합니다."""
        return self._config.robot

    @property
    def mpc(self) -> MPCConfig:
        """MPC 설정을 반환합니다."""
        return self._config.mpc

    @property
    def simulation(self) -> SimulationConfig:
        """시뮬레이션 설정을 반환합니다."""
        return self._config.simulation

    @property
    def obstacle(self) -> ObstacleConfig:
        """장애물 설정을 반환합니다."""
        return self._config.obstacle

    def load(self, config_path: str) -> None:
        """
        YAML 파일에서 설정을 로드합니다.

        Args:
            config_path: Path to configuration file
        """
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        self._parse_config(data)

    def _parse_config(self, data: dict) -> None:
        """설정 데이터를 파싱합니다."""
        if "robot" in data:
            robot_data = data["robot"]
            self._config.robot = RobotConfig(
                wheel_base=robot_data.get("wheel_base", 0.5),
                wheel_radius=robot_data.get("wheel_radius", 0.1),
                track_width=robot_data.get("track_width", 0.4),
                max_linear_velocity=robot_data.get("max_linear_velocity", 2.0),
                max_angular_velocity=robot_data.get("max_angular_velocity", 2.0),
                max_linear_acceleration=robot_data.get("max_linear_acceleration", 1.0),
                max_angular_acceleration=robot_data.get("max_angular_acceleration", 2.0),
            )

        if "mpc" in data:
            mpc_data = data["mpc"]
            weights = mpc_data.get("weights", {})
            solver = mpc_data.get("solver", {})

            self._config.mpc = MPCConfig(
                prediction_horizon=mpc_data.get("prediction_horizon", 20),
                control_horizon=mpc_data.get("control_horizon", 10),
                dt=mpc_data.get("dt", 0.1),
                position_weight=weights.get("position", 10.0),
                orientation_weight=weights.get("orientation", 5.0),
                velocity_weight=weights.get("velocity", 1.0),
                control_effort_weight=weights.get("control_effort", 0.1),
                terminal_weight=weights.get("terminal", 100.0),
                solver_type=solver.get("type", "ipopt"),
                max_iterations=solver.get("max_iterations", 100),
                tolerance=solver.get("tolerance", 1e-6),
            )

        if "simulation" in data:
            sim_data = data["simulation"]
            initial = sim_data.get("initial_state", {})
            noise = sim_data.get("noise", {})

            self._config.simulation = SimulationConfig(
                dt=sim_data.get("dt", 0.01),
                max_time=sim_data.get("max_time", 60.0),
                initial_x=initial.get("x", 0.0),
                initial_y=initial.get("y", 0.0),
                initial_theta=initial.get("theta", 0.0),
                noise_enabled=noise.get("enabled", False),
                position_noise_std=noise.get("position_std", 0.01),
                orientation_noise_std=noise.get("orientation_std", 0.01),
            )

        if "obstacle_avoidance" in data:
            obs_data = data["obstacle_avoidance"]
            self._config.obstacle = ObstacleConfig(
                enabled=obs_data.get("enabled", True),
                safety_margin=obs_data.get("safety_margin", 0.3),
                detection_range=obs_data.get("detection_range", 5.0),
            )

    def save(self, config_path: str) -> None:
        """
        현재 설정을 YAML 파일로 저장합니다.

        Args:
            config_path: Path to save configuration
        """
        data = {
            "robot": {
                "wheel_base": self._config.robot.wheel_base,
                "wheel_radius": self._config.robot.wheel_radius,
                "track_width": self._config.robot.track_width,
                "max_linear_velocity": self._config.robot.max_linear_velocity,
                "max_angular_velocity": self._config.robot.max_angular_velocity,
                "max_linear_acceleration": self._config.robot.max_linear_acceleration,
                "max_angular_acceleration": self._config.robot.max_angular_acceleration,
            },
            "mpc": {
                "prediction_horizon": self._config.mpc.prediction_horizon,
                "control_horizon": self._config.mpc.control_horizon,
                "dt": self._config.mpc.dt,
                "weights": {
                    "position": self._config.mpc.position_weight,
                    "orientation": self._config.mpc.orientation_weight,
                    "velocity": self._config.mpc.velocity_weight,
                    "control_effort": self._config.mpc.control_effort_weight,
                    "terminal": self._config.mpc.terminal_weight,
                },
                "solver": {
                    "type": self._config.mpc.solver_type,
                    "max_iterations": self._config.mpc.max_iterations,
                    "tolerance": self._config.mpc.tolerance,
                },
            },
            "simulation": {
                "dt": self._config.simulation.dt,
                "max_time": self._config.simulation.max_time,
                "initial_state": {
                    "x": self._config.simulation.initial_x,
                    "y": self._config.simulation.initial_y,
                    "theta": self._config.simulation.initial_theta,
                },
                "noise": {
                    "enabled": self._config.simulation.noise_enabled,
                    "position_std": self._config.simulation.position_noise_std,
                    "orientation_std": self._config.simulation.orientation_noise_std,
                },
            },
            "obstacle_avoidance": {
                "enabled": self._config.obstacle.enabled,
                "safety_margin": self._config.obstacle.safety_margin,
                "detection_range": self._config.obstacle.detection_range,
            },
        }

        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(data, f, default_flow_style=False, allow_unicode=True)

    def get(self, key: str, default: Any = None) -> Any:
        """
        점 표기법으로 설정 값을 가져옵니다.

        Args:
            key: Configuration key (e.g., "robot.wheel_base")
            default: Default value if key not found

        Returns:
            Configuration value
        """
        keys = key.split(".")
        value = self._config

        try:
            for k in keys:
                value = getattr(value, k)
            return value
        except AttributeError:
            return default


# 전역 설정 인스턴스
_config_manager: Optional[ConfigManager] = None


def get_config() -> ConfigManager:
    """전역 설정 관리자를 반환합니다."""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager


def load_config(config_path: str) -> ConfigManager:
    """
    설정 파일을 로드하고 전역 설정 관리자를 반환합니다.

    Args:
        config_path: Path to configuration file

    Returns:
        ConfigManager instance
    """
    global _config_manager
    _config_manager = ConfigManager(config_path)
    return _config_manager
