"""로깅 유틸리티 모듈.

이 모듈은 컬러 출력을 지원하는 로깅 유틸리티를 제공합니다.
"""

import logging
import sys
from typing import Optional


class ColoredFormatter(logging.Formatter):
    """컬러 출력을 지원하는 로그 포매터.

    ANSI 이스케이프 코드를 사용하여 터미널에 컬러 로그를 출력합니다.
    """

    # ANSI 컬러 코드
    COLORS = {
        'DEBUG': '\033[36m',      # Cyan (하늘색)
        'INFO': '\033[32m',       # Green (초록색)
        'WARNING': '\033[33m',    # Yellow (노란색)
        'ERROR': '\033[31m',      # Red (빨간색)
        'CRITICAL': '\033[35m',   # Magenta (자홍색)
    }
    RESET = '\033[0m'             # Reset (색상 초기화)
    BOLD = '\033[1m'              # Bold (굵게)

    def format(self, record: logging.LogRecord) -> str:
        """로그 레코드를 포맷팅합니다.

        Args:
            record: 로그 레코드 객체

        Returns:
            str: 포맷팅된 로그 메시지
        """
        # 로그 레벨에 따른 컬러 적용
        levelname = record.levelname
        if levelname in self.COLORS:
            # 로그 레벨을 컬러로 표시
            colored_levelname = (
                f"{self.BOLD}{self.COLORS[levelname]}{levelname}{self.RESET}"
            )
            record.levelname = colored_levelname

        # 기본 포맷팅 적용
        return super().format(record)


def setup_logger(
    name: str = __name__,
    level: int = logging.INFO,
    use_color: bool = True,
    log_format: Optional[str] = None,
) -> logging.Logger:
    """로거를 설정합니다.

    Args:
        name: 로거 이름 (기본값: 현재 모듈 이름)
        level: 로그 레벨 (기본값: INFO)
        use_color: 컬러 출력 사용 여부 (기본값: True)
        log_format: 로그 포맷 문자열 (기본값: None, 기본 포맷 사용)

    Returns:
        logging.Logger: 설정된 로거 객체

    Examples:
        >>> logger = setup_logger("my_app")
        >>> logger.info("애플리케이션 시작")
        >>> logger.debug("디버그 메시지")
        >>> logger.warning("경고 메시지")
        >>> logger.error("에러 메시지")
    """
    # 로거 생성
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # 핸들러가 이미 있으면 중복 추가 방지
    if logger.handlers:
        return logger

    # 콘솔 핸들러 생성
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)

    # 로그 포맷 설정
    if log_format is None:
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

    # 포매터 생성 및 적용
    if use_color:
        formatter = ColoredFormatter(log_format, datefmt='%Y-%m-%d %H:%M:%S')
    else:
        formatter = logging.Formatter(log_format, datefmt='%Y-%m-%d %H:%M:%S')

    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger


def get_logger(name: str = __name__) -> logging.Logger:
    """기존 로거를 가져오거나 새로 생성합니다.

    Args:
        name: 로거 이름 (기본값: 현재 모듈 이름)

    Returns:
        logging.Logger: 로거 객체

    Examples:
        >>> logger = get_logger("my_module")
        >>> logger.info("모듈 초기화 완료")
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        return setup_logger(name)
    return logger


# 기본 로거 생성
default_logger = setup_logger("mpc_controller")


def debug(msg: str, *args, **kwargs) -> None:
    """디버그 메시지를 출력합니다."""
    default_logger.debug(msg, *args, **kwargs)


def info(msg: str, *args, **kwargs) -> None:
    """정보 메시지를 출력합니다."""
    default_logger.info(msg, *args, **kwargs)


def warning(msg: str, *args, **kwargs) -> None:
    """경고 메시지를 출력합니다."""
    default_logger.warning(msg, *args, **kwargs)


def error(msg: str, *args, **kwargs) -> None:
    """에러 메시지를 출력합니다."""
    default_logger.error(msg, *args, **kwargs)


def critical(msg: str, *args, **kwargs) -> None:
    """치명적인 에러 메시지를 출력합니다."""
    default_logger.critical(msg, *args, **kwargs)
