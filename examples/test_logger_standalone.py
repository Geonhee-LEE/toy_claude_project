#!/usr/bin/env python3
"""Logger 유틸리티 독립 실행 데모 스크립트.

이 스크립트는 logger 유틸리티의 기능을 시연합니다.
mpc_controller 패키지를 임포트하지 않고 logger 모듈만 직접 로드합니다.
"""

import sys
import os
import logging

# logger.py 파일을 직접 실행
logger_path = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'mpc_controller', 'utils', 'logger.py'
)

# logger 모듈을 동적으로 로드
import importlib.util
spec = importlib.util.spec_from_file_location("logger", logger_path)
logger_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(logger_module)

# 모듈에서 함수들 가져오기
setup_logger = logger_module.setup_logger
get_logger = logger_module.get_logger


def main():
    """메인 함수."""
    print("=" * 60)
    print("Logger 유틸리티 데모")
    print("=" * 60)
    print()

    # 1. 기본 로거 생성
    print("1. 기본 로거 (INFO 레벨, 컬러 출력)")
    print("-" * 60)
    logger1 = setup_logger("demo_logger_1")
    logger1.debug("이 메시지는 표시되지 않습니다 (DEBUG < INFO)")
    logger1.info("정보 메시지 - 애플리케이션 시작")
    logger1.warning("경고 메시지 - 리소스 사용량 높음")
    logger1.error("에러 메시지 - 파일을 찾을 수 없음")
    logger1.critical("치명적 에러 - 시스템 종료 필요")
    print()

    # 2. DEBUG 레벨 로거
    print("2. DEBUG 레벨 로거")
    print("-" * 60)
    logger2 = setup_logger("demo_logger_2", level=logging.DEBUG)
    logger2.debug("디버그 메시지 - 변수 값 확인")
    logger2.info("정보 메시지 - 데이터 로딩 완료")
    print()

    # 3. 컬러 없는 로거
    print("3. 컬러 없는 로거")
    print("-" * 60)
    logger3 = setup_logger("demo_logger_3", use_color=False)
    logger3.info("컬러 없는 정보 메시지")
    logger3.warning("컬러 없는 경고 메시지")
    logger3.error("컬러 없는 에러 메시지")
    print()

    # 4. 커스텀 포맷 로거
    print("4. 커스텀 포맷 로거 (간단한 포맷)")
    print("-" * 60)
    logger4 = setup_logger(
        "demo_logger_4",
        log_format='%(levelname)s: %(message)s'
    )
    logger4.info("간단한 포맷의 메시지")
    logger4.warning("포맷에 시간과 이름이 없습니다")
    print()

    # 5. 기존 로거 가져오기
    print("5. 기존 로거 가져오기")
    print("-" * 60)
    logger5 = get_logger("demo_logger_1")  # 위에서 생성한 logger1
    logger5.info("기존 로거 재사용")
    print()

    # 6. 편의 함수 사용
    print("6. 편의 함수 사용 (기본 로거)")
    print("-" * 60)
    debug = logger_module.debug
    info_func = logger_module.info
    warning_func = logger_module.warning
    error_func = logger_module.error
    critical_func = logger_module.critical

    info_func("편의 함수로 정보 메시지 출력")
    warning_func("편의 함수로 경고 메시지 출력")
    error_func("편의 함수로 에러 메시지 출력")
    print()

    # 7. 실제 사용 예시
    print("7. 실제 사용 예시 - MPC Controller 시뮬레이션")
    print("-" * 60)
    robot_logger = setup_logger("RobotController", level=logging.DEBUG)

    robot_logger.info("로봇 컨트롤러 초기화 중...")
    robot_logger.debug("설정 파일 로딩: config.yaml")
    robot_logger.info("MPC 파라미터 설정 완료")
    robot_logger.debug("제어 주기: 0.1초, 예측 구간: 10단계")
    robot_logger.warning("장애물 감지 - 경로 재계획 필요")
    robot_logger.info("새로운 경로 생성 완료")
    robot_logger.debug("목표 지점까지 거리: 5.2m")
    robot_logger.info("목표 지점 도달 완료!")
    print()

    print("=" * 60)
    print("데모 완료!")
    print("=" * 60)


if __name__ == "__main__":
    main()
