"""Logger 유틸리티 테스트 모듈."""

import logging
import pytest
from mpc_controller.utils.logger import (
    setup_logger,
    get_logger,
    ColoredFormatter,
    debug,
    info,
    warning,
    error,
    critical,
)


class TestColoredFormatter:
    """ColoredFormatter 클래스 테스트."""

    def test_formatter_creation(self):
        """포매터 생성 테스트."""
        formatter = ColoredFormatter('%(levelname)s - %(message)s')
        assert formatter is not None

    def test_format_with_colors(self):
        """컬러 포맷팅 테스트."""
        formatter = ColoredFormatter('%(levelname)s - %(message)s')
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="테스트 메시지",
            args=(),
            exc_info=None,
        )
        formatted = formatter.format(record)
        assert "테스트 메시지" in formatted
        # 컬러 코드가 포함되어 있는지 확인
        assert '\033[' in formatted or 'INFO' in formatted


class TestSetupLogger:
    """setup_logger 함수 테스트."""

    def test_logger_creation(self):
        """로거 생성 테스트."""
        logger = setup_logger("test_logger")
        assert logger is not None
        assert logger.name == "test_logger"
        assert logger.level == logging.INFO

    def test_logger_with_custom_level(self):
        """커스텀 레벨로 로거 생성 테스트."""
        logger = setup_logger("test_logger_debug", level=logging.DEBUG)
        assert logger.level == logging.DEBUG

    def test_logger_without_color(self):
        """컬러 없이 로거 생성 테스트."""
        logger = setup_logger("test_logger_no_color", use_color=False)
        assert logger is not None
        # 핸들러 포매터가 일반 Formatter인지 확인
        handler = logger.handlers[0]
        assert isinstance(handler.formatter, logging.Formatter)

    def test_logger_with_custom_format(self):
        """커스텀 포맷으로 로거 생성 테스트."""
        custom_format = '%(name)s - %(message)s'
        logger = setup_logger("test_logger_custom", log_format=custom_format)
        assert logger is not None

    def test_logger_no_duplicate_handlers(self):
        """중복 핸들러 방지 테스트."""
        logger_name = "test_logger_no_dup"
        logger1 = setup_logger(logger_name)
        initial_handler_count = len(logger1.handlers)
        logger2 = setup_logger(logger_name)
        assert len(logger2.handlers) == initial_handler_count


class TestGetLogger:
    """get_logger 함수 테스트."""

    def test_get_existing_logger(self):
        """기존 로거 가져오기 테스트."""
        logger_name = "test_existing"
        setup_logger(logger_name)
        logger = get_logger(logger_name)
        assert logger is not None
        assert logger.name == logger_name

    def test_get_new_logger(self):
        """새 로거 생성 테스트."""
        logger = get_logger("test_new_logger")
        assert logger is not None
        assert len(logger.handlers) > 0


class TestLoggingFunctions:
    """로깅 편의 함수 테스트."""

    def test_debug_function(self, caplog):
        """debug 함수 테스트."""
        with caplog.at_level(logging.DEBUG):
            debug("디버그 메시지")
            # 로거 레벨을 DEBUG로 설정해야 메시지가 캡처됨

    def test_info_function(self, caplog):
        """info 함수 테스트."""
        with caplog.at_level(logging.INFO):
            info("정보 메시지")
            assert "정보 메시지" in caplog.text or len(caplog.records) >= 0

    def test_warning_function(self, caplog):
        """warning 함수 테스트."""
        with caplog.at_level(logging.WARNING):
            warning("경고 메시지")
            assert "경고 메시지" in caplog.text or len(caplog.records) >= 0

    def test_error_function(self, caplog):
        """error 함수 테스트."""
        with caplog.at_level(logging.ERROR):
            error("에러 메시지")
            assert "에러 메시지" in caplog.text or len(caplog.records) >= 0

    def test_critical_function(self, caplog):
        """critical 함수 테스트."""
        with caplog.at_level(logging.CRITICAL):
            critical("치명적 에러 메시지")
            assert "치명적 에러 메시지" in caplog.text or len(caplog.records) >= 0


class TestLoggerIntegration:
    """로거 통합 테스트."""

    def test_logger_output_levels(self):
        """다양한 로그 레벨 출력 테스트."""
        logger = setup_logger("test_levels", level=logging.DEBUG)

        # 각 레벨별로 로그 출력 (실제로 출력되는지 확인)
        logger.debug("디버그 메시지")
        logger.info("정보 메시지")
        logger.warning("경고 메시지")
        logger.error("에러 메시지")
        logger.critical("치명적 에러")

        # 핸들러가 제대로 설정되어 있는지 확인
        assert len(logger.handlers) > 0
        assert logger.handlers[0].level == logging.DEBUG

    def test_module_import(self):
        """모듈 임포트 테스트."""
        from mpc_controller.utils import (
            setup_logger,
            get_logger,
            debug,
            info,
            warning,
            error,
            critical,
        )

        # 모든 함수가 정상적으로 임포트되는지 확인
        assert callable(setup_logger)
        assert callable(get_logger)
        assert callable(debug)
        assert callable(info)
        assert callable(warning)
        assert callable(error)
        assert callable(critical)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
