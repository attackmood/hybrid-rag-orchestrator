"""
Smart-RAG 서비스 로깅 시스템
loguru 기반의 구조화된 로깅 및 파일 로테이션 지원
"""

import sys
import json
from pathlib import Path
from typing import Any, Dict, Optional, Union
from loguru import logger
from config.settings import settings


class SmartRAGLogger:
    """Smart-RAG 전용 로거 클래스"""

    def __init__(self):
        """로거 초기화"""
        self._setup_logger()
        self._logger = logger

    def _setup_logger(self):
        """로거 설정 구성"""
        # 기존 핸들러 제거
        logger.remove()

        # 콘솔 출력 핸들러
        self._add_console_handler()

        # 파일 출력 핸들러 (설정된 경우)
        if settings.logging.FILE_PATH:
            self._add_file_handler()

        # 로그 레벨 설정
        logger.level(settings.logging.LEVEL)

    def _add_console_handler(self):
        """콘솔 출력 핸들러 추가"""
        console_format = (
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        )

        logger.add(
            sys.stdout,
            format=console_format,
            level=settings.logging.LEVEL,
            colorize=True,
            backtrace=True,
            diagnose=True,
        )

    def _add_file_handler(self):
        """파일 출력 핸들러 추가"""
        log_file = Path(settings.logging.FILE_PATH)
        log_file.parent.mkdir(parents=True, exist_ok=True)

        file_format = (
            "{time:YYYY-MM-DD HH:mm:ss} | "
            "{level: <8} | "
            "{name}:{function}:{line} | "
            "{message}"
        )

        logger.add(
            str(log_file),
            format=file_format,
            level=settings.logging.LEVEL,
            rotation="1 day",
            retention="30 days",
            compression="zip",
            backtrace=True,
            diagnose=True,
            enqueue=True,  # 비동기 파일 쓰기
        )

    def _format_extra(self, **kwargs) -> str:
        """추가 정보를 JSON 형태로 포맷팅"""
        if not kwargs:
            return ""

        try:
            # JSON 직렬화 가능한 데이터만 필터링
            serializable_data = {}
            for key, value in kwargs.items():
                try:
                    json.dumps(value)
                    serializable_data[key] = value
                except (TypeError, ValueError):
                    serializable_data[key] = str(value)

            if serializable_data:
                return f" | {json.dumps(serializable_data, ensure_ascii=False)}"
        except Exception:
            pass

        return ""

    def debug(self, message: str, **kwargs):
        """DEBUG 레벨 로그"""
        extra = self._format_extra(**kwargs)
        self._logger.debug(f"{message}{extra}")

    def info(self, message: str, **kwargs):
        """INFO 레벨 로그"""
        extra = self._format_extra(**kwargs)
        self._logger.info(f"{message}{extra}")

    def warning(self, message: str, **kwargs):
        """WARNING 레벨 로그"""
        extra = self._format_extra(**kwargs)
        self._logger.warning(f"{message}{extra}")

    def error(self, message: str, **kwargs):
        """ERROR 레벨 로그"""
        extra = self._format_extra(**kwargs)
        self._logger.error(f"{message}{extra}")

    def critical(self, message: str, **kwargs):
        """CRITICAL 레벨 로그"""
        extra = self._format_extra(**kwargs)
        self._logger.critical(f"{message}{extra}")

    def exception(self, message: str, **kwargs):
        """예외 정보와 함께 로그"""
        extra = self._format_extra(**kwargs)
        self._logger.exception(f"{message}{extra}")

    def log_request(
        self, method: str, url: str, status_code: int, response_time: float, **kwargs
    ):
        """HTTP 요청 로그"""
        extra = self._format_extra(**kwargs)
        self._logger.info(
            f"HTTP {method} {url} - {status_code} ({response_time:.3f}s){extra}"
        )

    def log_query(self, query: str, source: str, response_time: float, **kwargs):
        """사용자 쿼리 로그"""
        extra = self._format_extra(**kwargs)
        self._logger.info(
            f"Query from {source}: '{query}' ({response_time:.3f}s){extra}"
        )

    def log_pdf_processing(self, filename: str, pages: int, chunks: int, **kwargs):
        """PDF 처리 로그"""
        extra = self._format_extra(**kwargs)
        self._logger.info(
            f"PDF processed: {filename} - {pages} pages, {chunks} chunks{extra}"
        )

    def log_vector_search(
        self, query: str, results_count: int, search_time: float, **kwargs
    ):
        """벡터 검색 로그"""
        extra = self._format_extra(**kwargs)
        self._logger.info(
            f"Vector search: '{query}' - {results_count} results ({search_time:.3f}s){extra}"
        )

    def log_mcp_request(
        self, service: str, endpoint: str, response_time: float, **kwargs
    ):
        """MCP 서비스 요청 로그"""
        extra = self._format_extra(**kwargs)
        self._logger.info(f"MCP {service}:{endpoint} ({response_time:.3f}s){extra}")

    def log_error_with_context(self, error: Exception, context: str, **kwargs):
        """컨텍스트와 함께 에러 로그"""
        extra = self._format_extra(**kwargs)
        self._logger.error(f"Error in {context}: {str(error)}{extra}")
        if settings.base.DEBUG:
            self._logger.exception(f"Full traceback for error in {context}")

    def log_performance(self, operation: str, duration: float, **kwargs):
        """성능 관련 로그"""
        extra = self._format_extra(**kwargs)
        if duration > 1.0:  # 1초 이상 걸리는 작업은 WARNING
            self._logger.warning(
                f"Slow operation: {operation} took {duration:.3f}s{extra}"
            )
        else:
            self._logger.debug(f"Operation: {operation} took {duration:.3f}s{extra}")

    def log_configuration(self):
        """설정 정보 로그"""
        config_info = {
            "app_name": settings.base.APP_NAME,
            "version": settings.base.APP_VERSION,
            "environment": settings.base.ENVIRONMENT,
            "debug": settings.base.DEBUG,
            "log_level": settings.logging.LEVEL,
            "ollama_url": settings.ollama.BASE_URL,
            "chroma_dir": settings.chroma_db.PERSIST_DIRECTORY,
            "pdf_upload_dir": settings.rag.PDF_TEMP_DIR,
        }

        self._logger.info("Service configuration loaded", **config_info)

    def get_logger(self):
        """기본 loguru 로거 반환"""
        return self._logger


# 전역 로거 인스턴스
smart_logger = SmartRAGLogger()

# 편의를 위한 별칭
log = smart_logger
