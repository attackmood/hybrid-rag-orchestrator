"""
Smart-RAG 웹 애플리케이션 설정

애플리케이션 전반의 설정을 관리합니다.
"""

import os
from typing import Optional
from dataclasses import dataclass


@dataclass
class AppConfig:
    """애플리케이션 설정"""

    # 기본 설정
    app_name: str = "Smart-RAG Chat"
    app_version: str = "1.0.0"
    debug: bool = False

    # 서버 설정
    host: str = "0.0.0.0"
    port: int = 11434

    # 템플릿 설정
    template_dir: str = "templates"
    static_dir: str = "static"

    # 채팅 설정
    max_chat_history: int = 100
    max_message_length: int = 1000
    chat_timeout: int = 30

    # 로깅 설정
    log_level: str = "INFO"

    def __post_init__(self):
        """환경변수에서 설정 로드"""
        self.debug = os.getenv("DEBUG", "false").lower() == "true"
        self.host = os.getenv("HOST", self.host)
        self.port = int(os.getenv("PORT", self.port))
        self.log_level = os.getenv("LOG_LEVEL", self.log_level)

        # 디렉토리 경로 설정
        self.template_dir = os.path.join(os.getcwd(), self.template_dir)
        self.static_dir = os.path.join(os.getcwd(), self.static_dir)

    def get_database_url(self) -> Optional[str]:
        """데이터베이스 URL 반환 (필요시)"""
        return os.getenv("DATABASE_URL")

    def get_redis_url(self) -> Optional[str]:
        """Redis URL 반환 (필요시)"""
        return os.getenv("REDIS_URL")

    def is_production(self) -> bool:
        """프로덕션 환경 여부"""
        return os.getenv("ENVIRONMENT", "development").lower() == "production"

    def get_cors_origins(self) -> list[str]:
        """CORS 허용 오리진 목록"""
        origins = os.getenv("CORS_ORIGINS", "*")
        if origins == "*":
            return ["*"]
        return [origin.strip() for origin in origins.split(",")]
