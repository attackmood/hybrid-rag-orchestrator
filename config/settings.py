"""
Smart-RAG 서비스 설정 관리 모듈
dataclasses와 os.environ을 사용한 환경변수 관리
"""

import os
from pathlib import Path
from typing import Optional, Dict, List
from dataclasses import dataclass


# 프로젝트 루트 디렉토리
PROJECT_ROOT = Path(__file__).parent.parent


@dataclass
class BaseConfig:
    """기본 설정"""

    APP_NAME: str
    APP_VERSION: str
    DEBUG: bool
    ENVIRONMENT: str


@dataclass
class ServerConfig:
    """서버 설정"""

    HOST: str
    PORT: int
    WORKERS: int


@dataclass
class OllamaConfig:
    """Ollama 설정"""

    BASE_URL: str
    MODEL_NAME: str
    TIMEOUT: int
    MAX_TOKENS: int
    TEMPERATURE: float


@dataclass
class GoogleSearchConfig:
    """Google Search API 설정"""

    API_KEY: str
    CSE_ID: str
    MAX_RESULTS: int
    TIMEOUT: int


@dataclass
class ChromaDBConfig:
    """ChromaDB 설정"""

    PERSIST_DIRECTORY: str
    COLLECTION_NAME: str
    DISTANCE_THRESHOLD: float
    MAX_RESULTS: int


@dataclass
class EmbeddingConfig:
    """임베딩 모델 설정"""

    EMBEDDING_MODEL: str
    EMBEDDING_DEVICE: str
    EMBEDDING_BATCH_SIZE: int
    EMBEDDING_MAX_LENGTH: int
    EMBEDDING_CACHE_DIR: str
    ENABLE_CACHE: bool
    NORMALIZE_VECTORS: bool


@dataclass
class MCPConfig:
    """MCP 서버 설정"""

    SERVERS: Dict[str, Dict]
    TIMEOUT: int
    RETRY_ATTEMPTS: int
    RETRY_DELAY: int
    PING_INTERVAL: int
    PING_TIMEOUT: int
    CLOSE_TIMEOUT: int


@dataclass
class RAGConfig:
    """RAG 설정"""

    CHUNK_SIZE: int
    CHUNK_OVERLAP: int
    SIMILARITY_THRESHOLD: float
    MAX_CONTEXT_LENGTH: int
    PDF_TEMP_DIR: str


@dataclass
class LoggingConfig:
    """로깅 설정"""

    LEVEL: str
    FORMAT: str
    FILE_PATH: str
    MAX_BYTES: int
    BACKUP_COUNT: int


@dataclass
class CacheConfig:
    """캐시 설정"""

    CACHE_TTL: int
    CACHE_MAX_SIZE: int
    CACHE_DIR: str


@dataclass
class PerformanceConfig:
    """성능 설정"""

    MAX_CONCURRENT_REQUESTS: int
    REQUEST_TIMEOUT: int


@dataclass
class SecurityConfig:
    """보안 설정"""

    SECRET_KEY: str
    ALLOWED_HOSTS: List[str]
    MAX_FILE_SIZE: int


class Settings:
    """Smart-RAG 서비스 설정 클래스"""

    def __init__(self):
        self._load_environment()
        self.base = self._get_base_config()
        self.server = self._get_server_config()
        self.ollama = self._get_ollama_config()
        self.google_search = self._get_google_search_config()
        self.chroma_db = self._get_chroma_db_config()
        self.embedding = self._get_embedding_config()
        self.mcp = self._get_mcp_config()
        self.rag = self._get_rag_config()
        self.logging = self._get_logging_config()
        self.cache = self._get_cache_config()
        self.performance = self._get_performance_config()
        self.security = self._get_security_config()

    def _load_environment(self):
        """환경변수 로드"""
        env_path = PROJECT_ROOT / ".env"
        if env_path.exists():
            try:
                from dotenv import load_dotenv

                load_dotenv(env_path)
            except ImportError:
                print(
                    "Warning: python-dotenv not installed. Using system environment variables."
                )

    def _get_base_config(self) -> BaseConfig:
        """기본 설정 로드"""
        return BaseConfig(
            APP_NAME=self._get_env("APP_NAME", "Smart-RAG"),
            APP_VERSION=self._get_env("APP_VERSION", "1.0.0"),
            DEBUG=self._get_env_bool("DEBUG", False),
            ENVIRONMENT=self._get_env("ENVIRONMENT", "development"),
        )

    def _get_server_config(self) -> ServerConfig:
        """서버 관련 환경변수 로드"""
        return ServerConfig(
            HOST=self._get_env("HOST", "0.0.0.0"),
            PORT=self._get_env_int("PORT", 8000),
            WORKERS=self._get_env_int("WORKERS", 1),
        )

    def _get_ollama_config(self) -> OllamaConfig:
        """Ollama 관련 환경변수 로드"""
        return OllamaConfig(
            BASE_URL=self._get_env("OLLAMA_BASE_URL", "http://localhost:11434"),
            MODEL_NAME=self._get_env("OLLAMA_MODEL", "llama3.1:8b"),
            TIMEOUT=self._get_env_int("OLLAMA_TIMEOUT", 60),
            MAX_TOKENS=self._get_env_int("OLLAMA_MAX_TOKENS", 1024),
            TEMPERATURE=self._get_env_float("OLLAMA_TEMPERATURE", 0.5),
        )

    def _get_embedding_config(self) -> EmbeddingConfig:
        """임베딩 모델 관련 환경변수 로드"""
        default_cache = str(PROJECT_ROOT / "data" / "embeddings_cache")
        return EmbeddingConfig(
            EMBEDDING_MODEL=self._get_env("EMBEDDING_MODEL", "jhgan/ko-sbert-nli"),
            EMBEDDING_DEVICE=self._get_env("EMBEDDING_DEVICE", "cuda"),
            EMBEDDING_BATCH_SIZE=self._get_env_int("EMBEDDING_BATCH_SIZE", 128),
            EMBEDDING_MAX_LENGTH=self._get_env_int("EMBEDDING_MAX_LENGTH", 512),
            EMBEDDING_CACHE_DIR=self._get_env("EMBEDDING_CACHE_DIR", default_cache),
            ENABLE_CACHE=self._get_env_bool("EMBEDDING_ENABLE_CACHE", True),
            NORMALIZE_VECTORS=self._get_env_bool("EMBEDDING_NORMALIZE_VECTORS", True),
        )

    def _get_google_search_config(self) -> GoogleSearchConfig:
        """Google Search API 관련 환경변수 로드"""
        return GoogleSearchConfig(
            API_KEY=self._get_env("GOOGLE_API_KEY", required=True),
            CSE_ID=self._get_env("GOOGLE_CSE_ID", required=True),
            MAX_RESULTS=self._get_env_int("GOOGLE_MAX_RESULTS", 10),
            TIMEOUT=self._get_env_int("GOOGLE_TIMEOUT", 10),
        )

    def _get_chroma_db_config(self) -> ChromaDBConfig:
        """ChromaDB 관련 환경변수 로드"""
        default_path = str(PROJECT_ROOT / "data" / "chroma_db")
        return ChromaDBConfig(
            PERSIST_DIRECTORY=self._get_env("CHROMA_PERSIST_DIR", default_path),
            COLLECTION_NAME=self._get_env("CHROMA_COLLECTION", "smart_rag_documents"),
            DISTANCE_THRESHOLD=self._get_env_float("CHROMA_DISTANCE_THRESHOLD", 0.5),
            MAX_RESULTS=self._get_env_int("CHROMA_MAX_RESULTS", 10),
        )

    def _get_mcp_config(self) -> MCPConfig:
        """MCP 서버 관련 환경변수 로드"""
        # 기본 MCP 서버 구성
        default_servers = {
            "mcp_server_jsonrpc": {
                "jsonrpc": self._get_env("MCP_JSON_RPC", "2.0"),
                "method": self._get_env("MCP_METHOD", "tools/call"),
                "url": self._get_env("MCP_WEBSOCKET_URL", "ws://localhost:8765/ws"),
            },
        }

        return MCPConfig(
            SERVERS=default_servers,
            TIMEOUT=self._get_env_int("MCP_TIMEOUT", 10),
            RETRY_ATTEMPTS=self._get_env_int("MCP_RETRY_ATTEMPTS", 3),
            RETRY_DELAY=self._get_env_int("MCP_RETRY_DELAY", 1),
            PING_INTERVAL=self._get_env_int("MCP_PING_INTERVAL", 30),
            PING_TIMEOUT=self._get_env_int("MCP_PING_TIMEOUT", 10),
            CLOSE_TIMEOUT=self._get_env_int("MCP_CLOSE_TIMEOUT", 5),
        )

    def _get_rag_config(self) -> RAGConfig:
        """RAG 설정 로드"""
        default_temp_dir = str(PROJECT_ROOT / "data" / "pdf_temp")
        return RAGConfig(
            CHUNK_SIZE=self._get_env_int("RAG_CHUNK_SIZE", 500),
            CHUNK_OVERLAP=self._get_env_int("RAG_CHUNK_OVERLAP", 100),
            SIMILARITY_THRESHOLD=self._get_env_float("RAG_SIMILARITY_THRESHOLD", 0.5),
            MAX_CONTEXT_LENGTH=self._get_env_int("RAG_MAX_CONTEXT_LENGTH", 1000),
            PDF_TEMP_DIR=self._get_env("RAG_PDF_TEMP_DIR", default_temp_dir),
        )

    def _get_logging_config(self) -> LoggingConfig:
        """로깅 설정 로드"""
        default_log_path = str(PROJECT_ROOT / "logs" / "smart_rag.log")
        return LoggingConfig(
            LEVEL=self._get_env("LOG_LEVEL", "INFO"),
            FORMAT=self._get_env(
                "LOG_FORMAT",
                "{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {message}",
            ),
            FILE_PATH=self._get_env("LOG_FILE_PATH", default_log_path),
            MAX_BYTES=self._get_env_int("LOG_MAX_BYTES", 10 * 1024 * 1024),  # 10MB
            BACKUP_COUNT=self._get_env_int("LOG_BACKUP_COUNT", 5),
        )

    def _get_cache_config(self) -> CacheConfig:
        default_cache_dir = str(PROJECT_ROOT / "data" / "cache")
        return CacheConfig(
            CACHE_TTL=self._get_env_int("CACHE_TTL", 3600),  # 1시간
            CACHE_MAX_SIZE=self._get_env_int("CACHE_MAX_SIZE", 1000),
            CACHE_DIR=self._get_env("CACHE_DIR", default_cache_dir),
        )

    def _get_performance_config(self) -> PerformanceConfig:
        """성능 설정 로드"""
        return PerformanceConfig(
            MAX_CONCURRENT_REQUESTS=self._get_env_int("MAX_CONCURRENT_REQUESTS", 10),
            REQUEST_TIMEOUT=self._get_env_int("REQUEST_TIMEOUT", 30),
        )

    def _get_security_config(self) -> SecurityConfig:
        """보안 설정 로드"""
        return SecurityConfig(
            SECRET_KEY=self._get_env(
                "SECRET_KEY", "your-secret-key-change-in-production"
            ),
            ALLOWED_HOSTS=self._get_env("ALLOWED_HOSTS", "localhost,127.0.0.1").split(
                ","
            ),
            MAX_FILE_SIZE=self._get_env_int("PDF_MAX_SIZE_MB", 50)
            * 1024
            * 1024,  # MB to bytes
        )

    def _get_env(self, key: str, default: str = None, required: bool = False) -> str:
        """환경 변수 가져오기"""
        value = os.getenv(key, default)
        if required and not value:
            raise ValueError(f"Required environment variable {key} is not set")
        return value

    def _get_env_int(self, key: str, default: int) -> int:
        """정수형 환경 변수 가져오기"""
        try:
            return int(self._get_env(key, str(default)))
        except ValueError:
            return default

    def _get_env_float(self, key: str, default: float) -> float:
        """실수형 환경 변수 가져오기"""
        try:
            return float(self._get_env(key, str(default)))
        except ValueError:
            return default

    def _get_env_bool(self, key: str, default: bool) -> bool:
        """불린형 환경 변수 가져오기"""
        value = self._get_env(key, str(default)).lower()
        return value in ("true", "1", "yes", "on")

    def create_directories(self):
        """필요한 디렉토리 생성"""
        directories = [
            self.chroma_db.PERSIST_DIRECTORY,
            self.embedding.EMBEDDING_CACHE_DIR,
            self.rag.PDF_TEMP_DIR,
            os.path.dirname(self.logging.FILE_PATH),
            self.cache.CACHE_DIR,
        ]

        for directory in directories:
            if directory:  # None이나 빈 문자열 체크
                Path(directory).mkdir(parents=True, exist_ok=True)

    def _validate_settings(self):
        """설정 검증"""
        # 로그 레벨 검증
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if self.logging.LEVEL not in valid_levels:
            raise ValueError(f"LOG_LEVEL must be one of {valid_levels}")

        # Google API 설정 검증
        if self.google_search.CSE_ID and not self.google_search.API_KEY:
            print("⚠️  GOOGLE_CSE_ID가 설정되었지만 GOOGLE_API_KEY가 없습니다.")

        # 프로덕션 환경 보안 검증
        if (
            self.is_production
            and self.security.SECRET_KEY == "your-secret-key-change-in-production"
        ):
            print("⚠️  프로덕션 환경에서는 SECRET_KEY를 변경해야 합니다.")

    @property
    def is_production(self) -> bool:
        """프로덕션 환경 여부"""
        return self.base.ENVIRONMENT.lower() == "production"

    @property
    def is_development(self) -> bool:
        """개발 환경 여부"""
        return self.base.ENVIRONMENT.lower() == "development"

    def get_mcp_urls(self) -> Dict[str, Optional[str]]:
        """MCP 서버 URL들을 딕셔너리로 반환"""
        return {
            name: server.get("url") if server.get("url") else None
            for name, server in self.mcp.SERVERS.items()
        }

    def get_mcp_keywords(self) -> Dict[str, List[str]]:
        """MCP 서비스별 키워드 반환"""
        return {
            name: server.get("keywords", [])
            for name, server in self.mcp.SERVERS.items()
        }


# 전역 설정 인스턴스 생성
try:
    settings = Settings()
    settings.create_directories()
    settings._validate_settings()
    print("✅ Smart-RAG 설정 로드 완료")
except Exception as e:
    print(f"⚠️  설정 로드 중 오류 발생: {e}")
    print("기본값으로 서비스를 시작합니다.")
    # 기본값으로 설정 생성 (검증 없이)
    settings = Settings()
    settings.create_directories()
