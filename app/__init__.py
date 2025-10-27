"""
Smart-RAG 웹 애플리케이션

HybridRouter를 기반으로 한 LLM 웹 애플리케이션
"""

from .main import app
from .config import AppConfig

__all__ = [
    "app",
    "AppConfig",
]
