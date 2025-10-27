"""
Smart-RAG 웹 애플리케이션 메인 엔트리포인트

HybridRouter를 기반으로 한 LLM 웹 애플리케이션
- 반응형 UI
"""

import sys
import os
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Optional

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime

from utils.logger import log
from core.hybrid_router import HybridRouter
from app.config import AppConfig
from api import chat_router, health_router
from api.chat import set_hybrid_router as set_chat_router
from api.health import set_hybrid_router as set_health_router, set_start_time

# 전역 변수
hybrid_router: Optional[HybridRouter] = None
app_config = AppConfig()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """애플리케이션 생명주기 관리"""
    global hybrid_router

    # 시작 시 초기화
    log.info("🚀 Smart-RAG 웹 애플리케이션 시작")
    try:
        hybrid_router = HybridRouter()
        log.info("✅ HybridRouter 초기화 완료")

        # API 라우터들에 HybridRouter 설정
        set_chat_router(hybrid_router)
        set_health_router(hybrid_router)
        set_start_time(datetime.now().timestamp())

        log.info("✅ API 라우터 설정 완료")
    except Exception as e:
        log.error(f"❌ HybridRouter 초기화 실패: {e}")
        hybrid_router = None

    yield

    # 종료 시 정리
    log.info("🔄 Smart-RAG 웹 애플리케이션 종료 중...")
    if hybrid_router:
        await hybrid_router.aclose()
    log.info("✅ Smart-RAG 웹 애플리케이션 정리 완료")


# FastAPI 앱 생성
app = FastAPI(
    title="Smart-RAG Chat",
    description="HybridRouter 기반 LLM 채팅 애플리케이션",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# CORS 미들웨어
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 개발 환경에서는 모든 도메인 허용
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API 라우터 등록
app.include_router(chat_router)
app.include_router(health_router)

# 정적 파일 서빙
app.mount("/static", StaticFiles(directory="static"), name="static")

# 템플릿 엔진 설정
templates = Jinja2Templates(directory="templates")


# 루트 엔드포인트
@app.get("/")
async def root(request: Request):
    """메인 페이지"""
    current_time = datetime.now().strftime("%H:%M")
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "title": "Smart-RAG Chat",
            "version": "1.0.0",
            "current_time": current_time,
        },
    )


# 헬스체크 엔드포인트
@app.get("/health")
async def health_check():
    """시스템 상태 확인"""
    return {
        "status": "healthy" if hybrid_router else "unhealthy",
        "service": "Smart-RAG Chat",
        "version": "1.0.0",
        "router_available": hybrid_router is not None,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=app_config.host,
        port=app_config.port,
        reload=True,
        log_level=app_config.log_level.lower(),
    )
