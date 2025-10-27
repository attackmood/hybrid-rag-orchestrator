"""
헬스체크 API 엔드포인트

시스템 상태를 모니터링하는 API입니다.
"""

from __future__ import annotations

import time
import psutil
from datetime import datetime
from typing import Optional, Dict, Any
from fastapi import APIRouter, HTTPException, Depends

from .models import HealthResponse, ProcessingStats
from core.hybrid_router import HybridRouter
from utils.logger import log


# 전역 라우터 인스턴스 (app/main.py에서 주입)
_hybrid_router: Optional[HybridRouter] = None
_start_time: Optional[float] = None


def set_hybrid_router(router: HybridRouter) -> None:
    """HybridRouter 인스턴스를 설정합니다."""
    global _hybrid_router
    _hybrid_router = router


def set_start_time(start_time: float) -> None:
    """애플리케이션 시작 시간을 설정합니다."""
    global _start_time
    _start_time = start_time


def get_hybrid_router() -> Optional[HybridRouter]:
    """HybridRouter 인스턴스를 가져옵니다."""
    return _hybrid_router


def get_uptime() -> Optional[float]:
    """서비스 가동 시간을 계산합니다."""
    if _start_time is None:
        return None
    return time.time() - _start_time


# API 라우터 생성
router = APIRouter(prefix="/api/health", tags=["health"])


@router.get("/", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """
    기본 헬스체크를 수행합니다.

    Returns:
        HealthResponse: 헬스체크 결과
    """
    try:
        router_available = _hybrid_router is not None
        uptime = get_uptime()

        log.debug("헬스체크 수행")

        return HealthResponse(
            status="healthy" if router_available else "degraded",
            service="Smart-RAG Chat",
            version="1.0.0",
            router_available=router_available,
            uptime=uptime,
        )

    except Exception as e:
        log.error(f"헬스체크 중 오류 발생: {e}")

        return HealthResponse(
            status="unhealthy",
            service="Smart-RAG Chat",
            version="1.0.0",
            router_available=False,
            uptime=get_uptime(),
        )


@router.get("/detailed")
async def detailed_health_check() -> Dict[str, Any]:
    """
    상세한 헬스체크를 수행합니다.

    Returns:
        Dict[str, Any]: 상세 헬스체크 결과
    """
    try:
        # 시스템 리소스 정보
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage("/")

        # 프로세스 정보
        process = psutil.Process()
        process_memory = process.memory_info()

        # HybridRouter 상태
        router_status = "available" if _hybrid_router is not None else "unavailable"

        # 서비스 상태
        overall_status = "healthy"
        if cpu_percent > 90:
            overall_status = "warning"
        if memory.percent > 90:
            overall_status = "warning"
        if router_status == "unavailable":
            overall_status = "unhealthy"

        log.debug("상세 헬스체크 수행")

        return {
            "status": overall_status,
            "timestamp": datetime.now().isoformat(),
            "service": "Smart-RAG Chat",
            "version": "1.0.0",
            "uptime": get_uptime(),
            "system": {
                "cpu_percent": cpu_percent,
                "memory": {
                    "total": memory.total,
                    "available": memory.available,
                    "percent": memory.percent,
                    "used": memory.used,
                },
                "disk": {
                    "total": disk.total,
                    "used": disk.used,
                    "free": disk.free,
                    "percent": (disk.used / disk.total) * 100,
                },
            },
            "process": {
                "pid": process.pid,
                "memory_rss": process_memory.rss,
                "memory_vms": process_memory.vms,
                "cpu_percent": process.cpu_percent(),
            },
            "components": {
                "hybrid_router": router_status,
                "ollama_client": "available" if _hybrid_router else "unavailable",
                "mcp_client": "available" if _hybrid_router else "unavailable",
                "google_search": "available" if _hybrid_router else "unavailable",
                "rag_client": "available" if _hybrid_router else "unavailable",
            },
        }

    except Exception as e:
        log.error(f"상세 헬스체크 중 오류 발생: {e}")

        return {
            "status": "unhealthy",
            "timestamp": datetime.now().isoformat(),
            "service": "Smart-RAG Chat",
            "version": "1.0.0",
            "error": str(e),
        }


@router.get("/stats", response_model=ProcessingStats)
async def get_processing_stats() -> ProcessingStats:
    """
    처리 통계를 조회합니다.

    Returns:
        ProcessingStats: 처리 통계 정보
    """
    try:
        # TODO: 실제 통계 데이터 수집
        # 현재는 임시 데이터
        log.debug("처리 통계 조회")

        return ProcessingStats(
            total_queries=0,
            successful_queries=0,
            failed_queries=0,
            average_processing_time=0.0,
            mode_usage={"parallel": 0, "adaptive": 0},
            tool_usage={
                "weather": 0,
                "stock": 0,
                "calculator": 0,
                "web_search": 0,
                "knowledge_base": 0,
            },
        )

    except Exception as e:
        log.error(f"처리 통계 조회 중 오류 발생: {e}")

        raise HTTPException(status_code=500, detail=f"통계 조회 중 오류 발생: {str(e)}")


@router.get("/ready")
async def readiness_check() -> Dict[str, Any]:
    """
    서비스 준비 상태를 확인합니다.

    Returns:
        Dict[str, Any]: 준비 상태 결과
    """
    try:
        # 모든 핵심 컴포넌트가 준비되었는지 확인
        router_ready = _hybrid_router is not None

        if router_ready:
            # HybridRouter의 내부 컴포넌트들도 확인
            try:
                # 간단한 테스트 쿼리로 라우터 상태 확인
                test_result = await _hybrid_router._analyze_query({"query": "test"})
                router_functional = test_result is not None
            except Exception:
                router_functional = False
        else:
            router_functional = False

        is_ready = router_ready and router_functional

        log.debug(f"준비 상태 확인: {is_ready}")

        return {
            "ready": is_ready,
            "timestamp": datetime.now().isoformat(),
            "checks": {
                "hybrid_router_initialized": router_ready,
                "hybrid_router_functional": router_functional,
            },
        }

    except Exception as e:
        log.error(f"준비 상태 확인 중 오류 발생: {e}")

        return {
            "ready": False,
            "timestamp": datetime.now().isoformat(),
            "error": str(e),
        }


@router.get("/live")
async def liveness_check() -> Dict[str, Any]:
    """
    서비스 생존 상태를 확인합니다.

    Returns:
        Dict[str, Any]: 생존 상태 결과
    """
    try:
        # 기본적인 생존 확인 (프로세스가 살아있는지)
        uptime = get_uptime()
        is_alive = uptime is not None and uptime > 0

        log.debug(f"생존 상태 확인: {is_alive}")

        return {
            "alive": is_alive,
            "timestamp": datetime.now().isoformat(),
            "uptime": uptime,
        }

    except Exception as e:
        log.error(f"생존 상태 확인 중 오류 발생: {e}")

        return {
            "alive": False,
            "timestamp": datetime.now().isoformat(),
            "error": str(e),
        }
