"""
API 데이터 모델

FastAPI 엔드포인트에서 사용하는 Pydantic 모델들을 정의합니다.
"""

from __future__ import annotations

from datetime import datetime
from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    """채팅 메시지 모델"""

    role: str = Field(..., description="메시지 역할 (user, assistant, system)")
    content: str = Field(..., description="메시지 내용")
    timestamp: Optional[datetime] = Field(
        default_factory=datetime.now, description="메시지 시간"
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default_factory=dict, description="추가 메타데이터"
    )


class ChatRequest(BaseModel):
    """채팅 요청 모델"""

    message: str = Field(
        ..., description="사용자 메시지", min_length=1, max_length=1000
    )
    session_id: Optional[str] = Field(None, description="세션 ID")
    mode: Optional[str] = Field(
        "parallel", description="처리 모드 (parallel, adaptive)"
    )
    context: Optional[Dict[str, Any]] = Field(
        default_factory=dict, description="추가 컨텍스트"
    )


class ChatResponse(BaseModel):
    """채팅 응답 모델"""

    success: bool = Field(..., description="처리 성공 여부")
    message: str = Field(..., description="응답 메시지")
    session_id: Optional[str] = Field(None, description="세션 ID")
    processing_time: Optional[float] = Field(None, description="처리 시간 (초)")
    mode_used: Optional[str] = Field(None, description="사용된 처리 모드")
    metadata: Optional[Dict[str, Any]] = Field(
        default_factory=dict, description="추가 메타데이터"
    )
    error: Optional[str] = Field(None, description="에러 메시지 (실패 시)")


class HealthResponse(BaseModel):
    """헬스체크 응답 모델"""

    status: str = Field(..., description="서비스 상태")
    timestamp: datetime = Field(default_factory=datetime.now, description="체크 시간")
    service: str = Field(..., description="서비스 이름")
    version: str = Field(..., description="서비스 버전")
    router_available: bool = Field(..., description="라우터 사용 가능 여부")
    uptime: Optional[float] = Field(None, description="서비스 가동 시간 (초)")


class QueryAnalysis(BaseModel):
    """쿼리 분석 결과 모델"""

    query: str = Field(..., description="원본 쿼리")
    complexity_score: float = Field(..., description="복잡도 점수")
    primary_intent: str = Field(..., description="주요 의도")
    selected_tools: List[str] = Field(..., description="선택된 도구들")
    confidence: float = Field(..., description="분석 신뢰도")


class ToolResult(BaseModel):
    """도구 실행 결과 모델"""

    tool_name: str = Field(..., description="도구 이름")
    success: bool = Field(..., description="실행 성공 여부")
    result: Optional[Any] = Field(None, description="실행 결과")
    execution_time: float = Field(..., description="실행 시간 (초)")
    error: Optional[str] = Field(None, description="에러 메시지")


class ProcessingStats(BaseModel):
    """처리 통계 모델"""

    total_queries: int = Field(..., description="총 쿼리 수")
    successful_queries: int = Field(..., description="성공한 쿼리 수")
    failed_queries: int = Field(..., description="실패한 쿼리 수")
    average_processing_time: float = Field(..., description="평균 처리 시간 (초)")
    mode_usage: Dict[str, int] = Field(..., description="모드별 사용 횟수")
    tool_usage: Dict[str, int] = Field(..., description="도구별 사용 횟수")
