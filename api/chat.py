"""
채팅 API 엔드포인트

HybridRouter를 사용한 채팅 기능을 제공하는 API입니다.
"""

from __future__ import annotations

import os
import time
import uuid
from typing import Optional, Dict, Any
from fastapi import APIRouter, HTTPException, Depends, UploadFile, File
from fastapi.responses import JSONResponse

from .models import ChatRequest, ChatResponse, QueryAnalysis, ToolResult
from core.hybrid_router import HybridRouter
from services.rag import VectorSearchManager
from utils.logger import log


# 전역 라우터 인스턴스 (app/main.py에서 주입)
_hybrid_router: Optional[HybridRouter] = None


def set_hybrid_router(router: HybridRouter) -> None:
    """HybridRouter 인스턴스를 설정합니다."""
    global _hybrid_router
    _hybrid_router = router


def get_hybrid_router() -> HybridRouter:
    """HybridRouter 인스턴스를 가져옵니다."""
    if _hybrid_router is None:
        raise HTTPException(
            status_code=503, detail="HybridRouter가 초기화되지 않았습니다."
        )
    return _hybrid_router


# API 라우터 생성
router = APIRouter(prefix="/api/chat", tags=["chat"])


@router.post("/query", response_model=ChatResponse)
async def process_chat_query(
    request: ChatRequest, hybrid_router: HybridRouter = Depends(get_hybrid_router)
) -> ChatResponse:
    """
    채팅 쿼리를 처리합니다.

    Args:
        request: 채팅 요청 데이터
        hybrid_router: HybridRouter 인스턴스

    Returns:
        ChatResponse: 처리 결과
    """
    start_time = time.time()
    session_id = request.session_id or str(uuid.uuid4())

    try:
        log.info(f"채팅 쿼리 처리 시작: {request.message[:50]}...")

        # HybridRouter로 쿼리 처리 (병렬 워크플로우)
        result = await hybrid_router.process_query(query=request.message)

        processing_time = time.time() - start_time

        # 결과 파싱
        answer = result.get("final_answer", "처리 완료")
        metadata = {
            "complexity_score": result.get("complexity_score", 0.0),
            "selected_tools": result.get("selected_tools", []),
        }

        log.info(f"채팅 쿼리 처리 완료: {processing_time:.2f}초")

        return ChatResponse(
            success=True,
            message=answer,
            session_id=session_id,
            processing_time=processing_time,
            mode_used="parallel",
            metadata=metadata,
        )

    except Exception as e:
        processing_time = time.time() - start_time
        error_msg = f"쿼리 처리 중 오류 발생: {str(e)}"

        log.error(f"채팅 쿼리 처리 실패: {error_msg}")

        return ChatResponse(
            success=False,
            message="죄송합니다. 처리 중 오류가 발생했습니다.",
            session_id=session_id,
            processing_time=processing_time,
            error=error_msg,
        )


@router.post("/analyze", response_model=QueryAnalysis)
async def analyze_query(
    request: ChatRequest, hybrid_router: HybridRouter = Depends(get_hybrid_router)
) -> QueryAnalysis:
    """
    쿼리를 분석합니다 (도구 실행 없이).

    Args:
        request: 분석할 쿼리 요청
        hybrid_router: HybridRouter 인스턴스

    Returns:
        QueryAnalysis: 쿼리 분석 결과
    """
    try:
        log.info(f"쿼리 분석 시작: {request.message[:50]}...")

        # HybridRouter의 분석 기능 사용
        analysis_result = await hybrid_router._analyze_query({"query": request.message})

        parallel_results = analysis_result.get("parallel_results", {})

        log.info("쿼리 분석 완료")

        return QueryAnalysis(
            query=request.message,
            complexity_score=parallel_results.get("complexity", 0.0),
            primary_intent=parallel_results.get("intent_analysis", {}).get(
                "primary_intent", "unknown"
            ),
            selected_tools=parallel_results.get("selected_tools", []),
            confidence=parallel_results.get("intent_analysis", {}).get(
                "confidence", 0.0
            ),
        )

    except Exception as e:
        error_msg = f"쿼리 분석 중 오류 발생: {str(e)}"
        log.error(error_msg)

        raise HTTPException(status_code=500, detail=error_msg)


@router.get("/history/{session_id}")
async def get_chat_history(session_id: str, limit: int = 50) -> JSONResponse:
    """
    채팅 히스토리를 조회합니다.

    Args:
        session_id: 세션 ID
        limit: 조회할 메시지 수 제한

    Returns:
        JSONResponse: 채팅 히스토리
    """
    try:
        # TODO: 실제 세션 저장소에서 히스토리 조회
        # 현재는 임시 응답
        log.info(f"채팅 히스토리 조회: {session_id}")

        return JSONResponse(
            {
                "success": True,
                "session_id": session_id,
                "messages": [],
                "total_count": 0,
                "message": "채팅 히스토리 조회 완료",
            }
        )

    except Exception as e:
        error_msg = f"채팅 히스토리 조회 중 오류 발생: {str(e)}"
        log.error(error_msg)

        raise HTTPException(status_code=500, detail=error_msg)


@router.delete("/history/{session_id}")
async def clear_chat_history(session_id: str) -> JSONResponse:
    """
    채팅 히스토리를 삭제합니다.

    Args:
        session_id: 세션 ID

    Returns:
        JSONResponse: 삭제 결과
    """
    try:
        # TODO: 실제 세션 저장소에서 히스토리 삭제
        log.info(f"채팅 히스토리 삭제: {session_id}")

        return JSONResponse(
            {
                "success": True,
                "session_id": session_id,
                "message": "채팅 히스토리가 삭제되었습니다.",
            }
        )

    except Exception as e:
        error_msg = f"채팅 히스토리 삭제 중 오류 발생: {str(e)}"
        log.error(error_msg)

        raise HTTPException(status_code=500, detail=error_msg)


@router.post("/upload-pdf")
async def upload_pdf_file(
    file: UploadFile = File(...),
    add_to_chroma: bool = True,
    hybrid_router: HybridRouter = Depends(get_hybrid_router),
) -> JSONResponse:
    """
    PDF 파일을 업로드하고 RAG 시스템에 추가합니다.

    Args:
        file: 업로드된 PDF 파일
        add_to_chroma: ChromaDB에 영구 저장 여부
        hybrid_router: HybridRouter 인스턴스

    Returns:
        JSONResponse: 업로드 결과
    """
    start_time = time.time()

    try:
        log.info(f"PDF 파일 업로드 시작: {file.filename}")

        # 파일 확장자 확인
        if not file.filename.endswith(".pdf"):
            raise HTTPException(status_code=400, detail="PDF 파일만 업로드 가능합니다.")

        # 임시 파일로 저장
        import tempfile
        import shutil

        temp_file_path = None
        try:
            # 임시 파일 생성
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                temp_file_path = temp_file.name
                # 업로드된 파일을 임시 파일로 복사
                shutil.copyfileobj(file.file, temp_file)

            log.info(f"임시 파일 저장 완료: {temp_file_path}")

            # PDF 처리
            pdf_processor = hybrid_router.tools_registry.rag_client.pdf_processor
            processing_result = pdf_processor.process_pdf(
                temp_file_path, metadata={"filename": file.filename, "source": "upload"}
            )

            log.info(f"PDF 처리 완료: {processing_result.total_chunks}개 청크 생성")

            # ChromaDB에 영구 저장 (선택적)
            saved_to_chroma = False
            if add_to_chroma:
                try:
                    chroma_client = (
                        hybrid_router.tools_registry.rag_client.chroma_client
                    )

                    # 청크를 ChromaDB에 저장
                    chunk_texts = [chunk.text for chunk in processing_result.chunks]
                    chunk_metadatas = [
                        {**chunk.metadata, "pdf_filename": file.filename}
                        for chunk in processing_result.chunks
                    ]

                    doc_ids = chroma_client.add_texts(
                        texts=chunk_texts, metadatas=chunk_metadatas
                    )

                    log.info(f"ChromaDB에 {len(doc_ids)}개 청크 저장 완료")
                    saved_to_chroma = True

                except Exception as e:
                    log.warning(f"ChromaDB 저장 실패 (실시간 검색은 가능): {e}")

            processing_time = time.time() - start_time

            return JSONResponse(
                {
                    "success": True,
                    "message": f"PDF 파일이 성공적으로 업로드되었습니다.",
                    "data": {
                        "filename": file.filename,
                        "file_hash": processing_result.file_hash,
                        "total_pages": processing_result.total_pages,
                        "total_chunks": processing_result.total_chunks,
                        "file_size": processing_result.file_size,
                        "saved_to_chroma": saved_to_chroma,
                        "processing_time": processing_time,
                    },
                }
            )

        finally:
            # 임시 파일 삭제
            if temp_file_path and os.path.exists(temp_file_path):
                os.remove(temp_file_path)
                log.debug(f"임시 파일 삭제: {temp_file_path}")

    except HTTPException:
        raise
    except Exception as e:
        processing_time = time.time() - start_time
        error_msg = f"PDF 업로드 중 오류 발생: {str(e)}"
        log.error(error_msg)

        raise HTTPException(status_code=500, detail=error_msg)
