"""
RAG 통합 모듈 (Vector Search Manager)

크로마 DB와 실시간 PDF 검색을 통합하여 병렬 검색을 수행하고
결과를 통합, 랭킹, 중복 제거하는 핵심 모듈입니다.
"""

from __future__ import annotations

import time
import asyncio
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
import numpy as np

from utils.logger import log
from utils.embeddings import KoreanEmbeddingModel, create_embedding_model
from config.settings import settings
from .chroma_client import ChromaDBClient, SearchResult
from .pdf_processor import PDFProcessor, PDFChunk, create_pdf_processor


@dataclass
class UnifiedSearchResult:
    """RAG 검색 결과 데이터 구조"""

    content: str
    source_type: str  # 'chroma', 'pdf_realtime' (RAG 전용)
    source_id: str
    metadata: Dict[str, Any]
    similarity_score: float
    rank: int
    context_length: int


@dataclass
class SearchContext:
    """검색 컨텍스트 정보"""

    query: str
    query_embedding: List[float]
    max_results: int
    similarity_threshold: float
    max_context_length: int
    search_start_time: float = field(default_factory=time.time)


class VectorSearchManager:
    """RAG 벡터 검색 관리자

    크로마 DB와 실시간 PDF 검색을 병렬로 실행하고
    RAG 결과를 통합하여 최적의 컨텍스트를 제공합니다.

    주의: 이 클래스는 RAG 검색 전용이며, MCP나 Google Search는 포함하지 않습니다.
    """

    def __init__(
        self,
        chroma_client: Optional[ChromaDBClient] = None,
        pdf_processor: Optional[PDFProcessor] = None,
        embedding_model: Optional[KoreanEmbeddingModel] = None,
        max_workers: Optional[int] = None,
        similarity_threshold: Optional[float] = None,
        max_context_length: Optional[int] = None,
    ) -> None:
        """벡터 검색 관리자를 초기화합니다.

        Args:
            chroma_client: 크로마 DB 클라이언트
            pdf_processor: PDF 프로세서
            embedding_model: 임베딩 모델 (공유 인스턴스 권장)
            max_workers: 병렬 처리 최대 워커 수
            similarity_threshold: 유사도 임계값
            max_context_length: 최대 컨텍스트 길이
        """
        try:
            # 설정값 적용
            self.similarity_threshold = (
                similarity_threshold or settings.rag.SIMILARITY_THRESHOLD
            )
            self.max_context_length = (
                max_context_length or settings.rag.MAX_CONTEXT_LENGTH
            )
            self.max_workers = max_workers or 4

            # 핵심 컴포넌트 초기화 (임베딩 모델은 외부에서 전달받은 것 우선 사용)
            if embedding_model is not None:
                self.embedding_model = embedding_model
                log.info("외부에서 전달받은 임베딩 모델 사용")
            else:
                self.embedding_model = create_embedding_model()
                log.info("새로운 임베딩 모델 생성")

            self.chroma_client = chroma_client or ChromaDBClient()

            # PDFProcessor도 동일한 임베딩 모델 사용하도록 설정
            if pdf_processor is not None:
                self.pdf_processor = pdf_processor
                # PDFProcessor의 임베딩 모델이 현재 모델과 다른 경우 경고
                if (
                    hasattr(self.pdf_processor, "embedding_model")
                    and self.pdf_processor.embedding_model is not self.embedding_model
                ):
                    log.warning("PDFProcessor와 다른 임베딩 모델 사용 중")
            else:
                self.pdf_processor = create_pdf_processor(
                    embedding_model=self.embedding_model
                )
                log.info("PDFProcessor에 동일한 임베딩 모델 전달")

            log.info(
                f"VectorSearchManager 초기화 완료: "
                f"threshold={self.similarity_threshold}, "
                f"max_context={self.max_context_length}, "
                f"workers={self.max_workers}, "
                f"embedding_model_id={id(self.embedding_model)}"
            )
        except Exception as e:
            log.error(f"VectorSearchManager 초기화 실패: {e}")
            raise

    async def search_async(
        self,
        query: str,
        max_results: Optional[int] = None,
        similarity_threshold: Optional[float] = None,
        include_pdf_realtime: bool = True,
        include_chroma: bool = True,
    ) -> List[UnifiedSearchResult]:
        """비동기 RAG 벡터 검색을 수행합니다.

        Args:
            query: 검색 쿼리
            max_results: 최대 결과 수
            similarity_threshold: 유사도 임계값
            include_pdf_realtime: 실시간 PDF 검색 포함 여부
            include_chroma: 크로마 DB 검색 포함 여부

        Returns:
            RAG 검색 결과 리스트 (크로마 DB + 실시간 PDF)
        """
        try:
            search_start = time.time()
            log.info(f"🔍 비동기 RAG 검색 시작: query='{query[:50]}...'")

            # 검색 컨텍스트 생성
            context = self._create_search_context(
                query, max_results, similarity_threshold
            )

            # 병렬 검색 실행
            tasks = []
            task_names = []

            if include_chroma:
                tasks.append(self._search_chroma_async(context))
                task_names.append("chroma")
            if include_pdf_realtime:
                tasks.append(self._search_pdf_realtime_async(context))
                task_names.append("pdf_realtime")

            if not tasks:
                log.warning("실행할 검색 작업이 없습니다.")
                return []

            # asyncio.gather로 병렬 실행
            parallel_start = time.time()
            search_results_with_exceptions = await asyncio.gather(
                *tasks, return_exceptions=True
            )
            parallel_time = time.time() - parallel_start
            log.info(
                f"⚡ 병렬 검색 실행 완료: {len(tasks)}개 작업, 소요시간={parallel_time:.3f}s"
            )

            # 예외 처리 및 결과 수집
            search_results = []
            for i, result in enumerate(search_results_with_exceptions):
                if isinstance(result, Exception):
                    log.error(f"{task_names[i]} 검색 실패: {result}")
                    search_results.append([])  # 빈 결과로 처리
                else:
                    search_results.append(result)

            # RAG 결과 통합 및 후처리
            unified_results = self._integrate_results(search_results)

            # 컨텍스트 최적화 및 중복 제거
            final_results = self._optimize_context(
                unified_results, context.max_context_length
            )

            search_time = time.time() - search_start
            log.info(
                f"✅ 비동기 RAG 검색 완료: {len(final_results)}개 결과, "
                f"소요시간={search_time:.3f}s (병렬 처리)"
            )

            return final_results

        except Exception as e:
            log.error(f"비동기 통합 검색 실패: {e}")
            raise

    def _create_search_context(
        self,
        query: str,
        max_results: Optional[int],
        similarity_threshold: Optional[float],
    ) -> SearchContext:
        """검색 컨텍스트를 생성합니다."""
        try:
            context_start = time.time()
            # 쿼리 임베딩 생성
            query_embedding = self._get_query_embedding(query)

            # 설정값 적용
            max_results = max_results or self.chroma_client.max_results
            similarity_threshold = similarity_threshold or self.similarity_threshold

            context = SearchContext(
                query=query,
                query_embedding=query_embedding,
                max_results=max_results,
                similarity_threshold=similarity_threshold,
                max_context_length=self.max_context_length,
            )

            context_time = time.time() - context_start
            log.info(f"📋 검색 컨텍스트 생성 완료: {context_time:.3f}s")
            return context
        except Exception as e:
            log.error(f"검색 컨텍스트 생성 실패: {e}")
            raise

    def _get_query_embedding(self, query: str) -> List[float]:
        """쿼리 텍스트의 임베딩을 생성합니다."""
        try:
            embedding_start = time.time()
            # 직접 임베딩 생성
            embedding = self.embedding_model.encode_text(query)

            embedding_time = time.time() - embedding_start
            log.info(f"🔢 쿼리 임베딩 생성 완료: {embedding_time:.3f}s")

            return embedding.tolist()
        except Exception as e:
            log.error(f"쿼리 임베딩 생성 실패: {e}")
            raise

    async def _search_chroma_async(
        self, context: SearchContext
    ) -> List[UnifiedSearchResult]:
        """비동기 크로마 DB 검색을 실행합니다 (고급 랭킹 및 중복 제거 포함)."""
        try:
            chroma_start = time.time()
            # I/O 바운드 작업을 스레드 풀에서 실행
            loop = asyncio.get_event_loop()

            # ChromaDB 쿼리를 별도 스레드에서 실행
            search_result = await loop.run_in_executor(
                None,
                lambda: self.chroma_client.query_with_ranking(
                    query_text=context.query,
                    top_k=context.max_results,
                    where=None,
                    include_documents=True,
                    remove_duplicates=True,  # ChromaDB 수준에서 의미적 중복 제거
                    similarity_threshold=0.95,  # 중복 판단 임계값 (95% 이상 유사하면 중복)
                ),
            )

            # UnifiedSearchResult로 변환
            unified_results = []
            for i, (doc_id, document, metadata, distance) in enumerate(
                zip(
                    search_result.ids,
                    search_result.documents,
                    search_result.metadatas,
                    search_result.distances,
                )
            ):
                # 거리를 유사도 점수로 변환 (cosine distance -> similarity)
                similarity_score = 1.0 - distance

                # 검색 임계값 필터링 (여전히 적용)
                if similarity_score >= context.similarity_threshold:
                    unified_results.append(
                        UnifiedSearchResult(
                            content=document,
                            source_type="chroma",
                            source_id=doc_id,
                            metadata=metadata or {},
                            similarity_score=similarity_score,
                            rank=i + 1,  # 이미 정렬된 순서
                            context_length=len(document),
                        )
                    )

            chroma_time = time.time() - chroma_start
            log.info(
                f"🗄️ 비동기 크로마 DB 검색 완료: {len(unified_results)}개 결과, "
                f"소요시간={chroma_time:.3f}s (정렬 및 중복 제거 적용됨)"
            )
            return unified_results

        except Exception as e:
            log.error(f"비동기 크로마 DB 검색 실패: {e}")
            return []

    async def _search_pdf_realtime_async(
        self, context: SearchContext
    ) -> List[UnifiedSearchResult]:
        """비동기 실시간 PDF 검색을 실행합니다."""
        try:
            pdf_start = time.time()
            # I/O 바운드 작업을 스레드 풀에서 실행
            loop = asyncio.get_event_loop()

            # PDF 검색을 별도 스레드에서 실행
            pdf_chunks = await loop.run_in_executor(
                None,
                lambda: self.pdf_processor.search_chunks(
                    query=context.query, top_k=context.max_results, file_filter=None
                ),
            )

            # 유사도 임계값 필터링
            filtered_chunks = [
                (chunk, score)
                for chunk, score in pdf_chunks
                if score >= context.similarity_threshold
            ]

            # UnifiedSearchResult로 변환
            unified_results = []
            for i, (chunk, similarity_score) in enumerate(filtered_chunks):
                unified_results.append(
                    UnifiedSearchResult(
                        content=chunk.text,
                        source_type="pdf_realtime",
                        source_id=chunk.id,
                        metadata={
                            **chunk.metadata,
                            "page": chunk.page,
                            "chunk_index": chunk.chunk_index,
                            "created_at": chunk.created_at,
                        },
                        similarity_score=similarity_score,
                        rank=i + 1,
                        context_length=len(chunk.text),
                    )
                )

            pdf_time = time.time() - pdf_start
            log.info(
                f"📄 비동기 실시간 PDF 검색 완료: {len(unified_results)}개 결과, 소요시간={pdf_time:.3f}s"
            )
            return unified_results

        except Exception as e:
            log.error(f"비동기 실시간 PDF 검색 실패: {e}")
            return []

    def _integrate_results(
        self, search_results: List[List[UnifiedSearchResult]]
    ) -> List[UnifiedSearchResult]:
        """RAG 다중 소스 검색 결과를 통합합니다."""
        try:
            integrate_start = time.time()
            all_results = []

            # 검색 결과 수집
            for result in search_results:
                all_results.extend(result)

            if not all_results:
                log.warning("통합할 RAG 검색 결과가 없습니다.")
                return []

            # 점수 정규화 및 랭킹 재계산
            normalized_results = self._normalize_scores(all_results)

            # 소스별 가중치 적용
            weighted_results = self._apply_source_weights(normalized_results)

            # 최종 랭킹
            final_results = sorted(
                weighted_results, key=lambda x: x.similarity_score, reverse=True
            )

            # 중복 제거 적용
            final_results = self._remove_duplicates(final_results)

            integrate_time = time.time() - integrate_start
            log.info(
                f"🔗 RAG 결과 통합 완료: {len(final_results)}개 결과, 소요시간={integrate_time:.3f}s"
            )
            return final_results

        except Exception as e:
            log.error(f"결과 통합 실패: {e}")
            return []

    def _normalize_scores(
        self, results: List[UnifiedSearchResult]
    ) -> List[UnifiedSearchResult]:
        """유사도 점수를 정규화합니다."""
        try:
            if not results:
                return results

            # 점수 범위 정규화 (0-1)
            scores = [r.similarity_score for r in results]
            min_score, max_score = min(scores), max(scores)

            if max_score == min_score:
                # 모든 점수가 동일한 경우 균등 분배
                normalized_score = 0.5
                for result in results:
                    result.similarity_score = normalized_score
            else:
                # Min-Max 정규화
                for result in results:
                    result.similarity_score = (result.similarity_score - min_score) / (
                        max_score - min_score
                    )

            return results

        except Exception as e:
            log.error(f"점수 정규화 실패: {e}")
            return results

    def _apply_source_weights(
        self, results: List[UnifiedSearchResult]
    ) -> List[UnifiedSearchResult]:
        """RAG 소스별 가중치를 적용합니다."""
        try:
            # RAG 소스별 가중치 정의 (MCP, Google 제외)
            source_weights = {
                "chroma": 1.0,  # 기존 저장 문서 (기본 가중치)
                "pdf_realtime": 1.2,  # 새로 업로드된 문서 (높은 가중치)
            }

            for result in results:
                weight = source_weights.get(result.source_type, 1.0)
                result.similarity_score *= weight

            return results

        except Exception as e:
            log.error(f"RAG 소스 가중치 적용 실패: {e}")
            return results

    def _optimize_context(
        self, results: List[UnifiedSearchResult], max_context_length: int
    ) -> List[UnifiedSearchResult]:
        """컨텍스트 길이를 최적화합니다."""
        optimize_start = time.time()
        try:
            if not results:
                return []

            # 현재 컨텍스트 길이 계산
            current_length = sum(len(result.content) for result in results)

            if current_length <= max_context_length:
                return results

            # 길이 제한을 초과하는 경우, 중요도 기반으로 선택
            optimized = []
            accumulated_length = 0

            for result in results:
                content_length = len(result.content)

                # 이 결과를 추가해도 제한을 초과하지 않는 경우
                if accumulated_length + content_length <= max_context_length:
                    optimized.append(result)
                    accumulated_length += content_length
                else:
                    # 남은 공간에 맞는지 확인
                    remaining_space = max_context_length - accumulated_length
                    if remaining_space > 100:  # 최소 100자 이상 남은 경우만
                        # 결과를 잘라서 추가
                        truncated_result = UnifiedSearchResult(
                            content=result.content[:remaining_space] + "...",
                            source_type=result.source_type,
                            source_id=result.source_id,
                            metadata=result.metadata,
                            similarity_score=result.similarity_score,
                            rank=result.rank,
                            context_length=remaining_space,
                        )
                        optimized.append(truncated_result)
                    break

            # 중복 제거 적용
            optimized = self._remove_duplicates(optimized)
            optimize_time = time.time() - optimize_start
            log.info(
                f"⚡ 컨텍스트 최적화 완료: {len(optimized)}개 결과, 소요시간={optimize_time:.3f}s"
            )

            return optimized

        except Exception as e:
            log.error(f"❌ 컨텍스트 최적화 실패: {e}")
            return results[:5]  # 기본적으로 상위 5개만 반환

    def _remove_duplicates(
        self, results: List[UnifiedSearchResult]
    ) -> List[UnifiedSearchResult]:
        """중복 내용을 제거합니다 (소스별 최적화 적용)."""
        try:
            dedup_start = time.time()
            # 소스별로 분류
            chroma_results = [r for r in results if r.source_type == "chroma"]
            other_results = [r for r in results if r.source_type != "chroma"]

            # ChromaDB 결과는 이미 의미적 중복 제거가 적용되었으므로 그대로 유지
            deduplicated = chroma_results.copy()

            # 다른 소스 결과들에 대해서만 문자열 기반 중복 제거 적용
            seen_contents = set()

            # ChromaDB 결과의 내용을 seen_contents에 추가
            for result in chroma_results:
                normalized_content = self._normalize_content(result.content)
                seen_contents.add(normalized_content)

            # 다른 소스 결과 중복 제거
            for result in other_results:
                normalized_content = self._normalize_content(result.content)

                if normalized_content not in seen_contents:
                    seen_contents.add(normalized_content)
                    deduplicated.append(result)

            dedup_time = time.time() - dedup_start
            log.info(
                f"🔄 소스별 중복 제거: {len(results)} → {len(deduplicated)}개, "
                f"소요시간={dedup_time:.3f}s "
                f"(ChromaDB: {len(chroma_results)}개 유지, 기타: {len(deduplicated) - len(chroma_results)}개)"
            )
            return deduplicated

        except Exception as e:
            log.error(f"중복 제거 실패: {e}")
            return results

    def _normalize_content(self, content: str) -> str:
        """내용을 정규화하여 비교용으로 사용합니다."""
        try:
            # 기본 정규화
            normalized = content.lower().strip()

            # 공백 정규화
            import re

            normalized = re.sub(r"\s+", " ", normalized)

            # 특수문자 제거 (한글, 영문, 숫자만 유지)
            normalized = re.sub(r"[^\w\s가-힣]", "", normalized)

            return normalized

        except Exception as e:
            log.error(f"내용 정규화 실패: {e}")
            return content

    def get_search_summary(self, results: List[UnifiedSearchResult]) -> Dict[str, Any]:
        """검색 결과 요약 정보를 반환합니다."""
        try:
            if not results:
                return {"total_results": 0, "sources": {}, "avg_score": 0.0}

            # 소스별 통계
            source_stats = {}
            total_score = 0.0

            for result in results:
                source_type = result.source_type
                if source_type not in source_stats:
                    source_stats[source_type] = {
                        "count": 0,
                        "total_score": 0.0,
                        "avg_score": 0.0,
                    }

                source_stats[source_type]["count"] += 1
                source_stats[source_type]["total_score"] += result.similarity_score
                total_score += result.similarity_score

            # 평균 점수 계산
            for source_type in source_stats:
                count = source_stats[source_type]["count"]
                total = source_stats[source_type]["total_score"]
                source_stats[source_type]["avg_score"] = total / count

            return {
                "total_results": len(results),
                "sources": source_stats,
                "avg_score": total_score / len(results),
                "context_length": sum(r.context_length for r in results),
            }

        except Exception as e:
            log.error(f"검색 요약 생성 실패: {e}")
            return {"error": str(e)}

    def __del__(self):
        """리소스 정리"""
        pass

    async def aclose(self) -> None:
        """비동기 리소스 정리 함수

        VectorSearchManager가 사용하는 외부 리소스(예: DB 커넥션, 임베딩 모델 등)를 안전하게 해제합니다.
        """
        try:
            # 예시: ChromaDBClient, PDFProcessor, EmbeddingModel에 비동기 close 메서드가 있다고 가정
            if hasattr(self, "chroma_client") and hasattr(self.chroma_client, "aclose"):
                await self.chroma_client.aclose()
            if hasattr(self, "pdf_processor") and hasattr(self.pdf_processor, "aclose"):
                await self.pdf_processor.aclose()
            if (
                hasattr(self, "embedding_model")
                and hasattr(self.embedding_model, "aclose")
            ):
                await self.embedding_model.aclose()
            log.info("VectorSearchManager 리소스 비동기 정리 완료")
        except Exception as e:
            log.error(f"VectorSearchManager 리소스 정리 실패: {e}")



# 편의 함수들
def create_vector_search_manager(
    chroma_client: Optional[ChromaDBClient] = None,
    pdf_processor: Optional[PDFProcessor] = None,
    embedding_model: Optional[KoreanEmbeddingModel] = None,
    **kwargs,
) -> VectorSearchManager:
    """VectorSearchManager를 생성하는 편의 함수입니다."""
    try:
        factory_start = time.time()

        manager = VectorSearchManager(
            chroma_client=chroma_client,
            pdf_processor=pdf_processor,
            embedding_model=embedding_model,
            **kwargs,
        )

        factory_time = time.time() - factory_start
        log.info(f"🏭 VectorSearchManager 팩토리 생성 완료: {factory_time:.3f}s")

        return manager
    except Exception as e:
        log.error(f"VectorSearchManager 생성 실패: {e}")
        raise


async def search_rag_async(
    query: str, vector_manager: Optional[VectorSearchManager] = None, **kwargs
) -> List[UnifiedSearchResult]:
    """비동기 RAG 검색을 수행하는 편의 함수입니다."""
    try:
        convenience_start = time.time()

        if vector_manager is None:
            manager_start = time.time()
            vector_manager = create_vector_search_manager()
            manager_time = time.time() - manager_start
            log.info(f"🏗️ VectorSearchManager 생성 완료: {manager_time:.3f}s")

        result = await vector_manager.search_async(query, **kwargs)

        convenience_time = time.time() - convenience_start
        log.info(
            f"🎯 RAG 편의 함수 실행 완료: {len(result)}개 결과, 총 소요시간={convenience_time:.3f}s"
        )

        return result
    except Exception as e:
        log.error(f"비동기 RAG 검색 실패: {e}")
        raise
