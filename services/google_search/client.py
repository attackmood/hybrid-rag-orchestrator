"""
Google Search 클라이언트

LangChain의 GoogleSearchAPIWrapper를 사용하여
웹 검색을 수행하고 결과를 파싱/정제하는 모듈입니다.
"""

from __future__ import annotations

import time
import re
import asyncio
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from urllib.parse import urlparse
from langchain_google_community import GoogleSearchAPIWrapper
from langchain.prompts import PromptTemplate
from utils.logger import log
from config.settings import settings


@dataclass
class GoogleSearchResult:
    """Google 검색 결과 데이터 구조"""

    title: str
    link: str
    snippet: str
    source: str
    relevance_score: float
    search_rank: int
    content_length: int


class GoogleSearchClient:
    """Google 검색 클라이언트 (LangChain 전용)"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        cse_id: Optional[str] = None,
        max_results: Optional[int] = None,
        timeout: Optional[int] = None,
        ollama_client=None,
    ) -> None:
        """Google 검색 클라이언트를 초기화합니다.

        Args:
                api_key: Google Custom Search API 키
                cse_id: Custom Search Engine ID
                max_results: 최대 검색 결과 수
                timeout: 요청 타임아웃 (초)
                ollama_client: 쿼리 최적화용 Ollama 클라이언트
        """
        try:
            # 설정값 적용
            self.api_key = api_key or settings.google_search.API_KEY
            self.cse_id = cse_id or settings.google_search.CSE_ID
            self.max_results = max_results or settings.google_search.MAX_RESULTS
            self.timeout = timeout or settings.google_search.TIMEOUT

            # LangChain 래퍼 초기화 (필수)
            self._lc_wrapper = None
            try:
                self._lc_wrapper = GoogleSearchAPIWrapper(
                    google_api_key=self.api_key,
                    google_cse_id=self.cse_id,
                    k=self.max_results,
                    siterestrict=False,
                )
                log.info("Google Search Tool이 성공적으로 초기화되었습니다.")
            except Exception as e:  # pragma: no cover
                # 초기화 실패 시 즉시 에러 발생
                log.error(f"LangChain GoogleSearchAPIWrapper 초기화 실패: {e}")
                raise

            log.info(
                f"Google Search 클라이언트 초기화 완료: "
                f"max_results={self.max_results}, timeout={self.timeout}, "
                f"langchain={'ON' if self._lc_wrapper else 'OFF'}"
            )
        except Exception as e:
            log.error(f"Google Search 클라이언트 초기화 실패: {e}")
            raise

    async def process_search(
        self,
        query: str,
        max_results: Optional[int] = None,
        language: str = "ko",
        region: str = "kr",
        date_restrict: Optional[str] = None,
    ) -> List[GoogleSearchResult]:
        """비동기 Google 검색을 수행합니다 (병렬 실행용).

        Args:
            query: 검색 쿼리
            max_results: 최대 검색 결과 수
            language: 검색 언어
            region: 검색 지역
            date_restrict: 날짜 제한
        """
        try:
            if not query or not query.strip():
                log.warning("빈 검색 쿼리가 제공되었습니다.")
                return []

            search_start = time.time()

            log.info(f"🔍 비동기 Google 검색 시작: query='{query[:50]}...'")
            # 기존 방식 사용
            max_results = max_results or self.max_results
            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(
                None, lambda: self._search_with_langchain(query, max_results)
            )

            search_time = time.time() - search_start

            log.info(
                f"✅ 비동기 Google 검색 완료: {len(results)}개 결과, 소요시간={search_time:.3f}s (스레드 풀 사용)"
            )

            return results
        except Exception as e:
            log.error(f"❌ 비동기 Google 검색 실패: {e}")
            return []

    # --- LangChain path ---
    def _search_with_langchain(
        self, query: str, max_results: int
    ) -> List[GoogleSearchResult]:
        """LangChain GoogleSearchAPIWrapper로 검색합니다."""
        try:
            langchain_start = time.time()
            assert self._lc_wrapper is not None
            log.info(f"🔍 LangChain API 호출 시작")

            # LangChain API 호출
            items = self._lc_wrapper.results(query, num_results=max_results)

            langchain_time = time.time() - langchain_start
            log.info(
                f"🔗 LangChain API 호출 완료: {len(items)}개 원시 결과, 소요시간={langchain_time:.3f}s"
            )

            # 결과 파싱 및 구조화
            parse_start = time.time()
            results: List[GoogleSearchResult] = []

            for i, item in enumerate(items):
                title = item.get("title") or item.get("snippet") or ""
                link = item.get("link") or item.get("href") or ""
                snippet = item.get("snippet") or item.get("content") or ""
                source = self._extract_source(link)

                # 기본적인 관련성 점수 계산 (쿼리와의 텍스트 매칭 기반)
                relevance_score = self._calculate_relevance_score(
                    query, title, snippet, i
                )

                results.append(
                    GoogleSearchResult(
                        title=title,
                        link=link,
                        snippet=snippet,
                        source=source,
                        relevance_score=relevance_score,
                        search_rank=i + 1,
                        content_length=len(snippet),
                    )
                )

            parse_time = time.time() - parse_start
            log.info(
                f"🔄 결과 파싱 및 구조화 완료: {len(results)}개 결과, 소요시간={parse_time:.3f}s"
            )

            return results
        except Exception as e:
            log.error(f"❌ LangChain 검색 실패: {e}")
            return []

    def _extract_source(self, url: str) -> str:
        """URL에서 소스 도메인을 추출합니다."""
        try:
            parsed = urlparse(url)
            domain = parsed.netloc.lower()
            if domain.startswith("www."):
                domain = domain[4:]
            return domain or "unknown"
        except Exception:
            return "unknown"

    def _calculate_relevance_score(
        self, query: str, title: str, snippet: str, rank: int
    ) -> float:
        """기본적인 관련성 점수를 계산합니다."""
        try:
            query_lower = query.lower()
            title_lower = title.lower()
            snippet_lower = snippet.lower()

            # 기본 점수 (랭킹 기반: 1위=1.0, 2위=0.9, ...)
            base_score = max(0.1, 1.0 - (rank * 0.1))

            # 제목에서 쿼리 단어 매칭
            title_matches = sum(
                1 for word in query_lower.split() if word in title_lower
            )
            title_bonus = (
                (title_matches / len(query_lower.split())) * 0.3
                if query_lower.split()
                else 0
            )

            # 스니펫에서 쿼리 단어 매칭
            snippet_matches = sum(
                1 for word in query_lower.split() if word in snippet_lower
            )
            snippet_bonus = (
                (snippet_matches / len(query_lower.split())) * 0.2
                if query_lower.split()
                else 0
            )

            # 최종 점수 계산
            final_score = min(1.0, base_score + title_bonus + snippet_bonus)

            return round(final_score, 3)
        except Exception as e:
            log.warning(f"관련성 점수 계산 실패: {e}")
            return max(0.1, 1.0 - (rank * 0.1))  # 기본 점수만 반환

    def get_search_summary(self, results: List[GoogleSearchResult]) -> Dict[str, Any]:
        """검색 결과 요약 정보를 반환합니다."""
        try:
            summary_start = time.time()

            if not results:
                return {"total_results": 0, "avg_score": 0.0, "sources": {}}

            source_stats: Dict[str, Dict[str, float]] = {}
            total_score = 0.0

            for r in results:
                source_stats.setdefault(
                    r.source, {"count": 0, "total_score": 0.0, "avg_score": 0.0}
                )
                s = source_stats[r.source]
                s["count"] += 1
                s["total_score"] += r.relevance_score
                total_score += r.relevance_score

            for src in source_stats:
                cnt = source_stats[src]["count"]
                ttl = source_stats[src]["total_score"]
                source_stats[src]["avg_score"] = ttl / cnt if cnt else 0.0

            summary_time = time.time() - summary_start
            log.info(f"📊 검색 요약 생성 완료: 소요시간={summary_time:.3f}s")

            return {
                "total_results": len(results),
                "avg_score": total_score / len(results) if results else 0.0,
                "sources": source_stats,
                "total_content_length": sum(r.content_length for r in results),
                "best_score": (
                    max(r.relevance_score for r in results) if results else 0.0
                ),
                "worst_score": (
                    min(r.relevance_score for r in results) if results else 0.0
                ),
            }
        except Exception as e:
            log.error(f"❌ 검색 요약 생성 실패: {e}")
            return {"error": str(e)}

    async def aclose(self) -> None:
        """리소스 정리"""
        pass
