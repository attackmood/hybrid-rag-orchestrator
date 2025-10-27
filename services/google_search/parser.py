"""
Google 검색 결과 파서

Google 검색 결과를 파싱하고 정제하여
RAG 시스템에서 사용할 수 있는 형태로 변환하는 모듈입니다.
"""

from __future__ import annotations

import re
import html
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from urllib.parse import urlparse, unquote

from utils.logger import log
from .client import GoogleSearchResult


@dataclass
class ParsedSearchResult:
    """파싱된 검색 결과 데이터 구조"""

    title: str
    content: str
    source: str
    url: str
    relevance_score: float
    content_type: str  # 'article', 'news', 'document', 'other'
    language: str
    word_count: int
    summary: str


class GoogleSearchResultParser:
    """Google 검색 결과 파서"""

    def __init__(self):
        """파서를 초기화합니다."""
        self.html_entities = {
            "&amp;": "&",
            "&lt;": "<",
            "&gt;": ">",
            "&quot;": '"',
            "&#39;": "'",
            "&nbsp;": " ",
            "&hellip;": "...",
        }

    def parse_results(
        self, search_results: List[GoogleSearchResult], max_content_length: int = 2000
    ) -> List[ParsedSearchResult]:
        """검색 결과를 파싱하고 정제합니다.

        Args:
            search_results: 원본 검색 결과
            max_content_length: 최대 콘텐츠 길이

        Returns:
            파싱된 검색 결과 리스트
        """
        try:
            parsed_results = []

            for result in search_results:
                parsed_result = self._parse_single_result(result, max_content_length)
                if parsed_result:
                    parsed_results.append(parsed_result)

            log.info(f"검색 결과 파싱 완료: {len(parsed_results)}개 결과")
            return parsed_results

        except Exception as e:
            log.error(f"검색 결과 파싱 실패: {e}")
            return []

    def _parse_single_result(
        self, result: GoogleSearchResult, max_content_length: int
    ) -> Optional[ParsedSearchResult]:
        """단일 검색 결과를 파싱합니다."""
        try:
            # HTML 엔티티 디코딩
            title = self._decode_html_entities(result.title)
            snippet = self._decode_html_entities(result.snippet)

            # 콘텐츠 정제
            content = self._clean_content(title, snippet)

            # 콘텐츠 길이 제한
            if len(content) > max_content_length:
                content = content[:max_content_length] + "..."

            # 콘텐츠 타입 분류
            content_type = self._classify_content_type(result.link, title, snippet)

            # 언어 감지
            language = self._detect_language(title + " " + snippet)

            # 요약 생성
            summary = self._generate_summary(content, max_length=200)

            # 단어 수 계산
            word_count = len(content.split())

            return ParsedSearchResult(
                title=title,
                content=content,
                source=result.source,
                url=result.link,
                relevance_score=result.relevance_score,
                content_type=content_type,
                language=language,
                word_count=word_count,
                summary=summary,
            )

        except Exception as e:
            log.error(f"단일 결과 파싱 실패: {e}")
            return None

    def _decode_html_entities(self, text: str) -> str:
        """HTML 엔티티를 디코딩합니다."""
        try:
            # 기본 HTML 엔티티 디코딩
            decoded = html.unescape(text)

            # 추가 엔티티 처리
            for entity, replacement in self.html_entities.items():
                decoded = decoded.replace(entity, replacement)

            return decoded
        except Exception:
            return text

    def _clean_content(self, title: str, snippet: str) -> str:
        """콘텐츠를 정제합니다."""
        try:
            # 제목과 스니펫 결합
            content = f"{title}\n\n{snippet}"

            # 불필요한 공백 제거
            content = re.sub(r"\s+", " ", content)

            # 특수 문자 정리
            content = re.sub(r'[^\w\s가-힣.,!?;:()[\]{}"\'-]', "", content)

            # 연속된 구두점 정리
            content = re.sub(r"[.!?]{2,}", ".", content)

            return content.strip()

        except Exception as e:
            log.error(f"콘텐츠 정제 실패: {e}")
            return f"{title}\n\n{snippet}"

    def _classify_content_type(self, url: str, title: str, snippet: str) -> str:
        """콘텐츠 타입을 분류합니다."""
        try:
            url_lower = url.lower()
            title_lower = title.lower()
            snippet_lower = snippet.lower()

            # 뉴스 사이트 패턴
            news_patterns = [
                r"news\.",
                r"press\.",
                r"media\.",
                r"journal\.",
                r"뉴스",
                r"기사",
                r"보도",
                r"언론",
            ]

            # 문서 사이트 패턴
            doc_patterns = [
                r"docs\.",
                r"documentation\.",
                r"manual\.",
                r"guide\.",
                r"문서",
                r"매뉴얼",
                r"가이드",
                r"설명서",
            ]

            # 블로그/기사 패턴
            article_patterns = [
                r"blog\.",
                r"article\.",
                r"post\.",
                r"column\.",
                r"블로그",
                r"칼럼",
                r"기고",
                r"포스트",
            ]

            # 뉴스 분류
            for pattern in news_patterns:
                if (
                    re.search(pattern, url_lower)
                    or re.search(pattern, title_lower)
                    or re.search(pattern, snippet_lower)
                ):
                    return "news"

            # 문서 분류
            for pattern in doc_patterns:
                if (
                    re.search(pattern, url_lower)
                    or re.search(pattern, title_lower)
                    or re.search(pattern, snippet_lower)
                ):
                    return "document"

            # 기사 분류
            for pattern in article_patterns:
                if (
                    re.search(pattern, url_lower)
                    or re.search(pattern, title_lower)
                    or re.search(pattern, snippet_lower)
                ):
                    return "article"

            return "other"

        except Exception as e:
            log.error(f"콘텐츠 타입 분류 실패: {e}")
            return "other"

    def _detect_language(self, text: str) -> str:
        """텍스트의 언어를 감지합니다."""
        try:
            # 한글 문자 패턴
            korean_pattern = re.compile(r"[가-힣]")
            # 영어 문자 패턴
            english_pattern = re.compile(r"[a-zA-Z]")

            korean_count = len(korean_pattern.findall(text))
            english_count = len(english_pattern.findall(text))

            if korean_count > english_count:
                return "ko"
            elif english_count > korean_count:
                return "en"
            else:
                return "mixed"

        except Exception as e:
            log.error(f"언어 감지 실패: {e}")
            return "unknown"

    def _generate_summary(self, content: str, max_length: int = 200) -> str:
        """콘텐츠의 요약을 생성합니다."""
        try:
            # 문장 단위로 분리
            sentences = re.split(r"[.!?]+", content)

            # 빈 문장 제거
            sentences = [s.strip() for s in sentences if s.strip()]

            if not sentences:
                return (
                    content[:max_length] + "..."
                    if len(content) > max_length
                    else content
                )

            # 첫 번째 문장을 요약으로 사용
            summary = sentences[0]

            # 길이 제한
            if len(summary) > max_length:
                summary = summary[:max_length] + "..."

            return summary

        except Exception as e:
            log.error(f"요약 생성 실패: {e}")
            return (
                content[:max_length] + "..." if len(content) > max_length else content
            )

    def filter_results_by_relevance(
        self, results: List[ParsedSearchResult], min_score: float = 0.3
    ) -> List[ParsedSearchResult]:
        """관련성 점수로 결과를 필터링합니다."""
        try:
            filtered = [r for r in results if r.relevance_score >= min_score]
            log.info(f"관련성 필터링: {len(results)} → {len(filtered)}개")
            return filtered
        except Exception as e:
            log.error(f"관련성 필터링 실패: {e}")
            return results

    def filter_results_by_content_type(
        self, results: List[ParsedSearchResult], content_types: List[str]
    ) -> List[ParsedSearchResult]:
        """콘텐츠 타입으로 결과를 필터링합니다."""
        try:
            filtered = [r for r in results if r.content_type in content_types]
            log.info(f"콘텐츠 타입 필터링: {len(results)} → {len(filtered)}개")
            return filtered
        except Exception as e:
            log.error(f"콘텐츠 타입 필터링 실패: {e}")
            return results

    def get_parsing_summary(self, results: List[ParsedSearchResult]) -> Dict[str, Any]:
        """파싱 결과 요약 정보를 반환합니다."""
        try:
            if not results:
                return {"total_results": 0, "content_types": {}, "languages": {}}

            # 콘텐츠 타입별 통계
            content_types = {}
            for result in results:
                content_type = result.content_type
                if content_type not in content_types:
                    content_types[content_type] = 0
                content_types[content_type] += 1

            # 언어별 통계
            languages = {}
            for result in results:
                language = result.language
                if language not in languages:
                    languages[language] = 0
                languages[language] += 1

            # 평균 관련성 점수
            avg_relevance = sum(r.relevance_score for r in results) / len(results)

            # 평균 단어 수
            avg_word_count = sum(r.word_count for r in results) / len(results)

            return {
                "total_results": len(results),
                "content_types": content_types,
                "languages": languages,
                "avg_relevance_score": avg_relevance,
                "avg_word_count": avg_word_count,
                "total_content_length": sum(r.word_count for r in results),
            }

        except Exception as e:
            log.error(f"파싱 요약 생성 실패: {e}")
            return {"error": str(e)}
