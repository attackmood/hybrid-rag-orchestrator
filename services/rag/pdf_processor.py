"""
실시간 PDF 처리 모듈 (Dynamic RAG)

settings.py 설정을 활용하여 PDF를 청킹하고 임시 벡터 저장소에 저장합니다.
"""

from __future__ import annotations

import os
import tempfile
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import time
import numpy as np
import fitz  # PyMuPDF

from utils.logger import log
from config.settings import settings
from utils.embeddings import KoreanEmbeddingModel, create_embedding_model


@dataclass
class PDFChunk:
    """PDF 청크 정보"""

    id: str
    text: str
    page: int
    chunk_index: int
    metadata: Dict[str, Any]
    embedding: Optional[List[float]] = None
    created_at: float = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = time.time()


@dataclass
class PDFProcessingResult:
    """PDF 처리 결과"""

    file_hash: str
    filename: str
    total_pages: int
    total_chunks: int
    chunks: List[PDFChunk]
    processing_time: float
    file_size: int


class PDFProcessor:
    """PDF 처리 및 청킹 클래스"""

    def __init__(
        self,
        embedding_model: Optional[KoreanEmbeddingModel] = None,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        temp_dir: Optional[str] = None,
    ) -> None:
        """
        PDF 프로세서를 초기화합니다.

        Args:
            embedding_model: 임베딩 모델
            chunk_size: 청크 크기 (문자 기준)
            chunk_overlap: 청크 간 중복 길이
            temp_dir: 임시 파일 저장 디렉토리
        """
        try:
            # 설정값 적용
            self.chunk_size = chunk_size or settings.rag.CHUNK_SIZE
            self.chunk_overlap = chunk_overlap or settings.rag.CHUNK_OVERLAP
            self.temp_dir = temp_dir or settings.rag.PDF_TEMP_DIR

            # 임베딩 모델 초기화
            self.embedding_model = embedding_model or create_embedding_model()

            # 임시 벡터 저장소 (메모리 기반)
            self._vector_store: Dict[str, PDFChunk] = {}
            self._file_chunks: Dict[str, List[str]] = {}  # file_hash -> chunk_ids

            # 임시 디렉토리 생성
            Path(self.temp_dir).mkdir(parents=True, exist_ok=True)

            log.info(
                f"PDF 프로세서 초기화 완료: chunk_size={self.chunk_size}, overlap={self.chunk_overlap}"
            )
        except Exception as e:
            log.error(f"PDF 프로세서 초기화 실패: {e}")
            raise

    def process_pdf(
        self, pdf_path: str, metadata: Optional[Dict[str, Any]] = None
    ) -> PDFProcessingResult:
        """
        PDF 파일을 처리하여 청킹하고 임시 벡터 저장소에 저장합니다.

        Args:
            pdf_path: PDF 파일 경로
            metadata: 추가 메타데이터

        Returns:
            PDFProcessingResult
        """
        start_time = time.time()

        try:
            # 1. PDF 파일 검증
            if not os.path.exists(pdf_path):
                raise FileNotFoundError(f"PDF 파일을 찾을 수 없습니다: {pdf_path}")

            # 2. 파일 해시 생성
            file_hash = self._calculate_file_hash(pdf_path)
            filename = Path(pdf_path).name
            file_size = os.path.getsize(pdf_path)

            # 3. 텍스트 추출
            text_content = self._extract_text(pdf_path)
            if not text_content.strip():
                raise ValueError("PDF에서 텍스트를 추출할 수 없습니다")

            # 4. 청킹
            chunks = self._create_chunks(text_content, file_hash, filename, metadata)

            # 5. 임베딩 생성 및 저장
            self._process_chunks(chunks)

            # 6. 임시 벡터 저장소에 저장
            self._store_chunks(file_hash, chunks)

            processing_time = time.time() - start_time

            result = PDFProcessingResult(
                file_hash=file_hash,
                filename=filename,
                total_pages=len(text_content.split("\f")),  # 페이지 구분자로 추정
                total_chunks=len(chunks),
                chunks=chunks,
                processing_time=processing_time,
                file_size=file_size,
            )

            log.info(
                f"PDF 처리 완료: {filename} -> {len(chunks)}개 청크, {processing_time:.2f}초"
            )
            return result

        except Exception as e:
            log.error(f"PDF 처리 실패 ({pdf_path}): {e}")
            raise

    def _calculate_file_hash(self, file_path: str) -> str:
        """파일의 MD5 해시를 계산합니다."""
        try:
            hash_md5 = hashlib.md5()
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()
        except Exception as e:
            log.error(f"파일 해시 계산 실패 ({file_path}): {e}")
            raise

    def _extract_text(self, pdf_path: str) -> str:
        """PDF에서 텍스트를 추출합니다."""
        try:
            return self._extract_text_pymupdf(pdf_path)
        except Exception as e:
            log.error(f"텍스트 추출 실패: {e}")
            raise

    def _extract_text_pymupdf(self, pdf_path: str) -> str:
        """PyMuPDF를 사용하여 텍스트 추출"""
        try:
            doc = fitz.open(pdf_path)
            text_content = []

            for page_num in range(len(doc)):
                try:
                    page = doc.load_page(page_num)
                    text = page.get_text()
                    text_content.append(text)
                except Exception as e:
                    log.warning(f"페이지 {page_num} 텍스트 추출 실패: {e}")
                    text_content.append("")  # 빈 페이지로 처리

            doc.close()
            return "\f".join(text_content)  # 페이지 구분자
        except Exception as e:
            log.error(f"PyMuPDF 텍스트 추출 실패 ({pdf_path}): {e}")
            raise

    def _create_chunks(
        self,
        text: str,
        file_hash: str,
        filename: str,
        metadata: Optional[Dict[str, Any]],
    ) -> List[PDFChunk]:
        """텍스트를 청킹합니다."""
        try:
            chunks = []
            chunk_index = 0

            # 페이지별로 분리
            pages = text.split("\f")

            for page_num, page_text in enumerate(pages):
                try:
                    if not page_text.strip():
                        continue

                    # 페이지 내에서 청킹
                    page_chunks = self._chunk_text(page_text, page_num)

                    for chunk_text in page_chunks:
                        chunk_id = f"{file_hash}_page{page_num}_chunk{chunk_index}"

                        chunk_metadata = {
                            "source_file": filename,
                            "file_hash": file_hash,
                            "page": page_num,
                            "chunk_index": chunk_index,
                            "chunk_size": len(chunk_text),
                            **(metadata or {}),
                        }

                        chunk = PDFChunk(
                            id=chunk_id,
                            text=chunk_text,
                            page=page_num,
                            chunk_index=chunk_index,
                            metadata=chunk_metadata,
                        )

                        chunks.append(chunk)
                        chunk_index += 1
                except Exception as e:
                    log.warning(f"페이지 {page_num} 청킹 실패: {e}")
                    continue

            return chunks
        except Exception as e:
            log.error(f"청크 생성 실패: {e}")
            raise

    def _chunk_text(self, text: str, page_num: int) -> List[str]:
        """단일 페이지 텍스트를 청킹합니다."""
        try:
            if len(text) <= self.chunk_size:
                return [text]

            chunks = []
            start = 0

            while start < len(text):
                try:
                    end = start + self.chunk_size

                    # 문장 경계에서 자르기 시도
                    if end < len(text):
                        # 마침표, 물음표, 느낌표로 끝나는 지점 찾기
                        sentence_end = text.rfind(".", start, end)
                        if sentence_end == -1:
                            sentence_end = text.rfind("?", start, end)
                        if sentence_end == -1:
                            sentence_end = text.rfind("!", start, end)

                        if sentence_end > start + self.chunk_size // 2:
                            end = sentence_end + 1

                    chunk = text[start:end].strip()
                    if chunk:
                        chunks.append(chunk)

                    start = end - self.chunk_overlap
                    if start >= len(text):
                        break
                except Exception as e:
                    log.warning(
                        f"청크 생성 중 오류 (페이지 {page_num}, 시작점 {start}): {e}"
                    )
                    break

            return chunks
        except Exception as e:
            log.error(f"텍스트 청킹 실패 (페이지 {page_num}): {e}")
            return [text]  # 실패 시 전체 텍스트를 하나의 청크로 반환

    def _process_chunks(self, chunks: List[PDFChunk]) -> None:
        """청크들을 임베딩하여 벡터화합니다.
        -> 청크: 긴 문장을 작은 문장으로 나눈 것
        -> 임베딩: 문장을 컴퓨터가 비교할 수 있는 숫자로 바꾸는 작업
        -> 벡터화: 임베딩 벡터들의 2D 배열로 만드는 것
        """

        try:
            if not chunks:
                return

            # 배치 임베딩
            texts = [chunk.text for chunk in chunks]
            embeddings = self.embedding_model.encode_batch(texts, show_progress=False)

            # 각 청크에 임베딩 할당
            for chunk, embedding in zip(chunks, embeddings):
                try:
                    chunk.embedding = embedding.tolist()
                except Exception as e:
                    log.warning(f"청크 임베딩 할당 실패 (ID: {chunk.id}): {e}")
                    chunk.embedding = None
        except Exception as e:
            log.error(f"청크 임베딩 처리 실패: {e}")
            raise

    def _store_chunks(self, file_hash: str, chunks: List[PDFChunk]) -> None:
        """청크들을 임시 벡터 저장소에 저장합니다."""
        try:
            chunk_ids = []

            for chunk in chunks:
                try:
                    self._vector_store[chunk.id] = chunk
                    chunk_ids.append(chunk.id)
                except Exception as e:
                    log.warning(f"청크 저장 실패 (ID: {chunk.id}): {e}")
                    continue

            self._file_chunks[file_hash] = chunk_ids

            log.info(f"청크 {len(chunks)}개를 임시 벡터 저장소에 저장: {file_hash}")
        except Exception as e:
            log.error(f"청크 저장소 저장 실패: {e}")
            raise

    def search_chunks(
        self, query: str, top_k: int = 5, file_filter: Optional[str] = None
    ) -> List[Tuple[PDFChunk, float]]:
        """
        임시 벡터 저장소에서 유사한 청크를 검색합니다.

        Args:
            query: 검색 쿼리
            top_k: 최대 결과 수
            file_filter: 특정 파일 해시로 필터링

        Returns:
            (청크, 유사도 점수) 튜플 리스트
        """
        start_time = time.time()
        try:
            if not self._vector_store:
                return []

            # 검색 대상 청크들 선별
            search_chunks = []
            if file_filter:
                # 특정 파일의 청크만 검색
                chunk_ids = self._file_chunks.get(file_filter, [])
                search_chunks = [
                    self._vector_store[cid]
                    for cid in chunk_ids
                    if cid in self._vector_store and self._vector_store[cid].embedding
                ]
            else:
                # 모든 청크 검색 (임베딩이 있는 것만)
                search_chunks = [
                    chunk
                    for chunk in self._vector_store.values()
                    if chunk.embedding and len(chunk.embedding) > 0
                ]

            if not search_chunks:
                return []

            # 텍스트 리스트 생성
            candidate_texts = [chunk.text for chunk in search_chunks]

            # find_most_similar() 사용하여 검색
            similarity_results = self.embedding_model.find_most_similar(
                query=query, candidates=candidate_texts, top_k=top_k
            )

            # 결과를 (PDFChunk, similarity) 형태로 변환
            chunk_results = []
            for idx, text, similarity_score in similarity_results:
                chunk = search_chunks[idx]
                chunk_results.append((chunk, similarity_score))

            processing_time = time.time() - start_time
            log.info(
                f"청크 검색 완료: {len(chunk_results)}개 결과, {processing_time:.2f}초"
            )

            return chunk_results

        except Exception as e:
            log.error(f"청크 검색 실패: {e}")
            return []

    def get_file_chunks(self, file_hash: str) -> List[PDFChunk]:
        """특정 파일의 모든 청크를 반환합니다."""
        # 잠재적 용도: 특정 파일의 청크 조회, 디버깅, 관리 UI
        try:
            chunk_ids = self._file_chunks.get(file_hash, [])
            chunks = []
            for cid in chunk_ids:
                try:
                    if cid in self._vector_store:
                        chunks.append(self._vector_store[cid])
                except Exception as e:
                    log.warning(f"청크 조회 실패 (ID: {cid}): {e}")
                    continue
            return chunks
        except Exception as e:
            log.error(f"파일 청크 조회 실패 (file_hash: {file_hash}): {e}")
            return []

    def remove_file(self, file_hash: str) -> None:
        """특정 파일의 모든 청크를 임시 저장소에서 제거합니다."""
        # 잠재적 용도: 파일 삭제 시 메모리 정리, 임시 파일 관리
        try:
            if file_hash not in self._file_chunks:
                return

            chunk_ids = self._file_chunks[file_hash]
            removed_count = 0

            for chunk_id in chunk_ids:
                try:
                    if chunk_id in self._vector_store:
                        del self._vector_store[chunk_id]
                        removed_count += 1
                except Exception as e:
                    log.warning(f"청크 제거 실패 (ID: {chunk_id}): {e}")
                    continue

            del self._file_chunks[file_hash]
            log.info(
                f"파일 {file_hash}의 청크 {removed_count}개를 임시 저장소에서 제거"
            )
        except Exception as e:
            log.error(f"파일 제거 실패 (file_hash: {file_hash}): {e}")
            raise

    def clear_all(self) -> None:
        """모든 임시 데이터를 정리합니다."""
        # 잠재적 용도: 세션 종료, 메모리 초기화, 테스트 간 정리
        try:
            chunks_count = len(self._vector_store)
            files_count = len(self._file_chunks)

            self._vector_store.clear()
            self._file_chunks.clear()

            log.info(
                f"임시 벡터 저장소 완전 정리: 청크 {chunks_count}개, 파일 {files_count}개"
            )
        except Exception as e:
            log.error(f"저장소 정리 실패: {e}")
            raise

    def get_stats(self) -> Dict[str, Any]:
        """현재 상태 통계를 반환합니다."""
        # 잠재적 용도: 시스템 상태 모니터링, 관리자 대시보드
        try:
            total_chunks = len(self._vector_store)
            total_files = len(self._file_chunks)

            return {
                "total_chunks": total_chunks,
                "total_files": total_files,
                "file_hashes": list(self._file_chunks.keys()),
                "chunk_ids": list(self._vector_store.keys()),
            }
        except Exception as e:
            log.error(f"통계 정보 조회 실패: {e}")
            return {
                "total_chunks": 0,
                "total_files": 0,
                "file_hashes": [],
                "chunk_ids": [],
                "error": str(e),
            }


# 편의 함수
def create_pdf_processor(**kwargs) -> PDFProcessor:
    """PDF 프로세서를 생성하는 편의 함수입니다."""
    try:
        return PDFProcessor(**kwargs)
    except Exception as e:
        log.error(f"PDF 프로세서 생성 실패: {e}")
        raise
