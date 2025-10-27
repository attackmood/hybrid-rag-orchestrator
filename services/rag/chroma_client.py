"""
ChromaDB 클라이언트

"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import uuid

import numpy as np

from utils.logger import log
from config.settings import settings
import chromadb
from chromadb.api import ClientAPI
from chromadb.api.models import Collection

from utils.embeddings import KoreanEmbeddingModel, create_embedding_model


@dataclass
class SearchResult:
    """크로마 검색 결과 데이터 구조"""

    ids: List[str]
    documents: List[str]
    metadatas: List[Dict[str, Any]]
    distances: List[float]


class ChromaDBClient:
    """ChromaDB 연동 클라이언트"""

    def __init__(
        self,
        persist_directory: Optional[str] = None,
        collection_name: Optional[str] = None,
        embedding_model: Optional[KoreanEmbeddingModel] = None,
        distance_threshold: Optional[float] = None,
        max_results: Optional[int] = None,
    ) -> None:
        """클라이언트를 초기화합니다.

        Args:
            persist_directory: 크로마 퍼시스턴스 디렉토리
            collection_name: 접근할 컬렉션명
            embedding_model: 임베딩 모델(미지정 시 중앙 설정으로 생성)
            distance_threshold: 검색 거리 임계값
            max_results: 최대 검색 결과 수
        """
        self.persist_directory = (
            persist_directory or settings.chroma_db.PERSIST_DIRECTORY
        )
        self.collection_name = collection_name or settings.chroma_db.COLLECTION_NAME
        self.distance_threshold = (
            distance_threshold or settings.chroma_db.DISTANCE_THRESHOLD
        )
        self.max_results = max_results or settings.chroma_db.MAX_RESULTS

        self.embedding_model = embedding_model or create_embedding_model()
        self._client: Optional[ClientAPI] = None
        self._collection: Optional[Collection] = None

        self._initialize_client()
        self._initialize_collection()

    def _initialize_client(self) -> None:
        """Persistent 클라이언트를 초기화합니다."""
        if chromadb is None:
            raise RuntimeError(
                "chromadb 패키지가 설치되어 있지 않습니다. 설치 후 다시 시도하세요."
            )

        start_time = time.time()
        try:
            self._client = chromadb.PersistentClient(path=self.persist_directory)
            duration = time.time() - start_time
            log.info(
                f"Chroma PersistentClient 초기화 완료: path={self.persist_directory} | 소요시간: {duration:.3f}s"
            )
        except Exception as exc:
            log.error(f"Chroma 클라이언트 초기화 실패: {exc}")
            raise

    def _initialize_collection(self) -> None:
        """컬렉션을 생성/가져옵니다."""
        assert self._client is not None
        try:
            # cosine 거리 사용을 명시하기 위해 메타데이터 설정
            self._collection = self._client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )
            log.info(f"Chroma 컬렉션 준비 완료: {self.collection_name}")
        except Exception as exc:
            log.error(f"컬렉션 초기화 실패: {exc}")
            raise

    def add_texts(
        self,
        texts: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None,
    ) -> List[str]:
        """문서를 컬렉션에 추가합니다.

        Args:
            texts: 문서 본문 리스트
            metadatas: 각 문서의 메타데이터 리스트(선택)
            ids: 각 문서의 고유 ID(선택, 미지정시 자동 생성)

        Returns:
            실제 저장된 문서 ID 리스트
        """
        if not texts:
            return []

        if metadatas is not None and len(metadatas) != len(texts):
            raise ValueError("metadatas의 길이는 texts와 동일해야 합니다")

        start_time = time.time()
        doc_ids = ids or [str(uuid.uuid4()) for _ in texts]

        try:
            # 임베딩 계산(배치)
            embeddings = self.embedding_model.encode_batch(texts, show_progress=False)

            assert self._collection is not None
            self._collection.add(
                ids=doc_ids,
                documents=texts,
                embeddings=embeddings.tolist(),
                metadatas=metadatas,
            )

            duration = time.time() - start_time
            log.info(
                f"문서 {len(texts)}건 저장 완료 (collection={self.collection_name}) | 소요시간: {duration:.3f}s"
            )
            return doc_ids
        except Exception as exc:
            log.error(f"문서 추가 실패: {exc}")
            raise

    def query(
        self,
        query_text: str,
        top_k: Optional[int] = None,
        where: Optional[Dict[str, Any]] = None,
        include_documents: bool = True,
    ) -> SearchResult:
        """쿼리 텍스트로 유사도 검색을 수행합니다.

        Args:
            query_text: 검색어
            top_k: 최대 검색 수(미지정 시 settings 사용)
            where: 메타데이터 필터
            include_documents: 본문 포함 여부

        Returns:
            SearchResult
        """
        if not query_text or not query_text.strip():
            raise ValueError("빈 쿼리는 검색할 수 없습니다")

        start_time = time.time()
        n_results = top_k or self.max_results

        try:
            query_vec = self.embedding_model.encode_text(query_text)

            assert self._collection is not None
            result = self._collection.query(
                query_embeddings=[query_vec.tolist()],
                n_results=n_results,
                where=where,
                include=[
                    "distances",
                    "metadatas",
                    "documents" if include_documents else None,
                    "embeddings",
                    "uris",
                    "data",
                ],
            )

            ids: List[str] = result.get("ids", [[""]])[0]
            documents: List[str] = (
                result.get("documents", [[""]])[0] if include_documents else []
            )
            metadatas: List[Dict[str, Any]] = result.get("metadatas", [[{}]])[0]
            distances: List[float] = result.get("distances", [[1.0]])[0]

            # 거리 임계값 필터링
            filtered = [
                (i, d, m, dist)
                for i, d, m, dist in zip(ids, documents, metadatas, distances)
                if dist <= self.distance_threshold
            ]

            if filtered:
                ids, documents, metadatas, distances = map(list, zip(*filtered))
            else:
                ids, documents, metadatas, distances = [], [], [], []

            duration = time.time() - start_time
            log.info(
                f"크로마 검색 완료 | 결과: {len(ids)}개 | 소요시간: {duration:.3f}s"
            )

            return SearchResult(
                ids=ids, documents=documents, metadatas=metadatas, distances=distances
            )
        except Exception as exc:
            log.error(f"크로마 검색 실패: {exc}")
            raise

    def delete_by_ids(self, ids: List[str]) -> None:
        """ID로 문서를 삭제합니다."""
        if not ids:
            return

        start_time = time.time()
        try:
            assert self._collection is not None
            self._collection.delete(ids=ids)
            duration = time.time() - start_time
            log.info(f"문서 {len(ids)}건 삭제 완료 | 소요시간: {duration:.3f}s")
        except Exception as exc:
            log.error(f"문서 삭제 실패: {exc}")
            raise

    def delete_where(self, where: Dict[str, Any]) -> None:
        """메타데이터 조건으로 문서를 삭제합니다."""
        start_time = time.time()
        try:
            assert self._collection is not None
            self._collection.delete(where=where)
            duration = time.time() - start_time
            log.info(
                f"조건부 문서 삭제 완료 | 조건: {where} | 소요시간: {duration:.3f}s"
            )
        except Exception as exc:
            log.error(f"조건부 삭제 실패: {exc}")
            raise

    def _remove_duplicates(
        self,
        result: SearchResult,
        sorted_indices: np.ndarray,
        similarity_threshold: float = 0.95,
    ) -> SearchResult:
        """
        검색 결과에서 중복/유사 문서를 제거합니다.

        Args:
            result: 원본 검색 결과
            sorted_indices: 거리 기준 정렬된 인덱스
            similarity_threshold: 유사도 임계값 (0.95 = 95% 유사)

        Returns:
            중복이 제거된 SearchResult
        """
        if not result.ids:
            return result

        start_time = time.time()
        try:
            # 정렬된 순서로 재배열
            sorted_ids = [result.ids[i] for i in sorted_indices]
            sorted_docs = [result.documents[i] for i in sorted_indices]
            sorted_metas = [result.metadatas[i] for i in sorted_indices]
            sorted_dists = [result.distances[i] for i in sorted_indices]

            # find_most_similar()를 활용한 중복 제거
            unique_indices = [0]  # 첫 번째 문서는 항상 포함
            unique_docs = [sorted_docs[0]]

            for i in range(1, len(sorted_docs)):
                current_doc = sorted_docs[i]

                # 이미 선택된 문서들과 유사도 비교 (find_most_similar 활용)
                similarity_results = self.embedding_model.find_most_similar(
                    query=current_doc,
                    candidates=unique_docs,
                    top_k=1,  # 가장 유사한 하나만 확인
                )

                # 가장 유사한 문서의 유사도가 임계값보다 낮으면 추가
                if (
                    not similarity_results
                    or similarity_results[0][2] < similarity_threshold
                ):
                    unique_indices.append(i)
                    unique_docs.append(current_doc)

            duration = time.time() - start_time
            log.info(
                f"중복 제거 완료 | {len(result.ids)}개 → {len(unique_indices)}개 | 소요시간: {duration:.3f}s"
            )

            # 중복 제거된 결과 반환
            return SearchResult(
                ids=[sorted_ids[i] for i in unique_indices],
                documents=[sorted_docs[i] for i in unique_indices],
                metadatas=[sorted_metas[i] for i in unique_indices],
                distances=[sorted_dists[i] for i in unique_indices],
            )

        except Exception as e:
            log.error(f"중복 제거 실패: {e}")
            # 실패 시 원본 결과 반환
            return result

    def query_with_ranking(
        self,
        query_text: str,
        top_k: Optional[int] = None,
        where: Optional[Dict[str, Any]] = None,
        include_documents: bool = True,
        remove_duplicates: bool = True,
        similarity_threshold: float = 0.95,
    ) -> SearchResult:
        """
        랭킹과 중복 제거가 포함된 고급 검색을 수행합니다.

        Args:
            query_text: 검색어
            top_k: 최대 검색 수
            where: 메타데이터 필터
            include_documents: 본문 포함 여부
            remove_duplicates: 중복 제거 여부
            similarity_threshold: 중복 판단 유사도 임계값

        Returns:
            정렬 및 중복 제거된 SearchResult
        """
        try:
            # 1. 기본 검색 수행
            result = self.query(query_text, top_k, where, include_documents)

            if not result.ids:
                log.info("검색 결과가 없습니다.")
                return result

            # 2. 거리 기반 정렬 (가까울수록 유사)
            sorted_indices = np.argsort(result.distances)

            # 3. 중복 제거 (선택적)
            if remove_duplicates:
                result = self._remove_duplicates(
                    result, sorted_indices, similarity_threshold
                )
                log.info(f"중복 제거 후 결과: {len(result.ids)}개")
            else:
                # 중복 제거하지 않을 경우에도 정렬 적용
                result = SearchResult(
                    ids=[result.ids[i] for i in sorted_indices],
                    documents=[result.documents[i] for i in sorted_indices],
                    metadatas=[result.metadatas[i] for i in sorted_indices],
                    distances=[result.distances[i] for i in sorted_indices],
                )
                log.info(f"정렬된 검색 결과: {len(result.ids)}개")

            # 4. 최종 결과 검증
            if result.ids:
                avg_distance = sum(result.distances) / len(result.distances)
                log.info(
                    f"평균 거리: {avg_distance:.3f}, 최고 점수: {1.0 - min(result.distances):.3f}"
                )

            return result

        except Exception as e:
            log.error(f"고급 검색 실패: {e}")
            # 실패 시 기본 검색 결과 반환
            return self.query(query_text, top_k, where, include_documents)
