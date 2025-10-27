"""
임베딩 시스템 모듈

한국어 텍스트를 위한 벡터 임베딩 및 유사도 계산 기능을 제공합니다.
"""

import re
import hashlib
import pickle
import time
from typing import List, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from utils.logger import log
from config.settings import settings


@dataclass
class EmbeddingConfig:
    """임베딩 설정 클래스"""

    model_name: str = settings.embedding.EMBEDDING_MODEL
    device: str = settings.embedding.EMBEDDING_DEVICE
    batch_size: int = settings.embedding.EMBEDDING_BATCH_SIZE
    max_seq_length: int = settings.embedding.EMBEDDING_MAX_LENGTH
    cache_dir: str = settings.embedding.EMBEDDING_CACHE_DIR
    enable_cache: bool = settings.embedding.ENABLE_CACHE
    normalize_vectors: bool = settings.embedding.NORMALIZE_VECTORS
    force_cpu: bool = True  # GPU 메모리 문제 방지를 위해 기본값을 True로 설정


class TextPreprocessor:
    """텍스트 전처리 전담 클래스"""

    def __init__(self, max_length: Optional[int] = None):
        """
        텍스트 전처리기를 초기화합니다.

        Args:
            max_length: 최대 텍스트 길이 (토큰 기준)
        """
        self.max_length = max_length or settings.embedding.EMBEDDING_MAX_LENGTH

        # 한국어 정규식 패턴들
        self.patterns = {
            "whitespace": re.compile(r"\s+"),
            "special_chars": re.compile(r"[^\w\s가-힣]"),
        }

    def preprocess(self, text: str) -> str:
        """
        텍스트를 전처리합니다.

        Args:
            text: 원본 텍스트

        Returns:
            전처리된 텍스트
        """
        if not text or not text.strip():
            return ""

        # 1. 기본 정리
        text = text.strip()

        # 2. 줄바꿈 및 탭 처리
        text = text.replace("\n", " ").replace("\r", " ").replace("\t", " ")

        # 3. 여러 공백을 하나로
        text = self.patterns["whitespace"].sub(" ", text)

        # 4. 길이 제한 (대략적인 토큰 기준)
        if len(text) > self.max_length * 4:  # 한국어는 토큰당 평균 4자
            text = text[: self.max_length * 4]

        return text.strip()


class VectorUtils:
    """벡터 연산 유틸리티 클래스"""

    @staticmethod
    def normalize_vector(vector: np.ndarray) -> np.ndarray:
        """L2 정규화를 수행합니다."""
        norm = np.linalg.norm(vector)
        if norm == 0:
            return vector
        return vector / norm

    @staticmethod
    def normalize_batch(vectors: np.ndarray) -> np.ndarray:
        """배치 벡터들을 정규화합니다."""
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        return vectors / norms

    @staticmethod
    def cosine_similarity_batch(
        query_vec: np.ndarray, vectors: np.ndarray
    ) -> np.ndarray:
        """쿼리 벡터와 여러 벡터들 간의 코사인 유사도를 계산합니다."""
        # 정규화 확인
        if not VectorUtils.is_normalized(query_vec):
            query_vec = VectorUtils.normalize_vector(query_vec)

        if vectors.ndim == 2:
            vectors = VectorUtils.normalize_batch(vectors)

        similarities = np.dot(vectors, query_vec)
        return np.clip(similarities, -1.0, 1.0)

    @staticmethod
    def is_normalized(vector: np.ndarray, tolerance: float = 1e-6) -> bool:
        """벡터가 정규화되었는지 확인합니다."""
        norm = np.linalg.norm(vector)
        return abs(norm - 1.0) < tolerance


class EmbeddingCache:
    """임베딩 결과를 캐싱하는 클래스"""

    def __init__(self, cache_dir: Optional[str] = None, memory_limit: int = 1000):
        """
        임베딩 캐시를 초기화합니다.

        Args:
            cache_dir: 캐시 저장 디렉토리
            memory_limit: 메모리 캐시 최대 항목 수
        """
        self.cache_dir = Path(cache_dir or settings.embedding.EMBEDDING_CACHE_DIR)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.memory_limit = memory_limit
        self._memory_cache = {}

    def _get_cache_key(self, text: str, model_hash: str) -> str:
        """캐시 키를 생성합니다."""
        content = f"{model_hash}:{text}"
        return hashlib.md5(content.encode()).hexdigest()

    def get(self, text: str, model_hash: str) -> Optional[np.ndarray]:
        """캐시에서 임베딩을 가져옵니다."""
        try:
            key = self._get_cache_key(text, model_hash)

            # 메모리 캐시 확인
            if key in self._memory_cache:
                return self._memory_cache[key]

            # 디스크 캐시 확인
            cache_file = self.cache_dir / f"{key}.pkl"
            if cache_file.exists():
                try:
                    with open(cache_file, "rb") as f:
                        embedding = pickle.load(f)

                    # 메모리 캐시에 추가 (크기 제한)
                    if len(self._memory_cache) < self.memory_limit:
                        self._memory_cache[key] = embedding

                    return embedding
                except (pickle.PickleError, EOFError, ValueError) as e:
                    log.warning(f"캐시 파일 로드 실패: {e}")
                    # 손상된 캐시 파일 제거
                    cache_file.unlink(missing_ok=True)
                except Exception as e:
                    log.error(f"캐시 파일 처리 중 예상치 못한 오류: {e}")

        except Exception as e:
            log.error(f"캐시 조회 실패: {e}")

        return None

    def set(self, text: str, model_hash: str, embedding: np.ndarray) -> None:
        """캐시에 임베딩을 저장합니다."""
        try:
            key = self._get_cache_key(text, model_hash)

            # 메모리 캐시에 저장 (크기 제한)
            if len(self._memory_cache) >= self.memory_limit:
                # LRU 방식으로 오래된 항목 제거 (간단한 구현)
                oldest_key = next(iter(self._memory_cache))
                del self._memory_cache[oldest_key]

            self._memory_cache[key] = embedding

            # 디스크 캐시에 저장
            cache_file = self.cache_dir / f"{key}.pkl"
            try:
                with open(cache_file, "wb") as f:
                    pickle.dump(embedding, f, protocol=pickle.HIGHEST_PROTOCOL)
            except (pickle.PickleError, OSError, IOError) as e:
                log.warning(f"캐시 파일 저장 실패: {e}")
            except Exception as e:
                log.error(f"캐시 파일 저장 중 예상치 못한 오류: {e}")

        except Exception as e:
            log.error(f"캐시 저장 실패: {e}")

    def clear_memory(self) -> None:
        """메모리 캐시를 정리합니다."""
        self._memory_cache.clear()

    def clear_disk(self) -> None:
        """디스크 캐시를 정리합니다."""
        try:
            for cache_file in self.cache_dir.glob("*.pkl"):
                cache_file.unlink()
            log.info("디스크 캐시 정리 완료")
        except Exception as e:
            log.error(f"디스크 캐시 정리 실패: {e}")


class KoreanEmbeddingModel:
    """한국어 텍스트 임베딩을 위한 모델 클래스"""

    def __init__(self, config: Optional[EmbeddingConfig] = None):
        """
        한국어 임베딩 모델을 초기화합니다.

        Args:
            config: 임베딩 설정
        """
        self.config: EmbeddingConfig = config or EmbeddingConfig()
        self.model = None
        self.model_hash = None

        # 구성 요소 초기화
        self.preprocessor = TextPreprocessor(max_length=self.config.max_seq_length)
        self.vector_utils = VectorUtils()
        # enable_cache 옵션이 True이면 EmbeddingCache 인스턴스를 생성하고, 그렇지 않으면 None을 할당합니다.
        self.cache = (
            EmbeddingCache(self.config.cache_dir) if self.config.enable_cache else None
        )

        try:
            self._load_model()
            log.info(f"임베딩 모델 로드 완료: {self.config.model_name} ({self.config.device})")
        except Exception as e:
            log.error(f"모델 로드 실패: {e}")
            raise

    def _load_model(self) -> None:
        """임베딩 모델을 로드합니다."""
        start_time = time.time()
        try:
            self.model = SentenceTransformer(
                self.config.model_name, device=self.config.device
            )

            # 모델을 CPU로 강제 설정 (GPU 메모리 문제 방지)
            if self.config.force_cpu:
                self.model = self.model.to("cpu")
                log.info(f"디바이스를 강제로 'cpu'로 설정합니다.")

            # 모델 해시 생성 (캐시 시스템에서 필요)
            self.model_hash = hashlib.md5(
                f"{self.config.model_name}:{self.config.max_seq_length}".encode()
            ).hexdigest()[:8]

            duration = time.time() - start_time
            log.info(
                f"임베딩 모델 로드 완료: {self.config.model_name} | 소요시간: {duration:.3f}s"
            )

        except Exception as e:
            log.error(f"❌ 임베딩 모델 로드 실패: {e}")
            raise

    def encode_text(
        self, text: str, use_cache: bool = True, preprocess: bool = True
    ) -> np.ndarray:
        """
        단일 텍스트를 임베딩 벡터로 변환합니다.

        Args:
            text: 임베딩할 텍스트
            use_cache: 캐시 사용 여부
            preprocess: 전처리 수행 여부

        Returns:
            임베딩 벡터 (numpy array)
        """
        if not text or not text.strip():
            raise ValueError("빈 텍스트는 임베딩할 수 없습니다")

        # 캐시 확인
        if use_cache and self.cache:
            cached_embedding = self.cache.get(text, self.model_hash)
            if cached_embedding is not None:
                return cached_embedding

        try:
            # 텍스트 전처리
            if preprocess:
                processed_text = self.preprocessor.preprocess(text)
                if not processed_text:
                    raise ValueError("전처리 후 텍스트가 비어있습니다")
            else:
                processed_text = text

            # 임베딩 생성
            embedding = self.model.encode(processed_text, convert_to_numpy=True)
            # 벡터 정규화
            if self.config.normalize_vectors:
                embedding = self.vector_utils.normalize_vector(embedding)

            # 캐시 저장
            if use_cache and self.cache:
                self.cache.set(text, self.model_hash, embedding)

            return embedding

        except Exception as e:
            log.error(f"텍스트 임베딩 실패 ('{text[:50]}...'): {e}")
            raise

    def encode_batch(
        self,
        texts: List[str],
        show_progress: bool = True,
        use_cache: bool = False,
        preprocess: bool = True,
    ) -> np.ndarray:
        """
        여러 텍스트를 배치로 임베딩합니다.

        Args:
            texts: 임베딩할 텍스트 리스트
            show_progress: 진행률 표시 여부
            use_cache: 캐시 사용 여부
            preprocess: 전처리 수행 여부

        Returns:
            임베딩 벡터들의 2D 배열
        """
        if not texts:
            return np.array([]).reshape(0, -1)

        start_time = time.time()
        try:
            embeddings = []
            uncached_texts = []
            uncached_indices = []

            # 캐시된 임베딩 확인
            for i, text in enumerate(texts):
                if use_cache and self.cache:
                    cached = self.cache.get(text, self.model_hash)
                    if cached is not None:
                        embeddings.append((i, cached))
                        continue

                uncached_texts.append(text)
                uncached_indices.append(i)

            # 캐시되지 않은 텍스트들 처리
            if uncached_texts:
                try:
                    # 전처리
                    if preprocess:
                        processed_texts = [
                            self.preprocessor.preprocess(text)
                            for text in uncached_texts
                        ]
                    else:
                        processed_texts = uncached_texts

                    # 배치 임베딩
                    batch_embeddings = self.model.encode(
                        processed_texts,
                        batch_size=self.config.batch_size,
                        show_progress_bar=show_progress,
                        convert_to_numpy=True,
                    )

                    # 벡터 정규화
                    if self.config.normalize_vectors:
                        batch_embeddings = self.vector_utils.normalize_batch(
                            batch_embeddings
                        )

                    # 결과 추가 및 캐시 저장
                    for i, (original_idx, embedding) in enumerate(
                        zip(uncached_indices, batch_embeddings)
                    ):
                        embeddings.append((original_idx, embedding))

                        # 캐시 저장
                        if use_cache and self.cache:
                            self.cache.set(
                                uncached_texts[i], self.model_hash, embedding
                            )

                except Exception as e:
                    log.error(f"배치 임베딩 처리 실패: {e}")
                    raise

            # 원래 순서로 정렬
            embeddings.sort(key=lambda x: x[0])
            result = np.array([emb for _, emb in embeddings])

            duration = time.time() - start_time
            cache_hits = len(texts) - len(uncached_texts)
            log.info(
                f"배치 임베딩 완료 | 총 {len(texts)}개 | 캐시 적중: {cache_hits}개 | 소요시간: {duration:.3f}s"
            )

            return result

        except Exception as e:
            log.error(f"배치 임베딩 실패: {e}")
            raise

    def find_most_similar(
        self, query: str, candidates: List[str], top_k: int = 5
    ) -> List[Tuple[int, str, float]]:
        """
        쿼리와 가장 유사한 후보 텍스트들을 찾습니다.

        Returns:
            (인덱스, 텍스트, 유사도 점수) 튜플의 리스트
        """
        if not candidates:
            return []

        start_time = time.time()
        try:
            query_vec = self.encode_text(query)  # 쿼리 임베딩
            candidate_vecs = self.encode_batch(candidates)  # 후보 텍스트 임베딩

            similarities = self.vector_utils.cosine_similarity_batch(
                query_vec, candidate_vecs
            )  # 코사인 유사도 계산

            # 상위 k개 선택
            top_indices = np.argsort(similarities)[::-1][:top_k]

            results = []
            for idx in top_indices:
                results.append((idx, candidates[idx], float(similarities[idx])))

            duration = time.time() - start_time
            log.info(
                f"유사도 검색 완료 | 쿼리 vs {len(candidates)}개 후보 | 상위 {len(results)}개 반환 | 소요시간: {duration:.3f}s"
            )

            return results

        except Exception as e:
            log.error(f"유사도 검색 실패: {e}")
            return []


# 편의 함수들
def create_embedding_model(model_name: Optional[str] = None) -> KoreanEmbeddingModel:
    """한국어 임베딩 모델을 생성하는 편의 함수입니다."""
    try:
        if model_name:
            config = EmbeddingConfig(model_name=model_name)
        else:
            config = EmbeddingConfig()

        return KoreanEmbeddingModel(config)
    except Exception as e:
        log.error(f"임베딩 모델 생성 실패: {e}")
        raise
