"""
Smart-RAG Ollama 클라이언트 (LangChain 기반)
langchain_ollama.llms.OllamaLLM을 사용한 간단하고 효율적인 구현
"""

import time
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from langchain_ollama.llms import OllamaLLM
from langchain.schema import BaseMessage, HumanMessage, SystemMessage, AIMessage
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
import httpx
from config.settings import settings
from utils.logger import log


@dataclass
class OllamaConfig:
    """Ollama 설정을 위한 데이터 클래스"""

    temperature: float = 0.7
    timeout: int = 30
    max_tokens: int = 2048
    top_p: float = 0.9
    top_k: int = 40
    repeat_penalty: float = 1.1
    stream: bool = False


@dataclass
class OllamaResponse:
    """Ollama 응답 데이터 (LangChain 호환)"""

    content: str
    model_name: str
    usage: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None


class OllamaClient:
    """LangChain 기반 Ollama 클라이언트"""

    def __init__(
        self,
        model_name: Optional[str] = None,
        base_url: Optional[str] = None,
        config: Optional[OllamaConfig] = None,
    ) -> None:
        """Ollama 클라이언트 초기화"""
        self.model_name = model_name or settings.ollama.MODEL_NAME
        self.base_url = base_url or settings.ollama.BASE_URL

        # 설정 초기화
        self.config = config or OllamaConfig()

        if not config:
            self.config = OllamaConfig()

        # LangChain OllamaLLM 인스턴스 생성
        self.llm = self._init_llm()

        log.info(
            f"LangChain Ollama 클라이언트 초기화 완료 | 모델: {self.model_name} | 베이스 URL: {self.base_url}"
        )

    def _init_llm(self, **kwargs):
        """LangChain OllamaLLM 인스턴스 생성"""
        try:
            return OllamaLLM(
                model=self.model_name,
                base_url=self.base_url,
                temperature=kwargs.get("temperature", settings.ollama.TEMPERATURE),
                timeout=kwargs.get("timeout", settings.ollama.TIMEOUT),
                num_predict=kwargs.get("max_tokens", settings.ollama.MAX_TOKENS),
                top_p=kwargs.get("top_p", 0.9),
                top_k=kwargs.get("top_k", 40),
                repeat_penalty=kwargs.get("repeat_penalty", 1.1),
                # stop 파라미터 제거 (AgentExecutor와의 충돌 방지)
                callback_manager=(
                    CallbackManager([StreamingStdOutCallbackHandler()])
                    if kwargs.get("stream", False)
                    else None
                ),
            )
        except Exception as e:
            log.error(f"LangChain OllamaLLM 초기화 실패: {e}")
            raise

    async def generate(
        self, prompt: str, system: Optional[str] = None, **kwargs
    ) -> OllamaResponse:
        """비동기 텍스트 생성 (LangChain 기반)"""
        start_time = time.time()
        try:
            # 시스템 메시지가 있으면 메시지 리스트로 구성
            if system:
                messages = [SystemMessage(content=system), HumanMessage(content=prompt)]
                response = await self.llm.ainvoke(messages)
            else:
                response = await self.llm.ainvoke(prompt)

            content = (
                response.content if hasattr(response, "content") else str(response)
            )

            duration = time.time() - start_time

            # JSON 응답 미리보기 (로그에 영향 없도록)
            preview = content[:100].replace("\n", " ").replace("\r", "")
            log.info(
                f"텍스트 생성 완료 | 응답길이: {len(content)}자 | 소요시간: {duration:.3f}s"
            )
            log.info(f"응답 미리보기: {preview}...")

            # 응답 파싱
            ollama_response = OllamaResponse(
                content=content,
                model_name=self.model_name,
                usage=getattr(response, "usage", None),
                metadata=getattr(response, "metadata", None),
            )

            return ollama_response

        except Exception as e:
            log.error(f"❌ 텍스트 생성 실패: {e}")
            raise

    async def chat(self, messages: List[BaseMessage], **kwargs) -> OllamaResponse:
        """비동기 채팅 (LangChain 기반)"""
        start_time = time.time()
        try:
            # LangChain 메시지 리스트로 직접 호출
            response = await self.llm.ainvoke(messages)

            content = (
                response.content if hasattr(response, "content") else str(response)
            )

            duration = time.time() - start_time

            log.info(
                f"채팅 완료 | 응답내용: {content[:100]}... | 소요시간: {duration:.3f}s"
            )

            # 응답 파싱
            ollama_response = OllamaResponse(
                content=content,
                model_name=self.model_name,
                usage=getattr(response, "usage", None),
                metadata=getattr(response, "metadata", None),
            )

            return ollama_response

        except Exception as e:
            log.error(f"❌ 채팅 실패: {e}")
            raise

    async def list_models(self) -> List[str]:
        """사용 가능한 모델 목록을 조회합니다."""
        start_time = time.time()
        try:
            # Ollama API를 통해 모델 목록 조회
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{self.base_url}/api/tags")
                response.raise_for_status()

                data = response.json()
                models = [model["name"] for model in data.get("models", [])]

                duration = time.time() - start_time
                log.info(
                    f"모델 목록 조회 완료 | {len(models)}개 모델 | 소요시간: {duration:.3f}s"
                )

                return models

        except httpx.TimeoutException:
            log.error("❌ 모델 목록 조회 타임아웃")
            raise
        except httpx.HTTPStatusError as e:
            log.error(f"❌ 모델 목록 조회 HTTP 오류: {e.response.status_code}")
            raise
        except httpx.RequestError as e:
            log.error(f"❌ 모델 목록 조회 요청 오류: {e}")
            raise
        except Exception as e:
            log.error(f"❌ 모델 목록 조회 실패: {e}")
            raise

    async def health_check_async(self) -> bool:
        """Ollama 서버 상태 확인(비동기, httpx)"""
        url = f"{self.base_url}/api/ps"
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(url)
                resp.raise_for_status()
            log.info("Ollama 서버 상태 정상")
            return True
        except httpx.TimeoutException as e:
            log.error(f"서버 상태 확인 타임아웃: {e}")
            return False
        except httpx.HTTPStatusError as e:
            log.error(f"서버 상태 확인 HTTP 오류 ({e.response.status_code}): {e}")
            return False
        except httpx.RequestError as e:
            log.error(f"서버 상태 확인 연결 오류: {e}")
            return False
        except Exception as e:
            log.error(f"서버 상태 확인 실패: {e}")
            return False


def create_ollama_client(
    model_name: Optional[str] = None, base_url: Optional[str] = None, **kwargs
) -> OllamaClient:
    """Ollama 클라이언트 생성"""
    try:
        return OllamaClient(model_name=model_name, base_url=base_url, **kwargs)
    except Exception as e:
        log.error(f"Ollama 클라이언트 생성 실패: {e}")
        raise
    
    
