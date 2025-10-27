"""
하이브리드 라우터: ReAct Router + Multi-LangGraph Router의 장점 통합

주요 특징:
1. LLM 기반 의도 분석 (ReAct Router에서 가져옴)
2. 병렬 도구 실행 및 점수 기반 통합 (Multi-LangGraph Router에서 가져옴)
3. 실제 서비스 연동 (MCP, Google Search, RAG)
4. 적응형 워크플로우 (복잡도 기반 분기)
5. 다중 워크플로우 패턴 지원
"""

import asyncio
import json
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, TypedDict, Annotated
import operator

from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langgraph.graph import StateGraph, END

from utils.logger import log
from core.ollama_client import create_ollama_client
from core.tools_registry import ToolsRegistry, ToolSchema, create_tools_registry


# ============================================================================
# 하이브리드 라우터 상태 정의
# ============================================================================


class HybridQueryState(TypedDict):
    """하이브리드 라우터 상태"""

    messages: Annotated[List[BaseMessage], operator.add]
    query: str
    intent_analysis: Dict[str, Any]
    selected_tools: List[str]
    parallel_results: Dict[str, Any]
    scores: Dict[str, float]
    final_answer: str
    processing_stage: str
    complexity_score: float


@dataclass
class ScoredResult:
    """점수가 포함된 결과"""

    content: str
    confidence: float
    source: str
    reasoning: str
    execution_time: float


# ============================================================================
# 하이브리드 라우터 구현
# ============================================================================


class HybridRouter:
    """ReAct Router + Multi-LangGraph Router의 장점을 통합한 하이브리드 라우터"""

    def __init__(self):
        # LLM 클라이언트 초기화
        self.ollama_client = create_ollama_client()

        # 도구 레지스트리 설정
        self.tools_registry = create_tools_registry(
            ollama_client=self.ollama_client,
        )

        # 도구 가져오기
        self.tools = self.tools_registry.tools

        # 워크플로우 설정
        self.workflow = self._create_parallel_workflow()

        log.info("하이브리드 라우터 초기화 완료")

    # ========================================================================
    # 병렬 처리 워크플로우
    # ========================================================================

    def _create_parallel_workflow(self):
        """병렬 처리 워크플로우 정의:
        LangGraph를 사용하여 분석-병렬-실행-점수 부여-통합의 단계를 순차적으로 실행하는 워크플로우를 정의합니다.
        """
        workflow = StateGraph(HybridQueryState)

        # 노드 정의
        workflow.add_node("analyze_query", self._analyze_query)
        workflow.add_node("parallel_execution", self._parallel_execution)
        workflow.add_node("score_results", self._score_results)
        workflow.add_node("integrate_results", self._integrate_results)

        # 순차적 실행 흐름
        workflow.set_entry_point("analyze_query")
        workflow.add_edge("analyze_query", "parallel_execution")
        workflow.add_edge("parallel_execution", "score_results")
        workflow.add_edge("score_results", "integrate_results")
        workflow.add_edge("integrate_results", END)

        return workflow.compile()

    async def _analyze_query(self, state: HybridQueryState) -> Dict:
        """쿼리 분석 노드:
        쿼리 복잡도 분석, LLM 의도 분석, 도구 선택을 수행하고 상태를 업데이트합니다.
        """
        query = state["query"]

        # 1. 쿼리 복잡도 분석
        complexity_score = self._calculate_complexity(query)

        # 2. LLM 기반 의도 분석
        intent_analysis = await self._llm_intent_analysis(query)

        # 3. 도구 선택 (LLM 결과)
        selected_tools = await self._select_tools_with_llm(query, intent_analysis)

        log.info(
            f"쿼리 분석 완료: 복잡도={complexity_score:.2f}, "
            f"선택된 도구={selected_tools}, "
            f"신뢰도={intent_analysis.get('confidence', 0.0):.2f}"
        )

        return {
            "intent_analysis": intent_analysis,
            "selected_tools": selected_tools,
            "complexity_score": complexity_score,
            "parallel_results": {
                "selected_tools": selected_tools,
                "complexity": complexity_score,
            },
            "processing_stage": "analyzed",
        }

    async def _select_tools_with_llm(
        self, query: str, intent_analysis: Dict
    ) -> List[str]:
        """최종 도구 선택 로직:
        LLM이 제안한 도구 리스트와, 키워드 기반의 백업 로직을 조합하여 실제로 실행할 최종 도구 리스트를 결정
        """

        selected_tools = []

        # 1. LLM이 제안한 도구들 추가
        llm_tools = intent_analysis.get("required_tools", [])
        selected_tools = [tool for tool in llm_tools if tool in self.tools]

        # LLM 신뢰도 확인
        confidence = intent_analysis.get("confidence", 0.0)

        # 신뢰도가 높으면 LLM이 선택한 도구들만 사용
        if confidence >= 0.7:
            log.info(f"LLM 신뢰도 높음 ({confidence:.2f}), 백업 로직 스킵")
            return selected_tools if selected_tools else ["reasoning"]

        # 신뢰도가 낮거나 도구를 선택하지 못한 경우에만 백업로직 탐
        log.warning(f"LLM 신뢰도 낮음 ({confidence:.2f}), 백업 로직 적용")

        if "계산" in query and "calculator" not in selected_tools:
            selected_tools.append("calculator")

        if "날씨" in query and "weather" not in selected_tools:
            selected_tools.append("weather")

        if (
            "주식" in query
            or "주가" in query
            or "종목" in query
            and "stock_info" not in selected_tools
        ):
            selected_tools.append("stock_info")

        # 문서/자료 참조 키워드 체크
        doc_keywords = ["문서", "자료", "파일", "업로드", "PDF", "요약", "정리"]
        if (
            any(keyword in query for keyword in doc_keywords)
            and "knowledge_base" not in selected_tools
        ):
            selected_tools.append("knowledge_base")

        # 최소 1개 보장
        if not selected_tools:
            selected_tools.append("reasoning")

        # 4. 도구 중복 제거
        selected_tools = list(dict.fromkeys(selected_tools))

        return selected_tools

    async def _parallel_execution(self, state: HybridQueryState) -> Dict:
        """도구 병렬 실행 노드:
        선택된 도구들을 동시에 실행하고, 각 도구의 실행 결과를 수집하여 반환합니다.
        """
        query = state["query"]
        selected_tools = state["selected_tools"]
        arguments = state["intent_analysis"].get("arguments", {})

        start_time = time.time()

        # 병렬 실행
        tasks = []
        for tool_name in selected_tools:
            if tool_name in self.tools:
                # 🆕 도구의 arguments가 리스트인지 확인
                tool_args_list = arguments.get(tool_name, [])

                # 리스트가 아니면 리스트로 변환 (하위 호환성)
                if not isinstance(tool_args_list, list):
                    tool_args_list = [tool_args_list]

                # 🆕 각 매개변수 세트마다 태스크 생성
                for idx, tool_args in enumerate(tool_args_list):
                    task = asyncio.create_task(
                        self._execute_tool_with_timing(tool_name, query, tool_args)
                    )
                    tasks.append((f"{tool_name}_{idx}", task))

        # 모든 도구 결과 수집
        parallel_results = {}
        for tool_name, task in tasks:
            try:
                result = await task
                parallel_results[tool_name] = result
                log.info(f"{tool_name} 실행 완료: {result.execution_time:.2f}초")
            except Exception as e:
                parallel_results[tool_name] = ScoredResult(
                    content=f"실행 오류: {str(e)}",
                    confidence=0.0,
                    source=tool_name,
                    reasoning="도구 실행 중 오류 발생",
                    execution_time=0.0,
                )
                log.error(f"{tool_name} 실행 실패: {e}")

        total_time = time.time() - start_time
        log.info(
            f"병렬 실행 완료: {len(parallel_results)}개 도구, 총 {total_time:.2f}초"
        )

        return {
            "parallel_results": {
                **state["parallel_results"],
                "execution_results": parallel_results,
            },
            "processing_stage": "executed",
        }

    async def _execute_tool_with_timing(
        self, tool_name: str, query: str, arguments: Dict
    ) -> ScoredResult:
        """개별 도구 실행 및 타이밍 측정:
        특정 도구를 호출(ainvoke)하고 실행 시간을 측정하며, 결과의 신뢰도를 계산하여 반환합니다."""
        start_time = time.time()

        log.info(f"도구 실행 시작: {tool_name}, 매개변수={arguments}")
        try:
            tool = self.tools[tool_name]

            # 도구별 매개변수 설정
            if tool_name == "weather":
                city = arguments.get("city", "")
                mode = arguments.get("mode", "")
                result = await tool.ainvoke({"city": city, "mode": mode})
            elif tool_name == "stock_info":
                stock_code = arguments.get("stock_code", "")
                mode = arguments.get("mode", "")
                result = await tool.ainvoke({"stock_code": stock_code, "mode": mode})
            else:
                result = await tool.ainvoke({"query": arguments.get("query", query)})

            execution_time = time.time() - start_time

            # 결과의 신뢰도 계산
            confidence = self._calculate_confidence(result, tool_name, execution_time)

            return ScoredResult(
                content=result,
                confidence=confidence,
                source=tool_name,
                reasoning=f"{tool_name}에서 실행된 결과",
                execution_time=execution_time,
            )

        except Exception as e:
            execution_time = time.time() - start_time
            raise e

    async def _score_results(self, state: HybridQueryState) -> Dict:
        """결과 점수 부여 노드:
        병렬 실행된 각 도구 결과에 대해 관련성, 신뢰도, 속도를 기반으로 점수를 부여하고, 점수를 기반으로 결과를 통합합니다."""
        execution_results = state["parallel_results"]["execution_results"]
        query = state["query"]

        scores = {}

        for tool_name, result in execution_results.items():
            if isinstance(result, ScoredResult):
                # 다양한 기준으로 점수 계산
                relevance_score = self._calculate_relevance(result.content, query)
                confidence_score = result.confidence
                speed_score = 1.0 / (result.execution_time + 0.1)

                # 가중 평균
                final_score = (
                    relevance_score * 0.5
                    + confidence_score * 0.3
                    + min(speed_score, 1.0) * 0.2
                )

                scores[tool_name] = final_score
                log.info(
                    f"{tool_name} 점수: {final_score:.3f} (관련성: {relevance_score:.3f}, 신뢰도: {confidence_score:.3f})"
                )
            else:
                scores[tool_name] = 0.0

        return {"scores": scores, "processing_stage": "scored"}

    async def _integrate_results(self, state: HybridQueryState) -> Dict:
        """결과 통합 노드:
        복잡도에 따라 다른 통합 전략을 사용하여 최종 답변을 생성합니다.
        """
        execution_results = state["parallel_results"]["execution_results"]
        scores = state["scores"]
        query = state["query"]
        complexity_score = state["complexity_score"]

        # 점수 순으로 정렬
        sorted_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        # 임계점 이상 결과 필터링
        filtered_results = [
            (tool_name, execution_results[tool_name], score)
            for tool_name, score in sorted_results
            if score > 0.3
            and isinstance(execution_results.get(tool_name), ScoredResult)
        ]

        if not filtered_results:
            return {
                "final_answer": "적절한 답변을 생성할 수 없습니다.",
                "processing_stage": "completed",
            }

        # 복잡도 기반 통합 전략 선택
        if complexity_score < 0.3 and len(filtered_results) == 1:
            # 간단한 쿼리 + 단일 도구: 직접 답변
            tool_name, result, score = filtered_results[0]
            final_answer = result.content

        elif complexity_score > 0.5 or len(filtered_results) > 2:
            # 복잡한 쿼리 또는 다중 도구: LLM 통합
            final_answer = await self._integrate_with_llm(query, filtered_results)
        else:
            # 중간 복잡도: 점수 기반 나열
            integrated_content = []
            total_weight = 0
            for tool_name, result, score in filtered_results:
                integrated_content.append(
                    f"[{tool_name} ({score:.2f})]: {result.content}"
                )
                total_weight += score
            final_answer = (
                f"다음은 {len(integrated_content)}개 소스를 통합한 답변입니다:\n\n"
            )
            final_answer += "\n\n".join(integrated_content)
            avg_confidence = total_weight / len(integrated_content)
            final_answer += f"\n\n종합 신뢰도: {avg_confidence:.2f}"
        log.info(
            f"결과 통합 완료: {len(filtered_results)}개 소스, "
            f"복잡도={complexity_score:.2f}"
        )

        return {"final_answer": final_answer, "processing_stage": "completed"}

    async def _integrate_with_llm(self, query: str, filtered_results: list) -> str:
        """LLM을 사용한 지능형 결과 통합"""
        try:
            # 결과를 구조화된 텍스트로 변환
            collected_info = []
            for tool_name, result, score in filtered_results:
                collected_info.append(
                    f"[{tool_name}] (신뢰도: {score:.2f})\n{result.content}"
                )

            system_prompt = """당신은 정보 통합 전문가입니다.
            여러 소스의 정보를 종합하여 일관성 있고 유용한 답변을 제공하세요."""

            user_prompt = f"""
            질문: {query}

            수집된 정보:
            {chr(10).join(collected_info)}

            위 정보들을 종합하여 사용자 질문에 대한 명확하고 유용한 답변을 생성해주세요.
            """

            response = await self.ollama_client.generate(
                prompt=user_prompt, system=system_prompt
            )

            return response.content

        except Exception as e:
            log.error(f"LLM 통합 실패, 기본 통합 사용: {e}")
            # Fallback: 점수 기반 나열
            return "\n\n".join(
                [f"[{name}]: {result.content}" for name, result, _ in filtered_results]
            )

    # ========================================================================
    # LLM 기반 의도 분석
    # ========================================================================

    def _get_all_tools_schema(self) -> str:
        """도구 스키마 문자열 생성 :
        LLM에게 제공하기 위해 등록된 모든 도구의 이름, 설명, 매개변수, 예시 정보를 포맷팅된 문자열로 반환
        """
        return self.tools_registry.get_all_tools_schema_text()

    async def _llm_intent_analysis(self, query: str) -> Dict[str, Any]:
        """LLM을 사용한 고급 의도 분석 (ReAct Router 방식):
        시스템 프롬프트를 사용하여 LLM에게 쿼리를 보내고, 주요/보조 의도, 복잡도 수준, 필수 도구 리스트 등의 정보를 포함하는 JSON 형태의 분석 결과를 받습니다. (ReAct Router의 핵심 아이디어)
        """

        system_prompt = """당신은 쿼리 의도 분석 전문가입니다.
        사용자의 쿼리를 분석하여 다음 정보를 JSON 형태로 제공하세요:

        1. required_tools: 필요한 도구들 (리스트)
        2. arguments: 각 도구에 필요한 매개변수 (딕셔너리)
        3. reasoning: 분석 근거
        4. confidence: 신뢰도 (0.0-1.0)

        사용 가능한 도구들:
        - weather: 날씨 정보 조회
          매개변수: city(도시명), lat/lon(좌표), mode(current/forecast)
        - stock_info: 주식 정보 조회
          매개변수: stock_code(종목코드), mode(info/price/fundamental/market_cap)
        - calculator: 수학 계산
          매개변수: query(수식)
        - web_search: 웹 검색
          매개변수: query(검색어)
            🔥 쿼리 최적화 필수 규칙:
            1. 핵심 키워드를 쉼표로 구분하여 나열
            2. 한국어 키워드와 영어 동의어 모두 포함
            3. 불용어 완전 제거 (은, 는, 이, 가, 를, the, a 등)
            4. 동의어 3-5개 이상 포함
            5. 연도(2024, 2025) 추가로 최신 정보 확보
        - knowledge_base: 내부 지식베이스 및 업로드된 문서 검색
          매개변수: query(검색어)
          🔥 사용 시점:
          1. "문서에서", "자료에서", "과제에서" 등 문서 참조 요청
          2. 업로드된 PDF 내용에 대한 질문
          3. 구체적인 학술/기술 내용 검색
          4. "요약", "정리", "설명" 등 문서 기반 작업
        - reasoning: 논리적 추론 (문서 참조 없이 순수 분석)
          매개변수: query(질문)
          🔥 사용 시점:
          1. "어떻게 생각해?", "전망은?" 등 의견 요청
          2. 비교/분석/평가 (문서 없이)
          3. 논리적 추론이 필요한 질문

        예시 응답 1 (날씨):
        {
            "required_tools": ["weather"],
            "arguments": {
                "weather": [
                    {"city": "성남", "mode": "current", "days": 1},
                    {"city": "부산", "mode": "current", "days": 1}
                ]
            },
            "reasoning": "성남과 부산의 날씨를 요청하므로 weather 도구가 필요합니다",
            "confidence": 0.9
        }


        중요 사항:
        - required_tools에 필요한 모든 도구를 나열하세요
        - arguments에는 required_tools에 포함된 도구만 매개변수를 제공하세요
        - 각 도구의 매개변수는 위의 도구 스키마를 참고하세요
        - web_search를 사용할 경우 반드시 검색 쿼리 최적화 규칙을 적용하세요"""

        user_prompt = f"""
        다음 쿼리를 분석해주세요:
        쿼리: "{query}"

        사용 가능한 도구와 사용법:
        {self._get_all_tools_schema()}

        위 쿼리의 의도를 분석하고 JSON 형태로 응답해주세요.
        """

        try:
            # LLM 요청
            response = await self.ollama_client.generate(
                prompt=user_prompt, system=system_prompt
            )

            # 원본 응답 정리 (백틱 제거 등)
            cleaned_content = self._clean_llm_response(response.content)

            # JSON 파싱 시도
            intent_analysis = json.loads(cleaned_content)
            log.info(f"LLM 의도 분석 완료: {intent_analysis}")
            return intent_analysis

        except json.JSONDecodeError as e:
            # JSON 파싱 실패 시 상세 로그
            log.error(f"JSON 파싱 실패: {e}")
            log.error(f"원본 응답:\n{response.content}")
            log.error(f"정리된 응답:\n{cleaned_content}")

            # 기본값 반환
            return {
                "required_tools": ["reasoning"],
                "arguments": {"reasoning": {"query": query}},
                "reasoning": "LLM 분석 실패로 reasoning 도구 사용",
                "confidence": 0.3,
            }
        except Exception as e:
            log.error(f"LLM 의도 분석 오류: {e}")
            return {
                "required_tools": ["reasoning"],
                "arguments": {"reasoning": {"query": query}},
                "reasoning": f"분석 오류: {str(e)}",
                "confidence": 0.1,
            }

    # ========================================================================
    # 유틸리티 함수들
    # ========================================================================
    def _clean_llm_response(self, response_content: str) -> str:
        """LLM JSON 응답 정리:
        LLM 응답에서 JSON 추출하고, 백틱을 제거하여 순수한 JSON 문자열로 반환합니다.
        """
        content = response_content.strip()

        # 방법 1: 백틱 블록으로 감싸진 경우 추출
        if "```" in content:
            # 시작 백틱 제거
            if content.startswith("```json"):
                content = content[7:]
            elif content.startswith("```"):
                content = content[3:]

            # 끝 백틱 제거
            if content.endswith("```"):
                content = content[:-3]

            content = content.strip()

        # 방법 2: 첫 번째 { 부터 마지막 } 까지 추출 (중첩 JSON 지원)
        first_brace = content.find("{")
        last_brace = content.rfind("}")

        if first_brace != -1 and last_brace != -1 and first_brace < last_brace:
            content = content[first_brace : last_brace + 1]

        # 방법 3: 중괄호 균형 확인 및 복구 시도
        content = self._fix_incomplete_json(content)

        return content.strip()

    def _fix_incomplete_json(self, content: str) -> str:
        """불완전한 JSON 복구 시도"""
        try:
            # 중괄호 개수 확인
            open_count = content.count("{")
            close_count = content.count("}")

            # 닫는 중괄호가 부족한 경우 추가
            if open_count > close_count:
                missing = open_count - close_count
                log.warning(
                    f"JSON 중괄호 불균형 감지: {missing}개 부족, 자동 추가 시도"
                )
                content = content + ("}" * missing)

            # 대괄호 개수 확인
            open_bracket = content.count("[")
            close_bracket = content.count("]")

            if open_bracket > close_bracket:
                missing = open_bracket - close_bracket
                log.warning(
                    f"JSON 대괄호 불균형 감지: {missing}개 부족, 자동 추가 시도"
                )
                content = content + ("]" * missing)

            return content
        except Exception as e:
            log.warning(f"JSON 복구 실패: {e}")
            return content

    def _calculate_complexity(self, query: str) -> float:
        """쿼리 복잡도 계산:
        쿼리 길이, 물음표 개수, 접속사, 특정 키워드(분석, 비교 등)의 출현 빈도 등을 종합적으로 고려하여 0.0에서 1.0 사이의 쿼리 복잡도 수준을 계산합니다.
        """
        factors = {
            "length": min(len(query) / 50, 1.0) * 0.3,  # 50자 기준, 가중치 증가
            "questions": query.count("?") * 0.15,  # 가중치 증가
            "conjunctions": sum(
                query.count(word)
                for word in ["그리고", "또한", "하지만", "그러나", "또는"]
            )
            * 0.2,  # 가중치 증가
            "complexity_keywords": sum(
                query.count(word)
                for word in [
                    "분석",
                    "비교",
                    "평가",
                    "추론",
                    "설명",
                    "왜",
                    "어떻게",
                    "전망",
                    "예측",
                    "동향",
                    "영향",
                ]
            )
            * 0.15,  # 키워드 확장, 가중치 조정
            "multi_concept": (
                0.2 if any(word in query for word in ["와", "과", "및"]) else 0
            ),  # 다중 개념
            "sequential_keywords": sum(
                query.count(word) for word in ["결과로", "이후에", "~하고", "다음으로"]
            )
            * 0.3,  # 순차 패턴 감지
        }

        return min(sum(factors.values()), 1.0)

    def _calculate_confidence(
        self, result: str, tool_name: str, execution_time: float
    ) -> float:
        """결과 신뢰도 계산
        도구별 기본 신뢰도와 실행 시간을 고려하여 결과의 신뢰도 점수를 계산합니다.
        """
        base_confidence = 0.7

        # 도구별 기본 신뢰도
        tool_confidence = {
            "weather": 0.9,
            "stock_info": 0.9,
            "calculator": 0.95,
            "web_search": 0.65,
            "knowledge_base": 0.7,
            "reasoning": 0.7,
        }.get(tool_name, 0.5)

        # 실행 시간 고려
        if execution_time < 0.1 or execution_time > 5.0:
            time_penalty = 0.1
        else:
            time_penalty = 0.0

        return max(tool_confidence - time_penalty, 0.1)

    def _calculate_relevance(self, content: str, query: str) -> float:
        """관련성 점수 계산:
        쿼리 단어와 결과 내용 단어의 **겹치는 정도(교집합)**를 기반으로 관련성 점수를 계산합니다.
        """
        query_words = set(query.lower().split())
        content_words = set(content.lower().split())

        if not query_words:
            return 0.0

        intersection = query_words.intersection(content_words)
        return len(intersection) / len(query_words)

    # ========================================================================
    # 메인 실행 인터페이스
    # ========================================================================

    async def process_query(self, query: str) -> Dict:
        """쿼리 처리 (병렬 워크플로우)"""
        initial_state = {
            "messages": [HumanMessage(content=query)],
            "query": query,
            "intent_analysis": {},
            "selected_tools": [],
            "parallel_results": {},
            "scores": {},
            "final_answer": "",
            "processing_stage": "start",
            "complexity_score": 0.0,
        }
        result = await self.workflow.ainvoke(initial_state)

        return {
            "success": True,
            "final_answer": result.get("final_answer", "답변을 생성할 수 없습니다."),
            "complexity_score": result.get("complexity_score", 0.0),
            "selected_tools": result.get("parallel_results", {}).get(
                "selected_tools", []
            ),
        }

    async def aclose(self):
        """리소스 정리"""
        try:
            # ToolsRegistry의 리소스 정리
            await self.tools_registry.mcp_client.aclose()
            await self.tools_registry.google_search_client.aclose()
            if self.tools_registry.rag_client:
                await self.tools_registry.rag_client.aclose()
            log.info("하이브리드 라우터 리소스 정리 완료")
        except Exception as e:
            log.error(f"리소스 정리 중 오류: {e}")


# ========================
# 팩토리 함수
# ========================


def create_hybrid_router() -> HybridRouter:
    """하이브리드 라우터 생성"""
    return HybridRouter()


async def main():
    """하이브리드 라우터 테스트"""
    router = HybridRouter()
    await router.aclose()


if __name__ == "__main__":
    asyncio.run(main())
