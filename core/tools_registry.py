"""
도구 레지스트리 및 스키마 정의

HybridRouter에서 사용하는 모든 도구 함수와 스키마를 정의합니다.
"""

from typing import Dict, Any, Optional
from dataclasses import dataclass
from langchain_core.tools import tool

from utils.logger import log
from services.google_search import GoogleSearchClient, GoogleSearchResultParser
from services.mcp import WeatherService, StockService, MCPClient
from services.rag import VectorSearchManager, search_rag_async


@dataclass
class ToolSchema:
    """도구 스키마 정의"""

    name: str
    description: str
    parameters: Dict[str, Any]
    example: str


class ToolsRegistry:
    """도구 레지스트리: 도구 함수와 스키마를 통합 관리"""

    def __init__(
        self,
        mcp_client: Optional[MCPClient] = None,
        ollama_client=None,
    ):
        """도구 레지스트리 초기화

        Args:
            mcp_client: MCP 클라이언트 (optional, 없으면 자동 생성)
            ollama_client: Ollama LLM 클라이언트 (필수)
        """
        # MCP 클라이언트 초기화
        if mcp_client is None:
            self.mcp_client = MCPClient()
            log.info("ToolsRegistry: MCP 클라이언트 자체 생성")
        else:
            self.mcp_client = mcp_client
            log.info("ToolsRegistry: MCP 클라이언트 외부에서 주입")

        # MCP 서비스들 초기화
        self._weather_service = WeatherService(client=self.mcp_client)
        self._stock_service = StockService(client=self.mcp_client)

        # Google Search 초기화
        self.google_search_client = GoogleSearchClient(ollama_client=ollama_client)
        self.google_search_parser = GoogleSearchResultParser()

        # RAG 클라이언트 초기화
        try:
            self.rag_client = VectorSearchManager()
            log.info("ToolsRegistry: RAG 클라이언트 초기화 성공")
        except Exception as e:
            log.warning(f"ToolsRegistry: RAG 클라이언트 초기화 실패: {e}")
            self.rag_client = None

        # Ollama 클라이언트 저장
        self.ollama_client = ollama_client

        # 도구와 스키마 설정
        self.tools = self._create_tools()
        self.tool_schemas = self._create_tool_schemas()

    def _create_tools(self) -> Dict[str, Any]:
        """도구 함수 생성"""

        @tool
        async def weather_tool(
            city: str = "",
            lat: float = None,
            lon: float = None,
            mode: str = "current",
            days: int = None,
            date: str = None,
            weekday: str = None,
            top_n: int = 3,
        ) -> str:
            """날씨 정보 조회 (MCP JSON-RPC 사용)"""
            try:
                log.info(
                    f"날씨 정보 조회 요청: city={city}, lat={lat}, lon={lon}, mode={mode}, days={days}, date={date}, weekday={weekday}, top_n={top_n}"
                )
                service = self._weather_service

                is_forecast = (mode or "").lower() == "forecast" or any(
                    [days is not None, bool(date), bool(weekday)]
                )

                if lat is not None and lon is not None:
                    resp = await service.get_weather_by_coordinates(lat=lat, lon=lon)
                else:
                    if is_forecast:
                        resp = await service.get_weather_forecast(city)
                    else:
                        resp = await service.get_current_weather(city)

                if isinstance(resp, dict):
                    success = resp.get("success")
                    message = resp.get("message", "")
                    if success:
                        return message or "날씨 조회가 완료되었습니다."
                    return message or "날씨 조회에 실패했습니다."
                return str(resp)
            except Exception as e:
                return f"날씨 조회 오류: {str(e)}"

        @tool
        async def stock_info_tool(
            stock_code: str = "",
            query: str = "",
            mode: str = "info",
            include_price: bool = False,
            include_fundamental: bool = False,
            include_market_cap: bool = False,
        ) -> str:
            """주식 정보 조회 (MCP JSON-RPC 사용)"""
            try:
                service = self._stock_service

                if query and not stock_code:
                    resp = await service.search_stock(query)
                elif mode == "info":
                    resp = await service.get_stock_info(stock_code)
                elif mode == "price":
                    resp = await service.get_stock_price_data(stock_code)
                elif mode == "fundamental":
                    resp = await service.get_stock_fundamental(stock_code)
                elif mode == "market_cap":
                    resp = await service.get_stock_market_cap(stock_code)

                if isinstance(resp, dict):
                    success = resp.get("success")
                    message = resp.get("message", "")
                    if success:
                        return message or f"주식 정보 조회가 완료되었습니다."
                    return message or f"주식 정보 조회에 실패했습니다."
                return str(resp)
            except Exception as e:
                return f"주식 정보 조회 오류: {str(e)}"

        @tool
        async def calculator_tool(query: str) -> str:
            """수학 계산"""
            try:
                result = eval(query)
                return f"계산 결과: {query} = {result}"
            except Exception as e:
                return f"계산 오류: {query} - {str(e)}"

        @tool
        async def web_search_tool(query: str) -> str:
            """웹 검색 (실제 Google Search API 사용)"""
            try:
                results = await self.google_search_client.process_search(query=query)

                if not results:
                    return "검색 결과가 없습니다."

                summary = self.google_search_client.get_search_summary(results)
                log.info(f"웹 검색 결과 요약: {summary}")

                parsed = self.google_search_parser.parse_results(
                    results, max_content_length=600
                )

                if not parsed:
                    return "검색 결과 파싱에 실패했습니다."

                top = parsed[:3]
                lines = []
                # 검색 결과 요약 정보 추가
                lines.append(
                    f"📊 {len(results)}개 결과 (평균 {summary['avg_score']:.2f}, 최고 {summary['best_score']:.2f})"
                )
                lines.append(f"🌐 출처: {len(summary['sources'])}개 도메인")
                lines.append("")
                for idx, item in enumerate(top, start=1):
                    confidence_emoji = (
                        "🟢"
                        if item.relevance_score >= 0.7
                        else "🟡" if item.relevance_score >= 0.5 else "🔴"
                    )

                    lines.append(
                        f"{idx}. {confidence_emoji} {item.title} ({item.source})"
                    )
                    lines.append(
                        f"   신뢰도: {item.relevance_score:.2f} | URL: {item.url}"
                    )
                    lines.append(f"   {item.summary}")

                log.info(f"웹 검색 결과: {'\n'.join(lines)}")
                return "\n".join(lines)
            except Exception as e:
                return f"웹 검색 오류: {str(e)}"

        @tool
        async def knowledge_base_tool(query: str) -> str:
            """내부 지식베이스 검색 (실제 RAG 시스템 사용)"""
            try:
                if not self.rag_client:
                    return "RAG 시스템이 사용 불가능합니다."

                from services.rag import search_rag_async

                results = await search_rag_async(
                    query=query, vector_manager=self.rag_client, max_results=3
                )

                if not results:
                    return "관련 문서를 찾을 수 없습니다."

                formatted_results = []
                for result in results:
                    formatted_results.append(
                        f"📄 {result.content} (신뢰도: {result.similarity_score:.2f})"
                    )

                return "\n\n".join(formatted_results)

            except Exception as e:
                return f"지식베이스 검색 오류: {str(e)}"

        @tool
        async def reasoning_tool(query: str) -> str:
            """논리적 추론 (LLM 기반)"""
            try:
                system_prompt = """당신은 논리적 분석 전문가입니다.
                주어진 질문에 대해 체계적이고 논리적인 분석을 제공하세요."""

                response = await self.ollama_client.generate(
                    prompt=f"다음 질문에 대해 논리적으로 분석해주세요: {query}",
                    system=system_prompt,
                )

                return f"추론 결과:\n{response.content}"

            except Exception as e:
                return f"추론 분석 오류: {str(e)}"

        return {
            "weather": weather_tool,
            "stock_info": stock_info_tool,
            "calculator": calculator_tool,
            "web_search": web_search_tool,
            "knowledge_base": knowledge_base_tool,
            "reasoning": reasoning_tool,
        }

    def _create_tool_schemas(self) -> Dict[str, ToolSchema]:
        """도구 스키마 생성"""
        return {
            "weather": ToolSchema(
                name="weather",
                description="MCP 기반 날씨 조회. 도시명 또는 좌표로 현재/예보 정보를 반환",
                parameters={
                    "city": "string(optional)",
                    "mode": "string(optional: current|forecast, default=current)",
                    "days": "int(optional)",
                    "date": "string(optional, YYYY-MM-DD)",
                    "weekday": "string(optional, 월|화|수|목|금|토|일)",
                    "top_n": "int(optional)",
                },
                example='weather 도구 사용 시: {"city": "서울", "mode": "forecast"}',
            ),
            "stock_info": ToolSchema(
                name="stock_info",
                description="MCP 기반 주식 정보 조회. 종목 코드, 검색어, 모드로 다양한 주식 정보를 반환",
                parameters={
                    "stock_code": "string(optional, e.g. 005930)",
                    "mode": "string(optional: info|price|fundamental|market_cap, default=info)",
                },
                example='stock_info 도구 사용 시: {"stock_code": "005930", "mode": "info"}',
            ),
            "calculator": ToolSchema(
                name="calculator",
                description="수학 계산을 수행합니다",
                parameters={"query": "string"},
                example='calculator 도구 사용 시: {"query": "2 + 3 * 4"}',
            ),
            "web_search": ToolSchema(
                name="web_search",
                description="웹에서 최신 정보를 검색합니다",
                parameters={"query": "string"},
                example='web_search 도구 사용 시: {"query": "AI 기술 발전"}',
            ),
            "knowledge_base": ToolSchema(
                name="knowledge_base",
                description="내부 지식베이스에서 정보를 검색합니다",
                parameters={"query": "string"},
                example='knowledge_base 도구 사용 시: {"query": "기업 정책"}',
            ),
            "reasoning": ToolSchema(
                name="reasoning",
                description="논리적 추론 및 분석을 수행합니다",
                parameters={"query": "string"},
                example='reasoning 도구 사용 시: {"query": "AI의 미래 전망"}',
            ),
        }

    def get_all_tools_schema_text(self) -> str:
        """도구 스키마를 텍스트로 반환 (LLM 프롬프트용)"""
        schema_text = ""
        for tool_name, schema in self.tool_schemas.items():
            schema_text += f"""
                                도구명: {schema.name}
                                설명: {schema.description}
                                매개변수: {schema.parameters}
                                예시: {schema.example}
                                ---
                            """
        return schema_text


def create_tools_registry(
    ollama_client,
    mcp_client: Optional[MCPClient] = None,
) -> ToolsRegistry:
    """도구 레지스트리 생성

    Args:
        ollama_client: Ollama LLM 클라이언트 (필수)
        mcp_client: MCP 클라이언트 (optional, 없으면 자동 생성)
    """
    return ToolsRegistry(
        mcp_client=mcp_client,
        ollama_client=ollama_client,
    )
