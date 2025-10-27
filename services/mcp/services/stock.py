"""
MCP 주식 서비스 (Service Module)

JSON-RPC(WebSocket) 기반 MCP 서버의 주식 관련 도구들을 호출하는 서비스 레이어입니다.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from utils.logger import log
from ..client import MCPClient, MCPResponse
from ..parser import StockDataFormatter


class StockService:
    """주식 서비스 핸들러.

    - JSON-RPC 도구 호출 → 응답 표준 포맷으로 변환
    """

    def __init__(self, client: Optional[MCPClient] = None) -> None:
        """StockService를 초기화합니다.

        Args:
            client: 주입 가능한 MCPClient 인스턴스. 미제공 시 기본값 사용
        """
        self.client = client or MCPClient()
        self.formatter = StockDataFormatter()

        log.info(f"StockService 초기화: {self.client}")

    async def load_all_tickers(self) -> Dict[str, Any]:
        """상장 종목 코드 전체 로드."""
        try:
            raw = await self.client.call_tool_jsonrpc_async(
                method_name="load_all_tickers", arguments={}
            )
            resp: MCPResponse = await self.client.to_response_async(
                "load_all_tickers", raw
            )
            return self._format_response(resp.success, resp.raw_data, resp.content_text)
        except Exception as exc:
            log.error(f"티커 로드 실패: {exc}")
            return self._format_response(False, None, f"티커 로드 실패: {exc}")

    async def get_stock_info(self, stock_code: str) -> Dict[str, Any]:
        """단일 종목 기본 정보 조회.

        Args:
            stock_code: 종목 코드 (예: "005930")

        Returns:
            성공 여부, 원시 데이터, 요약 메시지를 포함한 표준 응답 딕셔너리
        """
        try:
            raw = await self.client.call_tool_jsonrpc_async(
                method_name="get_stock_info", arguments={"stock_code": stock_code}
            )
            resp: MCPResponse = await self.client.to_response_async(
                "get_stock_info", raw
            )
            # 포매터 사용
            if resp.success and resp.raw_data.get("result", {}).get("content"):
                content = resp.raw_data["result"]["content"]
                formatted_message = self.formatter.format_info(content)
            else:
                formatted_message = resp.content_text
            return self._format_response(resp.success, resp.raw_data, formatted_message)
        except Exception as exc:
            log.error(f"주식 정보 조회 실패: {exc}")
            return self._format_response(False, None, f"주식 정보 조회 실패: {exc}")

    async def get_stock_price_data(self, stock_code: str) -> Dict[str, Any]:
        """주식 가격 데이터 조회."""
        try:
            raw = await self.client.call_tool_jsonrpc_async(
                method_name="get_stock_price_data", arguments={"stock_code": stock_code}
            )
            resp: MCPResponse = await self.client.to_response_async(
                "get_stock_price_data", raw
            )
            if resp.success and resp.raw_data.get("result", {}).get("content"):
                content = resp.raw_data["result"]["content"]
                formatted_message = self.formatter.format_price_data(content)
            else:
                formatted_message = resp.content_text
            return self._format_response(resp.success, resp.raw_data, formatted_message)
        except Exception as exc:
            log.error(f"주식 가격 데이터 조회 실패: {exc}")
            return self._format_response(
                False, None, f"주식 가격 데이터 조회 실패: {exc}"
            )

    async def get_stock_market_cap(self, stock_code: str) -> Dict[str, Any]:
        """시가총액 정보 조회."""
        try:
            raw = await self.client.call_tool_jsonrpc_async(
                method_name="get_stock_market_cap", arguments={"stock_code": stock_code}
            )
            resp: MCPResponse = await self.client.to_response_async(
                "get_stock_market_cap", raw
            )
            # 포매터 사용
            if resp.success and resp.raw_data.get("result", {}).get("content"):
                content = resp.raw_data["result"]["content"]
                formatted_message = self.formatter.format_market_cap(content)
            else:
                formatted_message = resp.content_text
            return self._format_response(resp.success, resp.raw_data, formatted_message)
        except Exception as exc:
            log.error(f"시가총액 조회 실패: {exc}")
            return self._format_response(False, None, f"시가총액 조회 실패: {exc}")

    async def get_stock_fundamental(self, stock_code: str) -> Dict[str, Any]:
        """재무 지표(펀더멘털) 조회.

        Args:
            stock_code: 종목 코드
        """
        try:
            raw = await self.client.call_tool_jsonrpc_async(
                method_name="get_stock_fundamental",
                arguments={"stock_code": stock_code},
            )
            resp: MCPResponse = await self.client.to_response_async(
                "get_stock_fundamental", raw
            )
            # 포매터 사용
            if resp.success and resp.raw_data.get("result", {}).get("content"):
                content = resp.raw_data["result"]["content"]
                formatted_message = self.formatter.format_fundamental(content)
            else:
                formatted_message = resp.content_text
            return self._format_response(resp.success, resp.raw_data, formatted_message)
        except Exception as exc:
            log.error(f"재무 지표 조회 실패: {exc}")
            return self._format_response(False, None, f"재무 지표 조회 실패: {exc}")

    async def search_stock(self, query: str) -> Dict[str, Any]:
        """종목 검색.

        Args:
            query: 종목명 또는 키워드
        """
        try:
            raw = await self.client.call_tool_jsonrpc_async(
                method_name="search_stock", arguments={"keyword": query}
            )
            resp: MCPResponse = await self.client.to_response_async("search_stock", raw)
            return self._format_response(resp.success, resp.raw_data, resp.content_text)
        except Exception as exc:
            log.error(f"종목 검색 실패: {exc}")
            return self._format_response(False, None, f"종목 검색 실패: {exc}")

    def _format_response(
        self, success: bool, data: Optional[Dict[str, Any]], message: str
    ) -> Dict[str, Any]:
        """일관된 응답 포맷으로 변환합니다.

        Args:
            success: 성공 여부
            data: 원시 데이터(JSON)
            message: 사람이 읽을 수 있는 요약 메시지
        """
        return {
            "success": success,
            "data": data if isinstance(data, dict) else {"raw": data},
            "message": message,
        }

    async def aclose(self) -> None:
        """MCP 클라이언트 리소스 정리."""
        if self.client:
            await self.client.aclose()

    def __del__(self) -> None:
        """소멸자에서 리소스 정리(이벤트 루프 비동작 시)."""
        if self.client:
            try:
                import asyncio

                loop = asyncio.get_event_loop()
                if not loop.is_running():
                    loop.run_until_complete(self.aclose())
            except Exception:
                pass


__all__ = ["StockService"]
