"""
MCP 날씨 서비스 (Service Module)

쿼리 파싱(간단 NER), 엔드포인트 선택, JSON-RPC 호출, 응답 데이터 구조화를 수행합니다.
"""

from __future__ import annotations

from typing import Any, Dict, Optional
from utils.logger import log

from ..client import MCPClient, MCPResponse


class WeatherService:
    """날씨 서비스 핸들러.

    - 쿼리 파싱 → 매개변수 추출 → JSON-RPC 툴 호출 → 응답 구조화
    """

    def __init__(self, client: Optional[MCPClient] = None) -> None:
        """WeatherService를 초기화합니다."""
        """
        Args:
            client: 주입 가능한 MCPClient 인스턴스. 미제공 시 기본값 사용
        """
        self.client = client or MCPClient()
        log.info(f"WeatherService 초기화: {self.client}")

    async def get_current_weather(self, city: str) -> Dict[str, Any]:
        """도시명으로 현재 날씨 조회."""
        try:
            raw = await self.client.call_tool_jsonrpc_async(
                method_name="get_current_weather", arguments={"city": city}
            )
            resp: MCPResponse = await self.client.to_response_async(
                "get_current_weather", raw
            )
            return self._format_response(resp.success, resp.raw_data, resp.content_text)
        except Exception as exc:
            log.error(f"현재 날씨 조회 실패: {exc}")
            return self._format_response(False, None, f"현재 날씨 조회 실패: {exc}")

    async def get_weather_forecast(self, city: str) -> Dict[str, Any]:
        """도시명으로 날씨 예보 조회."""
        try:
            raw = await self.client.call_tool_jsonrpc_async(
                method_name="get_weather_forecast", arguments={"city": city}
            )
            resp: MCPResponse = await self.client.to_response_async(
                "get_weather_forecast", raw
            )
            return self._format_response(resp.success, resp.raw_data, resp.content_text)
        except Exception as exc:
            log.error(f"도시 날씨 예보 조회 실패: {exc}")
            return self._format_response(
                False, None, f"도시 날씨 예보 조회 실패: {exc}"
            )

    async def get_weather_by_coordinates(
        self, lat: float, lon: float
    ) -> Dict[str, Any]:
        """좌표로 날씨 조회."""
        try:
            raw = await self.client.call_tool_jsonrpc_async(
                method_name="get_weather_by_coordinates",
                arguments={"lat": lat, "lon": lon},
            )
            resp: MCPResponse = await self.client.to_response_async(
                "get_weather_by_coordinates", raw
            )
            return self._format_response(resp.success, resp.raw_data, resp.content_text)
        except Exception as exc:
            log.error(f"좌표 기반 날씨 조회 실패: {exc}")
            return self._format_response(
                False, None, f"좌표 기반 날씨 조회 실패: {exc}"
            )

    def _format_response(
        self, success: bool, data: Optional[Dict[str, Any]], message: str
    ) -> Dict[str, Any]:
        """일관된 응답 포맷으로 반환합니다.
        Args:
            success: 성공 여부
            data: 데이터
            message: 메시지
        Returns:
            Dict[str, Any]: 응답 데이터
        """
        return {
            "success": success,
            "data": data if isinstance(data, dict) else {"raw": data},
            "message": message,
        }

    async def aclose(self) -> None:
        """MCP 클라이언트 리소스를 정리합니다."""
        if self.client:
            await self.client.aclose()

    def __del__(self):
        """소멸자에서 리소스 정리"""
        if self.client:
            try:
                import asyncio

                loop = asyncio.get_event_loop()
                if not loop.is_running():
                    loop.run_until_complete(self.aclose())
            except Exception:
                pass


__all__ = ["WeatherService"]
