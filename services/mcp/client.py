"""
MCP (Model Context Protocol) 클라이언트 - WebSocket 기반 JSON-RPC 통신

MCP는 WebSocket을 통한 JSON-RPC 프로토콜을 사용합니다:
1. WebSocket 연결
2. initialize 메서드로 핸드셰이크
3. tools/call 메서드로 도구 호출
"""

import asyncio
import json
import time
from dataclasses import dataclass
from typing import Dict, Any, Optional
import websockets
from config.settings import settings
from utils.logger import log


@dataclass
class MCPResponse:
    """MCP 서버 응답 데이터"""

    success: bool
    content_text: str
    raw_data: Dict[str, Any]
    tool_name: str


class MCPClient:
    """WebSocket 기반 MCP 클라이언트 - 개선된 연결 재사용"""

    def __init__(self, websocket_url: Optional[str] = None) -> None:
        self.mcp_config = settings.mcp.SERVERS["mcp_server_jsonrpc"]
        self.websocket_url = websocket_url or self.mcp_config["url"]

        # WebSocket 연결 상태
        self.websocket: Optional[websockets.WebSocketServerProtocol] = None
        self.initialized: bool = False
        self.request_id_counter: int = 1
        self._connection_lock = asyncio.Lock()  # 동시 연결 시도 방지
        self._request_lock = asyncio.Lock()  # 동시 요청 방지 (추가!)

    async def ensure_connected(self) -> bool:
        """연결 상태를 확인하고 필요시에만 연결"""

        # 기본 상태 확인
        basic_connected = self._is_websocket_connected()
        log.info(
            f"기본 WebSocket 상태: {basic_connected}, 초기화 상태: {self.initialized}"
        )

        # 기본 확인이 성공하고 초기화되어 있으면 ping으로 실제 확인
        if basic_connected and self.initialized:
            ping_success = await self._is_websocket_connected_with_ping()
            log.info(f"Ping 테스트 결과: {ping_success}")
            if ping_success:
                log.info("기존 연결 재사용")
                return True

        # 동시 연결 시도 방지
        async with self._connection_lock:
            # 락 획득 후 다시 한번 확인
            basic_connected = self._is_websocket_connected()
            if basic_connected and self.initialized:
                ping_success = await self._is_websocket_connected_with_ping()
                if ping_success:
                    log.info("락 획득 후 확인: 다른 태스크가 이미 연결함")
                    return True

            log.info("WebSocket 연결 필요 - 새로운 연결 시도")
            return await self._connect_with_retry()

    async def _connect_with_retry(self) -> bool:
        """재시도 로직이 포함된 연결"""
        for attempt in range(settings.mcp.RETRY_ATTEMPTS + 1):
            try:
                log.info(
                    f"MCP 연결 시도 {attempt + 1}/{settings.mcp.RETRY_ATTEMPTS + 1}"
                )

                # 기존 연결 정리
                await self._close_websocket_safely()

                # 새 WebSocket 연결
                self.websocket = await websockets.connect(
                    self.websocket_url,
                    ping_interval=settings.mcp.PING_INTERVAL,
                    ping_timeout=settings.mcp.PING_TIMEOUT,
                    close_timeout=settings.mcp.CLOSE_TIMEOUT,
                )

                # MCP 초기화
                if await self._initialize_mcp():
                    self.initialized = True
                    return True
                else:
                    await self._close_websocket_safely()

            except Exception as e:
                log.error(f"연결 시도 {attempt + 1} 실패: {e}")
                await self._close_websocket_safely()

            if attempt < settings.mcp.RETRY_ATTEMPTS:
                await asyncio.sleep(settings.mcp.RETRY_DELAY)

        return False

    def _is_websocket_connected(self) -> bool:
        """WebSocket 연결 상태 안전 확인"""
        try:
            if self.websocket is None:
                log.debug("WebSocket이 None입니다")
                return False

            # websockets 라이브러리 버전에 따라 다른 방식으로 상태 확인
            try:
                # 방법 1: closed 속성 확인 (일부 버전)
                if hasattr(self.websocket, "closed"):
                    log.debug(f"WebSocket closed 상태: {self.websocket.closed}")
                    if self.websocket.closed:
                        return False

                # 방법 2: state 속성 확인
                if hasattr(self.websocket, "state"):
                    log.debug(f"WebSocket state: {self.websocket.state}")
                    # OPEN 상태인지 확인
                    if hasattr(websockets, "State"):
                        is_open = self.websocket.state == websockets.State.OPEN
                        log.debug(f"WebSocket state == OPEN: {is_open}")
                        return is_open
                    else:
                        # State enum이 없는 경우 문자열로 확인
                        return str(self.websocket.state) == "OPEN"

                # 방법 3: close_code 확인 (연결이 끊어지면 close_code가 설정됨)
                if hasattr(self.websocket, "close_code"):
                    log.debug(
                        f"WebSocket close_code: {getattr(self.websocket, 'close_code', None)}"
                    )
                    return getattr(self.websocket, "close_code", None) is None

                # 모든 방법이 실패하면 객체 존재만으로 판단
                log.debug(
                    "WebSocket 상태 확인 방법을 찾을 수 없음 - 객체 존재만으로 판단"
                )
                return True

            except Exception as e:
                log.warning(f"WebSocket 상태 속성 확인 중 오류: {e}")
                return False

        except Exception as e:
            log.warning(f"WebSocket 연결 상태 확인 중 예외: {e}")
            return False

    async def _is_websocket_connected_with_ping(self) -> bool:
        """ping을 통한 실제 연결 상태 확인"""
        try:
            if self.websocket is None:
                return False

            # 실제 ping을 보내서 연결 상태 확인
            await asyncio.wait_for(self.websocket.ping(), timeout=1.0)
            log.debug("WebSocket ping 성공 - 연결 활성화")
            return True

        except asyncio.TimeoutError:
            log.debug("WebSocket ping 타임아웃 - 연결 끊어짐")
            return False
        except Exception as e:
            log.debug(f"WebSocket ping 실패: {e}")
            return False

    async def health_check(self) -> bool:
        """연결 상태 헬스체크"""
        if not self._is_websocket_connected():
            return False

        try:
            # ping으로 연결 상태 확인
            await self.websocket.ping()
            return True
        except Exception:
            log.warning("헬스체크 실패 - 연결이 끊어진 것 같습니다")
            self.initialized = False
            return False

    async def call_tool_jsonrpc_async(
        self, method_name: str, arguments: Dict[str, Any]
    ) -> Dict[str, Any]:
        """WebSocket을 통한 MCP 도구 호출 - 연결 재사용 + 순차 처리"""
        start_time = time.time()

        # 동시 요청 방지 - 하나씩 순차적으로 처리
        async with self._request_lock:
            try:
                # 연결 상태 확인 및 필요시에만 연결
                if not await self.ensure_connected():
                    return self._create_error_response(
                        "connection_failed", method_name, time.time() - start_time
                    )

                # 도구 호출
                tool_call = {
                    "jsonrpc": "2.0",
                    "id": self._get_next_request_id(),
                    "method": "tools/call",
                    "params": {"name": method_name, "arguments": arguments or {}},
                }

                log.info(f"MCP 도구 호출: {method_name}")

                # 요청 전송
                await self.websocket.send(json.dumps(tool_call, ensure_ascii=False))

                # 응답 대기 (타임아웃 설정)
                response = await asyncio.wait_for(
                    self.websocket.recv(), timeout=settings.mcp.TIMEOUT
                )

                response_data = json.loads(response)
                total_time = time.time() - start_time
                log.info(f"MCP 호출 완료: {method_name} ({total_time:.3f}s)")

                return response_data

            except websockets.exceptions.ConnectionClosed:
                log.warning(f"연결 끊어짐 감지: {method_name}")
                # 연결 상태 리셋하고 한 번만 재시도
                self.initialized = False
                self.websocket = None

                if await self.ensure_connected():
                    log.info("재연결 성공 - 도구 호출 재시도")
                    return await self.call_tool_jsonrpc_async(method_name, arguments)
                else:
                    return self._create_error_response(
                        "connection_lost", method_name, time.time() - start_time
                    )

            except asyncio.TimeoutError:
                log.error(f"응답 타임아웃: {method_name}")
                return self._create_error_response(
                    "timeout", method_name, time.time() - start_time
                )
            except Exception as e:
                log.error(f"도구 호출 오류: {method_name} - {e}")
                return self._create_error_response(
                    f"error: {str(e)}", method_name, time.time() - start_time
                )

    async def _initialize_mcp(self) -> bool:
        """MCP 초기화"""
        try:
            init_message = {
                "jsonrpc": "2.0",
                "id": self._get_next_request_id(),
                "method": "initialize",
                "params": {
                    "protocolVersion": "2025-06-18",
                    "capabilities": {"tools": {}, "resources": {}},
                    "clientInfo": {"name": "SmartRAG", "version": "1.0.0"},
                },
            }

            await self.websocket.send(json.dumps(init_message, ensure_ascii=False))

            # 응답 대기 (여러 메시지 처리 가능)
            max_wait_time = settings.mcp.TIMEOUT
            start_time = time.time()

            while time.time() - start_time < max_wait_time:
                try:
                    response = await asyncio.wait_for(
                        self.websocket.recv(), timeout=2.0
                    )
                    response_data = json.loads(response)
                    log.debug(f"MCP 초기화 응답: {response_data}")

                    # 초기화 성공 응답 확인
                    if (
                        response_data.get("id") == init_message["id"]
                        and "result" in response_data
                    ):
                        log.info("MCP 초기화 성공")
                        return True

                    # 초기화 실패 응답 확인
                    elif (
                        response_data.get("id") == init_message["id"]
                        and "error" in response_data
                    ):
                        error_info = response_data.get("error", {})
                        error_msg = (
                            error_info.get("message", "알 수 없는 오류")
                            if isinstance(error_info, dict)
                            else str(error_info)
                        )
                        log.error(f"MCP 초기화 실패: {error_msg}")
                        return False

                    # notifications/initialized 메시지 (무시하고 계속 대기)
                    elif response_data.get("method") == "notifications/initialized":
                        log.debug("MCP 서버 초기화 알림 수신")
                        continue

                    # 기타 메시지
                    else:
                        log.info(f"기타 MCP 메시지: {response_data}")
                        continue

                except asyncio.TimeoutError:
                    log.debug("MCP 초기화 응답 대기 중...")
                    continue
                except json.JSONDecodeError as e:
                    log.error(f"MCP 응답 JSON 파싱 오류: {e}")
                    continue
                except Exception as e:
                    log.warning(f"MCP 응답 처리 중 오류: {e}")
                    continue

            log.error("MCP 초기화 타임아웃")
            return False

        except Exception as e:
            log.error(f"MCP 초기화 중 오류: {e}")
            return False

    async def _close_websocket_safely(self) -> None:
        """안전한 WebSocket 연결 종료"""
        try:
            if self.websocket:
                # closed 속성이 있는 경우에만 확인
                if hasattr(self.websocket, "closed") and not self.websocket.closed:
                    await self.websocket.close()
                elif not hasattr(self.websocket, "closed"):
                    # closed 속성이 없는 경우 무조건 close 시도
                    await self.websocket.close()
        except Exception as e:
            log.debug(f"WebSocket 종료 중 예외 (정상): {e}")
        finally:
            self.websocket = None
            self.initialized = False

    def _get_next_request_id(self) -> int:
        """다음 요청 ID 생성"""
        request_id = self.request_id_counter
        self.request_id_counter += 1
        return request_id

    def _create_error_response(
        self, error_type: str, method_name: str, duration: float
    ) -> Dict[str, Any]:
        return {
            "success": False,
            "error": error_type,
            "method": method_name,
            "duration": duration,
        }

    async def to_response_async(
        self, method_name: str, data: Dict[str, Any]
    ) -> MCPResponse:
        """비동기 응답을 MCPResponse로 변환"""
        try:
            response = MCPResponse(
                success=bool(data.get("result")),
                content_text=self._extract_content_text(data),
                raw_data=data,
                tool_name=method_name,
            )
            return response

        except Exception as e:
            log.error(f"MCPResponse 변환 실패: {e}")
            # 기본 응답 반환
            return MCPResponse(
                success=False,
                content_text=f"응답 변환 실패: {str(e)}",
                raw_data=data,
                tool_name=method_name,
            )

    def _extract_content_text(self, data: Dict[str, Any]) -> str:
        """MCP JSON-RPC 응답에서 content의 text 부분을 추출합니다."""
        try:
            # 데이터가 문자열인 경우 직접 반환
            if isinstance(data, str):
                return data

            # JSON-RPC 응답 구조: {"jsonrpc": "2.0", "id": X, "result": {...}}
            if "error" in data:
                error = data["error"]
                if isinstance(error, dict):
                    return (
                        f"오류: {error.get('message', '알 수 없는 오류')} "
                        f"(코드: {error.get('code', 'N/A')})"
                    )
                else:
                    return f"오류: {str(error)}"

            result = data.get("result", {})
            if not result:
                return "응답 결과가 없습니다"

            # 일반화된 content 추출 로직
            if isinstance(result, dict):
                # 1) result.content 우선
                if "content" in result:
                    content_obj = result["content"]

                    # dict 형태
                    if isinstance(content_obj, dict):
                        # content.content 배열 내 첫 text
                        inner_list = content_obj.get("content")
                        if isinstance(inner_list, list) and inner_list:
                            first = inner_list[0] or {}
                            if isinstance(first, dict) and "text" in first:
                                return str(first["text"])

                        # content.message
                        if "message" in content_obj:
                            return str(content_obj["message"])  # 도메인 중립 메시지

                        # content.data는 도메인별로 상이 → JSON 문자열로 반환
                        if isinstance(content_obj.get("data"), dict):
                            try:
                                return json.dumps(
                                    content_obj["data"], ensure_ascii=False
                                )
                            except Exception:
                                pass

                        # content.text 직접 필드
                        if "text" in content_obj:
                            return str(content_obj["text"])

                    # list 형태
                    if isinstance(content_obj, list) and content_obj:
                        first = content_obj[0] or {}
                        if isinstance(first, dict) and "text" in first:
                            return str(first["text"])

                # 2) result 최상위에서 대체 필드 탐색
                if "message" in result:
                    return str(result["message"])  # 도메인 중립 메시지

                top_list = result.get("content")
                if isinstance(top_list, list) and top_list:
                    first = top_list[0] or {}
                    if isinstance(first, dict) and "text" in first:
                        return str(first["text"])

                # dict 자체를 JSON 문자열로 반환 (마지막 우선순위)
                try:
                    return json.dumps(result, ensure_ascii=False)
                except Exception:
                    return str(result)

            # result가 딕셔너리가 아닌 경우 직접 문자열화
            return str(result)

        except Exception as exc:
            log.warning(f"content 텍스트 추출 실패: {exc}")
            return "응답 처리 중 오류가 발생했습니다"

    async def aclose(self) -> None:
        """리소스 정리"""
        await self._close_websocket_safely()
        log.info("MCP 클라이언트 정리 완료")
