"""주식 데이터 포맷터"""

from typing import Dict, Any, List


class StockDataFormatter:
    """주식 데이터를 사용자 친화적으로 포맷팅"""

    @staticmethod
    def format_info(content: Dict[str, Any]) -> str:
        """기본 정보 포맷팅"""
        basic = content.get("Basic Information", {})
        financial = content.get("Financial Data", {})
        freshness = content.get("Data Freshness", {})

        lines = []

        # 헤더
        company_name = basic.get("Company Name", "N/A")
        lines.append(f"{'='*60}")
        lines.append(f"📊 {company_name}")
        lines.append(f"{'='*60}")
        lines.append("")

        # 기본 정보
        lines.append("🏢 기본 정보")
        lines.append(f"  • 업종: {basic.get('Industry Classification', 'N/A')}")
        lines.append(f"  • 시장: {basic.get('Listed Market', 'N/A')}")
        lines.append(f"  • 직원 수: {basic.get('Number of Employees', 0):,}명")
        lines.append(f"  • 웹사이트: {basic.get('Official Website', 'N/A')}")
        lines.append("")

        # 재무 정보
        lines.append("💰 재무 지표")
        price = financial.get("Latest Stock Price", 0)
        lines.append(f"  • 현재가: {price:,}원")
        lines.append(f"  • PER: {financial.get('Price-Earnings Ratio', 0):.2f}")
        lines.append(f"  • PBR: {financial.get('Price-Book Ratio', 0):.2f}")
        lines.append(f"  • 배당수익률: {financial.get('Dividend Yield', 0):.2f}%")
        lines.append("")

        # 데이터 출처
        lines.append(
            f"📡 데이터: {freshness.get('Data Source', 'N/A')} "
            f"({freshness.get('Data Quality', 'N/A')})"
        )

        return "\n".join(lines)

    @staticmethod
    def format_price_data(content: Dict[str, Any], days: int = 5) -> str:
        """가격 데이터 포맷팅"""
        price_data = content.get("price_data", [])

        if not price_data:
            return "❌ 가격 데이터가 없습니다."

        recent = price_data[-days:]

        lines = []
        lines.append(f"{'='*60}")
        lines.append(f"📈 주가 동향 (최근 {len(recent)}일)")
        lines.append(f"기간: {content.get('start_date')} ~ {content.get('end_date')}")
        lines.append(f"{'='*60}")
        lines.append("")

        for day in recent:
            date = day["Date"]
            open_price = day["Open"]
            close_price = day["Close"]
            high = day["High"]
            low = day["Low"]
            volume = day["Volume"]

            change = close_price - open_price
            change_pct = (change / open_price * 100) if open_price else 0

            # 이모지 선택
            if change > 0:
                emoji = "🔺"
                color = "상승"
            elif change < 0:
                emoji = "🔻"
                color = "하락"
            else:
                emoji = "➖"
                color = "보합"

            lines.append(f"{emoji} {date} ({color})")
            lines.append(
                f"   종가: {close_price:,}원 (시가 대비 {change:+,}원, {change_pct:+.2f}%)"
            )
            lines.append(f"   고가: {high:,}원 | 저가: {low:,}원")
            lines.append(f"   거래량: {volume:,}주")
            lines.append("")

        # 통계 정보
        first_close = recent[0]["Close"]
        last_close = recent[-1]["Close"]
        period_change = last_close - first_close
        period_change_pct = (period_change / first_close * 100) if first_close else 0

        lines.append(f"📊 기간 통계")
        lines.append(f"  • 시작가: {first_close:,}원")
        lines.append(f"  • 종가: {last_close:,}원")
        lines.append(f"  • 변동: {period_change:+,}원 ({period_change_pct:+.2f}%)")

        return "\n".join(lines)

    @staticmethod
    def format_market_cap(content: Dict[str, Any], days: int = 5) -> str:
        """시가총액 데이터 포맷팅

        Args:
            content: MCP 서버로부터 받은 시가총액 데이터
            days: 표시할 일수 (기본값: 5일)

        Returns:
            포맷팅된 시가총액 정보 문자열
        """
        try:
            market_cap_data = content.get("market_cap_data", [])

            if not market_cap_data:
                return "❌ 시가총액 데이터가 없습니다."

            recent = market_cap_data[-days:]

            lines = []
            lines.append(f"{'='*60}")
            lines.append(f"💎 시가총액 추이 (최근 {len(recent)}일)")
            lines.append(
                f"기간: {content.get('start_date')} ~ {content.get('end_date')}"
            )
            lines.append(f"{'='*60}")
            lines.append("")

            for day in recent:
                date = day["Date"]
                close_price = day["Close_Price"]
                market_cap = day["Market_Cap"]
                volume = day["Volume"]

                # 시가총액을 조 단위로 변환
                market_cap_trillion = market_cap / 1_000_000_000_000

                lines.append(f"📅 {date}")
                lines.append(f"   종가: {close_price:,}원")
                lines.append(f"   시가총액: {market_cap_trillion:.2f}조원")
                lines.append(f"   거래량: {volume:,}주")
                lines.append("")

            # 통계 정보
            first_cap = recent[0]["Market_Cap"]
            last_cap = recent[-1]["Market_Cap"]
            cap_change = last_cap - first_cap
            cap_change_pct = (cap_change / first_cap * 100) if first_cap else 0

            lines.append(f"📊 기간 통계")
            lines.append(f"  • 시작 시가총액: {first_cap / 1_000_000_000_000:.2f}조원")
            lines.append(f"  • 종료 시가총액: {last_cap / 1_000_000_000_000:.2f}조원")
            lines.append(
                f"  • 변동: {cap_change / 1_000_000_000_000:+.2f}조원 ({cap_change_pct:+.2f}%)"
            )

            return "\n".join(lines)
        except Exception as e:
            return f"시가총액 데이터 포맷팅 실패: {str(e)}"

    @staticmethod
    def format_fundamental(content: Dict[str, Any]) -> str:
        """기본 재무지표 포맷팅

        Args:
            content: MCP 서버로부터 받은 재무지표 데이터

        Returns:
            포맷팅된 재무지표 정보 문자열
        """
        try:
            fundamental = content.get("fundamental_data", {})

            if not fundamental:
                return "❌ 재무지표 데이터가 없습니다."

            lines = []
            lines.append(f"{'='*60}")
            lines.append(f"📈 재무 지표 (Fundamental)")
            lines.append(f"{'='*60}")
            lines.append("")

            # 기업 규모
            lines.append("🏢 기업 규모")

            # 안전하게 숫자로 변환
            try:
                market_cap = float(fundamental.get("Market Cap", 0))
                market_cap_str = f"{market_cap / 1_000_000_000_000:.2f}조원"
            except (ValueError, TypeError):
                market_cap_str = "N/A"

            try:
                enterprise_value = float(fundamental.get("Enterprise Value", 0))
                enterprise_value_str = f"{enterprise_value / 1_000_000_000_000:.2f}조원"
            except (ValueError, TypeError):
                enterprise_value_str = "N/A"

            lines.append(f"  • 시가총액: {market_cap_str}")
            lines.append(f"  • 기업가치(EV): {enterprise_value_str}")
            lines.append("")

            # 수익성 지표
            lines.append("💰 수익성 지표")
            per = fundamental.get("Price-Earnings Ratio (PER)", "N/A")
            pbr = fundamental.get("Price-Book Ratio (PBR)", "N/A")
            dividend = fundamental.get("Dividend Yield", "N/A")

            lines.append(f"  • PER: {per}")
            lines.append(f"  • PBR: {pbr}")
            lines.append(f"  • 배당수익률: {dividend}")
            lines.append("")

            # 재무 안정성
            lines.append("🛡️ 재무 안정성")

            # 안전하게 숫자로 변환
            try:
                current_ratio = float(fundamental.get("Current Ratio", 0))
                current_ratio_str = f"{current_ratio:.2f}"
            except (ValueError, TypeError):
                current_ratio_str = "N/A"

            try:
                quick_ratio = float(fundamental.get("Quick Ratio", 0))
                quick_ratio_str = f"{quick_ratio:.2f}"
            except (ValueError, TypeError):
                quick_ratio_str = "N/A"

            try:
                debt_to_equity = float(fundamental.get("Debt to Equity", 0))
                debt_to_equity_str = f"{debt_to_equity:.2f}"
            except (ValueError, TypeError):
                debt_to_equity_str = "N/A"

            lines.append(f"  • 유동비율: {current_ratio_str}")
            lines.append(f"  • 당좌비율: {quick_ratio_str}")
            lines.append(f"  • 부채비율: {debt_to_equity_str}")
            lines.append("")

            # 수익률 지표
            lines.append("📊 수익률 지표")

            # 안전하게 숫자로 변환
            try:
                roe = float(fundamental.get("Return on Equity (ROE)", 0))
                roe_str = f"{roe * 100:.2f}%"
            except (ValueError, TypeError):
                roe_str = "N/A"

            try:
                roa = float(fundamental.get("Return on Assets (ROA)", 0))
                roa_str = f"{roa * 100:.2f}%"
            except (ValueError, TypeError):
                roa_str = "N/A"

            lines.append(f"  • ROE (자기자본이익률): {roe_str}")
            lines.append(f"  • ROA (총자산이익률): {roa_str}")

            return "\n".join(lines)
        except Exception as e:
            return f"재무지표 데이터 포맷팅 실패: {str(e)}"
