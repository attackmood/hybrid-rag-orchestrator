# Hybrid RAG Orchestrator 서비스 아키텍처

## 🏗️ 전체 아키텍처

```mermaid
graph TB
    subgraph "Frontend Layer"
        UI[Web UI<br/>FastAPI + Jinja2]
    end
    
    subgraph "API Layer"
        API[FastAPI Server<br/>RESTful API]
        CHAT[Chat API<br/>/api/chat]
        HEALTH[Health API<br/>/api/health]
    end
    
    subgraph "Core Orchestration Layer"
        HR[HybridRouter<br/>LangGraph StateGraph]
        TR[ToolsRegistry<br/>도구 관리]
    end
    
    subgraph "LLM Layer"
        OLLAMA[Ollama Client<br/>LangChain 기반]
        MODEL[LLM Model<br/>llama3.1:8b]
    end
    
    subgraph "Tool Layer"
        subgraph "External Services"
            GS[Google Search<br/>LangChain API]
            MCP[MCP Services<br/>WebSocket JSON-RPC]
        end
        
        subgraph "RAG System"
            VS[VectorSearchManager<br/>통합 검색]
            CHROMA[ChromaDB<br/>영구 저장소]
            PDF[PDFProcessor<br/>실시간 처리]
        end
        
        subgraph "Utility Tools"
            CALC[Calculator<br/>수학 계산]
            REASON[Reasoning<br/>논리적 추론]
        end
    end
    
    subgraph "Data Layer"
        EMBED[Korean Embedding<br/>jhgan/ko-sbert-nli]
        CACHE[Cache System<br/>임베딩 캐시]
        LOGS[Logging<br/>Loguru]
    end
    
    UI --> API
    API --> CHAT
    API --> HEALTH
    CHAT --> HR
    HR --> TR
    HR --> OLLAMA
    OLLAMA --> MODEL
    TR --> GS
    TR --> MCP
    TR --> VS
    TR --> CALC
    TR --> REASON
    VS --> CHROMA
    VS --> PDF
    VS --> EMBED
    EMBED --> CACHE
    HR --> LOGS
```



## 🛠️ 도구 시스템 상세

### 1. 도구 레지스트리 (ToolsRegistry)

```python
# 도구 목록
tools = {
    "weather": weather_tool,           # MCP 기반 날씨 조회
    "stock_info": stock_info_tool,      # MCP 기반 주식 정보
    "calculator": calculator_tool,      # 수학 계산
    "web_search": web_search_tool,      # Google Search API
    "knowledge_base": knowledge_base_tool, # RAG 시스템
    "reasoning": reasoning_tool         # LLM 기반 추론
}
```

### 2. MCP (Model Context Protocol) 서비스

```mermaid
graph LR
    subgraph "MCP Client"
        WS[WebSocket Client<br/>JSON-RPC 2.0]
        INIT[Initialize<br/>Handshake]
        CALL[Tool Call<br/>tools/call]
    end
    
    subgraph "MCP Server"
        WEATHER[Weather Service<br/>날씨 API]
        STOCK[Stock Service<br/>주식 API]
    end
    
    WS --> INIT
    INIT --> CALL
    CALL --> WEATHER
    CALL --> STOCK
```

**특징:**
- WebSocket 기반 JSON-RPC 통신
- 연결 재사용 및 자동 재연결
- 순차적 요청 처리로 안정성 확보

### 3. Google Search 통합

```mermaid
graph TB
    subgraph "Google Search Flow"
        QUERY[검색 쿼리]
        LC[LangChain API Wrapper]
        GS[Google Custom Search API]
        PARSE[결과 파싱]
        SCORE[관련성 점수 계산]
        FORMAT[형식화된 결과]
    end
    
    QUERY --> LC
    LC --> GS
    GS --> PARSE
    PARSE --> SCORE
    SCORE --> FORMAT
```

**특징:**
- LangChain GoogleSearchAPIWrapper 활용
- 비동기 스레드 풀 실행
- 관련성 점수 기반 랭킹

### 4. RAG 시스템 아키텍처

```mermaid
graph TB
    subgraph "RAG System"
        subgraph "VectorSearchManager"
            VS[통합 검색 관리자]
            INTEGRATE[결과 통합]
            OPTIMIZE[컨텍스트 최적화]
        end
        
        subgraph "ChromaDB (영구 저장소)"
            CHROMA[ChromaDB Client]
            COLLECTION[Document Collection]
            EMBED_STORE[임베딩 저장소]
        end
        
        subgraph "PDFProcessor (실시간 처리)"
            PDF[PDF 처리기]
            CHUNK[청킹 시스템]
            TEMP_STORE[임시 벡터 저장소]
        end
        
        subgraph "Embedding System"
            KO_EMBED[Korean Embedding Model<br/>jhgan/ko-sbert-nli]
            CACHE[임베딩 캐시]
        end
    end
    
    VS --> CHROMA
    VS --> PDF
    CHROMA --> COLLECTION
    CHROMA --> EMBED_STORE
    PDF --> CHUNK
    PDF --> TEMP_STORE
    CHROMA --> KO_EMBED
    PDF --> KO_EMBED
    KO_EMBED --> CACHE
    VS --> INTEGRATE
    INTEGRATE --> OPTIMIZE
```

**RAG 시스템 특징:**
- **이중 저장소**: ChromaDB(영구) + 메모리(임시)
- **병렬 검색**: 두 저장소를 동시에 검색
- **중복 제거**: 의미적 유사도 기반 중복 제거
- **컨텍스트 최적화**: 길이 제한 내 최적 결과 선택




