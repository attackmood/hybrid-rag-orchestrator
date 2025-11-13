# Hybrid RAG Orchestrator

## 📋 프로젝트 개요

**Hybrid RAG Orchestrator**는 ReAct 패턴을 기반으로 한 지능형 쿼리 처리 시스템으로, 사용자의 질의를 LLM이 의도 분석하여 적절한 도구들을 병렬로 실행하고, 신뢰도와 복잡도에 따라 최적의 답변을 제공합니다.


### ✨ 주요 특징

- 🧠 **LLM 기반 의도 분석**: ReAct 방식으로 사용자 질의의 의도를 자동 분석
- ⚡ **병렬 도구 실행**: 여러 도구를 동시에 실행하여 성능 최적화
- 🔄 **적응형 워크플로우**: 복잡도에 따라 다른 답변 생성 전략 적용
- 🔗 **다중 소스 통합**: 날씨, 주식, 웹 검색, RAG 시스템을 하나로 통합


### 🔧 사용 가능한 도구

- **🌤️ Weather**: MCP 기반 날씨 정보 조회
- **📈 Stock Info**: MCP 기반 주식 정보 조회
- **🔍 Web Search**: Google Search API
- **📚 Knowledge Base**: RAG 기반 문서 검색
- **🧮 Calculator**: 수학 계산
- **🤔 Reasoning**: LLM 기반 논리적 추론
 

---
## 🔧 기술 스택 상세

### Backend Framework
- **FastAPI**: 비동기 웹 프레임워크
- **Uvicorn**: ASGI 서버
- **Pydantic**: 데이터 검증 및 설정 관리

### LLM & AI Framework
- **LangChain**: LLM 체인 및 프롬프트 관리
- **LangGraph**: StateGraph 기반 워크플로우 오케스트레이션
- **Ollama**: 로컬 LLM 서버 (llama3.1:8b)

### Vector Database & Embeddings
- **ChromaDB**: 벡터 저장소 (코사인 유사도)
- **Sentence-Transformers**: 한국어 임베딩 모델 (jhgan/ko-sbert-nli)
- **PyMuPDF**: PDF 텍스트 추출


### External Services
- **Google Custom Search API**: 웹 검색
- **MCP (Model Context Protocol)**: 날씨/주식 API 통합
- **WebSockets**: MCP 서버 통신

### Development & Monitoring
- **Loguru**: 구조화된 로깅
- **Python-dotenv**: 환경변수 관리
- **Jinja2**: 템플릿 엔진


---

## 🏗️ 전체 아키텍처

1. **의도 분석**: LLM이 사용자 쿼리를 분석하여 필요한 도구 선택
2. **병렬 실행**: 선택된 도구들을 동시에 실행
3. **점수 계산**: 각 결과의 관련성, 신뢰도, 속도 평가
4. **적응형 통합**: 복잡도에 따라 다른 방식으로 결과 통합
   - 간단한 쿼리: 직접 반환
   - 복잡한 쿼리: LLM으로 통합


### 아키텍처 다이어그램

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

---

**이 프로젝트는 AI 개발 학습 과정에서 만들어진 토이 프로젝트입니다.** 
더 자세한 내용은 [SERVICE_ARCHITECTURE.md](./SERVICE_ARCHITECTURE.md)와 [IMPLEMENTATION_GUIDE.md](./IMPLEMENTATION_GUIDE.md)를 참고하세요.
