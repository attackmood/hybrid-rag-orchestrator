# Hybrid RAG Orchestrator

AI 기반 하이브리드 쿼리 처리 시스템으로, LLM이 의도를 분석하여 적절한 도구들을 병렬로 실행하고 최적의 답변을 제공합니다.

## 📋 프로젝트 소개

**Hybrid RAG Orchestrator**는 사용자의 질의를 LLM이 의도 분석하여 여러 도구(MCP 서비스, Google Search, RAG 시스템)를 병렬로 실행하고, 신뢰도와 복잡도에 따라 적응형 답변을 생성하는 시스템입니다.

### ✨ 주요 특징

- 🧠 **LLM 기반 의도 분석**: ReAct 방식으로 사용자 질의의 의도를 자동 분석
- ⚡ **병렬 도구 실행**: 여러 도구를 동시에 실행하여 성능 최적화
- 🔄 **적응형 워크플로우**: 복잡도에 따라 다른 답변 생성 전략 적용
- 🔗 **다중 소스 통합**: 날씨, 주식, 웹 검색, RAG 시스템을 하나로 통합

## 🚀 시작하기

### 환경 요구사항

- Python 3.12 이상
- Ollama (로컬 LLM 서버)
- Google Custom Search API 키

### 설치

```bash
# 레포지토리 클론
git clone https://github.com/attackmood/hybrid-rag-orchestrator.git
cd hybrid-rag-orchestrator

# 가상환경 생성 및 활성화
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 의존성 설치
pip install -r requirements.txt

# 환경변수 설정
cp env.example .env
# .env 파일을 열어 API 키 및 설정 값 입력
```

### 환경변수 설정

`.env` 파일에 다음 정보를 설정하세요:

```env
# Ollama 설정
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.1:8b

# Google Search API
GOOGLE_API_KEY=your_api_key
GOOGLE_CSE_ID=your_cse_id

# MCP 서버 설정
MCP_WEBSOCKET_URL=ws://localhost:8765/ws

# 서버 설정
HOST=0.0.0.0
PORT=8000
```

### 실행

```bash
# 서버 시작
python -m app.main

# 또는 uvicorn 직접 실행
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

브라우저에서 `http://localhost:8000` 접속

## 📁 프로젝트 구조

```
hybrid-rag-orchestrator/
├── api/                    # API 엔드포인트
│   ├── chat.py            # 채팅 API
│   ├── health.py          # 헬스체크
│   └── models.py          # Pydantic 모델
├── app/                    # FastAPI 애플리케이션
│   ├── config.py          # 앱 설정
│   └── main.py            # 엔트리포인트
├── core/                   # 핵심 로직
│   ├── hybrid_router.py   # 하이브리드 라우터 (LangGraph)
│   ├── ollama_client.py   # Ollama 클라이언트
│   └── tools_registry.py  # 도구 레지스트리
├── services/               # 외부 서비스 통합
│   ├── google_search/     # Google Search API
│   ├── mcp/               # MCP 서비스 (날씨, 주식)
│   └── rag/               # RAG 시스템
│       ├── chroma_client.py
│       ├── pdf_processor.py
│       └── vector_search.py
├── config/                 # 설정 관리
│   └── settings.py         # 전역 설정
├── utils/                  # 유틸리티
│   ├── embeddings.py       # 임베딩 모델
│   └── logger.py           # 로깅
├── static/                 # 정적 파일
│   ├── css/
│   └── js/
├── templates/              # HTML 템플릿
├── data/                   # 데이터 저장소
│   ├── cache/             # 임베딩 캐시
│   ├── chroma_db/         # ChromaDB 저장소
│   └── pdf_temp/          # PDF 임시 파일
└── requirements.txt        # Python 의존성
```

## 🛠️ 사용 방법

### 1. 웹 UI 사용

브라우저에서 접속하여 채팅 인터페이스를 사용합니다.

### 2. API 호출

#### 채팅 요청

```bash
curl -X POST http://localhost:8000/api/chat/query \
  -H "Content-Type: application/json" \
  -d '{
    "message": "서울 날씨가 어때?",
    "session_id": "test_session"
  }'
```

#### 응답 예시

```json
{
  "success": true,
  "message": "서울의 현재 날씨는 맑음, 기온은 15도입니다.",
  "session_id": "test_session",
  "processing_time": 2.3,
  "mode_used": "parallel",
  "metadata": {
    "complexity_score": 0.25,
    "selected_tools": ["weather"]
  }
}
```

### 3. PDF 업로드

RAG 시스템에 PDF를 추가할 수 있습니다:

```bash
curl -X POST http://localhost:8000/api/chat/upload-pdf \
  -F "file=@document.pdf" \
  -F "add_to_chroma=true"
```

## 🔧 사용 가능한 도구

- **🌤️ Weather**: MCP 기반 날씨 정보 조회
- **📈 Stock Info**: MCP 기반 주식 정보 조회
- **🔍 Web Search**: Google Search API
- **📚 Knowledge Base**: RAG 기반 문서 검색
- **🧮 Calculator**: 수학 계산
- **🤔 Reasoning**: LLM 기반 논리적 추론

## 📊 기술 스택

- **Framework**: FastAPI, Uvicorn
- **LLM**: LangChain, LangGraph, Ollama
- **Database**: ChromaDB (벡터 DB)
- **Embedding**: Sentence-Transformers (Ko-SBERT)
- **External**: Google Search API, MCP Protocol

## 🎯 작동 방식

1. **의도 분석**: LLM이 사용자 쿼리를 분석하여 필요한 도구 선택
2. **병렬 실행**: 선택된 도구들을 동시에 실행
3. **점수 계산**: 각 결과의 관련성, 신뢰도, 속도 평가
4. **적응형 통합**: 복잡도에 따라 다른 방식으로 결과 통합
   - 간단한 쿼리: 직접 반환
   - 복잡한 쿼리: LLM으로 통합

## 📝 라이선스

MIT License

## 🤝 기여

이슈 등록 및 Pull Request 환영합니다.

---

**이 프로젝트는 AI 개발 학습 과정에서 만들어진 토이 프로젝트입니다.** 
더 자세한 내용은 `SERVICE_ARCHITECTURE.md`와 `IMPLEMENTATION_GUIDE.md`를 참고하세요.
