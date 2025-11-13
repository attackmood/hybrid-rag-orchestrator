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



## 🚀 프로젝트 발전 로드맵

### Phase 1: ReAct 순수성 강화 (1-2개월)

#### 🎯 **목표**: 전통적 ReAct 루프 구현
- **현재 상태**: 병렬 처리 중심 (75% 구현)
- **목표 상태**: 순차적 ReAct 루프 추가 (90% 구현)

#### 📋 **구현 계획**

**1.1 반복적 ReAct 루프 추가**
```python
# 새로운 워크플로우 추가
class HybridRouter:
    def __init__(self):
        # 기존 병렬 워크플로우
        self.parallel_workflow = self._create_parallel_workflow()
        # 새로운 ReAct 루프 워크플로우
        self.react_workflow = self._create_react_loop_workflow()
    
    async def process_query(self, query: str) -> Dict:
        complexity_score = self._calculate_complexity(query)
        
        if complexity_score < 0.3:
            # 간단한 쿼리: 병렬 처리 (기존)
            return await self._process_parallel(query)
        else:
            # 복잡한 쿼리: ReAct 루프 (신규)
            return await self._process_react_loop(query)
```

**1.2 동적 도구 선택 메커니즘**
```python
async def _reason_step(self, state: ReActLoopState) -> Dict:
    """추론 단계: 다음 행동을 생각"""
    reasoning_prompt = f"""
    현재 단계: {state['current_step']}/{state['max_steps']}
    원래 질문: {state['query']}
    
    이전 실행 결과:
    {self._build_context_from_history(state['execution_history'])}
    
    다음에 해야 할 일을 생각해보세요:
    1. 현재까지 얻은 정보로 질문에 답할 수 있나요?
    2. 아니면 추가 정보가 필요한가요?
    3. 어떤 도구를 사용해야 할까요?
    """
    
    response = await self.ollama_client.generate(prompt=reasoning_prompt)
    analysis = json.loads(self._clean_llm_response(response.content))
    
    return {
        "reasoning": analysis["reasoning"],
        "selected_tool": analysis.get("selected_tool"),
        "current_step": state["current_step"] + 1
    }
```

**1.3 실행 중간 도구 추가**
```python
def _should_continue(self, state: ReActLoopState) -> str:
    """계속할지 결정하는 조건부 함수"""
    current_step = state["current_step"]
    max_steps = state["max_steps"]
    
    # 최대 단계 수 도달
    if current_step >= max_steps:
        return "complete"
    
    # 마지막 관찰에서 답변이 완료되었다고 판단
    last_observation = state["execution_history"][-1] if state["execution_history"] else {}
    if last_observation.get("answer_complete", False):
        return "complete"
    
    # 신뢰도가 충분히 높음
    if last_observation.get("confidence", 0) > 0.8:
        return "complete"
    
    return "continue"  # 다시 추론 단계로
```

#### 📊 **성과 지표**
- **ReAct 구현도**: 75% → 90%
- **복잡한 쿼리 처리 능력**: 70% → 85%
- **동적 적응성**: 60% → 80%

### Phase 2: 멀티 에이전트 시스템 (2-3개월)

#### 🎯 **목표**: 전문 에이전트 기반 협력 시스템
- **현재 상태**: 단일 라우터 중심
- **목표 상태**: 전문 에이전트들의 협력 시스템

#### 📋 **구현 계획**

**2.1 전문 에이전트 설계**
```python
class MultiAgentReActRouter:
    def __init__(self):
        self.agents = {
            "planner": PlannerAgent(),      # 계획 수립 전문
            "executor": ExecutorAgent(),     # 도구 실행 전문
            "evaluator": EvaluatorAgent(),   # 결과 평가 전문
            "synthesizer": SynthesizerAgent() # 최종 통합 전문
        }
    
    async def _reason_step(self, state: ReActLoopState) -> Dict:
        """계획 에이전트가 다음 행동 계획"""
        planner_response = await self.agents["planner"].plan(
            query=state["query"],
            history=state["execution_history"],
            available_tools=list(self.tools.keys())
        )
        
        return {
            "reasoning": planner_response["plan"],
            "selected_tool": planner_response["selected_tool"],
            "tool_arguments": planner_response["arguments"]
        }
```

**2.2 에이전트 간 통신 프로토콜**
```python
class AgentCommunication:
    def __init__(self):
        self.message_bus = MessageBus()
        self.shared_memory = SharedMemory()
    
    async def broadcast_message(self, sender: str, message: Dict):
        """에이전트 간 메시지 브로드캐스트"""
        await self.message_bus.publish({
            "sender": sender,
            "timestamp": time.time(),
            "content": message
        })
```

#### 📊 **성과 지표**
- **시스템 복잡도 처리**: 75% → 90%
- **에이전트 협력 효율성**: 0% → 80%
- **전문성 향상**: 70% → 85%

### Phase 3: 학습 및 개선 시스템 (3-4개월)

#### 🎯 **목표**: 사용자 패턴 학습 및 자동 개선
- **현재 상태**: 정적 도구 선택 로직
- **목표 상태**: 학습 기반 동적 개선

#### 📋 **구현 계획**

**3.1 사용자 패턴 학습**
```python
class LearningSystem:
    def __init__(self):
        self.pattern_analyzer = PatternAnalyzer()
        self.feedback_collector = FeedbackCollector()
        self.model_updater = ModelUpdater()
    
    async def learn_from_interaction(self, interaction: Dict):
        """사용자 상호작용에서 학습"""
        # 패턴 분석
        patterns = await self.pattern_analyzer.analyze(interaction)
        
        # 피드백 수집
        feedback = await self.feedback_collector.collect(interaction)
        
        # 모델 업데이트
        await self.model_updater.update(patterns, feedback)
```

**3.2 자동 도구 선택 개선**
```python
class AdaptiveToolSelector:
    def __init__(self):
        self.historical_data = HistoricalDataStore()
        self.success_metrics = SuccessMetrics()
    
    async def select_tools(self, query: str, context: Dict) -> List[str]:
        """학습된 패턴 기반 도구 선택"""
        # 유사한 과거 쿼리 분석
        similar_queries = await self.historical_data.find_similar(query)
        
        # 성공률 기반 도구 추천
        recommended_tools = await self.success_metrics.recommend_tools(
            query, similar_queries
        )
        
        return recommended_tools
```

#### 📊 **성과 지표**
- **학습 능력**: 0% → 70%
- **사용자 만족도**: 80% → 90%
- **자동 개선률**: 0% → 60%

### Phase 4: 고급 기능 확장 (4-6개월)

#### 🎯 **목표**: 차세대 AI 시스템 기능
- **현재 상태**: 텍스트 기반 쿼리 처리
- **목표 상태**: 멀티모달, 실시간, 분산 처리

#### 📋 **구현 계획**

**4.1 멀티모달 처리**
```python
class MultiModalRouter:
    def __init__(self):
        self.text_processor = TextProcessor()
        self.image_processor = ImageProcessor()
        self.audio_processor = AudioProcessor()
    
    async def process_multimodal_query(self, query: MultiModalQuery):
        """텍스트, 이미지, 음성을 통합 처리"""
        if query.has_text():
            text_analysis = await self.text_processor.analyze(query.text)
        
        if query.has_image():
            image_analysis = await self.image_processor.analyze(query.image)
        
        if query.has_audio():
            audio_analysis = await self.audio_processor.analyze(query.audio)
        
        # 통합 분석 및 도구 선택
        return await self._integrate_multimodal_analysis(
            text_analysis, image_analysis, audio_analysis
        )
```

**4.2 실시간 스트리밍**
```python
class StreamingReActRouter:
    def __init__(self):
        self.stream_manager = StreamManager()
        self.progressive_processor = ProgressiveProcessor()
    
    async def process_streaming_query(self, query_stream: AsyncIterator):
        """실시간 스트리밍 쿼리 처리"""
        async for query_chunk in query_stream:
            # 점진적 처리
            partial_result = await self.progressive_processor.process_chunk(
                query_chunk
            )
            
            # 실시간 결과 스트리밍
            await self.stream_manager.stream_result(partial_result)
```

**4.3 분산 처리**
```python
class DistributedReActRouter:
    def __init__(self):
        self.cluster_manager = ClusterManager()
        self.load_balancer = LoadBalancer()
        self.consensus_manager = ConsensusManager()
    
    async def process_distributed_query(self, query: str):
        """분산 환경에서 쿼리 처리"""
        # 클러스터 노드 선택
        selected_nodes = await self.cluster_manager.select_nodes(query)
        
        # 부하 분산
        tasks = await self.load_balancer.distribute_tasks(
            query, selected_nodes
        )
        
        # 결과 합의
        final_result = await self.consensus_manager.reach_consensus(tasks)
        
        return final_result
```

#### 📊 **성과 지표**
- **멀티모달 처리**: 0% → 80%
- **실시간 성능**: 70% → 95%
- **분산 확장성**: 60% → 90%
