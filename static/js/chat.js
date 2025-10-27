/**
 * Smart-RAG Chat - 채팅 기능 JavaScript
 *
 * 채팅 인터페이스를 관리합니다.
 */

// 전역 변수
let currentSessionId = null;
let isTyping = false;
let isProcessing = false;
let abortController = null;

// DOM 로드 완료 시 실행
document.addEventListener('DOMContentLoaded', function () {
    initializeChat();
});

/**
 * 채팅 초기화
 */
function initializeChat() {
    console.log('💬 채팅 초기화 시작');

    // 세션 ID 생성
    currentSessionId = generateSessionId();

    // 이벤트 리스너 설정
    setupChatEventListeners();

    console.log('✅ 채팅 초기화 완료');
}

/**
 * 채팅 이벤트 리스너 설정
 */
function setupChatEventListeners() {
    const chatInput = document.getElementById('chat-input');
    const sendButton = document.getElementById('send-button');
    const newChatButton = document.querySelector('.new-chat-button');

    // 메시지 입력 이벤트
    if (chatInput) {
        // Enter 키 이벤트 (Shift+Enter는 줄바꿈, Enter만 누르면 전송)
        chatInput.addEventListener('keydown', function (e) {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault(); // 기본 동작(줄바꿈) 방지
                sendMessage();
            }
        });

        // 입력창 자동 크기 조절
        chatInput.addEventListener('input', function () {
            autoResizeTextarea(this);
        });

        // 입력 중 상태 표시
        chatInput.addEventListener('input', function () {
            if (!isTyping && this.value.trim()) {
                isTyping = true;
                showTypingIndicator();
            } else if (isTyping && !this.value.trim()) {
                isTyping = false;
                hideTypingIndicator();
            }
        });

        // 입력창 포커스 시 자동 스크롤
        chatInput.addEventListener('focus', function () {
            const chatMessages = document.getElementById('chat-messages');
            if (chatMessages) {
                setTimeout(() => {
                    chatMessages.scrollTop = chatMessages.scrollHeight;
                }, 100);
            }
        });
    }

    // 전송 버튼 이벤트
    if (sendButton) {
        sendButton.addEventListener('click', sendMessage);
    }

    // Stop 버튼 이벤트
    const stopButton = document.getElementById('stop-button');
    if (stopButton) {
        stopButton.addEventListener('click', stopGeneration);
    }

    // 새 채팅 버튼 이벤트
    if (newChatButton) {
        newChatButton.addEventListener('click', startNewChat);
    }

    // PDF 업로드 버튼 이벤트
    const uploadPdfButton = document.getElementById('upload-pdf-button');
    const pdfFileInput = document.getElementById('pdf-file-input');

    if (uploadPdfButton && pdfFileInput) {
        // 업로드 버튼 클릭 시 파일 선택 다이얼로그 열기
        uploadPdfButton.addEventListener('click', function () {
            pdfFileInput.click();
        });

        // 파일 선택 시 업로드 실행
        pdfFileInput.addEventListener('change', function () {
            if (this.files && this.files[0]) {
                uploadPDF(this.files[0]);
            }
        });
    }

    // 채팅 옵션 버튼 이벤트
    const chatOptionsButtons = document.querySelectorAll('.chat-options-button');
    chatOptionsButtons.forEach(button => {
        button.addEventListener('click', function (e) {
            e.stopPropagation();
            showChatOptions(this);
        });
    });

    // 채팅 아이템 클릭 이벤트
    const chatItems = document.querySelectorAll('.chat-item');
    chatItems.forEach(item => {
        item.addEventListener('click', function () {
            // 활성 상태 변경
            chatItems.forEach(ci => ci.classList.remove('active'));
            this.classList.add('active');

            // 채팅 제목 업데이트
            const chatTitle = this.querySelector('.chat-title').textContent;
            const chatHeader = document.querySelector('.chat-header h3');
            if (chatHeader) {
                chatHeader.textContent = chatTitle;
            }
        });
    });
}


/**
 * 메시지 전송
 */
async function sendMessage() {
    const chatInput = document.getElementById('chat-input');
    const message = chatInput.value.trim();

    if (!message || isProcessing) {
        return;
    }

    // 사용자 메시지를 채팅에 추가
    addMessageToChat('user', message);

    // 입력창 초기화
    chatInput.value = '';
    autoResizeTextarea(chatInput);
    hideTypingIndicator();

    // 저장된 채팅 모드 가져오기
    const savedMode = localStorage.getItem('chat_mode') || 'parallel';

    // AbortController 생성 (중단 기능)
    abortController = new AbortController();

    // 처리 상태 및 UI 업데이트
    isProcessing = true;
    toggleStopButton(true);
    showAITypingIndicator();

    try {
        const response = await fetch('/api/chat/query', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                message: message,
                mode: savedMode,
                session_id: currentSessionId
            }),
            signal: abortController.signal
        });

        const result = await response.json();
        console.log('📤 메시지 전송 완료:', result);

        if (result.success) {
            // AI 응답을 채팅에 추가
            addMessageToChat('assistant', result.message, {
                processing_time: result.processing_time,
                selected_tools: result.metadata?.selected_tools || [],
                complexity_score: result.metadata?.complexity_score || 0
            });
        } else {
            // 에러 메시지 표시
            addMessageToChat('assistant', result.message || '처리 중 오류가 발생했습니다.', {
                isError: true,
                processing_time: result.processing_time
            });
        }

    } catch (error) {
        if (error.name === 'AbortError') {
            console.log('🛑 사용자가 생성을 중단했습니다.');
            addMessageToChat('assistant', '⚠️ 답변 생성이 중단되었습니다.', {
                isError: false
            });
        } else {
            console.error('메시지 전송 실패:', error);
            addMessageToChat('assistant', '서버와의 통신에 실패했습니다. 다시 시도해주세요.', {
                isError: true
            });
        }
    } finally {
        // 처리 상태 해제
        isProcessing = false;
        toggleStopButton(false);
        hideAITypingIndicator();
        abortController = null;
    }
}

/**
 * 채팅에 메시지 추가
 */
function addMessageToChat(role, content, metadata = {}) {
    const chatMessages = document.getElementById('chat-messages');
    if (!chatMessages) return;

    const messageDiv = document.createElement('div');
    messageDiv.className = `message-bubble ${role}`;

    const messageContent = document.createElement('div');
    messageContent.className = 'message-content';

    // 🆕 Markdown → HTML 변환 (AI 응답만)
    if (role === 'assistant') {
        // 라이브러리 로드 상태 확인
        console.log('🔍 Markdown 라이브러리 체크:', {
            marked: typeof marked,
            DOMPurify: typeof DOMPurify,
            role: role
        });

        if (typeof marked !== 'undefined' && typeof DOMPurify !== 'undefined') {
            try {
                // Marked.js로 Markdown → HTML 변환
                const rawHtml = marked.parse(content);
                // DOMPurify로 XSS 방지 (보안)
                const cleanHtml = DOMPurify.sanitize(rawHtml);
                messageContent.innerHTML = cleanHtml;
                console.log('✅ Markdown 렌더링 성공!');
            } catch (error) {
                console.error('❌ Markdown 렌더링 실패:', error);
                messageContent.textContent = content;  // Fallback
            }
        } else {
            console.warn('⚠️ Markdown 라이브러리 미로드! plain text로 표시합니다.');
            messageContent.textContent = content;
        }
    } else {
        // 사용자 메시지는 plain text
        messageContent.textContent = content;
    }

    const messageInfo = document.createElement('div');
    messageInfo.className = 'message-info';

    const messageTime = document.createElement('span');
    messageTime.className = 'message-time';
    messageTime.textContent = getCurrentTime();

    messageInfo.appendChild(messageTime);

    // 메타데이터가 있으면 추가 정보 표시
    if (metadata && Object.keys(metadata).length > 0) {
        const metadataDiv = document.createElement('div');
        metadataDiv.className = 'message-metadata';
        metadataDiv.style.display = 'none'; // 기본적으로 숨김

        if (metadata.processing_time) {
            const processingTime = document.createElement('span');
            processingTime.className = 'processing-time';
            processingTime.textContent = `⏱️ ${metadata.processing_time.toFixed(2)}초`;
            metadataDiv.appendChild(processingTime);
        }

        if (metadata.selected_tools && metadata.selected_tools.length > 0) {
            const toolsUsed = document.createElement('span');
            toolsUsed.className = 'tools-used';

            // 도구 아이콘 매핑
            const toolIcons = {
                'weather': '🌤️',
                'stock_info': '📈',
                'calculator': '🔢',
                'web_search': '🔍',
                'knowledge_base': '📚',
                'reasoning': '🧠'
            };

            const toolsText = metadata.selected_tools.map(tool => {
                const icon = toolIcons[tool] || '🛠️';
                return `${icon} ${tool}`;
            }).join(', ');

            toolsUsed.textContent = `도구: ${toolsText}`;
            metadataDiv.appendChild(toolsUsed);
        }

        if (metadata.complexity_score !== undefined) {
            const complexity = document.createElement('span');
            complexity.className = 'complexity-score';
            complexity.textContent = `복잡도: ${metadata.complexity_score.toFixed(2)}`;
            metadataDiv.appendChild(complexity);
        }

        messageInfo.appendChild(metadataDiv);

        // 메시지 클릭 시 메타데이터 토글
        messageDiv.addEventListener('click', function () {
            metadataDiv.style.display = metadataDiv.style.display === 'none' ? 'block' : 'none';
        });
    }

    messageDiv.appendChild(messageContent);
    messageDiv.appendChild(messageInfo);

    chatMessages.appendChild(messageDiv);

    // 스크롤을 맨 아래로
    chatMessages.scrollTop = chatMessages.scrollHeight;

    // 에러 메시지인 경우 스타일 적용
    if (metadata.isError) {
        messageDiv.classList.add('error-message');
    }
}

/**
 * 시스템 메시지 표시
 */
function showSystemMessage(message) {
    const chatMessages = document.getElementById('chat-messages');
    if (!chatMessages) return;

    const systemDiv = document.createElement('div');
    systemDiv.className = 'system-message';
    systemDiv.textContent = message;

    chatMessages.appendChild(systemDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;

    // 3초 후 자동 제거
    setTimeout(() => {
        if (systemDiv.parentNode) {
            systemDiv.parentNode.removeChild(systemDiv);
        }
    }, 3000);
}

/**
 * 채팅 히스토리 로드
 */
function loadChatHistory(messages) {
    const chatMessages = document.getElementById('chat-messages');
    if (!chatMessages) return;

    // 기존 메시지 제거 (시스템 메시지 제외)
    const existingMessages = chatMessages.querySelectorAll('.message-bubble');
    existingMessages.forEach(msg => msg.remove());

    // 히스토리 메시지 추가
    messages.forEach(msg => {
        addMessageToChat(msg.role, msg.content, msg.metadata || {});
    });
}

/**
 * 새 채팅 시작
 */
function startNewChat() {

    // 새 세션 ID 생성
    currentSessionId = generateSessionId();

    // 채팅 히스토리 초기화
    const chatMessages = document.getElementById('chat-messages');
    if (chatMessages) {
        chatMessages.innerHTML = '';

        // 환영 메시지 다시 추가
        const welcomeMessage = document.createElement('div');
        welcomeMessage.className = 'message-bubble assistant';
        welcomeMessage.innerHTML = `
            <div class="message-content">안녕하세요! 저는 Smart-RAG AI 어시스턴트입니다. 무엇을 도와드릴까요? 날씨, 주식, 계산, 웹 검색, 지식베이스 검색 등 다양한 작업을 도와드릴 수 있습니다.</div>
            <div class="message-info">
                <span class="message-time">${getCurrentTime()}</span>
            </div>
        `;
        chatMessages.appendChild(welcomeMessage);
    }

    showSystemMessage('새로운 채팅을 시작합니다.');
}

/**
 * 연결 상태 업데이트
 */
function updateConnectionStatus(connected) {
    const statusIndicator = document.getElementById('connection-status');
    if (statusIndicator) {
        statusIndicator.className = `connection-status ${connected ? 'connected' : 'disconnected'}`;
        statusIndicator.textContent = connected ? '연결됨' : '연결 끊김';
    }
}

/**
 * 타이핑 인디케이터 표시
 */
function showTypingIndicator() {
    const chatMessages = document.getElementById('chat-messages');
    if (!chatMessages) return;

    const typingDiv = document.createElement('div');
    typingDiv.className = 'typing-indicator';
    typingDiv.id = 'typing-indicator';
    typingDiv.innerHTML = '<div class="typing-dots"><span></span><span></span><span></span></div>';

    chatMessages.appendChild(typingDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

/**
 * 타이핑 인디케이터 숨김
 */
function hideTypingIndicator() {
    const typingIndicator = document.getElementById('typing-indicator');
    if (typingIndicator) {
        typingIndicator.remove();
    }
}

/**
 * AI 타이핑 인디케이터 표시 (답변 생성 중)
 */
function showAITypingIndicator() {
    const chatMessages = document.getElementById('chat-messages');
    if (!chatMessages) return;

    // 기존 AI 타이핑 인디케이터가 있으면 제거
    hideAITypingIndicator();

    const typingDiv = document.createElement('div');
    typingDiv.className = 'message-bubble assistant';
    typingDiv.id = 'ai-typing-indicator';
    typingDiv.innerHTML = `
        <div class="message-content">
            <div class="typing-dots">
                <span></span><span></span><span></span>
            </div>
        </div>
    `;

    chatMessages.appendChild(typingDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

/**
 * AI 타이핑 인디케이터 숨김
 */
function hideAITypingIndicator() {
    const aiTypingIndicator = document.getElementById('ai-typing-indicator');
    if (aiTypingIndicator) {
        aiTypingIndicator.remove();
    }
}

/**
 * Stop 버튼 토글
 */
function toggleStopButton(show) {
    const stopButton = document.getElementById('stop-button');
    const sendButton = document.getElementById('send-button');
    const chatInput = document.getElementById('chat-input');

    if (stopButton && sendButton) {
        if (show) {
            // 처리 중: Stop 버튼 표시, Send 버튼 숨김, 입력창 비활성화
            stopButton.classList.remove('hidden');
            sendButton.classList.add('hidden');
            if (chatInput) {
                chatInput.disabled = true;
                chatInput.placeholder = '답변 생성 중...';
            }
        } else {
            // 대기 중: Send 버튼 표시, Stop 버튼 숨김, 입력창 활성화
            stopButton.classList.add('hidden');
            sendButton.classList.remove('hidden');
            if (chatInput) {
                chatInput.disabled = false;
                chatInput.placeholder = '메시지를 입력하세요...';
            }
        }
    }
}

/**
 * 답변 생성 중단
 */
function stopGeneration() {
    console.log('🛑 답변 생성 중단 요청');

    if (abortController) {
        abortController.abort();
        showSystemMessage('답변 생성이 중단되었습니다.');
    }
}

/**
 * 텍스트 영역 자동 크기 조절
 */
function autoResizeTextarea(textarea) {
    textarea.style.height = 'auto';
    textarea.style.height = Math.min(textarea.scrollHeight, 120) + 'px';
}

/**
 * 현재 시간 반환
 */
function getCurrentTime() {
    const now = new Date();
    return now.toLocaleTimeString('ko-KR', {
        hour: '2-digit',
        minute: '2-digit'
    });
}

/**
 * 세션 ID 생성
 */
function generateSessionId() {
    return 'session_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
}

/**
 * 로딩 오버레이 표시/숨김
 */
function showLoadingOverlay(show) {
    const overlay = document.getElementById('loading-overlay');
    if (overlay) {
        if (show) {
            overlay.classList.remove('hidden');
        } else {
            overlay.classList.add('hidden');
        }
    }
}

/**
 * 채팅 옵션 표시
 */
function showChatOptions(button) {
    // 간단한 컨텍스트 메뉴 구현
    const chatItem = button.closest('.chat-item');
    const chatTitle = chatItem.querySelector('.chat-title').textContent;

    if (confirm(`"${chatTitle}" 채팅을 삭제하시겠습니까?`)) {
        chatItem.remove();
        showSystemMessage('채팅이 삭제되었습니다.');
    }
}

/**
 * PDF 파일 업로드
 */
async function uploadPDF(file) {
    console.log('📄 PDF 업로드 시작:', file.name);

    // 파일 크기 확인 (10MB 제한)
    const maxSize = 10 * 1024 * 1024; // 10MB
    if (file.size > maxSize) {
        showSystemMessage('❌ 파일 크기는 10MB 이하로 제한됩니다.');
        return;
    }

    // 파일 타입 확인
    if (!file.name.endsWith('.pdf')) {
        showSystemMessage('❌ PDF 파일만 업로드 가능합니다.');
        return;
    }

    // 업로드 시작 메시지
    showSystemMessage(`📄 ${file.name} 업로드 중...`);
    showAITypingIndicator();

    try {
        // FormData 생성
        const formData = new FormData();
        formData.append('file', file);
        formData.append('add_to_chroma', 'true');

        // API 요청
        const response = await fetch('/api/chat/upload-pdf', {
            method: 'POST',
            body: formData
        });

        const result = await response.json();
        console.log('📤 PDF 업로드 완료:', result);

        if (result.success) {
            const data = result.data;

            // 성공 메시지를 채팅에 추가
            const successMessage = `✅ PDF 업로드 완료!\n\n` +
                `📄 파일명: ${data.filename}\n` +
                `📊 총 페이지: ${data.total_pages}페이지\n` +
                `📦 청크 수: ${data.total_chunks}개\n` +
                `💾 저장 위치: ${data.saved_to_chroma ? 'ChromaDB (영구)' : '메모리 (임시)'}\n` +
                `⏱️ 처리 시간: ${data.processing_time.toFixed(2)}초\n\n` +
                `이제 이 문서에 대해 질문할 수 있습니다!`;

            addMessageToChat('assistant', successMessage);
            showSystemMessage('✅ PDF 파일이 성공적으로 업로드되었습니다.');
        } else {
            showSystemMessage('❌ PDF 업로드에 실패했습니다.');
            addMessageToChat('assistant', `❌ 업로드 실패: ${result.message}`, { isError: true });
        }

    } catch (error) {
        console.error('PDF 업로드 실패:', error);
        showSystemMessage('❌ PDF 업로드 중 오류가 발생했습니다.');
        addMessageToChat('assistant', '서버와의 통신에 실패했습니다. 다시 시도해주세요.', { isError: true });
    } finally {
        // 타이핑 인디케이터 해제 및 파일 입력 초기화
        hideAITypingIndicator();
        document.getElementById('pdf-file-input').value = '';
    }
}
