/**
 * Smart-RAG Chat - 메인 JavaScript
 *
 * 전역 기능들을 관리합니다.
 */

// 전역 변수
let currentTheme = localStorage.getItem('theme') || 'light';
let isConnected = false;

// DOM 로드 완료 시 실행
document.addEventListener('DOMContentLoaded', function () {
    initializeApp();
});

/**
 * 애플리케이션 초기화
 */
function initializeApp() {
    console.log('🚀 Smart-RAG Chat 초기화 시작');

    // 테마 설정
    setTheme(currentTheme);

    // 이벤트 리스너 등록
    setupEventListeners();

    // 연결 상태 확인
    checkConnectionStatus();

    // 설정 로드
    loadSettings();

    console.log('✅ Smart-RAG Chat 초기화 완료');
}

/**
 * 이벤트 리스너 설정
 */
function setupEventListeners() {
    // 테마 토글
    const themeToggle = document.getElementById('theme-toggle');
    if (themeToggle) {
        themeToggle.addEventListener('click', toggleTheme);
    }

    // 설정 모달
    const settingsBtn = document.getElementById('settings-btn');
    const settingsModal = document.getElementById('settings-modal');
    const closeSettings = document.getElementById('close-settings');
    const cancelSettings = document.getElementById('cancel-settings');
    const saveSettings = document.getElementById('save-settings');

    if (settingsBtn && settingsModal) {
        settingsBtn.addEventListener('click', () => openModal('settings-modal'));
    }

    if (closeSettings) {
        closeSettings.addEventListener('click', () => closeModal('settings-modal'));
    }

    if (cancelSettings) {
        cancelSettings.addEventListener('click', () => closeModal('settings-modal'));
    }

    if (saveSettings) {
        saveSettings.addEventListener('click', saveSettingsHandler);
    }

    // 모달 배경 클릭으로 닫기
    document.addEventListener('click', function (e) {
        if (e.target.classList.contains('modal')) {
            closeModal(e.target.id);
        }
    });

    // ESC 키로 모달 닫기
    document.addEventListener('keydown', function (e) {
        if (e.key === 'Escape') {
            closeAllModals();
        }
    });
}

/**
 * 테마 토글
 */
function toggleTheme() {
    currentTheme = currentTheme === 'light' ? 'dark' : 'light';
    setTheme(currentTheme);
    localStorage.setItem('theme', currentTheme);
}

/**
 * 테마 설정
 */
function setTheme(theme) {
    document.documentElement.setAttribute('data-theme', theme);

    const themeIcon = document.querySelector('.theme-icon');
    if (themeIcon) {
        themeIcon.textContent = theme === 'light' ? '🌙' : '☀️';
    }

    // 설정 모달의 테마 선택 옵션 업데이트
    const themeSelect = document.getElementById('theme-select');
    if (themeSelect) {
        themeSelect.value = theme;
    }
}

/**
 * 모달 열기
 */
function openModal(modalId) {
    const modal = document.getElementById(modalId);
    if (modal) {
        modal.classList.remove('hidden');
        document.body.style.overflow = 'hidden';
    }
}

/**
 * 모달 닫기
 */
function closeModal(modalId) {
    const modal = document.getElementById(modalId);
    if (modal) {
        modal.classList.add('hidden');
        document.body.style.overflow = '';
    }
}

/**
 * 모든 모달 닫기
 */
function closeAllModals() {
    const modals = document.querySelectorAll('.modal');
    modals.forEach(modal => {
        modal.classList.add('hidden');
    });
    document.body.style.overflow = '';
}

/**
 * 연결 상태 확인
 */
async function checkConnectionStatus() {
    try {
        const response = await fetch('/health');
        const data = await response.json();

        isConnected = data.status === 'healthy';
        updateConnectionStatus(isConnected);

        if (isConnected) {
            console.log('✅ 서버 연결 성공');
        } else {
            console.warn('⚠️ 서버 연결 실패');
        }
    } catch (error) {
        console.error('❌ 연결 상태 확인 실패:', error);
        isConnected = false;
        updateConnectionStatus(false);
    }
}

/**
 * 연결 상태 UI 업데이트
 */
function updateConnectionStatus(connected) {
    const statusIndicator = document.getElementById('connection-status');
    const statusDot = document.querySelector('.status-dot');

    if (statusIndicator && statusDot) {
        if (connected) {
            statusIndicator.textContent = '연결됨';
            statusDot.classList.remove('disconnected');
        } else {
            statusIndicator.textContent = '연결 끊김';
            statusDot.classList.add('disconnected');
        }
    }
}

/**
 * 토스트 메시지 표시
 */
function showToast(message, type = 'info') {
    // 기존 토스트 제거
    const existingToast = document.querySelector('.toast');
    if (existingToast) {
        existingToast.remove();
    }

    // 새 토스트 생성
    const toast = document.createElement('div');
    toast.className = `toast toast-${type}`;
    toast.textContent = message;

    // 스타일 적용
    Object.assign(toast.style, {
        position: 'fixed',
        top: '20px',
        right: '20px',
        padding: '12px 16px',
        borderRadius: '8px',
        color: 'white',
        fontWeight: '500',
        zIndex: '3000',
        transform: 'translateX(100%)',
        transition: 'transform 0.3s ease',
        backgroundColor: type === 'error' ? '#dc3545' : type === 'success' ? '#28a745' : '#007bff'
    });

    document.body.appendChild(toast);

    // 애니메이션
    setTimeout(() => {
        toast.style.transform = 'translateX(0)';
    }, 100);

    // 자동 제거
    setTimeout(() => {
        toast.style.transform = 'translateX(100%)';
        setTimeout(() => {
            if (toast.parentNode) {
                toast.parentNode.removeChild(toast);
            }
        }, 300);
    }, 3000);
}

/**
 * 로딩 상태 표시
 */
function showLoading(show = true) {
    const loadingOverlay = document.getElementById('loading-overlay');
    if (loadingOverlay) {
        if (show) {
            loadingOverlay.classList.remove('hidden');
        } else {
            loadingOverlay.classList.add('hidden');
        }
    }
}

/**
 * 유틸리티 함수들
 */
const utils = {
    /**
     * 문자열을 안전하게 이스케이프
     */
    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    },

    /**
     * 현재 시간을 포맷팅
     */
    formatTime(date = new Date()) {
        return date.toLocaleTimeString('ko-KR', {
            hour: '2-digit',
            minute: '2-digit'
        });
    },

    /**
     * 텍스트 길이 제한
     */
    truncateText(text, maxLength = 100) {
        if (text.length <= maxLength) return text;
        return text.substring(0, maxLength) + '...';
    },

    /**
     * 디바운스 함수
     */
    debounce(func, wait) {
        let timeout;
        return function executedFunction(...args) {
            const later = () => {
                clearTimeout(timeout);
                func(...args);
            };
            clearTimeout(timeout);
            timeout = setTimeout(later, wait);
        };
    },

    /**
     * 쿠키 설정
     */
    setCookie(name, value, days = 7) {
        const expires = new Date();
        expires.setTime(expires.getTime() + (days * 24 * 60 * 60 * 1000));
        document.cookie = `${name}=${value};expires=${expires.toUTCString()};path=/`;
    },

    /**
     * 쿠키 가져오기
     */
    getCookie(name) {
        const nameEQ = name + "=";
        const ca = document.cookie.split(';');
        for (let i = 0; i < ca.length; i++) {
            let c = ca[i];
            while (c.charAt(0) === ' ') c = c.substring(1, c.length);
            if (c.indexOf(nameEQ) === 0) return c.substring(nameEQ.length, c.length);
        }
        return null;
    }
};

/**
 * 설정 저장 핸들러
 */
function saveSettingsHandler() {
    const themeSelect = document.getElementById('theme-select');
    const chatModeSelect = document.getElementById('chat-mode');

    if (themeSelect) {
        const selectedTheme = themeSelect.value;
        setTheme(selectedTheme);
        localStorage.setItem('theme', selectedTheme);
    }

    if (chatModeSelect) {
        const selectedMode = chatModeSelect.value;
        localStorage.setItem('chat_mode', selectedMode);
    }

    closeModal('settings-modal');
    showToast('설정이 저장되었습니다.', 'success');
}

/**
 * 설정 로드
 */
function loadSettings() {
    const savedTheme = localStorage.getItem('theme') || 'light';
    const savedChatMode = localStorage.getItem('chat_mode') || 'parallel';

    setTheme(savedTheme);

    const themeSelect = document.getElementById('theme-select');
    const chatModeSelect = document.getElementById('chat-mode');

    if (themeSelect) {
        themeSelect.value = savedTheme;
    }

    if (chatModeSelect) {
        chatModeSelect.value = savedChatMode;
    }
}

// 전역 객체에 유틸리티 추가
window.SmartRAG = {
    utils,
    showToast,
    showLoading,
    checkConnectionStatus,
    loadSettings
};
