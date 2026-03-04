"""
LLM 팩토리 — ChatOllama 인스턴스 생성 및 캐싱.

로컬 Ollama 서버의 LLM 모델을 용도별로 생성하는 팩토리 함수 모음.
동일 파라미터 조합에 대해 싱글턴 캐싱을 적용하여 중복 인스턴스 생성을 방지한다.

모델 라우팅:
- 구조화 출력 (intent, emotion): Qwen 14B (정확한 JSON 생성)
- 한국어 생성 (preference, conversation, question, explanation): EXAONE 32B
- 구조화 출력은 with_structured_output(schema)으로 Pydantic 모델 자동 검증

캐싱:
- (model, temperature, format) 튜플을 키로 모듈 레벨 dict 캐싱
- get_structured_llm은 schema별로 별도 캐싱 (.with_structured_output은 새 Runnable)
"""

from __future__ import annotations

from typing import Any

from langchain_core.runnables import Runnable
from langchain_ollama import ChatOllama
from pydantic import BaseModel

from monglepick.config import settings

# ============================================================
# 모듈 레벨 캐시: (model, temperature, format) → ChatOllama 인스턴스
# ============================================================

_llm_cache: dict[tuple[str, float, str | None], ChatOllama] = {}

# 구조화 출력 캐시: (model, temperature, schema_name) → Runnable
_structured_cache: dict[tuple[str, float, str], Runnable] = {}


def get_llm(
    model: str | None = None,
    temperature: float | None = None,
    format: str | None = None,
    num_predict: int | None = None,
) -> ChatOllama:
    """
    ChatOllama 인스턴스를 생성하거나 캐시에서 반환한다.

    동일 (model, temperature, format) 조합은 싱글턴으로 재사용된다.
    num_predict는 캐시 키에 포함되지 않으며, 호출 시마다 적용할 수 있다.

    Args:
        model: Ollama 모델명 (기본값: settings.CONVERSATION_MODEL)
        temperature: 생성 온도 (기본값: 0.5)
        format: 응답 형식 ("json" 또는 None)
        num_predict: 최대 생성 토큰 수 (None이면 모델 기본값)

    Returns:
        ChatOllama 인스턴스
    """
    # 기본값 설정
    model = model or settings.CONVERSATION_MODEL
    temperature = temperature if temperature is not None else 0.5

    # 캐시 키 생성
    cache_key = (model, temperature, format)

    if cache_key not in _llm_cache:
        # ChatOllama 인스턴스 생성
        kwargs: dict[str, Any] = {
            "model": model,
            "temperature": temperature,
            "base_url": settings.OLLAMA_BASE_URL,
        }
        if format is not None:
            kwargs["format"] = format
        if num_predict is not None:
            kwargs["num_predict"] = num_predict

        _llm_cache[cache_key] = ChatOllama(**kwargs)

    return _llm_cache[cache_key]


def get_structured_llm(
    schema: type[BaseModel],
    model: str | None = None,
    temperature: float | None = None,
) -> Runnable:
    """
    구조화 출력(Pydantic 모델 자동 검증) LLM을 반환한다.

    ChatOllama에 .with_structured_output(schema, method="json_schema")를 적용하여
    LLM 응답을 자동으로 Pydantic 모델로 파싱한다.

    Args:
        schema: 출력 Pydantic 모델 클래스
        model: Ollama 모델명 (기본값: settings.INTENT_MODEL)
        temperature: 생성 온도 (기본값: 0.1)

    Returns:
        구조화 출력이 적용된 Runnable
    """
    model = model or settings.INTENT_MODEL
    temperature = temperature if temperature is not None else 0.1

    # 구조화 출력 캐시 키
    cache_key = (model, temperature, schema.__name__)

    if cache_key not in _structured_cache:
        # 기본 ChatOllama 인스턴스 가져오기
        llm = get_llm(model=model, temperature=temperature, format="json")
        # 구조화 출력 적용
        _structured_cache[cache_key] = llm.with_structured_output(
            schema, method="json_schema",
        )

    return _structured_cache[cache_key]


# ============================================================
# 용도별 편의 함수
# ============================================================

def get_intent_llm() -> Runnable:
    """
    의도 분류용 구조화 출력 LLM (Qwen 14B, temp=0.1).

    IntentResult Pydantic 모델을 자동 파싱한다.
    """
    from monglepick.agents.chat.models import IntentResult

    return get_structured_llm(
        schema=IntentResult,
        model=settings.INTENT_MODEL,
        temperature=0.1,
    )


def get_emotion_llm() -> Runnable:
    """
    감정 분석용 구조화 출력 LLM (Qwen 14B, temp=0.1).

    EmotionResult Pydantic 모델을 자동 파싱한다.
    """
    from monglepick.agents.chat.models import EmotionResult

    return get_structured_llm(
        schema=EmotionResult,
        model=settings.EMOTION_MODEL,
        temperature=0.1,
    )


def get_preference_llm() -> Runnable:
    """
    선호 추출용 구조화 출력 LLM (EXAONE 32B, temp=0.3).

    ExtractedPreferences Pydantic 모델을 자동 파싱한다.
    """
    from monglepick.agents.chat.models import ExtractedPreferences

    return get_structured_llm(
        schema=ExtractedPreferences,
        model=settings.PREFERENCE_MODEL,
        temperature=0.3,
    )


def get_conversation_llm() -> ChatOllama:
    """
    일반 대화용 LLM (EXAONE 32B, temp=0.5).

    자유 텍스트 생성 — 구조화 출력 미적용.
    """
    return get_llm(
        model=settings.CONVERSATION_MODEL,
        temperature=0.5,
    )


def get_question_llm() -> ChatOllama:
    """
    후속 질문 생성용 LLM (EXAONE 32B, temp=0.5).

    자유 텍스트 생성 — 구조화 출력 미적용.
    """
    return get_llm(
        model=settings.QUESTION_MODEL,
        temperature=0.5,
    )


def get_explanation_llm() -> ChatOllama:
    """
    추천 이유 생성용 LLM (EXAONE 32B, temp=0.5).

    자유 텍스트 생성 — 구조화 출력 미적용.
    """
    return get_llm(
        model=settings.EXPLANATION_MODEL,
        temperature=0.5,
    )
