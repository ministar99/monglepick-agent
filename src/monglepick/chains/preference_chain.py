"""
선호 추출 체인 (§6-2 Node 4).

사용자 메시지에서 영화 선호 조건 7개 필드를 추출하고,
이전 선호 조건과 병합하는 체인.
EXAONE 32B 구조화 출력으로 ExtractedPreferences를 반환한다.

처리 흐름:
1. 이전 선호 조건을 문자열로 포맷하여 프롬프트에 포함
2. get_preference_llm() (EXAONE 32B, structured output) 호출
3. merge_preferences(previous, current) 로 병합
4. 에러 시: 이전 선호 조건 그대로 반환 (첫 턴이면 빈 ExtractedPreferences)
"""

from __future__ import annotations

import structlog
from langchain_core.prompts import ChatPromptTemplate

from monglepick.agents.chat.models import (
    ExtractedPreferences,
    merge_preferences,
)
from monglepick.llm.factory import get_preference_llm
from monglepick.prompts.preference import (
    PREFERENCE_HUMAN_PROMPT,
    PREFERENCE_SYSTEM_PROMPT,
)

logger = structlog.get_logger()


def _format_existing_preferences(prefs: ExtractedPreferences | None) -> str:
    """
    이전 선호 조건을 프롬프트용 문자열로 포맷한다.

    Args:
        prefs: 이전 턴까지 누적된 선호 조건 (None이면 없음)

    Returns:
        프롬프트에 삽입할 선호 조건 문자열
    """
    if prefs is None:
        return "(아직 파악된 선호 조건 없음)"

    parts = []
    if prefs.genre_preference:
        parts.append(f"- 장르: {prefs.genre_preference}")
    if prefs.mood:
        parts.append(f"- 분위기: {prefs.mood}")
    if prefs.viewing_context:
        parts.append(f"- 시청 상황: {prefs.viewing_context}")
    if prefs.platform:
        parts.append(f"- 플랫폼: {prefs.platform}")
    if prefs.reference_movies:
        parts.append(f"- 참조 영화: {', '.join(prefs.reference_movies)}")
    if prefs.era:
        parts.append(f"- 시대: {prefs.era}")
    if prefs.exclude:
        parts.append(f"- 제외: {prefs.exclude}")

    if not parts:
        return "(아직 파악된 선호 조건 없음)"

    return "\n".join(parts)


async def extract_preferences(
    current_input: str,
    previous_preferences: ExtractedPreferences | None = None,
) -> ExtractedPreferences:
    """
    사용자 메시지에서 선호 조건을 추출하고 이전 선호와 병합한다.

    Args:
        current_input: 현재 사용자 입력 텍스트
        previous_preferences: 이전 턴까지 누적된 선호 조건

    Returns:
        병합된 ExtractedPreferences
        - 에러 시: 이전 선호 조건 그대로 (첫 턴이면 빈 ExtractedPreferences)
    """
    # 프롬프트 구성
    prompt = ChatPromptTemplate.from_messages([
        ("system", PREFERENCE_SYSTEM_PROMPT),
        ("human", PREFERENCE_HUMAN_PROMPT),
    ])

    # 구조화 출력 LLM (EXAONE 32B, ExtractedPreferences 자동 파싱)
    llm = get_preference_llm()

    # 이전 선호 조건을 포맷
    existing_prefs_str = _format_existing_preferences(previous_preferences)

    # 입력 변수
    inputs = {
        "current_input": current_input,
        "existing_prefs": existing_prefs_str,
    }

    try:
        # 프롬프트 포맷 → LLM 호출 (명시적 2단계)
        prompt_value = await prompt.ainvoke(inputs)
        extracted: ExtractedPreferences = await llm.ainvoke(prompt_value)

        # 이전 선호와 병합
        merged = merge_preferences(previous_preferences, extracted)

        logger.info(
            "preferences_extracted",
            genre=merged.genre_preference,
            mood=merged.mood,
            reference_movies=merged.reference_movies,
            input_preview=current_input[:50],
        )
        return merged

    except Exception as e:
        logger.error(
            "preference_extraction_error",
            error=str(e),
            input_preview=current_input[:50],
        )
        # 에러 시: 이전 선호 유지 (첫 턴이면 빈 선호)
        return previous_preferences or ExtractedPreferences()
