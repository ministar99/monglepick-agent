"""
의도 분류 체인 단위 테스트 (Task 5).

테스트 대상:
- Mock LLM: recommend/0.95 → 그대로 반환
- Mock LLM: recommend/0.4 → general로 보정
- Mock LLM: 에러 → fallback general 반환
- 6가지 intent 모두 유효한지 확인
"""

from __future__ import annotations

import pytest

from monglepick.agents.chat.models import IntentResult
from monglepick.chains.intent_chain import classify_intent


@pytest.mark.asyncio
async def test_intent_recommend_high_confidence(mock_ollama):
    """recommend/0.95 → 그대로 반환."""
    mock_ollama.set_structured_response(
        IntentResult(intent="recommend", confidence=0.95),
    )
    result = await classify_intent("영화 추천해줘")
    assert result.intent == "recommend"
    assert result.confidence == 0.95


@pytest.mark.asyncio
async def test_intent_low_confidence_corrected_to_general(mock_ollama):
    """recommend/0.4 → confidence < 0.6이므로 general로 보정."""
    mock_ollama.set_structured_response(
        IntentResult(intent="recommend", confidence=0.4),
    )
    result = await classify_intent("음...")
    assert result.intent == "general"
    assert result.confidence == 0.4


@pytest.mark.asyncio
async def test_intent_error_returns_fallback(mock_ollama):
    """LLM 에러 → IntentResult(intent='general', confidence=0.0)."""
    mock_ollama.set_error(RuntimeError("LLM connection failed"))
    result = await classify_intent("영화 추천해줘")
    assert result.intent == "general"
    assert result.confidence == 0.0


@pytest.mark.asyncio
async def test_intent_search(mock_ollama):
    """search intent가 정상 반환된다."""
    mock_ollama.set_structured_response(
        IntentResult(intent="search", confidence=0.85),
    )
    result = await classify_intent("인터스텔라 검색해줘")
    assert result.intent == "search"


@pytest.mark.asyncio
async def test_intent_info(mock_ollama):
    """info intent가 정상 반환된다."""
    mock_ollama.set_structured_response(
        IntentResult(intent="info", confidence=0.9),
    )
    result = await classify_intent("인셉션 출연진 알려줘")
    assert result.intent == "info"


@pytest.mark.asyncio
async def test_intent_theater(mock_ollama):
    """theater intent가 정상 반환된다."""
    mock_ollama.set_structured_response(
        IntentResult(intent="theater", confidence=0.8),
    )
    result = await classify_intent("근처 영화관 알려줘")
    assert result.intent == "theater"


@pytest.mark.asyncio
async def test_intent_booking(mock_ollama):
    """booking intent가 정상 반환된다."""
    mock_ollama.set_structured_response(
        IntentResult(intent="booking", confidence=0.75),
    )
    result = await classify_intent("CGV 예매하고 싶어")
    assert result.intent == "booking"


@pytest.mark.asyncio
async def test_intent_general(mock_ollama):
    """general intent가 정상 반환된다."""
    mock_ollama.set_structured_response(
        IntentResult(intent="general", confidence=0.9),
    )
    result = await classify_intent("안녕하세요")
    assert result.intent == "general"
