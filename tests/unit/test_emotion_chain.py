"""
감정 분석 체인 단위 테스트 (Task 6).

테스트 대상:
- emotion="sad" → mood_tags에 매핑 테이블 값 포함
- emotion=None → LLM 추출 tags 그대로
- 잘못된 무드태그 → MOOD_WHITELIST로 필터링
- 에러 → 빈 결과 반환
"""

from __future__ import annotations

import pytest

from monglepick.agents.chat.models import EmotionResult
from monglepick.chains.emotion_chain import _validate_mood_tags, analyze_emotion
from monglepick.prompts.emotion import EMOTION_TO_MOOD_MAP


@pytest.mark.asyncio
async def test_emotion_sad_includes_mapped_tags(mock_ollama):
    """emotion='sad' → mood_tags에 매핑 테이블 값(힐링, 감동 등)이 포함된다."""
    mock_ollama.set_structured_response(
        EmotionResult(emotion="sad", mood_tags=["슬픔"]),
    )
    result = await analyze_emotion("오늘 너무 우울해")
    assert result.emotion == "sad"
    # 매핑 테이블에서 sad → [힐링, 감동, 따뜻, 잔잔, 카타르시스]
    for tag in EMOTION_TO_MOOD_MAP["sad"]:
        assert tag in result.mood_tags


@pytest.mark.asyncio
async def test_emotion_none_keeps_llm_tags(mock_ollama):
    """emotion=None → LLM이 추출한 mood_tags 그대로 유지."""
    mock_ollama.set_structured_response(
        EmotionResult(emotion=None, mood_tags=["스릴", "긴장감"]),
    )
    result = await analyze_emotion("무서운 영화 보고 싶어")
    assert result.emotion is None
    # 매핑 테이블 적용 없이 LLM 추출 값 유지
    assert "스릴" in result.mood_tags
    assert "긴장감" in result.mood_tags


@pytest.mark.asyncio
async def test_emotion_happy_maps_correctly(mock_ollama):
    """emotion='happy' → 유쾌, 모험, 따뜻, 로맨틱, 카타르시스가 포함된다."""
    mock_ollama.set_structured_response(
        EmotionResult(emotion="happy", mood_tags=["유쾌"]),
    )
    result = await analyze_emotion("오늘 기분 좋아!")
    assert result.emotion == "happy"
    for tag in EMOTION_TO_MOOD_MAP["happy"]:
        assert tag in result.mood_tags


@pytest.mark.asyncio
async def test_invalid_mood_tags_filtered(mock_ollama):
    """MOOD_WHITELIST에 없는 태그는 필터링된다."""
    mock_ollama.set_structured_response(
        EmotionResult(emotion=None, mood_tags=["유쾌", "존재하지않는태그", "힐링"]),
    )
    result = await analyze_emotion("재미있는 영화")
    assert "유쾌" in result.mood_tags
    assert "힐링" in result.mood_tags
    assert "존재하지않는태그" not in result.mood_tags


@pytest.mark.asyncio
async def test_emotion_error_returns_empty(mock_ollama):
    """LLM 에러 → EmotionResult(emotion=None, mood_tags=[])."""
    mock_ollama.set_error(RuntimeError("LLM timeout"))
    result = await analyze_emotion("우울한 날")
    assert result.emotion is None
    assert result.mood_tags == []


class TestValidateMoodTags:
    """_validate_mood_tags 유틸 함수 테스트."""

    def test_valid_tags_pass(self):
        """화이트리스트에 있는 태그는 통과한다."""
        result = _validate_mood_tags(["유쾌", "힐링", "감동"])
        assert result == ["유쾌", "힐링", "감동"]

    def test_invalid_tags_removed(self):
        """화이트리스트에 없는 태그는 제거된다."""
        result = _validate_mood_tags(["유쾌", "잘못된태그", "힐링"])
        assert result == ["유쾌", "힐링"]

    def test_empty_input(self):
        """빈 리스트 입력 → 빈 리스트 반환."""
        result = _validate_mood_tags([])
        assert result == []

    def test_all_invalid(self):
        """모두 무효한 태그 → 빈 리스트 반환."""
        result = _validate_mood_tags(["없는태그1", "없는태그2"])
        assert result == []
