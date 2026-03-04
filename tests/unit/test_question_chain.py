"""
후속 질문 생성 체인 단위 테스트 (Task 8).

테스트 대상:
- Mock LLM → 한국어 질문 반환
- _get_missing_fields: 전부 None → 7개 필드 가중치순
- _get_missing_fields: genre+mood 채워짐 → 5개 필드
- 에러 → DEFAULT_QUESTION 반환
"""

from __future__ import annotations

import pytest

from monglepick.agents.chat.models import ExtractedPreferences
from monglepick.chains.question_chain import (
    DEFAULT_QUESTIONS,
    _format_known_preferences,
    _get_missing_fields,
    generate_question,
)


@pytest.mark.asyncio
async def test_generate_question_returns_korean(mock_ollama):
    """Mock LLM → 한국어 질문 문자열을 반환한다."""
    mock_ollama.set_response("어떤 장르의 영화를 좋아하세요? 🎬")
    result = await generate_question(ExtractedPreferences())
    assert isinstance(result, str)
    assert len(result) > 0


@pytest.mark.asyncio
async def test_error_returns_default_question(mock_ollama):
    """LLM 에러 → 최고 가중치 부족 필드의 기본 질문."""
    mock_ollama.set_error(RuntimeError("LLM timeout"))
    result = await generate_question(ExtractedPreferences())
    # genre_preference가 최고 가중치(2.0)이므로 해당 기본 질문
    assert result == DEFAULT_QUESTIONS["genre_preference"]


@pytest.mark.asyncio
async def test_error_with_genre_filled(mock_ollama):
    """genre 채워진 상태에서 에러 → mood 기본 질문."""
    mock_ollama.set_error(RuntimeError("LLM timeout"))
    prefs = ExtractedPreferences(genre_preference="SF")
    result = await generate_question(prefs)
    # genre 채워짐 → mood가 최고 가중치
    assert result == DEFAULT_QUESTIONS["mood"]


class TestGetMissingFields:
    """_get_missing_fields 유틸 함수 테스트."""

    def test_all_none_returns_seven(self):
        """모든 필드가 None → 7개 부족 필드."""
        missing = _get_missing_fields(ExtractedPreferences())
        assert len(missing) == 7

    def test_sorted_by_weight_desc(self):
        """가중치 내림차순으로 정렬된다."""
        missing = _get_missing_fields(ExtractedPreferences())
        weights = [w for _, w in missing]
        assert weights == sorted(weights, reverse=True)

    def test_genre_mood_filled_returns_five(self):
        """genre+mood 채워짐 → 5개 부족 필드."""
        prefs = ExtractedPreferences(genre_preference="SF", mood="웅장한")
        missing = _get_missing_fields(prefs)
        assert len(missing) == 5
        # genre_preference와 mood는 빠져야 함
        field_names = [f for f, _ in missing]
        assert "genre_preference" not in field_names
        assert "mood" not in field_names

    def test_all_filled_returns_empty(self):
        """모든 필드 채워짐 → 빈 리스트."""
        prefs = ExtractedPreferences(
            genre_preference="SF",
            mood="웅장한",
            viewing_context="혼자",
            platform="넷플릭스",
            reference_movies=["인터스텔라"],
            era="2020년대",
            exclude="공포 제외",
        )
        missing = _get_missing_fields(prefs)
        assert len(missing) == 0


class TestFormatKnownPreferences:
    """_format_known_preferences 유틸 함수 테스트."""

    def test_empty_prefs(self):
        """빈 선호 → '(아직 없음)'."""
        result = _format_known_preferences(ExtractedPreferences())
        assert result == "(아직 없음)"

    def test_filled_prefs(self):
        """채워진 필드가 포맷된다."""
        prefs = ExtractedPreferences(genre_preference="SF", mood="웅장한")
        result = _format_known_preferences(prefs)
        assert "SF" in result
        assert "웅장한" in result
