"""
Chat Agent 그래프 통합 테스트 (Phase 3).

전체 그래프 흐름을 테스트한다 (mock LLM + mock DB).
run_chat_agent_sync()로 동기 실행하여 최종 State를 검증한다.

테스트 시나리오:
1. 추천 흐름: recommend 의도 → 감정 → 선호 → 검색 → 순위 → 설명 → 응답
2. 일반 대화 흐름: general 의도 → general_responder → 응답
3. 후속 질문 흐름: 선호 부족 → question_generator → 질문 응답
4. 빈 검색 결과 → 빈 추천 + 기본 응답
5. tool_executor 흐름: info/theater/booking → 안내 메시지
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from monglepick.agents.chat.models import (
    EmotionResult,
    ExtractedPreferences,
    IntentResult,
)
from monglepick.rag.hybrid_search import SearchResult


# ============================================================
# 통합 테스트용 mock 래퍼
# ============================================================

def _make_all_mocks(
    intent: IntentResult | None = None,
    emotion: EmotionResult | None = None,
    preferences: ExtractedPreferences | None = None,
    search_results: list[SearchResult] | None = None,
    general_response: str = "안녕하세요!",
    question_response: str = "어떤 장르를 좋아하세요?",
    explanation_response: str = "좋은 영화입니다.",
):
    """
    모든 외부 의존성(LLM 체인, DB, 검색)을 mock하는 패치 목록을 반환한다.

    Returns:
        list of mock.patch context managers
    """
    if intent is None:
        intent = IntentResult(intent="recommend", confidence=0.9)
    if emotion is None:
        emotion = EmotionResult(emotion="happy", mood_tags=["유쾌"])
    if preferences is None:
        preferences = ExtractedPreferences(genre_preference="SF", mood="웅장한")
    if search_results is None:
        search_results = [
            SearchResult(
                movie_id="157336",
                title="인터스텔라",
                score=0.95,
                source="rrf",
                metadata={
                    "genres": ["SF", "드라마"],
                    "director": "놀란",
                    "rating": 8.7,
                    "release_year": 2014,
                    "overview": "우주 탐사 영화",
                },
            ),
        ]

    patches = [
        # MySQL mock (익명 사용자로 처리하기 위해 get_mysql을 에러로)
        patch(
            "monglepick.agents.chat.nodes.get_mysql",
            side_effect=Exception("mock: DB 미사용"),
        ),
        # LLM 체인 mock
        patch(
            "monglepick.agents.chat.nodes.classify_intent",
            new_callable=AsyncMock,
            return_value=intent,
        ),
        patch(
            "monglepick.agents.chat.nodes.analyze_emotion",
            new_callable=AsyncMock,
            return_value=emotion,
        ),
        patch(
            "monglepick.agents.chat.nodes.extract_preferences",
            new_callable=AsyncMock,
            return_value=preferences,
        ),
        patch(
            "monglepick.agents.chat.nodes.generate_question",
            new_callable=AsyncMock,
            return_value=question_response,
        ),
        patch(
            "monglepick.agents.chat.nodes.generate_explanations_batch",
            new_callable=AsyncMock,
            return_value=[explanation_response] * len(search_results),
        ),
        patch(
            "monglepick.agents.chat.nodes.generate_general_response",
            new_callable=AsyncMock,
            return_value=general_response,
        ),
        # 하이브리드 검색 mock
        patch(
            "monglepick.agents.chat.nodes.hybrid_search",
            new_callable=AsyncMock,
            return_value=search_results,
        ),
    ]
    return patches


# ============================================================
# 통합 테스트
# ============================================================

class TestChatGraphIntegration:
    """Chat Agent 그래프 통합 테스트."""

    @pytest.mark.asyncio
    async def test_recommend_flow(self):
        """추천 흐름: recommend → emotion → preference → search → rank → explain → format."""
        patches = _make_all_mocks(
            intent=IntentResult(intent="recommend", confidence=0.9),
            preferences=ExtractedPreferences(genre_preference="SF", mood="웅장한"),
        )

        from monglepick.agents.chat.graph import run_chat_agent_sync

        with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5], patches[6], patches[7]:
            state = await run_chat_agent_sync(
                user_id="",
                session_id="test",
                message="우울한데 영화 추천해줘",
            )

        # 최종 State 검증
        assert state.get("intent") is not None
        assert state["intent"].intent == "recommend"
        assert state.get("response")
        assert "인터스텔라" in state["response"]
        assert len(state.get("ranked_movies", [])) >= 1

    @pytest.mark.asyncio
    async def test_general_flow(self):
        """일반 대화 흐름: general → general_responder → format."""
        patches = _make_all_mocks(
            intent=IntentResult(intent="general", confidence=0.8),
            general_response="안녕하세요! 영화 추천 도우미 몽글이에요!",
        )

        from monglepick.agents.chat.graph import run_chat_agent_sync

        with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5], patches[6], patches[7]:
            state = await run_chat_agent_sync(
                user_id="",
                session_id="test",
                message="안녕",
            )

        assert state["intent"].intent == "general"
        assert "몽글" in state["response"]

    @pytest.mark.asyncio
    async def test_question_flow(self):
        """후속 질문 흐름: 선호 부족 → question_generator → 응답."""
        patches = _make_all_mocks(
            intent=IntentResult(intent="recommend", confidence=0.9),
            preferences=ExtractedPreferences(),  # 빈 선호 → 불충분
            question_response="어떤 장르의 영화를 좋아하세요?",
        )

        from monglepick.agents.chat.graph import run_chat_agent_sync

        with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5], patches[6], patches[7]:
            state = await run_chat_agent_sync(
                user_id="",
                session_id="test",
                message="영화 추천해줘",
            )

        assert state.get("needs_clarification") is True
        assert "장르" in state["response"]

    @pytest.mark.asyncio
    async def test_empty_search_results(self):
        """빈 검색 결과 → 기본 응답."""
        patches = _make_all_mocks(
            intent=IntentResult(intent="recommend", confidence=0.9),
            preferences=ExtractedPreferences(genre_preference="SF", mood="웅장한"),
            search_results=[],
            explanation_response="",
        )

        from monglepick.agents.chat.graph import run_chat_agent_sync

        with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5], patches[6], patches[7]:
            state = await run_chat_agent_sync(
                user_id="",
                session_id="test",
                message="엄청 특이한 영화 추천해줘",
            )

        # 빈 검색 결과 → ranked_movies 비어있음
        assert len(state.get("ranked_movies", [])) == 0
        assert state.get("response")  # 기본 응답은 존재

    @pytest.mark.asyncio
    async def test_tool_executor_flow(self):
        """도구 실행 흐름: info → tool_executor_node → format."""
        patches = _make_all_mocks(
            intent=IntentResult(intent="info", confidence=0.9),
        )

        from monglepick.agents.chat.graph import run_chat_agent_sync

        with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5], patches[6], patches[7]:
            state = await run_chat_agent_sync(
                user_id="",
                session_id="test",
                message="인터스텔라 상세 정보 알려줘",
            )

        assert state["intent"].intent == "info"
        assert "준비" in state["response"]
