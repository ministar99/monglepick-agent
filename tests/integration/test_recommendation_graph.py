"""
추천 엔진 서브그래프 통합 테스트 (Phase 4).

서브그래프의 end-to-end 흐름을 테스트한다:
- Cold Start 경로: cold_start_checker → popularity_fallback → diversity_reranker → score_finalizer
- 정상 경로: cold_start_checker → collaborative_filter → content_based_filter → hybrid_merger → diversity_reranker → score_finalizer
- Chat Agent 통합: recommendation_ranker가 서브그래프를 호출하고, 에러 시 fallback 동작
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from monglepick.agents.chat.models import (
    CandidateMovie,
    ChatAgentState,
    EmotionResult,
    ExtractedPreferences,
    RankedMovie,
)
from monglepick.agents.recommendation.graph import run_recommendation_engine


# ============================================================
# 테스트 헬퍼
# ============================================================

def _make_candidate(
    id: str = "1",
    title: str = "테스트 영화",
    genres: list[str] | None = None,
    director: str = "",
    cast: list[str] | None = None,
    rating: float = 7.0,
    mood_tags: list[str] | None = None,
    rrf_score: float = 0.5,
) -> CandidateMovie:
    """테스트용 CandidateMovie를 생성한다."""
    return CandidateMovie(
        id=id,
        title=title,
        genres=genres or ["드라마"],
        director=director,
        cast=cast or [],
        rating=rating,
        mood_tags=mood_tags or [],
        rrf_score=rrf_score,
    )


def _make_diverse_candidates() -> list[CandidateMovie]:
    """다양한 장르의 후보 영화 6편을 생성한다."""
    return [
        _make_candidate(id="1", title="인터스텔라", genres=["SF", "드라마"], rating=8.7, rrf_score=0.95),
        _make_candidate(id="2", title="기생충", genres=["드라마", "스릴러"], rating=8.5, rrf_score=0.90),
        _make_candidate(id="3", title="어벤져스", genres=["액션", "SF"], rating=8.0, rrf_score=0.85),
        _make_candidate(id="4", title="라라랜드", genres=["로맨스"], rating=7.9, rrf_score=0.80),
        _make_candidate(id="5", title="겟아웃", genres=["공포", "스릴러"], rating=7.5, rrf_score=0.75),
        _make_candidate(id="6", title="코코", genres=["애니메이션"], rating=8.3, rrf_score=0.70),
    ]


# ============================================================
# 서브그래프 통합 테스트
# ============================================================

class TestRecommendationGraphColdStart:
    """Cold Start 경로 통합 테스트."""

    @pytest.mark.asyncio
    async def test_cold_start_returns_ranked_movies(self, mock_redis_cf):
        """Cold Start 유저에게 인기도 기반 추천을 반환한다."""
        candidates = _make_diverse_candidates()

        result = await run_recommendation_engine(
            candidate_movies=candidates,
            user_id="",
            user_profile={},
            watch_history=[],  # 시청 이력 0편 → Cold Start
            emotion=None,
            preferences=ExtractedPreferences(genre_preference="SF"),
        )

        # RankedMovie 리스트 반환
        assert len(result) > 0
        assert len(result) <= 5
        assert all(isinstance(m, RankedMovie) for m in result)

        # rank가 순서대로 부여됨
        assert [m.rank for m in result] == list(range(1, len(result) + 1))

        # Cold Start: CF 점수 0.0
        for m in result:
            assert m.score_detail.cf_score == 0.0

    @pytest.mark.asyncio
    async def test_cold_start_with_genre_preference(self, mock_redis_cf):
        """Cold Start이지만 장르 선호가 있으면 해당 장르에 부스트를 준다."""
        candidates = [
            _make_candidate(id="1", title="SF영화", genres=["SF"], rating=7.0, rrf_score=0.5),
            _make_candidate(id="2", title="로맨스영화", genres=["로맨스"], rating=7.0, rrf_score=0.5),
        ]

        result = await run_recommendation_engine(
            candidate_movies=candidates,
            user_id="",
            user_profile={},
            watch_history=[],
            emotion=None,
            preferences=ExtractedPreferences(genre_preference="SF"),
        )

        assert len(result) == 2
        # SF 장르 부스트로 1순위가 SF영화
        assert result[0].title == "SF영화"


class TestRecommendationGraphNormal:
    """정상 경로 (CF+CBF) 통합 테스트."""

    @pytest.mark.asyncio
    async def test_normal_path_returns_ranked_movies(self, mock_redis_cf):
        """정상 유저에게 CF+CBF 하이브리드 추천을 반환한다."""
        # 유사 유저 + 평점 설정
        mock_redis_cf.set_similar_users("user1", [("user2", 0.9)])
        mock_redis_cf.set_user_ratings("user2", {"1": "5.0", "2": "4.0"})

        candidates = _make_diverse_candidates()
        watch_history = [
            {"movie_id": str(i), "title": f"영화{i}", "genres": ["SF"]}
            for i in range(10)
        ]

        result = await run_recommendation_engine(
            candidate_movies=candidates,
            user_id="user1",
            user_profile={},
            watch_history=watch_history,  # 10편 → Warm Start
            emotion=EmotionResult(emotion="excited", mood_tags=["웅장"]),
            preferences=ExtractedPreferences(genre_preference="SF"),
        )

        assert len(result) > 0
        assert len(result) <= 5
        assert all(isinstance(m, RankedMovie) for m in result)

        # ScoreDetail이 채워져 있음
        for m in result:
            assert m.score_detail.hybrid_score >= 0.0

    @pytest.mark.asyncio
    async def test_diversity_in_results(self, mock_redis_cf):
        """MMR이 장르 다양성을 보장한다."""
        # CF 캐시 미스 (CBF에 의존)
        # RRF 점수가 비슷하게 설정하여 다양성이 발동하도록 함
        candidates = [
            _make_candidate(id="1", genres=["SF"], rating=9.0, rrf_score=0.90),
            _make_candidate(id="2", genres=["SF"], rating=8.5, rrf_score=0.89),
            _make_candidate(id="3", genres=["로맨스"], rating=8.0, rrf_score=0.88),
        ]

        # 시청 이력에 장르 포함 (CBF 장르 매칭용)
        watch_history = [
            {"movie_id": str(i), "genres": ["SF", "로맨스"]}
            for i in range(10)
        ]

        result = await run_recommendation_engine(
            candidate_movies=candidates,
            user_id="user1",
            user_profile={},
            watch_history=watch_history,
            emotion=None,
            preferences=None,
        )

        assert len(result) == 3
        # 결과에 SF와 로맨스가 모두 포함되어 있으면 다양성 보장
        all_genres = set()
        for m in result:
            all_genres.update(m.genres)
        assert "SF" in all_genres
        assert "로맨스" in all_genres


class TestChatAgentIntegration:
    """Chat Agent recommendation_ranker ↔ 서브그래프 통합 테스트."""

    @pytest.mark.asyncio
    async def test_recommendation_ranker_calls_subgraph(self, mock_redis_cf, mock_ollama):
        """recommendation_ranker가 서브그래프를 호출하여 ranked_movies를 반환한다."""
        from monglepick.agents.chat.nodes import recommendation_ranker

        candidates = _make_diverse_candidates()[:3]
        state: ChatAgentState = {
            "candidate_movies": candidates,
            "user_id": "",
            "user_profile": {},
            "watch_history": [],
            "emotion": None,
            "preferences": None,
        }

        result = await recommendation_ranker(state)

        ranked = result["ranked_movies"]
        assert len(ranked) > 0
        assert all(isinstance(m, RankedMovie) for m in ranked)

    @pytest.mark.asyncio
    async def test_recommendation_ranker_fallback_on_error(self, mock_ollama):
        """서브그래프 에러 시 RRF 기반 fallback으로 복원한다."""
        from monglepick.agents.chat.nodes import recommendation_ranker

        candidates = [
            _make_candidate(id="1", title="인터스텔라", rrf_score=0.9),
            _make_candidate(id="2", title="기생충", rrf_score=0.8),
        ]
        state: ChatAgentState = {
            "candidate_movies": candidates,
            "user_id": "user1",
            "user_profile": {},
            "watch_history": [],
            "emotion": None,
            "preferences": None,
        }

        # 서브그래프 실행을 에러로 만들기 (lazy import이므로 소스 모듈에서 패치)
        with patch(
            "monglepick.agents.recommendation.graph.run_recommendation_engine",
            side_effect=RuntimeError("Subgraph error"),
        ):
            result = await recommendation_ranker(state)

        ranked = result["ranked_movies"]
        assert len(ranked) == 2
        # RRF 점수 순서로 정렬
        assert ranked[0].title == "인터스텔라"
        assert ranked[1].title == "기생충"
        # fallback이므로 CF/CBF 점수는 0.0
        assert ranked[0].score_detail.cf_score == 0.0
