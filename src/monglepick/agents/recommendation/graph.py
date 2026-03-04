"""
추천 엔진 서브그래프 (§7-2).

7노드 LangGraph StateGraph로 CF+CBF 하이브리드 추천, Cold Start 처리,
MMR 다양성 재정렬을 수행한다.

그래프 흐름:
    START → cold_start_checker
      → route_after_cold_start (조건부 분기)
          │
          ├─ is_cold_start=True  → popularity_fallback → diversity_reranker
          │                        → score_finalizer → END
          │
          └─ is_cold_start=False → collaborative_filter → content_based_filter
                                   → hybrid_merger → diversity_reranker
                                   → score_finalizer → END

공개 인터페이스:
- run_recommendation_engine(...) → list[RankedMovie]
"""

from __future__ import annotations

import structlog
from langgraph.graph import END, START, StateGraph

from monglepick.agents.chat.models import (
    CandidateMovie,
    EmotionResult,
    ExtractedPreferences,
    RankedMovie,
)
from monglepick.agents.recommendation.models import RecommendationEngineState
from monglepick.agents.recommendation.nodes import (
    cold_start_checker,
    collaborative_filter,
    content_based_filter,
    diversity_reranker,
    hybrid_merger,
    popularity_fallback,
    score_finalizer,
)

logger = structlog.get_logger()


# ============================================================
# 라우팅 함수
# ============================================================

def route_after_cold_start(state: RecommendationEngineState) -> str:
    """
    cold_start_checker 이후 분기 결정.

    - is_cold_start=True  → popularity_fallback (인기도 기반)
    - is_cold_start=False → collaborative_filter (CF+CBF 경로)

    Args:
        state: RecommendationEngineState (is_cold_start 필요)

    Returns:
        다음 노드 이름 문자열
    """
    if state.get("is_cold_start", False):
        return "popularity_fallback"
    return "collaborative_filter"


# ============================================================
# 그래프 빌드
# ============================================================

def build_recommendation_graph():
    """
    추천 엔진 StateGraph를 구성하고 컴파일한다.

    7개 노드와 1개 조건부 분기를 등록하여 CF+CBF 하이브리드 추천 흐름을 정의한다.

    Returns:
        컴파일된 StateGraph (CompiledGraph)
    """
    graph = StateGraph(RecommendationEngineState)

    # ── 노드 등록 (7개) ──
    graph.add_node("cold_start_checker", cold_start_checker)
    graph.add_node("collaborative_filter", collaborative_filter)
    graph.add_node("content_based_filter", content_based_filter)
    graph.add_node("hybrid_merger", hybrid_merger)
    graph.add_node("popularity_fallback", popularity_fallback)
    graph.add_node("diversity_reranker", diversity_reranker)
    graph.add_node("score_finalizer", score_finalizer)

    # ── 엣지 정의 ──

    # START → cold_start_checker
    graph.add_edge(START, "cold_start_checker")

    # cold_start_checker → 조건부 분기
    graph.add_conditional_edges(
        "cold_start_checker",
        route_after_cold_start,
        {
            "popularity_fallback": "popularity_fallback",
            "collaborative_filter": "collaborative_filter",
        },
    )

    # 정상 경로: CF → CBF → hybrid_merger → diversity_reranker
    graph.add_edge("collaborative_filter", "content_based_filter")
    graph.add_edge("content_based_filter", "hybrid_merger")
    graph.add_edge("hybrid_merger", "diversity_reranker")

    # Cold Start 경로: popularity_fallback → diversity_reranker
    graph.add_edge("popularity_fallback", "diversity_reranker")

    # 공통: diversity_reranker → score_finalizer → END
    graph.add_edge("diversity_reranker", "score_finalizer")
    graph.add_edge("score_finalizer", END)

    # 그래프 컴파일
    compiled = graph.compile()
    logger.info("recommendation_graph_compiled", node_count=7)
    return compiled


# ── 모듈 레벨 싱글턴: 컴파일 1회 ──
recommendation_engine = build_recommendation_graph()


# ============================================================
# 공개 인터페이스
# ============================================================

async def run_recommendation_engine(
    candidate_movies: list[CandidateMovie],
    user_id: str,
    user_profile: dict,
    watch_history: list[dict],
    emotion: EmotionResult | None,
    preferences: ExtractedPreferences | None,
) -> list[RankedMovie]:
    """
    추천 엔진 서브그래프를 실행하고 ranked_movies를 반환한다.

    Chat Agent의 recommendation_ranker 노드에서 호출한다.
    서브그래프 내부에서 CF+CBF 하이브리드 추천, Cold Start 처리,
    MMR 다양성 재정렬을 수행한다.

    Args:
        candidate_movies: rag_retriever 출력 후보 영화 목록
        user_id: 사용자 ID (빈 문자열이면 익명)
        user_profile: MySQL 유저 프로필
        watch_history: MySQL 시청 이력 (최근 50건)
        emotion: 감정 분석 결과 (None이면 감정 없음)
        preferences: 사용자 선호 조건 (None이면 빈 선호)

    Returns:
        list[RankedMovie]: 최종 순위 결과 (최대 5편)
    """
    # 무드 태그 추출
    mood_tags: list[str] = []
    if emotion and emotion.mood_tags:
        mood_tags = emotion.mood_tags

    # 서브그래프 초기 State 구성
    initial_state: RecommendationEngineState = {
        "candidate_movies": candidate_movies,
        "user_id": user_id,
        "user_profile": user_profile,
        "watch_history": watch_history,
        "emotion": emotion,
        "mood_tags": mood_tags,
        "preferences": preferences,
    }

    # 서브그래프 실행
    result = await recommendation_engine.ainvoke(initial_state)

    # 결과에서 ranked_movies 추출
    ranked_movies = result.get("ranked_movies", [])

    logger.info(
        "recommendation_engine_completed",
        input_count=len(candidate_movies),
        output_count=len(ranked_movies),
        is_cold_start=result.get("is_cold_start", False),
    )

    return ranked_movies
