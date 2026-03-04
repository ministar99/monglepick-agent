"""
Chat Agent 노드 함수 (§6-2 Node 1~11 + general_responder + tool_executor_node).

LangGraph StateGraph의 각 노드로 등록되는 13개 async 함수.
시그니처: async def node_name(state: ChatAgentState) -> dict

모든 노드는 try/except로 감싸고, 에러 시 유효한 기본값을 반환한다 (에러 전파 금지).
반환값은 dict — LangGraph 컨벤션 (TypedDict State 일부 업데이트).

노드 목록:
1. context_loader       — 유저 프로필/시청이력/대화이력 로드 (MySQL)
2. intent_classifier    — 의도 분류 (6가지 intent)
3. emotion_analyzer     — 감정 분석 + 무드 태그 매핑
4. preference_refiner   — 선호 추출 + 누적 병합 + 충분성 판정
5. question_generator   — 부족 정보 후속 질문 생성
6. query_builder        — RAG 검색 쿼리 구성 (규칙 기반, LLM 없음)
7. rag_retriever        — 하이브리드 검색 (Qdrant+ES+Neo4j RRF)
8. recommendation_ranker — 추천 순위 정렬 (Phase 4 스텁)
9. explanation_generator — 영화별 추천 이유 생성
10. response_formatter  — 응답 포맷팅 (추천/질문/일반/에러)
11. error_handler       — 에러 처리 + 친절한 안내 메시지
12. general_responder   — 일반 대화 응답 (몽글 페르소나)
13. tool_executor_node  — 도구 실행 (Phase 6 스텁)
"""

from __future__ import annotations

import re
from typing import Any

import aiomysql
import structlog
from langsmith import traceable

from monglepick.agents.chat.models import (
    CandidateMovie,
    ChatAgentState,
    EmotionResult,
    ExtractedPreferences,
    IntentResult,
    RankedMovie,
    ScoreDetail,
    SearchQuery,
    is_sufficient,
)
from monglepick.chains import (
    analyze_emotion,
    classify_intent,
    extract_preferences,
    generate_explanations_batch,
    generate_general_response,
    generate_question,
)
from monglepick.db.clients import get_mysql
from monglepick.rag.hybrid_search import SearchResult, hybrid_search

logger = structlog.get_logger()


# ============================================================
# 1. context_loader — 유저 프로필/시청이력/대화이력 로드
# ============================================================

@traceable(name="context_loader", run_type="chain", metadata={"node": "1/13", "db": "mysql"})
async def context_loader(state: ChatAgentState) -> dict:
    """
    MySQL에서 유저 프로필과 시청 이력을 로드하고, 메시지 리스트를 구성한다.

    - user_id가 비어있으면(익명 사용자) 빈 기본값을 반환한다.
    - messages에 현재 입력을 user 메시지로 추가한다.
    - turn_count는 기존 user 메시지 수 + 1로 계산한다.

    Args:
        state: ChatAgentState (user_id, session_id, current_input 필수)

    Returns:
        dict: user_profile, watch_history, messages, turn_count 업데이트
    """
    try:
        user_id = state.get("user_id", "")
        current_input = state.get("current_input", "")

        # 기존 메시지 복사 + 현재 입력 추가
        messages: list[dict[str, str]] = list(state.get("messages", []))
        messages.append({"role": "user", "content": current_input})

        # 턴 카운트: user 메시지 수
        turn_count = sum(1 for m in messages if m.get("role") == "user")

        # 익명 사용자: 빈 기본값
        if not user_id:
            logger.info("context_loader_anonymous")
            return {
                "user_profile": {},
                "watch_history": [],
                "messages": messages,
                "turn_count": turn_count,
            }

        # MySQL에서 유저 프로필 + 시청 이력 로드
        user_profile: dict[str, Any] = {}
        watch_history: list[dict[str, Any]] = []

        try:
            pool = await get_mysql()
            async with pool.acquire() as conn:
                async with conn.cursor(aiomysql.DictCursor) as cursor:
                    # 유저 프로필 조회
                    await cursor.execute(
                        "SELECT * FROM users WHERE user_id = %s LIMIT 1",
                        (user_id,),
                    )
                    row = await cursor.fetchone()
                    if row:
                        user_profile = dict(row)

                    # 시청 이력 조회 (최근 50건, 영화 제목 포함)
                    await cursor.execute(
                        """
                        SELECT wh.movie_id, m.title, wh.rating, wh.watched_at
                        FROM watch_history wh
                        LEFT JOIN movies m ON wh.movie_id = m.movie_id
                        WHERE wh.user_id = %s
                        ORDER BY wh.watched_at DESC
                        LIMIT 50
                        """,
                        (user_id,),
                    )
                    rows = await cursor.fetchall()
                    watch_history = [dict(r) for r in rows]
        except Exception as db_err:
            # DB 에러 시에도 빈 기본값으로 계속 진행
            logger.warning("context_loader_db_error", error=str(db_err))

        logger.info(
            "context_loaded",
            user_id=user_id,
            profile_exists=bool(user_profile),
            history_count=len(watch_history),
            turn_count=turn_count,
        )

        return {
            "user_profile": user_profile,
            "watch_history": watch_history,
            "messages": messages,
            "turn_count": turn_count,
        }

    except Exception as e:
        logger.error("context_loader_error", error=str(e))
        return {
            "user_profile": {},
            "watch_history": [],
            "messages": [{"role": "user", "content": state.get("current_input", "")}],
            "turn_count": 1,
            "error": str(e),
        }


# ============================================================
# 2. intent_classifier — 의도 분류
# ============================================================

@traceable(name="intent_classifier", run_type="chain", metadata={"node": "2/13", "llm": "qwen2.5:14b"})
async def intent_classifier(state: ChatAgentState) -> dict:
    """
    사용자 메시지의 의도를 6가지 중 하나로 분류한다.

    recent_messages: 최근 6개 메시지를 "role: content" 포맷으로 구성하여 맥락 제공.
    신뢰도 < 0.6이면 classify_intent 체인 내부에서 general로 보정.

    Args:
        state: ChatAgentState (current_input, messages 필요)

    Returns:
        dict: intent(IntentResult) 업데이트
    """
    try:
        current_input = state.get("current_input", "")
        messages = state.get("messages", [])

        # 최근 6개 메시지를 포맷 (현재 입력 제외)
        recent = messages[-7:-1] if len(messages) > 1 else []
        recent_messages = "\n".join(
            f"{m.get('role', 'user')}: {m.get('content', '')}"
            for m in recent[-6:]
        )

        result = await classify_intent(
            current_input=current_input,
            recent_messages=recent_messages,
        )

        logger.info(
            "intent_classified_node",
            intent=result.intent,
            confidence=result.confidence,
        )
        return {"intent": result}

    except Exception as e:
        logger.error("intent_classifier_error", error=str(e))
        return {"intent": IntentResult(intent="general", confidence=0.0)}


# ============================================================
# 3. emotion_analyzer — 감정 분석
# ============================================================

@traceable(name="emotion_analyzer", run_type="chain", metadata={"node": "3/13", "llm": "qwen2.5:14b"})
async def emotion_analyzer(state: ChatAgentState) -> dict:
    """
    사용자 메시지의 감정을 분석하고 무드 태그를 추출한다.

    analyze_emotion 체인이 감정→무드 매핑 + MOOD_WHITELIST 필터링을 수행한다.

    Args:
        state: ChatAgentState (current_input, messages 필요)

    Returns:
        dict: emotion(EmotionResult) 업데이트
    """
    try:
        current_input = state.get("current_input", "")
        messages = state.get("messages", [])

        # 최근 6개 메시지 포맷
        recent = messages[-7:-1] if len(messages) > 1 else []
        recent_messages = "\n".join(
            f"{m.get('role', 'user')}: {m.get('content', '')}"
            for m in recent[-6:]
        )

        result = await analyze_emotion(
            current_input=current_input,
            recent_messages=recent_messages,
        )

        logger.info(
            "emotion_analyzed_node",
            emotion=result.emotion,
            mood_tags=result.mood_tags,
        )
        return {"emotion": result}

    except Exception as e:
        logger.error("emotion_analyzer_error", error=str(e))
        return {"emotion": EmotionResult(emotion=None, mood_tags=[])}


# ============================================================
# 4. preference_refiner — 선호 추출 + 충분성 판정
# ============================================================

@traceable(name="preference_refiner", run_type="chain", metadata={"node": "4/13", "llm": "exaone-32b"})
async def preference_refiner(state: ChatAgentState) -> dict:
    """
    사용자 메시지에서 선호 조건을 추출하고, 이전 선호와 병합한 후 충분성을 판정한다.

    - extract_preferences 체인이 추출 + 병합을 수행한다.
    - is_sufficient()로 가중치 합산 ≥ 3.0 또는 turn_count ≥ 3 판정.
    - needs_clarification=True면 후속 질문 필요, False면 추천 진행.

    Args:
        state: ChatAgentState (current_input, preferences, emotion, turn_count 필요)

    Returns:
        dict: preferences(ExtractedPreferences), needs_clarification(bool) 업데이트
    """
    try:
        current_input = state.get("current_input", "")
        prev_prefs = state.get("preferences")
        emotion = state.get("emotion")
        turn_count = state.get("turn_count", 0)

        # 선호 추출 + 병합
        merged = await extract_preferences(
            current_input=current_input,
            previous_preferences=prev_prefs,
        )

        # 감정 존재 여부 확인 (무드 가중치 부여용)
        has_emotion = emotion is not None and emotion.emotion is not None

        # 충분성 판정
        sufficient = is_sufficient(
            prefs=merged,
            turn_count=turn_count,
            has_emotion=has_emotion,
        )

        logger.info(
            "preference_refined_node",
            needs_clarification=not sufficient,
            turn_count=turn_count,
            genre=merged.genre_preference,
            mood=merged.mood,
        )
        return {
            "preferences": merged,
            "needs_clarification": not sufficient,
        }

    except Exception as e:
        logger.error("preference_refiner_error", error=str(e))
        return {
            "preferences": state.get("preferences", ExtractedPreferences()),
            "needs_clarification": True,
        }


# ============================================================
# 5. question_generator — 후속 질문 생성
# ============================================================

@traceable(name="question_generator", run_type="chain", metadata={"node": "5/13", "llm": "exaone-32b"})
async def question_generator(state: ChatAgentState) -> dict:
    """
    부족한 선호 정보를 파악하기 위한 후속 질문을 생성한다.

    needs_clarification=True일 때만 호출된다.
    response 필드에도 질문 텍스트를 설정하여 response_formatter에서 바로 사용한다.

    Args:
        state: ChatAgentState (preferences, emotion, turn_count 필요)

    Returns:
        dict: follow_up_question, response 업데이트
    """
    try:
        prefs = state.get("preferences", ExtractedPreferences())
        emotion = state.get("emotion")
        turn_count = state.get("turn_count", 0)

        emotion_str = emotion.emotion if emotion else None

        question = await generate_question(
            extracted_preferences=prefs,
            emotion=emotion_str,
            turn_count=turn_count,
        )

        logger.info("question_generated_node", question_preview=question[:50])
        return {
            "follow_up_question": question,
            "response": question,
        }

    except Exception as e:
        logger.error("question_generator_error", error=str(e))
        fallback = "어떤 영화를 찾으시는지 좀 더 알려주세요!"
        return {
            "follow_up_question": fallback,
            "response": fallback,
        }


# ============================================================
# 6. query_builder — RAG 검색 쿼리 구성 (규칙 기반)
# ============================================================

def _parse_era(era: str) -> tuple[int, int] | None:
    """
    시대/연도 문자열을 (시작연도, 끝연도) 튜플로 변환한다.

    지원 포맷:
    - "2020년대" → (2020, 2029)
    - "90년대" → (1990, 1999)
    - "2020" → (2020, 2020)

    Args:
        era: 시대/연도 문자열

    Returns:
        (시작연도, 끝연도) 튜플, 파싱 실패 시 None
    """
    if not era:
        return None

    # "2020년대", "90년대" 패턴
    match = re.match(r"(\d{2,4})년대", era)
    if match:
        year_str = match.group(1)
        if len(year_str) == 2:
            # "90년대" → 1990
            base = 1900 + int(year_str)
        else:
            # "2020년대" → 2020
            base = int(year_str)
        return (base, base + 9)

    # 단순 연도 "2020"
    match = re.match(r"(\d{4})", era)
    if match:
        year = int(match.group(1))
        return (year, year)

    return None


@traceable(name="query_builder", run_type="chain", metadata={"node": "6/13", "llm": "none"})
async def query_builder(state: ChatAgentState) -> dict:
    """
    선호 조건과 감정 분석 결과를 기반으로 RAG 검색 쿼리를 구성한다.

    규칙 기반 (LLM 없음):
    - semantic_query: 사용자 입력 + 장르 + 무드 + 참조 영화 결합
    - keyword_query: 사용자 원문 입력
    - filters: 장르, 무드태그, OTT, 연도 범위
    - boost_keywords: 무드태그 + 참조영화
    - exclude_ids: 시청 이력 영화 ID

    Args:
        state: ChatAgentState (current_input, preferences, emotion, watch_history 필요)

    Returns:
        dict: search_query(SearchQuery) 업데이트
    """
    try:
        current_input = state.get("current_input", "")
        prefs = state.get("preferences", ExtractedPreferences())
        emotion = state.get("emotion", EmotionResult())
        watch_history = state.get("watch_history", [])

        # semantic_query 구성: 사용자 입력 + 장르 + 무드 + 참조 영화
        query_parts = [current_input]
        if prefs.genre_preference:
            query_parts.append(prefs.genre_preference)
        if prefs.mood:
            query_parts.append(prefs.mood)
        if prefs.reference_movies:
            query_parts.append(" ".join(prefs.reference_movies))
        semantic_query = " ".join(query_parts)

        # filters 구성
        filters: dict[str, Any] = {}
        if prefs.genre_preference:
            # 쉼표나 공백으로 구분된 장르를 리스트로 변환
            genres = [g.strip() for g in re.split(r"[,\s]+", prefs.genre_preference) if g.strip()]
            if genres:
                filters["genres"] = genres
        if prefs.platform:
            filters["platform"] = prefs.platform

        # 연도 범위 파싱
        year_range = _parse_era(prefs.era) if prefs.era else None
        if year_range:
            filters["year_range"] = year_range

        # boost_keywords: 무드태그 + 참조영화
        boost_keywords: list[str] = []
        if emotion and emotion.mood_tags:
            boost_keywords.extend(emotion.mood_tags)
        if prefs.reference_movies:
            boost_keywords.extend(prefs.reference_movies)

        # exclude_ids: 시청 이력 영화 ID
        exclude_ids = [str(wh.get("movie_id", "")) for wh in watch_history if wh.get("movie_id")]

        search_query = SearchQuery(
            semantic_query=semantic_query,
            keyword_query=current_input,
            filters=filters,
            boost_keywords=boost_keywords,
            exclude_ids=exclude_ids,
            limit=15,
        )

        logger.info(
            "query_built_node",
            semantic_query_preview=semantic_query[:80],
            filter_count=len(filters),
            exclude_count=len(exclude_ids),
        )
        return {"search_query": search_query}

    except Exception as e:
        logger.error("query_builder_error", error=str(e))
        # 최소한 사용자 입력으로 검색 쿼리 구성
        return {
            "search_query": SearchQuery(
                semantic_query=state.get("current_input", "영화 추천"),
                keyword_query=state.get("current_input", "영화 추천"),
            )
        }


# ============================================================
# 7. rag_retriever — 하이브리드 검색
# ============================================================

def _search_result_to_candidate(result: SearchResult, rank: int) -> CandidateMovie:
    """
    SearchResult를 CandidateMovie로 변환한다.

    메타데이터에서 영화 정보를 추출하여 CandidateMovie 필드를 채운다.

    Args:
        result: 하이브리드 검색 결과
        rank: 검색 순위 (0-based)

    Returns:
        CandidateMovie 인스턴스
    """
    meta = result.metadata or {}
    return CandidateMovie(
        id=result.movie_id,
        title=result.title or meta.get("title", ""),
        title_en=meta.get("title_en", ""),
        genres=meta.get("genres", []) if isinstance(meta.get("genres"), list) else [],
        director=meta.get("director", ""),
        cast=meta.get("cast", []) if isinstance(meta.get("cast"), list) else [],
        rating=float(meta.get("rating", 0.0) or 0.0),
        release_year=int(meta.get("release_year", 0) or 0),
        overview=meta.get("overview", ""),
        mood_tags=meta.get("mood_tags", []) if isinstance(meta.get("mood_tags"), list) else [],
        poster_path=meta.get("poster_path", ""),
        ott_platforms=meta.get("ott_platforms", []) if isinstance(meta.get("ott_platforms"), list) else [],
        certification=meta.get("certification", ""),
        trailer_url=meta.get("trailer_url", ""),
        rrf_score=result.score,
        retrieval_source=result.source,
    )


@traceable(name="rag_retriever", run_type="retriever", metadata={"node": "7/13", "fusion": "RRF"})
async def rag_retriever(state: ChatAgentState) -> dict:
    """
    하이브리드 검색(Qdrant+ES+Neo4j RRF)을 실행하여 후보 영화를 검색한다.

    SearchQuery의 필터와 부스트를 hybrid_search()에 전달한다.
    결과를 CandidateMovie 리스트로 변환한다.

    Args:
        state: ChatAgentState (search_query 필요)

    Returns:
        dict: candidate_movies(list[CandidateMovie]) 업데이트
    """
    try:
        search_query = state.get("search_query", SearchQuery())
        emotion = state.get("emotion", EmotionResult())

        # 필터 파라미터 추출
        filters = search_query.filters
        genre_filter = filters.get("genres")
        mood_tags = emotion.mood_tags if emotion else None
        ott_filter = [filters["platform"]] if filters.get("platform") else None
        year_range = filters.get("year_range")

        # 하이브리드 검색 실행
        results = await hybrid_search(
            query=search_query.semantic_query or search_query.keyword_query,
            top_k=search_query.limit,
            genre_filter=genre_filter,
            mood_tags=mood_tags,
            ott_filter=ott_filter,
            year_range=year_range,
        )

        # SearchResult → CandidateMovie 변환
        candidates = [
            _search_result_to_candidate(r, i)
            for i, r in enumerate(results)
        ]

        # exclude_ids로 시청한 영화 제외
        if search_query.exclude_ids:
            exclude_set = set(search_query.exclude_ids)
            candidates = [c for c in candidates if c.id not in exclude_set]

        logger.info(
            "rag_retrieved_node",
            candidate_count=len(candidates),
            query_preview=search_query.semantic_query[:50] if search_query.semantic_query else "",
        )
        return {"candidate_movies": candidates}

    except Exception as e:
        logger.error("rag_retriever_error", error=str(e))
        return {"candidate_movies": []}


# ============================================================
# 8. recommendation_ranker — 추천 엔진 서브그래프 호출 (Phase 4)
# ============================================================

@traceable(name="recommendation_ranker", run_type="chain", metadata={"node": "8/13"})
async def recommendation_ranker(state: ChatAgentState) -> dict:
    """
    추천 엔진 서브그래프(§7)를 호출하여 CF+CBF 하이브리드 추천을 수행한다.

    서브그래프 흐름:
    - Cold Start 판정 → (정상: CF→CBF→hybrid / Cold: popularity_fallback)
    - MMR 다양성 재정렬 → ScoreDetail 첨부

    에러 시 RRF 점수 기반 fallback으로 복원한다.

    Args:
        state: ChatAgentState (candidate_movies, user_id, user_profile,
               watch_history, emotion, preferences 필요)

    Returns:
        dict: ranked_movies(list[RankedMovie]) 업데이트
    """
    try:
        candidates = state.get("candidate_movies", [])

        if not candidates:
            logger.warning("recommendation_ranker_no_candidates")
            return {"ranked_movies": []}

        # 추천 엔진 서브그래프 호출
        from monglepick.agents.recommendation.graph import run_recommendation_engine

        ranked = await run_recommendation_engine(
            candidate_movies=candidates,
            user_id=state.get("user_id", ""),
            user_profile=state.get("user_profile", {}),
            watch_history=state.get("watch_history", []),
            emotion=state.get("emotion"),
            preferences=state.get("preferences"),
        )

        logger.info(
            "recommendation_ranked_node",
            ranked_count=len(ranked),
            top_title=ranked[0].title if ranked else "",
        )
        return {"ranked_movies": ranked}

    except Exception as e:
        # fallback: RRF 점수 기준 정렬 (서브그래프 에러 시 기존 스텁 로직)
        logger.error("recommendation_ranker_error", error=str(e))
        candidates = state.get("candidate_movies", [])
        if not candidates:
            return {"ranked_movies": []}

        sorted_candidates = sorted(candidates, key=lambda c: c.rrf_score, reverse=True)
        ranked: list[RankedMovie] = []
        for i, c in enumerate(sorted_candidates[:5]):
            ranked.append(RankedMovie(
                id=c.id,
                title=c.title,
                title_en=c.title_en,
                genres=c.genres,
                director=c.director,
                cast=c.cast,
                rating=c.rating,
                release_year=c.release_year,
                overview=c.overview,
                mood_tags=c.mood_tags,
                poster_path=c.poster_path,
                ott_platforms=c.ott_platforms,
                certification=c.certification,
                trailer_url=c.trailer_url,
                rank=i + 1,
                score_detail=ScoreDetail(
                    cf_score=0.0,
                    cbf_score=0.0,
                    hybrid_score=c.rrf_score,
                    genre_match=0.0,
                    mood_match=0.0,
                    similar_to=[],
                ),
                explanation="",
            ))
        return {"ranked_movies": ranked}


# ============================================================
# 9. explanation_generator — 추천 이유 생성
# ============================================================

@traceable(name="explanation_generator", run_type="chain", metadata={"node": "9/13", "llm": "exaone-32b"})
async def explanation_generator(state: ChatAgentState) -> dict:
    """
    각 추천 영화에 대해 사용자 맞춤 추천 이유를 생성한다.

    generate_explanations_batch()로 병렬 생성하고, 각 RankedMovie.explanation에 할당한다.

    Args:
        state: ChatAgentState (ranked_movies, emotion, preferences, watch_history 필요)

    Returns:
        dict: ranked_movies(list[RankedMovie], explanation 채워짐) 업데이트
    """
    try:
        ranked = state.get("ranked_movies", [])
        emotion = state.get("emotion")
        prefs = state.get("preferences")
        watch_history = state.get("watch_history", [])

        if not ranked:
            return {"ranked_movies": []}

        # 시청 이력 제목 목록 (상위 5개)
        watch_titles = [
            wh.get("title", "") for wh in watch_history[:5] if wh.get("title")
        ]

        emotion_str = emotion.emotion if emotion else None

        # 배치 병렬 생성
        explanations = await generate_explanations_batch(
            movies=ranked,
            emotion=emotion_str,
            preferences=prefs,
            watch_history_titles=watch_titles,
        )

        # 각 RankedMovie에 explanation 할당 (불변 모델이므로 새 인스턴스 생성)
        updated_ranked: list[RankedMovie] = []
        for movie, explanation in zip(ranked, explanations):
            updated = movie.model_copy(update={"explanation": explanation})
            updated_ranked.append(updated)

        logger.info(
            "explanations_generated_node",
            count=len(updated_ranked),
        )
        return {"ranked_movies": updated_ranked}

    except Exception as e:
        logger.error("explanation_generator_error", error=str(e))
        return {"ranked_movies": state.get("ranked_movies", [])}


# ============================================================
# 10. response_formatter — 응답 포맷팅
# ============================================================

@traceable(name="response_formatter", run_type="chain", metadata={"node": "10/13"})
async def response_formatter(state: ChatAgentState) -> dict:
    """
    응답 유형별로 최종 텍스트를 포맷팅하고, messages에 assistant 메시지를 추가한다.

    응답 유형:
    - 추천: ranked_movies가 있으면 영화 카드 포맷
    - 질문: follow_up_question / response가 이미 설정된 경우
    - 에러: error가 설정된 경우
    - 일반: response가 이미 설정된 경우

    포맷:
    - 추천: "{rank}. **{title}** ({release_year})\n- 장르: ...\n- 감독: ...\n- 평점: ...\n{explanation}"
    - 질문/일반/에러: 텍스트 그대로

    Args:
        state: ChatAgentState (ranked_movies, response, error, messages 필요)

    Returns:
        dict: response, messages 업데이트
    """
    try:
        ranked = state.get("ranked_movies", [])
        existing_response = state.get("response", "")
        error = state.get("error")
        messages = list(state.get("messages", []))

        # 에러 응답
        if error and not existing_response:
            response = "죄송해요, 지금은 추천이 어려워요. 다시 시도해주세요!"
        # 추천 응답: ranked_movies가 있으면 영화 카드 포맷
        elif ranked:
            parts = ["추천 영화를 찾았어요! 🎬\n"]
            for movie in ranked:
                genres_str = ", ".join(movie.genres[:3]) if movie.genres else "-"
                year_str = f" ({movie.release_year})" if movie.release_year else ""
                card = (
                    f"{movie.rank}. **{movie.title}**{year_str}\n"
                    f"   - 장르: {genres_str}\n"
                    f"   - 감독: {movie.director or '-'}\n"
                    f"   - 평점: {movie.rating:.1f}\n"
                )
                if movie.explanation:
                    card += f"   > {movie.explanation}\n"
                parts.append(card)
            response = "\n".join(parts)
        # 기존 response 사용 (질문/일반 대화)
        elif existing_response:
            response = existing_response
        else:
            response = "무엇을 도와드릴까요? 영화 추천이 필요하시면 말씀해주세요!"

        # assistant 메시지 추가
        messages.append({"role": "assistant", "content": response})

        logger.info(
            "response_formatted_node",
            response_length=len(response),
            has_movies=bool(ranked),
        )
        return {
            "response": response,
            "messages": messages,
        }

    except Exception as e:
        logger.error("response_formatter_error", error=str(e))
        fallback = "죄송해요, 응답을 구성하는 중 문제가 생겼어요."
        messages = list(state.get("messages", []))
        messages.append({"role": "assistant", "content": fallback})
        return {
            "response": fallback,
            "messages": messages,
        }


# ============================================================
# 11. error_handler — 에러 처리
# ============================================================

@traceable(name="error_handler", run_type="chain", metadata={"node": "11/13"})
async def error_handler(state: ChatAgentState) -> dict:
    """
    에러 로깅 후 친절한 안내 메시지를 설정한다.

    그래프에서 unknown/None 의도가 감지되었을 때 호출된다.

    Args:
        state: ChatAgentState

    Returns:
        dict: response, error 업데이트
    """
    try:
        error_msg = state.get("error", "알 수 없는 오류")
        intent = state.get("intent")
        logger.error(
            "error_handler_node",
            error=error_msg,
            intent=intent.intent if intent else None,
        )
        return {
            "response": "죄송해요, 잠시 문제가 생겼어요. 다시 한번 말씀해주세요! 🙏",
            "error": error_msg,
        }

    except Exception as e:
        logger.error("error_handler_inner_error", error=str(e))
        return {
            "response": "죄송해요, 잠시 문제가 생겼어요. 다시 한번 말씀해주세요! 🙏",
            "error": str(e),
        }


# ============================================================
# 12. general_responder — 일반 대화 응답
# ============================================================

@traceable(name="general_responder", run_type="chain", metadata={"node": "12/13", "llm": "exaone-32b"})
async def general_responder(state: ChatAgentState) -> dict:
    """
    일반 대화(intent=general)에 대해 몽글 페르소나로 응답한다.

    generate_general_response 체인을 호출하고 response에 설정한다.

    Args:
        state: ChatAgentState (current_input, messages 필요)

    Returns:
        dict: response 업데이트
    """
    try:
        current_input = state.get("current_input", "")
        messages = state.get("messages", [])

        # 최근 6개 메시지 포맷
        recent = messages[-7:-1] if len(messages) > 1 else []
        recent_messages = "\n".join(
            f"{m.get('role', 'user')}: {m.get('content', '')}"
            for m in recent[-6:]
        )

        response = await generate_general_response(
            current_input=current_input,
            recent_messages=recent_messages,
        )

        logger.info(
            "general_response_node",
            response_preview=response[:50],
        )
        return {"response": response}

    except Exception as e:
        logger.error("general_responder_error", error=str(e))
        return {"response": "안녕하세요! 영화 추천이 필요하시면 말씀해주세요 😊"}


# ============================================================
# 13. tool_executor_node — 도구 실행 (Phase 6 스텁)
# ============================================================

@traceable(name="tool_executor_node", run_type="tool", metadata={"node": "13/13"})
async def tool_executor_node(state: ChatAgentState) -> dict:
    """
    도구 실행 노드 (Phase 6 스텁).

    info/theater/booking 의도에 대해 아직 구현되지 않은 기능임을 안내한다.
    NotImplementedError를 호출하지 않고, 친절한 안내 메시지를 반환한다.

    Args:
        state: ChatAgentState (intent 필요)

    Returns:
        dict: response 업데이트
    """
    try:
        intent = state.get("intent")
        intent_str = intent.intent if intent else "unknown"

        # 의도별 안내 메시지
        messages_map = {
            "info": "영화 상세 정보 조회 기능은 곧 준비될 예정이에요! 🎬 "
                    "궁금한 영화가 있으시면 제목을 알려주세요, 아는 범위에서 추천해드릴게요!",
            "theater": "가까운 영화관 검색 기능은 아직 준비 중이에요! 🏢 "
                       "대신 보고 싶은 영화를 추천해드릴까요?",
            "booking": "예매 링크 연결 기능은 아직 준비 중이에요! 🎟️ "
                       "대신 보고 싶은 영화를 추천해드릴까요?",
        }
        response = messages_map.get(
            intent_str,
            "해당 기능은 아직 준비 중이에요. 영화 추천이 필요하시면 말씀해주세요! 🎬",
        )

        logger.info(
            "tool_executor_stub_node",
            intent=intent_str,
        )
        return {"response": response}

    except Exception as e:
        logger.error("tool_executor_node_error", error=str(e))
        return {"response": "해당 기능은 아직 준비 중이에요. 영화 추천이 필요하시면 말씀해주세요!"}
