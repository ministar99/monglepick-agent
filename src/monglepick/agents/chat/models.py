"""
Chat Agent Pydantic 모델 정의.

§6 Chat Agent 그래프의 각 노드가 사용하는 데이터 모델.
Phase 2에서 체인 입출력으로 사용하고, Phase 3에서 LangGraph State에 통합한다.

모델 목록:
- IntentResult: 의도 분류 결과 (6가지 intent + confidence)
- EmotionResult: 감정 분석 결과 (emotion + mood_tags)
- ExtractedPreferences: 사용자 선호 조건 7개 필드
- SearchQuery: RAG 검색 쿼리 구조 (Phase 3 준비)
- CandidateMovie: 검색 결과 후보 영화 (Phase 3 준비)
- ScoreDetail: 추천 점수 분해 (Phase 4 준비)
- RankedMovie: 최종 추천 영화 (Phase 3 준비)
- ChatAgentState: LangGraph TypedDict State (Phase 3 준비)

유틸 함수:
- calculate_sufficiency: 가중치 기반 충분성 점수 계산
- is_sufficient: 추천 진행 가능 여부 판정
- merge_preferences: 이전 선호 + 현재 선호 병합
"""

from __future__ import annotations

from typing import Any, Literal, TypedDict

from pydantic import BaseModel, Field


# ============================================================
# 의도 분류 결과
# ============================================================

# 6가지 의도 타입 (§6-2 Node 2)
IntentType = Literal["recommend", "search", "info", "theater", "booking", "general"]


class IntentResult(BaseModel):
    """
    의도 분류 LLM 출력 모델.

    intent: 6가지 중 하나 (recommend, search, info, theater, booking, general)
    confidence: 0.0~1.0 신뢰도 (< 0.6이면 general로 보정)
    """

    intent: IntentType = Field(
        default="general",
        description="분류된 사용자 의도 (6가지 중 하나)",
    )
    confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="의도 분류 신뢰도 (0.0~1.0)",
    )


# ============================================================
# 감정 분석 결과
# ============================================================

class EmotionResult(BaseModel):
    """
    감정 분석 LLM 출력 모델.

    emotion: 감지된 감정 (happy, sad, excited, angry, calm, None)
    mood_tags: 감정→무드 매핑 결과 (25개 화이트리스트 한정)
    """

    emotion: str | None = Field(
        default=None,
        description="감지된 감정 (happy/sad/excited/angry/calm 또는 None)",
    )
    mood_tags: list[str] = Field(
        default_factory=list,
        description="감정에서 매핑된 무드 태그 목록 (25개 화이트리스트 한정)",
    )


# ============================================================
# 사용자 선호 조건
# ============================================================

class ExtractedPreferences(BaseModel):
    """
    사용자 선호 조건 7개 필드 (§6-2 Node 4).

    모든 필드는 Optional — 아직 파악되지 않은 선호는 None으로 유지.
    가중치를 기반으로 충분성을 판단하여 추천 진행 여부를 결정한다.

    가중치 테이블:
    - genre_preference: 2.0
    - mood: 2.0
    - viewing_context: 1.0
    - platform: 1.0
    - reference_movies: 1.5
    - era: 0.5
    - exclude: 0.5
    """

    genre_preference: str | None = Field(
        default=None,
        description="선호 장르 (예: 'SF', '액션 코미디')",
    )
    mood: str | None = Field(
        default=None,
        description="원하는 분위기 (예: '따뜻한', '긴장감 있는')",
    )
    viewing_context: str | None = Field(
        default=None,
        description="시청 상황 (예: '혼자', '연인과', '가족과')",
    )
    platform: str | None = Field(
        default=None,
        description="시청 플랫폼 (예: '넷플릭스', '극장')",
    )
    reference_movies: list[str] = Field(
        default_factory=list,
        description="참조 영화 제목 목록 (예: ['인셉션', '인터스텔라'])",
    )
    era: str | None = Field(
        default=None,
        description="선호 시대/연도 (예: '2020년대', '90년대')",
    )
    exclude: str | None = Field(
        default=None,
        description="제외 조건 (예: '공포는 빼주세요', '한국 영화 말고')",
    )


# ============================================================
# 선호 충분성 가중치 테이블 (§6-2 Node 4)
# ============================================================

PREFERENCE_WEIGHTS: dict[str, float] = {
    "genre_preference": 2.0,
    "mood": 2.0,
    "viewing_context": 1.0,
    "platform": 1.0,
    "reference_movies": 1.5,
    "era": 0.5,
    "exclude": 0.5,
}

# 충분성 판정 임계값
SUFFICIENCY_THRESHOLD: float = 3.0
# 턴 카운트 오버라이드 임계값 (선호 부족해도 3턴 이상이면 추천 진행)
TURN_COUNT_OVERRIDE: int = 3


# ============================================================
# RAG 검색 쿼리 (Phase 3 준비)
# ============================================================

class SearchQuery(BaseModel):
    """
    RAG 검색용 구조화된 쿼리 (§6-2 Node 6).

    query_builder 노드가 선호 조건을 기반으로 생성한다.
    rag_retriever 노드가 이 쿼리로 Qdrant + ES + Neo4j 하이브리드 검색을 수행한다.
    """

    semantic_query: str = Field(
        default="",
        description="벡터 검색용 자연어 쿼리 (Qdrant 임베딩 검색)",
    )
    keyword_query: str = Field(
        default="",
        description="BM25 키워드 검색용 쿼리 (Elasticsearch)",
    )
    filters: dict[str, Any] = Field(
        default_factory=dict,
        description="필터 조건 (장르, 연도, 플랫폼 등)",
    )
    boost_keywords: list[str] = Field(
        default_factory=list,
        description="가산점 부여 키워드 목록",
    )
    exclude_ids: list[str] = Field(
        default_factory=list,
        description="제외할 영화 ID 목록 (이미 추천/시청한 영화)",
    )
    limit: int = Field(
        default=15,
        description="검색 결과 최대 수 (10~15편)",
    )


# ============================================================
# 검색 결과 후보 영화 (Phase 3 준비)
# ============================================================

class CandidateMovie(BaseModel):
    """
    하이브리드 검색 결과 후보 영화 (§6-2 Node 7 출력).

    RRF 합산된 점수와 함께 영화 메타데이터를 포함한다.
    recommendation_ranker 노드의 입력으로 전달된다.
    """

    id: str = Field(..., description="영화 ID (TMDB/KOBIS/KMDb)")
    title: str = Field(default="", description="영화 제목")
    title_en: str = Field(default="", description="영문 제목")
    genres: list[str] = Field(default_factory=list, description="장르 목록")
    director: str = Field(default="", description="감독명")
    cast: list[str] = Field(default_factory=list, description="출연 배우 목록")
    rating: float = Field(default=0.0, description="TMDB 평점")
    release_year: int = Field(default=0, description="개봉 연도")
    overview: str = Field(default="", description="줄거리")
    mood_tags: list[str] = Field(default_factory=list, description="무드 태그")
    poster_path: str = Field(default="", description="포스터 경로")
    ott_platforms: list[str] = Field(default_factory=list, description="OTT 플랫폼 목록")
    certification: str = Field(default="", description="관람등급")
    trailer_url: str = Field(default="", description="트레일러 URL")
    rrf_score: float = Field(default=0.0, description="RRF 합산 점수")
    retrieval_source: str = Field(
        default="hybrid",
        description="검색 소스 (qdrant/es/neo4j/hybrid)",
    )


# ============================================================
# 추천 점수 분해 (Phase 4 준비)
# ============================================================

class ScoreDetail(BaseModel):
    """
    추천 점수 분해 (§7-2 Node 6 출력).

    추천 엔진 서브그래프가 생성하는 CF/CBF/하이브리드 점수 상세 정보.
    explanation_generator에서 추천 이유 생성 시 참조한다.
    """

    cf_score: float = Field(default=0.0, description="협업 필터링(CF) 점수")
    cbf_score: float = Field(default=0.0, description="컨텐츠 기반 필터링(CBF) 점수")
    hybrid_score: float = Field(default=0.0, description="하이브리드 합산 점수")
    genre_match: float = Field(default=0.0, description="장르 일치도 (0.0~1.0)")
    mood_match: float = Field(default=0.0, description="무드 일치도 (0.0~1.0)")
    similar_to: list[str] = Field(
        default_factory=list,
        description="유사 영화 제목 목록 (추천 이유 보조)",
    )


# ============================================================
# 최종 추천 영화 (Phase 3 준비)
# ============================================================

class RankedMovie(BaseModel):
    """
    최종 추천 영화 (§6-2 Node 8 출력).

    recommendation_ranker가 CandidateMovie를 점수별로 정렬하고
    ScoreDetail을 첨부하여 생성한다. explanation_generator의 입력.
    """

    id: str = Field(..., description="영화 ID")
    title: str = Field(default="", description="영화 제목")
    title_en: str = Field(default="", description="영문 제목")
    genres: list[str] = Field(default_factory=list, description="장르 목록")
    director: str = Field(default="", description="감독명")
    cast: list[str] = Field(default_factory=list, description="출연 배우 목록")
    rating: float = Field(default=0.0, description="TMDB 평점")
    release_year: int = Field(default=0, description="개봉 연도")
    overview: str = Field(default="", description="줄거리")
    mood_tags: list[str] = Field(default_factory=list, description="무드 태그")
    poster_path: str = Field(default="", description="포스터 경로")
    ott_platforms: list[str] = Field(default_factory=list, description="OTT 플랫폼 목록")
    certification: str = Field(default="", description="관람등급")
    trailer_url: str = Field(default="", description="트레일러 URL")
    rank: int = Field(default=0, description="추천 순위 (1부터)")
    score_detail: ScoreDetail = Field(
        default_factory=ScoreDetail,
        description="추천 점수 분해 상세 정보",
    )
    explanation: str = Field(default="", description="추천 이유 텍스트 (explanation_generator 생성)")


# ============================================================
# LangGraph State (Phase 3 준비)
# ============================================================

class ChatAgentState(TypedDict, total=False):
    """
    Chat Agent LangGraph TypedDict State (§6-1).

    11개 노드가 이 State를 읽고 쓰며 그래프를 진행한다.
    TypedDict를 사용하는 이유: LangGraph StateGraph 호환 (Pydantic X).
    total=False: 모든 키가 Optional (초기 State에 일부만 존재).
    """

    # ── 입력 ──
    user_id: str
    session_id: str
    current_input: str

    # ── context_loader 출력 ──
    user_profile: dict[str, Any]
    watch_history: list[dict[str, Any]]
    messages: list[dict[str, str]]

    # ── intent_classifier 출력 ──
    intent: IntentResult

    # ── emotion_analyzer 출력 ──
    emotion: EmotionResult

    # ── preference_refiner 출력 ──
    preferences: ExtractedPreferences
    needs_clarification: bool
    turn_count: int

    # ── question_generator 출력 ──
    follow_up_question: str

    # ── query_builder 출력 ──
    search_query: SearchQuery

    # ── rag_retriever 출력 ──
    candidate_movies: list[CandidateMovie]

    # ── recommendation_ranker 출력 ──
    ranked_movies: list[RankedMovie]

    # ── response_formatter 출력 ──
    response: str

    # ── error_handler ──
    error: str | None


# ============================================================
# 유틸 함수: 선호 충분성 판정 + 병합
# ============================================================

def calculate_sufficiency(
    prefs: ExtractedPreferences,
    has_emotion: bool = False,
) -> float:
    """
    선호 조건의 가중치 합산 점수를 계산한다 (§6-2 Node 4).

    채워진 필드의 가중치를 합산하여 충분성 점수를 반환.
    has_emotion이 True이면 mood 가중치(2.0)를 추가한다
    (감정 분석 결과가 있으면 무드가 암시적으로 파악된 것으로 간주).

    Args:
        prefs: 현재까지 파악된 사용자 선호 조건
        has_emotion: 감정 분석 결과 존재 여부

    Returns:
        가중치 합산 점수 (float)
    """
    score = 0.0
    # 각 필드가 채워져 있으면 해당 가중치를 합산
    if prefs.genre_preference:
        score += PREFERENCE_WEIGHTS["genre_preference"]
    if prefs.mood:
        score += PREFERENCE_WEIGHTS["mood"]
    elif has_emotion:
        # 감정이 감지되면 mood가 없어도 무드 가중치 부여
        score += PREFERENCE_WEIGHTS["mood"]
    if prefs.viewing_context:
        score += PREFERENCE_WEIGHTS["viewing_context"]
    if prefs.platform:
        score += PREFERENCE_WEIGHTS["platform"]
    if prefs.reference_movies:
        score += PREFERENCE_WEIGHTS["reference_movies"]
    if prefs.era:
        score += PREFERENCE_WEIGHTS["era"]
    if prefs.exclude:
        score += PREFERENCE_WEIGHTS["exclude"]
    return score


def is_sufficient(
    prefs: ExtractedPreferences,
    turn_count: int = 0,
    has_emotion: bool = False,
) -> bool:
    """
    추천 진행 가능 여부를 판정한다 (§6-2 Node 4).

    판정 기준 (OR 조건):
    1. 가중치 합산 >= 3.0 (SUFFICIENCY_THRESHOLD)
    2. turn_count >= 3 (TURN_COUNT_OVERRIDE)

    Args:
        prefs: 현재까지 파악된 사용자 선호 조건
        turn_count: 현재 대화 턴 수
        has_emotion: 감정 분석 결과 존재 여부

    Returns:
        True면 추천 진행, False면 후속 질문 필요
    """
    # 턴 카운트 오버라이드: 3턴 이상이면 선호 부족해도 추천 진행
    if turn_count >= TURN_COUNT_OVERRIDE:
        return True
    # 가중치 합산 기반 판정
    return calculate_sufficiency(prefs, has_emotion) >= SUFFICIENCY_THRESHOLD


def merge_preferences(
    prev: ExtractedPreferences | None,
    curr: ExtractedPreferences,
) -> ExtractedPreferences:
    """
    이전 선호 조건과 현재 추출된 선호 조건을 병합한다.

    병합 규칙:
    - 새 값이 None이 아니면 덮어쓰기
    - 새 값이 None이면 이전 값 유지
    - reference_movies: 합집합 (중복 제거)

    Args:
        prev: 이전 턴까지 누적된 선호 조건 (None이면 빈 조건)
        curr: 현재 턴에서 추출된 선호 조건

    Returns:
        병합된 ExtractedPreferences
    """
    if prev is None:
        return curr

    return ExtractedPreferences(
        genre_preference=curr.genre_preference if curr.genre_preference is not None else prev.genre_preference,
        mood=curr.mood if curr.mood is not None else prev.mood,
        viewing_context=curr.viewing_context if curr.viewing_context is not None else prev.viewing_context,
        platform=curr.platform if curr.platform is not None else prev.platform,
        # reference_movies: 합집합 (이전 + 현재, 중복 제거, 순서 유지)
        reference_movies=list(dict.fromkeys(prev.reference_movies + curr.reference_movies)),
        era=curr.era if curr.era is not None else prev.era,
        exclude=curr.exclude if curr.exclude is not None else prev.exclude,
    )
