"""
데이터 파이프라인 Pydantic 모델 정의

§11 데이터 파이프라인에서 사용하는 핵심 데이터 모델.
TMDB/KOBIS/Kaggle에서 수집한 원본 데이터를 정규화하여
Qdrant, Neo4j, Elasticsearch에 적재하는 중간 표현(MovieDocument)을 정의한다.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class MovieDocument(BaseModel):
    """
    정규화된 영화 문서 모델.

    TMDB + KOBIS + Kaggle 데이터를 병합·전처리한 결과물.
    이 모델이 Qdrant(벡터), Neo4j(그래프), Elasticsearch(BM25)에 적재되는 공통 입력이다.

    §11-1 전체 흐름: raw data → preprocessor → MovieDocument → 4개 저장소 적재
    """

    # ── 식별자 ──
    id: str = Field(..., description="TMDB ID (문자열, 예: '157336')")

    # ── 기본 메타데이터 ──
    title: str = Field(..., description="한국어 제목 (TMDB 우선, KOBIS 보강)")
    title_en: str = Field(default="", description="영문 제목")
    overview: str = Field(default="", description="줄거리 (TMDB, 한국어 우선)")
    release_year: int = Field(default=0, description="개봉 연도 (1900~현재)")
    runtime: int = Field(default=0, description="러닝타임 (분, KOBIS 우선)")
    rating: float = Field(default=0.0, description="TMDB 평점 (0~10)")
    popularity_score: float = Field(default=0.0, description="인기도 (TMDB + KOBIS 관객수 보정)")
    poster_path: str = Field(default="", description="포스터 이미지 경로")

    # ── 분류 정보 ──
    genres: list[str] = Field(default_factory=list, description="장르 한국어 배열 (예: ['SF', '드라마'])")
    keywords: list[str] = Field(default_factory=list, description="키워드 배열")

    # ── 인물 정보 ──
    director: str = Field(default="", description="감독명 (KOBIS 우선, TMDB 보강)")
    cast: list[str] = Field(default_factory=list, description="출연 배우 상위 5명")

    # ── AI 생성 필드 ──
    mood_tags: list[str] = Field(
        default_factory=list,
        description="무드 태그 3~5개 (GPT-4o-mini 생성, 25개 화이트리스트 한정)",
    )

    # ── 플랫폼 정보 ──
    ott_platforms: list[str] = Field(
        default_factory=list,
        description="OTT 플랫폼 한국어 배열 (예: ['넷플릭스', '왓챠'])",
    )

    # ── 임베딩 입력 텍스트 ──
    embedding_text: str = Field(
        default="",
        description=(
            "임베딩 모델 입력용 구조화 텍스트. "
            "형식: [제목] {title} [장르] {genres} [감독] {director} "
            "[키워드] {keywords} [무드] {mood_tags} [줄거리] {overview[:200]}"
        ),
    )

    # ── 데이터 출처 추적 ──
    source: str = Field(default="tmdb", description="데이터 출처 ('tmdb', 'kobis', 'kaggle', 'merged')")


class TMDBRawMovie(BaseModel):
    """
    TMDB API에서 수집한 원본 영화 데이터.

    /movie/{id}?append_to_response=credits,keywords 응답을 파싱한 구조.
    전처리기(preprocessor)가 이 모델을 MovieDocument로 변환한다.
    """

    id: int
    title: str = ""
    original_title: str = ""
    overview: str = ""
    release_date: str = ""
    vote_average: float = 0.0
    popularity: float = 0.0
    poster_path: str | None = None
    runtime: int | None = None

    # 장르 (TMDB genre 객체 배열)
    genres: list[dict] = Field(default_factory=list, description="[{'id': 28, 'name': 'Action'}, ...]")

    # 크레딧 (감독 + 배우)
    credits: dict = Field(default_factory=dict, description="{'crew': [...], 'cast': [...]}")

    # 키워드
    keywords: dict = Field(default_factory=dict, description="{'keywords': [{'id': 1, 'name': '...'}]}")

    # OTT 제공 정보 (watch/providers)
    watch_providers: dict = Field(default_factory=dict, description="{'KR': {'flatrate': [...]}}")


class KOBISRawMovie(BaseModel):
    """
    KOBIS API에서 수집한 원본 한국 영화 데이터.

    영화 상세정보 API (/movie/searchMovieInfo) 응답을 파싱한 구조.
    """

    movie_cd: str = Field(..., description="KOBIS 영화 코드")
    movie_nm: str = Field(default="", description="한국어 영화명")
    movie_nm_en: str = Field(default="", description="영문 영화명")
    open_dt: str = Field(default="", description="개봉일 (YYYYMMDD)")
    show_tm: str = Field(default="", description="상영시간 (분)")
    genre_alt: str = Field(default="", description="장르 (쉼표 구분)")
    directors: list[dict] = Field(default_factory=list, description="[{'peopleNm': '...'}]")
    actors: list[dict] = Field(default_factory=list, description="[{'peopleNm': '...'}]")
    audi_acc: int = Field(default=0, description="누적관객수")
    watch_grade_nm: str = Field(default="", description="관람등급")


class KOBISBoxOffice(BaseModel):
    """KOBIS 박스오피스 데이터 (트렌딩 분석용)."""

    movie_cd: str
    movie_nm: str = ""
    rank: int = 0
    rank_inten: int = Field(default=0, description="전일 대비 순위 변동")
    rank_old_and_new: str = Field(default="OLD", description="'NEW' or 'OLD'")
    audi_cnt: int = Field(default=0, description="해당일 관객수")
    scrn_cnt: int = Field(default=0, description="상영 스크린 수")
    open_dt: str = ""


class PipelineState(BaseModel):
    """
    배치 파이프라인 진행 상태 (data/pipeline_state.json).

    §11-9: 중단점 재개를 위해 마지막 처리 지점을 기록한다.
    """

    last_movie_id: str = Field(default="", description="마지막 처리 완료한 TMDB ID")
    current_step: str = Field(default="", description="현재 진행 단계 (collect/preprocess/embed/load)")
    total_collected: int = Field(default=0, description="수집 완료 영화 수")
    total_processed: int = Field(default=0, description="전처리 완료 영화 수")
    total_loaded: int = Field(default=0, description="적재 완료 영화 수")
    failed_ids: list[str] = Field(default_factory=list, description="처리 실패한 TMDB ID 목록")
    timestamp: str = Field(default="", description="마지막 업데이트 시각 (ISO 8601)")
