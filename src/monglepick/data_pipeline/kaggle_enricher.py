"""
Kaggle 데이터 → MovieDocument 변환기.

TMDB API로 수집되지 않은 영화를 Kaggle CSV에서 직접 변환하여 보강한다.
Kaggle에는 45,000+건의 영화가 있으나, TMDB API로는 ~3,700건만 수집 가능하므로
나머지 ~44,000건을 Kaggle 데이터로 채운다.

Kaggle CSV 구조:
- movies_metadata.csv: id, title, overview, genres, release_date, vote_average, popularity, poster_path, runtime
- credits.csv: id, cast(JSON), crew(JSON)
- keywords.csv: id, keywords(JSON)

Kaggle에서 제공되지 않는 필드:
- OTT 플랫폼 (watch_providers): 빈 리스트로 설정
- 한국어 제목: original_title 사용 (대부분 영문)
"""

from __future__ import annotations

import structlog
import pandas as pd

from monglepick.data_pipeline.kaggle_loader import KaggleLoader
from monglepick.data_pipeline.models import MovieDocument
from monglepick.data_pipeline.preprocessor import (
    convert_genres,
    build_embedding_text,
    get_fallback_mood_tags,
    validate_movie,
)

logger = structlog.get_logger()


def load_kaggle_movies(
    kaggle_dir: str,
    exclude_ids: set[int] | None = None,
) -> list[MovieDocument]:
    """
    Kaggle CSV에서 MovieDocument 리스트를 생성한다.

    이미 적재된 ID는 제외하고, 유효한 영화만 반환한다.

    Args:
        kaggle_dir: Kaggle 데이터 디렉토리 경로
        exclude_ids: 제외할 TMDB ID 집합 (이미 적재된 영화)

    Returns:
        list[MovieDocument]: 변환된 영화 문서 리스트
    """
    exclude_ids = exclude_ids or set()
    loader = KaggleLoader(kaggle_dir)

    # 1. 메타데이터 로드
    logger.info("kaggle_enricher_loading_metadata")
    metadata_df = loader.load_movies_metadata()

    # 2. 크레딧 로드 (감독/배우)
    logger.info("kaggle_enricher_loading_credits")
    credits_df = loader.load_credits()

    # 3. 키워드 로드
    logger.info("kaggle_enricher_loading_keywords")
    keywords_df = loader.load_keywords()

    # 4. 크레딧, 키워드를 메타데이터에 조인 (TMDB ID 기준)
    merged = metadata_df.merge(credits_df, on="id", how="left")
    merged = merged.merge(keywords_df, on="id", how="left")

    # 이미 적재된 ID 제외
    if exclude_ids:
        before = len(merged)
        merged = merged[~merged["id"].isin(exclude_ids)]
        excluded = before - len(merged)
        logger.info("kaggle_enricher_excluded_existing", excluded=excluded, remaining=len(merged))

    # 5. MovieDocument로 변환
    logger.info("kaggle_enricher_converting", count=len(merged))
    documents: list[MovieDocument] = []
    failed = 0

    for _, row in merged.iterrows():
        try:
            doc = _row_to_movie_document(row)
            if doc and validate_movie(doc):
                documents.append(doc)
            else:
                failed += 1
        except Exception as e:
            failed += 1
            if failed <= 10:  # 처음 10개만 로그
                logger.warning("kaggle_enricher_convert_failed", id=row.get("id"), error=str(e))

    logger.info(
        "kaggle_enricher_complete",
        total=len(merged),
        success=len(documents),
        failed=failed,
    )

    return documents


def _row_to_movie_document(row: pd.Series) -> MovieDocument | None:
    """
    Kaggle 메타데이터 DataFrame 행을 MovieDocument로 변환한다.

    Args:
        row: merged DataFrame의 한 행
             (metadata + credits + keywords 조인 결과)

    Returns:
        MovieDocument 또는 None (변환 불가 시)
    """
    tmdb_id = int(row["id"])

    # 제목 (한국어 제목이 없으면 원제 사용)
    title = row.get("title", "") or row.get("original_title", "")
    title_en = row.get("original_title", "") or title
    if not title:
        return None

    # 개봉 연도 추출
    release_date = str(row.get("release_date", ""))
    release_year = 0
    if release_date and len(release_date) >= 4:
        try:
            release_year = int(release_date[:4])
        except ValueError:
            pass

    # 장르 변환 (Kaggle의 genres_parsed는 [{"id": 28, "name": "Action"}, ...] 형태)
    genres_parsed = row.get("genres_parsed", [])
    if not isinstance(genres_parsed, list):
        genres_parsed = []
    genres = convert_genres(genres_parsed)

    # 감독/배우 (credits_df에서 조인된 컬럼)
    director = row.get("director", "") or ""
    cast_names = row.get("cast_names", [])
    if not isinstance(cast_names, list):
        cast_names = []

    # 키워드 (keywords_df에서 조인된 컬럼)
    keywords_list = row.get("keywords_list", [])
    if not isinstance(keywords_list, list):
        keywords_list = []

    # 기본 필드
    overview = str(row.get("overview", "")) if pd.notna(row.get("overview")) else ""
    rating = float(row.get("vote_average", 0)) if pd.notna(row.get("vote_average")) else 0.0
    popularity = float(row.get("popularity", 0)) if pd.notna(row.get("popularity")) else 0.0
    runtime = int(row.get("runtime", 0)) if pd.notna(row.get("runtime")) else 0
    poster_path = str(row.get("poster_path", "")) if pd.notna(row.get("poster_path")) else ""

    # 무드태그 (장르 기반 fallback)
    mood_tags = get_fallback_mood_tags(genres)

    # 줄거리 빈 문자열 대체
    if not overview or len(overview) < 10:
        overview = f"{', '.join(genres)} 장르의 영화. 키워드: {', '.join(keywords_list[:5])}"

    # MovieDocument 생성
    doc = MovieDocument(
        id=str(tmdb_id),
        title=title,
        title_en=title_en,
        overview=overview,
        release_year=release_year,
        runtime=runtime,
        rating=rating,
        popularity_score=popularity,
        poster_path=poster_path,
        genres=genres,
        keywords=keywords_list,
        director=director,
        cast=cast_names,
        ott_platforms=[],  # Kaggle에는 OTT 정보 없음
        mood_tags=mood_tags,
        source="kaggle",
    )

    # 임베딩 텍스트 구성
    doc.embedding_text = build_embedding_text(doc)

    return doc
