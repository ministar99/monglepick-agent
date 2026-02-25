"""
Elasticsearch BM25 인덱스 적재기.

§10-8 Elasticsearch 인덱스 설정:
- 인덱스명: movies_bm25
- Nori 한국어 분석기 (형태소 분석)
- 12개 필드 매핑 (text + keyword + numeric)
- 배치 적재: bulk API 사용
"""

from __future__ import annotations

import structlog
from elasticsearch.helpers import async_bulk

from monglepick.data_pipeline.models import MovieDocument
from monglepick.db.clients import ES_INDEX_NAME, get_elasticsearch

logger = structlog.get_logger()


def _movie_to_es_doc(doc: MovieDocument) -> dict:
    """MovieDocument를 Elasticsearch 인덱싱용 dict로 변환한다."""
    return {
        "_index": ES_INDEX_NAME,
        "_id": doc.id,
        "_source": {
            "id": doc.id,
            "title": doc.title,
            "title_en": doc.title_en,
            "director": doc.director,
            "overview": doc.overview,
            # cast: 배열 → 공백 구분 텍스트 (text 필드용)
            "cast": " ".join(doc.cast),
            "keywords": " ".join(doc.keywords),
            "genres": doc.genres,
            "mood_tags": doc.mood_tags,
            "ott_platforms": doc.ott_platforms,
            "release_year": doc.release_year,
            "rating": doc.rating,
            "popularity_score": doc.popularity_score,
        },
    }


async def load_to_elasticsearch(
    documents: list[MovieDocument],
    chunk_size: int = 500,
) -> int:
    """
    MovieDocument 리스트를 Elasticsearch에 bulk 적재한다.

    §10-8: movies_bm25 인덱스에 Nori 분석기로 인덱싱.
    배치 적재 시 refresh_interval을 -1로 비활성화 후 완료 시 복원한다.

    Args:
        documents: 적재할 MovieDocument 리스트
        chunk_size: bulk API 배치 크기 (기본 500)

    Returns:
        int: 성공적으로 적재된 문서 수
    """
    client = await get_elasticsearch()

    # 배치 적재 중 refresh 비활성화 (성능 최적화)
    await client.indices.put_settings(
        index=ES_INDEX_NAME,
        body={"refresh_interval": "-1"},
    )

    # bulk 적재
    actions = [_movie_to_es_doc(doc) for doc in documents]

    success_count, errors = await async_bulk(
        client,
        actions,
        chunk_size=chunk_size,
        raise_on_error=False,
    )

    if errors:
        logger.warning("es_bulk_errors", error_count=len(errors))

    # refresh 복원 + 강제 refresh
    await client.indices.put_settings(
        index=ES_INDEX_NAME,
        body={"refresh_interval": "30s"},
    )
    await client.indices.refresh(index=ES_INDEX_NAME)

    # 적재 확인
    count_resp = await client.count(index=ES_INDEX_NAME)
    total = count_resp["count"]

    logger.info(
        "es_load_complete",
        loaded=success_count,
        total_in_index=total,
    )

    return success_count
