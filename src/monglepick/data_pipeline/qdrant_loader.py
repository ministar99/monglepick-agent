"""
Qdrant 벡터 DB 적재기.

§11-7-1 Qdrant 적재 명세:
- upsert 배치 크기: 100 points
- 병렬 요청 수: 4 (asyncio.Semaphore(4))
- 재시도: 3회 (지수 백오프: 1s, 2s, 4s)
- wait 옵션: True (upsert 완료 확인)

적재 흐름: MovieDocument[] → 임베딩 배치 생성 → PointStruct 생성 → 배치 upsert
"""

from __future__ import annotations

import asyncio

import numpy as np
import structlog
from qdrant_client.models import PointStruct
from tenacity import retry, stop_after_attempt, wait_exponential

from monglepick.config import settings
from monglepick.data_pipeline.models import MovieDocument
from monglepick.db.clients import get_qdrant

logger = structlog.get_logger()

# §11-7-1: 병렬 upsert 제한
_upsert_semaphore = asyncio.Semaphore(4)


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, max=4))
async def _upsert_batch(points: list[PointStruct]) -> None:
    """단일 배치를 Qdrant에 upsert한다. 최대 3회 재시도."""
    async with _upsert_semaphore:
        client = await get_qdrant()
        await client.upsert(
            collection_name=settings.QDRANT_COLLECTION,
            points=points,
            wait=True,  # §11-7-1: 완료 확인 후 다음 배치
        )


def _movie_to_point(doc: MovieDocument, vector: np.ndarray) -> PointStruct:
    """MovieDocument + 임베딩 벡터를 Qdrant PointStruct로 변환한다."""
    return PointStruct(
        id=int(doc.id),  # §11-7-1: TMDB ID (정수)
        vector=vector.tolist(),
        payload={
            "title": doc.title,
            "title_en": doc.title_en,
            "genres": doc.genres,
            "director": doc.director,
            "cast": doc.cast,
            "mood_tags": doc.mood_tags,
            "ott_platforms": doc.ott_platforms,
            "release_year": doc.release_year,
            "rating": doc.rating,
            "popularity_score": doc.popularity_score,
            "overview": doc.overview,
            "keywords": doc.keywords,
            "poster_path": doc.poster_path,
            "runtime": doc.runtime,
        },
    )


async def load_to_qdrant(
    documents: list[MovieDocument],
    embeddings: np.ndarray,
    batch_size: int = 100,
) -> int:
    """
    MovieDocument 리스트와 임베딩을 Qdrant에 적재한다.

    §11-7-1 적재 흐름:
    [1] PointStruct 생성 → [2] 100건씩 배치 → [3] upsert (4 병렬)

    Args:
        documents: 적재할 MovieDocument 리스트
        embeddings: 임베딩 벡터 배열 (shape: len(documents) × 1024)
        batch_size: upsert 배치 크기 (기본 100)

    Returns:
        int: 적재 완료된 포인트 수
    """
    # PointStruct 변환
    points = [
        _movie_to_point(doc, embeddings[i])
        for i, doc in enumerate(documents)
    ]

    # 배치 분할 및 병렬 upsert
    tasks = []
    for i in range(0, len(points), batch_size):
        batch = points[i : i + batch_size]
        tasks.append(_upsert_batch(batch))

    await asyncio.gather(*tasks)

    # 적재 검증 (§11-7-1 [4])
    client = await get_qdrant()
    info = await client.get_collection(settings.QDRANT_COLLECTION)
    logger.info(
        "qdrant_load_complete",
        loaded=len(points),
        total_in_collection=info.points_count,
    )

    return len(points)
