"""
Upstage Solar 임베딩 생성기.

Upstage Embedding API (OpenAI 호환)를 사용하여 텍스트를 벡터로 변환한다.

공식 문서: https://console.upstage.ai/docs/capabilities/embed

모델:
- embedding-passage: 문서/passage 임베딩 (긴 텍스트 단락용)
- embedding-query:   검색 쿼리 임베딩 (짧은 질문/검색어용)

API:
- Base URL: https://api.upstage.ai/v1
- 인증: Authorization: Bearer {UPSTAGE_API_KEY}
- Rate Limit: 100 RPM / 300,000 TPM
- 벡터 정규화: magnitude=1 (코사인 유사도 = 내적)

OpenAI 호환 API이므로 openai 패키지를 그대로 사용한다.
"""

from __future__ import annotations

import asyncio

import numpy as np
import structlog
from openai import OpenAI

from monglepick.config import settings

logger = structlog.get_logger()

# Upstage API 클라이언트 (싱글턴)
_client: OpenAI | None = None

# Upstage API Base URL (공식 문서 기준)
UPSTAGE_BASE_URL = "https://api.upstage.ai/v1"

# Rate Limit: 100 RPM → 안전하게 배치당 딜레이 추가
RATE_LIMIT_DELAY = 0.7  # 초 (100 RPM = 1.67 req/sec → 여유 확보)


def _get_client() -> OpenAI:
    """Upstage API 클라이언트를 반환한다 (싱글턴)."""
    global _client
    if _client is None:
        _client = OpenAI(
            api_key=settings.UPSTAGE_API_KEY,
            base_url=UPSTAGE_BASE_URL,
        )
        logger.info("upstage_embedding_client_initialized")
    return _client


def embed_texts(texts: list[str], batch_size: int = 50) -> np.ndarray:
    """
    텍스트 리스트를 벡터로 변환한다 (문서/passage 용도).

    Upstage 'embedding-passage' 모델을 사용한다.
    긴 텍스트 단락에 최적화되어 있다.

    Rate Limit (100 RPM)을 준수하기 위해 배치 간 딜레이를 적용한다.

    Args:
        texts: 임베딩할 텍스트 리스트
        batch_size: API 배치 크기 (기본 50, TPM 고려)

    Returns:
        np.ndarray: shape (len(texts), embedding_dimension)
    """
    client = _get_client()
    all_embeddings: list[list[float]] = []

    # 배치 단위로 API 호출
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]

        response = client.embeddings.create(
            model="embedding-passage",
            input=batch,
        )

        # 응답에서 임베딩 벡터 추출 (인덱스 순서 보장)
        batch_embeddings = [
            item.embedding
            for item in sorted(response.data, key=lambda x: x.index)
        ]
        all_embeddings.extend(batch_embeddings)

        completed = min(i + batch_size, len(texts))
        if completed % 500 == 0 or completed >= len(texts):
            logger.info("embedding_progress", completed=completed, total=len(texts))

        # Rate Limit 준수 (100 RPM)
        if i + batch_size < len(texts):
            import time
            time.sleep(RATE_LIMIT_DELAY)

    result = np.array(all_embeddings)
    logger.info("texts_embedded", count=len(texts), dimension=result.shape[1])
    return result


def embed_query(query: str) -> np.ndarray:
    """
    검색 쿼리를 벡터로 변환한다.

    Upstage 'embedding-query' 모델을 사용한다.
    짧은 검색 질문에 최적화되어 있다.

    Returns:
        np.ndarray: shape (embedding_dimension,)
    """
    client = _get_client()

    response = client.embeddings.create(
        model="embedding-query",
        input=[query],
    )

    return np.array(response.data[0].embedding)
