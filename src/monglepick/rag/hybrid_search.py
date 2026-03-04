"""
하이브리드 검색 (Qdrant 벡터 + Elasticsearch BM25 + Neo4j 그래프 + RRF 합산).

§11-1 하이브리드 검색 시 5개 저장소 협업 흐름:
1. Qdrant 벡터 검색: 의미적 유사도 Top-30
2. ES BM25 검색: 키워드 매칭 Top-20
3. Neo4j 그래프 검색: 무드+장르 관계 탐색 Top-15
4. RRF 합산: Reciprocal Rank Fusion (k=60) → 최종 후보 15편

§6-2-1 Chat Agent의 query_builder/rag_retriever 노드에서 호출된다.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import structlog
from langsmith import traceable
from qdrant_client.models import FieldCondition, Filter, MatchAny, MatchValue

from monglepick.config import settings
from monglepick.data_pipeline.embedder import embed_query
from monglepick.db.clients import ES_INDEX_NAME, get_elasticsearch, get_neo4j, get_qdrant

logger = structlog.get_logger()

# RRF 상수 (§11-1: k=60)
RRF_K = 60


@dataclass
class SearchResult:
    """검색 결과 단일 항목."""
    movie_id: str
    title: str = ""
    score: float = 0.0
    source: str = ""  # "qdrant", "es", "neo4j"
    metadata: dict = field(default_factory=dict)


# ============================================================
# 1. Qdrant 벡터 검색
# ============================================================

@traceable(name="search_qdrant", run_type="retriever", metadata={"db": "qdrant", "type": "vector"})
async def search_qdrant(
    query: str,
    top_k: int = 30,
    genre_filter: list[str] | None = None,
    mood_filter: list[str] | None = None,
    ott_filter: list[str] | None = None,
    min_rating: float | None = None,
    year_range: tuple[int, int] | None = None,
) -> list[SearchResult]:
    """
    Qdrant 벡터 검색: 쿼리의 의미적 유사도로 영화를 검색한다.

    §11-1 ①: 쿼리 벡터 생성 → 코사인 유사도 Top-30 + 메타데이터 필터
    """
    client = await get_qdrant()

    # 쿼리 임베딩 (multilingual-e5: "query: " 접두사)
    query_vector = embed_query(query).tolist()

    # 필터 조건 구성 (§10-2-1 payload 인덱스 활용)
    conditions = []

    if genre_filter:
        conditions.append(FieldCondition(key="genres", match=MatchAny(any=genre_filter)))
    if mood_filter:
        conditions.append(FieldCondition(key="mood_tags", match=MatchAny(any=mood_filter)))
    if ott_filter:
        conditions.append(FieldCondition(key="ott_platforms", match=MatchAny(any=ott_filter)))
    if min_rating is not None:
        conditions.append(FieldCondition(key="rating", range={"gte": min_rating}))
    if year_range is not None:
        conditions.append(FieldCondition(key="release_year", range={"gte": year_range[0], "lte": year_range[1]}))

    query_filter = Filter(must=conditions) if conditions else None

    # 벡터 검색 실행 (qdrant-client v1.17+: search → query_points)
    response = await client.query_points(
        collection_name=settings.QDRANT_COLLECTION,
        query=query_vector,
        query_filter=query_filter,
        limit=top_k,
        with_payload=True,
    )

    return [
        SearchResult(
            movie_id=str(hit.id),
            title=hit.payload.get("title", "") if hit.payload else "",
            score=hit.score,
            source="qdrant",
            metadata=dict(hit.payload) if hit.payload else {},
        )
        for hit in response.points
    ]


# ============================================================
# 2. Elasticsearch BM25 검색
# ============================================================

@traceable(name="search_elasticsearch", run_type="retriever", metadata={"db": "elasticsearch", "type": "bm25"})
async def search_elasticsearch(
    query: str,
    top_k: int = 20,
    genre_filter: list[str] | None = None,
    mood_filter: list[str] | None = None,
) -> list[SearchResult]:
    """
    Elasticsearch BM25 검색: Nori 한국어 형태소 분석 기반 키워드 매칭.

    §11-1 ②: multi_match + function_score (무드태그 부스트)
    """
    client = await get_elasticsearch()

    # multi_match 쿼리 (title, director, overview, cast, keywords 대상)
    must_query: dict = {
        "multi_match": {
            "query": query,
            "fields": [
                "title^3.0",
                "director^2.5",
                "cast^2.0",
                "keywords^1.5",
                "overview^1.0",
            ],
            "type": "best_fields",
        }
    }

    # 필터 조건
    filter_clauses = []
    if genre_filter:
        filter_clauses.append({"terms": {"genres": genre_filter}})
    if mood_filter:
        filter_clauses.append({"terms": {"mood_tags": mood_filter}})

    # function_score로 인기도 부스트
    body = {
        "query": {
            "function_score": {
                "query": {
                    "bool": {
                        "must": [must_query],
                        "filter": filter_clauses if filter_clauses else [],
                    }
                },
                "functions": [
                    {
                        "field_value_factor": {
                            "field": "popularity_score",
                            "modifier": "log1p",
                            "factor": 0.1,
                        }
                    }
                ],
                "boost_mode": "sum",
            }
        },
        "size": top_k,
    }

    resp = await client.search(index=ES_INDEX_NAME, body=body)
    hits = resp["hits"]["hits"]

    return [
        SearchResult(
            movie_id=hit["_id"],
            title=hit["_source"].get("title", ""),
            score=hit["_score"],
            source="es",
            metadata=hit["_source"],
        )
        for hit in hits
    ]


# ============================================================
# 3. Neo4j 그래프 검색
# ============================================================

@traceable(name="search_neo4j", run_type="retriever", metadata={"db": "neo4j", "type": "graph"})
async def search_neo4j(
    mood_tags: list[str] | None = None,
    genres: list[str] | None = None,
    director: str | None = None,
    similar_to_movie_id: str | None = None,
    top_k: int = 15,
) -> list[SearchResult]:
    """
    Neo4j 그래프 검색: 무드/장르/감독 관계를 기반으로 영화를 탐색한다.

    §11-1 ③:
    - MATCH (m)-[:HAS_MOOD]->(mt:MoodTag) 무드 기반
    - SIMILAR_TO 관계로 후보 확장
    - 감독/배우 관계 탐색
    """
    driver = await get_neo4j()

    results: list[SearchResult] = []

    async with driver.session() as session:
        # 전략 1: 무드태그 + 장르 조합 검색
        if mood_tags or genres:
            conditions = []
            params: dict = {}

            if mood_tags:
                conditions.append("(m)-[:HAS_MOOD]->(:MoodTag {name: mood})")
                params["mood_tags"] = mood_tags

            if genres:
                conditions.append("(m)-[:HAS_GENRE]->(:Genre {name: genre})")
                params["genres"] = genres

            # 무드 + 장르 조합 쿼리
            cypher = """
            MATCH (m:Movie)
            WHERE
            """
            where_clauses = []
            if mood_tags:
                where_clauses.append(
                    "EXISTS { MATCH (m)-[:HAS_MOOD]->(mt:MoodTag) WHERE mt.name IN $mood_tags }"
                )
            if genres:
                where_clauses.append(
                    "EXISTS { MATCH (m)-[:HAS_GENRE]->(g:Genre) WHERE g.name IN $genres }"
                )

            cypher += " AND ".join(where_clauses)
            cypher += """
            OPTIONAL MATCH (m)-[:HAS_MOOD]->(mt:MoodTag) WHERE mt.name IN $mood_tags
            WITH m, count(mt) AS mood_match
            RETURN m.id AS movie_id, m.title AS title,
                   m.rating AS rating, m.popularity_score AS popularity,
                   mood_match
            ORDER BY mood_match DESC, m.popularity_score DESC
            LIMIT $top_k
            """
            params["top_k"] = top_k
            if "mood_tags" not in params:
                params["mood_tags"] = []

            result = await session.run(cypher, params)
            records = await result.data()

            for i, record in enumerate(records):
                results.append(SearchResult(
                    movie_id=str(record["movie_id"]),
                    title=record.get("title", ""),
                    score=float(top_k - i),  # 순위 기반 점수
                    source="neo4j",
                    metadata={"rating": record.get("rating"), "mood_match": record.get("mood_match")},
                ))

        # 전략 2: SIMILAR_TO 관계 확장
        if similar_to_movie_id:
            cypher = """
            MATCH (source:Movie {id: $movie_id})-[r:SIMILAR_TO]->(m:Movie)
            RETURN m.id AS movie_id, m.title AS title,
                   r.score AS similarity, m.rating AS rating
            ORDER BY r.score DESC
            LIMIT $top_k
            """
            result = await session.run(cypher, {"movie_id": similar_to_movie_id, "top_k": top_k})
            records = await result.data()

            for record in records:
                results.append(SearchResult(
                    movie_id=str(record["movie_id"]),
                    title=record.get("title", ""),
                    score=float(record.get("similarity", 0)),
                    source="neo4j",
                    metadata={"rating": record.get("rating")},
                ))

        # 전략 3: 감독 기반 탐색
        if director:
            cypher = """
            MATCH (p:Person {name: $director})-[:DIRECTED]->(m:Movie)
            RETURN m.id AS movie_id, m.title AS title, m.rating AS rating
            ORDER BY m.rating DESC
            LIMIT $top_k
            """
            result = await session.run(cypher, {"director": director, "top_k": top_k})
            records = await result.data()

            for i, record in enumerate(records):
                results.append(SearchResult(
                    movie_id=str(record["movie_id"]),
                    title=record.get("title", ""),
                    score=float(top_k - i),
                    source="neo4j",
                    metadata={"rating": record.get("rating")},
                ))

    return results


# ============================================================
# 4. RRF (Reciprocal Rank Fusion) 합산
# ============================================================

def reciprocal_rank_fusion(
    result_lists: list[list[SearchResult]],
    k: int = RRF_K,
) -> list[SearchResult]:
    """
    Reciprocal Rank Fusion으로 여러 검색 결과를 합산한다.

    §6-2-1 rag_retriever 노드의 RRF 합산 로직:
    RRF_score(d) = Σ 1 / (k + rank_i(d))

    Args:
        result_lists: 각 검색 엔진의 결과 리스트 (순위 순서)
        k: RRF 상수 (기본 60)

    Returns:
        RRF 점수 내림차순으로 정렬된 합산 결과
    """
    # 영화별 RRF 점수 누적
    rrf_scores: dict[str, float] = {}
    metadata_cache: dict[str, dict] = {}
    title_cache: dict[str, str] = {}

    for results in result_lists:
        for rank, result in enumerate(results, start=1):
            mid = result.movie_id
            rrf_scores[mid] = rrf_scores.get(mid, 0.0) + 1.0 / (k + rank)

            # 메타데이터는 가장 상세한 것을 보존
            if mid not in metadata_cache or len(result.metadata) > len(metadata_cache[mid]):
                metadata_cache[mid] = result.metadata
                title_cache[mid] = result.title

    # RRF 점수 내림차순 정렬
    sorted_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)

    return [
        SearchResult(
            movie_id=mid,
            title=title_cache.get(mid, ""),
            score=rrf_scores[mid],
            source="rrf",
            metadata=metadata_cache.get(mid, {}),
        )
        for mid in sorted_ids
    ]


# ============================================================
# 통합 하이브리드 검색 인터페이스
# ============================================================

@traceable(name="hybrid_search", run_type="chain", metadata={"fusion": "RRF", "k": 60})
async def hybrid_search(
    query: str,
    top_k: int = 15,
    genre_filter: list[str] | None = None,
    mood_tags: list[str] | None = None,
    ott_filter: list[str] | None = None,
    min_rating: float | None = None,
    year_range: tuple[int, int] | None = None,
    director: str | None = None,
    similar_to_movie_id: str | None = None,
) -> list[SearchResult]:
    """
    3개 검색 엔진을 동시 실행하고 RRF로 합산하여 최종 후보를 반환한다.

    §11-1 하이브리드 검색 흐름:
    ① Qdrant 벡터 검색 (의미)
    ② ES BM25 검색 (키워드)
    ③ Neo4j 그래프 검색 (관계)
    ④ RRF 합산 → 최종 후보

    Args:
        query: 사용자 검색 쿼리
        top_k: 최종 반환 결과 수 (기본 15)
        genre_filter: 장르 필터
        mood_tags: 무드태그 필터
        ott_filter: OTT 플랫폼 필터
        min_rating: 최소 평점
        year_range: (시작연도, 끝연도)
        director: 감독명 (Neo4j 검색용)
        similar_to_movie_id: 유사 영화 기준 ID (Neo4j SIMILAR_TO)

    Returns:
        RRF 합산 점수 기준 상위 top_k 결과
    """
    import asyncio

    # 3개 검색 엔진 동시 실행
    qdrant_task = search_qdrant(
        query=query,
        top_k=30,
        genre_filter=genre_filter,
        mood_filter=mood_tags,
        ott_filter=ott_filter,
        min_rating=min_rating,
        year_range=year_range,
    )

    es_task = search_elasticsearch(
        query=query,
        top_k=20,
        genre_filter=genre_filter,
        mood_filter=mood_tags,
    )

    neo4j_task = search_neo4j(
        mood_tags=mood_tags,
        genres=genre_filter,
        director=director,
        similar_to_movie_id=similar_to_movie_id,
        top_k=15,
    )

    qdrant_results, es_results, neo4j_results = await asyncio.gather(
        qdrant_task, es_task, neo4j_task,
    )

    logger.info(
        "hybrid_search_results",
        qdrant=len(qdrant_results),
        es=len(es_results),
        neo4j=len(neo4j_results),
    )

    # RRF 합산 (§11-1 ④)
    fused = reciprocal_rank_fusion(
        [qdrant_results, es_results, neo4j_results],
        k=RRF_K,
    )

    final = fused[:top_k]

    logger.info(
        "hybrid_search_complete",
        query=query[:50],
        result_count=len(final),
        top_score=final[0].score if final else 0,
    )

    return final
